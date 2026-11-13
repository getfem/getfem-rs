//! Matrix assembly fused over the points of an element (M14 step 4): the
//! assembled tensor is a sum of products of a tensor with the tests of one
//! variable by one with the tests of the other, scaled by scalars. Summed
//! over the points, each product is one GEMM with the points merged into
//! the contracted index: per point the operands are stashed (the weight
//! and the scalars on one of them), at the last point of the element the
//! GEMMs give the element matrix.

use getfem_xsmm::{Gemm, GemmShape};

use super::{ExecState, Instruction, InstructionSet, NONE, kernel_gemm};

/// A product, in the convention of Contraction:
/// K[m + M n] = sum_k a[n + N k] b[m + M k].
#[derive(Clone, Copy, Debug)]
pub(crate) enum Leaf {
  /// Contraction over i
  Contr { a: u32, b: u32, i: u32 },
  /// SimpleTmult t[i + s1 j] = a[i] b[j]
  Tmult { a: u32, b: u32 },
  /// ContractionOpt11, assembled by MatAssemblyOpt10: the scalar block
  Opt11 { a: u32, b: u32, nn: u32 },
  /// ContractionOpt22, assembled by MatAssemblyOpt10: the scalar block
  Opt22 { a: u32, b: u32, n: u32, q: u32 },
  /// ContractionOpt20 (a vectorized gradient, n derivatives, q components)
  Opt20 { a: u32, b: u32, n: u32, q: u32 },
  /// ContractionOpt02 (b a vectorized gradient)
  Opt02 { a: u32, b: u32, n: u32, q: u32 },
  /// ContractionOpt01 (b a vectorized test value), accumulated per point
  Opt01 { a: u32, b: u32, nn: u32 },
  /// not a product: the tensor t itself, accumulated per point
  Dense { t: u32 },
}

/// A product times the scalars mul, divided by div, negated if neg.
#[derive(Clone, Debug)]
pub(crate) struct Term {
  leaf: Leaf,
  mul: Vec<u32>,
  div: Vec<u32>,
  neg: bool,
}

/// A fused matrix assembly: tests of v1 (rows), v2 (NONE: a vector); q >
/// 0: the scalar block on the q components (matrix_opt10).
#[derive(Clone, Debug)]
pub(crate) struct Fused {
  v1: u32,
  v2: u32,
  q: u32,
  terms: Vec<Term>,
}

impl Fused {
  /// Test only: the products' leaves
  #[cfg(test)]
  pub(crate) fn leaves(&self) -> impl Iterator<Item = Leaf> + '_ {
    self.terms.iter().map(|t| t.leaf)
  }
}

impl Leaf {
  /// the operands (Dense: its tensor twice)
  fn args(self) -> [u32; 2] {
    match self {
      Leaf::Contr { a, b, .. }
      | Leaf::Tmult { a, b }
      | Leaf::Opt11 { a, b, .. }
      | Leaf::Opt22 { a, b, .. }
      | Leaf::Opt20 { a, b, .. }
      | Leaf::Opt02 { a, b, .. }
      | Leaf::Opt01 { a, b, .. } => [a, b],
      Leaf::Dense { t } => [t, t],
    }
  }
}

#[cfg(test)]
thread_local! {
  /// tests: compile without fusion
  pub(crate) static NO_FUSE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
  /// tests: another GEMM threshold
  pub(crate) static TEST_GEMM_MIN: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

impl InstructionSet {
  /// the tensors read by instruction i (with the components of CMatrix and
  /// the operands of fused assemblies)
  fn reads(&self, i: &Instruction) -> Vec<u32> {
    let mut r = i.tensor_args();
    match *i {
      Instruction::CMatrix { l, .. } | Instruction::CMatrixTests { l, .. } => {
        r.extend_from_slice(&self.maps[l as usize]);
      }
      Instruction::Condense { c } => {
        let cd = &self.conds[c as usize];
        r.extend(cd.kqq.iter().chain(&cd.kqj).filter(|&&x| x != NONE));
      }
      Instruction::CondenseK { k, .. } | Instruction::CondenseR { k, .. } => {
        let ck = &self.cond_k[k as usize];
        let cd = &self.conds[ck.c as usize];
        r.extend(ck.kiq.iter().filter(|&&x| x != NONE));
        r.extend(cd.kqjp.iter().chain(&cd.rqp));
      }
      Instruction::FusedAssembly { f } => {
        for t in &self.fused[f as usize].terms {
          r.extend(t.leaf.args());
          r.extend(&t.mul);
          r.extend(&t.div);
        }
      }
      _ => {}
    }
    r
  }

  /// the terms of tensor s (producers: instruction of each
  /// tensor); false if it is not such a sum
  fn terms(&self, s: u32, prod: &[usize], opt10: bool, t: &Term, out: &mut Vec<Term>) -> bool {
    let k = prod[s as usize];
    if k == usize::MAX {
      return false;
    }
    let with = |leaf: Leaf| Term { leaf, ..t.clone() };
    match self.instructions[k] {
      Instruction::Add { a, b, .. } => {
        self.terms(a, prod, opt10, t, out) && self.terms(b, prod, opt10, t, out)
      }
      Instruction::Sub { a, b, .. } => {
        let tn = Term {
          neg: !t.neg,
          ..t.clone()
        };
        self.terms(a, prod, opt10, t, out) && self.terms(b, prod, opt10, &tn, out)
      }
      Instruction::ScalarMult { a, c, .. } => {
        let mut tm = t.clone();
        tm.mul.push(c);
        self.terms(a, prod, opt10, &tm, out)
      }
      Instruction::ScalarDiv { a, c, .. } => {
        let mut tm = t.clone();
        tm.div.push(c);
        self.terms(a, prod, opt10, &tm, out)
      }
      Instruction::Contraction { a, b, i, .. } if !opt10 => {
        out.push(with(Leaf::Contr { a, b, i }));
        true
      }
      Instruction::SimpleTmult { a, b, .. } if !opt10 => {
        out.push(with(Leaf::Tmult { a, b }));
        true
      }
      Instruction::ContractionOpt11 { a, b, nn, .. } if opt10 => {
        out.push(with(Leaf::Opt11 { a, b, nn }));
        true
      }
      Instruction::ContractionOpt22 { a, b, nq, .. } if opt10 => {
        let (n, q) = (nq & 0xffff, nq >> 16);
        out.push(with(Leaf::Opt22 { a, b, n, q }));
        true
      }
      Instruction::ContractionOpt01 { a, b, nn, .. } if !opt10 => {
        out.push(with(Leaf::Opt01 { a, b, nn }));
        true
      }
      Instruction::ContractionOpt20 { a, b, nq, .. } if !opt10 => {
        let (n, q) = (nq & 0xffff, nq >> 16);
        out.push(with(Leaf::Opt20 { a, b, n, q }));
        true
      }
      Instruction::ContractionOpt02 { a, b, nq, .. } if !opt10 => {
        let (n, q) = (nq & 0xffff, nq >> 16);
        out.push(with(Leaf::Opt02 { a, b, n, q }));
        true
      }
      // not a product (not in the scalar block of opt10)
      _ if !opt10 => {
        out.push(with(Leaf::Dense { t: s }));
        true
      }
      _ => false,
    }
  }

  /// the matrix assemblies of the point instructions fused where their
  /// tensor is a sum of products; the instructions no longer read removed
  pub(crate) fn fuse(&mut self) {
    #[cfg(test)]
    if NO_FUSE.get() {
      return;
    }
    // filters skip a number of instructions: not with removals
    if self
      .instructions
      .iter()
      .any(|i| matches!(i, Instruction::Filter { .. }))
    {
      return;
    }
    let mut prod = vec![usize::MAX; self.shapes.len()];
    for (k, i) in self.instructions.iter().enumerate() {
      if let Some(o) = i.out() {
        prod[o as usize] = k;
      }
    }
    let mut any = false;
    for k in 0..self.instructions.len() {
      let (t, v1, v2, q) = match self.instructions[k] {
        Instruction::MatAssembly { t, v1, v2, .. } => (t, v1, v2, 0),
        Instruction::VecAssembly { t, v, .. } => (t, v, NONE, 0),
        Instruction::MatAssemblyOpt10 { t, v1, v2, .. } => {
          let Some(q) = self.leaf_opt10(t, &prod) else {
            continue;
          };
          (t, v1, v2, q)
        }
        _ => continue,
      };
      let t0 = Term {
        leaf: Leaf::Tmult { a: NONE, b: NONE },
        mul: vec![],
        div: vec![],
        neg: false,
      };
      let mut terms = vec![];
      if !self.terms(t, &prod, q > 0, &t0, &mut terms)
        || terms.iter().all(|t| matches!(t.leaf, Leaf::Dense { .. }))
      {
        continue;
      }
      let f = self.fused.len() as u32;
      self.fused.push(Fused { v1, v2, q, terms });
      self.instructions[k] = Instruction::FusedAssembly { f };
      any = true;
    }
    if any {
      self.remove_dead();
    }
  }

  /// the qdim of the ContractionOpt11 or Opt22 at the leaves of the
  /// tensor t (the first found)
  fn leaf_opt10(&self, t: u32, prod: &[usize]) -> Option<u32> {
    let k = prod[t as usize];
    if k == usize::MAX {
      return None;
    }
    match self.instructions[k] {
      Instruction::ContractionOpt11 { nn, .. } => Some(nn),
      Instruction::ContractionOpt22 { nq, .. } => Some(nq >> 16),
      Instruction::Add { a, .. }
      | Instruction::Sub { a, .. }
      | Instruction::ScalarMult { a, .. }
      | Instruction::ScalarDiv { a, .. } => self.leaf_opt10(a, prod),
      _ => None,
    }
  }

  /// point instructions whose result is not read (by the point or element
  /// instructions) removed, last first
  fn remove_dead(&mut self) {
    let mut nread = vec![0u32; self.shapes.len()];
    // a read of an alias reads the tensor it shares
    let base = |s: u32| *self.alias.get(&s).unwrap_or(&s);
    let all: Vec<Instruction> = self
      .elt_instructions
      .iter()
      .chain(&self.instructions)
      .copied()
      .collect();
    for i in &all {
      for s in self.reads(i) {
        if let Some(c) = nread.get_mut(base(s) as usize) {
          *c += 1;
        }
      }
    }
    let mut keep = vec![true; self.instructions.len()];
    for k in (0..self.instructions.len()).rev() {
      let i = self.instructions[k];
      let Some(o) = i.out() else { continue };
      if nread[o as usize] > 0 {
        continue;
      }
      keep[k] = false;
      for s in self.reads(&i) {
        if let Some(c) = nread.get_mut(base(s) as usize) {
          *c -= 1;
        }
      }
    }
    let mut k = 0;
    self.instructions.retain(|_| {
      k += 1;
      keep[k - 1]
    });
  }

  /// point: the products of fused assembly f accumulated (small) or their
  /// operands stashed (large: one GEMM per product at the last point); at
  /// the last point of the element the element matrix
  pub(crate) fn fused_assembly(&self, f: usize, m: &mut ExecState, ext: &mut impl super::Ext) {
    let fu = &self.fused[f];
    let (ipt, nq) = (m.ipt, m.nbpt);
    let mut el = std::mem::take(&mut m.facc[f]);
    if ipt == 0 {
      // by GEMM: the products with tests on both operands and large GEMM
      // calls (sizes known for the points and the layout)
      if m.fkey[f] != (nq, m.layouts) {
        let mut g = std::mem::take(&mut m.fgemm[f]);
        g.clear();
        g.extend(fu.terms.iter().map(|t| {
          let c = sizes(t.leaf, m, nq).1;
          c > 0 && c >= gemm_min()
        }));
        m.fgemm[f] = g;
        m.flen[f] = sizes(fu.terms[0].leaf, m, nq).0;
        m.fkey[f] = (nq, m.layouts);
      }
      el.clear();
      el.resize(m.flen[f], 0.0);
    }
    let gemm = m.fgemm[f].iter().any(|&g| g);
    let scale = |term: &Term, m: &ExecState| {
      let mut s = m.coeff;
      for &c in &term.mul {
        s *= m.buffer[m.range(c).start];
      }
      for &c in &term.div {
        s /= m.buffer[m.range(c).start];
      }
      if term.neg { -s } else { s }
    };
    if gemm {
      // the points by chunks of STASH_POINTS, one product per chunk (the
      // stash bounded on elements with many points: cut elements)
      let start = ipt - ipt % STASH_POINTS;
      let nc = STASH_POINTS.min(nq - start);
      let mut st = std::mem::take(&mut m.stash[f]);
      st.resize(fu.terms.len(), (vec![], vec![]));
      for (k, (term, (sa, sb))) in fu.terms.iter().zip(st.iter_mut()).enumerate() {
        let s = scale(term, m);
        if m.fgemm[f][k] {
          stash(term.leaf, m, ipt - start, nc, s, sa, sb);
        } else {
          add_tensor_to_element_matrix(term.leaf, m, s, &mut el);
        }
      }
      if ipt + 1 == start + nc {
        for (k, (term, (sa, sb))) in fu.terms.iter().zip(&st).enumerate() {
          if m.fgemm[f][k] {
            product(term.leaf, m, nc, sa, sb, &mut el);
          }
        }
      }
      m.stash[f] = st;
    } else {
      for term in &fu.terms {
        let s = scale(term, m);
        add_tensor_to_element_matrix(term.leaf, m, s, &mut el);
      }
    }
    if ipt + 1 == nq {
      if fu.v2 == NONE {
        ext.vector(fu.v1 as usize, &el);
      } else if fu.q > 0 {
        ext.matrix_opt10(fu.v1 as usize, fu.v2 as usize, fu.q as usize, &el);
      } else {
        ext.matrix(fu.v1 as usize, fu.v2 as usize, &el);
      }
    }
    m.facc[f] = el;
  }
}

/// products of at least this many multiply-adds per GEMM call by GEMM, the
/// smaller ones accumulated per point (i9-13900K, serial OpenBLAS: 3D
/// stiffness contractions, 1200, faster accumulated; 3D mass, 1500, and
/// elasticity faster by GEMM; m14-fused.md); all accumulated without
/// libxsmm
const GEMM_MIN: usize = 1400;

fn gemm_min() -> usize {
  #[cfg(test)]
  if let Some(g) = TEST_GEMM_MIN.get() {
    return g;
  }
  if Gemm::available() {
    GEMM_MIN
  } else {
    usize::MAX
  }
}

/// size of the element matrix of a product, the multiply-adds of one of its
/// GEMM calls for nq points (0: accumulated per point)
fn sizes(leaf: Leaf, m: &ExecState, nq: usize) -> (usize, usize) {
  let [a, b] = leaf.args();
  let (la, lb) = (m.range(a).len(), m.range(b).len());
  match leaf {
    // (multiply-adds 0: accumulated, if an operand is a single row)
    Leaf::Contr { i, .. } => {
      let i = i as usize;
      let (nn, mm) = (la / i, lb / i);
      (
        nn * mm,
        if nn > 1 && mm > 1 {
          la * lb / i * nq
        } else {
          0
        },
      )
    }
    Leaf::Tmult { .. } => (la * lb, if la > 1 && lb > 1 { la * lb * nq } else { 0 }),
    Leaf::Opt11 { nn, .. } => {
      let nn2 = (nn * nn) as usize;
      ((la / nn2) * (lb / nn2), (la / nn2) * (lb / nn2) * nq)
    }
    Leaf::Opt22 { n, q, .. } => {
      let nqq = (n * q * q) as usize;
      let (s1, s2) = (la / nqq, lb / nqq);
      (s1 * s2, s1 * s2 * n as usize * nq)
    }
    Leaf::Opt20 { n, q, .. } => {
      let nq2 = (n * q) as usize;
      let (s1, s2) = (la / nq2, lb / nq2);
      // q calls
      (s1 * s2, s1 * s2 / q as usize * n as usize * nq)
    }
    Leaf::Opt02 { n, q, .. } => {
      let nq2 = (n * q) as usize;
      let (s1, s2) = (la / nq2, lb / nq2);
      // q calls of s2 / q rows, s1 columns, n nq products
      (s1 * s2, s1 * s2 / q as usize * n as usize * nq)
    }
    Leaf::Opt01 { nn, .. } => {
      let nn = nn as usize;
      let (s1, s2n) = (la / nn, lb / (nn * nn));
      (s1 * s2n * nn, 0)
    }
    Leaf::Dense { .. } => (la, 0),
  }
}

/// el += s times the product at the point
fn add_tensor_to_element_matrix(leaf: Leaf, m: &ExecState, s: f64, el: &mut [f64]) {
  let [a, b] = leaf.args();
  let (a, b) = (&m.buffer[m.range(a)], &m.buffer[m.range(b)]);
  match leaf {
    Leaf::Contr { i, .. } => match i {
      1 => acc_contract::<1>(el, a, b, s),
      2 => acc_contract::<2>(el, a, b, s),
      3 => acc_contract::<3>(el, a, b, s),
      4 => acc_contract::<4>(el, a, b, s),
      5 => acc_contract::<5>(el, a, b, s),
      6 => acc_contract::<6>(el, a, b, s),
      _ => {
        let i = i as usize;
        let (nn, mm) = (a.len() / i, b.len() / i);
        for k in 0..i {
          let bk = &b[mm * k..mm * (k + 1)];
          for (n, &x) in a[nn * k..nn * (k + 1)].iter().enumerate() {
            let x = s * x;
            let row = el[mm * n..mm * (n + 1)].iter_mut().zip(bk);
            row.for_each(|(o, y)| *o += x * y);
          }
        }
      }
    },
    // t[i + s1 j] = a[i] b[j]
    Leaf::Tmult { .. } => acc_contract::<1>(el, b, a, s),
    Leaf::Dense { .. } => el.iter_mut().zip(a).for_each(|(o, x)| *o += s * x),
    Leaf::Opt01 { nn, .. } => match nn {
      2 => acc_opt01::<2>(el, a, b, s),
      3 => acc_opt01::<3>(el, a, b, s),
      _ => {
        // t[k + nn (jj + s2n ii)] = a[ii + s1 k] b[jj nn]
        let nn = nn as usize;
        let (s1, s2n) = (a.len() / nn, b.len() / (nn * nn));
        for (ii, row) in el.chunks_exact_mut(nn * s2n).enumerate().take(s1) {
          for (jj, o) in row.chunks_exact_mut(nn).enumerate() {
            let y = s * b[jj * nn];
            for (k, ok) in o.iter_mut().enumerate() {
              *ok += a[ii + s1 * k] * y;
            }
          }
        }
      }
    },
    Leaf::Opt11 { nn, .. } => {
      // el[i + nb j] += s b[i nn] a[j nn]
      let nn = nn as usize;
      let (na, nb) = (a.len() / (nn * nn), b.len() / (nn * nn));
      for (j, col) in el.chunks_exact_mut(nb).enumerate().take(na) {
        let x = s * a[j * nn];
        for (i, o) in col.iter_mut().enumerate() {
          *o += x * b[i * nn];
        }
      }
    }
    Leaf::Opt22 { n, q, .. } => {
      // el[i + nb j] += s sum_mm b[i q + s2 q mm] a[j q + s1 q mm]
      let (n, q) = (n as usize, q as usize);
      let (s1, s2) = (a.len() / (n * q), b.len() / (n * q));
      let nb = s2 / q;
      for (j, col) in el.chunks_exact_mut(nb).enumerate().take(s1 / q) {
        for mm in 0..n {
          let x = s * a[j * q + s1 * q * mm];
          let bm = b[s2 * q * mm..s2 * q * mm + s2].iter().step_by(q);
          col.iter_mut().zip(bm).for_each(|(o, y)| *o += x * y);
        }
      }
    }
    Leaf::Opt02 { n, q, .. } => match (n, q) {
      (2, 2) => acc_opt02n::<2, 2>(el, a, b, s),
      (3, 3) => acc_opt02n::<3, 3>(el, a, b, s),
      _ => acc_opt02(el, a, b, n as usize, q as usize, s),
    },
    Leaf::Opt20 { n, q, .. } => match n {
      2 => acc_opt20::<2>(el, a, b, q as usize, s),
      3 => acc_opt20::<3>(el, a, b, q as usize, s),
      _ => {
        let (n, q) = (n as usize, q as usize);
        let (s1, s2) = (a.len() / (n * q), b.len() / (n * q));
        for ii in 0..s1 / q {
          for l in 0..q {
            let row = &mut el[s2 * (l + q * ii)..s2 * (l + q * ii + 1)];
            for mm in 0..n {
              let x = s * a[ii * q + s1 * q * mm];
              let br = &b[l * s2 + s2 * q * mm..l * s2 + s2 * q * mm + s2];
              row.iter_mut().zip(br).for_each(|(o, y)| *o += x * y);
            }
          }
        }
      }
    },
  }
}

/// ContractionOpt02 accumulated: el[l + q j + s2 i] += s sum_m a[i + s1 l
/// + s1 q m] b[j q + s2 q m] (the sum as the instruction computes it)
fn acc_opt02(el: &mut [f64], a: &[f64], b: &[f64], n: usize, q: usize, s: f64) {
  let (s1, s2) = (a.len() / (n * q), b.len() / (n * q));
  let (s1q, s2q) = (s1 * q, s2 * q);
  for (i, row) in el.chunks_exact_mut(s2).enumerate().take(s1) {
    for (j, o) in row.chunks_exact_mut(q).enumerate() {
      for (l, ol) in o.iter_mut().enumerate() {
        let mut x = a[i + s1 * l] * b[j * q];
        for m in 1..n {
          x += a[i + s1 * l + s1q * m] * b[j * q + s2q * m];
        }
        *ol += s * x;
      }
    }
  }
}

/// acc_opt02 for n = N, q = Q (as ga's opt02n): the same sums
fn acc_opt02n<const N: usize, const Q: usize>(el: &mut [f64], av: &[f64], bv: &[f64], s: f64) {
  let (s1, s2) = (av.len() / (N * Q), bv.len() / (N * Q));
  let (s1qq, s2qq) = (s1 * Q, s2 * Q);
  for (ii, row) in el.chunks_exact_mut(s2).enumerate().take(s1) {
    let al: [[f64; Q]; N] =
      std::array::from_fn(|m| std::array::from_fn(|l| av[ii + s1 * l + s1qq * m]));
    let (ra, _) = row.as_chunks_mut::<Q>();
    for (j, o) in ra.iter_mut().enumerate() {
      let bm: [f64; N] = std::array::from_fn(|m| bv[j * Q + s2qq * m]);
      for l in 0..Q {
        let mut x = al[0][l] * bm[0];
        for m in 1..N {
          x += al[m][l] * bm[m];
        }
        o[l] += s * x;
      }
    }
  }
}

/// el[m + M n] += sum_k (s a[n + N k]) b[m + M k]
fn acc_contract<const I: usize>(el: &mut [f64], av: &[f64], bv: &[f64], s: f64) {
  let (nn, mm) = (av.len() / I, bv.len() / I);
  let bk: [&[f64]; I] = std::array::from_fn(|k| &bv[mm * k..mm * (k + 1)]);
  for (n, row) in el.chunks_exact_mut(mm).enumerate().take(nn) {
    let ak: [f64; I] = std::array::from_fn(|k| s * av[n + nn * k]);
    for (m, o) in row.iter_mut().enumerate() {
      let mut x = ak[0] * bk[0][m];
      for k in 1..I {
        x += ak[k] * bk[k][m];
      }
      *o += x;
    }
  }
}

/// el[k + N (jj + s2n ii)] += (s a[ii + s1 k]) b[jj N] (as opt01)
fn acc_opt01<const N: usize>(el: &mut [f64], av: &[f64], bv: &[f64], s: f64) {
  let s1 = av.len() / N;
  let s2n = bv.len() / N / N;
  let (ea, _) = el.as_chunks_mut::<N>();
  let (ba, _) = bv.as_chunks::<N>();
  for (ii, row) in ea.chunks_exact_mut(s2n).enumerate().take(s1) {
    let ai: [f64; N] = std::array::from_fn(|k| s * av[ii + s1 * k]);
    for (o, b) in row.iter_mut().zip(ba) {
      let y = b[0];
      for k in 0..N {
        o[k] += ai[k] * y;
      }
    }
  }
}

/// el[j + s2 (l + q ii)] += sum_m (s a[ii q + s1 q m]) b[l s2 + j + s2 q m]
fn acc_opt20<const N: usize>(el: &mut [f64], av: &[f64], bv: &[f64], q: usize, s: f64) {
  let nn = N * q;
  let (s1, s2) = (av.len() / nn, bv.len() / nn);
  let (s1q, s1qq, s2qq) = (s1 / q, s1 * q, s2 * q);
  for ii in 0..s1q {
    let x: [f64; N] = std::array::from_fn(|m| s * av[ii * q + s1qq * m]);
    for l in 0..q {
      let row = &mut el[s2 * (l + q * ii)..s2 * (l + q * ii + 1)];
      let br: [&[f64]; N] = std::array::from_fn(|m| &bv[l * s2 + s2qq * m..l * s2 + s2qq * m + s2]);
      for (j, o) in row.iter_mut().enumerate() {
        let mut y = x[0] * br[0][j];
        for m in 1..N {
          y += x[m] * br[m][j];
        }
        *o += y;
      }
    }
  }
}

/// points stashed at most for one product
const STASH_POINTS: usize = 128;

/// the operands of point ipt (of nq) into sa (times s), sb
fn stash(
  leaf: Leaf,
  m: &ExecState,
  ipt: usize,
  nq: usize,
  s: f64,
  sa: &mut Vec<f64>,
  sb: &mut Vec<f64>,
) {
  let [a, b] = leaf.args();
  let (a, b) = (&m.buffer[m.range(a)], &m.buffer[m.range(b)]);
  // sa gets the scaled operand (the one indexed by n)
  let (xa, xb) = match leaf {
    Leaf::Tmult { .. } => (b, a),
    _ => (a, b),
  };
  match leaf {
    // accumulated, never stashed
    Leaf::Dense { .. } | Leaf::Opt01 { .. } => {}
    Leaf::Contr { .. } | Leaf::Tmult { .. } => {
      let (la, lb) = (xa.len(), xb.len());
      sa.resize(la * nq, 0.0);
      sb.resize(lb * nq, 0.0);
      for (o, x) in sa[la * ipt..la * (ipt + 1)].iter_mut().zip(xa) {
        *o = s * x;
      }
      sb[lb * ipt..lb * (ipt + 1)].copy_from_slice(xb);
    }
    Leaf::Opt11 { nn, .. } => {
      // scalar parts: a[j nn] (tests of a), b[i nn] (tests of b)
      let nn = nn as usize;
      let (na, nb) = (a.len() / (nn * nn), b.len() / (nn * nn));
      sa.resize(na * nq, 0.0);
      sb.resize(nb * nq, 0.0);
      for j in 0..na {
        sa[na * ipt + j] = s * a[j * nn];
      }
      for i in 0..nb {
        sb[nb * ipt + i] = b[i * nn];
      }
    }
    Leaf::Opt22 { n, q, .. } => {
      // scalar parts, the derivatives with the points in the contracted
      // index: sa[j + na (mm + n ipt)] = s a[j q + s1 q mm], sb likewise
      let (n, q) = (n as usize, q as usize);
      let (s1, s2) = (a.len() / (n * q), b.len() / (n * q));
      let (na, nb) = (s1 / q, s2 / q);
      sa.resize(na * n * nq, 0.0);
      sb.resize(nb * n * nq, 0.0);
      for mm in 0..n {
        let c = mm + n * ipt;
        for j in 0..na {
          sa[j + na * c] = s * a[j * q + s1 * q * mm];
        }
        for i in 0..nb {
          sb[i + nb * c] = b[i * q + s2 * q * mm];
        }
      }
    }
    Leaf::Opt20 { n, q, .. } => {
      // t[j + s2 (l + q ii)] = sum_mm a[ii q + s1 q mm] b[l s2 + j + s2 q mm]:
      // sa[ii + s1q (mm + n ipt)], per l: sb[l s2 K + j + s2 (mm + n ipt)]
      let (n, q) = (n as usize, q as usize);
      let (s1, s2) = (a.len() / (n * q), b.len() / (n * q));
      let s1q = s1 / q;
      let kk = n * nq;
      sa.resize(s1q * kk, 0.0);
      sb.resize(q * s2 * kk, 0.0);
      for mm in 0..n {
        let c = mm + n * ipt;
        for ii in 0..s1q {
          sa[ii + s1q * c] = s * a[ii * q + s1 * q * mm];
        }
        for l in 0..q {
          let src = &b[l * s2 + s2 * q * mm..l * s2 + s2 * q * mm + s2];
          sb[l * s2 * kk + s2 * c..l * s2 * kk + s2 * (c + 1)].copy_from_slice(src);
        }
      }
    }
    Leaf::Opt02 { n, q, .. } => {
      // t[l + q j + s2 i] = sum_mm a[i + s1 l + s1 q mm] b[j q + s2 q mm]:
      // per l: sa[l s1 K + i + s1 (mm + n ipt)], sb[j + s2/q (mm + n ipt)]
      let (n, q) = (n as usize, q as usize);
      let (s1, s2) = (a.len() / (n * q), b.len() / (n * q));
      let (s2j, kk) = (s2 / q, n * nq);
      sa.resize(q * s1 * kk, 0.0);
      sb.resize(s2j * kk, 0.0);
      for mm in 0..n {
        let c = mm + n * ipt;
        for l in 0..q {
          let src = &a[s1 * l + s1 * q * mm..s1 * l + s1 * q * mm + s1];
          let dst = &mut sa[l * s1 * kk + s1 * c..l * s1 * kk + s1 * (c + 1)];
          dst.iter_mut().zip(src).for_each(|(o, x)| *o = s * x);
        }
        for j in 0..s2j {
          sb[j + s2j * c] = b[j * q + s2 * q * mm];
        }
      }
    }
  }
}

/// C (+)= A op(B) (B transposed), the shape's kernel or, without one, a
/// plain loop
fn gemm(
  (m, n, k): (usize, usize, usize),
  (lda, ldb, ldc): (usize, usize, usize),
  accumulate: bool,
  a: &[f64],
  b: &[f64],
  c: &mut [f64],
) {
  let s = GemmShape {
    ta: false,
    tb: true,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    accumulate,
  };
  if kernel_gemm(&s, a, b, c) {
    return;
  }
  for j in 0..n {
    for i in 0..m {
      let v: f64 = (0..k).map(|l| a[i + l * lda] * b[j + l * ldb]).sum();
      let o = &mut c[i + j * ldc];
      *o = if accumulate { *o + v } else { v };
    }
  }
}

/// el += the product of the stashed operands of nq points
fn product(leaf: Leaf, m: &ExecState, nq: usize, sa: &[f64], sb: &[f64], el: &mut [f64]) {
  match leaf {
    Leaf::Dense { .. } | Leaf::Opt01 { .. } => {}
    Leaf::Contr { i, .. } => {
      let kk = i as usize * nq;
      let (nn, mm) = (sa.len() / kk, sb.len() / kk);
      gemm((mm, nn, kk), (mm, nn, mm), true, sb, sa, el);
    }
    Leaf::Tmult { .. } | Leaf::Opt11 { .. } => {
      let (nn, mm) = (sa.len() / nq, sb.len() / nq);
      gemm((mm, nn, nq), (mm, nn, mm), true, sb, sa, el);
    }
    Leaf::Opt22 { n, .. } => {
      let kk = n as usize * nq;
      let (nn, mm) = (sa.len() / kk, sb.len() / kk);
      gemm((mm, nn, kk), (mm, nn, mm), true, sb, sa, el);
    }
    Leaf::Opt20 { b, n, q, .. } => {
      let (n, q) = (n as usize, q as usize);
      let kk = n * nq;
      let s1q = sa.len() / kk;
      let s2 = m.range(b).len() / (n * q);
      for l in 0..q {
        let bl = &sb[l * s2 * kk..(l + 1) * s2 * kk];
        gemm(
          (s2, s1q, kk),
          (s2, s1q, q * s2),
          true,
          bl,
          sa,
          &mut el[l * s2..],
        );
      }
    }
    Leaf::Opt02 { a, n, q, .. } => {
      // per l, (s2/q x s1) = G A_l^T, added at el[l + q j + s2 i]
      let (n, q) = (n as usize, q as usize);
      let kk = n * nq;
      let s1 = m.range(a).len() / (n * q);
      let s2j = sb.len() / kk;
      let mut tmp = vec![0.0; s2j * s1];
      for l in 0..q {
        let al = &sa[l * s1 * kk..(l + 1) * s1 * kk];
        gemm((s2j, s1, kk), (s2j, s1, s2j), false, sb, al, &mut tmp);
        for (i, ti) in tmp.chunks_exact(s2j).enumerate() {
          let row = &mut el[s2j * q * i..s2j * q * (i + 1)];
          for (j, x) in ti.iter().enumerate() {
            row[l + q * j] += x;
          }
        }
      }
    }
  }
}
