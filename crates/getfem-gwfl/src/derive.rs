//! Symbolic derivation of an analysed tree w.r.t. a variable (GetFEM
//! ga_node_derivation): the result has one more test function (Test_ for
//! order 1, Test2_ for order 2) or, for Diff, a given direction. Subtrees not
//! depending on the variable are copied; the result is analysed again.

use crate::defs::{Defs, Support};
use crate::lexer::Tok;
use crate::semantic::{Der, Node, Src, Tree, Ty};
use crate::{Error, Result};

struct D<'a> {
  t: Tree,
  defs: &'a Defs,
  var: String,
  /// the transformation through which var is read ("" for var itself)
  trans: String,
  /// var through any transformation (GetFEM any_trans: Diff)
  any_trans: bool,
  order: usize,
  dir: Option<usize>,
  marked: Vec<bool>,
  /// spatial gradient instead of a variation (GetFEM ga_node_grad)
  meshdim: Option<usize>,
}

/// Derivative of `tree` w.r.t. `var` read through the interpolate
/// transformation `trans` ("" for var itself, GetFEM's var_trans_pair);
/// None if it does not depend on it. `dir`: direction subtree (a node of
/// `tree`) instead of a test function.
pub fn derive(
  tree: &Tree,
  defs: &Defs,
  var: &str,
  trans: &str,
  order: usize,
  dir: Option<usize>,
) -> Result<Option<Tree>> {
  derive_trans(tree, defs, var, trans, false, order, dir)
}

/// As derive; any_trans: var read through any transformation too (GetFEM
/// Diff: ga_node_mark_tree_for_variable, ga_node_derivation with any_trans)
pub(crate) fn derive_trans(
  tree: &Tree,
  defs: &Defs,
  var: &str,
  trans: &str,
  any_trans: bool,
  order: usize,
  dir: Option<usize>,
) -> Result<Option<Tree>> {
  let Some(r) = tree.root else { return Ok(None) };
  let mut d = D {
    t: tree.clone(),
    defs,
    var: var.to_string(),
    trans: trans.to_string(),
    any_trans,
    order,
    dir,
    marked: vec![false; tree.n.len()],
    meshdim: None,
  };
  if !d.mark(r) {
    return Ok(None);
  }
  let nr = d.d(r)?;
  d.t.n[nr].parent = None;
  d.t.root = Some(nr);
  Ok(Some(d.t))
}

/// Spatial gradient of `tree` (GetFEM ga_node_grad): a trailing index of
/// size meshdim; None if the tree does not depend on the position.
pub fn grad(tree: &Tree, defs: &Defs, meshdim: usize) -> Result<Option<Tree>> {
  let Some(r) = tree.root else { return Ok(None) };
  let mut d = D {
    t: tree.clone(),
    defs,
    var: String::new(),
    trans: String::new(),
    any_trans: false,
    order: 0,
    dir: None,
    marked: vec![false; tree.n.len()],
    meshdim: Some(meshdim),
  };
  if !d.mark(r) {
    return Ok(None);
  }
  let nr = d.g(r)?;
  d.t.n[nr].parent = None;
  d.t.root = Some(nr);
  Ok(Some(d.t))
}

/// Variables (not data) whose values an analysed tree reads, with the
/// transformation they are read through ("" for directly), including the
/// variables the transformations depend on (GetFEM ga_extract_variables).
pub fn variables(tree: &Tree, defs: &Defs) -> Vec<(String, String)> {
  let mut out: Vec<(String, String)> = vec![];
  let Some(r) = tree.root else { return out };
  let is_var = |v: &str| defs.var(v).is_some_and(|v| !v.is_data);
  let add = |out: &mut Vec<(String, String)>, v: &str, t: &str| {
    let p = (v.to_string(), t.to_string());
    if is_var(v) && !out.contains(&p) {
      out.push(p);
    }
  };
  let mut stack = vec![r];
  while let Some(i) = stack.pop() {
    let n = &tree.n[i];
    let trans = match n.ty {
      Ty::Field {
        src: Src::Plain | Src::Elem | Src::XfemPlus | Src::XfemMinus,
        test: false,
        ..
      } => {
        add(&mut out, &n.name, "");
        None
      }
      Ty::Field {
        src: Src::Interp,
        test,
        ..
      } => {
        if !test {
          add(&mut out, &n.name, &n.interpolate_name);
        }
        Some(&n.interpolate_name)
      }
      Ty::InterpX | Ty::InterpNormal | Ty::InterpEltK | Ty::InterpEltB | Ty::InterpolateFilter => {
        Some(&n.interpolate_name)
      }
      _ => None,
    };
    if let Some(deps) = trans.and_then(|t| defs.interpolate_transformations.get(t)) {
      for (v, t) in deps {
        add(&mut out, v, t);
      }
    }
    stack.extend(&n.children);
  }
  out.sort();
  out
}

/// The data a tree reads (GetFEM's data list of a brick), sorted; None
/// when it reads through an interpolate transformation, which may read
/// any data.
pub fn data(tree: &Tree, defs: &Defs) -> Option<Vec<String>> {
  let mut out: Vec<String> = vec![];
  let Some(r) = tree.root else {
    return Some(out);
  };
  let mut stack = vec![r];
  while let Some(i) = stack.pop() {
    let n = &tree.n[i];
    match n.ty {
      Ty::Field {
        src: Src::Interp, ..
      }
      | Ty::InterpX
      | Ty::InterpNormal
      | Ty::InterpEltK
      | Ty::InterpEltB
      | Ty::InterpolateFilter => return None,
      Ty::Field { test: false, .. }
        if defs.var(&n.name).is_some_and(|v| v.is_data) && !out.contains(&n.name) =>
      {
        out.push(n.name.clone())
      }
      _ => {}
    }
    stack.extend(&n.children);
  }
  out.sort();
  Some(out)
}

impl D<'_> {
  fn mark(&mut self, i: usize) -> bool {
    let mut m = false;
    for c in self.t.n[i].children.clone() {
      m |= self.mark(c);
    }
    let n = &self.t.n[i];
    if self.meshdim.is_some() {
      let fem = |v: &str| {
        self
          .defs
          .var(v)
          .is_some_and(|v| matches!(v.support, Support::Fem { .. }))
      };
      m |= match n.ty {
        Ty::Field { .. } => fem(&n.name),
        // (element_K, element_B constant on the element, as GetFEM)
        Ty::X | Ty::Normal | Ty::InterpX | Ty::InterpNormal | Ty::InterpEltK | Ty::InterpEltB => {
          true
        }
        _ => false,
      };
    } else {
      m |= match n.ty {
        Ty::Field {
          src: Src::Plain | Src::Elem | Src::XfemPlus | Src::XfemMinus,
          test: false,
          ..
        } => n.name == self.var && (self.any_trans || self.trans.is_empty()),
        Ty::Field {
          src: Src::Interp,
          test,
          ..
        } => {
          (!test && n.name == self.var && self.trans_is(&n.interpolate_name))
            || self.dep(&n.interpolate_name)
        }
        Ty::InterpX | Ty::InterpNormal | Ty::InterpEltK | Ty::InterpEltB => {
          self.dep(&n.interpolate_name)
        }
        // not marked: second order transformation derivatives are ignored
        // (as GetFEM)
        Ty::InterpolateDerivative => false,
        _ => false,
      };
    }
    self.marked[i] = m;
    m
  }

  fn cst(&mut self, sizes: &[usize], data: Vec<f64>, pos: usize) -> usize {
    let k = self.node(Ty::Constant, pos, &[]);
    self.t.n[k].sizes = sizes.to_vec();
    self.t.n[k].data = data;
    self.t.n[k].test = 0;
    k
  }

  fn int(&mut self, v: usize, pos: usize) -> usize {
    self.cst(&[], vec![v as f64], pos)
  }

  fn ones(&mut self, pos: usize) -> usize {
    let n = self.meshdim.unwrap();
    self.cst(&[n], vec![1.0; n], pos)
  }

  /// f(args) with an operator node of type ty (Contract, Index_move_last, ...)
  fn special(&mut self, ty: Ty, args: &[usize], pos: usize) -> usize {
    let f = self.node(ty, pos, &[]);
    self.t.n[f].test = 0;
    let mut ch = vec![f];
    ch.extend(args);
    self.node(Ty::Params, pos, &ch)
  }

  fn proper_size(&self, i: usize) -> usize {
    self.t.n[i].tensor_proper_size()
  }

  fn order_of(&self, i: usize) -> usize {
    self.t.n[i].tensor_order()
  }

  /// spatial gradient of node i (marked)
  fn g(&mut self, i: usize) -> Result<usize> {
    let n = self.t.n[i].clone();
    let pos = n.pos;
    let ch = n.children.clone();
    let big_n = self.meshdim.unwrap();
    match n.ty {
      Ty::Field {
        src: src @ (Src::Plain | Src::XfemPlus | Src::XfemMinus),
        der,
        test,
      } => {
        let t = match n.test {
          0 => "",
          1 => "Test_",
          _ => "Test2_",
        };
        let t = if test { t } else { "" };
        match der {
          Der::Val | Der::Grad => {
            let p = if der == Der::Val { "Grad_" } else { "Hess_" };
            let g = self.field(src, &format!("{p}{t}{}", n.name), pos);
            // GetFEM ga_node_grad sizes: a last index of size N is added
            // (1D: Grad_u, Hess_u of a vector do not have it)
            if big_n > 1 || n.sizes.last() == Some(&1) {
              return Ok(g);
            }
            let skip = usize::from(n.test > 0 && test);
            let mut args = vec![g];
            for &k in n.sizes[skip..].iter().chain(&[1]) {
              args.push(self.int(k, pos));
            }
            Ok(self.special(Ty::Reshape, &args, pos))
          }
          // grad(div u)_k = sum_i d_k d_i u_i (GetFEM: Id:Hess_u)
          Der::Div => {
            let h = self.field(src, &format!("Hess_{t}{}", n.name), pos);
            let (a, b) = (self.int(1, pos), self.int(2, pos));
            Ok(self.special(Ty::Contract, &[h, a, b], pos))
          }
          Der::Hess => Err(Error::Semantic {
            expr: self.t.expr.clone(),
            pos,
            msg: "Sorry, cannot derive an Hessian once more".into(),
          }),
        }
      }
      Ty::X => {
        if n.nbc1 > 0 {
          let mut e = vec![0.0; big_n];
          e[n.nbc1 - 1] = 1.0;
          Ok(self.cst(&[big_n], e, pos))
        } else {
          let mut e = vec![0.0; big_n * big_n];
          for k in 0..big_n {
            e[k * (big_n + 1)] = 1.0;
          }
          Ok(self.cst(&[big_n, big_n], e, pos))
        }
      }
      Ty::Normal | Ty::InterpNormal => Err(Error::Semantic {
        expr: self.t.expr.clone(),
        pos,
        msg: "Sorry, Gradient of Normal vector not implemented".into(),
      }),
      Ty::Op(tok) => match tok {
        Tok::Plus | Tok::Minus => {
          let (a, b) = (ch[0], ch[1]);
          match (self.is_marked(a), self.is_marked(b)) {
            (true, true) => {
              let (ga, gb) = (self.g(a)?, self.g(b)?);
              Ok(self.op(tok, pos, &[ga, gb]))
            }
            (true, false) => self.g(a),
            _ => {
              let gb = self.g(b)?;
              Ok(if tok == Tok::Minus {
                self.op(Tok::UnaryMinus, pos, &[gb])
              } else {
                gb
              })
            }
          }
        }
        Tok::UnaryMinus => {
          let ga = self.g(ch[0])?;
          Ok(self.op(tok, pos, &[ga]))
        }
        Tok::Quote => {
          let a = ch[0];
          if self.order_of(a) == 1 {
            let nn = self.t.n[a].proper_size(0);
            let ga = self.g(a)?;
            let (x, y, z) = (self.int(1, pos), self.int(nn, pos), self.int(big_n, pos));
            Ok(self.special(Ty::Reshape, &[ga, x, y, z], pos))
          } else {
            let ga = self.g(a)?;
            Ok(self.op(Tok::Quote, pos, &[ga]))
          }
        }
        Tok::Sym | Tok::Skew => {
          let (g1, g2) = (self.g(ch[0])?, self.g(ch[0])?);
          let q = self.op(Tok::Quote, pos, &[g2]);
          let s = self.op(
            if tok == Tok::Sym {
              Tok::Plus
            } else {
              Tok::Minus
            },
            pos,
            &[g1, q],
          );
          let h = self.cst(&[], vec![0.5], pos);
          Ok(self.op(Tok::Mult, pos, &[s, h]))
        }
        Tok::Trace => {
          let ga = self.g(ch[0])?;
          let (x, y) = (self.int(1, pos), self.int(2, pos));
          Ok(self.special(Ty::Contract, &[ga, x, y], pos))
        }
        Tok::Deviator => {
          // grad(T - Trace(T) Id/m) with m the size of T
          let m = self.t.n[ch[0]].proper_size(0);
          let (g1, g2) = (self.g(ch[0])?, self.g(ch[0])?);
          let (x, y) = (self.int(1, pos), self.int(2, pos));
          let tr = self.special(Ty::Contract, &[g2, x, y], pos);
          let mut id = vec![0.0; m * m];
          for k in 0..m {
            id[k * (m + 1)] = 1.0 / m as f64;
          }
          let idc = self.cst(&[m, m], id, pos);
          let p = self.op(Tok::TMult, pos, &[idc, tr]);
          Ok(self.op(Tok::Minus, pos, &[g1, p]))
        }
        Tok::Dot | Tok::Mult | Tok::Colon | Tok::TMult | Tok::DotMult => self.g_product(i, tok),
        Tok::Div | Tok::DotDiv => {
          let (a, b) = (ch[0], ch[1]);
          let mut terms = vec![];
          if self.is_marked(a) {
            let (ga, cb) = (self.g(a)?, self.copy(b));
            let den = if tok == Tok::DotDiv {
              let o = self.ones(pos);
              self.op(Tok::TMult, pos, &[cb, o])
            } else {
              cb
            };
            terms.push(self.op(tok, pos, &[ga, den]));
          }
          if self.is_marked(b) {
            // (GetFEM ga_node_grad: sqr(b), a constant a negated, t1 - m)
            let (ca, b1) = (self.copy(a), self.copy(b));
            let cst = self.t.n[ca].ty == Ty::Constant;
            if cst {
              self.t.n[ca].data.iter_mut().for_each(|x| *x = -*x);
            }
            let b2 = self.call_fn("sqr", &[b1], pos);
            let q = self.op(
              if tok == Tok::Div {
                Tok::Div
              } else {
                Tok::DotDiv
              },
              pos,
              &[ca, b2],
            );
            let gb = self.g(b)?;
            let m = if tok == Tok::Div {
              self.op(Tok::TMult, pos, &[q, gb])
            } else {
              let o = self.ones(pos);
              let qo = self.op(Tok::TMult, pos, &[q, o]);
              self.op(Tok::DotMult, pos, &[qo, gb])
            };
            return Ok(match terms.pop() {
              Some(t1) => self.op(Tok::Minus, pos, &[t1, m]),
              None if cst => m,
              None => self.op(Tok::UnaryMinus, pos, &[m]),
            });
          }
          Ok(self.sum(terms, pos))
        }
        _ => Err(self.unsupported(i, &format!("gradient of operator {tok:?}"))),
      },
      Ty::CMatrix => {
        let nt = n.nb_test_functions();
        let mut m = n.clone();
        m.sizes = n.sizes[nt..].to_vec();
        if big_n > 1 {
          m.sizes.push(big_n);
          m.nbc1 += 1;
        }
        m.data = vec![0.0; m.sizes.iter().product()];
        m.children.clear();
        m.parent = None;
        m.test = usize::MAX;
        let reps = if big_n > 1 { big_n } else { 1 };
        let mut kids = vec![];
        for d in 0..reps {
          for &c in &ch {
            kids.push(if self.is_marked(c) {
              let gc = self.g(c)?;
              if big_n > 1 {
                let k = self.int(d + 1, pos);
                self.node(Ty::Params, pos, &[gc, k])
              } else {
                gc
              }
            } else {
              self.scalar(0.0, true, pos)
            });
          }
        }
        let k = self.t.push(m);
        for c in kids {
          self.t.adopt(k, c);
        }
        Ok(k)
      }
      // GetFEM: the gradient of the filtered expression
      Ty::InterpolateFilter => self.with_child(i, 0, Self::g),
      Ty::Params => self.g_params(i, &n, &ch),
      _ => Err(self.unsupported(i, &format!("gradient of {:?}", n.ty))),
    }
  }

  /// gradient of a bilinear operation (GetFEM ga_node_grad)
  fn g_product(&mut self, i: usize, tok: Tok) -> Result<usize> {
    let n = self.t.n[i].clone();
    let pos = n.pos;
    let (a, b) = (n.children[0], n.children[1]);
    let (pa, pb) = (self.proper_size(a), self.proper_size(b));
    let (ra, rb) = (self.order_of(a), self.order_of(b));
    // a op a: (a op grad(a)) 2 (as the derivative)
    if self.is_marked(a)
      && self.is_marked(b)
      && self.t.same(a, b)
      && !(matches!(tok, Tok::Mult | Tok::Dot) && ra >= 2)
    {
      let (ca, gb) = (self.copy(a), self.g(b)?);
      let t = self.g_second(tok, ca, gb, pa, pb, ra, pos);
      let two = self.scalar(2.0, false, pos);
      return Ok(self.op(Tok::Mult, pos, &[t, two]));
    }
    let mut terms = vec![];
    if self.is_marked(a) {
      let swap = (tok == Tok::Colon && ra == 2)
        || (tok == Tok::Dot && ra == 1 && rb == 1)
        || tok == Tok::DotMult
        || pa == 1
        || pb == 1;
      if swap {
        // symmetric use: b op grad(a)
        let (cb, ga) = (self.copy(b), self.g(a)?);
        terms.push(self.g_second(tok, cb, ga, pb, pa, rb, pos));
      } else {
        let ga = self.g(a)?;
        let cb = self.copy(b);
        let (t, nred) = if tok == Tok::TMult {
          (self.op(Tok::TMult, pos, &[ga, cb]), ra + 1)
        } else if (ra <= 2 && tok == Tok::Mult) || tok == Tok::Dot {
          let (x, y) = (self.int(ra, pos), self.int(1, pos));
          (self.special(Ty::Contract, &[ga, x, cb, y], pos), ra)
        } else {
          let (x1, x2, y1, y2) = (
            self.int(ra - 1, pos),
            self.int(ra, pos),
            self.int(1, pos),
            self.int(2, pos),
          );
          (
            self.special(Ty::Contract, &[ga, x1, x2, cb, y1, y2], pos),
            ra - 1,
          )
        };
        let k = self.int(nred, pos);
        terms.push(self.special(Ty::IndMoveLast, &[t, k], pos));
      }
    }
    if self.is_marked(b) {
      let (ca, gb) = (self.copy(a), self.g(b)?);
      terms.push(self.g_second(tok, ca, gb, pa, pb, ra, pos));
    }
    Ok(self.sum(terms, pos))
  }

  /// x op grad(y), where x is left as is (GetFEM's pg2 branch); px, py are
  /// proper sizes, rx the order of x
  #[allow(clippy::too_many_arguments)]
  fn g_second(
    &mut self,
    tok: Tok,
    x: usize,
    gy: usize,
    px: usize,
    py: usize,
    rx: usize,
    pos: usize,
  ) -> usize {
    match tok {
      Tok::Mult | Tok::Colon | Tok::Dot => {
        if py == 1 {
          self.op(if px == 1 { Tok::Mult } else { Tok::TMult }, pos, &[x, gy])
        } else if px == 1 {
          self.op(Tok::Mult, pos, &[x, gy])
        } else if (rx <= 2 && tok == Tok::Mult) || tok == Tok::Dot {
          self.op(Tok::Dot, pos, &[x, gy])
        } else {
          let (x1, x2, y1, y2) = (
            self.int(rx - 1, pos),
            self.int(rx, pos),
            self.int(1, pos),
            self.int(2, pos),
          );
          self.special(Ty::Contract, &[x, x1, x2, gy, y1, y2], pos)
        }
      }
      Tok::TMult => self.op(Tok::TMult, pos, &[x, gy]),
      _ => {
        // DotMult
        if px == 1 {
          self.op(Tok::Mult, pos, &[x, gy])
        } else {
          let o = self.ones(pos);
          let xo = self.op(Tok::TMult, pos, &[x, o]);
          self.op(Tok::DotMult, pos, &[xo, gy])
        }
      }
    }
  }

  fn g_params(&mut self, i: usize, n: &Node, ch: &[usize]) -> Result<usize> {
    let pos = n.pos;
    let f = self.t.n[ch[0]].clone();
    let big_n = self.meshdim.unwrap();
    let rebuild = |s: &mut Self, gk: usize, k: usize| -> Vec<usize> {
      ch.iter()
        .enumerate()
        .map(|(j, &c)| if j == k { gk } else { s.copy(c) })
        .collect()
    };
    match f.ty {
      Ty::Reshape => {
        let g1 = self.g(ch[1])?;
        let mut kids = rebuild(self, g1, 1);
        kids.push(self.int(big_n, pos));
        Ok(self.node(Ty::Params, pos, &kids))
      }
      Ty::SwapInd => {
        let g1 = self.g(ch[1])?;
        let kids = rebuild(self, g1, 1);
        Ok(self.node(Ty::Params, pos, &kids))
      }
      Ty::IndMoveLast => {
        let order = self.order_of(i);
        let g1 = self.g(ch[1])?;
        let kids = rebuild(self, g1, 1);
        let inner = self.node(Ty::Params, pos, &kids);
        let (x, y) = (self.int(order, pos), self.int(order + 1, pos));
        Ok(self.special(Ty::SwapInd, &[inner, x, y], pos))
      }
      Ty::Contract if ch.len() == 4 => {
        let g1 = self.g(ch[1])?;
        let kids = rebuild(self, g1, 1);
        Ok(self.node(Ty::Params, pos, &kids))
      }
      Ty::Contract => {
        let k2 = if ch.len() == 5 { 3 } else { 4 };
        let mut terms = vec![];
        if self.is_marked(ch[1]) {
          let mut nred = self.order_of(ch[1]);
          if ch.len() == 7 {
            nred -= 1;
          }
          let g1 = self.g(ch[1])?;
          let kids = rebuild(self, g1, 1);
          let c = self.node(Ty::Params, pos, &kids);
          let k = self.int(nred, pos);
          terms.push(self.special(Ty::IndMoveLast, &[c, k], pos));
        }
        if self.is_marked(ch[k2]) {
          let g2 = self.g(ch[k2])?;
          let kids = rebuild(self, g2, k2);
          terms.push(self.node(Ty::Params, pos, &kids));
        }
        Ok(self.sum(terms, pos))
      }
      Ty::CrossProduct => Err(Error::Semantic {
        expr: self.t.expr.clone(),
        pos,
        msg: "Sorry, the gradient of a cross product has not been implemented".into(),
      }),
      Ty::PredefFunc => {
        let mut terms = vec![];
        for k in 1..ch.len() {
          if !self.is_marked(ch[k]) {
            continue;
          }
          let dname = self
            .defs
            .derivative_name(&f.name, k)
            .ok_or_else(|| Error::Semantic {
              expr: self.t.expr.clone(),
              pos,
              msg: format!("{} cannot be differentiated", f.name),
            })?;
          let args: Vec<usize> = ch[1..].iter().map(|&c| self.copy(c)).collect();
          let fp = match self.inline_der(&f.name, k, &args) {
            Some(d) => d,
            None => self.call_der(&dname, &args, pos),
          };
          let gk = self.g(ch[k])?;
          terms.push(if self.order_of(ch[k]) == 0 {
            self.op(Tok::Mult, pos, &[fp, gk])
          } else {
            let o = self.ones(pos);
            let fo = self.op(Tok::TMult, pos, &[fp, o]);
            self.op(Tok::DotMult, pos, &[fo, gk])
          });
        }
        Ok(self.sum(terms, pos))
      }
      Ty::Operator => {
        if f.der2 != 0 {
          return Err(Error::Semantic {
            expr: self.t.expr.clone(),
            pos,
            msg: format!(
              "Error in derivation of the assembly string. Cannot derive again operator {}",
              f.name
            ),
          });
        }
        let mut terms = vec![];
        for k in 1..ch.len() {
          if !self.is_marked(ch[k]) {
            continue;
          }
          let mut op = f.clone();
          op.children.clear();
          op.parent = None;
          if op.der1 == 0 {
            op.der1 = k;
          } else {
            op.der2 = k;
          }
          if let Some(nl) = self.defs.operators.get(&f.name)
            && !nl.has_derivative((op.der1, op.der2))
          {
            return Err(Error::Semantic {
              expr: self.t.expr.clone(),
              pos,
              msg: format!(
                "Error in derivation of the assembly string. {} derivative of operator {} w.r.t. its argument {k} not yet available",
                if op.der2 == 0 { "The" } else { "The second" },
                f.name
              ),
            });
          }
          let call = self.op_der(op, ch, pos);
          let red = match self.order_of(ch[k]) {
            0 => Tok::Mult,
            1 => Tok::Dot,
            2 => Tok::Colon,
            _ => return Err(self.unsupported(i, "an operator of an argument of order > 2")),
          };
          let gk = self.g(ch[k])?;
          terms.push(self.op(red, pos, &[call, gk]));
        }
        Ok(self.sum(terms, pos))
      }
      Ty::SpecFunc | Ty::Name => Err(self.unsupported(i, &f.name)),
      // component access: one more free index
      _ => {
        let g0 = self.g(ch[0])?;
        let mut kids = rebuild(self, g0, 0);
        kids.push(self.node(Ty::AllIndices, pos, &[]));
        Ok(self.node(Ty::Params, pos, &kids))
      }
    }
  }

  /// does transformation t depend on (var, trans)?
  fn dep(&self, t: &str) -> bool {
    self
      .defs
      .interpolate_transformations
      .get(t)
      .is_some_and(|d| d.iter().any(|(v, tr)| *v == self.var && self.trans_is(tr)))
  }

  /// tr is the transformation of var (any, for Diff)
  fn trans_is(&self, tr: &str) -> bool {
    self.any_trans || tr == self.trans
  }

  /// Interpolate(name, trans), analysed again later
  fn interp(&mut self, name: &str, trans: &str, pos: usize) -> usize {
    let k = self.node(Ty::Interpolate, pos, &[]);
    self.t.n[k].name = name.to_string();
    self.t.n[k].interpolate_name = trans.to_string();
    k
  }

  /// derivative of the target point of transformation t w.r.t. (var,
  /// trans), times the test functions of var: [test, n] (GetFEM
  /// Interpolate_derivative)
  fn ider(&mut self, t: &str, n: usize, pos: usize) -> usize {
    let k = self.node(Ty::InterpolateDerivative, pos, &[]);
    let nd = &mut self.t.n[k];
    nd.name = self.var.clone();
    nd.interpolate_name = self.trans.clone();
    nd.interpolate_name_der = t.to_string();
    nd.test = self.order;
    nd.sizes = if n > 1 { vec![2, n] } else { vec![2] };
    k
  }

  /// dimension of the mesh of a finite element variable
  fn mesh_dim(&self, v: &str) -> usize {
    match self.defs.var(v).map(|v| &v.support) {
      Some(Support::Fem { mesh_dim }) => *mesh_dim,
      _ => 1,
    }
  }

  /// derivative of Interpolate(u, trans) (GetFEM ga_node_derivation):
  /// the target test functions if (u, trans) is the variable, plus the
  /// gradient times the derivative of the transformation if it depends on it
  fn d_interp(&mut self, i: usize, der: Der, test: bool) -> Result<usize> {
    let n = self.t.n[i].clone();
    let (pos, tr) = (n.pos, n.interpolate_name.clone());
    if self.dir.is_some() {
      return Err(self.unsupported(i, "Diff of an interpolated field"));
    }
    let ivar = !test && n.name == self.var && self.trans_is(&tr);
    let itrans = self.dep(&tr);
    let pre = |d: Der| match d {
      Der::Val => "",
      Der::Grad => "Grad_",
      Der::Hess => "Hess_",
      Der::Div => "Div_",
    };
    let mut terms = vec![];
    if ivar {
      let t = if self.order == 1 { "Test_" } else { "Test2_" };
      terms.push(self.interp(&format!("{}{t}{}", pre(der), n.name), &tr, pos));
    }
    if itrans {
      if der == Der::Hess {
        return Err(Error::Semantic {
          expr: self.t.expr.clone(),
          pos,
          msg: "Sorry, cannot derive a hessian once more".into(),
        });
      }
      let tp = match (test, n.test) {
        (false, _) => "",
        (true, 1) => "Test_",
        (true, _) => "Test2_",
      };
      let higher = if der == Der::Val { "Grad_" } else { "Hess_" };
      let dim = self.mesh_dim(&n.name);
      let f = self.interp(&format!("{higher}{tp}{}", n.name), &tr, pos);
      let dn = self.ider(&tr, dim, pos);
      let mut p = self.op(if dim > 1 { Tok::Dot } else { Tok::Mult }, pos, &[f, dn]);
      if der == Der::Div {
        p = self.op(Tok::Trace, pos, &[p]);
      }
      terms.push(p);
    }
    Ok(self.sum(terms, pos))
  }

  fn is_marked(&self, i: usize) -> bool {
    self.marked.get(i).copied().unwrap_or(false)
  }

  fn node(&mut self, ty: Ty, pos: usize, children: &[usize]) -> usize {
    let k = self.t.push(Node::new(ty, pos));
    for &c in children {
      self.t.adopt(k, c);
    }
    k
  }

  fn copy(&mut self, i: usize) -> usize {
    self.t.copy_subtree(i)
  }

  fn op(&mut self, tok: Tok, pos: usize, args: &[usize]) -> usize {
    self.node(Ty::Op(tok), pos, args)
  }

  fn name(&mut self, s: &str, pos: usize) -> usize {
    let k = self.node(Ty::Name, pos, &[]);
    self.t.n[k].name = s.to_string();
    k
  }

  /// the field s (with its prefixes): a name, or Xfem_plus(s), Xfem_minus(s)
  fn field(&mut self, src: Src, s: &str, pos: usize) -> usize {
    let ty = match src {
      Src::XfemPlus => Ty::XfemPlus,
      Src::XfemMinus => Ty::XfemMinus,
      _ => return self.name(s, pos),
    };
    let k = self.node(ty, pos, &[]);
    self.t.n[k].name = s.to_string();
    k
  }

  fn scalar(&mut self, v: f64, zero: bool, pos: usize) -> usize {
    let k = self.node(if zero { Ty::Zero } else { Ty::Constant }, pos, &[]);
    self.t.n[k].data = vec![v];
    self.t.n[k].test = 0;
    k
  }

  /// the derivative op of an operator at the arguments ch[1..]; that of
  /// Norm_sqr as 2*x (GetFEM ga_node_derivation, ga_node_grad)
  fn op_der(&mut self, op: Node, ch: &[usize], pos: usize) -> usize {
    if op.name == "Norm_sqr" && op.der1 == 1 {
      let (two, x) = (self.scalar(2.0, false, pos), self.copy(ch[1]));
      return self.op(Tok::Mult, pos, &[two, x]);
    }
    let ok = self.t.push(op);
    let mut kids = vec![ok];
    kids.extend(ch[1..].iter().map(|&c| self.copy(c)).collect::<Vec<_>>());
    self.node(Ty::Params, pos, &kids)
  }

  /// the k-th derivative of the built-in function f at args, when given
  /// as an expression of t (and u), inlined (not GetFEM's, which inlines
  /// affine ones only): derivable again, simplified with the rest
  fn inline_der(&mut self, f: &str, k: usize, args: &[usize]) -> Option<usize> {
    let b = crate::names::function(f)?;
    let d = if k == 1 { b.der1 } else { b.der2 };
    if d.is_empty() || crate::names::function(d).is_some() {
      return None;
    }
    let e = crate::parser::parse(d, &Default::default()).ok()??;
    let r = self.t.build(&e);
    Some(self.subst_args(r, args))
  }

  /// t, u in subtree i replaced by copies of args[0], args[1]
  fn subst_args(&mut self, i: usize, args: &[usize]) -> usize {
    let n = &self.t.n[i];
    if n.ty == Ty::Name
      && n.children.is_empty()
      && let Some(j) = ["t", "u"].iter().position(|v| *v == n.name)
    {
      return self.copy(args[j]);
    }
    for k in 0..self.t.n[i].children.len() {
      let c = self.subst_args(self.t.n[i].children[k], args);
      self.t.n[i].children[k] = c;
      self.t.n[c].parent = Some(i);
    }
    i
  }

  /// the derivative f of a function at args: a t + b inlined when f is
  /// an affine expression of t (GetFEM: DER_PDFUNC_ of sqr, 2*t)
  fn call_der(&mut self, f: &str, args: &[usize], pos: usize) -> usize {
    match (args, self.affine(f)) {
      ([x], Some((a, b))) => {
        let ca = self.scalar(a, a == 0.0, pos);
        let m = self.op(Tok::Mult, pos, &[ca, *x]);
        if b == 0.0 {
          return m;
        }
        let cb = self.scalar(b.abs(), false, pos);
        self.op(if b > 0.0 { Tok::Plus } else { Tok::Minus }, pos, &[m, cb])
      }
      _ => self.call_fn(f, args, pos),
    }
  }

  /// (a, b) if the function f of one argument is a t + b (GetFEM
  /// ga_predef_function::is_affine)
  fn affine(&self, f: &str) -> Option<(f64, f64)> {
    let u = crate::defs::function(f)?;
    if u.nargs != 1 || u.native.is_some() {
      return None;
    }
    let body = u.body.as_ref()?;
    let e = crate::parser::parse(body, &Default::default()).ok()??;
    let der = |g: &str, j: usize| self.defs.derivative_name(g, j);
    if let Some(d1) = crate::symbolic::diff(&e, "t", &der).ok()?
      && crate::symbolic::diff(&d1, "t", &der).ok()?.is_some()
    {
      return None;
    }
    // f(x): the body with t = x, folded by the semantic analysis
    let at = |x: f64| -> Option<f64> {
      let mut d = self.defs.clone();
      d.macros.add("t", &format!("({x})")).ok()?;
      let ctx = crate::semantic::Ctx {
        meshdim: 1,
        ref_elt_dim: 1,
        ..Default::default()
      };
      let (t, _) = crate::semantic::analyse_str(body, &d, &ctx).ok()?;
      let n = &t.n[t.root?];
      match n.ty {
        Ty::Constant if n.data.len() == 1 => Some(n.data[0]),
        Ty::Zero => Some(0.0),
        _ => None,
      }
    };
    let b = at(0.0)?;
    Some((at(1.0)? - b, b))
  }

  /// f(args) with a function node named f
  fn call_fn(&mut self, f: &str, args: &[usize], pos: usize) -> usize {
    let fk = self.node(Ty::PredefFunc, pos, &[]);
    self.t.n[fk].name = f.to_string();
    self.t.n[fk].test = 0;
    let mut ch = vec![fk];
    ch.extend(args);
    self.node(Ty::Params, pos, &ch)
  }

  fn sum(&mut self, terms: Vec<usize>, pos: usize) -> usize {
    let mut it = terms.into_iter();
    let first = it.next().expect("at least one term");
    it.fold(first, |a, b| self.op(Tok::Plus, pos, &[a, b]))
  }

  fn unsupported(&self, i: usize, msg: &str) -> Error {
    self.t.unsupported(i, format!("derivative of {msg}"))
  }

  /// derivative of a bilinear operation on children (ka, kb) of node i;
  /// `rebuild(a, b)` makes the operation from two node indices
  fn product(
    &mut self,
    i: usize,
    ka: usize,
    kb: usize,
    rebuild: &dyn Fn(&mut Self, usize, usize) -> usize,
  ) -> Result<usize> {
    let ch = self.t.n[i].children.clone();
    let (a, b) = (ch[ka], ch[kb]);
    let mut terms = vec![];
    if self.is_marked(a) {
      let (da, cb) = (self.d(a)?, self.copy(b));
      terms.push(rebuild(self, da, cb));
    }
    if self.is_marked(b) {
      let (ca, db) = (self.copy(a), self.d(b)?);
      terms.push(rebuild(self, ca, db));
    }
    Ok(self.sum(terms, self.t.n[i].pos))
  }

  /// the node i with children replaced: marked ones by their derivative at
  /// position k, the others copied
  fn with_child_derived(&mut self, i: usize, k: usize) -> Result<usize> {
    self.with_child(i, k, Self::d)
  }

  /// a copy of node i with child k replaced by f(child)
  fn with_child(
    &mut self,
    i: usize,
    k: usize,
    f: fn(&mut Self, usize) -> Result<usize>,
  ) -> Result<usize> {
    let n = self.t.n[i].clone();
    let mut ch = vec![];
    for (j, &c) in n.children.iter().enumerate() {
      ch.push(if j == k { f(self, c)? } else { self.copy(c) });
    }
    let mut m = n.clone();
    m.children.clear();
    m.parent = None;
    let nk = self.t.push(m);
    for c in ch {
      self.t.adopt(nk, c);
    }
    Ok(nk)
  }

  fn d(&mut self, i: usize) -> Result<usize> {
    let n = self.t.n[i].clone();
    let pos = n.pos;
    let ch = n.children.clone();
    match n.ty {
      Ty::Field {
        src: Src::Plain,
        der,
        test: false,
      } => {
        if let Some(dir) = self.dir {
          return match der {
            Der::Val => Ok(self.copy(dir)),
            Der::Grad => {
              let g = self.name("Grad", pos);
              let c = self.copy(dir);
              Ok(self.node(Ty::Params, pos, &[g, c]))
            }
            Der::Hess => {
              let (g1, g2) = (self.name("Grad", pos), self.name("Grad", pos));
              let c = self.copy(dir);
              let inner = self.node(Ty::Params, pos, &[g2, c]);
              Ok(self.node(Ty::Params, pos, &[g1, inner]))
            }
            Der::Div => {
              let g = self.name("Grad", pos);
              let c = self.copy(dir);
              let gr = self.node(Ty::Params, pos, &[g, c]);
              Ok(self.op(Tok::Trace, pos, &[gr]))
            }
          };
        }
        let pre = match der {
          Der::Val => "",
          Der::Grad => "Grad_",
          Der::Hess => "Hess_",
          Der::Div => "Div_",
        };
        let t = if self.order == 1 { "Test_" } else { "Test2_" };
        Ok(self.name(&format!("{pre}{t}{}", n.name), pos))
      }
      Ty::Field {
        src: Src::Interp,
        der,
        test,
      } => self.d_interp(i, der, test),
      Ty::Field {
        src: Src::Elem,
        der,
        test: false,
      } => {
        if self.dir.is_some() {
          return Err(self.unsupported(i, "directional derivatives of elementary transformations"));
        }
        let pre = match der {
          Der::Val => "",
          Der::Grad => "Grad_",
          Der::Hess => "Hess_",
          Der::Div => "Div_",
        };
        let t = if self.order == 1 { "Test_" } else { "Test2_" };
        let k = self.node(Ty::Elementary, pos, &[]);
        self.t.n[k].name = format!("{pre}{t}{}", n.name);
        self.t.n[k].elementary_name = n.elementary_name.clone();
        self.t.n[k].elementary_target = n.elementary_target.clone();
        Ok(k)
      }
      Ty::Field {
        src: src @ (Src::XfemPlus | Src::XfemMinus),
        der,
        test: false,
      } => {
        // GetFEM: no directional derivative of Xfem_plus, Xfem_minus
        if self.dir.is_some() {
          return Err(self.unsupported(i, "directional derivatives of Xfem_plus and Xfem_minus"));
        }
        let pre = match der {
          Der::Val => "",
          Der::Grad => "Grad_",
          Der::Hess => "Hess_",
          Der::Div => "Div_",
        };
        let t = if self.order == 1 { "Test_" } else { "Test2_" };
        Ok(self.field(src, &format!("{pre}{t}{}", n.name), pos))
      }
      Ty::Field { .. } => Err(self.unsupported(i, "test functions of elementary transformations")),
      Ty::InterpX => {
        let dim = n.sizes.last().copied().unwrap_or(1);
        Ok(self.ider(&n.interpolate_name, dim, pos))
      }
      Ty::InterpNormal => Err(self.unsupported(i, "the interpolated Normal")),
      Ty::InterpEltK => Err(self.unsupported(i, "the interpolated element_K")),
      Ty::InterpEltB => Err(self.unsupported(i, "the interpolated element_B")),
      Ty::InterpolateDerivative => Err(self.unsupported(
        i,
        "a transformation derivative (second order transformation derivatives)",
      )),
      Ty::Op(tok) => match tok {
        Tok::Plus | Tok::Minus => {
          let (a, b) = (ch[0], ch[1]);
          match (self.is_marked(a), self.is_marked(b)) {
            (true, true) => {
              let (da, db) = (self.d(a)?, self.d(b)?);
              Ok(self.op(tok, pos, &[da, db]))
            }
            (true, false) => self.d(a),
            _ => {
              let db = self.d(b)?;
              Ok(if tok == Tok::Minus {
                self.op(Tok::UnaryMinus, pos, &[db])
              } else {
                db
              })
            }
          }
        }
        Tok::UnaryMinus | Tok::Quote | Tok::Sym | Tok::Skew | Tok::Trace | Tok::Deviator => {
          let da = self.d(ch[0])?;
          Ok(self.op(tok, pos, &[da]))
        }
        Tok::Dot | Tok::Mult | Tok::Colon | Tok::TMult | Tok::DotMult => {
          let (a, b) = (ch[0], ch[1]);
          if !(self.is_marked(a) && self.is_marked(b)) {
            return self.product(i, 0, 1, &|s, a, b| s.op(tok, pos, &[a, b]));
          }
          // GetFEM ga_node_derivation: a op a as (a op da) 2, else the
          // first term as b op da where op commutes (equal terms up to the
          // test functions: compiled once, transposed)
          let (na, nb) = (&self.t.n[a], &self.t.n[b]);
          let ra = na.tensor_order();
          if self.t.same(a, b) && !(matches!(tok, Tok::Mult | Tok::Dot) && ra >= 2) {
            let (ca, db) = (self.copy(a), self.d(b)?);
            let t = self.op(tok, pos, &[ca, db]);
            let two = self.scalar(2.0, false, pos);
            return Ok(self.op(Tok::Mult, pos, &[t, two]));
          }
          let swap = (tok == Tok::Colon && ra == 2)
            || (tok == Tok::Dot && ra == 1 && nb.tensor_order() == 1)
            || tok == Tok::DotMult
            || (na.tensor_proper_size() == 1 && nb.tensor_proper_size() == 1);
          let (cb, da) = (self.copy(b), self.d(a)?);
          let t1 = if swap {
            self.op(tok, pos, &[cb, da])
          } else {
            self.op(tok, pos, &[da, cb])
          };
          let (ca, db) = (self.copy(a), self.d(b)?);
          let t2 = self.op(tok, pos, &[ca, db]);
          Ok(self.op(Tok::Plus, pos, &[t1, t2]))
        }
        Tok::Div | Tok::DotDiv => {
          let (a, b) = (ch[0], ch[1]);
          let (mul, div) = if tok == Tok::Div {
            (Tok::Mult, Tok::Div)
          } else {
            (Tok::DotMult, Tok::DotDiv)
          };
          let mut terms = vec![];
          if self.is_marked(a) {
            let (da, cb) = (self.d(a)?, self.copy(b));
            terms.push(self.op(div, pos, &[da, cb]));
          }
          if self.is_marked(b) {
            // - a / sqr(b) * db (GetFEM: a constant a negated, da / b - m)
            let (ca, b1) = (self.copy(a), self.copy(b));
            let cst = self.t.n[ca].ty == Ty::Constant;
            if cst {
              self.t.n[ca].data.iter_mut().for_each(|x| *x = -*x);
            }
            let b2 = self.call_fn("sqr", &[b1], pos);
            let q = self.op(div, pos, &[ca, b2]);
            let db = self.d(b)?;
            let m = self.op(mul, pos, &[q, db]);
            return Ok(match terms.pop() {
              Some(t1) => self.op(Tok::Minus, pos, &[t1, m]),
              None if cst => m,
              None => self.op(Tok::UnaryMinus, pos, &[m]),
            });
          }
          Ok(self.sum(terms, pos))
        }
        _ => Err(self.unsupported(i, &format!("operator {tok:?}"))),
      },
      Ty::CMatrix => {
        let mut m = n.clone();
        let nt = n.nb_test_functions();
        m.sizes = n.sizes[nt..].to_vec();
        m.data = vec![0.0; m.sizes.iter().product()];
        m.children.clear();
        m.parent = None;
        m.test = usize::MAX;
        let mut kids = vec![];
        for &c in &ch {
          kids.push(if self.is_marked(c) {
            self.d(c)?
          } else {
            self.scalar(0.0, true, pos)
          });
        }
        let k = self.t.push(m);
        for c in kids {
          self.t.adopt(k, c);
        }
        Ok(k)
      }
      Ty::Params => self.params(i, &n, &ch),
      // the filter of the derivative (GetFEM)
      Ty::InterpolateFilter => self.with_child_derived(i, 0),
      _ => Err(self.unsupported(i, &format!("{:?}", n.ty))),
    }
  }

  fn params(&mut self, i: usize, n: &Node, ch: &[usize]) -> Result<usize> {
    let pos = n.pos;
    let f = self.t.n[ch[0]].clone();
    match f.ty {
      Ty::Reshape | Ty::SwapInd | Ty::IndMoveLast => self.with_child_derived(i, 1),
      Ty::Contract if ch.len() == 4 => self.with_child_derived(i, 1),
      Ty::Contract | Ty::CrossProduct => {
        let kb = match (f.ty, ch.len()) {
          (Ty::CrossProduct, _) => 2,
          (_, 5) => 3,
          _ => 4,
        };
        let shape: Vec<usize> = ch.to_vec();
        self.product(i, 1, kb, &|s, a, b| {
          let mut kids = vec![];
          for (j, &c) in shape.iter().enumerate() {
            kids.push(if j == 1 {
              a
            } else if j == kb {
              b
            } else {
              s.copy(c)
            });
          }
          s.node(Ty::Params, pos, &kids)
        })
      }
      Ty::PredefFunc => {
        let mut terms = vec![];
        for k in 1..ch.len() {
          if !self.is_marked(ch[k]) {
            continue;
          }
          let dname = self
            .defs
            .derivative_name(&f.name, k)
            .ok_or_else(|| Error::Semantic {
              expr: self.t.expr.clone(),
              pos,
              msg: format!("{} cannot be differentiated", f.name),
            })?;
          let args: Vec<usize> = ch[1..].iter().map(|&c| self.copy(c)).collect();
          let fp = match self.inline_der(&f.name, k, &args) {
            Some(d) => d,
            None => self.call_der(&dname, &args, pos),
          };
          let order = self.t.n[ch[k]].tensor_order();
          let da = self.d(ch[k])?;
          terms.push(self.op(
            if order == 0 { Tok::Mult } else { Tok::DotMult },
            pos,
            &[fp, da],
          ));
        }
        Ok(self.sum(terms, pos))
      }
      Ty::Operator => {
        if f.der2 != 0 {
          return Err(Error::Semantic {
            expr: self.t.expr.clone(),
            pos,
            msg: format!(
              "Error in derivation of the assembly string. Cannot derive again operator {}",
              f.name
            ),
          });
        }
        let mut terms = vec![];
        for k in 1..ch.len() {
          if !self.is_marked(ch[k]) {
            continue;
          }
          let mut op = f.clone();
          op.children.clear();
          op.parent = None;
          if op.der1 == 0 {
            op.der1 = k;
          } else {
            op.der2 = k;
          }
          if let Some(nl) = self.defs.operators.get(&f.name)
            && !nl.has_derivative((op.der1, op.der2))
          {
            return Err(Error::Semantic {
              expr: self.t.expr.clone(),
              pos,
              msg: format!(
                "Error in derivation of the assembly string. {} derivative of operator {} w.r.t. its argument {k} not yet available",
                if op.der2 == 0 { "The" } else { "The second" },
                f.name
              ),
            });
          }
          let call = self.op_der(op, ch, pos);
          let red = match self.t.n[ch[k]].tensor_order() {
            0 => Tok::Mult,
            1 => Tok::Dot,
            2 => Tok::Colon,
            _ => return Err(self.unsupported(i, "an operator of an argument of order > 2")),
          };
          let da = self.d(ch[k])?;
          terms.push(self.op(red, pos, &[call, da]));
        }
        Ok(self.sum(terms, pos))
      }
      Ty::SpecFunc | Ty::Name => Err(self.unsupported(i, &f.name)),
      // component access A(i, :)
      _ => self.with_child_derived(i, 0),
    }
  }
}
