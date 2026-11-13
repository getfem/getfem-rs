//! Products of scalar powers of the same base merged (not GetFEM's):
//! x*(1/x) -> 1, pow(x,a)*pow(x,b) -> pow(x,a+b), x/x -> 1. The chain of a
//! product: the factors of `*`, `/` (the denominator with the opposite
//! sign), unary `-` and the operands of `'`, `:`, `.`, `@`, `.*` with test
//! functions or a non-scalar result. Bases compared by structure (`same`).

use super::*;

/// a scalar factor base^(sign exp) at child k of parent
struct Slot {
  node: usize,
  parent: usize,
  base: usize,
  exp: f64,
  sign: f64,
}

/// an exponent within a few ulps of an integer is that integer
fn snap(k: f64) -> f64 {
  let r = k.round();
  if (k - r).abs() <= 4.0 * f64::EPSILON * k.abs().max(1.0) {
    r
  } else {
    k
  }
}

impl An<'_> {
  fn scalar(&self, i: usize) -> bool {
    let n = &self.t.n[i];
    n.test == 0 && n.tensor_proper_size() == 1
  }

  /// a node the chain passes through
  fn link(&self, i: usize) -> bool {
    match self.t.n[i].ty {
      Ty::Op(Tok::Mult | Tok::Div | Tok::UnaryMinus) => true,
      Ty::Op(Tok::Quote | Tok::Colon | Tok::Dot | Tok::TMult | Tok::DotMult) => !self.scalar(i),
      _ => false,
    }
  }

  /// (base, exponent) of a scalar factor: pow(x, c) -> (x, c), x -> (x, 1)
  fn power(&self, i: usize) -> (usize, f64) {
    let n = &self.t.n[i];
    if n.ty == Ty::Params && n.children.len() == 3 {
      let (f, c) = (&self.t.n[n.children[0]], &self.t.n[n.children[2]]);
      if f.ty == Ty::PredefFunc
        && f.name == "pow"
        && f.der1 == 0
        && f.der2 == 0
        && c.ty == Ty::Constant
        && c.data.len() == 1
      {
        return (n.children[1], c.data[0]);
      }
    }
    (i, 1.0)
  }

  fn gather(&self, i: usize, sign: f64, out: &mut Vec<Slot>) {
    let n = &self.t.n[i];
    for (k, &c) in n.children.iter().enumerate() {
      let s = if n.ty == Ty::Op(Tok::Div) && k == 1 {
        -sign
      } else {
        sign
      };
      if self.link(c) {
        self.gather(c, s, out);
      } else if self.scalar(c) && !matches!(self.t.n[c].ty, Ty::Constant | Ty::Zero) {
        let (base, exp) = self.power(c);
        out.push(Slot {
          node: c,
          parent: i,
          base,
          exp,
          sign: s,
        });
      }
    }
  }

  fn put(&mut self, parent: usize, old: usize, new: usize) {
    self.t.replace_child(parent, old, new);
    self.t.n[new].parent = Some(parent);
  }

  fn constant(&mut self, v: f64, pos: usize) -> usize {
    let mut c = Node::new(Ty::Constant, pos);
    c.init_scalar(v);
    self.t.push(c)
  }

  /// the chain of product i with equal bases merged; i if none
  pub(super) fn merge_powers(&mut self, i: usize) -> Result<usize> {
    let mut slots = vec![];
    self.gather(i, 1.0, &mut slots);
    let mut done = vec![false; slots.len()];
    let mut changed = false;
    for a in 0..slots.len() {
      if done[a] {
        continue;
      }
      let group: Vec<usize> = (a..slots.len())
        .filter(|&b| !done[b] && self.t.same(slots[a].base, slots[b].base))
        .collect();
      if group.len() < 2 {
        continue;
      }
      group.iter().for_each(|&b| done[b] = true);
      changed = true;
      let total = snap(group.iter().map(|&b| slots[b].sign * slots[b].exp).sum());
      // the merged power at a numerator slot if any
      let keep = *group
        .iter()
        .find(|&&b| slots[b].sign > 0.0)
        .unwrap_or(&group[0]);
      let s = &slots[keep];
      let (pos, place) = (self.t.n[s.node].pos, s.sign * total);
      let r = if place == 0.0 {
        self.constant(1.0, pos)
      } else if place == 1.0 {
        s.base
      } else {
        let p = self.t.push(Node::new(Ty::Params, pos));
        let mut f = Node::new(Ty::Name, pos);
        f.name = "pow".into();
        let f = self.t.push(f);
        let e = self.constant(place, pos);
        for c in [f, s.base, e] {
          self.t.adopt(p, c);
        }
        p
      };
      let (node, parent) = (s.node, s.parent);
      self.put(parent, node, r);
      for &b in &group {
        if b != keep {
          let one = self.constant(1.0, self.t.n[slots[b].node].pos);
          self.put(slots[b].parent, slots[b].node, one);
        }
      }
    }
    if changed {
      // folds the ones left (GetFEM's 1*x, x/1, constants)
      return self.node(i);
    }
    Ok(i)
  }
}
