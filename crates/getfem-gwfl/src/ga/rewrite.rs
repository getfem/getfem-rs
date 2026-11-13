//! Contractions pushed into tensor products (compile time, on a copy of the
//! analysed tree, which stays GetFEM's): with c without test functions,
//! contracting the last indices of x ⊗ y,
//!
//! - (x ⊗ y) : c = x ⊗ (y : c) if the contracted indices are y's,
//! - (x ⊗ y) : c = (x : c) ⊗ y if y is a scalar (per test function),
//! - (s A) : c = s (A : c) for a scalar s,
//!
//! the same for `.` (one index). `((Grad_Test2_u @ Grad_Test_u) : Id) : Id`
//! becomes `tr(Grad_Test2_u) ⊗ tr(Grad_Test_u)`: an outer product of
//! vectors instead of the product of the gradients (81 times larger in 3D).

use crate::lexer::Tok;
use crate::semantic::{Node, Tree, Ty};

// test only: the rewrite off (the reference of the tests)
#[cfg(test)]
thread_local! {
  pub(crate) static OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// the trees rewritten, None if nothing applies to any
pub(crate) fn all(trees: &[Tree]) -> Option<Vec<Tree>> {
  #[cfg(test)]
  if OFF.with(|o| o.get()) {
    return None;
  }
  let rw: Vec<Option<Tree>> = trees.iter().map(push_contractions).collect();
  rw.iter().any(Option::is_some).then(|| {
    rw.into_iter()
      .zip(trees)
      .map(|(r, t)| r.unwrap_or_else(|| t.clone()))
      .collect()
  })
}

/// the rewritten tree, None if nothing applies
fn push_contractions(t: &Tree) -> Option<Tree> {
  let r = t.root?;
  let mut w = t.clone();
  let mut changed = false;
  walk(&mut w, r, &mut changed);
  changed.then_some(w)
}

fn walk(t: &mut Tree, i: usize, changed: &mut bool) {
  for c in t.n[i].children.clone() {
    walk(t, c, changed);
  }
  rule(t, i, changed);
}

/// number of contracted indices of a contraction node with these operands
fn contracted(t: &Tree, i: usize) -> Option<(usize, usize, usize)> {
  let k = match t.n[i].ty {
    Ty::Op(Tok::Colon) => 2,
    Ty::Op(Tok::Dot) => 1,
    _ => return None,
  };
  let [a, c] = t.n[i].children[..] else {
    return None;
  };
  let nc = &t.n[c];
  (nc.test == 0 && nc.tensor_order() == k).then_some((k, a, c))
}

fn order(t: &Tree, i: usize) -> usize {
  t.n[i].tensor_order()
}

fn scalar(t: &Tree, i: usize) -> bool {
  t.n[i].test == 0 && t.n[i].tensor_order() == 0
}

fn rule(t: &mut Tree, i: usize, changed: &mut bool) {
  let Some((k, a, c)) = contracted(t, i) else {
    return;
  };
  let op = t.n[i].ty;
  // the rules below only for tensors with tests (without, nothing to gain)
  let tests = t.n[a].nb_test_functions() > 0;
  // a sum, a difference, a negation, a division by a scalar: the
  // contraction of each term ((x + y) : c = x : c + y : c)
  match (t.n[a].ty, &t.n[a].children[..]) {
    _ if !tests => {}
    (Ty::Op(Tok::Plus | Tok::Minus), &[x, y]) => {
      let c2 = t.copy_subtree(c);
      let (zx, zy) = (contraction(t, op, i, x, c), contraction(t, op, i, y, c2));
      return distributed(t, i, t.n[a].ty, vec![zx, zy], &[zx, zy], changed);
    }
    (Ty::Op(Tok::UnaryMinus), &[x]) => {
      let z = contraction(t, op, i, x, c);
      return distributed(t, i, t.n[a].ty, vec![z], &[z], changed);
    }
    (Ty::Op(Tok::Div), &[x, y]) if scalar(t, y) => {
      let z = contraction(t, op, i, x, c);
      return distributed(t, i, t.n[a].ty, vec![z, y], &[z], changed);
    }
    _ => {}
  }
  let [x, y] = t.n[a].children[..] else {
    return;
  };
  // a matrix product: the test-free factor moved onto c, (x M) : c =
  // x : (c M'), (M y) : c = y : (M' c); with tests on both, (x y) : c =
  // x : (c y'), a contraction of the tests of x with those of c y'
  if tests && k == 2 && t.n[a].ty == Ty::Op(Tok::Mult) && order(t, x) == 2 && order(t, y) == 2 {
    let (sx, sy) = (proper(t, x), proper(t, y));
    let (keep, w) = if t.n[y].test == 0 {
      let mt = new_node(t, Ty::Op(Tok::Quote), i, None, &[sy[1], sy[0]], vec![y]);
      (
        x,
        new_node(t, Ty::Op(Tok::Mult), i, None, &[sx[0], sx[1]], vec![c, mt]),
      )
    } else if t.n[x].test == 0 {
      let mt = new_node(t, Ty::Op(Tok::Quote), i, None, &[sx[1], sx[0]], vec![x]);
      (
        y,
        new_node(t, Ty::Op(Tok::Mult), i, None, &[sy[0], sy[1]], vec![mt, c]),
      )
    } else {
      let yt = new_node(t, Ty::Op(Tok::Quote), i, Some(y), &[sy[1], sy[0]], vec![y]);
      (
        x,
        new_node(
          t,
          Ty::Op(Tok::Mult),
          i,
          Some(y),
          &[sx[0], sx[1]],
          vec![c, yt],
        ),
      )
    };
    t.n[i].children = vec![keep, w];
    t.n[keep].parent = Some(i);
    t.n[w].parent = Some(i);
    *changed = true;
    if t.n[w].test == 0 {
      rule(t, i, changed);
    }
    return;
  }
  let (ty, left, right) = match t.n[a].ty {
    Ty::Op(Tok::Mult) if order(t, x) == 0 => {
      let z = contraction(t, op, i, y, c);
      (Ty::Op(Tok::Mult), x, z)
    }
    Ty::Op(Tok::Mult) if order(t, y) == 0 => {
      let z = contraction(t, op, i, x, c);
      (Ty::Op(Tok::Mult), y, z)
    }
    Ty::Op(Tok::TMult) if order(t, y) >= k => {
      let z = contraction(t, op, i, y, c);
      (Ty::Op(Tok::TMult), x, z)
    }
    Ty::Op(Tok::TMult) if order(t, y) == 0 && order(t, x) >= k => {
      let z = contraction(t, op, i, x, c);
      (Ty::Op(Tok::TMult), z, y)
    }
    _ => return,
  };
  t.n[i].ty = ty;
  t.n[i].children = vec![left, right];
  t.n[left].parent = Some(i);
  t.n[right].parent = Some(i);
  *changed = true;
  // the new contraction may push further
  let z = if left == x || left == y { right } else { left };
  rule(t, z, changed);
}

/// node i becomes ty over children; the new contractions zs may push
/// further
fn distributed(
  t: &mut Tree,
  i: usize,
  ty: Ty,
  children: Vec<usize>,
  zs: &[usize],
  changed: &mut bool,
) {
  t.n[i].ty = ty;
  for &ch in &children {
    t.n[ch].parent = Some(i);
  }
  t.n[i].children = children;
  *changed = true;
  for &z in zs {
    rule(t, z, changed);
  }
}

/// the proper sizes of node i (without the test indices)
fn proper(t: &Tree, i: usize) -> Vec<usize> {
  let n = &t.n[i];
  n.sizes[n.sizes.len() - n.tensor_order()..].to_vec()
}

/// a new node ty over children: the tests of node tests (none if None),
/// then the proper sizes
fn new_node(
  t: &mut Tree,
  ty: Ty,
  at: usize,
  tests: Option<usize>,
  proper: &[usize],
  children: Vec<usize>,
) -> usize {
  let mut z = Node::new(ty, t.n[at].pos);
  z.test = 0;
  if let Some(b) = tests {
    let nb = &t.n[b];
    z.sizes = nb.sizes[..nb.nb_test_functions()].to_vec();
    z.test = nb.test;
    z.name_test1 = nb.name_test1.clone();
    z.name_test2 = nb.name_test2.clone();
    z.interp_test1 = nb.interp_test1.clone();
    z.interp_test2 = nb.interp_test2.clone();
    z.nbc1 = nb.nbc1;
  }
  z.sizes.extend_from_slice(proper);
  z.data = vec![0.0; z.sizes.iter().product()];
  z.parent = Some(at);
  let z = t.push(z);
  for &ch in &children {
    t.n[ch].parent = Some(z);
  }
  t.n[z].children = children;
  z
}

/// a new node b op c (op: `:` or `.` over all indices of c), the tests of b
fn contraction(t: &mut Tree, op: Ty, at: usize, b: usize, c: usize) -> usize {
  let nb = &t.n[b];
  let keep = nb.sizes.len() - t.n[c].tensor_order();
  let mut z = Node::new(op, t.n[at].pos);
  z.sizes = nb.sizes[..keep].to_vec();
  z.data = vec![0.0; z.sizes.iter().product()];
  z.test = nb.test;
  z.name_test1 = nb.name_test1.clone();
  z.name_test2 = nb.name_test2.clone();
  z.interp_test1 = nb.interp_test1.clone();
  z.interp_test2 = nb.interp_test2.clone();
  z.nbc1 = nb.nbc1;
  z.children = vec![b, c];
  z.parent = Some(at);
  let z = t.push(z);
  t.n[b].parent = Some(z);
  t.n[c].parent = Some(z);
  z
}
