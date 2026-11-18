//! GWFL semantic analysis (GetFEM's ga_node_analysis): resolves names
//! against a Defs store, computes shapes and test-function order, folds
//! constants. Operates on a mutable arena tree like GetFEM so that the
//! simplifications (and thus the results) are the same.

use std::collections::BTreeSet;

use crate::defs::{Defs, Support, UserFn};
use crate::lexer::Tok;
use crate::names;
use crate::parser::{Expr, ExprKind, strip_op, strip_test};
use crate::{Error, Result};

mod simplify;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Der {
  Val,
  Grad,
  Hess,
  Div,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Src {
  Plain,
  Interp,
  Elem,
  /// Xfem_plus, Xfem_minus: evaluated on the side + or - of the level set
  /// the point is on
  XfemPlus,
  XfemMinus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ty {
  Constant,
  Zero,
  AllIndices,
  Name,
  Op(Tok),
  Params,
  CMatrix,
  PredefFunc,
  SpecFunc,
  Operator,
  Reshape,
  CrossProduct,
  SwapInd,
  IndMoveLast,
  Contract,
  X,
  EltSize,
  EltK,
  EltB,
  EltCurvature,
  Normal,
  InterpX,
  InterpEltK,
  InterpEltB,
  InterpNormal,
  /// value/derivatives of a variable (test: its test function)
  Field {
    src: Src,
    der: Der,
    test: bool,
  },
  Interpolate,
  InterpolateFilter,
  InterpolateDerivative,
  Elementary,
  SecondaryDomain,
  XfemPlus,
  XfemMinus,
}

const UNDET: usize = usize::MAX;

#[derive(Clone, Debug)]
pub struct Node {
  pub ty: Ty,
  pub sizes: Vec<usize>,
  pub data: Vec<f64>,
  /// 0 none, 1 first order, 2 second order, 3 both
  pub test: usize,
  pub name_test1: String,
  pub name_test2: String,
  pub interp_test1: String,
  pub interp_test2: String,
  pub nbc1: usize,
  pub pos: usize,
  pub name: String,
  pub interpolate_name: String,
  pub interpolate_name_der: String,
  pub elementary_name: String,
  pub elementary_target: String,
  pub der1: usize,
  pub der2: usize,
  pub children: Vec<usize>,
  pub parent: Option<usize>,
}

impl Node {
  pub(crate) fn new(ty: Ty, pos: usize) -> Self {
    Node {
      ty,
      sizes: vec![],
      data: vec![0.0],
      test: UNDET,
      name_test1: String::new(),
      name_test2: String::new(),
      interp_test1: String::new(),
      interp_test2: String::new(),
      nbc1: 0,
      pos,
      name: String::new(),
      interpolate_name: String::new(),
      interpolate_name_der: String::new(),
      elementary_name: String::new(),
      elementary_target: String::new(),
      der1: 0,
      der2: 0,
      children: vec![],
      parent: None,
    }
  }

  pub fn nb_test_functions(&self) -> usize {
    match self.test {
      UNDET => 0,
      t if t >= 2 => t - 1,
      t => t,
    }
  }

  pub fn tensor_order(&self) -> usize {
    self.sizes.len().saturating_sub(self.nb_test_functions())
  }

  fn tensor_test_size(&self) -> usize {
    let st = self.nb_test_functions();
    (if st >= 1 { self.sizes[0] } else { 1 }) * (if st == 2 { self.sizes[1] } else { 1 })
  }

  pub fn tensor_proper_size(&self) -> usize {
    self.data.len() / self.tensor_test_size()
  }

  pub fn proper_size(&self, i: usize) -> usize {
    self.sizes[self.nb_test_functions() + i]
  }

  pub fn is_constant(&self) -> bool {
    self.ty == Ty::Constant || (self.ty == Ty::Zero && self.test == 0)
  }

  fn tensor_is_zero(&self) -> bool {
    self.ty == Ty::Zero || (self.ty == Ty::Constant && self.data.iter().all(|&v| v == 0.0))
  }

  fn adjust_sizes(&mut self, mi: &[usize]) {
    self.sizes = mi.to_vec();
    self.data.resize(mi.iter().product(), 0.0);
  }

  fn init_scalar(&mut self, v: f64) {
    self.sizes.clear();
    self.data = vec![v];
    self.test = 0;
  }

  fn init_vector(&mut self, d: usize) {
    self.sizes = vec![d];
    self.data = vec![0.0; d];
    self.test = 0;
  }

  fn init_matrix(&mut self, n: usize, m: usize) {
    self.sizes = vec![n, m];
    self.data = vec![0.0; n * m];
    self.test = 0;
  }

  fn init_identity(&mut self, n: usize) {
    self.init_matrix(n, n);
    for i in 0..n {
      self.data[i * (n + 1)] = 1.0;
    }
  }

  fn clear_data(&mut self) {
    self.data.iter_mut().for_each(|v| *v = 0.0);
  }

  /// value at a multi-index (column-major)
  fn at(&self, idx: &[usize]) -> f64 {
    self.data[flat(&self.sizes, idx)]
  }
}

fn flat(sizes: &[usize], idx: &[usize]) -> usize {
  let (mut k, mut s) = (0, 1);
  for (i, &j) in idx.iter().enumerate() {
    k += j * s;
    s *= sizes[i];
  }
  k
}

/// Tensor with its indices permuted: result index k is input index perm[k].
fn permute(sizes: &[usize], data: &[f64], perm: &[usize]) -> Vec<f64> {
  let out: Vec<usize> = perm.iter().map(|&p| sizes[p]).collect();
  let mut r = vec![0.0; data.len()];
  let mut idx = vec![0; out.len()];
  let mut src = vec![0; out.len()];
  for (k, v) in r.iter_mut().enumerate() {
    let mut q = k;
    for (j, m) in out.iter().enumerate() {
      idx[j] = q % m;
      q /= m;
    }
    for (j, &p) in perm.iter().enumerate() {
      src[p] = idx[j];
    }
    *v = data[flat(sizes, &src)];
  }
  r
}

fn nz(s: &str) -> &str {
  if s.is_empty() { "-" } else { s }
}

/// GetFEM's multi_index printing: (2, 3)
fn mi_str(mi: &[usize]) -> String {
  let v: Vec<String> = mi.iter().map(|x| x.to_string()).collect();
  format!("({})", v.join(", "))
}

#[derive(Clone, Debug, Default)]
pub struct Tree {
  pub n: Vec<Node>,
  pub root: Option<usize>,
  pub expr: String,
}

impl Tree {
  pub fn from_expr(src: &str, e: &Expr) -> Self {
    let mut t = Tree {
      n: vec![],
      root: None,
      expr: src.to_string(),
    };
    let r = t.build(e);
    t.root = Some(r);
    t
  }

  /// Equal subtrees a and b (GetFEM sub_tree_are_equal, version 0): the
  /// same nodes but their positions, a zero as a constant
  pub(crate) fn same(&self, a: usize, b: usize) -> bool {
    let (x, y) = (&self.n[a], &self.n[b]);
    let ty = |t| if t == Ty::Zero { Ty::Constant } else { t };
    ty(x.ty) == ty(y.ty)
      && (&x.sizes, &x.data, x.test, x.nbc1, x.der1, x.der2)
        == (&y.sizes, &y.data, y.test, y.nbc1, y.der1, y.der2)
      && [
        &x.name,
        &x.name_test1,
        &x.name_test2,
        &x.interp_test1,
        &x.interp_test2,
      ] == [
        &y.name,
        &y.name_test1,
        &y.name_test2,
        &y.interp_test1,
        &y.interp_test2,
      ]
      && [
        &x.interpolate_name,
        &x.interpolate_name_der,
        &x.elementary_name,
        &x.elementary_target,
      ] == [
        &y.interpolate_name,
        &y.interpolate_name_der,
        &y.elementary_name,
        &y.elementary_target,
      ]
      && x.children.len() == y.children.len()
      && x
        .children
        .iter()
        .zip(&y.children)
        .all(|(&p, &q)| self.same(p, q))
  }

  /// Indented dump of the analysed tree: kind, name, shape, test functions.
  pub fn debug_string(&self) -> String {
    let mut out = String::new();
    if let Some(r) = self.root {
      self.dump(r, 0, &mut out);
    }
    out
  }

  fn dump(&self, i: usize, depth: usize, out: &mut String) {
    let n = &self.n[i];
    let kind = match n.ty {
      Ty::Op(t) => format!(
        "op {}",
        if t == Tok::UnaryMinus {
          "neg"
        } else {
          t.symbol()
        }
      ),
      Ty::Field { src, der, test } => format!("{src:?} {der:?}{}", if test { " test" } else { "" }),
      t => format!("{t:?}"),
    };
    let name = if n.name.is_empty() {
      String::new()
    } else {
      format!(" {}", n.name)
    };
    let tests = match n.test {
      0 | UNDET => String::new(),
      _ => format!(
        " order {} ({} {})",
        n.test,
        nz(&n.name_test1),
        nz(&n.name_test2)
      ),
    };
    let val = if n.ty == Ty::Constant {
      format!(" = {:?}", n.data)
    } else {
      String::new()
    };
    out.push_str(&format!(
      "{}{kind}{name} {}{tests}{val}\n",
      "  ".repeat(depth),
      mi_str(&n.sizes)
    ));
    for &c in &n.children {
      self.dump(c, depth + 1, out);
    }
  }

  pub(crate) fn push(&mut self, node: Node) -> usize {
    self.n.push(node);
    self.n.len() - 1
  }

  /// the number of nodes of subtree i
  pub(crate) fn count(&self, i: usize) -> usize {
    1 + self.n[i]
      .children
      .iter()
      .map(|&c| self.count(c))
      .sum::<usize>()
  }

  /// a copy of subtree i, without parent; the nodes allocated first and
  /// the children not cloned (GetFEM ga_tree::copy_node with preallocation)
  pub(crate) fn copy_subtree(&mut self, i: usize) -> usize {
    self.n.reserve(self.count(i));
    self.copy_rec(i)
  }

  fn copy_rec(&mut self, i: usize) -> usize {
    let ch = std::mem::take(&mut self.n[i].children);
    let mut n = self.n[i].clone();
    n.parent = None;
    let k = self.push(n);
    let mut kids = Vec::with_capacity(ch.len());
    for &c in &ch {
      let cc = self.copy_rec(c);
      self.n[cc].parent = Some(k);
      kids.push(cc);
    }
    self.n[i].children = ch;
    self.n[k].children = kids;
    k
  }

  /// a copy of the subtree i of another tree
  fn graft(&mut self, o: &Tree, i: usize) -> usize {
    let mut n = o.n[i].clone();
    n.parent = None;
    n.children.clear();
    let k = self.push(n);
    for &c in &o.n[i].children {
      let cc = self.graft(o, c);
      self.adopt(k, cc);
    }
    k
  }

  /// self + o, to be analysed again (GetFEM add_tree: the terms of the same
  /// test functions in one tree)
  pub(crate) fn join(mut self, o: &Tree) -> Tree {
    let (Some(a), Some(b)) = (self.root, o.root) else {
      return self;
    };
    let b = self.graft(o, b);
    let p = self.push(Node::new(Ty::Op(Tok::Plus), self.n[a].pos));
    self.adopt(p, a);
    self.adopt(p, b);
    self.root = Some(p);
    self.expr = format!("{} + {}", self.expr, o.expr);
    self
  }

  pub(crate) fn adopt(&mut self, p: usize, c: usize) {
    self.n[p].children.push(c);
    self.n[c].parent = Some(p);
  }

  pub(crate) fn build(&mut self, e: &Expr) -> usize {
    let mut node = Node::new(Ty::Name, e.pos);
    let kids: Vec<&Expr> = match &e.kind {
      ExprKind::Number(v) => {
        node.ty = Ty::Constant;
        node.data = vec![*v];
        vec![]
      }
      ExprKind::AllIndices => {
        node.ty = Ty::AllIndices;
        vec![]
      }
      ExprKind::Name(s) => {
        node.name = s.clone();
        vec![]
      }
      ExprKind::Unary(t, a) => {
        node.ty = Ty::Op(*t);
        vec![a]
      }
      ExprKind::Binary(t, a, b) => {
        node.ty = Ty::Op(*t);
        vec![a, b]
      }
      ExprKind::Call(f, args) => {
        node.ty = Ty::Params;
        std::iter::once(&**f).chain(args).collect()
      }
      ExprKind::Tensor { shape, comps } => {
        node.ty = Ty::CMatrix;
        node.adjust_sizes(shape);
        node.nbc1 = shape.len();
        comps.iter().collect()
      }
      ExprKind::Interpolate { name, trans } => {
        node.ty = Ty::Interpolate;
        node.name = name.clone();
        node.interpolate_name = trans.clone();
        vec![]
      }
      ExprKind::InterpolateFilter { trans, expr, id } => {
        node.ty = Ty::InterpolateFilter;
        node.interpolate_name = trans.clone();
        // no third argument: every point with a target (GetFEM size_type(-1))
        node.nbc1 = usize::MAX;
        std::iter::once(&**expr).chain(id.as_deref()).collect()
      }
      ExprKind::InterpolateDerivative {
        trans_der,
        name,
        trans,
      } => {
        node.ty = Ty::InterpolateDerivative;
        node.name = name.clone();
        node.interpolate_name = trans.clone();
        node.interpolate_name_der = trans_der.clone();
        vec![]
      }
      ExprKind::Elementary {
        name,
        trans,
        target,
      } => {
        node.ty = Ty::Elementary;
        node.name = name.clone();
        node.elementary_name = trans.clone();
        node.elementary_target = target.clone();
        vec![]
      }
      ExprKind::SecondaryDomain { name } => {
        node.ty = Ty::SecondaryDomain;
        node.name = name.clone();
        vec![]
      }
      ExprKind::XfemPlus { name } => {
        node.ty = Ty::XfemPlus;
        node.name = name.clone();
        vec![]
      }
      ExprKind::XfemMinus { name } => {
        node.ty = Ty::XfemMinus;
        node.name = name.clone();
        vec![]
      }
    };
    let i = self.push(node);
    for k in kids {
      let c = self.build(k);
      self.adopt(i, c);
    }
    i
  }

  fn replace_child(&mut self, p: usize, old: usize, new: usize) {
    for c in self.n[p].children.iter_mut() {
      if *c == old {
        *c = new;
      }
    }
  }

  /// Node i replaced by its child k
  fn replace_by_child(&mut self, i: usize, k: usize) -> usize {
    let c = self.n[i].children[k];
    let p = self.n[i].parent;
    self.n[c].parent = p;
    match p {
      Some(p) => self.replace_child(p, i, c),
      None => self.root = Some(c),
    }
    self.n[i].children.clear();
    c
  }

  /// remove child c from its parent
  fn clear_node(&mut self, c: usize) {
    match self.n[c].parent {
      Some(p) => self.n[p].children.retain(|&x| x != c),
      None => self.root = None,
    }
  }

  fn copy_test(&mut self, dst: usize, src: usize) {
    let s = self.n[src].clone();
    let d = &mut self.n[dst];
    d.test = s.test;
    d.name_test1 = s.name_test1;
    d.name_test2 = s.name_test2;
    d.interp_test1 = s.interp_test1;
    d.interp_test2 = s.interp_test2;
  }

  fn copy_t(&mut self, dst: usize, src: usize) {
    self.n[dst].sizes = self.n[src].sizes.clone();
    self.n[dst].data = self.n[src].data.clone();
  }

  fn set_zero(&mut self, i: usize) {
    self.n[i].clear_data();
    self.n[i].ty = Ty::Zero;
    self.n[i].children.clear();
  }

  /// ga_tree_node::mult_test
  fn mult_test(&mut self, i: usize, c0: usize, c1: usize) -> Result<()> {
    let (n0, n1) = (self.n[c0].clone(), self.n[c1].clone());
    let (t0, t1) = (n0.test, n1.test);
    if t0 != 0 && t1 != 0 && (t0 == t1 || t0 >= 3 || t1 >= 3) {
      return Err(self.err(i, "Incompatibility of test functions in product."));
    }
    let d = &mut self.n[i];
    d.test = t0 + t1;
    let st = d.nb_test_functions();
    let mut mi = vec![0; st];
    for (t, n) in [(t0, &n0), (t1, &n1)] {
      match t {
        1 => mi[0] = n.sizes[0],
        2 => mi[st - 1] = n.sizes[0],
        3 => {
          mi[0] = n.sizes[0];
          mi[1] = n.sizes[1];
        }
        _ => {}
      }
    }
    let a = if !n0.name_test1.is_empty() { &n0 } else { &n1 };
    d.name_test1 = a.name_test1.clone();
    d.interp_test1 = a.interp_test1.clone();
    let b = if !n0.name_test2.is_empty() { &n0 } else { &n1 };
    d.name_test2 = b.name_test2.clone();
    d.interp_test2 = b.interp_test2.clone();
    d.adjust_sizes(&mi);
    Ok(())
  }

  pub(crate) fn err(&self, i: usize, msg: impl Into<String>) -> Error {
    Error::Semantic {
      expr: self.expr.clone(),
      pos: self.n[i].pos,
      msg: msg.into(),
    }
  }

  pub(crate) fn unsupported(&self, i: usize, msg: impl Into<String>) -> Error {
    Error::Unsupported {
      expr: self.expr.clone(),
      pos: self.n[i].pos,
      msg: msg.into(),
    }
  }
}

/// Analysis context (GetFEM ga_semantic_analysis arguments).
#[derive(Clone, Debug, Default)]
pub struct Ctx {
  pub meshdim: usize,
  pub ref_elt_dim: usize,
  pub ignore_x: bool,
  /// 0 plain, 1 collect test functions, 2 select one pair of test functions
  pub option: u8,
  pub selected_test1: (String, String),
  pub selected_test2: (String, String),
}

/// Test functions (variable, transformation) met with option 1.
#[derive(Clone, Debug, Default)]
pub struct Tests {
  pub test1: BTreeSet<(String, String)>,
  pub test2: BTreeSet<(String, String)>,
}

fn valid_operand(t: &Tree, c: usize) -> Result<()> {
  match t.n[c].ty {
    Ty::PredefFunc | Ty::SpecFunc | Ty::Name | Ty::Operator | Ty::AllIndices => {
      Err(t.err(c, "Invalid term"))
    }
    _ => Ok(()),
  }
}

/// strips the prefix in place (parser::strip_op, strip_test)
fn strip_prefix(name: &mut String, f: fn(&str) -> (u8, &str)) -> usize {
  let (k, r) = f(name);
  *name = r.to_string();
  k as usize
}

/// A scalar function: built-in or user defined.
struct Func {
  nargs: usize,
  builtin: Option<names::Eval>,
  user: Option<UserFn>,
}

fn func(name: &str) -> Option<Func> {
  if let Some(f) = names::function(name) {
    return Some(Func {
      nargs: f.eval.nargs(),
      builtin: Some(f.eval),
      user: None,
    });
  }
  crate::defs::function(name).map(|u| Func {
    nargs: u.nargs,
    builtin: None,
    user: Some(u),
  })
}

/// Value of a user function body at (t, u): substitute and fold.
fn eval_user(defs: &Defs, f: &UserFn, t: f64, u: f64, depth: usize) -> Option<f64> {
  fn subst(e: &mut Expr, t: f64, u: f64) {
    match &mut e.kind {
      ExprKind::Name(n) if n == "t" || n == "u" => {
        e.kind = ExprKind::Number(if n == "t" { t } else { u })
      }
      ExprKind::Unary(_, a) => subst(a, t, u),
      ExprKind::Binary(_, a, b) => {
        subst(a, t, u);
        subst(b, t, u)
      }
      ExprKind::Call(g, args) => {
        subst(g, t, u);
        args.iter_mut().for_each(|a| subst(a, t, u))
      }
      ExprKind::Tensor { comps, .. } => comps.iter_mut().for_each(|a| subst(a, t, u)),
      _ => {}
    }
  }
  if depth > 32 {
    return None;
  }
  if let Some(nf) = &f.native {
    return Some((nf.0)(t, u));
  }
  let body = f.body.as_ref()?;
  let mut e = crate::parser::parse(body, &Default::default()).ok()??;
  subst(&mut e, t, u);
  let ctx = Ctx {
    meshdim: 1,
    ref_elt_dim: 1,
    ..Ctx::default()
  };
  let mut a = An {
    t: Tree::from_expr(body, &e),
    defs,
    ctx: &ctx,
    tests: Tests::default(),
    depth: depth + 1,
  };
  let r = a.t.root?;
  let r = a.node(r).ok()?;
  let n = &a.t.n[r];
  (n.ty == Ty::Constant && n.data.len() == 1).then(|| n.data[0])
}

struct An<'a> {
  t: Tree,
  defs: &'a Defs,
  ctx: &'a Ctx,
  tests: Tests,
  /// nesting of user function evaluations
  depth: usize,
}

/// Parse (with the macros of defs) and analyse an expression.
pub fn analyse_str(src: &str, defs: &Defs, ctx: &Ctx) -> Result<(Tree, Tests)> {
  match crate::parser::parse(src, &defs.macros)? {
    Some(e) => analyse(Tree::from_expr(src, &e), defs, ctx),
    None => Ok((
      Tree {
        expr: src.to_string(),
        ..Tree::default()
      },
      Tests::default(),
    )),
  }
}

/// GetFEM ga_semantic_analysis.
pub fn analyse(mut tree: Tree, defs: &Defs, ctx: &Ctx) -> Result<(Tree, Tests)> {
  let Some(r) = tree.root else {
    return Ok((tree, Tests::default()));
  };
  tree.n[r].parent = None;
  let mut a = An {
    t: tree,
    defs,
    ctx,
    tests: Tests::default(),
    depth: 0,
  };
  a.node(r)?;
  let mut t = a.t;
  if let (Some(r), 2) = (t.root, ctx.option) {
    let n = &t.n[r];
    if ((n.test & 1) != 0
      && (n.name_test1 != ctx.selected_test1.0 || n.interp_test1 != ctx.selected_test1.1))
      || ((n.test & 2) != 0
        && (n.name_test2 != ctx.selected_test2.0 || n.interp_test2 != ctx.selected_test2.1))
    {
      t.root = None;
    }
  }
  if let Some(r) = t.root {
    valid_operand(&t, r)?;
  }
  Ok((t, a.tests))
}

impl An<'_> {
  /// analyse node i; returns the node that took its place
  fn node(&mut self, mut i: usize) -> Result<usize> {
    let meshdim = self.ctx.meshdim;
    let (mut all_cte, mut all_sc) = (true, true);
    let mut k = 0;
    while k < self.t.n[i].children.len() {
      let c = self.t.n[i].children[k];
      let c = self.node(c)?;
      let cn = &self.t.n[c];
      all_cte = all_cte && cn.is_constant();
      all_sc = all_sc && cn.tensor_proper_size() == 1;
      if cn.test == UNDET {
        return Err(Error::Internal(format!("undetermined child {k}")));
      }
      if self.t.n[i].ty != Ty::Params {
        valid_operand(&self.t, c)?;
      }
      k += 1;
    }
    let ch = self.t.n[i].children.clone();
    let c0 = ch.first().copied();
    let c1 = ch.get(1).copied();
    let size0 = c0.map(|c| self.t.n[c].sizes.clone()).unwrap_or_default();
    let size1 = c1.map(|c| self.t.n[c].sizes.clone()).unwrap_or_default();
    let dim0 = c0.map_or(0, |c| self.t.n[c].tensor_order());
    let dim1 = c1.map_or(0, |c| self.t.n[c].tensor_order());

    match self.t.n[i].ty {
      Ty::PredefFunc
      | Ty::Operator
      | Ty::SpecFunc
      | Ty::Constant
      | Ty::X
      | Ty::EltSize
      | Ty::EltK
      | Ty::EltB
      | Ty::EltCurvature
      | Ty::Normal
      | Ty::Reshape
      | Ty::CrossProduct
      | Ty::IndMoveLast
      | Ty::SwapInd
      | Ty::Contract
      | Ty::InterpX
      | Ty::InterpEltK
      | Ty::InterpEltB
      | Ty::InterpNormal
      | Ty::AllIndices => self.t.n[i].test = 0,
      Ty::Field { test: true, .. } => {
        // re-analysis: register the test function again (GetFEM does)
        let n = self.t.n[i].clone();
        if n.test == 1 {
          self.test_insert(1, &n.name_test1, &n.interp_test1);
        } else if n.test == 2 {
          self.test_insert(2, &n.name_test2, &n.interp_test2);
        }
      }
      Ty::Zero | Ty::Field { .. } => {}
      Ty::InterpolateDerivative => {
        // made by derivation: the derivative of transformation
        // interpolate_name_der w.r.t. variable name (read through
        // interpolate_name), times its test functions (GetFEM)
        let n = self.t.n[i].clone();
        if !self
          .defs
          .interpolate_transformations
          .contains_key(&n.interpolate_name_der)
        {
          return Err(self.t.err(i, "Unknown interpolate transformation"));
        }
        let q = self.defs.var(&n.name).map_or(0, |v| v.qdim());
        if q == 0 {
          return Err(self.t.err(i, "Invalid null size of variable"));
        }
        let nd = &mut self.t.n[i];
        // written by hand: GetFEM's test type 0 (second test function)
        if nd.test == UNDET {
          nd.test = 0;
        }
        let len = nd.sizes.iter().product();
        nd.data.resize(len, 0.0);
        if n.test == 1 {
          nd.name_test1 = n.name.clone();
          nd.interp_test1 = n.interpolate_name.clone();
          nd.name_test2.clear();
          nd.interp_test2.clear();
          self.test_insert(1, &n.name, &n.interpolate_name);
        } else {
          nd.name_test2 = n.name.clone();
          nd.interp_test2 = n.interpolate_name.clone();
          nd.name_test1.clear();
          nd.interp_test1.clear();
          self.test_insert(2, &n.name, &n.interpolate_name);
        }
      }
      Ty::SecondaryDomain => return Err(self.t.unsupported(i, "Secondary_domain")),
      Ty::Interpolate | Ty::Elementary | Ty::XfemPlus | Ty::XfemMinus => self.interpolate(i)?,
      Ty::InterpolateFilter => {
        let (c0, c1) = (c0.unwrap(), c1);
        if let Some(c1) = c1 {
          let n1 = &self.t.n[c1];
          let n = if n1.ty == Ty::Constant {
            n1.data[0].round() as i64
          } else {
            -1
          };
          if !(0..=100).contains(&n) || n1.tensor_order() > 0 {
            return Err(self.t.err(
              i,
              "The third argument of Interpolate_filter should be a (small) non-negative integer.",
            ));
          }
          self.t.n[i].nbc1 = n as usize;
          self.t.clear_node(c1);
        }
        if !self
          .defs
          .interpolate_transformations
          .contains_key(&self.t.n[i].interpolate_name)
        {
          return Err(self.t.err(i, "Unknown interpolate transformation"));
        }
        self.t.copy_t(i, c0);
        self.t.copy_test(i, c0);
      }
      Ty::Op(op) => {
        i = self.op(i, op, c0.unwrap(), c1, &size0, &size1, dim0, dim1, all_cte)?;
        if matches!(self.t.n[i].ty, Ty::Op(Tok::Mult | Tok::Div)) {
          i = self.merge_powers(i)?;
        }
      }
      Ty::CMatrix => self.cmatrix(i, all_cte, all_sc)?,
      Ty::Name => self.name(i, meshdim)?,
      Ty::Params => i = self.params(i, &size1, dim1)?,
    }
    Ok(i)
  }

  fn test_insert(&mut self, which: usize, name: &str, trans: &str) {
    if self.ctx.option == 1 {
      let s = if which == 1 {
        &mut self.tests.test1
      } else {
        &mut self.tests.test2
      };
      s.insert((name.to_string(), trans.to_string()));
    }
  }

  /// Interpolate(u, trans), Elementary_transformation(u, trans, target),
  /// Xfem_plus(u), Xfem_minus(u)
  fn interpolate(&mut self, i: usize) -> Result<()> {
    let meshdim = self.ctx.meshdim;
    let red = self.ctx.ref_elt_dim;
    let ty = self.t.n[i].ty;
    let elem = ty == Ty::Elementary;
    if ty == Ty::Interpolate {
      let nm = self.t.n[i].name.clone();
      let n = &mut self.t.n[i];
      match nm.as_str() {
        "X" => {
          n.ty = Ty::InterpX;
          n.init_vector(meshdim);
          return Ok(());
        }
        "Normal" => {
          n.ty = Ty::InterpNormal;
          n.init_vector(meshdim);
          return Ok(());
        }
        "element_K" => {
          n.ty = Ty::InterpEltK;
          if red == 1 {
            n.init_vector(meshdim)
          } else {
            n.init_matrix(meshdim, red)
          }
          return Ok(());
        }
        "element_B" => {
          n.ty = Ty::InterpEltB;
          // N x P as K (B = K (K^T K)^-1), a vector when P = 1
          if red == 1 {
            n.init_vector(meshdim)
          } else {
            n.init_matrix(meshdim, red)
          }
          return Ok(());
        }
        _ => {}
      }
    }
    let op_name = match ty {
      Ty::Elementary => "Elementary_transformation",
      Ty::XfemPlus => "Xfem_plus",
      Ty::XfemMinus => "Xfem_minus",
      _ => "Interpolation",
    };
    let mut name = self.t.n[i].name.clone();
    let prefix = strip_prefix(&mut name, strip_op);
    let test = strip_prefix(&mut name, strip_test);
    self.t.n[i].name = name.clone();
    if elem {
      let mut tg = self.t.n[i].elementary_target.clone();
      strip_prefix(&mut tg, strip_op);
      strip_prefix(&mut tg, strip_test);
      self.t.n[i].elementary_target = tg.clone();
      name = tg;
    }
    let Some(v) = self.defs.var(&name) else {
      return Err(self.t.err(
        i,
        format!("Unknown variable or group of variables \"{name}\""),
      ));
    };
    let Support::Fem { mesh_dim: n } = v.support else {
      return Err(self.t.err(
        i,
        format!("{op_name} can only apply to finite element variables/data"),
      ));
    };
    let q = v.qdim();
    if q == 0 {
      return Err(self.t.err(i, "Invalid null size of variable"));
    }
    let mut mii = v.qdims.clone();
    if mii.len() > 6 {
      return Err(
        self
          .t
          .err(i, "Tensor with too many dimensions. Limited to 6"),
      );
    }
    let (nm, trans) = (
      self.t.n[i].name.clone(),
      self.t.n[i].interpolate_name.clone(),
    );
    if test == 1 {
      self.t.n[i].name_test1 = nm.clone();
      self.t.n[i].interp_test1 = trans.clone();
      self.test_insert(1, &nm, &trans);
    } else if test == 2 {
      self.t.n[i].name_test2 = nm.clone();
      self.t.n[i].interp_test2 = trans.clone();
      self.test_insert(2, &nm, &trans);
    }
    let is_test = test != 0;
    let lead = |mii: &mut Vec<usize>| {
      if q == 1 && mii.len() <= 1 {
        *mii = vec![2];
      } else {
        mii.insert(0, 2);
      }
    };
    let der = match prefix {
      0 => {
        if is_test {
          lead(&mut mii);
        }
        Der::Val
      }
      1 => {
        if is_test {
          lead(&mut mii);
          if n > 1 {
            mii.push(n);
          }
        } else if n > 1 {
          if q == 1 && mii.len() == 1 {
            mii[0] = n;
          } else {
            mii.push(n);
          }
        }
        Der::Grad
      }
      2 => {
        if is_test {
          lead(&mut mii);
          if n > 1 {
            mii.push(n);
            mii.push(n);
          }
        } else if n > 1 {
          if q == 1 && mii.len() == 1 {
            mii[0] = n;
          } else {
            mii.push(n);
          }
          mii.push(n);
        }
        Der::Hess
      }
      _ => {
        if q != n {
          return Err(self.t.err(
            i,
            format!("Divergence operator requires fem qdim ({q}) to be equal to dim ({n})"),
          ));
        }
        mii = vec![if is_test { 2 } else { 1 }];
        Der::Div
      }
    };
    let src = match ty {
      Ty::Elementary => Src::Elem,
      Ty::XfemPlus => Src::XfemPlus,
      Ty::XfemMinus => Src::XfemMinus,
      _ => Src::Interp,
    };
    self.t.n[i].ty = Ty::Field {
      src,
      der,
      test: is_test,
    };
    self.t.n[i].adjust_sizes(&mii);
    self.t.n[i].test = test;
    if ty == Ty::Interpolate {
      if !self
        .defs
        .interpolate_transformations
        .contains_key(&self.t.n[i].interpolate_name)
      {
        return Err(self.t.err(i, "Unknown interpolate transformation"));
      }
    } else if elem {
      if !self
        .defs
        .elementary_transformations
        .contains(&self.t.n[i].elementary_name)
      {
        return Err(self.t.err(i, "Unknown elementary transformation"));
      }
      let tg = self.t.n[i].elementary_target.clone();
      if self.defs.var(&tg).is_none() {
        return Err(self.t.err(i, format!("Unknown data or variable {tg}")));
      }
    }
    Ok(())
  }

  #[allow(clippy::too_many_arguments)]
  fn op(
    &mut self,
    i: usize,
    op: Tok,
    c0: usize,
    c1: Option<usize>,
    size0: &[usize],
    size1: &[usize],
    dim0: usize,
    dim1: usize,
    all_cte: bool,
  ) -> Result<usize> {
    let t = &mut self.t;
    match op {
      Tok::Plus | Tok::Minus => {
        let c1 = c1.unwrap();
        let (n0, n1) = (t.n[c0].clone(), t.n[c1].clone());
        let c_size = size0.len().min(size1.len());
        let mut compatible = true;
        let f_ind = if n0.test != 0 && n1.test == n0.test {
          if n0.test == 3 { 2 } else { 1 }
        } else {
          0
        };
        for k in f_ind..c_size {
          compatible &= size0[k] == size1[k];
        }
        compatible &= size0[c_size..].iter().all(|&s| s == 1);
        compatible &= size1[c_size..].iter().all(|&s| s == 1);
        if !compatible {
          return Err(t.err(
            i,
            format!(
              "Addition or substraction of expressions of different sizes: {} != {}",
              mi_str(size0),
              mi_str(size1)
            ),
          ));
        }
        let option = self.ctx.option;
        if (n0.test != 0 || n1.test != 0)
          && (option == 0 || option == 2)
          && (n0.name_test1 != n1.name_test1
            || n0.name_test2 != n1.name_test2
            || n0.interp_test1 != n1.interp_test1
            || n0.interp_test2 != n1.interp_test2)
        {
          compatible = false;
        }
        if n0.test != n1.test || (!compatible && option != 2) {
          return Err(t.err(i, "Addition or substraction of incompatible test functions"));
        }
        if all_cte {
          let n = &mut t.n[i];
          n.ty = Ty::Constant;
          n.test = 0;
          n.sizes = n0.sizes.clone();
          n.data = n0.data.clone();
          for (a, b) in n.data.iter_mut().zip(&n1.data) {
            if op == Tok::Minus { *a -= b } else { *a += b }
          }
          n.children.clear();
          return Ok(i);
        }
        t.copy_t(i, c0);
        t.copy_test(i, c0);
        if n0.tensor_is_zero() {
          if n1.tensor_is_zero() {
            // both zero: zero, not -0 (GetFEM leaves -0, its compiler
            // takes zeros with test functions, ours does not)
            t.set_zero(i);
          } else if op == Tok::Minus {
            t.n[i].ty = Ty::Op(Tok::UnaryMinus);
            t.clear_node(c0);
          } else {
            return Ok(t.replace_by_child(i, 1));
          }
        } else if n1.tensor_is_zero() {
          return Ok(t.replace_by_child(i, 0));
        } else if option == 2 && !compatible {
          let (s1, s2) = (&self.ctx.selected_test1, &self.ctx.selected_test2);
          let tt = t.n[i].test;
          let (mut ok0, mut ok1) = (true, true);
          if tt & 1 != 0 {
            ok0 &= n0.name_test1 == s1.0 && n0.interp_test1 == s1.1;
            ok1 &= n1.name_test1 == s1.0 && n1.interp_test1 == s1.1;
          }
          if tt & 2 != 0 {
            ok0 &= n0.name_test2 == s2.0 && n0.interp_test2 == s2.1;
            // GetFEM compares interpolate_name_test1 here (a GetFEM slip)
            ok1 &= n1.name_test2 == s2.0 && n1.interp_test2 == s2.1;
          }
          if ok0 {
            return Ok(t.replace_by_child(i, 0));
          } else if ok1 {
            if op == Tok::Minus {
              t.n[i].ty = Ty::Op(Tok::UnaryMinus);
              t.copy_t(i, c1);
              t.copy_test(i, c1);
              t.clear_node(c0);
            } else {
              return Ok(t.replace_by_child(i, 1));
            }
          }
        }
      }
      Tok::DotMult | Tok::DotDiv => {
        let c1 = c1.unwrap();
        let (n0, n1) = (t.n[c0].clone(), t.n[c1].clone());
        let mut compatible = n0.tensor_proper_size() == n1.tensor_proper_size();
        if n0.tensor_proper_size() != 1 {
          compatible &= n0.tensor_order() == n1.tensor_order();
          for k in 0..n0.tensor_order().min(n1.tensor_order()) {
            compatible &= n0.proper_size(k) == n1.proper_size(k);
          }
        }
        if !compatible {
          return Err(t.err(i, "Arguments of different sizes for .* or ./"));
        }
        if op == Tok::DotDiv && n1.test != 0 {
          return Err(t.err(i, "Division by test functions is not allowed"));
        }
        t.mult_test(i, c0, c1)?;
        let mut mi = t.n[i].sizes.clone();
        mi.extend((0..n0.tensor_order()).map(|k| n0.proper_size(k)));
        t.n[i].adjust_sizes(&mi);
        if all_cte {
          t.n[i].ty = Ty::Constant;
          t.n[i].test = 0;
          for k in 0..n0.data.len() {
            if op == Tok::DotMult {
              t.n[i].data[k] = n0.data[k] * n1.data[k];
            } else {
              if n1.data[k] == 0.0 {
                return Err(t.err(i, "Division by zero."));
              }
              t.n[i].data[k] = n0.data[k] / n1.data[k];
            }
          }
          t.n[i].children.clear();
        } else {
          if n0.tensor_is_zero() || n1.tensor_is_zero() {
            t.set_zero(i);
          }
          if n1.tensor_is_zero() && op == Tok::DotDiv {
            return Err(t.err(i, "Division by zero."));
          }
          // 1 .* x, x .* 1, x ./ 1 -> x of the same shape (not GetFEM's)
          let one = |n: &Node| n.ty == Ty::Constant && n.data == [1.0];
          for (k, (a, b)) in [(1, (&n0, &n1)), (0, (&n1, &n0))] {
            if one(a) && (k == 1 || op == Tok::DotMult) && b.sizes == t.n[i].sizes {
              return Ok(t.replace_by_child(i, k));
            }
          }
        }
      }
      Tok::UnaryMinus => {
        t.copy_t(i, c0);
        t.copy_test(i, c0);
        if all_cte {
          t.n[i].ty = Ty::Constant;
          t.n[i].test = 0;
          t.n[i].data.iter_mut().for_each(|v| *v = -*v);
          t.n[i].children.clear();
        } else if t.n[c0].ty == Ty::Zero {
          return Ok(t.replace_by_child(i, 0));
        }
      }
      Tok::Quote => {
        let n0 = t.n[c0].clone();
        let mut mi = size0.to_vec();
        if n0.tensor_proper_size() == 1 {
          return Ok(t.replace_by_child(i, 0));
        } else if dim0 == 1 {
          let nn = *mi.last().unwrap();
          *mi.last_mut().unwrap() = 1;
          mi.push(nn);
        } else {
          let nt = n0.nb_test_functions();
          mi.swap(nt, nt + 1);
        }
        t.n[i].adjust_sizes(&mi);
        t.copy_test(i, c0);
        if n0.ty == Ty::Zero {
          t.set_zero(i);
        } else if all_cte {
          t.n[i].ty = Ty::Constant;
          t.n[i].test = 0;
          if dim0 == 1 {
            t.n[i].data.copy_from_slice(&n0.data);
          } else {
            let (n1, n2) = (n0.proper_size(0), n0.proper_size(1));
            let nn = n0.data.len() / (n1 * n2);
            let mut it = 0;
            for a in 0..nn {
              for j in 0..n1 {
                for k in 0..n2 {
                  t.n[i].data[it] = n0.data[j + k * n1 + a * n1 * n2];
                  it += 1;
                }
              }
            }
          }
          t.n[i].children.clear();
        }
      }
      Tok::Sym | Tok::Skew => {
        let n0 = t.n[c0].clone();
        if n0.tensor_proper_size() != 1
          && (dim0 != 2 || size0[size0.len() - 1] != size0[size0.len() - 2])
        {
          return Err(t.err(i, "Sym and Skew operators are for square matrices only."));
        }
        if n0.tensor_proper_size() == 1 {
          if op == Tok::Sym {
            return Ok(t.replace_by_child(i, 0));
          }
          t.set_zero(i);
          return Ok(i);
        }
        t.n[i].adjust_sizes(size0);
        t.copy_test(i, c0);
        if all_cte {
          t.n[i].ty = Ty::Constant;
          t.n[i].test = 0;
          let m = *size0.last().unwrap();
          for a in 0..m {
            for b in 0..m {
              let (x, y) = (n0.data[b + m * a], n0.data[a + m * b]);
              t.n[i].data[b + m * a] = 0.5 * if op == Tok::Sym { x + y } else { x - y };
            }
          }
          t.n[i].children.clear();
        } else if n0.ty == Ty::Zero {
          t.set_zero(i);
        }
      }
      Tok::Trace | Tok::Deviator => {
        let n0 = t.n[c0].clone();
        let mut mi = size0.to_vec();
        let nn = if n0.tensor_proper_size() == 1 {
          1
        } else {
          *mi.last().unwrap()
        };
        if op == Tok::Trace && n0.tensor_proper_size() == 1 {
          return Ok(t.replace_by_child(i, 0));
        }
        if (dim0 != 2 && n0.tensor_proper_size() != 1) || (dim0 == 2 && mi[mi.len() - 2] != nn) {
          let w = if op == Tok::Trace {
            "Trace"
          } else {
            "Deviator"
          };
          return Err(t.err(i, format!("{w} operator is for square matrices only.")));
        }
        if op == Tok::Deviator && n0.tensor_proper_size() == 1 {
          t.set_zero(i);
          return Ok(i);
        }
        if op == Tok::Trace && dim0 == 2 {
          mi.truncate(mi.len() - 2);
        }
        t.n[i].adjust_sizes(&mi);
        t.copy_test(i, c0);
        if all_cte {
          t.n[i].ty = Ty::Constant;
          t.n[i].test = 0;
          let tr: f64 = (0..nn).map(|a| n0.data[a * (nn + 1)]).sum();
          if op == Tok::Trace {
            t.n[i].data[0] = tr;
          } else {
            t.n[i].data.copy_from_slice(&n0.data);
            for a in 0..nn {
              t.n[i].data[a * (nn + 1)] -= tr / nn as f64;
            }
          }
          t.n[i].children.clear();
        } else if n0.ty == Ty::Zero {
          t.set_zero(i);
        }
      }
      Tok::Print => return Err(t.unsupported(i, "Print")),
      Tok::Dot => {
        let c1 = c1.unwrap();
        let (n0, n1) = (t.n[c0].clone(), t.n[c1].clone());
        let s0 = if dim0 == 0 { 1 } else { *size0.last().unwrap() };
        let s1 = if dim1 == 0 {
          1
        } else {
          size1[size1.len() - dim1]
        };
        if s0 != s1 {
          return Err(t.err(
            i,
            format!("Dot product of expressions of different sizes.({s0} != {s1}).  Sizes of operands: {} and {}", mi_str(size0), mi_str(size1)),
          ));
        }
        t.mult_test(i, c0, c1)?;
        if dim0 > 1 || dim1 > 1 {
          let mut mi = t.n[i].sizes.clone();
          mi.extend((1..dim0).map(|k| n0.proper_size(k - 1)));
          mi.extend((1..dim1).map(|k| n1.proper_size(k)));
          t.n[i].adjust_sizes(&mi);
        }
        if n0.tensor_is_zero() || n1.tensor_is_zero() {
          t.set_zero(i);
        } else if all_cte {
          // contraction of the last index of c0 with the first of c1
          t.n[i].clear_data();
          let (m0, m1) = (n0.data.len() / s0, n1.data.len() / s1);
          for a in 0..m0 {
            for b in 0..m1 {
              for c in 0..s0 {
                t.n[i].data[a + m0 * b] += n0.data[a + m0 * c] * n1.data[c + s1 * b];
              }
            }
          }
          t.n[i].ty = Ty::Constant;
          t.n[i].children.clear();
        }
      }
      Tok::Colon => {
        let c1 = c1.unwrap();
        let (n0, n1) = (t.n[c0].clone(), t.n[c1].clone());
        let s00 = match dim0 {
          0 => 1,
          1 => *size0.last().unwrap(),
          _ => size0[size0.len() - 2],
        };
        let s01 = if dim0 >= 2 { *size0.last().unwrap() } else { 1 };
        let s10 = if dim1 == 0 { 1 } else { n1.proper_size(0) };
        let s11 = if dim1 < 2 { 1 } else { n1.proper_size(1) };
        if s00 != s10 || s01 != s11 {
          return Err(t.err(
            i,
            format!(
              "Frobenius product of expressions of different sizes ({s00},{s01} != {s10},{s11}). Sizes of arguments (last two): {} and {}",
              mi_str(size0),
              mi_str(size1)
            ),
          ));
        }
        t.mult_test(i, c0, c1)?;
        if dim0 > 2 || dim1 > 2 {
          let mut mi = t.n[i].sizes.clone();
          mi.extend((0..dim0.saturating_sub(2)).map(|k| n0.proper_size(k)));
          mi.extend((2..dim1).map(|k| n1.proper_size(k)));
          t.n[i].adjust_sizes(&mi);
        }
        if n0.tensor_is_zero() || n1.tensor_is_zero() {
          t.set_zero(i);
        } else if all_cte {
          t.n[i].ty = Ty::Constant;
          t.n[i].clear_data();
          let len = t.n[i].data.len();
          let (mut j, mut k) = (0, 0);
          for a in 0..n0.data.len() {
            t.n[i].data[j] += n0.data[a] * n1.data[k];
            j += 1;
            if j == len {
              j = 0;
              k += 1;
            }
          }
          t.n[i].children.clear();
        }
      }
      Tok::TMult => {
        let c1 = c1.unwrap();
        let (n0, n1) = (t.n[c0].clone(), t.n[c1].clone());
        if all_cte {
          t.n[i].ty = Ty::Constant;
          t.n[i].test = 0;
          if n0.data.len() == 1 && n1.data.len() == 1 {
            t.n[i].init_scalar(n0.data[0] * n1.data[0]);
          } else if n0.data.len() == 1 {
            t.copy_t(i, c1);
            t.n[i].data.iter_mut().for_each(|v| *v *= n0.data[0]);
          } else if n1.data.len() == 1 {
            t.copy_t(i, c0);
            t.n[i].data.iter_mut().for_each(|v| *v *= n1.data[0]);
          } else {
            if dim0 + dim1 > 6 {
              return Err(t.err(i, "Unauthorized tensor multiplication."));
            }
            let mi: Vec<usize> = n0.sizes.iter().chain(&n1.sizes).copied().collect();
            t.n[i].adjust_sizes(&mi);
            let l0 = n0.data.len();
            for a in 0..l0 {
              for b in 0..n1.data.len() {
                t.n[i].data[a + b * l0] = n0.data[a] * n1.data[b];
              }
            }
          }
          t.n[i].children.clear();
        } else {
          t.mult_test(i, c0, c1)?;
          let mut mi = t.n[i].sizes.clone();
          if n0.tensor_proper_size() != 1 || n1.tensor_proper_size() != 1 {
            if n0.tensor_proper_size() == 1 {
              mi.extend((0..dim1).map(|k| n1.proper_size(k)));
            } else if n1.data.len() == 1 {
              mi.extend((0..dim0).map(|k| n0.proper_size(k)));
            } else {
              if dim0 + dim1 > 6 {
                return Err(t.err(i, "Unauthorized tensor multiplication."));
              }
              mi.extend((0..dim0).map(|k| n0.proper_size(k)));
              mi.extend((0..dim1).map(|k| n1.proper_size(k)));
            }
            t.n[i].adjust_sizes(&mi);
          }
          if n0.tensor_is_zero() || n1.tensor_is_zero() {
            t.set_zero(i);
          }
        }
      }
      Tok::Mult => return self.mult(i, c0, c1.unwrap(), dim0, dim1, all_cte),
      Tok::Div => {
        let c1 = c1.unwrap();
        let (n0, n1) = (t.n[c0].clone(), t.n[c1].clone());
        if n1.tensor_proper_size() > 1 {
          return Err(t.err(
            i,
            format!(
              "Only the division by a scalar is allowed. Got a size of {}",
              n1.tensor_proper_size()
            ),
          ));
        }
        if n1.test != 0 {
          return Err(t.err(i, "Division by test functions is not allowed."));
        }
        if n1.ty == Ty::Constant && n1.data[0] == 0.0 {
          return Err(t.err(c1, "Division by zero"));
        }
        t.copy_t(i, c0);
        t.copy_test(i, c0);
        if all_cte {
          t.n[i].ty = Ty::Constant;
          t.n[i].test = 0;
          t.n[i].data.iter_mut().for_each(|v| *v *= 1.0 / n1.data[0]);
          t.n[i].children.clear();
        } else if n0.tensor_is_zero() {
          t.set_zero(i);
        } else if n1.ty == Ty::Constant && n1.data.len() == 1 && n1.data[0] == 1.0 {
          return Ok(t.replace_by_child(i, 0));
        }
      }
      _ => return Err(Error::Internal("unexpected operation".into())),
    }
    Ok(i)
  }

  fn mult(
    &mut self,
    i: usize,
    c0: usize,
    c1: usize,
    dim0: usize,
    dim1: usize,
    all_cte: bool,
  ) -> Result<usize> {
    let t = &mut self.t;
    let (n0, n1) = (t.n[c0].clone(), t.n[c1].clone());
    let (p0, p1) = (n0.tensor_proper_size(), n1.tensor_proper_size());
    if all_cte {
      t.n[i].ty = Ty::Constant;
      t.n[i].test = 0;
      if p0 == 1 && p1 == 1 {
        t.n[i].init_scalar(n0.data[0] * n1.data[0]);
      } else if p0 == 1 {
        t.copy_t(i, c1);
        t.n[i].data.iter_mut().for_each(|v| *v *= n0.data[0]);
      } else if p1 == 1 {
        t.copy_t(i, c0);
        t.n[i].data.iter_mut().for_each(|v| *v *= n1.data[0]);
      } else if dim0 == 2 && dim1 == 1 {
        let (m, n) = (n0.sizes[0], n0.sizes[1]);
        if n != n1.sizes[0] {
          return Err(t.err(
            i,
            format!(
              "Incompatible sizes in matrix-vector multiplication ({n} != {}).",
              n1.sizes[0]
            ),
          ));
        }
        t.n[i].init_vector(m);
        for a in 0..m {
          for b in 0..n {
            t.n[i].data[a] += n0.at(&[a, b]) * n1.data[b];
          }
        }
      } else if dim0 == 2 && dim1 == 2 {
        let (m, n, p) = (n0.sizes[0], n0.sizes[1], n1.sizes[1]);
        if n != n1.sizes[0] {
          return Err(t.err(
            i,
            format!(
              "Incompatible sizes in matrix-matrix multiplication ({n} != {}).",
              n1.sizes[0]
            ),
          ));
        }
        t.n[i].init_matrix(m, p);
        for a in 0..m {
          for b in 0..n {
            for c in 0..p {
              t.n[i].data[a + m * c] += n0.at(&[a, b]) * n1.at(&[b, c]);
            }
          }
        }
      } else if dim0 == 4 && dim1 == 2 {
        let s = &n0.sizes;
        let (m, n, o, p) = (s[0], s[1], s[2], s[3]);
        if o != n1.sizes[0] || p != n1.sizes[1] {
          return Err(t.err(
            i,
            format!(
              "Incompatible sizes in tensor-matrix multiplication ({o},{p} != {},{}).",
              n1.sizes[0], n1.sizes[1]
            ),
          ));
        }
        t.n[i].init_matrix(m, n);
        for a in 0..m {
          for b in 0..n {
            for c in 0..o {
              for d in 0..p {
                t.n[i].data[a + m * b] += n0.at(&[a, b, c, d]) * n1.at(&[c, d]);
              }
            }
          }
        }
      } else {
        return Err(t.err(i, "Unauthorized multiplication."));
      }
      t.n[i].children.clear();
      return Ok(i);
    }
    t.mult_test(i, c0, c1)?;
    let mut mi = t.n[i].sizes.clone();
    if p0 == 1 && p1 == 1 {
    } else if p0 == 1 {
      mi.extend((0..dim1).map(|k| n1.proper_size(k)));
    } else if p1 == 1 {
      mi.extend((0..dim0).map(|k| n0.proper_size(k)));
    } else if n0.tensor_order() == 2 && n1.tensor_order() == 1 {
      let (m, n) = (n0.proper_size(0), n0.proper_size(1));
      mi.push(m);
      if n != n1.proper_size(0) {
        return Err(t.err(
          i,
          format!(
            "Incompatible sizes in matrix-vector multiplication ({n} != {}).",
            n1.proper_size(0)
          ),
        ));
      }
    } else if n0.tensor_order() == 2 && n1.tensor_order() == 2 {
      let (m, n, p) = (n0.proper_size(0), n0.proper_size(1), n1.proper_size(1));
      mi.push(m);
      mi.push(p);
      if n != n1.proper_size(0) {
        return Err(t.err(
          i,
          format!(
            "Incompatible sizes in matrix-matrix multiplication ({n} != {}).",
            n1.proper_size(0)
          ),
        ));
      }
    } else if n0.tensor_order() == 4 && n1.tensor_order() == 2 {
      let (m, n, o, p) = (
        n0.proper_size(0),
        n0.proper_size(1),
        n0.proper_size(2),
        n0.proper_size(3),
      );
      mi.push(m);
      mi.push(n);
      if o != n1.proper_size(0) || p != n1.proper_size(1) {
        return Err(t.err(
          i,
          format!(
            "Incompatible sizes in tensor-matrix multiplication ({o},{p} != {},{}).",
            n1.proper_size(0),
            n1.proper_size(1)
          ),
        ));
      }
    } else {
      return Err(t.err(i, "Unauthorized multiplication."));
    }
    t.n[i].adjust_sizes(&mi);
    let one = |n: &Node| n.ty == Ty::Constant && n.data.len() == 1 && n.data[0] == 1.0;
    if n0.tensor_is_zero() || n1.tensor_is_zero() {
      t.set_zero(i);
    } else if one(&n0) {
      return Ok(t.replace_by_child(i, 1));
    } else if one(&n1) {
      return Ok(t.replace_by_child(i, 0));
    }
    Ok(i)
  }

  fn cmatrix(&mut self, i: usize, all_cte: bool, all_sc: bool) -> Result<()> {
    let t = &mut self.t;
    if !all_sc {
      return Err(t.err(
        i,
        "Constant vector/matrix/tensor components should be scalar valued.",
      ));
    }
    t.n[i].test = 0;
    for c in t.n[i].children.clone() {
      let cn = t.n[c].clone();
      if cn.test == 0 {
        continue;
      }
      if t.n[i].test == 0 {
        t.copy_test(i, c);
      } else {
        let n = &t.n[i];
        if n.test != cn.test
          || n.name_test1 != cn.name_test1
          || n.name_test2 != cn.name_test2
          || n.interp_test1 != cn.interp_test1
          || n.interp_test2 != cn.interp_test2
        {
          return Err(t.err(i, "Inconsistent use of test function in constant matrix."));
        }
      }
    }
    let n = &t.n[i];
    let to_add = (n.nb_test_functions() + n.nbc1) as isize - n.sizes.len() as isize;
    if !(0..=2).contains(&to_add) {
      return Err(Error::Internal("constant matrix sizes".into()));
    }
    let to_add = to_add as usize;
    if to_add > 0 {
      let mut mi = n.sizes.clone();
      mi.resize(n.nbc1 + n.nb_test_functions(), 0);
      for k in (to_add..mi.len()).rev() {
        mi[k] = mi[k - to_add];
      }
      for m in mi.iter_mut().take(to_add) {
        *m = 2;
      }
      let (tt, nt1, nt2) = (n.test, n.name_test1.clone(), n.name_test2.clone());
      for (bit, name, slot) in [(1, &nt1, 0), (2, &nt2, usize::from(tt & 1 != 0))] {
        // global or im_data test functions carry their true size
        if tt & bit != 0
          && let Some(v) = self.defs.var(name)
          && !matches!(v.support, Support::Fem { .. })
        {
          mi[slot] = v.qdim();
        }
      }
      t.n[i].adjust_sizes(&mi);
    }
    if all_cte {
      let ch = t.n[i].children.clone();
      let mut all_zero = true;
      for (k, &c) in ch.iter().enumerate() {
        let v = t.n[c].data[0];
        t.n[i].data[k] = v;
        all_zero &= v == 0.0;
      }
      t.n[i].ty = if all_zero { Ty::Zero } else { Ty::Constant };
      t.n[i].children.clear();
    }
    Ok(())
  }

  fn name(&mut self, i: usize, meshdim: usize) -> Result<()> {
    let red = self.ctx.ref_elt_dim;
    let t = &mut self.t;
    let mut name = t.n[i].name.clone();
    let n = &mut t.n[i];
    match name.as_str() {
      "X" if !self.ctx.ignore_x => {
        n.ty = Ty::X;
        n.nbc1 = 0;
        n.init_vector(meshdim);
        return Ok(());
      }
      "Diff" | "Grad" => {
        n.test = 0;
        return Ok(());
      }
      "element_size" => {
        n.ty = Ty::EltSize;
        n.init_scalar(0.0);
        return Ok(());
      }
      "Cross_product" => {
        n.ty = Ty::CrossProduct;
        n.test = 0;
        return Ok(());
      }
      "element_K" => {
        n.ty = Ty::EltK;
        if red == 1 {
          n.init_vector(meshdim)
        } else {
          n.init_matrix(meshdim, red)
        }
        return Ok(());
      }
      "element_B" => {
        n.ty = Ty::EltB;
        // N x P as K, a vector when P = 1
        if red == 1 {
          n.init_vector(meshdim)
        } else {
          n.init_matrix(meshdim, red)
        }
        return Ok(());
      }
      "element_curvature" => {
        n.ty = Ty::EltCurvature;
        n.init_matrix(meshdim, meshdim);
        return Ok(());
      }
      "Normal" => {
        n.ty = Ty::Normal;
        n.init_vector(meshdim);
        return Ok(());
      }
      "Reshape" | "Swap_indices" | "Index_move_last" | "Contract" => {
        n.ty = match name.as_str() {
          "Reshape" => Ty::Reshape,
          "Swap_indices" => Ty::SwapInd,
          "Index_move_last" => Ty::IndMoveLast,
          _ => Ty::Contract,
        };
        n.init_scalar(0.0);
        return Ok(());
      }
      _ => {}
    }
    if let Some(rest) = name.strip_prefix("Derivative_") {
      let mut rest = rest.to_string();
      let mut valid = true;
      n.der1 = 1;
      n.der2 = 0;
      for k in 0..2 {
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if !digits.is_empty() {
          let d = digits.parse().unwrap_or(0);
          if k == 0 {
            n.der1 = d;
          } else {
            n.der2 = d;
          }
          match rest[digits.len()..].strip_prefix('_') {
            Some(r) => rest = r.to_string(),
            None => valid = false,
          }
        }
      }
      if !valid || n.der1 == 0 {
        return Err(t.err(i, "Invalid derivative format"));
      }
      name = rest;
    }
    let n = &mut t.n[i];
    if let Some(f) = func(&name) {
      n.ty = Ty::PredefFunc;
      n.name = name.clone();
      n.test = 0;
      if n.der1 != 0 {
        if n.der1 > f.nargs || n.der2 > f.nargs {
          return Err(t.err(i, "Invalid derivative."));
        }
        // renamed to the function giving the derivative
        if n.der2 == 0 {
          match self.defs.derivative_name(&name, n.der1) {
            Some(d) => {
              n.name = d;
              n.der1 = 0;
            }
            None if f.user.is_none() => {
              return Err(t.err(i, format!("{name} cannot be differentiated")));
            }
            None => {}
          }
        }
      }
      return Ok(());
    }
    if names::SPEC_FUNCTIONS.contains(&name.as_str()) {
      if n.der1 != 0 {
        return Err(t.err(i, "Special functions do not support derivatives."));
      }
      n.ty = Ty::SpecFunc;
      n.name = name.clone();
      n.test = 0;
      match name.as_str() {
        "pi" => {
          n.ty = Ty::Constant;
          n.init_scalar(std::f64::consts::PI);
        }
        "meshdim" => {
          n.ty = Ty::Constant;
          n.init_scalar(meshdim as f64);
        }
        _ => {}
      }
      return Ok(());
    }
    if self.defs.operators.contains(&name)
      || names::OPERATORS.contains(&name.as_str())
      || names::is_law(&name)
    {
      n.ty = Ty::Operator;
      n.name = name;
      n.test = 0;
      return Ok(());
    }
    let prefix = strip_prefix(&mut name, strip_op);
    let test = strip_prefix(&mut name, strip_test);
    let Some(v) = self.defs.var(&name) else {
      let msg = format!("Unknown variable, function, operator or data \"{name}\"");
      return Err(t.err(i, msg));
    };
    if n.der1 != 0 {
      return Err(t.err(
        i,
        "Derivative is for functions or operators, not for variables. Use Grad instead.",
      ));
    }
    n.name = name.clone();
    if test != 0 && v.is_data && !v.disabled {
      return Err(t.err(i, "Test functions of constants are not allowed."));
    }
    let q = v.qdim();
    if test == 1 {
      n.name_test1 = name.clone();
      n.interp_test1.clear();
      if q == 0 {
        return Err(t.err(i, "Invalid null size of variable"));
      }
    } else if test == 2 {
      n.name_test2 = name.clone();
      n.interp_test2.clear();
      if q == 0 {
        return Err(t.err(i, "Invalid null size of variable"));
      }
    }
    let is_test = test != 0;
    match v.support {
      Support::Fixed => {
        if prefix != 0 {
          return Err(t.err(
            i,
            "Gradient, Hessian or Divergence cannot be evaluated for fixed size data.",
          ));
        }
        n.ty = if is_test {
          Ty::Field {
            src: Src::Plain,
            der: Der::Val,
            test: true,
          }
        } else {
          Ty::Field {
            src: Src::Plain,
            der: Der::Val,
            test: false,
          }
        };
        if q == 1 {
          if is_test {
            n.init_vector(1);
            n.data[0] = 1.0;
          } else {
            n.init_scalar(0.0);
          }
        } else if is_test {
          n.init_identity(q);
        } else {
          n.adjust_sizes(&v.qdims);
        }
      }
      Support::ImData => {
        let mut mii = v.qdims.clone();
        if q == 0 {
          return Err(t.err(i, format!("Invalid null size of variable {name}")));
        }
        if mii.len() > 6 {
          return Err(t.err(i, "Tensor with too many dimensions. Limited to 6"));
        }
        if prefix != 0 {
          return Err(t.err(
            i,
            "Gradient, Hessian or Divergence cannot be evaluated for im data.",
          ));
        }
        n.ty = Ty::Field {
          src: Src::Plain,
          der: Der::Val,
          test: is_test,
        };
        if is_test {
          if q == 1 && mii.len() <= 1 {
            n.init_vector(1);
            n.data[0] = 1.0;
          } else {
            mii.insert(0, q);
            n.adjust_sizes(&mii);
            n.clear_data();
            for a in 0..q {
              n.data[a * (q + 1)] = 1.0;
            }
          }
        } else {
          n.adjust_sizes(&mii);
        }
      }
      Support::Fem { mesh_dim: nd } => {
        let mut mii = v.qdims.clone();
        if q == 0 {
          return Err(t.err(i, format!("Invalid null size of variable {name}")));
        }
        if mii.len() > 6 {
          return Err(t.err(i, "Tensor with too many dimensions. Limited to 6"));
        }
        let lead = |mii: &mut Vec<usize>| {
          if q == 1 && mii.len() <= 1 {
            *mii = vec![2];
          } else {
            mii.insert(0, 2);
          }
        };
        let der = match prefix {
          0 => {
            if is_test {
              lead(&mut mii);
            }
            Der::Val
          }
          1 | 2 => {
            if is_test {
              lead(&mut mii);
            }
            if nd > 1 {
              if mii.len() == 1 && mii[0] == 1 {
                mii[0] = nd;
              } else {
                mii.push(nd);
              }
              if prefix == 2 {
                mii.push(nd);
              }
            }
            if prefix == 1 { Der::Grad } else { Der::Hess }
          }
          _ => {
            if q != nd {
              return Err(t.err(
                i,
                format!("Divergence operator can only be applied toFields with qdim ({q}) equal to dim ({nd})"),
              ));
            }
            mii = vec![if is_test { 2 } else { 1 }];
            Der::Div
          }
        };
        n.ty = Ty::Field {
          src: Src::Plain,
          der,
          test: is_test,
        };
        n.adjust_sizes(&mii);
      }
    }
    t.n[i].test = test;
    if test != 0 {
      self.test_insert(test, &name, "");
    }
    Ok(())
  }

  fn params(&mut self, i: usize, size1: &[usize], dim1: usize) -> Result<usize> {
    let meshdim = self.ctx.meshdim;
    let t = &mut self.t;
    let ch = t.n[i].children.clone();
    let (c0, c1) = (ch[0], ch.get(1).copied());
    let n0 = t.n[c0].clone();
    match n0.ty {
      Ty::Name if n0.name == "Grad" => return self.grad(i, &ch),
      Ty::Name if n0.name == "Diff" => return self.diff(i, &ch),
      Ty::Name => return Err(t.err(c0, "Unknown special operator")),
      Ty::X => {
        let Some(c1) = c1.filter(|_| ch.len() == 2) else {
          return Err(t.err(
            ch.get(1).copied().unwrap_or(c0),
            "X stands for the coordinates on the real elements. It accepts only one index.",
          ));
        };
        let n1 = &t.n[c1];
        if n1.ty != Ty::Constant || n1.data.len() != 1 {
          return Err(t.err(c1, "Index for X has to be constant and of size 1."));
        }
        let k = n1.data[0].round() as usize;
        if k == 0 || k > meshdim {
          return Err(t.err(
            c1,
            format!("Index for X not convenient. Found {k} with meshdim = {meshdim}"),
          ));
        }
        t.n[c0].init_scalar(0.0);
        t.n[c0].nbc1 = k;
        return Ok(t.replace_by_child(i, 0));
      }
      Ty::Reshape => {
        let c1 = c1.unwrap_or(c0);
        if ch.len() < 3 {
          return Err(t.err(c1, "Not enough parameters for Reshape"));
        }
        if ch.len() > 12 {
          return Err(t.err(c1, "Too many parameters for Reshape"));
        }
        t.copy_t(i, c1);
        t.copy_test(i, c1);
        let nt = t.n[i].nb_test_functions();
        let mut mi: Vec<usize> = size1[..nt].to_vec();
        for &c in &ch[2..] {
          if t.n[c].ty != Ty::Constant {
            return Err(t.err(c, "Reshape sizes should be constant positive integers."));
          }
          let s = t.n[c].data[0].round() as usize;
          if s == 0 {
            return Err(t.err(c, "Wrong zero size for Reshape."));
          }
          mi.push(s);
        }
        let total: usize = mi[nt..].iter().product();
        if total != t.n[i].tensor_proper_size() {
          return Err(t.err(
            i,
            format!(
              "Invalid sizes for reshape, found a total of {total} should be {}.",
              t.n[i].tensor_proper_size()
            ),
          ));
        }
        t.n[i].adjust_sizes(&mi);
        match t.n[c1].ty {
          Ty::Constant => {
            t.n[i].ty = Ty::Constant;
            t.n[i].children.clear();
          }
          Ty::Zero => t.set_zero(i),
          _ => {}
        }
      }
      Ty::CrossProduct => {
        if ch.len() != 3 {
          return Err(t.err(
            c1.unwrap_or(c0),
            "Wrong number of parameters for Cross_product",
          ));
        }
        // GetFEM's constant folding of Cross_product is disabled
        t.mult_test(i, ch[1], ch[2])?;
        let mut mi = t.n[i].sizes.clone();
        mi.push(3);
        t.n[i].adjust_sizes(&mi);
      }
      Ty::SwapInd => {
        let c1 = c1.unwrap_or(c0);
        if ch.len() != 4 {
          return Err(t.err(c1, "Wrong number of parameters for Swap_indices"));
        }
        t.copy_t(i, c1);
        t.copy_test(i, c1);
        let n1 = t.n[c1].clone();
        let mut ind = [0; 2];
        for (k, &c) in ch[2..4].iter().enumerate() {
          let cn = &t.n[c];
          if cn.ty != Ty::Constant || cn.data.len() != 1 {
            return Err(t.err(c, "Indices for swap should be constant positive integers."));
          }
          let v = cn.data[0].round() as usize;
          if v < 1 || v > n1.tensor_proper_size() {
            return Err(t.err(c, "Index out of range for Swap_indices."));
          }
          ind[k] = v - 1;
        }
        if ind[0] == ind[1] {
          return Ok(t.replace_by_child(i, 1));
        }
        let nbtf = n1.nb_test_functions();
        let mut mi = t.n[i].sizes.clone();
        mi.swap(ind[0] + nbtf, ind[1] + nbtf);
        t.n[i].adjust_sizes(&mi);
        match n1.ty {
          Ty::Constant => {
            // GetFEM reads the operator node here (uninitialized values)
            let mut perm: Vec<usize> = (0..n1.sizes.len()).collect();
            perm.swap(ind[0], ind[1]);
            t.n[i].data = permute(&n1.sizes, &n1.data, &perm);
            t.n[i].ty = Ty::Constant;
            t.n[i].children.clear();
          }
          Ty::Zero => t.set_zero(i),
          _ => {}
        }
      }
      Ty::IndMoveLast => {
        let c1 = c1.unwrap_or(c0);
        if ch.len() != 3 {
          return Err(t.err(c1, "Wrong number of parameters for Index_move_last"));
        }
        t.copy_t(i, c1);
        t.copy_test(i, c1);
        let n1 = t.n[c1].clone();
        let c2 = &t.n[ch[2]];
        if c2.ty != Ty::Constant || c2.data.len() != 1 {
          return Err(t.err(
            ch[2],
            "Index for Index_move_last should be constant positive integers.",
          ));
        }
        let ind = c2.data[0].round() as usize;
        if ind < 1 || ind > n1.tensor_proper_size() {
          return Err(t.err(ch[2], "Index out of range for Index_move_last."));
        }
        if ind == n1.tensor_order() {
          return Ok(t.replace_by_child(i, 1));
        }
        let mut mi = t.n[i].sizes.clone();
        let nbtf = n1.nb_test_functions();
        for k in ind..n1.tensor_order() {
          mi.swap(k - 1 + nbtf, k + nbtf);
        }
        t.n[i].adjust_sizes(&mi);
        match n1.ty {
          Ty::Constant => {
            // GetFEM reads the operator node here (uninitialized values)
            let mut perm: Vec<usize> = (0..n1.sizes.len()).filter(|&k| k != ind - 1).collect();
            perm.push(ind - 1);
            t.n[i].data = permute(&n1.sizes, &n1.data, &perm);
            t.n[i].ty = Ty::Constant;
            t.n[i].children.clear();
          }
          Ty::Zero => t.set_zero(i),
          _ => {}
        }
      }
      Ty::Contract => return self.contract(i, &ch, dim1),
      Ty::PredefFunc => {
        for &c in &ch[1..] {
          valid_operand(t, c)?;
        }
        let f = func(&n0.name).ok_or_else(|| Error::Internal("unknown function".into()))?;
        let nargs = f.nargs;
        if nargs + 1 != ch.len() {
          return Err(t.err(
            i,
            format!(
              "Bad number of arguments for predefined function {}. Found {}, should be {nargs}.",
              n0.name,
              ch.len() - 1
            ),
          ));
        }
        t.n[i].test = 0;
        let c1 = ch[1];
        let c2 = if nargs == 2 { ch[2] } else { c1 };
        let (n1, n2) = (t.n[c1].clone(), t.n[c2].clone());
        let all_cte = n1.ty == Ty::Constant && n2.ty == Ty::Constant;
        if n1.test != 0 || n2.test != 0 {
          return Err(t.err(
            i,
            "Test functions cannot be passed as argument of a predefined function.",
          ));
        }
        let (s1, s2) = (n1.data.len(), n2.data.len());
        if s1 != s2 {
          return Err(t.err(
            i,
            format!("Invalid argument size for a scalar function. Size of first argument: {s1}. Size of second argument: {s2}."),
          ));
        }
        t.copy_t(i, c1);
        // pow(x, 1) -> x, pow(x, 0) -> 1 (not GetFEM's)
        let pow = n0.name == "pow" && n0.der1 == 0 && n0.der2 == 0;
        let e = (pow && n2.ty == Ty::Constant && n2.data.len() == 1).then(|| n2.data[0]);
        if !all_cte && e == Some(1.0) {
          return Ok(t.replace_by_child(i, 1));
        }
        if !all_cte && e == Some(0.0) {
          t.n[i].ty = Ty::Constant;
          t.n[i].data.iter_mut().for_each(|v| *v = 1.0);
          t.n[i].children.clear();
          return Ok(i);
        }
        if all_cte {
          if n0.der1 != 0 {
            return Err(t.unsupported(i, "constant derivative of a function"));
          }
          t.n[i].ty = Ty::Constant;
          let len = t.n[i].data.len();
          for k in 0..len {
            let (a, b) = (n1.data[k], n2.data[k]);
            let v = match (f.builtin, &f.user) {
              (Some(names::Eval::One(g)), _) => Some(g(a)),
              (Some(names::Eval::Two(g)), _) => Some(g(a, b)),
              (_, Some(u)) => eval_user(self.defs, u, a, b, self.depth),
              _ => None,
            };
            let Some(v) = v else {
              return Err(t.unsupported(i, format!("constant evaluation of {}", n0.name)));
            };
            t.n[i].data[k] = v;
          }
          t.n[i].children.clear();
        }
      }
      Ty::SpecFunc => {
        for &c in &ch[1..] {
          valid_operand(t, c)?;
        }
        if ch.len() != 2 {
          return Err(t.err(
            i,
            format!(
              "One and only one argument is allowed for function {}.",
              n0.name
            ),
          ));
        }
        let n1 = t.n[ch[1]].clone();
        match n0.name.as_str() {
          "qdim" | "qdims" => {
            if !matches!(
              n1.ty,
              Ty::Field {
                src: Src::Plain,
                der: Der::Val,
                test: false
              }
            ) {
              return Err(t.err(
                i,
                "The argument of qdim function can only be a variable name.",
              ));
            }
            let v = self.defs.var(&n1.name).unwrap();
            t.n[i].ty = Ty::Constant;
            if n0.name == "qdim" {
              t.n[i].init_scalar(v.qdim() as f64);
              if v.qdim() == 0 {
                return Err(t.err(i, "Invalid null size of variable"));
              }
            } else {
              let mii = &v.qdims;
              if mii.len() > 6 {
                return Err(t.err(i, "Tensor with too much dimensions. Limited to 6"));
              }
              if mii.is_empty() || mii[0] == 0 {
                return Err(t.err(i, "Invalid null size of variable"));
              }
              t.n[i].init_vector(mii.len());
              for (k, &m) in mii.iter().enumerate() {
                t.n[i].data[k] = m as f64;
              }
            }
          }
          "Id" => {
            let n = if n1.ty == Ty::Constant {
              n1.data[0].round() as i64
            } else {
              -1
            };
            if n <= 0 || n > 100 || n1.tensor_order() > 0 {
              return Err(t.err(
                i,
                "The argument of Id should be a (small) positive integer.",
              ));
            }
            t.n[i].ty = Ty::Constant;
            if n == 1 {
              t.n[i].init_scalar(1.0);
            } else {
              t.n[i].init_identity(n as usize);
            }
          }
          _ => return Err(t.err(c0, "Unknown special function.")),
        }
        t.n[i].children.clear();
      }
      Ty::Operator => {
        for &c in &ch[1..] {
          valid_operand(t, c)?;
        }
        let mut all_cte = true;
        for &c in &ch[1..] {
          let cn = &t.n[c];
          all_cte &= cn.ty == Ty::Constant;
          if cn.ty == Ty::AllIndices {
            return Err(t.err(
              c,
              "Colon operator is not allowed in nonlinear operator call.",
            ));
          }
          if cn.test != 0 {
            return Err(t.err(
              i,
              "Test functions cannot be passed as argument of a nonlinear operator.",
            ));
          }
          if cn.tensor_order() > 2 {
            return Err(t.err(
              i,
              "Sorry, arguments to nonlinear operators should only be scalar, vector or matrices",
            ));
          }
        }
        let args: Vec<Vec<usize>> = ch[1..].iter().map(|&c| t.n[c].sizes.clone()).collect();
        let shapes: Vec<&[usize]> = args.iter().map(|a| a.as_slice()).collect();
        let Some(op) = self.defs.operators.get(&n0.name) else {
          return Err(t.unsupported(
            i,
            format!("nonlinear operator {} (not ported yet)", n0.name),
          ));
        };
        let Some(mut mi) = op.result_shape(&shapes) else {
          return Err(t.err(
            i,
            format!(
              "Wrong number or wrong size of arguments for the call of nonlinear operator {}",
              n0.name
            ),
          ));
        };
        t.n[i].test = 0;
        if n0.der1 > args.len() || n0.der2 > args.len() {
          return Err(t.err(
            c0,
            format!(
              "Invalid derivative number for nonlinear operator {}",
              n0.name
            ),
          ));
        }
        if n0.der1 != 0 {
          mi.extend(&args[n0.der1 - 1]);
          if n0.der2 != 0 {
            mi.extend(&args[n0.der2 - 1]);
          }
        }
        t.n[i].adjust_sizes(&mi);
        if all_cte {
          let data: Vec<Vec<f64>> = ch[1..].iter().map(|&c| t.n[c].data.clone()).collect();
          let vals: Vec<&[f64]> = data.iter().map(|d| d.as_slice()).collect();
          let mut out = vec![0.0; mi.iter().product()];
          let r = match (n0.der1, n0.der2) {
            (0, _) => op.value(&vals, &shapes, &mut out),
            (d, 0) => op.derivative(&vals, &shapes, d - 1, &mut out),
            (d, e) => op.second_derivative(&vals, &shapes, d - 1, e - 1, &mut out),
          };
          r.map_err(|m| t.err(i, format!("{}: {m}", n0.name)))?;
          t.n[i].data = out;
          t.n[i].ty = Ty::Constant;
          t.n[i].children.clear();
        }
      }
      _ => return self.components(i, &ch),
    }
    Ok(i)
  }

  /// Diff(expr, u) and Diff(expr, u, direction)
  fn diff(&mut self, i: usize, ch: &[usize]) -> Result<usize> {
    let c0 = ch[0];
    if ch.len() != 3 && ch.len() != 4 {
      return Err(self.t.err(c0, "Bad number of parameters for Diff operator"));
    }
    let (e, v) = (ch[1], ch[2]);
    if !matches!(
      self.t.n[v].ty,
      Ty::Field {
        src: Src::Plain,
        der: Der::Val,
        test: false
      }
    ) {
      return Err(self.t.err(
        v,
        "Second parameter of Diff operator has to be a variable name",
      ));
    }
    let var = self.t.n[v].name.clone();
    let order = self.t.n[e].test;
    if order > 1 {
      return Err(
        self
          .t
          .err(v, "Cannot derive further this order two expression"),
      );
    }
    let dir = ch.get(3).copied();
    let mut sub = self.t.clone();
    sub.root = Some(e);
    // var through any transformation (any_trans)
    let r = match crate::derive::derive_trans(&sub, self.defs, &var, "", true, order + 1, dir)? {
      Some(d) => {
        // the derivative lives in a superset of the arena
        let r = d.root.unwrap();
        self.t.n = d.n;
        if dir.is_some() {
          // GetFEM derives with a test function first, then substitutes
          self.test_insert(order + 1, &var, "");
        }
        r
      }
      None => {
        let mut z = Node::new(Ty::Zero, self.t.n[e].pos);
        let mut sizes = self.t.n[e].sizes.clone();
        if dir.is_none() {
          sizes.insert(0, 2);
          z.test = if order == 0 { 1 } else { 3 };
          // the test data of the expression (GetFEM: the node itself)
          let ne = &self.t.n[e];
          z.name_test1 = ne.name_test1.clone();
          z.name_test2 = ne.name_test2.clone();
          z.interp_test1 = ne.interp_test1.clone();
          z.interp_test2 = ne.interp_test2.clone();
          if order == 0 {
            z.name_test1 = var.clone();
          } else {
            z.name_test2 = var.clone();
          }
        } else {
          z.test = order;
        }
        z.adjust_sizes(&sizes);
        z.clear_data();
        self.t.push(z)
      }
    };
    let p = self.t.n[i].parent;
    self.t.n[r].parent = p;
    match p {
      Some(p) => self.t.replace_child(p, i, r),
      None => self.t.root = Some(r),
    }
    self.node(r)
  }

  /// does the subtree depend on the position (GetFEM ga_node_mark_tree_for_grad)
  fn depends_on_x(&self, i: usize) -> bool {
    let n = &self.t.n[i];
    let fem = |name: &str| {
      self
        .defs
        .var(name)
        .is_some_and(|v| matches!(v.support, Support::Fem { .. }))
    };
    let own = match n.ty {
      Ty::Field { .. } => fem(&n.name),
      Ty::X | Ty::Normal | Ty::InterpX | Ty::InterpEltK | Ty::InterpEltB | Ty::InterpNormal => true,
      _ => false,
    };
    own || n.children.iter().any(|&c| self.depends_on_x(c))
  }

  /// Grad(expr): GetFEM differentiates the analysed subtree; values and
  /// gradients of fields become gradients and Hessians, a subtree that does
  /// not depend on the position becomes zero. Other expressions need M7.
  fn grad(&mut self, i: usize, ch: &[usize]) -> Result<usize> {
    let meshdim = self.ctx.meshdim;
    if ch.len() != 2 {
      return Err(
        self
          .t
          .err(ch[0], "Bad number of parameters for Grad operator"),
      );
    }
    let c1 = ch[1];
    if !self.depends_on_x(c1) {
      let n = &mut self.t.n[c1];
      let mut mi = n.sizes.clone();
      mi.push(meshdim);
      n.adjust_sizes(&mi);
      self.t.set_zero(c1);
      return Ok(self.t.replace_by_child(i, 1));
    }
    let n = &mut self.t.n[c1];
    match n.ty {
      Ty::Field {
        src: Src::Plain,
        der: der @ (Der::Val | Der::Grad),
        test,
      } => {
        n.ty = Ty::Field {
          src: Src::Plain,
          der: if der == Der::Val {
            Der::Grad
          } else {
            Der::Hess
          },
          test,
        };
        let mut mi = n.sizes.clone();
        if *mi.last().unwrap() != 1 {
          mi.push(meshdim);
        } else {
          *mi.last_mut().unwrap() = meshdim;
        }
        n.adjust_sizes(&mi);
        Ok(self.t.replace_by_child(i, 1))
      }
      Ty::Field {
        src: Src::Plain,
        der: Der::Hess,
        ..
      } => Err(self.t.err(c1, "Sorry, cannot derive an Hessian once more")),
      _ => {
        // general expression: build its gradient and analyse it
        let mut sub = self.t.clone();
        sub.root = Some(c1);
        let d = crate::derive::grad(&sub, self.defs, meshdim)?
          .ok_or_else(|| Error::Internal("gradient of a constant".into()))?;
        let r = d.root.unwrap();
        self.t.n = d.n;
        let p = self.t.n[i].parent;
        self.t.n[r].parent = p;
        match p {
          Some(p) => self.t.replace_child(p, i, r),
          None => self.t.root = Some(r),
        }
        self.node(r)
      }
    }
  }

  /// Contract(A, i, j), Contract(A, i, B, j), Contract(A, i, j, B, k, l)
  fn contract(&mut self, i: usize, ch: &[usize], dim1: usize) -> Result<usize> {
    let t = &mut self.t;
    let mut ind: Vec<usize> = match ch.len() {
      4 => vec![2, 3],
      5 => vec![2, 4],
      7 => vec![2, 3, 5, 6],
      _ => {
        return Err(t.err(
          ch.get(1).copied().unwrap_or(i),
          "Wrong number of parameters for Contract",
        ));
      }
    };
    let mut indsize = vec![0; ind.len()];
    let (mut order, mut kk, mut ll) = (0, 0, 1);
    for (k, &c) in ch.iter().enumerate().skip(1) {
      if kk < ind.len() && k == ind[kk] {
        let cn = &t.n[c];
        if cn.ty != Ty::Constant || cn.data.len() != 1 {
          return Err(t.err(
            c,
            "Invalid parameter for Contract. Should be an index number.",
          ));
        }
        let v = cn.data[0].round() as usize;
        order = t.n[ch[ll]].tensor_order();
        if v < 1 || v > order {
          return Err(t.err(
            c,
            format!("Parameter out of range for Contract (should be between 1 and {order})"),
          ));
        }
        ind[kk] = v - 1;
        indsize[kk] = t.n[ch[ll]].proper_size(ind[kk]);
        let h = ind.len() / 2;
        if kk >= h && indsize[kk] != indsize[kk - h] {
          return Err(t.err(
            ch[0],
            "Invalid parameters for Contract. Cannot contract on indices of different sizes",
          ));
        }
        kk += 1;
      } else {
        ll = k;
      }
    }
    let n1 = t.n[ch[1]].clone();
    match ch.len() {
      4 => {
        t.copy_test(i, ch[1]);
        let (i1, i2) = (ind[0], ind[1]);
        if i1 == i2 {
          return Err(t.err(ch[0], "Invalid parameters for Contract. Repeated index."));
        }
        let nt = t.n[i].nb_test_functions();
        let mut mi: Vec<usize> = n1.sizes[..nt].to_vec();
        mi.extend(
          (0..order)
            .filter(|&k| k != i1 && k != i2)
            .map(|k| n1.proper_size(k)),
        );
        t.n[i].adjust_sizes(&mi);
        match n1.ty {
          Ty::Constant => {
            let rest: Vec<usize> = (0..order).filter(|&k| k != i1 && k != i2).collect();
            let out = &t.n[i].sizes.clone();
            let mut src = vec![0; order];
            for k in 0..t.n[i].data.len() {
              let mut q = k;
              for (j, &r) in rest.iter().enumerate() {
                src[r] = q % out[j];
                q /= out[j];
              }
              let mut v = 0.0;
              for n in 0..indsize[0] {
                src[i1] = n;
                src[i2] = n;
                v += n1.at(&src);
              }
              t.n[i].data[k] = v;
            }
            t.n[i].ty = Ty::Constant;
            t.n[i].children.clear();
          }
          Ty::Zero => t.set_zero(i),
          _ => {}
        }
      }
      5 => {
        let c2 = ch[3];
        t.mult_test(i, ch[1], c2)?;
        let n2 = t.n[c2].clone();
        let (i1, i2) = (ind[0], ind[1]);
        let mut mi = t.n[i].sizes.clone();
        mi.extend((0..dim1).filter(|&k| k != i1).map(|k| n1.proper_size(k)));
        mi.extend((0..order).filter(|&k| k != i2).map(|k| n2.proper_size(k)));
        t.n[i].adjust_sizes(&mi);
        // GetFEM computes but does not mark constant two-tensor contractions
        if !(n1.ty == Ty::Constant && n2.ty == Ty::Constant)
          && (n1.tensor_is_zero() || n2.tensor_is_zero())
        {
          t.set_zero(i);
        }
      }
      _ => {
        let c2 = ch[4];
        t.mult_test(i, ch[1], c2)?;
        let n2 = t.n[c2].clone();
        if ind[0] == ind[1] || ind[2] == ind[3] {
          return Err(t.err(ch[0], "Invalid parameters for Contract. Repeated index."));
        }
        let mut mi = t.n[i].sizes.clone();
        mi.extend(
          (0..dim1)
            .filter(|&k| k != ind[0] && k != ind[1])
            .map(|k| n1.proper_size(k)),
        );
        mi.extend(
          (0..order)
            .filter(|&k| k != ind[2] && k != ind[3])
            .map(|k| n2.proper_size(k)),
        );
        t.n[i].adjust_sizes(&mi);
        if n1.tensor_is_zero() || n2.tensor_is_zero() {
          t.set_zero(i);
        }
      }
    }
    Ok(i)
  }

  /// A(i, :, j): component access
  fn components(&mut self, i: usize, ch: &[usize]) -> Result<usize> {
    let t = &mut self.t;
    let c0 = ch[0];
    let n0 = t.n[c0].clone();
    let all_cte = n0.ty == Ty::Constant;
    if ch.len() != n0.tensor_order() + 1 {
      return Err(t.err(i, "Bad number of indices."));
    }
    for &c in &ch[1..] {
      let cn = &t.n[c];
      if cn.ty != Ty::AllIndices && (cn.ty != Ty::Constant || cn.data.len() != 1) {
        return Err(t.err(c, "Indices should be constant integers or colon."));
      }
    }
    let mut mi1 = vec![0; n0.sizes.len()];
    let (mut mi2, mut indices) = (vec![], vec![]);
    for k in 0..n0.tensor_order() {
      let cn = &t.n[ch[k + 1]];
      if cn.ty == Ty::AllIndices {
        mi2.push(n0.proper_size(k));
        indices.push(k);
      } else {
        let v = cn.data[0].round() - 1.0;
        if v < 0.0 || v as usize >= n0.proper_size(k) {
          return Err(t.err(
            ch[k + 1],
            format!(
              "Index out of range, {}. Should be between 1 and {}.",
              v + 1.0,
              n0.proper_size(k)
            ),
          ));
        }
        mi1[k] = v as usize;
      }
    }
    let nt = n0.nb_test_functions();
    let mut mi: Vec<usize> = n0.sizes[..nt].to_vec();
    mi.extend(&mi2);
    t.n[i].adjust_sizes(&mi);
    t.copy_test(i, c0);
    if n0.tensor_is_zero() {
      t.set_zero(i);
    } else if all_cte {
      t.n[i].ty = Ty::Constant;
      // GetFEM indexes the child with mi1 as a full multi-index (no tests)
      let total: usize = mi2.iter().product();
      let mut mi3 = vec![0; mi2.len()];
      for k in 0..total {
        let mut r = k;
        for (j, m) in mi2.iter().enumerate() {
          mi3[j] = r % m;
          r /= m;
        }
        for (j, &ix) in indices.iter().enumerate() {
          mi1[ix] = mi3[j];
        }
        t.n[i].data[flat(&mi, &mi3)] = n0.at(&mi1);
      }
      t.n[i].children.clear();
    }
    Ok(i)
  }
}
