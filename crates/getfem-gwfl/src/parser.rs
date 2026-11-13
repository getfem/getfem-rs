//! GWFL syntax analysis: a port of GetFEM's pushdown reader (ga_read_term)
//! and tree construction (operator climbing), so parses are identical,
//! including macro expansion. The result is an owned Expr tree.

use std::collections::HashMap;

use crate::lexer::{Tok, next_token};
use crate::{Error, Result};

// --------------------------------------------------------------- the AST

#[derive(Clone, Debug, PartialEq)]
pub struct Expr {
  pub kind: ExprKind,
  /// byte offset in the source expression
  pub pos: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExprKind {
  Number(f64),
  /// ':' as a component index (all indices)
  AllIndices,
  Name(String),
  /// Unary minus, Sym, Skew, Trace, Deviator, Print, transpose (Quote)
  Unary(Tok, Box<Expr>),
  Binary(Tok, Box<Expr>, Box<Expr>),
  /// f(args): function, operator or component access
  Call(Box<Expr>, Vec<Expr>),
  /// explicit tensor, components in Fortran order
  Tensor {
    shape: Vec<usize>,
    comps: Vec<Expr>,
  },
  Interpolate {
    name: String,
    trans: String,
  },
  InterpolateFilter {
    trans: String,
    expr: Box<Expr>,
    id: Option<Box<Expr>>,
  },
  InterpolateDerivative {
    trans_der: String,
    name: String,
    trans: String,
  },
  Elementary {
    name: String,
    trans: String,
    target: String,
  },
  SecondaryDomain {
    name: String,
  },
  XfemPlus {
    name: String,
  },
  XfemMinus {
    name: String,
  },
}

// ------------------------------------------------------------ the arena

#[derive(Clone, Debug)]
enum Kind {
  Scalar(f64),
  AllIndices,
  Name(String),
  Op(Tok),
  Params,
  Matrix {
    shape: Vec<usize>,
    marked: bool,
  },
  Interpolate {
    name: String,
    trans: String,
  },
  InterpolateFilter {
    trans: String,
  },
  InterpolateDerivative {
    trans_der: String,
    name: String,
    trans: String,
  },
  Elementary {
    name: String,
    trans: String,
    target: String,
  },
  SecondaryDomain {
    name: String,
  },
  XfemPlus {
    name: String,
  },
  XfemMinus {
    name: String,
  },
  /// parameter i of a macro, with Grad_/Hess_/Div_ (po) and Test_/Test2_ (pt)
  /// prefixes, and the kind of the node it replaced
  MacroParam {
    index: usize,
    po: u8,
    pt: u8,
    orig: Box<Kind>,
  },
}

#[derive(Clone, Debug)]
struct Node {
  kind: Kind,
  children: Vec<usize>,
  parent: Option<usize>,
  pos: usize,
}

#[derive(Default, Clone, Debug)]
struct Arena {
  n: Vec<Node>,
}

#[derive(Default, Clone, Copy, Debug)]
struct Tree {
  root: Option<usize>,
  current: Option<usize>,
}

impl Arena {
  fn new_node(&mut self, kind: Kind, pos: usize) -> usize {
    self.n.push(Node {
      kind,
      children: vec![],
      parent: None,
      pos,
    });
    self.n.len() - 1
  }

  fn is_op(&self, i: usize) -> bool {
    matches!(self.n[i].kind, Kind::Op(_))
  }

  fn op_priority(&self, i: usize) -> Option<u8> {
    match self.n[i].kind {
      Kind::Op(t) => Some(t.priority()),
      _ => None,
    }
  }

  fn adopt(&mut self, parent: usize, child: usize) {
    self.n[parent].children.push(child);
    self.n[child].parent = Some(parent);
  }

  fn replace_child(&mut self, parent: usize, old: usize, new: usize) {
    for c in self.n[parent].children.iter_mut() {
      if *c == old {
        *c = new;
      }
    }
    self.n[new].parent = Some(parent);
  }

  /// a copy of subtree i (nodes allocated first, as copy_between)
  fn copy(&mut self, i: usize) -> usize {
    self.n.reserve(count(self, i));
    self.copy_rec(i)
  }

  fn copy_rec(&mut self, i: usize) -> usize {
    let (kind, pos, ch) = (
      self.n[i].kind.clone(),
      self.n[i].pos,
      self.n[i].children.clone(),
    );
    let c = self.new_node(kind, pos);
    self.n[c].children.reserve_exact(ch.len());
    for k in ch {
      let cc = self.copy_rec(k);
      self.adopt(c, cc);
    }
    c
  }

  /// attach a leaf at the nearest operator above current (GetFEM add_name etc.)
  fn add_leaf(&mut self, t: &mut Tree, kind: Kind, pos: usize) {
    let mut cur = t.current;
    while let Some(c) = cur {
      if self.is_op(c) {
        break;
      }
      cur = self.n[c].parent;
    }
    let nn = self.new_node(kind, pos);
    match cur {
      Some(c) => self.adopt(c, nn),
      None => t.root = Some(nn),
    }
    t.current = Some(nn);
  }

  fn add_op(&mut self, t: &mut Tree, op: Tok, pos: usize) {
    let pr = op.priority();
    let mut cur = t.current;
    while let Some(c) = cur {
      match self.n[c].parent.and_then(|p| self.op_priority(p)) {
        Some(pp) if pp >= pr => cur = self.n[c].parent,
        _ => break,
      }
    }
    let nn = self.new_node(Kind::Op(op), pos);
    match cur {
      Some(c) => {
        if op.is_prefix() {
          self.adopt(c, nn);
        } else {
          let parent = self.n[c].parent;
          match parent {
            Some(p) => self.replace_child(p, c, nn),
            None => {
              t.root = Some(nn);
              self.n[nn].parent = None;
            }
          }
          self.adopt(nn, c);
        }
      }
      None => {
        if let Some(r) = t.root {
          self.adopt(nn, r);
        }
        t.root = Some(nn);
      }
    }
    t.current = Some(nn);
  }

  fn add_params(&mut self, t: &mut Tree, pos: usize) {
    let mut c = t.current.expect("params after a term");
    while let Some(p) = self.n[c].parent {
      match self.op_priority(p) {
        Some(pp) if pp >= 4 => c = p,
        _ => break,
      }
    }
    let nn = self.new_node(Kind::Params, pos);
    match self.n[c].parent {
      Some(p) => self.replace_child(p, c, nn),
      None => t.root = Some(nn),
    }
    self.adopt(nn, c);
    t.current = Some(nn);
  }

  fn add_sub_tree(&mut self, t: &mut Tree, sub: &mut Tree) -> Result<()> {
    let r = sub
      .root
      .ok_or_else(|| Error::Internal("empty sub-expression".into()))?;
    let cur_is_container = t.current.is_some_and(|c| {
      matches!(
        self.n[c].kind,
        Kind::Params | Kind::InterpolateFilter { .. } | Kind::Matrix { .. }
      )
    });
    if cur_is_container {
      self.adopt(t.current.unwrap(), r);
    } else {
      let mut cur = t.current;
      while let Some(c) = cur {
        if self.is_op(c) {
          break;
        }
        cur = self.n[c].parent;
      }
      match cur {
        Some(c) => self.adopt(c, r),
        None => {
          t.root = Some(r);
          self.n[r].parent = None;
        }
      }
      t.current = Some(r);
    }
    *sub = Tree::default();
    Ok(())
  }
}

// ---------------------------------------------------------------- macros

#[derive(Clone, Debug)]
struct Macro {
  nb_params: usize,
  arena: Arena,
  root: Option<usize>,
}

/// Macros of a workspace (name -> definition).
#[derive(Clone, Debug, Default)]
pub struct MacroDict {
  macros: HashMap<String, Macro>,
}

impl MacroDict {
  pub fn exists(&self, name: &str) -> bool {
    self.macros.contains_key(name)
  }

  /// Add "name(a,b)" := "expr", as GetFEM's add_macro.
  pub fn add(&mut self, name: &str, expr: &str) -> Result<()> {
    let src = format!("Def {name}:={expr}");
    let mut p = Parser::new(&src, self);
    p.read_all()?;
    let local = p.local.macros;
    self.macros.extend(local);
    Ok(())
  }

  pub fn remove(&mut self, name: &str) -> bool {
    self.macros.remove(name).is_some()
  }
}

// ---------------------------------------------------------------- parser

pub const PREFIX_OP: [&str; 3] = ["Grad_", "Hess_", "Div_"];

/// derivative prefix: 0 none, 1 Grad, 2 Hess, 3 Div; the rest
pub(crate) fn strip_op(name: &str) -> (u8, &str) {
  for (i, p) in PREFIX_OP.iter().enumerate() {
    if let Some(r) = name.strip_prefix(p) {
      return (i as u8 + 1, r);
    }
  }
  (0, name)
}

/// test prefix: 0 none, 1 Test, 2 Test2; the rest
pub(crate) fn strip_test(name: &str) -> (u8, &str) {
  if let Some(r) = name.strip_prefix("Test_") {
    (1, r)
  } else if let Some(r) = name.strip_prefix("Test2_") {
    (2, r)
  } else {
    (0, name)
  }
}

struct Parser<'a> {
  src: &'a str,
  s: &'a [u8],
  pos: usize,
  a: Arena,
  /// the workspace's macros (borrowed, not copied per parse), those
  /// defined in the expression (Def)
  base: &'a MacroDict,
  local: MacroDict,
}

fn err(src: &str, pos: usize, msg: impl Into<String>) -> Error {
  Error::Syntax {
    expr: src.to_string(),
    pos,
    msg: msg.into(),
  }
}

impl<'a> Parser<'a> {
  fn new(src: &'a str, base: &'a MacroDict) -> Self {
    Parser {
      src,
      s: src.as_bytes(),
      pos: 0,
      a: Arena::default(),
      base,
      local: MacroDict::default(),
    }
  }

  /// the definition of macro n (the expression's first)
  fn macro_of(&self, n: &str) -> Option<&Macro> {
    self.local.macros.get(n).or_else(|| self.base.macros.get(n))
  }

  fn tok(&mut self) -> (Tok, usize, usize) {
    next_token(self.s, &mut self.pos)
  }

  fn text(&self, start: usize, len: usize) -> String {
    self.src[start..start + len].to_string()
  }

  fn e(&self, pos: usize, msg: &str) -> Error {
    err(self.src, pos, msg)
  }

  fn expect_name(&mut self, msg: &str) -> Result<String> {
    let (t, p, l) = self.tok();
    if t != Tok::Name {
      return Err(self.e(p, msg));
    }
    Ok(self.text(p, l))
  }

  fn expect(&mut self, t: Tok, msg: &str) -> Result<()> {
    let (tt, p, _) = self.tok();
    if tt != t {
      return Err(self.e(p, msg));
    }
    Ok(())
  }

  fn read_all(&mut self) -> Result<Tree> {
    let mut t = Tree::default();
    let r = self.read_term(&mut t)?;
    if let Some(root) = t.root {
      let root = self.expand(root)?;
      t.root = Some(root);
    }
    match r {
      Tok::End => Ok(t),
      Tok::RPar => Err(self.e(self.pos.saturating_sub(1), "Unbalanced parenthesis.")),
      Tok::RBracket => Err(self.e(self.pos.saturating_sub(1), "Unbalanced bracket.")),
      _ => Err(self.e(self.pos.saturating_sub(1), "Unexpected token.")),
    }
  }

  /// Read a term; returns the token that ended it (GetFEM ga_read_term).
  fn read_term(&mut self, t: &mut Tree) -> Result<Tok> {
    let mut state = 1;
    loop {
      let (tt, tp, tl) = self.tok();
      if state == 1 {
        match tt {
          Tok::Scalar => {
            let txt = &self.src[tp..tp + tl];
            let v: f64 = txt.parse().map_err(|_| self.e(tp, "Bad numeric format."))?;
            self.a.add_leaf(t, Kind::Scalar(v), tp);
            state = 2;
          }
          Tok::Colon => {
            self.a.add_leaf(t, Kind::AllIndices, tp);
            state = 2;
          }
          Tok::Name => {
            let n = self.text(tp, tl);
            self.a.add_leaf(t, Kind::Name(n), tp);
            state = 2;
          }
          Tok::Minus => self.a.add_op(t, Tok::UnaryMinus, tp),
          Tok::Plus => {}
          Tok::Sym | Tok::Skew | Tok::Trace | Tok::Deviator | Tok::Print => {
            self.a.add_op(t, tt, tp)
          }
          Tok::Def => {
            let r = self.read_macro_def()?;
            if r == Tok::End {
              return Ok(r);
            }
          }
          Tok::Interpolate => {
            self.expect(Tok::LPar, "Missing interpolate arguments.")?;
            let mut name = self.expect_name(
              "First argument of Interpolate should be a variable, test function, X or Normal.",
            )?;
            if name == "Grad" {
              self.expect(Tok::LPar, "Missing Grad argument.")?;
              name = format!(
                "Grad_{}",
                self.expect_name("Argument of Grad should be a variable name, here.")?
              );
              self.expect(
                Tok::RPar,
                "The sole argument of Grad should be a variable name, here.",
              )?;
            }
            self.expect(Tok::Comma, "Bad format for Interpolate arguments.")?;
            let trans = self
              .expect_name("Second argument of Interpolate should be a transformation name.")?;
            self.expect(
              Tok::RPar,
              "Missing a parenthesis after interpolate arguments.",
            )?;
            self.a.add_leaf(t, Kind::Interpolate { name, trans }, tp);
            state = 2;
          }
          Tok::InterpolateDerivative => {
            self.expect(Tok::LPar, "Missing Interpolate_derivative arguments.")?;
            let trans_der = self.expect_name(
              "First argument of Interpolate should the interpolate transformation name ",
            )?;
            self.expect(
              Tok::Comma,
              "Bad format for Interpolate_derivative arguments.",
            )?;
            let name =
              self.expect_name("Second argument of Interpolate should be a variable name.")?;
            let (mut tt2, mut tp2, _) = self.tok();
            let mut trans = String::new();
            if tt2 == Tok::Comma {
              trans = self.expect_name(
                "Third argument of Interpolate should be a interpolate transformation name.",
              )?;
              (tt2, tp2, _) = self.tok();
            }
            if tt2 != Tok::RPar {
              return Err(self.e(
                tp2,
                "Missing a parenthesis after Interpolate_derivative arguments.",
              ));
            }
            self.a.add_leaf(
              t,
              Kind::InterpolateDerivative {
                trans_der,
                name,
                trans,
              },
              tp,
            );
            state = 2;
          }
          Tok::Elementary => {
            self.expect(Tok::LPar, "Missing Elementary_transformation arguments.")?;
            let name = self.expect_name("First argument of Elementary_transformation should be a variable or a test function.")?;
            self.expect(
              Tok::Comma,
              "Bad format for Elementary_transformation arguments.",
            )?;
            let trans = self.expect_name(
              "Second argument of Elementary_transformation should be a transformation name.",
            )?;
            let (mut tt2, mut tp2, _) = self.tok();
            let mut target = name.clone();
            if tt2 == Tok::Comma {
              target = self.expect_name(
                "Third argument of Elementary_transformation should be a variable or data name.",
              )?;
              (tt2, tp2, _) = self.tok();
            }
            if tt2 != Tok::RPar {
              return Err(self.e(
                tp2,
                "Missing a parenthesis after Elementary_transformation arguments.",
              ));
            }
            self.a.add_leaf(
              t,
              Kind::Elementary {
                name,
                trans,
                target,
              },
              tp,
            );
            state = 2;
          }
          Tok::SecondaryDomain | Tok::XfemPlus | Tok::XfemMinus => {
            let what = match tt {
              Tok::SecondaryDomain => "Secondary_domain",
              Tok::XfemPlus => "Xfem_plus",
              _ => "Xfem_minus",
            };
            self.expect(Tok::LPar, &format!("Missing {what} arguments."))?;
            let msg = format!("The argument of {what} should be a variable or a test function.");
            let mut name = self.expect_name(&msg)?;
            // Grad(u), Hess(u), Div(u): Grad_u, Hess_u, Div_u
            let save = self.pos;
            if matches!(name.as_str(), "Grad" | "Hess" | "Div") && self.tok().0 == Tok::LPar {
              let inner = self.expect_name(&msg)?;
              self.expect(
                Tok::RPar,
                &format!("Missing a parenthesis after {what} argument."),
              )?;
              name = format!("{name}_{inner}");
            } else {
              self.pos = save;
            }
            self.expect(
              Tok::RPar,
              &format!("Missing a parenthesis after {what} argument."),
            )?;
            let k = match tt {
              Tok::SecondaryDomain => Kind::SecondaryDomain { name },
              Tok::XfemPlus => Kind::XfemPlus { name },
              _ => Kind::XfemMinus { name },
            };
            self.a.add_leaf(t, k, tp);
            state = 2;
          }
          Tok::InterpolateFilter => {
            self.expect(Tok::LPar, "Missing interpolate arguments.")?;
            let trans = self.expect_name(
              "First argument of Interpolate_filter should be a transformation name.",
            )?;
            self.expect(Tok::Comma, "Bad format for Interpolate_filter arguments.")?;
            self.a.add_leaf(t, Kind::InterpolateFilter { trans }, tp);
            let mut sub = Tree::default();
            let mut r = self.read_term(&mut sub)?;
            if r != Tok::RPar && r != Tok::Comma {
              return Err(self.e(
                self.pos.saturating_sub(1),
                "Bad format for Interpolate_filter arguments.",
              ));
            }
            self.a.add_sub_tree(t, &mut sub)?;
            if r == Tok::Comma {
              let mut sub2 = Tree::default();
              r = self.read_term(&mut sub2)?;
              self.a.add_sub_tree(t, &mut sub2)?;
            }
            if r != Tok::RPar {
              return Err(self.e(self.pos.saturating_sub(1), "Unbalanced parenthesis."));
            }
            state = 2;
          }
          Tok::LPar => {
            let mut sub = Tree::default();
            let r = self.read_term(&mut sub)?;
            if r != Tok::RPar {
              return Err(self.e(self.pos.saturating_sub(1), "Unbalanced parenthesis."));
            }
            self.a.add_sub_tree(t, &mut sub)?;
            state = 2;
          }
          Tok::LBracket => {
            self.read_tensor(t, tp)?;
            state = 2;
          }
          _ => return Err(self.e(tp, "Unexpected token.")),
        }
      } else {
        match tt {
          Tok::Plus
          | Tok::Minus
          | Tok::Mult
          | Tok::Div
          | Tok::Colon
          | Tok::Dot
          | Tok::DotMult
          | Tok::DotDiv
          | Tok::TMult => {
            self.a.add_op(t, tt, tp);
            state = 1;
          }
          Tok::Quote => self.a.add_op(t, tt, tp),
          Tok::End
          | Tok::RPar
          | Tok::Comma
          | Tok::DComma
          | Tok::RBracket
          | Tok::Semicolon
          | Tok::DSemicolon => return Ok(tt),
          Tok::LPar => {
            self.a.add_params(t, tp);
            loop {
              let mut sub = Tree::default();
              let r = self.read_term(&mut sub)?;
              if r != Tok::RPar && r != Tok::Comma {
                let p = self.pos - if r != Tok::End { 1 } else { 0 };
                return Err(self.e(
                  p,
                  "Parameters should be separated by ',' and parameter list ended by ')'.",
                ));
              }
              self.a.add_sub_tree(t, &mut sub)?;
              if r == Tok::RPar {
                break;
              }
            }
          }
          _ => return Err(self.e(tp, "Unexpected token.")),
        }
      }
    }
  }

  fn read_macro_def(&mut self) -> Result<Tok> {
    let (tt, tp, tl) = self.tok();
    if tt != Tok::Name {
      return Err(self.e(tp, "Macro definition should begin with macro name"));
    }
    let name = self.text(tp, tl);
    if crate::names::reserved(&name).is_some() {
      return Err(self.e(tp, "Invalid macro name."));
    }
    let (mut tt, mut tp, _) = self.tok();
    let mut params: Vec<String> = vec![];
    if tt == Tok::LPar {
      let mut tl;
      (tt, tp, tl) = self.tok();
      while tt == Tok::Name {
        let p = self.text(tp, tl);
        if crate::names::reserved(&p).is_some() {
          return Err(self.e(tp, "Invalid macro parameter name."));
        }
        if params.contains(&p) {
          return Err(self.e(tp, "Invalid repeated macro parameter name."));
        }
        params.push(p);
        (tt, tp, tl) = self.tok();
        if tt == Tok::Comma {
          (tt, tp, tl) = self.tok();
        }
      }
      if tt != Tok::RPar {
        return Err(self.e(tp, "Missing right parenthesis in macro definition."));
      }
      (tt, tp, _) = self.tok();
    }
    if tt != Tok::ColonEq {
      return Err(self.e(tp, "Missing := for macro definition."));
    }
    let mut body = Tree::default();
    let r = self.read_term(&mut body)?;
    let mut root = body.root;
    if let Some(rt) = root {
      root = Some(self.expand(rt)?);
    }
    if let Some(rt) = root
      && !params.is_empty()
    {
      self.mark_params(rt, &params);
      root = Some(self.expand(rt)?);
    }
    // the macro keeps its own copy of the body
    let mut ma = Arena::default();
    let mroot = root.map(|r| copy_between(&self.a, r, &mut ma));
    self.local.macros.insert(
      name,
      Macro {
        nb_params: params.len(),
        arena: ma,
        root: mroot,
      },
    );
    if r == Tok::End {
      return Ok(r);
    }
    if r != Tok::Semicolon {
      return Err(self.e(
        self.pos.saturating_sub(1),
        "Syntax error at the end of macro definition.",
      ));
    }
    Ok(Tok::Semicolon)
  }

  fn mark_params(&mut self, i: usize, params: &[String]) {
    for c in self.a.n[i].children.clone() {
      self.mark_params(c, params);
    }
    let nm = match &self.a.n[i].kind {
      Kind::Name(n)
      | Kind::Interpolate { name: n, .. }
      | Kind::Elementary { name: n, .. }
      | Kind::SecondaryDomain { name: n }
      | Kind::XfemPlus { name: n }
      | Kind::XfemMinus { name: n } => n.clone(),
      _ => return,
    };
    let (po, rest) = strip_op(&nm);
    let (pt, base) = strip_test(rest);
    if let Some(idx) = params.iter().position(|p| p == base) {
      let orig = Box::new(self.a.n[i].kind.clone());
      self.a.n[i].kind = Kind::MacroParam {
        index: idx,
        po,
        pt,
        orig,
      };
    }
  }

  /// a copy of the body of macro n in the expression's arena
  fn copy_macro(&mut self, n: &str, body: usize) -> usize {
    let m = self
      .local
      .macros
      .get(n)
      .or_else(|| self.base.macros.get(n))
      .unwrap();
    copy_between(&m.arena, body, &mut self.a)
  }

  /// GetFEM ga_expand_macro; returns the (possibly new) node index.
  fn expand(&mut self, i: usize) -> Result<usize> {
    match self.a.n[i].kind.clone() {
      Kind::Params => {
        let ch = self.a.n[i].children.clone();
        for &c in &ch[1..] {
          let nc = self.expand(c)?;
          if nc != c {
            self.a.replace_child(i, c, nc);
          }
        }
        let f = self.a.n[i].children[0];
        let fname = match &self.a.n[f].kind {
          Kind::Name(n) => Some(n.clone()),
          _ => None,
        };
        let found = fname.and_then(|n| self.macro_of(&n).map(|m| (n, m.nb_params, m.root)));
        match found {
          None => {
            if !matches!(self.a.n[f].kind, Kind::Name(_)) {
              let nf = self.expand(f)?;
              if nf != f {
                self.a.replace_child(i, f, nf);
              }
            }
            Ok(i)
          }
          Some((n, nb_params, root)) => {
            let body = root.ok_or_else(|| self.e(self.a.n[i].pos, "Empty macro"))?;
            if nb_params == 0 {
              let nf = self.copy_macro(&n, body);
              self.a.replace_child(i, f, nf);
              Ok(i)
            } else {
              let args = self.a.n[i].children[1..].to_vec();
              if nb_params != args.len() {
                return Err(self.e(
                  self.a.n[i].pos,
                  &format!(
                    "Bad number of parameters in the use of macro '{n}'. Expected {} found {}.",
                    nb_params,
                    args.len()
                  ),
                ));
              }
              let nb = self.copy_macro(&n, body);
              let parent = self.a.n[i].parent;
              let nb = self.replace_params(nb, &args)?;
              self.a.n[nb].parent = parent;
              if let Some(p) = parent {
                self.a.replace_child(p, i, nb);
              }
              Ok(nb)
            }
          }
        }
      }
      Kind::Name(n) if self.macro_of(&n).is_some() => {
        let (nb_params, root) = self.macro_of(&n).map(|m| (m.nb_params, m.root)).unwrap();
        if nb_params != 0 {
          return Err(self.e(
            self.a.n[i].pos,
            &format!(
              "Bad number of parameters in the use of macro '{n}'. Expected {nb_params} none found."
            ),
          ));
        }
        let body = root.ok_or_else(|| self.e(self.a.n[i].pos, "Empty macro"))?;
        let nb = self.copy_macro(&n, body);
        let parent = self.a.n[i].parent;
        self.a.n[nb].parent = parent;
        if let Some(p) = parent {
          self.a.replace_child(p, i, nb);
        }
        Ok(nb)
      }
      _ => {
        for c in self.a.n[i].children.clone() {
          let nc = self.expand(c)?;
          if nc != c {
            self.a.replace_child(i, c, nc);
          }
        }
        Ok(i)
      }
    }
  }

  fn replace_params(&mut self, i: usize, args: &[usize]) -> Result<usize> {
    for c in self.a.n[i].children.clone() {
      let nc = self.replace_params(c, args)?;
      if nc != c {
        self.a.replace_child(i, c, nc);
      }
    }
    let Kind::MacroParam {
      index,
      po,
      pt,
      orig,
    } = self.a.n[i].kind.clone()
    else {
      return Ok(i);
    };
    let arg = args[index];
    let plain_name = matches!(*orig, Kind::Name(_));
    if po != 0 || pt != 0 || !plain_name {
      let an = match (&self.a.n[arg].kind, self.a.n[arg].children.is_empty()) {
        (Kind::Name(n), true) => n.clone(),
        _ => {
          return Err(self.e(self.a.n[arg].pos, "Error in macro expansion. Only variable name are allowed for macro parameter preceded by Grad_ Hess_ Test_ or Test2_ prefixes."));
        }
      };
      let mut name = an;
      if pt == 1 {
        name = format!("Test_{name}");
      }
      if pt == 2 {
        name = format!("Test2_{name}");
      }
      if po > 0 {
        name = format!("{}{name}", PREFIX_OP[po as usize - 1]);
      }
      self.a.n[i].kind = match *orig {
        Kind::Name(_) => Kind::Name(name),
        Kind::Interpolate { trans, .. } => Kind::Interpolate { name, trans },
        Kind::Elementary { trans, target, .. } => Kind::Elementary {
          name,
          trans,
          target,
        },
        Kind::SecondaryDomain { .. } => Kind::SecondaryDomain { name },
        Kind::XfemPlus { .. } => Kind::XfemPlus { name },
        Kind::XfemMinus { .. } => Kind::XfemMinus { name },
        k => k,
      };
      Ok(i)
    } else {
      Ok(self.a.copy(arg))
    }
  }

  /// '[' ... ']' explicit tensors, nested ([[a,b],[c,d]]) and flat
  /// ([a,b;c,d], with ,, and ;; for orders 3 and 4) formats.
  fn read_tensor(&mut self, t: &mut Tree, tp: usize) -> Result<()> {
    let bad = |p: &Self| {
      p.e(
        p.pos.saturating_sub(1),
        "Bad explicit vector/matrix/tensor format.",
      )
    };
    let mut sub = Tree::default();
    let mut r = self.read_term(&mut sub)?;
    self.a.add_leaf(
      t,
      Kind::Matrix {
        shape: vec![],
        marked: false,
      },
      tp,
    );
    let mnode = t.current.unwrap();
    let is_mat = |a: &Arena, s: &Tree| {
      s.root
        .is_some_and(|r| matches!(a.n[r].kind, Kind::Matrix { .. }))
    };
    if is_mat(&self.a, &sub) {
      // nested format
      let mut nb = 0usize;
      let mut mii: Vec<usize> = vec![];
      loop {
        if nb > 0 {
          sub = Tree::default();
          r = self.read_term(&mut sub)?;
        }
        if !is_mat(&self.a, &sub) || (r != Tok::Comma && r != Tok::RBracket) {
          return Err(bad(self));
        }
        let sr = sub.root.unwrap();
        let (mut shape, marked) = match &self.a.n[sr].kind {
          Kind::Matrix { shape, marked, .. } => (shape.clone(), *marked),
          _ => unreachable!(),
        };
        // a row vector [a,b] inside a nested tensor is a column
        let size: usize = shape.iter().product();
        if marked && shape.first() == Some(&1) && size != 1 {
          shape.remove(0);
        }
        if nb == 0 {
          mii = shape;
        } else if shape != mii {
          return Err(bad(self));
        }
        let ch = std::mem::take(&mut self.a.n[sr].children);
        for c in ch {
          self.a.adopt(mnode, c);
        }
        nb += 1;
        if r == Tok::RBracket {
          break;
        }
      }
      mii.push(nb);
      self.a.n[mnode].kind = Kind::Matrix {
        shape: mii,
        marked: false,
      };
    } else {
      let (mut nbc1, mut nbc2, mut nbc3, mut n1, mut n2, mut n3) =
        (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
      let mut order = 1usize;
      let mut foundcomma = false;
      let mut nb = 0;
      let (mut fn1, mut fn2, mut fn3) = (0, 0, 0);
      loop {
        if nb > 0 {
          sub = Tree::default();
          r = self.read_term(&mut sub)?;
        }
        nb += 1;
        t.current = Some(mnode);
        self.a.add_sub_tree(t, &mut sub)?;
        n1 += 1;
        n2 += 1;
        n3 += 1;
        if order < 2 {
          nbc1 += 1;
        }
        if order < 3 {
          nbc2 += 1;
        }
        if order < 4 {
          nbc3 += 1;
        }
        match r {
          Tok::Comma => {
            if !foundcomma && order > 1 {
              return Err(bad(self));
            }
            foundcomma = true;
          }
          Tok::Semicolon => {
            if n1 != nbc1 {
              return Err(bad(self));
            }
            n1 = 0;
            order = order.max(2);
          }
          Tok::DComma => {
            if n1 != nbc1 || n2 != nbc2 {
              return Err(bad(self));
            }
            n2 = 0;
            n1 = 0;
            order = order.max(3);
          }
          Tok::DSemicolon => {
            if n1 != nbc1 || n2 != nbc2 || n3 != nbc3 || order < 3 {
              return Err(bad(self));
            }
            n3 = 0;
            n2 = 0;
            n1 = 0;
            order = order.max(4);
          }
          Tok::RBracket => {
            if n1 != nbc1 || n2 != nbc2 || n3 != nbc3 || order == 3 {
              return Err(bad(self));
            }
            fn1 = nbc1;
            if order == 4 {
              fn2 = nbc2 / nbc1;
              fn3 = nbc3 / nbc2;
            } else {
              fn2 = 1;
              fn3 = 1;
            }
          }
          _ => return Err(self.e(self.pos.saturating_sub(1), "The explicit vector/matrix/tensor components should be separated by ',', ';', ',,' and ';;' and be ended by ']'.")),
        }
        if r == Tok::RBracket {
          break;
        }
      }
      let (nbc1, nbc2, nbc3) = (fn1, fn2, fn3);
      let nch = self.a.n[mnode].children.len();
      let nbl = nch / (nbc1 * nbc2 * nbc3);
      let shape = match order {
        1 => vec![nbc1],
        2 => {
          if nbc1 > 1 {
            vec![nbl, nbc1]
          } else {
            vec![nbl]
          }
        }
        3 => vec![nbl, nbc2, nbc1],
        _ => vec![nbl, nbc3, nbc2, nbc1],
      };
      // reorder the components to Fortran order
      let ch = self.a.n[mnode].children.clone();
      let mut out = Vec::with_capacity(nch);
      for i in 0..nbc1 {
        for j in 0..nbc2 {
          for k in 0..nbc3 {
            for l in 0..nbl {
              out.push(ch[i + nbc1 * (j + nbc2 * (k + nbc3 * l))]);
            }
          }
        }
      }
      self.a.n[mnode].children = out;
      self.a.n[mnode].kind = Kind::Matrix {
        shape,
        marked: true,
      };
    }
    t.current = Some(mnode);
    Ok(())
  }

  fn to_expr(&self, i: usize) -> Result<Expr> {
    let n = &self.a.n[i];
    let ch = |k: usize| self.to_expr(n.children[k]).map(Box::new);
    let kind = match &n.kind {
      Kind::Scalar(v) => ExprKind::Number(*v),
      Kind::AllIndices => ExprKind::AllIndices,
      Kind::Name(s) => ExprKind::Name(s.clone()),
      Kind::Op(t) => match n.children.len() {
        1 => ExprKind::Unary(*t, ch(0)?),
        2 => ExprKind::Binary(*t, ch(0)?, ch(1)?),
        _ => return Err(self.e(n.pos, "Missing operand.")),
      },
      Kind::Params => ExprKind::Call(
        ch(0)?,
        n.children[1..]
          .iter()
          .map(|&c| self.to_expr(c))
          .collect::<Result<_>>()?,
      ),
      Kind::Matrix { shape, .. } => ExprKind::Tensor {
        shape: shape.clone(),
        comps: n
          .children
          .iter()
          .map(|&c| self.to_expr(c))
          .collect::<Result<_>>()?,
      },
      Kind::Interpolate { name, trans } => ExprKind::Interpolate {
        name: name.clone(),
        trans: trans.clone(),
      },
      Kind::InterpolateFilter { trans } => ExprKind::InterpolateFilter {
        trans: trans.clone(),
        expr: ch(0)?,
        id: if n.children.len() > 1 {
          Some(ch(1)?)
        } else {
          None
        },
      },
      Kind::InterpolateDerivative {
        trans_der,
        name,
        trans,
      } => ExprKind::InterpolateDerivative {
        trans_der: trans_der.clone(),
        name: name.clone(),
        trans: trans.clone(),
      },
      Kind::Elementary {
        name,
        trans,
        target,
      } => ExprKind::Elementary {
        name: name.clone(),
        trans: trans.clone(),
        target: target.clone(),
      },
      Kind::SecondaryDomain { name } => ExprKind::SecondaryDomain { name: name.clone() },
      Kind::XfemPlus { name } => ExprKind::XfemPlus { name: name.clone() },
      Kind::XfemMinus { name } => ExprKind::XfemMinus { name: name.clone() },
      Kind::MacroParam { .. } => return Err(Error::Internal("unexpanded macro parameter".into())),
    };
    Ok(Expr { kind, pos: n.pos })
  }
}

/// the number of nodes of subtree i
fn count(a: &Arena, i: usize) -> usize {
  1 + a.n[i].children.iter().map(|&c| count(a, c)).sum::<usize>()
}

/// a copy of subtree i of from in to, the nodes allocated first (GetFEM
/// ga_tree::copy_node with preallocation)
fn copy_between(from: &Arena, i: usize, to: &mut Arena) -> usize {
  to.n.reserve(count(from, i));
  copy_rec(from, i, to)
}

fn copy_rec(from: &Arena, i: usize, to: &mut Arena) -> usize {
  let n = &from.n[i];
  let c = to.new_node(n.kind.clone(), n.pos);
  to.n[c].children.reserve_exact(n.children.len());
  for &ch in &n.children {
    let cc = copy_rec(from, ch, to);
    to.adopt(c, cc);
  }
  c
}

/// Parse an expression, expanding macros of `dict` and of leading
/// "Def name(args) := expr;" definitions. None for an empty expression.
pub fn parse(src: &str, dict: &MacroDict) -> Result<Option<Expr>> {
  if next_token(src.as_bytes(), &mut 0).0 == Tok::End {
    return Ok(None);
  }
  let mut p = Parser::new(src, dict);
  let t = p.read_all()?;
  t.root.map(|r| p.to_expr(r)).transpose()
}

/// S-expression of a tree, numbers as IEEE bits (for exact comparisons).
pub fn sexpr(e: &Expr) -> String {
  let list = |h: &str, v: &[&Expr]| {
    let mut s = format!("({h}");
    for x in v {
      s += " ";
      s += &sexpr(x);
    }
    s + ")"
  };
  match &e.kind {
    ExprKind::Number(v) => format!("n:{:x}", v.to_bits()),
    ExprKind::AllIndices => ":".into(),
    ExprKind::Name(n) => n.clone(),
    ExprKind::Unary(Tok::UnaryMinus, a) => list("neg", &[a]),
    ExprKind::Unary(t, a) => list(t.symbol(), &[a]),
    ExprKind::Binary(t, a, b) => list(t.symbol(), &[a, b]),
    ExprKind::Call(f, args) => list(
      "call",
      &std::iter::once(&**f).chain(args).collect::<Vec<_>>(),
    ),
    ExprKind::Tensor { shape, comps } => {
      let sh: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
      list(
        &format!("tensor[{}]", sh.join(",")),
        &comps.iter().collect::<Vec<_>>(),
      )
    }
    ExprKind::Interpolate { name, trans } => format!("(interp {name} {trans})"),
    ExprKind::InterpolateFilter { trans, expr, id } => match id {
      Some(i) => list(&format!("ifilter {trans}"), &[expr, i]),
      None => list(&format!("ifilter {trans}"), &[expr]),
    },
    ExprKind::InterpolateDerivative {
      trans_der,
      name,
      trans,
    } => {
      format!(
        "(ider {trans_der} {name} {})",
        if trans.is_empty() { "-" } else { trans }
      )
    }
    ExprKind::Elementary {
      name,
      trans,
      target,
    } => format!("(elem {name} {trans} {target})"),
    ExprKind::SecondaryDomain { name } => format!("(secondary {name})"),
    ExprKind::XfemPlus { name } => format!("(xfem+ {name})"),
    ExprKind::XfemMinus { name } => format!("(xfem- {name})"),
  }
}
