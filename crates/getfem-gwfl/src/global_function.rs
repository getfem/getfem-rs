//! Global functions given by GWFL expressions of the point X (macros x, y,
//! z, w), as GetFEM's global_function_parser and the expression of
//! global_function_bounded: `GlobalFunction::parser`,
//! `GlobalFunction::bounded_by`; functions of the crack tip coordinates
//! (x, y, r, theta), as GetFEM's parser_xy_function:
//! `CrackTipFunction::parser`.

use getfem_fem::crack_tip_function::{CrackTipFunction, CrackTipFunctionImpl};
use getfem_fem::global_function::{EvalPoint, GlobalFunction, GlobalFunctionImpl};

use crate::defs::{Defs, Support, VarDef};
use crate::ga::{CompileCtx, GaFunction};
use crate::semantic::Tree;
use crate::semantic::{Ctx, analyse, analyse_str};
use crate::{Error, Result};

/// An expression of X (dimension n) and its first two derivatives (given,
/// derived, or none: zero).
struct PointFunction {
  n: usize,
  /// value, gradient (n), Hessian (n x n) (GetFEM global_function_parser)
  f_val: Option<GaFunction>,
  f_grad: Option<GaFunction>,
  f_hess: Option<GaFunction>,
}

/// X of dimension n, the macros x, y, z, w and more
fn defs(n: usize, more: &[(&str, &str)]) -> Result<Defs> {
  let mut d = Defs::default();
  d.vars.insert(
    "X".into(),
    VarDef {
      is_data: false,
      support: Support::Fixed,
      qdims: vec![n],
      disabled: false,
    },
  );
  for (k, m) in ["x", "y", "z", "w"].iter().enumerate().take(n) {
    d.macros.add(m, &format!("X({})", k + 1))?;
  }
  for (m, e) in more {
    d.macros.add(m, e)?;
  }
  Ok(d)
}

fn ctx(n: usize) -> Ctx {
  Ctx {
    meshdim: n,
    ref_elt_dim: n,
    option: 1,
    ignore_x: true,
    ..Ctx::default()
  }
}

impl PointFunction {
  /// val, and with ders its gradient and Hessian: the given expressions,
  /// else derived
  fn new(
    n: usize,
    more: &[(&str, &str)],
    val: &str,
    ders: Option<(Option<&str>, Option<&str>)>,
  ) -> Result<Self> {
    let d = defs(n, more)?;
    let ctx = ctx(n);
    let no_data = |_: &str| None;
    let no_trans = |_: &str| None;
    let no_der = |_: &str, _: &str, _: &str| None;
    let cx = CompileCtx {
      defs: &d,
      fixed: &no_data,
      inputs: &["X"],
      vectorized: &|_| false,
      trans: &no_trans,
      trans_der: &no_der,
      internal: &|_| false,
    };
    // a tree of the expected size (an empty tree: zero)
    let ga_function = |t: &Tree, e: &str, want: usize| -> Result<Option<GaFunction>> {
      let Some(r) = t.root else {
        return Ok(None);
      };
      let size: usize = t.n[r].sizes.iter().product();
      if size != want {
        return Err(Error::InvalidArgument(format!(
          "global function {e}: {size} components, {want} expected"
        )));
      }
      Ok(Some(GaFunction::new(t, &cx, &[n]).ok_or_else(|| {
        Error::Unsupported {
          expr: e.to_string(),
          pos: 0,
          msg: "global function: this expression is not supported".into(),
        }
      })?))
    };
    // the derivative of order k (Test_X, then Test2_X)
    let derived = |t: &Tree, k: usize| -> Result<Tree> {
      Ok(match t.root {
        None => Tree::default(),
        Some(_) => match crate::derive::derive(t, &d, "X", "", k, None)? {
          Some(dt) => analyse(dt, &d, &ctx)?.0,
          None => Tree::default(),
        },
      })
    };
    let (tv, _) = analyse_str(val, &d, &ctx)?;
    let f_val = ga_function(&tv, val, 1)?;
    let (mut f_grad, mut f_hess) = (None, None);
    if let Some((grad, hess)) = ders {
      let tg = match grad {
        Some(g) => analyse_str(g, &d, &ctx)?.0,
        None => derived(&tv, 1)?,
      };
      let th = match hess {
        Some(h) => analyse_str(h, &d, &ctx)?.0,
        None => derived(&tg, 2)?,
      };
      f_grad = ga_function(&tg, grad.unwrap_or(val), n)?;
      f_hess = ga_function(&th, hess.unwrap_or(val), n * n)?;
    }
    Ok(PointFunction {
      n,
      f_val,
      f_grad,
      f_hess,
    })
  }

  /// value (k = 0), gradient (1), Hessian (2) at x
  fn eval(&self, k: usize, x: &[f64]) -> Vec<f64> {
    let Some(p) = [&self.f_val, &self.f_grad, &self.f_hess][k] else {
      return vec![0.0; [1, self.n, self.n * self.n][k]];
    };
    let mut out = vec![];
    // compiled and size-checked: only a nonlinear operator can fail
    p.eval(&[&x[..self.n]], &mut out)
      .expect("global function evaluation");
    out
  }
}

struct Parser(PointFunction);

impl GlobalFunctionImpl for Parser {
  fn dim(&self) -> usize {
    self.0.n
  }
  fn val(&self, p: &EvalPoint) -> f64 {
    self.0.eval(0, p.x)[0]
  }
  fn grad(&self, p: &EvalPoint) -> Vec<f64> {
    self.0.eval(1, p.x)
  }
  fn hess(&self, p: &EvalPoint) -> Vec<f64> {
    self.0.eval(2, p.x)
  }
}

struct BoundedBy {
  f: GlobalFunction,
  min: Vec<f64>,
  max: Vec<f64>,
  is_in: PointFunction,
}

impl GlobalFunctionImpl for BoundedBy {
  fn dim(&self) -> usize {
    self.f.dim()
  }
  fn val(&self, p: &EvalPoint) -> f64 {
    self.f.val(p)
  }
  fn grad(&self, p: &EvalPoint) -> Vec<f64> {
    self.f.grad(p)
  }
  fn hess(&self, p: &EvalPoint) -> Vec<f64> {
    self.f.hess(p)
  }
  /// where the expression is positive
  fn is_in_support(&self, x: &[f64]) -> bool {
    self.is_in.eval(0, x)[0] > 0.0
  }
  fn bounding_box(&self) -> (Vec<f64>, Vec<f64>) {
    (self.min.clone(), self.max.clone())
  }
}

/// The global functions of GWFL expressions.
pub trait GlobalFunctionExpr: Sized {
  /// GetFEM's global_function_parser: the value, and optionally the
  /// gradient (a vector) and the Hessian (a matrix) of X in dimension dim
  /// (macros x, y, z, w); the derivatives not given are derived.
  fn parser(dim: usize, val: &str, grad: Option<&str>, hess: Option<&str>) -> Result<Self>;
  /// GetFEM's global_function_bounded with an expression: f on the box
  /// [min, max], in its support where is_in > 0.
  fn bounded_by(f: &GlobalFunction, min: &[f64], max: &[f64], is_in: &str) -> Result<Self>;
}

impl GlobalFunctionExpr for GlobalFunction {
  fn parser(dim: usize, val: &str, grad: Option<&str>, hess: Option<&str>) -> Result<Self> {
    Ok(GlobalFunction::new(Parser(PointFunction::new(
      dim,
      &[],
      val,
      Some((grad, hess)),
    )?)))
  }

  fn bounded_by(f: &GlobalFunction, min: &[f64], max: &[f64], is_in: &str) -> Result<Self> {
    let n = f.dim();
    if min.len() != n || max.len() != n {
      return Err(Error::InvalidArgument("Wrong dimensions".into()));
    }
    Ok(GlobalFunction::new(BoundedBy {
      f: f.clone(),
      min: min.to_vec(),
      max: max.to_vec(),
      is_in: PointFunction::new(n, &[], is_in, None)?,
    }))
  }
}

struct TipParser(PointFunction);

impl CrackTipFunctionImpl for TipParser {
  fn val(&self, x: f64, y: f64) -> f64 {
    self.0.eval(0, &[x, y])[0]
  }
  fn grad(&self, x: f64, y: f64) -> [f64; 2] {
    let g = self.0.eval(1, &[x, y]);
    [g[0], g[1]]
  }
  fn hess(&self, x: f64, y: f64) -> [f64; 4] {
    let h = self.0.eval(2, &[x, y]);
    [h[0], h[1], h[2], h[3]]
  }
}

/// The functions of the crack tip coordinates of GWFL expressions.
pub trait CrackTipFunctionExpr: Sized {
  /// GetFEM's parser_xy_function (Python GlobalFunction('parser')): the
  /// value, and optionally the gradient (a 2-vector) and the Hessian (2 x
  /// 2, or 4 values column-major) of x, y, r = sqrt(x*x+y*y), theta =
  /// atan2(y, x); the derivatives not given are derived.
  fn parser(val: &str, grad: Option<&str>, hess: Option<&str>) -> Result<Self>;
}

impl CrackTipFunctionExpr for CrackTipFunction {
  fn parser(val: &str, grad: Option<&str>, hess: Option<&str>) -> Result<Self> {
    let more = [("r", "sqrt(x*x+y*y)"), ("theta", "atan2(y,x)")];
    Ok(CrackTipFunction::new(TipParser(PointFunction::new(
      2,
      &more,
      val,
      Some((grad, hess)),
    )?)))
  }
}
