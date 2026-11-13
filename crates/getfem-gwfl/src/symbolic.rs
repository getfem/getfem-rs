//! Symbolic derivatives of scalar function bodies (expressions of t and u),
//! used when a function is defined so that its derivatives are functions
//! too (applied elementwise), and a printer back to GWFL source.

use crate::lexer::Tok;
use crate::parser::{Expr, ExprKind};

/// Name of the function computing the k-th partial derivative of f
/// (k = 1: w.r.t. t, 2: w.r.t. u), for derivatives given as expressions.
pub fn der_name(f: &str, k: usize) -> String {
  format!("DER{k}_{f}")
}

fn num(v: f64, pos: usize) -> Expr {
  Expr {
    kind: ExprKind::Number(v),
    pos,
  }
}

fn bin(op: Tok, a: Expr, b: Expr) -> Expr {
  let pos = a.pos;
  Expr {
    kind: ExprKind::Binary(op, Box::new(a), Box::new(b)),
    pos,
  }
}

fn call(f: &str, args: Vec<Expr>, pos: usize) -> Expr {
  let name = Expr {
    kind: ExprKind::Name(f.to_string()),
    pos,
  };
  Expr {
    kind: ExprKind::Call(Box::new(name), args),
    pos,
  }
}

/// d e / d var; None is zero. `fname(f, k)` names the derivative of
/// function f w.r.t. its k-th argument (None: not differentiable).
pub fn diff(
  e: &Expr,
  var: &str,
  fname: &dyn Fn(&str, usize) -> Option<String>,
) -> Result<Option<Expr>, String> {
  let p = e.pos;
  Ok(match &e.kind {
    ExprKind::Number(_) | ExprKind::AllIndices => None,
    ExprKind::Name(n) => (n == var).then(|| num(1.0, p)),
    ExprKind::Unary(op, a) => diff(a, var, fname)?.map(|d| Expr {
      kind: ExprKind::Unary(*op, Box::new(d)),
      pos: p,
    }),
    ExprKind::Binary(op, a, b) => {
      let (da, db) = (diff(a, var, fname)?, diff(b, var, fname)?);
      let (a, b) = ((**a).clone(), (**b).clone());
      match op {
        Tok::Plus | Tok::Minus => match (da, db) {
          (None, None) => None,
          (Some(x), None) => Some(x),
          (None, Some(y)) => Some(if *op == Tok::Plus {
            y
          } else {
            Expr {
              kind: ExprKind::Unary(Tok::UnaryMinus, Box::new(y)),
              pos: p,
            }
          }),
          (Some(x), Some(y)) => Some(bin(*op, x, y)),
        },
        Tok::Mult | Tok::Dot | Tok::Colon | Tok::TMult | Tok::DotMult => match (da, db) {
          (None, None) => None,
          (Some(x), None) => Some(bin(*op, x, b)),
          (None, Some(y)) => Some(bin(*op, a, y)),
          (Some(x), Some(y)) => Some(bin(Tok::Plus, bin(*op, x, b), bin(*op, a, y))),
        },
        Tok::Div | Tok::DotDiv => {
          let (mul, div) = if *op == Tok::Div {
            (Tok::Mult, Tok::Div)
          } else {
            (Tok::DotMult, Tok::DotDiv)
          };
          let t1 = da.map(|x| bin(div, x, b.clone()));
          let t2 = db.map(|y| bin(mul, bin(div, a.clone(), bin(mul, b.clone(), b.clone())), y));
          match (t1, t2) {
            (None, None) => None,
            (Some(x), None) => Some(x),
            (None, Some(y)) => Some(Expr {
              kind: ExprKind::Unary(Tok::UnaryMinus, Box::new(y)),
              pos: p,
            }),
            (Some(x), Some(y)) => Some(bin(Tok::Minus, x, y)),
          }
        }
        _ => return Err(format!("cannot differentiate operator {op:?}")),
      }
    }
    ExprKind::Call(f, args) => {
      let fname_s = match &f.kind {
        ExprKind::Name(n) => n.clone(),
        _ => {
          // component access: linear in the tensor
          return Ok(diff(f, var, fname)?.map(|d| Expr {
            kind: ExprKind::Call(Box::new(d), args.clone()),
            pos: p,
          }));
        }
      };
      if ["Id", "qdim", "qdims", "meshdim"].contains(&fname_s.as_str()) {
        return Ok(None);
      }
      let mut sum: Option<Expr> = None;
      for (k, a) in args.iter().enumerate() {
        let Some(da) = diff(a, var, fname)? else {
          continue;
        };
        let g =
          fname(&fname_s, k + 1).ok_or_else(|| format!("{fname_s} cannot be differentiated"))?;
        let term = bin(Tok::DotMult, call(&g, args.clone(), p), da);
        sum = Some(match sum {
          None => term,
          Some(s) => bin(Tok::Plus, s, term),
        });
      }
      sum
    }
    ExprKind::Tensor { shape, comps } => {
      let ds: Vec<Option<Expr>> = comps
        .iter()
        .map(|c| diff(c, var, fname))
        .collect::<Result<_, _>>()?;
      if ds.iter().all(|d| d.is_none()) {
        None
      } else {
        let comps = ds
          .into_iter()
          .map(|d| d.unwrap_or_else(|| num(0.0, p)))
          .collect();
        Some(Expr {
          kind: ExprKind::Tensor {
            shape: shape.clone(),
            comps,
          },
          pos: p,
        })
      }
    }
    _ => return Err("cannot differentiate this construct in a scalar function".into()),
  })
}

/// GWFL source of an expression, fully parenthesized.
pub fn to_src(e: &Expr) -> String {
  match &e.kind {
    ExprKind::Number(v) => {
      let s = format!("{v:e}");
      if *v < 0.0 { format!("({s})") } else { s }
    }
    ExprKind::AllIndices => ":".into(),
    ExprKind::Name(n) => n.clone(),
    ExprKind::Unary(Tok::UnaryMinus, a) => format!("(-{})", to_src(a)),
    ExprKind::Unary(Tok::Quote, a) => format!("({})'", to_src(a)),
    ExprKind::Unary(t, a) => format!("{}({})", t.symbol(), to_src(a)),
    ExprKind::Binary(t, a, b) => format!("({}{}{})", to_src(a), t.symbol(), to_src(b)),
    ExprKind::Call(f, args) => {
      let a: Vec<String> = args.iter().map(to_src).collect();
      format!("{}({})", to_src(f), a.join(", "))
    }
    ExprKind::Tensor { shape, comps } => {
      // flat format by columns: Fortran order back to rows
      let c: Vec<String> = comps.iter().map(to_src).collect();
      match shape.len() {
        0 | 1 => format!("[{}]", c.join(";")),
        2 => {
          let (m, n) = (shape[0], shape[1]);
          let rows: Vec<String> = (0..m)
            .map(|i| {
              (0..n)
                .map(|j| c[i + m * j].clone())
                .collect::<Vec<_>>()
                .join(",")
            })
            .collect();
          format!("[{}]", rows.join(";"))
        }
        _ => format!(
          "Reshape([{}], {})",
          c.join(";"),
          shape
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(", ")
        ),
      }
    }
    ExprKind::Interpolate { name, trans } => format!("Interpolate({name}, {trans})"),
    ExprKind::InterpolateFilter { trans, expr, id } => match id {
      Some(i) => format!(
        "Interpolate_filter({trans}, {}, {})",
        to_src(expr),
        to_src(i)
      ),
      None => format!("Interpolate_filter({trans}, {})", to_src(expr)),
    },
    ExprKind::InterpolateDerivative {
      trans_der,
      name,
      trans,
    } => format!("Interpolate_derivative({trans_der}, {name}, {trans})"),
    ExprKind::Elementary {
      name,
      trans,
      target,
    } => format!("Elementary_transformation({name}, {trans}, {target})"),
    ExprKind::SecondaryDomain { name } => format!("Secondary_domain({name})"),
    ExprKind::XfemPlus { name } => format!("Xfem_plus({name})"),
    ExprKind::XfemMinus { name } => format!("Xfem_minus({name})"),
  }
}
