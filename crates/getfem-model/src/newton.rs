//! Newton driver over an abstract nonlinear system. It knows nothing about
//! the problem: only residual, tangent, values, and the variables that
//! partition the unknowns (SPEC 10.2).
//!
//! Convergence is a residual test in the norm |R|_D (below; free of the
//! variables' units), |R|_D <= rtol * scale, where scale is set by the
//! user or else is |R|_D at the start of the solve; rows without a weight
//! in D are tested per variable, relative to their largest residual in
//! the solve. Or the absolute test |R| <= atol, a user's override (0 by
//! default). Optionally the absolute test on the equations of given
//! variables, all required. Full Newton also accepts a step that did not
//! halve |R|_D when |R|_D is at the rounding floor, FLOOR eps
//! || |J| |x| ||_D, and not above |R|_D or the floor at the start (the
//! relative test is then out of reach: a start at the solution, a too small
//! rtol; a singular tangent instead jumps to huge values, at the floor of
//! their own size).
//!
//! Methods tried in turn, each from the start values (Newton::methods,
//! default all three; docs/design/newton-phases.md):
//! - Newton: full steps, abandoned when |R|_D grows after 10 steps
//!   (or no update gives a finite residual); never stopped while |R|_D
//!   decreases;
//! - NleqErr: Deuflhard's error-oriented global Newton, damping from the
//!   simplified Newton correction (smooth problems);
//! - NleqRes: Deuflhard's residual-based global Newton in |R|_D (non-smooth
//!   problems: contact).
//!
//! The tangent is never changed. Problems that need a regularization
//! (pseudo-time, a viscous term) carry it as a term of the model.
//!
//! D = diag(J0); rows without diagonal (multipliers) get the Schur
//! estimate sum_j J0_ij^2 / |D_jj| over the columns with one. The norm
//! |R|_D^2 = sum R_i^2 / |D_ii| (energy units in every row) measures
//! progress and convergence. Rows without either (no diagonal, no
//! coupling to a row with one) have no weight. The condensed unknowns'
//! rows count in the convergence test by their own block K_QQ (current):
//! sum_i |R_i (K_QQ^-1 R_Q)_i|, the condensation's local correction;
//! progress is measured on the condensed residual of the others.

use std::ops::Range;

use getfem_linalg::{ColMatrix, Csr, LinearSolver, MatrixKind};
use getfem_mumps::Mumps;

use crate::{Error, Result};
use getfem_linalg::vect_norm2;

/// A linearized system at the current values: the rhs -R of all unknowns
/// (the convergence tests on its norms); with condensed unknowns (the last
/// ones), the rhs on the others and the recovery data. The tangent stays
/// with the system (tangent_matrix), as GetFEM's model owns it.
pub struct Linearized {
  pub rhs: Vec<f64>,
  pub condensed: Option<CondensedStep>,
}

/// GetFEM nonlin_condensed_model_pb: solve k Δu = rhs on the primary
/// unknowns, then Δq = rhs_q - kqj Δu for the condensed ones.
pub struct CondensedStep {
  pub rhs: Vec<f64>,
  pub kqj: Csr<f64>,
  pub rhs_q: Vec<f64>,
}

pub trait NonlinearSystem {
  fn values(&self) -> Vec<f64>;
  fn set_values(&mut self, u: &[f64]) -> Result<()>;
  /// assembles at the current values: the rhs -R(u), returned, and the
  /// tangent dR/du, kept by the system (tangent_matrix)
  fn residual_and_tangent(&mut self) -> Result<Linearized>;
  /// the tangent of the last residual_and_tangent (GetFEM's model keeps
  /// its tangent matrix; the solver takes it as is)
  fn tangent_matrix(&self) -> &ColMatrix<f64>;
  /// names and ranges of the variables (the whole vector if one)
  fn variables(&self) -> Vec<(String, Range<usize>)>;
}

/// Tolerance on the equations of some variables:
/// |R restricted to their equations| <= atol.
#[derive(Clone, Debug)]
pub struct VariableTolerance {
  pub variables: Vec<String>,
  pub atol: f64,
}

impl VariableTolerance {
  pub fn new(variables: &[&str], atol: f64) -> Self {
    let variables = variables.iter().map(|v| v.to_string()).collect();
    VariableTolerance { variables, atol }
  }

  /// the variables, as a label in messages and the report
  fn label(&self) -> String {
    format!("[{}]", self.variables.join(", "))
  }
}

#[derive(Clone, Debug)]
pub struct Newton {
  /// the methods tried in turn, each from the start values (default: full
  /// Newton, then NleqErr, then NleqRes)
  pub methods: Vec<Method>,
  /// safety cap on the linear solves of all methods
  pub max_iter: usize,
  /// absolute residual tolerance on |R|, an override of the relative
  /// test (f64::INFINITY: only the variable tolerances count)
  pub atol: f64,
  /// relative residual tolerance on |R|_D, w.r.t. |R|_D at the start
  pub rtol: f64,
  /// further tests on the equations of given variables, all required
  pub variable_tolerances: Vec<VariableTolerance>,
  pub verbose: bool,
}

impl Default for Newton {
  fn default() -> Self {
    Newton {
      methods: vec![Method::Newton, Method::NleqErr, Method::NleqRes],
      max_iter: 1000,
      atol: 0.0,
      rtol: 1e-8,
      variable_tolerances: vec![],
      verbose: false,
    }
  }
}

/// A Newton method of the cascade (Newton::methods)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
  /// full Newton: full steps (halved only to keep the residual finite),
  /// abandoned when |R|_D grows after 10 steps
  Newton,
  /// Deuflhard's error-oriented global Newton (NLEQ-ERR): smooth problems
  NleqErr,
  /// Deuflhard's residual-based global Newton (NLEQ-RES) in |R|_D:
  /// non-smooth problems (contact)
  NleqRes,
}

impl std::fmt::Display for Method {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    f.write_str(match self {
      Method::Newton => "full Newton",
      Method::NleqErr => "damped Newton NLEQ-ERR",
      Method::NleqRes => "damped Newton NLEQ-RES",
    })
  }
}

/// "1 linear solve", "3 linear solves"
fn solves(n: usize) -> String {
  format!("{n} linear solve{}", if n == 1 { "" } else { "s" })
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Status {
  Converged,
  #[default]
  MaxIterations,
}

#[derive(Clone, Debug, Default)]
pub struct NewtonReport {
  pub status: Status,
  /// number of linear solves (all methods)
  pub iterations: usize,
  /// the method that converged; None at the start values or not converged
  pub method: Option<Method>,
  /// the threshold of the relative test on |R|_D: rtol * scale
  pub tolerance: f64,
  /// |R|_D at the start
  pub scale: f64,
  /// per residual evaluation: |R|
  pub residuals: Vec<f64>,
  /// per residual evaluation: |R|_D
  pub residuals_d: Vec<f64>,
  /// per residual evaluation: |R| restricted to each variable's equations
  pub residuals_per_variable: Vec<Vec<(String, f64)>>,
  /// per residual evaluation: |R| of each variable tolerance, labelled by
  /// its variables
  pub residuals_per_tolerance: Vec<Vec<(String, f64)>>,
  /// per full Newton step: relative increment of each variable
  pub increments: Vec<Vec<(String, f64)>>,
  /// explanation when not converged
  pub message: String,
  /// a remark on a converged solve (no variable tolerances): equations of some
  /// variables reduced less than rtol
  pub hint: Option<String>,
}

impl NewtonReport {
  pub fn converged(&self) -> bool {
    self.status == Status::Converged
  }
}

/// full Newton: an increase of |R|_D after this many steps abandons it
const NEWTON_FREE: usize = 10;
/// damped methods: abandoned when |R|_D is not halved in this many steps
const STALL_WINDOW: usize = 20;
/// full Newton: |R|_D at most FLOOR eps || |J| |x| ||_D is rounding
const FLOOR: f64 = 1e3;

fn primary_rhs(lin: &Linearized) -> &[f64] {
  match &lin.condensed {
    Some(c) => &c.rhs,
    None => &lin.rhs,
  }
}

/// k d = rhs on the primary unknowns, the condensed ones recovered
fn step(solver: &mut dyn LinearSolver, lin: &Linearized, k: &ColMatrix<f64>) -> Result<Vec<f64>> {
  let mut d = primary_rhs(lin).to_vec();
  solver.solve_system(k, MatrixKind::General, &mut d)?;
  recover(lin, d)
}

/// d on the primary unknowns extended by the condensed ones (rhs_q - kqj d)
fn recover(lin: &Linearized, mut d: Vec<f64>) -> Result<Vec<f64>> {
  if let Some(c) = &lin.condensed {
    let dq = crate::model::matvec_summed(&c.kqj, &d)?;
    d.extend(c.rhs_q.iter().zip(&dq).map(|(a, b)| a - b));
  }
  Ok(d)
}

/// The norm weights 1 / |D| (D: see the module) from the tangent at the
/// start, on the primary unknowns.
fn weights_of(k: &ColMatrix<f64>, vars: &[(String, Range<usize>)]) -> Vec<f64> {
  let np = k.nrows;
  let mut dg: Vec<f64> = (0..np).map(|i| k.get(i, i)).collect();
  // MPI: each rank's k is its elements' part, the diagonal summed
  getfem_mpi::sum(&mut dg);
  // without diagonal: zero, or below 1e-12 of the variable's median
  let mut zero = vec![false; np];
  for (_, rg) in vars {
    let rg = rg.start.min(np)..rg.end.min(np);
    if rg.is_empty() {
      continue;
    }
    let mut s: Vec<f64> = dg[rg.clone()].iter().map(|x| x.abs()).collect();
    s.sort_by(|a, b| a.total_cmp(b));
    let med = s[s.len() / 2];
    for i in rg {
      zero[i] = dg[i] == 0.0 || dg[i].abs() <= 1e-12 * med;
    }
  }
  // rows without diagonal: the Schur estimate sum_j k_ij^2 / |k_jj| over
  // the columns j with one (|k_jj|: no cancellation, K indefinite)
  let mut d: Vec<f64> = (0..np).map(|i| if zero[i] { 0.0 } else { dg[i] }).collect();
  let coupling = k
    .cols
    .iter()
    .enumerate()
    .take(np)
    .filter(|&(j, _)| !zero[j])
    .flat_map(|(j, col)| {
      col
        .iter()
        .filter(|e| zero[e.0])
        .map(move |&(i, v)| (j, i, v))
    });
  if getfem_mpi::size() == 1 {
    coupling.for_each(|(j, i, v)| d[i] += v * v / dg[j].abs());
  } else {
    // whole entries before squaring: every rank's (j, i, k_ij) gathered,
    // summed per entry, in the order of one rank (columns, then rows)
    let mine: Vec<f64> = coupling
      .flat_map(|(j, i, v)| [j as f64, i as f64, v])
      .collect();
    let mut whole = std::collections::BTreeMap::new();
    for t in getfem_mpi::all_gather(&mine).chunks(3) {
      *whole.entry((t[0] as usize, t[1] as usize)).or_insert(0.0) += t[2];
    }
    whole
      .into_iter()
      .for_each(|((j, i), v)| d[i] += v * v / dg[j].abs());
  }
  d.iter()
    .map(|x| if *x != 0.0 { 1.0 / x.abs() } else { 0.0 })
    .collect()
}

/// || |k| |x| ||_D, the size of the terms rounding leaves in the residual
/// (x: the unknowns of k's rows first)
fn rounding_scale(k: &ColMatrix<f64>, x: &[f64], w: &[f64]) -> f64 {
  let mut f = vec![0.0; k.nrows];
  for (col, xj) in k.cols.iter().zip(x) {
    for &(i, v) in col.iter() {
      f[i] += v.abs() * xj.abs();
    }
  }
  // MPI: each rank's k is its elements' part
  getfem_mpi::sum(&mut f);
  dnorm(&f, w)
}

/// |r|_D (w: weights_of)
fn dnorm(r: &[f64], w: &[f64]) -> f64 {
  r.iter().zip(w).map(|(r, w)| r * r * w).sum::<f64>().sqrt()
}

/// |R|_D of all unknowns: the condensed rows weighted by their own block,
/// sum_i |R_i (K_QQ^-1 R_Q)_i| (R_i^2 / |K_ii| for a diagonal K_QQ), from
/// the condensation's local correction rhs_q
fn dnorm_all(lin: &Linearized, w: &[f64]) -> f64 {
  let r = &lin.rhs;
  let mut s = r.iter().zip(w).map(|(r, w)| r * r * w).sum::<f64>();
  if let Some(c) = &lin.condensed {
    let np = r.len() - c.rhs_q.len();
    s += r[np..]
      .iter()
      .zip(&c.rhs_q)
      .map(|(a, b)| (a * b).abs())
      .sum::<f64>();
  }
  s.sqrt()
}

type Failing = Vec<(String, f64, f64)>;

/// How a method ended
enum End {
  Converged,
  /// full Newton: |R|_D at the rounding floor (the hint says so)
  AtFloor(String),
  Abandoned(String),
}
/// the tests of a solve on a residual: (all passed, the failing ones)
type Test<'a> = dyn Fn(&mut NewtonReport, &Linearized) -> (bool, Failing) + 'a;

impl Newton {
  /// Solve with the given linear solver, or MUMPS if None.
  pub fn solve(
    &self,
    sys: &mut dyn NonlinearSystem,
    solver: Option<&mut dyn LinearSolver>,
  ) -> Result<NewtonReport> {
    let mut mumps;
    let solver: &mut dyn LinearSolver = match solver {
      Some(s) => s,
      None => {
        mumps = Mumps::new();
        &mut mumps
      }
    };
    let vars = sys.variables();
    let index = |v: &String, what: &str| -> Result<usize> {
      vars
        .iter()
        .position(|x| &x.0 == v)
        .ok_or_else(|| Error::InvalidArgument(format!("{what}: {v} is not an enabled variable")))
    };
    // variable indices of each variable tolerance
    let mut members = vec![];
    for g in &self.variable_tolerances {
      let m: Result<Vec<usize>> = g
        .variables
        .iter()
        .map(|v| index(v, &format!("variable tolerance {}", g.label())))
        .collect();
      members.push(m?);
    }
    let mut rep = NewtonReport::default();
    let x0 = sys.values();
    let lin0 = sys.residual_and_tangent()?;
    let r0n = vect_norm2(&lin0.rhs);
    if !r0n.is_finite() {
      return Err(Error::Solve(
        "non-finite residual at the start values".into(),
      ));
    }
    let w = weights_of(sys.tangent_matrix(), &vars);
    // per variable: |R| on its rows without weight (None: it has none)
    let unweighted = |r: &[f64]| -> Vec<Option<f64>> {
      vars
        .iter()
        .map(|(_, rg)| {
          let mut z = rg
            .clone()
            .filter(|&i| w.get(i).is_some_and(|&x| x == 0.0))
            .peekable();
          z.peek()?;
          Some(z.map(|i| r[i] * r[i]).sum::<f64>().sqrt())
        })
        .collect()
    };
    // their largest |R| in the solve, the reference of their rtol
    let zmax = std::cell::RefCell::new(vec![0.0_f64; vars.len()]);
    rep.scale = dnorm_all(&lin0, &w);
    rep.tolerance = self.rtol * rep.scale;
    // the tests on r: (all passed, the failing ones)
    let test = |rep: &mut NewtonReport, lin: &Linearized| -> (bool, Failing) {
      let r = &lin.rhs;
      let rn = vect_norm2(r);
      let dn = dnorm_all(lin, &w);
      let parts: Vec<(String, f64)> = vars
        .iter()
        .map(|(n, rg)| (n.clone(), vect_norm2(&r[rg.clone()])))
        .collect();
      let mut failing = vec![];
      // NaN fails too; |R| <= atol passes the relative tests
      if rn.is_nan() || rn > self.atol {
        if dn.is_nan() || dn > rep.tolerance {
          failing.push(("|R|_D".to_string(), dn, rep.tolerance));
        }
        let mut zm = zmax.borrow_mut();
        for (k, z) in unweighted(r).into_iter().enumerate() {
          let Some(z) = z else { continue };
          zm[k] = zm[k].max(z);
          if z.is_nan() || z > self.rtol * zm[k] {
            failing.push((
              format!("{} (rows without weight)", vars[k].0),
              z,
              self.rtol * zm[k],
            ));
          }
        }
      }
      let mut per_tol = vec![];
      for (g, m) in self.variable_tolerances.iter().zip(&members) {
        let x = m
          .iter()
          .map(|&k| parts[k].1 * parts[k].1)
          .sum::<f64>()
          .sqrt();
        if x.is_nan() || x > g.atol {
          failing.push((g.label(), x, g.atol));
        }
        per_tol.push((g.label(), x));
      }
      if self.verbose {
        let p: Vec<String> = parts.iter().map(|(n, x)| format!("{n} {x:.1e}")).collect();
        println!("  |R| = {rn:.3e}, scaled |R| = {dn:.3e} ({})", p.join(", "));
      }
      rep.residuals.push(rn);
      rep.residuals_d.push(dn);
      rep.residuals_per_variable.push(parts);
      rep.residuals_per_tolerance.push(per_tol);
      (failing.is_empty(), failing)
    };
    let done = |mut rep: NewtonReport, iters: usize, m: Option<Method>| -> NewtonReport {
      rep.status = Status::Converged;
      rep.iterations = iters;
      rep.method = m;
      if self.variable_tolerances.is_empty() {
        rep.hint = self.hint(&rep);
      }
      rep
    };

    if test(&mut rep, &lin0).0 {
      return Ok(done(rep, 0, None));
    }
    let mut iters = 0;
    let mut tried = vec![];
    let mut lin0 = Some(lin0);
    for &m in &self.methods {
      // each method from the start values
      let lin = match lin0.take() {
        Some(l) => l,
        None => {
          sys.set_values(&x0)?;
          sys.residual_and_tangent()?
        }
      };
      let end = match m {
        Method::Newton => self.full_newton(
          sys, solver, &x0, lin, &vars, &w, &mut iters, &test, &mut rep,
        )?,
        Method::NleqErr | Method::NleqRes => self.damped(
          m, sys, solver, &x0, lin, &vars, &w, &mut iters, &test, &mut rep,
        )?,
      };
      let why = match end {
        End::Converged => {
          self.log(format!("converged by {m} in {}", solves(iters)));
          return Ok(done(rep, iters, Some(m)));
        }
        End::AtFloor(hint) => {
          self.log(format!(
            "converged by {m} at round-off level in {}",
            solves(iters)
          ));
          let mut rep = done(rep, iters, Some(m));
          rep.hint = Some(hint);
          return Ok(rep);
        }
        End::Abandoned(why) => why,
      };
      self.log(format!("{m} abandoned: {why}"));
      tried.push(format!("{m} ({why})"));
      if iters >= self.max_iter {
        break;
      }
    }
    rep.iterations = iters;
    rep.message = format!(
      "no convergence in {}: {}",
      solves(iters),
      tried.join(", then ")
    );
    Ok(rep)
  }

  /// Full Newton from lin at x0: full steps, halved only while the values
  /// or the residual are not finite; abandoned when |R|_D grows after
  /// NEWTON_FREE steps; a step that does not halve |R|_D accepted when only
  /// |R|_D fails and it is at the rounding floor, not above the start. count: the linear solves
  /// so far.
  #[allow(clippy::too_many_arguments)]
  fn full_newton(
    &self,
    sys: &mut dyn NonlinearSystem,
    solver: &mut dyn LinearSolver,
    x0: &[f64],
    lin: Linearized,
    vars: &[(String, Range<usize>)],
    w: &[f64],
    count: &mut usize,
    test: &Test,
    rep: &mut NewtonReport,
  ) -> Result<End> {
    let start = *count;
    let mut iters = start;
    let mut x = x0.to_vec();
    let mut lin = lin;
    let mut prev = dnorm(primary_rhs(&lin), w);
    let start_bound = prev.max(FLOOR * f64::EPSILON * rounding_scale(sys.tangent_matrix(), x0, w));
    let mut first = vec![0.0; vars.len()];
    let why = loop {
      if iters >= self.max_iter {
        break "the safety cap max_iter".to_string();
      }
      let dx = step(solver, &lin, sys.tangent_matrix())?;
      iters += 1;
      // the update halved while the values or the residual are not finite
      let mut alpha = 1.0;
      let next = loop {
        let xt: Vec<f64> = x.iter().zip(&dx).map(|(a, b)| a + alpha * b).collect();
        sys.set_values(&xt)?;
        if vect_norm2(&xt).is_finite()
          && let Ok(l) = sys.residual_and_tangent()
          && vect_norm2(&l.rhs).is_finite()
        {
          break Some((xt, l));
        }
        alpha /= 2.0;
        if alpha < 1e-6 {
          break None;
        }
      };
      let Some((xt, lt)) = next else {
        sys.set_values(&x)?;
        break format!("no finite update at step {iters}");
      };
      if alpha < 1.0 {
        self.log(format!(
          "step {iters}: update scaled by {alpha} to keep the residual finite"
        ));
      }
      let mut inc = vec![];
      for (kv, (name, rg)) in vars.iter().enumerate() {
        let nd = alpha * vect_norm2(&dx[rg.clone()]);
        if iters - start == 1 {
          first[kv] = nd;
        }
        let nu = vect_norm2(&x[rg.clone()]).max(first[kv]);
        inc.push((name.clone(), if nu > 0.0 { nd / nu } else { 0.0 }));
      }
      (x, lin) = (xt, lt);
      rep.increments.push(inc);
      let (conv, failing) = test(rep, &lin);
      if conv {
        *count = iters;
        return Ok(End::Converged);
      }
      let nd = dnorm(primary_rhs(&lin), w);
      // no progress, only the relative test failing: rounding?
      if nd > 0.5 * prev && failing.len() == 1 && failing[0].0 == "|R|_D" {
        let floor = FLOOR * f64::EPSILON * rounding_scale(sys.tangent_matrix(), &x, w);
        // not above the start (a singular tangent: huge values, at the
        // floor of their own size)
        if nd <= floor && nd <= start_bound {
          *count = iters;
          return Ok(End::AtFloor(format!(
            "converged at round-off level: scaled residual {nd:.3e} <= round-off bound {floor:.3e}; the relative tolerance (rtol x initial residual = {:.3e}) is below round-off",
            failing[0].2
          )));
        }
      }
      if iters - start >= NEWTON_FREE && nd > prev {
        break format!("the scaled residual increased at step {iters}");
      }
      prev = nd;
    };
    *count = iters;
    Ok(End::Abandoned(why))
  }

  /// Deuflhard's global Newton methods (Newton Methods for Nonlinear
  /// Problems, 2004) from lin at x0, the damping lambda predicted and
  /// corrected: NleqErr, error oriented (3.3.3), monitors the simplified
  /// correction J_k^-1 R(x_k + lambda dx_k) (one more solve with J_k's
  /// factors) in a norm relative to each variable's size; NleqRes, residual
  /// based (3.2.2), monitors |R|_D (no Jacobian smoothness needed: contact).
  /// lambda_0 = 0.01 (highly nonlinear: full Newton failed). Abandoned when
  /// lambda < 1e-8, a linear solve fails, |R|_D is not halved in
  /// STALL_WINDOW steps, or at max_iter.
  #[allow(clippy::too_many_arguments)]
  fn damped(
    &self,
    m: Method,
    sys: &mut dyn NonlinearSystem,
    solver: &mut dyn LinearSolver,
    x0: &[f64],
    mut lin: Linearized,
    vars: &[(String, Range<usize>)],
    w: &[f64],
    iters: &mut usize,
    test: &Test,
    rep: &mut NewtonReport,
  ) -> Result<End> {
    let err = m == Method::NleqErr;
    let mut x = x0.to_vec();
    let mut lambda = 0.01;
    // NleqErr: per variable 1 / max(|x0_v|, |dx0_v|)
    let mut wv: Vec<f64> = vec![];
    let enorm = |wv: &[f64], y: &[f64]| -> f64 {
      vars
        .iter()
        .zip(wv)
        .map(|((_, rg), s)| {
          let rg = rg.start.min(y.len())..rg.end.min(y.len());
          s * s * y[rg].iter().map(|v| v * v).sum::<f64>()
        })
        .sum::<f64>()
        .sqrt()
    };
    let mut nf = dnorm(primary_rhs(&lin), w);
    let (mut best, mut best_k) = (nf, 0);
    // NleqErr: (|dx_{k-1}|, the simplified correction at x_k, lambda_{k-1});
    // NleqRes: mu_k
    let mut prev: Option<(f64, Vec<f64>, f64)> = None;
    let mut mu: Option<f64> = None;
    let mut k = 0;
    loop {
      if *iters >= self.max_iter {
        return Ok(End::Abandoned("the safety cap max_iter".into()));
      }
      if k - best_k >= STALL_WINDOW {
        return Ok(End::Abandoned(format!(
          "the scaled residual not halved in {STALL_WINDOW} steps"
        )));
      }
      let Ok(dx) = step(solver, &lin, sys.tangent_matrix()) else {
        sys.set_values(&x)?;
        return Ok(End::Abandoned(format!(
          "the linear solve failed at step {k}"
        )));
      };
      *iters += 1;
      if err && wv.is_empty() {
        wv = vars
          .iter()
          .map(|(_, rg)| {
            let rg = rg.start.min(dx.len())..rg.end.min(dx.len());
            let s = vect_norm2(&x0[rg.clone()]).max(vect_norm2(&dx[rg]));
            if s > 0.0 { 1.0 / s } else { 0.0 }
          })
          .collect();
      }
      let ndx = enorm(&wv, &dx);
      // the prediction from the last step
      if let Some((n0, bar, l)) = &prev {
        let diff: Vec<f64> = bar.iter().zip(&dx).map(|(a, b)| a - b).collect();
        lambda = (n0 * enorm(&wv, bar) / (enorm(&wv, &diff) * ndx) * l).min(1.0);
      }
      if let Some(mu) = mu {
        lambda = mu.min(1.0);
      }
      let f0 = primary_rhs(&lin).to_vec();
      let mut raised = false;
      let accepted = loop {
        if lambda < 1e-8 {
          break Err(format!("damping below 1e-8 at step {k}"));
        }
        let xt: Vec<f64> = x.iter().zip(&dx).map(|(a, b)| a + lambda * b).collect();
        sys.set_values(&xt)?;
        let lt = match sys.residual_and_tangent() {
          Ok(l) if vect_norm2(&l.rhs).is_finite() && vect_norm2(&xt).is_finite() => l,
          _ => {
            lambda /= 2.0;
            continue;
          }
        };
        // the contraction theta and mu' (Deuflhard's [h]^-1)
        let (theta, mup, bar) = if err {
          if *iters >= self.max_iter {
            break Err("the safety cap max_iter".into());
          }
          let mut bar = primary_rhs(&lt).to_vec();
          if solver.solve(&mut bar).is_err() {
            break Err(format!("the linear solve failed at step {k}"));
          }
          let bar = recover(&lt, bar)?;
          *iters += 1;
          let e: Vec<f64> = bar
            .iter()
            .zip(&dx)
            .map(|(a, b)| a - (1.0 - lambda) * b)
            .collect();
          let theta = enorm(&wv, &bar) / ndx;
          (
            theta,
            0.5 * ndx * lambda * lambda / enorm(&wv, &e),
            Some(bar),
          )
        } else {
          let e: Vec<f64> = primary_rhs(&lt)
            .iter()
            .zip(&f0)
            .map(|(a, b)| a - (1.0 - lambda) * b)
            .collect();
          let theta = dnorm(primary_rhs(&lt), w) / nf;
          (theta, 0.5 * nf * lambda * lambda / dnorm(&e, w), None)
        };
        if self.verbose {
          println!("    damping {lambda:.1e}, contraction {theta:.2}");
        }
        // monotonicity (restricted for NleqRes)
        let limit = if err { 1.0 } else { 1.0 - lambda / 4.0 };
        if !theta.is_finite() || theta >= limit {
          lambda = mup.min(lambda / 2.0);
          continue;
        }
        let ln = mup.min(1.0);
        if ln >= 4.0 * lambda && !raised {
          raised = true;
          lambda = ln;
          continue;
        }
        break Ok((xt, lt, mup, bar));
      };
      let (xt, lt, mup, bar) = match accepted {
        Ok(a) => a,
        Err(why) => {
          sys.set_values(&x)?;
          return Ok(End::Abandoned(why));
        }
      };
      k += 1;
      let nt = dnorm(primary_rhs(&lt), w);
      match bar {
        Some(bar) => prev = Some((ndx, bar, lambda)),
        // h_{k+1} = h_k |R_{k+1}| / |R_k|
        None => mu = Some(mup * nf / nt),
      }
      (x, lin, nf) = (xt, lt, nt);
      if test(rep, &lin).0 {
        return Ok(End::Converged);
      }
      if nf < 0.5 * best {
        (best, best_k) = (nf, k);
      }
    }
  }

  fn log(&self, s: String) {
    if self.verbose {
      println!("  {s}");
    }
  }

  /// Variables whose equations carried a significant residual in this solve
  /// (at least 1e-6 of the largest |R|) but were reduced less than rtol;
  /// none without a relative test (rtol = 0).
  fn hint(&self, rep: &NewtonReport) -> Option<String> {
    if self.rtol == 0.0 {
      return None;
    }
    let gmax = rep.residuals.iter().copied().fold(0.0, f64::max);
    let last = rep.residuals_per_variable.last()?;
    let mut weak = vec![];
    for (k, (name, fin)) in last.iter().enumerate() {
      let vmax = rep
        .residuals_per_variable
        .iter()
        .map(|p| p[k].1)
        .fold(0.0, f64::max);
      if vmax >= 1e-6 * gmax && *fin > self.rtol * vmax {
        weak.push(format!("{name} ({:.1e})", fin / vmax));
      }
    }
    (!weak.is_empty()).then(|| {
      format!(
        "the global test passed, but the equations of {} were reduced less than rtol = {:.0e} relative to their largest residual in this solve; tolerances per variable can be set in Newton::variable_tolerances",
        weak.join(", "),
        self.rtol
      )
    })
  }
}
