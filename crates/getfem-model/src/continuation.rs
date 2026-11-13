//! Moore-Penrose continuation (GetFEM getfem_continuation.h,
//! cont_struct_getfem_model) in a scalar data of a model: predictor,
//! Newton corrector on the bordered system by block elimination, step
//! control, limit points, smooth and non-smooth bifurcations. A port of
//! GetFEM's algorithm with its tests and metrics; the random borders of the
//! bifurcation test come from glibc's rand() with a fixed seed (GetFEM
//! seeds with the time). The point (x, gamma) is the model's, x all its
//! unknowns; the user holds the tangent (tx, tgamma), the step length and
//! the singularity indicators between steps.

use std::ops::Range;

use getfem_linalg::{ColMatrix, LinearSolver, MatrixKind, Rand};
use getfem_mumps::Mumps;

use crate::newton::NonlinearSystem;
use crate::{BuildVersion, Error, Model, Result};
use getfem_linalg::{vect_norm2, vect_sp};

/// GetFEM's tau_bp_init, diffeps
const TAU_BP_INIT: f64 = 1e6;
/// F_gamma by the forward difference of this step (gamma in [0, 1])
const DIFFEPS: f64 = 1e-8;

/// Variables of the continuation metric and their scale (a typical
/// magnitude of their values)
#[derive(Clone, Debug)]
pub struct VariableScale {
  pub variables: Vec<String>,
  pub scale: f64,
}

impl VariableScale {
  pub fn new(variables: &[&str], scale: f64) -> Self {
    let variables = variables.iter().map(|v| v.to_string()).collect();
    VariableScale { variables, scale }
  }
}

/// Options of a Moore-Penrose continuation step (GetFEM's
/// cont_struct_getfem_model, its defaults); no state between steps but
/// the tangent, the step length and the singularity indicators, held by
/// the user. The parameter is dimensionless, from 0 to 1 (a load factor;
/// parametrised_data for a load with units). The metric (tangent, step
/// length, max_diff, min_cos) is dimensionless:
///   |(x, gamma)|^2 = sum over variables of |x_v|^2 / (n_v scale_v^2) + gamma^2,
/// x_v the values of a group of variables (n_v of them); the other
/// variables do not count (GetFEM: scfac |x|^2 or scfac |x_var|^2 with
/// variable_name, its cosine unweighted).
#[derive(Clone, Debug)]
pub struct NumericalContinuation {
  /// the scalar data of the parameter
  pub parameter: String,
  /// (init, final, current): current = (1 - gamma) init + gamma final
  pub parametrised_data: Option<(String, String, String)>,
  /// the variables of the metric and their scales
  pub variables: Vec<VariableScale>,
  /// the end of the path: the predictor stops there, a point reached
  /// beyond it is brought back to it (Newton at gamma_max from the
  /// tangent's point); a step from gamma >= gamma_max fails
  pub gamma_max: f64,
  pub h_init: f64,
  pub h_max: f64,
  pub h_min: f64,
  pub h_inc: f64,
  pub h_dec: f64,
  pub max_iter: usize,
  pub thr_iter: usize,
  /// |F| absolute, in the units of the equations (GetFEM's; max_diff is
  /// the dimensionless test)
  pub max_res: f64,
  pub max_diff: f64,
  pub min_cos: f64,
  /// 0 none, 1 limit points, 2 also bifurcations
  pub singularities: u8,
  pub non_smooth: bool,
  pub delta_max: f64,
  pub delta_min: f64,
  pub thr_var: f64,
  pub nb_dir: usize,
  pub nb_span: usize,
  /// seed of the borders (GetFEM: the time)
  pub seed: u32,
  pub noisy: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContinuationStatus {
  /// the first call: the tangent at the model's point
  Initialized,
  /// a new point
  Converged,
  /// no new point (GetFEM: h = 0); the model back at the start point
  Failed,
}

/// The outcome of a step
#[derive(Clone, Debug)]
pub struct ContinuationReport {
  pub status: ContinuationStatus,
  /// the step length used (GetFEM h0; 0 if none)
  pub h0: f64,
  /// "", "limit point", "smooth bifurcation point", "non-smooth
  /// bifurcation point"
  pub sing_label: String,
  /// the singular point and the tangents of its branches
  pub x_sing: Vec<f64>,
  pub gamma_sing: f64,
  pub tx_sing: Vec<Vec<f64>>,
  pub tgamma_sing: Vec<f64>,
  /// the singularity indicators (tau_lp, tau_bp_1, tau_bp_2: their sign
  /// changes mark limit and bifurcation points), for the next step
  pub sing_indicators: [f64; 3],
  /// (alpha, tau_bp) along the convex combination of a non-smooth test
  pub tau_bp_graph: Vec<(f64, f64)>,
}

/// the working state of one step
struct Cont {
  o: NumericalContinuation,
  tau_lp: f64,
  tau_bp_1: f64,
  tau_bp_2: f64,
  tau_bp_graph: Vec<(f64, f64)>,
  sing_label: String,
  x_sing: Vec<f64>,
  gamma_sing: f64,
  tx_sing: Vec<Vec<f64>>,
  tgamma_sing: Vec<f64>,
  tx_predict: Vec<Vec<f64>>,
  tgamma_predict: Vec<f64>,
  x_next: Vec<f64>,
  gamma_next: f64,
  bb_x: Vec<f64>,
  cc_x: Vec<f64>,
  bb_gamma: f64,
  cc_gamma: f64,
  dd: f64,
  rand: Rand,
  /// the metric: ranges of a group of variables, 1 / (n scale^2)
  metric: Vec<(Vec<Range<usize>>, f64)>,
  solver: Mumps,
}

// [v, a] helpers (GetFEM's copy, scale, scaled_add)
fn axpy(v1: &[f64], v2: &[f64], c2: f64) -> Vec<f64> {
  v1.iter().zip(v2).map(|(a, b)| a + c2 * b).collect()
}
fn lin(v1: &[f64], c1: f64, v2: &[f64], c2: f64) -> Vec<f64> {
  v1.iter().zip(v2).map(|(a, b)| c1 * a + c2 * b).collect()
}
impl NumericalContinuation {
  pub fn new(parameter: &str, variables: Vec<VariableScale>) -> Self {
    NumericalContinuation {
      parameter: parameter.to_string(),
      parametrised_data: None,
      variables,
      gamma_max: f64::INFINITY,
      h_init: 1e-2,
      h_max: 1e-1,
      h_min: 1e-5,
      h_inc: 1.3,
      h_dec: 0.5,
      max_iter: 10,
      thr_iter: 4,
      max_res: 1e-6,
      max_diff: 1e-6,
      min_cos: 0.9,
      singularities: 0,
      non_smooth: false,
      delta_max: 0.005,
      delta_min: 0.00012,
      thr_var: 0.02,
      nb_dir: 40,
      nb_span: 1,
      seed: 7689,
      noisy: 0,
    }
  }

  /// One step from the model's point (its state and the parameter) along
  /// the tangent (tx, tgamma) with the step length h, both updated; the
  /// model holds the new point (the start point if failed). An empty tx:
  /// the tangent at the model's point, oriented by the sign of tgamma, and
  /// h = h_init (GetFEM init_Moore_Penrose_continuation). sing_indicators: the
  /// report's of the previous step (None: no history).
  pub fn step(
    &self,
    md: &mut Model,
    tx: &mut Vec<f64>,
    tgamma: &mut f64,
    h: &mut f64,
    sing_indicators: Option<[f64; 3]>,
  ) -> Result<ContinuationReport> {
    let mut x = md.values_of_variables(true);
    let mut gamma = md
      .variable(&self.parameter)
      .and_then(|v| v.first().copied())
      .ok_or_else(|| Error::InvalidArgument(format!("continuation: no data {}", self.parameter)))?;
    let [tau_lp, tau_bp_1, tau_bp_2] =
      sing_indicators.unwrap_or([*tgamma, TAU_BP_INIT, TAU_BP_INIT]);
    let mut c = Cont {
      o: self.clone(),
      tau_lp,
      tau_bp_1,
      tau_bp_2,
      tau_bp_graph: vec![],
      sing_label: String::new(),
      x_sing: vec![],
      gamma_sing: 0.0,
      tx_sing: vec![],
      tgamma_sing: vec![],
      tx_predict: vec![],
      tgamma_predict: vec![],
      x_next: vec![],
      gamma_next: 0.0,
      bb_x: vec![],
      cc_x: vec![],
      bb_gamma: 0.0,
      cc_gamma: 0.0,
      dd: 0.0,
      rand: Rand::new(1),
      metric: vec![],
      solver: Mumps::new(),
    };
    let (status, h0) = if tx.is_empty() {
      c.init_moore_penrose_continuation(md, &x, gamma, tx, tgamma, h)?;
      (ContinuationStatus::Initialized, 0.0)
    } else if gamma >= self.gamma_max {
      println!(
        "continuation: {} = {gamma} at gamma_max {}, no step",
        self.parameter, self.gamma_max
      );
      (ContinuationStatus::Failed, 0.0)
    } else {
      let h0 = c.moore_penrose_continuation(md, &mut x, &mut gamma, tx, tgamma, h)?;
      let status = if *h == 0.0 {
        ContinuationStatus::Failed
      } else {
        ContinuationStatus::Converged
      };
      (status, h0)
    };
    Ok(ContinuationReport {
      status,
      h0,
      sing_label: c.sing_label,
      x_sing: c.x_sing,
      gamma_sing: c.gamma_sing,
      tx_sing: c.tx_sing,
      tgamma_sing: c.tgamma_sing,
      sing_indicators: [c.tau_lp, c.tau_bp_1, c.tau_bp_2],
      tau_bp_graph: c.tau_bp_graph,
    })
  }
}

impl Cont {
  // ---- the model (GetFEM cont_struct_getfem_model) ----

  fn set_variables(&self, md: &mut Model, x: &[f64], gamma: f64) -> Result<()> {
    let ws = md.workspace_mut();
    ws.set_variable(&self.o.parameter, &[gamma])?;
    if let Some((i, f, c)) = &self.o.parametrised_data {
      let val = |n: &str| {
        ws.variable(n)
          .map(|v| v.to_vec())
          .ok_or_else(|| Error::InvalidArgument(format!("continuation: no data {n}")))
      };
      let cur = lin(&val(i)?, 1.0 - gamma, &val(f)?, gamma);
      ws.set_variable(c, &cur)?;
    }
    md.set_values_of_variables(x, true)
  }

  /// F(x, gamma): the residual (-rhs)
  fn f(&self, md: &mut Model, x: &[f64], gamma: f64) -> Result<Vec<f64>> {
    self.set_variables(md, x, gamma)?;
    md.assembly(BuildVersion::Rhs)?;
    Ok(md.rhs().iter().map(|v| -v).collect())
  }

  /// (F(x, gamma + eps) - f0) / eps
  fn f_gamma_with(&self, md: &mut Model, x: &[f64], gamma: f64, f0: &[f64]) -> Result<Vec<f64>> {
    let g = self.f(md, x, gamma + DIFFEPS)?;
    Ok(g.iter().zip(f0).map(|(a, b)| (a - b) / DIFFEPS).collect())
  }

  fn f_gamma(&self, md: &mut Model, x: &[f64], gamma: f64) -> Result<Vec<f64>> {
    let f0 = self.f(md, x, gamma)?;
    self.f_gamma_with(md, x, gamma, &f0)
  }

  /// F_x(x, gamma): a copy of the tangent (GetFEM's F_x copies the
  /// model's tangent matrix too)
  fn f_x(&self, md: &mut Model, x: &[f64], gamma: f64) -> Result<ColMatrix<f64>> {
    self.set_variables(md, x, gamma)?;
    let lin = md.residual_and_tangent()?;
    if lin.condensed.is_some() {
      return Err(Error::InvalidArgument(
        "continuation with internal variables: singularity detection not available".into(),
      ));
    }
    Ok(md.tangent_matrix().clone())
  }

  /// At (x, gamma), A = F_x(x, gamma): A^-1 f (f = F(x, gamma), if given)
  /// and A^-1 F_gamma, one factorization; A itself when there are no
  /// condensed unknowns. With condensed unknowns q (GetFEM's continuation
  /// leaves them at their start, DEVIATIONS) the solves go through the
  /// condensation: A^-1 f = (K^-1 rhs, rq - kqj (K^-1 rhs)) and F_gamma by
  /// the forward difference of the condensed rhs and of rq.
  #[allow(clippy::type_complexity)]
  fn grad_solves(
    &mut self,
    md: &mut Model,
    x: &[f64],
    gamma: f64,
    f: Option<&[f64]>,
  ) -> Result<(Option<Vec<f64>>, Vec<f64>, Option<ColMatrix<f64>>)> {
    if md.workspace().nb_internal_dof() == 0 {
      let g = match f {
        Some(f) => self.f_gamma_with(md, x, gamma, f)?,
        None => self.f_gamma(md, x, gamma)?,
      };
      let a = self.f_x(md, x, gamma)?;
      return Ok(match f {
        Some(f) => {
          let (dx, y) = self.solve2(&a, f, &g)?;
          (Some(dx), y, Some(a))
        }
        None => (None, self.solve1(&a, &g)?, Some(a)),
      });
    }
    self.set_variables(md, x, gamma + DIFFEPS)?;
    let l1 = md.residual_and_tangent()?;
    self.set_variables(md, x, gamma)?;
    let l0 = md.residual_and_tangent()?;
    let (c0, c1) = (
      l0.condensed.as_ref().unwrap(),
      l1.condensed.as_ref().unwrap(),
    );
    let fd = |a: &[f64], b: &[f64]| -> Vec<f64> {
      a.iter().zip(b).map(|(a, b)| (a - b) / DIFFEPS).collect()
    };
    // F = R = -rhs
    let neg = |v: &[f64]| -> Vec<f64> { v.iter().map(|x| -x).collect() };
    let (r0, r1, rq0, rq1) = (neg(&c0.rhs), neg(&c1.rhs), neg(&c0.rhs_q), neg(&c1.rhs_q));
    let (gp, gq) = (fd(&r1, &r0), fd(&rq1, &rq0));
    let full = |p: Vec<f64>, q: &[f64]| -> Result<Vec<f64>> {
      let kp = crate::model::matvec_summed(&c0.kqj, &p)?;
      let mut v = p;
      v.extend(q.iter().zip(&kp).map(|(a, b)| a - b));
      Ok(v)
    };
    Ok(match f {
      Some(_) => {
        // the tangent of the last assembly, l0's
        let (dp, yp) = self.solve2(md.tangent_matrix(), &r0, &gp)?;
        (Some(full(dp, &rq0)?), full(yp, &gq)?, None)
      }
      None => {
        let yp = self.solve1(md.tangent_matrix(), &gp)?;
        (None, full(yp, &gq)?, None)
      }
    })
  }

  /// A^-1 [l1, l2], one factorization
  fn solve2(&mut self, a: &ColMatrix<f64>, l1: &[f64], l2: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
    self.solver.analyze(a, MatrixKind::General)?;
    self.solver.factorize(a)?;
    let (mut g1, mut g2) = (l1.to_vec(), l2.to_vec());
    self.solver.solve(&mut g1)?;
    self.solver.solve(&mut g2)?;
    Ok((g1, g2))
  }

  fn solve1(&mut self, a: &ColMatrix<f64>, l: &[f64]) -> Result<Vec<f64>> {
    self.solver.analyze(a, MatrixKind::General)?;
    self.solver.factorize(a)?;
    let mut g = l.to_vec();
    self.solver.solve(&mut g)?;
    Ok(g)
  }

  fn w_sp(&self, v1: &[f64], v2: &[f64]) -> f64 {
    self
      .metric
      .iter()
      .map(|(rs, w)| {
        w * rs
          .iter()
          .map(|r| vect_sp(&v1[r.clone()], &v2[r.clone()]))
          .sum::<f64>()
      })
      .sum()
  }

  fn w_norm(&self, v: &[f64], w: f64) -> f64 {
    (self.w_sp(v, v) + w * w).sqrt()
  }

  /// in the metric (GetFEM: unweighted)
  fn cosang(&self, v1: &[f64], v2: &[f64], w1: f64, w2: f64) -> f64 {
    let no = self.w_norm(v1, w1) * self.w_norm(v2, w2);
    if no == 0.0 {
      0.0
    } else {
      (self.w_sp(v1, v2) + w1 * w2) / no
    }
  }

  /// [v, a] /= w_norm
  fn normalize(&self, v: &mut [f64], a: &mut f64) {
    let c = 1.0 / self.w_norm(v, *a);
    v.iter_mut().for_each(|x| *x *= c);
    *a *= c;
  }

  fn metric_of(&mut self, md: &Model) -> Result<()> {
    if self.o.variables.is_empty() {
      return Err(Error::InvalidArgument(
        "continuation: no variables for the metric".into(),
      ));
    }
    self.metric.clear();
    for vs in &self.o.variables {
      let rs = vs
        .variables
        .iter()
        .map(|v| {
          md.workspace()
            .interval_of_variable(v)
            .ok_or_else(|| Error::InvalidArgument(format!("continuation: no variable {v}")))
        })
        .collect::<Result<Vec<_>>>()?;
      let n: usize = rs.iter().map(|r| r.len()).sum();
      if n == 0 || vs.scale.is_nan() || vs.scale <= 0.0 {
        return Err(Error::InvalidArgument(format!(
          "continuation: {:?}: {n} unknowns, scale {}",
          vs.variables, vs.scale
        )));
      }
      self
        .metric
        .push((rs, 1.0 / (n as f64 * vs.scale * vs.scale)));
    }
    Ok(())
  }

  // ---- GetFEM virtual_cont_struct ----

  /// a unit tangent at (x, gamma), acute to the incoming (tx, tgamma)
  fn compute_tangent(
    &mut self,
    md: &mut Model,
    x: &[f64],
    gamma: f64,
    tx: &mut Vec<f64>,
    tgamma: &mut f64,
  ) -> Result<()> {
    let (_, y, a) = self.grad_solves(md, x, gamma, None)?;
    *tgamma = 1.0 / (*tgamma - self.w_sp(tx, &y));
    *tx = y.iter().map(|v| -*tgamma * v).collect();
    self.normalize(tx, tgamma);
    if let Some(a) = a
      && self.o.noisy > 0
    {
      let g = self.f_gamma(md, x, gamma)?;
      let r = vect_norm2(&axpy(&crate::model::mult_summed(&a, tx), &g, *tgamma));
      if r > 1e-10 {
        println!("warning: tangent computed with the residual {r:e}");
      }
    }
    Ok(())
  }

  /// GetFEM newton_corr: corrections of (X, Gamma) and (tX, tGamma);
  /// (converged, iterations index)
  #[allow(clippy::too_many_arguments)]
  fn newton_corr(
    &mut self,
    md: &mut Model,
    x: &mut Vec<f64>,
    gamma: &mut f64,
    tx: &mut Vec<f64>,
    tgamma: &mut f64,
    tx_old: &[f64],
    tgamma_old: f64,
  ) -> Result<(bool, usize)> {
    let mut f = self.f(md, x, *gamma)?;
    let mut res = 0.0;
    let mut it = 0;
    while it < self.o.max_iter && res < 1e8 {
      let (dx, y, _) = self.grad_solves(md, x, *gamma, Some(&f))?;
      let mut dx = dx.unwrap();
      // the constraint (tx, tgamma).(dx, dgamma) = 0 in the metric (GetFEM:
      // plain over all unknowns)
      let dgamma = self.w_sp(tx, &dx) / (self.w_sp(tx, &y) - *tgamma);
      if dgamma.is_nan() {
        return Ok((false, it));
      }
      dx = axpy(&dx, &y, -dgamma);
      *x = axpy(x, &dx, -1.0);
      *gamma -= dgamma;
      f = self.f(md, x, *gamma)?;
      res = vect_norm2(&f);
      *tgamma = 1.0 / (*tgamma - self.w_sp(tx, &y));
      *tx = y.iter().map(|v| -*tgamma * v).collect();
      self.normalize(tx, tgamma);
      let diff = self.w_norm(&dx, dgamma);
      if self.o.noisy > 1 {
        println!(
          " correction {it:3}: gamma {:.6} residual {res:.3e} difference {diff:.3e} cosang {:.6}",
          *gamma,
          self.cosang(tx, tx_old, *tgamma, tgamma_old)
        );
      }
      if res <= self.o.max_res && diff <= self.o.max_diff {
        self.compute_tangent(md, x, *gamma, tx, tgamma)?;
        return Ok((true, it));
      }
      it += 1;
    }
    Ok((false, it))
  }

  fn init_moore_penrose_continuation(
    &mut self,
    md: &mut Model,
    x: &[f64],
    gamma: f64,
    tx: &mut Vec<f64>,
    tgamma: &mut f64,
    h: &mut f64,
  ) -> Result<()> {
    self.metric_of(md)?;
    *tx = vec![0.0; x.len()];
    *tgamma = if *tgamma >= 0.0 { 1.0 } else { -1.0 };
    self.compute_tangent(md, x, gamma, tx, tgamma)?;
    *h = self.o.h_init;
    if self.o.singularities > 0 {
      self.init_test_functions(md, x, gamma, tx, *tgamma)?;
    }
    self.set_variables(md, x, gamma)
  }

  /// One step; h is the next step length (0: failed). Returns the length
  /// used (GetFEM h0). The model holds the new point.
  fn moore_penrose_continuation(
    &mut self,
    md: &mut Model,
    x: &mut Vec<f64>,
    gamma: &mut f64,
    tx: &mut Vec<f64>,
    tgamma: &mut f64,
    h: &mut f64,
  ) -> Result<f64> {
    self.metric_of(md)?;
    let (mut new_point, mut tangent_switched) = (false, false);
    let mut step_dec = 0;
    let (tx0, tgamma0) = (tx.clone(), *tgamma);
    let (mut xx, mut gg, mut txx, mut tgg): (Vec<f64>, f64, Vec<f64>, f64);
    let mut h0;
    self.tau_bp_graph.clear();
    self.clear_sing_data();
    loop {
      // the predictor at most to gamma_max
      let capped = *tgamma > 0.0 && *gamma + *h * *tgamma > self.o.gamma_max;
      h0 = if capped {
        (self.o.gamma_max - *gamma) / *tgamma
      } else {
        *h
      };
      xx = axpy(x, tx, h0);
      gg = *gamma + h0 * *tgamma;
      if self.o.noisy > 1 {
        println!(
          " prediction: gamma {gg} (h = {h0:.3e}, tgamma = {:.3e})",
          *tgamma
        );
      }
      txx = tx.clone();
      tgg = *tgamma;
      let (converged, it) =
        self.newton_corr(md, &mut xx, &mut gg, &mut txx, &mut tgg, tx, *tgamma)?;
      let cang = if converged {
        self.cosang(&txx, tx, tgg, *tgamma)
      } else {
        0.0
      };
      let mut accepted = converged && cang >= self.o.min_cos;
      // to gamma_max if the predictor stopped there (the corrector lands
      // short of it on a bending path) or the point is beyond it
      if accepted
        && (capped || gg > self.o.gamma_max)
        && !self.back_to_gamma_max(md, &mut xx, &mut gg, &mut txx, &mut tgg)?
      {
        accepted = gg <= self.o.gamma_max;
      }
      if accepted {
        new_point = true;
        if self.o.singularities > 0 {
          if self.test_limit_point(tgg) {
            self.sing_label = "limit point".into();
            if self.o.noisy > 0 {
              println!("limit point detected");
            }
          }
          if self.o.singularities > 1 {
            if !tangent_switched {
              if self.test_smooth_bifurcation(md, &xx, gg, &txx, tgg)? {
                self.sing_label = "smooth bifurcation point".into();
                if self.o.noisy > 0 {
                  println!("smooth bifurcation point detected");
                }
                self.treat_smooth_bif_point(md, &xx, gg, &txx, tgg, *h)?;
              }
            } else if self
              .test_nonsmooth_bifurcation(md, x, *gamma, &tx0, tgamma0, &xx, gg, &txx, tgg)?
            {
              self.sing_label = "non-smooth bifurcation point".into();
              if self.o.noisy > 0 {
                println!("non-smooth bifurcation point detected");
              }
              self.treat_nonsmooth_point(md, x, *gamma, &tx0, tgamma0, false)?;
            }
          }
        }
        if step_dec == 0 && it < self.o.thr_iter {
          *h = (self.o.h_inc * *h).min(self.o.h_max);
        }
      } else if *h > self.o.h_min {
        *h = (self.o.h_dec * *h).max(self.o.h_min);
        step_dec += 1;
      } else if self.o.non_smooth && !tangent_switched {
        if self.o.noisy > 0 {
          println!("classical continuation has failed");
        }
        if self.switch_tangent(md, x, *gamma, tx, tgamma, h)? {
          tangent_switched = true;
          step_dec = if *h >= self.o.h_init { 0 } else { 1 };
        } else {
          break;
        }
      } else {
        break;
      }
      if new_point {
        break;
      }
    }
    if new_point {
      *x = xx;
      *gamma = gg;
      *tx = txx;
      *tgamma = tgg;
    } else if self.o.non_smooth && self.o.singularities > 1 {
      h0 = self.o.h_min;
      self.treat_nonsmooth_point(md, x, *gamma, &tx0, tgamma0, true)?;
      if !self.x_next.is_empty() {
        // GetFEM tests the t_Gamma of the last corrector here (H:698)
        if self.test_limit_point(tgg) {
          self.sing_label = "limit point".into();
        }
        let mut bif = self.tx_sing.len() > 2;
        let (xn, gn) = (self.x_next.clone(), self.gamma_next);
        let (t1, tg1) = (self.tx_sing[1].clone(), self.tgamma_sing[1]);
        if bif {
          self.tau_bp_1 = TAU_BP_INIT;
          self.tau_bp_2 = self.test_function_bp_at(md, &xn, gn, &t1, tg1)?.0;
        } else {
          bif = self.test_nonsmooth_bifurcation(md, x, *gamma, tx, *tgamma, &xn, gn, &t1, tg1)?;
        }
        if bif {
          self.sing_label = "non-smooth bifurcation point".into();
        }
        *x = xn;
        *gamma = gn;
        *tx = t1;
        *tgamma = tg1;
        *h = self.o.h_init;
        new_point = true;
      }
    }
    if !new_point {
      if self.o.noisy > 0 {
        println!("continuation has failed");
      }
      h0 = 0.0;
      *h = 0.0;
    }
    self.set_variables(md, x, *gamma)?;
    Ok(h0)
  }

  /// Newton at gamma_max from the tangent's point there, then the
  /// tangent; (x, gamma, tx, tgamma) unchanged if it fails
  fn back_to_gamma_max(
    &mut self,
    md: &mut Model,
    x: &mut Vec<f64>,
    gamma: &mut f64,
    tx: &mut Vec<f64>,
    tgamma: &mut f64,
  ) -> Result<bool> {
    let g = self.o.gamma_max;
    let mut y = axpy(x, tx, (g - *gamma) / *tgamma);
    let mut f = self.f(md, &y, g)?;
    for _ in 0..self.o.max_iter {
      let (dx, _, _) = self.grad_solves(md, &y, g, Some(&f))?;
      let dx = dx.unwrap();
      y = axpy(&y, &dx, -1.0);
      f = self.f(md, &y, g)?;
      let (res, diff) = (vect_norm2(&f), self.w_norm(&dx, 0.0));
      if self.o.noisy > 1 {
        println!(" to gamma_max: residual {res:.3e} difference {diff:.3e}");
      }
      if !res.is_finite() {
        break;
      }
      if res <= self.o.max_res && diff <= self.o.max_diff {
        self.compute_tangent(md, &y, g, tx, tgamma)?;
        *x = y;
        *gamma = g;
        return Ok(true);
      }
    }
    Ok(false)
  }

  // ---- singular points ----

  fn test_limit_point(&mut self, tgamma: f64) -> bool {
    let old = self.tau_lp;
    self.tau_lp = tgamma;
    tgamma * old < 0.0
  }

  fn init_test_functions(
    &mut self,
    md: &mut Model,
    x: &[f64],
    gamma: f64,
    tx: &[f64],
    tgamma: f64,
  ) -> Result<()> {
    self.tau_lp = tgamma;
    if self.o.singularities > 1 {
      self.tau_bp_2 = self.test_function_bp_at(md, x, gamma, tx, tgamma)?.0;
    }
    Ok(())
  }

  /// GetFEM init_border: srand(seed) (GetFEM: the time), then the borders
  fn init_border(&mut self, n: usize) {
    self.rand = Rand::new(self.o.seed);
    let nf = n as f64;
    self.bb_x = vec![0.0; n];
    self.rand.fill_random(&mut self.bb_x);
    self.cc_x = vec![0.0; n];
    self.rand.fill_random(&mut self.cc_x);
    self.bb_gamma = self.rand.random() / nf;
    self.cc_gamma = self.rand.random() / nf;
    self.dd = self.rand.random() / nf;
    self.bb_x.iter_mut().for_each(|v| *v /= nf);
    self.cc_x.iter_mut().for_each(|v| *v /= nf);
  }

  /// test function for bifurcations with the matrix A: (tau, v_x, v_gamma)
  fn test_function_bp(
    &mut self,
    a: &ColMatrix<f64>,
    g: &[f64],
    tx: &[f64],
    tgamma: f64,
  ) -> Result<(f64, Vec<f64>, f64)> {
    let n = g.len();
    if self.bb_x.len() != n {
      self.init_border(n);
    }
    let bb = self.bb_x.clone();
    let (y, z) = self.solve2(a, g, &bb)?;
    // the tangent row in the metric, as the corrector's
    let mut v_gamma = (self.bb_gamma - self.w_sp(tx, &z)) / (tgamma - self.w_sp(tx, &y));
    let mut v_x = axpy(&z, &y, -v_gamma);
    let tau = 1.0 / (self.dd - vect_sp(&self.cc_x, &v_x) - self.cc_gamma * v_gamma);
    v_x.iter_mut().for_each(|v| *v *= -tau);
    v_gamma *= -tau;
    if self.o.noisy > 0 {
      let mut r = crate::model::mult_summed(a, &v_x);
      for i in 0..n {
        r[i] += v_gamma * g[i] + tau * bb[i];
      }
      let s = self.w_sp(tx, &v_x) + tgamma * v_gamma + self.bb_gamma * tau;
      let q = vect_sp(&self.cc_x, &v_x) + self.cc_gamma * v_gamma + self.dd * tau - 1.0;
      let res = (vect_sp(&r, &r) + s * s + q * q).sqrt();
      if res > 1e-10 {
        println!("warning: test function evaluated with the residual {res:e}");
      }
    }
    Ok((tau, v_x, v_gamma))
  }

  fn test_function_bp_at(
    &mut self,
    md: &mut Model,
    x: &[f64],
    gamma: f64,
    tx: &[f64],
    tgamma: f64,
  ) -> Result<(f64, Vec<f64>, f64)> {
    let a = self.f_x(md, x, gamma)?;
    let g = self.f_gamma(md, x, gamma)?;
    self.test_function_bp(&a, &g, tx, tgamma)
  }

  fn test_smooth_bifurcation(
    &mut self,
    md: &mut Model,
    x: &[f64],
    gamma: f64,
    tx: &[f64],
    tgamma: f64,
  ) -> Result<bool> {
    let old = self.tau_bp_1;
    self.tau_bp_1 = self.tau_bp_2;
    self.tau_bp_2 = self.test_function_bp_at(md, x, gamma, tx, tgamma)?.0;
    Ok(self.tau_bp_2 * self.tau_bp_1 < 0.0 && self.tau_bp_1.abs() < old.abs())
  }

  /// GetFEM test_predict_dir: one predictor-corrector step from (x, gamma)
  /// along (tx, tgamma), oriented by the secant
  fn test_predict_dir(
    &mut self,
    md: &mut Model,
    x: &mut Vec<f64>,
    gamma: &mut f64,
    tx: &mut Vec<f64>,
    tgamma: &mut f64,
  ) -> Result<bool> {
    let mut h = self.o.h_init;
    let (mut xx, mut gg, mut txx, mut tgg);
    loop {
      xx = axpy(x, tx, h);
      gg = *gamma + h * *tgamma;
      txx = tx.clone();
      tgg = *tgamma;
      let (conv, _) = self.newton_corr(md, &mut xx, &mut gg, &mut txx, &mut tgg, tx, *tgamma)?;
      if h > self.o.h_min {
        h = (0.199 * self.o.h_dec * h).max(self.o.h_min);
      } else if !conv {
        return Ok(false);
      }
      if conv {
        break;
      }
    }
    // the tangent along the secant
    let sx = axpy(&xx, x, -1.0);
    let sg = gg - *gamma;
    if self.w_sp(&txx, &sx) + tgg * sg < 0.0 {
      txx.iter_mut().for_each(|v| *v = -*v);
      tgg = -tgg;
    }
    *x = xx;
    *gamma = gg;
    *tx = txx;
    *tgamma = tgg;
    Ok(true)
  }

  /// GetFEM treat_smooth_bif_point: locate the point, the second branch
  fn treat_smooth_bif_point(
    &mut self,
    md: &mut Model,
    x: &[f64],
    gamma: f64,
    tx: &[f64],
    tgamma: f64,
    h: f64,
  ) -> Result<()> {
    let (mut tau0, mut tau1) = (self.tau_bp_1, self.tau_bp_2);
    let (mut x0, mut gamma0, mut tx0, mut tgamma0) = (x.to_vec(), gamma, tx.to_vec(), tgamma);
    let (mut txx, mut tgg) = (tx.to_vec(), tgamma);
    let (mut v_x, mut v_gamma) = (tx.to_vec(), 0.0);
    let mut h = h * tau1 / (tau0 - tau1);
    let mut i = 0;
    while i < 10 && h.abs() >= self.o.h_min {
      let mut xx = axpy(&x0, &tx0, h);
      let mut gg = gamma0 + h * tgamma0;
      let (t0, tg0) = (tx0.clone(), tgamma0);
      if self
        .newton_corr(md, &mut xx, &mut gg, &mut txx, &mut tgg, &t0, tg0)?
        .0
      {
        x0 = xx;
        gamma0 = gg;
        if self.cosang(&txx, &tx0, tgg, tgamma0) >= self.o.min_cos {
          tx0.clone_from(&txx);
          tgamma0 = tgg;
        }
        tau0 = tau1;
        let r = self.test_function_bp_at(md, &x0, gamma0, &tx0, tgamma0)?;
        (tau1, v_x, v_gamma) = r;
        h *= tau1 / (tau0 - tau1);
      } else {
        // GetFEM keeps the uncorrected predictor (H:448)
        x0 = axpy(&x0, &tx0, h);
        gamma0 += h * tgamma0;
        let r = self.test_function_bp_at(md, &x0, gamma0, &tx0, tgamma0)?;
        (_, v_x, v_gamma) = r;
        break;
      }
      i += 1;
    }
    self.x_sing = x0.clone();
    self.gamma_sing = gamma0;
    self.insert_tangent_sing(&tx0, tgamma0);
    let no = self.w_norm(&v_x, v_gamma);
    v_x.iter_mut().for_each(|v| *v /= no);
    v_gamma /= no;
    if self.test_predict_dir(md, &mut x0, &mut gamma0, &mut v_x, &mut v_gamma)?
      && self.insert_tangent_sing(&v_x, v_gamma)
      && self.o.noisy > 0
    {
      println!("second branch found");
    }
    Ok(())
  }

  fn clear_sing_data(&mut self) {
    self.sing_label.clear();
    self.x_sing.clear();
    self.x_next.clear();
    self.tx_sing.clear();
    self.tgamma_sing.clear();
    self.tx_predict.clear();
    self.tgamma_predict.clear();
  }

  fn insert_tangent_sing(&mut self, tx: &[f64], tgamma: f64) -> bool {
    let inc = (0..self.tx_sing.len())
      .any(|i| self.cosang(&self.tx_sing[i], tx, self.tgamma_sing[i], tgamma) >= self.o.min_cos);
    if !inc {
      self.tx_sing.push(tx.to_vec());
      self.tgamma_sing.push(tgamma);
    }
    !inc
  }

  fn insert_tangent_predict(&mut self, tx: &[f64], tgamma: f64) -> bool {
    let inc = (0..self.tx_predict.len()).any(|i| {
      self
        .cosang(&self.tx_predict[i], tx, self.tgamma_predict[i], tgamma)
        .abs()
        >= self.o.min_cos
    });
    if !inc {
      self.tx_predict.push(tx.to_vec());
      self.tgamma_predict.push(tgamma);
    }
    !inc
  }

  fn set_next_point(&mut self, x: &[f64], gamma: f64) {
    if self.x_next.is_empty() {
      self.x_next = x.to_vec();
      self.gamma_next = gamma;
    }
  }

  // ---- non-smooth points ----

  /// GetFEM test_tangent: is the tangent at (x, gamma) + h (tX, tGamma)
  /// close to (tX, tGamma)?
  #[allow(clippy::too_many_arguments)]
  fn test_tangent(
    &mut self,
    md: &mut Model,
    x: &[f64],
    gamma: f64,
    tx_t: &[f64],
    tgamma_t: f64,
    tx: &[f64],
    tgamma: f64,
    h: f64,
  ) -> Result<bool> {
    let x1 = axpy(x, tx_t, h);
    let g1 = gamma + h * tgamma_t;
    let (mut t1, mut tg1) = (tx.to_vec(), tgamma);
    self.compute_tangent(md, &x1, g1, &mut t1, &mut tg1)?;
    Ok(self.cosang(&t1, tx_t, tg1, tgamma_t) >= self.o.min_cos)
  }

  /// GetFEM switch_tangent: a simple tangent switch
  fn switch_tangent(
    &mut self,
    md: &mut Model,
    x: &[f64],
    gamma: f64,
    tx: &mut Vec<f64>,
    tgamma: &mut f64,
    h: &mut f64,
  ) -> Result<bool> {
    *h *= 1.5;
    let xx = axpy(x, tx, *h);
    let gg = gamma + *h * *tgamma;
    let (mut txx, mut tgg) = (tx.clone(), *tgamma);
    self.compute_tangent(md, &xx, gg, &mut txx, &mut tgg)?;
    let mut h_test = -0.9 * self.o.h_min;
    let mut accepted = false;
    while !accepted && h_test > -self.o.h_max {
      h_test = -h_test + 10f64.powf((-h_test / self.o.h_min).log10().floor()) * self.o.h_min;
      accepted = self.test_tangent(md, x, gamma, &txx, tgg, tx, *tgamma, h_test)?;
      if !accepted {
        h_test = -h_test;
        accepted = self.test_tangent(md, x, gamma, &txx, tgg, tx, *tgamma, h_test)?;
      }
    }
    if accepted {
      if h_test < 0.0 {
        txx.iter_mut().for_each(|v| *v = -*v);
        tgg = -tgg;
        h_test = -h_test;
      }
      *h = self.o.h_init;
      let mut adapted = false;
      while !adapted && *h > h_test {
        adapted = self.test_tangent(md, x, gamma, &txx, tgg, tx, *tgamma, *h)?;
        *h *= self.o.h_dec;
      }
      *h = if adapted { *h / self.o.h_dec } else { h_test };
      *tx = txx;
      *tgamma = tgg;
    } else if self.o.noisy > 0 {
      println!("simple tangent switch has failed");
    }
    Ok(accepted)
  }

  /// GetFEM test_nonsmooth_bifurcation: sign changes of the test function
  /// on the convex combination of the two Jacobians
  #[allow(clippy::too_many_arguments)]
  fn test_nonsmooth_bifurcation(
    &mut self,
    md: &mut Model,
    x1: &[f64],
    gamma1: f64,
    tx1: &[f64],
    tgamma1: f64,
    x2: &[f64],
    gamma2: f64,
    tx2: &[f64],
    tgamma2: f64,
  ) -> Result<bool> {
    let a2 = self.f_x(md, x2, gamma2)?;
    let g2 = self.f_gamma(md, x2, gamma2)?;
    let a1 = self.f_x(md, x1, gamma1)?;
    let g1 = self.f_gamma(md, x1, gamma1)?;
    let mut tau1 = self.test_function_bp(&a1, &g1, tx1, tgamma1)?.0;
    let mut tau2 = self.test_function_bp(&a2, &g2, tx2, tgamma2)?.0;
    let tau_var_ref = (tau2 - tau1).abs().max(1e-8);
    self.tau_bp_2 = tau1;
    self.tau_bp_graph = vec![(0.0, self.tau_bp_2)];
    let mut nb_changes = 0;
    let (mut delta, mut tau0, mut alpha) = (self.o.delta_min, TAU_BP_INIT, 0.0f64);
    while alpha < 1.0 {
      alpha = (alpha + delta).min(1.0);
      let a = scaled_add_matrices(&a1, 1.0 - alpha, &a2, alpha);
      let g = lin(&g1, 1.0 - alpha, &g2, alpha);
      let tx = lin(tx1, 1.0 - alpha, tx2, alpha);
      let tgamma = (1.0 - alpha) * tgamma1 + alpha * tgamma2;
      tau2 = self.test_function_bp(&a, &g, &tx, tgamma)?.0;
      if tau2 * tau1 < 0.0 && tau1.abs() < tau0.abs() {
        nb_changes += 1;
      }
      self.tau_bp_graph.push((alpha, tau2));
      if (tau2 - tau1).abs() < 0.5 * self.o.thr_var * tau_var_ref {
        delta = (2.0 * delta).min(self.o.delta_max);
      } else if (tau2 - tau1).abs() > self.o.thr_var * tau_var_ref {
        delta = (0.1 * delta).max(self.o.delta_min);
      }
      tau0 = tau1;
      tau1 = tau2;
    }
    self.tau_bp_1 = TAU_BP_INIT;
    self.tau_bp_2 = tau2;
    Ok(nb_changes % 2 == 1)
  }

  /// GetFEM treat_nonsmooth_point: locate the end point of the branch
  /// (x, gamma) + (tx, tgamma), then search the branches from it
  fn treat_nonsmooth_point(
    &mut self,
    md: &mut Model,
    x: &[f64],
    gamma: f64,
    tx: &[f64],
    tgamma: f64,
    set_next: bool,
  ) -> Result<()> {
    let (mut x0, mut gamma0, mut tx0, mut tgamma0) = (x.to_vec(), gamma, tx.to_vec(), tgamma);
    let (mut txx, mut tgg) = (tx.to_vec(), tgamma);
    let (mut h, mut mcos) = (self.o.h_min, self.o.min_cos);
    // the end point by bisection
    let mut xx = axpy(&x0, &tx0, h);
    let mut gg = gamma0 + h * tgamma0;
    let (t0, tg0) = (tx0.clone(), tgamma0);
    if self
      .newton_corr(md, &mut xx, &mut gg, &mut txx, &mut tgg, &t0, tg0)?
      .0
    {
      let cang = self.cosang(&txx, &tx0, tgg, tgamma0);
      if cang >= mcos {
        mcos = (cang + 1.0) / 2.0;
      }
    }
    txx.clone_from(&tx0);
    tgg = tgamma0;
    h /= 2.0;
    for _ in 0..15 {
      xx = axpy(&x0, &tx0, h);
      gg = gamma0 + h * tgamma0;
      let (t0, tg0) = (tx0.clone(), tgamma0);
      if self
        .newton_corr(md, &mut xx, &mut gg, &mut txx, &mut tgg, &t0, tg0)?
        .0
        && self.cosang(&txx, &tx0, tgg, tgamma0) >= mcos
      {
        x0.clone_from(&xx);
        gamma0 = gg;
        tx0.clone_from(&txx);
        tgamma0 = tgg;
      } else {
        txx.clone_from(&tx0);
        tgg = tgamma0;
      }
      h /= 2.0;
    }
    self.x_sing = x0.clone();
    self.gamma_sing = gamma0;
    // two reference vectors spanning directions from the end point
    let (mut tgamma1, mut tgamma2) = (-tgamma0, tgamma0);
    let mut tx1: Vec<f64> = tx0.iter().map(|v| -v).collect();
    let mut tx2 = tx0.clone();
    self.insert_tangent_sing(&tx1, tgamma1);
    h = self.o.h_min;
    xx = axpy(&x0, &tx0, h);
    gg = gamma0 + h * tgamma0;
    self.compute_tangent(md, &xx, gg, &mut tx2, &mut tgamma2)?;
    let (mut i1, mut i2, mut nspan) = (0usize, 0usize, 0usize);
    loop {
      for i in 0..self.o.nb_dir {
        let a = 2.0 * std::f64::consts::PI * i as f64 / self.o.nb_dir as f64;
        let (a1, a2) = (a.sin(), a.cos());
        txx = lin(&tx1, a1, &tx2, a2);
        tgg = a1 * tgamma1 + a2 * tgamma2;
        let no = self.w_norm(&txx, tgg);
        xx = axpy(&x0, &txx, h / no);
        gg = gamma0 + h / no * tgg;
        self.compute_tangent(md, &xx, gg, &mut txx, &mut tgg)?;
        if self.cosang(&txx, &tx0, tgg, tgamma0).abs() < self.o.min_cos || (i == 0 && nspan == 0) {
          tx0.clone_from(&txx);
          tgamma0 = tgg;
          if self.insert_tangent_predict(&txx, tgg) {
            xx.clone_from(&x0);
            gg = gamma0;
            if self.test_predict_dir(md, &mut xx, &mut gg, &mut txx, &mut tgg)? {
              if self.insert_tangent_sing(&txx, tgg) {
                if i == 0
                  && nspan == 0
                  && self.cosang(&txx, &tx0, tgg, tgamma0).abs() >= self.o.min_cos
                {
                  i2 = 1;
                }
                if self.o.noisy > 0 {
                  println!("a new branch located (for nspan = {nspan})");
                }
                if set_next {
                  self.set_next_point(&xx, gg);
                }
              }
              xx.clone_from(&x0);
              gg = gamma0;
              txx.clone_from(&tx0);
              tgg = tgamma0;
            }
            txx.iter_mut().for_each(|v| *v = -*v);
            tgg = -tgg;
            if self.test_predict_dir(md, &mut xx, &mut gg, &mut txx, &mut tgg)?
              && self.insert_tangent_sing(&txx, tgg)
            {
              if self.o.noisy > 0 {
                println!("a new branch located (for nspan = {nspan})");
              }
              if set_next {
                self.set_next_point(&xx, gg);
              }
            }
          }
        }
      }
      // varying the reference vectors
      let mut perturb = true;
      if i1 + 1 < i2 {
        i1 += 1;
        perturb = false;
      } else if i2 + 1 < self.tx_sing.len() {
        i2 += 1;
        i1 = 0;
        perturb = false;
      }
      if !perturb {
        tx1.clone_from(&self.tx_sing[i1]);
        tgamma1 = self.tgamma_sing[i1];
        tx2.clone_from(&self.tx_sing[i2]);
        tgamma2 = self.tgamma_sing[i2];
      } else {
        self.rand.fill_random(&mut txx);
        tgg = self.rand.random();
        let no = self.w_norm(&txx, tgg);
        tx2 = axpy(&tx2, &txx, 0.1 / no);
        tgamma2 += 0.1 / no * tgg;
        xx = axpy(&x0, &tx2, h);
        gg = gamma0 + h * tgamma2;
        self.compute_tangent(md, &xx, gg, &mut tx2, &mut tgamma2)?;
      }
      nspan += 1;
      if nspan >= self.o.nb_span {
        break;
      }
    }
    if self.o.noisy > 0 {
      println!("{} branches from the non-smooth point", self.tx_sing.len());
    }
    Ok(())
  }
}

/// c1 A1 + c2 A2 (gmm::add of scaled matrices)
fn scaled_add_matrices(
  a1: &ColMatrix<f64>,
  c1: f64,
  a2: &ColMatrix<f64>,
  c2: f64,
) -> ColMatrix<f64> {
  let mut c = a1.clone();
  c.cols.iter_mut().flatten().for_each(|e| e.1 *= c1);
  for (j, col) in a2.cols.iter().enumerate() {
    let s: Vec<(usize, f64)> = col.iter().map(|&(i, v)| (i, c2 * v)).collect();
    c.add_to_column(j, &s);
  }
  c
}
