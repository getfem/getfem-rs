//! Signed distances and projections onto them (the parts of GetFEM's
//! getfem_mesher used to cut elements by level sets): half spaces, the
//! reference simplex, parallelepiped and prism as their faces, projection
//! of a point onto one or several constraints, curvature radius estimate.

use crate::convex_structure::{self as cs, PCvs};
use crate::{Error, Result};
use getfem_linalg::tensor::{Tensor, inverse, sym_eigenvalues};
use getfem_linalg::{vect_dist2, vect_sp};

/// A constraint f <= 0 (GetFEM mesher_signed_distance): value, gradient,
/// Hessian (column-major).
pub trait SignedDistance: Send + Sync {
  fn val(&self, p: &[f64]) -> f64;
  /// the value and the gradient
  fn grad(&self, p: &[f64]) -> (f64, Vec<f64>);
  fn hess(&self, p: &[f64]) -> Vec<f64>;
}

/// |f| below this: on the constraint
pub const SEPS: f64 = 1e-8;

/// {x : (x - x0).n >= 0}, f = (x0 - x).n, n unit (GetFEM mesher_half_space)
#[derive(Clone, Debug)]
pub struct HalfSpace {
  n: Vec<f64>,
  xon: f64,
}

impl HalfSpace {
  pub fn new(x0: &[f64], n: &[f64]) -> Self {
    let l = n.iter().map(|v| v * v).sum::<f64>().sqrt();
    let n: Vec<f64> = n.iter().map(|v| v / l).collect();
    let xon = x0.iter().zip(&n).map(|(a, b)| a * b).sum();
    HalfSpace { n, xon }
  }
}

impl HalfSpace {
  /// p put exactly on the face if within tol of it (or beyond): an axis
  /// face sets the coordinate; a sloped face sum_j x_j = c (equal normal
  /// components) rounds those coordinates to multiples of 2^-40 and sets
  /// the largest to c minus the others, exact in floating point.
  pub fn snap(&self, p: &mut [f64], tol: f64) {
    if self.val(p) < -tol {
      return;
    }
    let j: Vec<usize> = (0..p.len()).filter(|&k| self.n[k] != 0.0).collect();
    // x.n = xon, n_j equal: sum_j x_j = xon / n_j
    let c = self.xon / self.n[j[0]];
    if j.len() == 1 {
      p[j[0]] = c.round();
      return;
    }
    let g = (2.0f64).powi(40);
    for &k in &j {
      p[k] = (p[k] * g).round() / g;
    }
    let &m = j.iter().max_by(|&&a, &&b| p[a].total_cmp(&p[b])).unwrap();
    p[m] = c.round() - j.iter().filter(|&&k| k != m).map(|&k| p[k]).sum::<f64>();
  }
}

impl SignedDistance for HalfSpace {
  fn val(&self, p: &[f64]) -> f64 {
    self.xon - p.iter().zip(&self.n).map(|(a, b)| a * b).sum::<f64>()
  }
  fn grad(&self, p: &[f64]) -> (f64, Vec<f64>) {
    (self.val(p), self.n.iter().map(|v| -v).collect())
  }
  fn hess(&self, p: &[f64]) -> Vec<f64> {
    vec![0.0; p.len() * p.len()]
  }
}

/// A reference convex as the intersection of its face half spaces
/// (GetFEM mesher_simplex_ref, mesher_rectangle, mesher_prism_ref): f is
/// the largest of theirs.
#[derive(Clone, Debug)]
pub struct RefElement {
  faces: Vec<HalfSpace>,
}

impl RefElement {
  /// The reference element of a basic structure: simplex,
  /// parallelepiped, prism.
  pub fn of(cvs: &PCvs) -> Result<Self> {
    let n = cvs.dim();
    let e = |k: usize, v: f64| {
      let mut x = vec![0.0; n];
      x[k] = v;
      x
    };
    let zero = vec![0.0; n];
    let mut faces: Vec<HalfSpace> = vec![];
    if **cvs == *cs::simplex_structure(n) {
      faces.extend((0..n).map(|k| HalfSpace::new(&zero, &e(k, 1.0))));
      let org = vec![1.0 / n as f64; n];
      faces.push(HalfSpace::new(
        &org,
        &org.iter().map(|v| -v).collect::<Vec<_>>(),
      ));
    } else if **cvs == *cs::parallelepiped_structure(n, 1) {
      let one = vec![1.0; n];
      for k in 0..n {
        faces.push(HalfSpace::new(&zero, &e(k, 1.0)));
        faces.push(HalfSpace::new(&one, &e(k, -1.0)));
      }
    } else if n >= 2 && **cvs == *cs::prism_p1_structure(n) {
      // the simplex of the first n - 1 directions times [0, 1]: x_k >= 0,
      // x_n <= 1, sum of the first n - 1 <= 1 (GetFEM's org 1/n puts
      // this face at sum = (n - 1)/n, DEVIATIONS)
      faces.extend((0..n).map(|k| HalfSpace::new(&zero, &e(k, 1.0))));
      faces.push(HalfSpace::new(&e(n - 1, 1.0), &e(n - 1, -1.0)));
      let mut org = vec![1.0 / (n - 1) as f64; n];
      org[n - 1] = 0.0;
      faces.push(HalfSpace::new(
        &org,
        &org.iter().map(|v| -v).collect::<Vec<_>>(),
      ));
    } else {
      return Err(Error::NotImplemented(
        "reference element: simplices, parallelepipeds and prisms only",
      ));
    }
    Ok(RefElement { faces })
  }

  /// the face constraints (GetFEM register_constraints)
  pub fn faces(&self) -> &[HalfSpace] {
    &self.faces
  }
}

impl SignedDistance for RefElement {
  fn val(&self, p: &[f64]) -> f64 {
    self.faces.iter().map(|f| f.val(p)).fold(f64::MIN, f64::max)
  }
  /// the gradient of the largest face distance
  fn grad(&self, p: &[f64]) -> (f64, Vec<f64>) {
    let mut best = &self.faces[0];
    for f in &self.faces[1..] {
      if f.val(p) > best.val(p) {
        best = f;
      }
    }
    best.grad(p)
  }
  fn hess(&self, p: &[f64]) -> Vec<f64> {
    vec![0.0; p.len() * p.len()]
  }
}

/// Newton projection of x onto f = 0 (or onto f <= 0 unless on_surface):
/// false after 1000 iterations or 20 without progress.
pub fn try_projection(f: &dyn SignedDistance, x: &mut [f64], on_surface: bool) -> bool {
  let (mut d, mut g) = f.grad(x);
  if !on_surface && d <= 0.0 {
    return true;
  }
  let mut dmin = d.abs();
  let (mut iter, mut stall) = (0, 0);
  loop {
    let y = x.to_vec();
    iter += 1;
    if iter > 1000 {
      return false;
    }
    let s = -d / vect_sp(&g, &g).max(1e-8);
    x.iter_mut().zip(&g).for_each(|(xi, gi)| *xi += s * gi);
    (d, g) = f.grad(x);
    if d.abs() >= dmin * 0.95 {
      stall += 1;
    } else {
      stall = 0;
      dmin = d.abs();
    }
    if stall > 20 {
      return false;
    }
    if dmin <= 1e-15 && vect_dist2(x, &y) <= 1e-15 {
      return true;
    }
  }
}

/// Newton projection of x onto the intersection of the constraints
/// (GetFEM pure_multi_constraint_projection): least-norm steps with
/// backtracking; linearly dependent constraints are dropped for a step,
/// then all are projected on in turn. True if all are met within SEPS.
pub fn multi_projection(cs: &[&dyn SignedDistance], x: &mut [f64]) -> bool {
  let (k, n) = (cs.len(), x.len());
  match k {
    0 => return true,
    1 => {
      try_projection(cs[0], x, true);
      return cs[0].val(x).abs() <= SEPS;
    }
    _ => {}
  }
  let eval = |x: &[f64]| -> (Vec<f64>, Vec<Vec<f64>>) {
    let (d, g): (Vec<f64>, Vec<Vec<f64>>) = cs
      .iter()
      .map(|c| {
        let (v, g) = c.grad(x);
        (-v, g)
      })
      .unzip();
    (d, g)
  };
  let norm = |d: &[f64]| vect_sp(d, d).sqrt();
  let (mut d, mut g) = eval(x);
  let mut residual = norm(&d);
  let mut iter = 0;
  while iter < 200 && residual > 1e-14 && (residual > 1e-11 || iter < 15) {
    // H = G^T G (k x k), the step G H^-1 d
    let h: Vec<Vec<f64>> = (0..k)
      .map(|i| (0..k).map(|j| vect_sp(&g[i], &g[j])).collect())
      .collect();
    let rows: Vec<&[f64]> = h.iter().map(|r| r.as_slice()).collect();
    let hinv = if k <= n {
      inverse(&Tensor::from_rows(&rows)).ok()
    } else {
      None
    };
    match hinv.filter(|hi| hi.data().iter().all(|v| v.is_finite() && v.abs() < 1e20)) {
      Some(hi) => {
        let v: Vec<f64> = (0..k)
          .map(|i| (0..k).map(|j| hi[(i, j)] * d[j]).sum())
          .collect();
        let dx: Vec<f64> = (0..n)
          .map(|r| (0..k).map(|i| g[i][r] * v[i]).sum())
          .collect();
        x.iter_mut().zip(&dx).for_each(|(a, b)| *a += b);
        (d, g) = eval(x);
        let mut alpha = 1.0;
        if iter > 0 {
          while norm(&d) > residual && alpha > 1e-10 {
            alpha /= 2.0;
            x.iter_mut().zip(&dx).for_each(|(a, b)| *a -= alpha * b);
            (d, g) = eval(x);
          }
        }
        if alpha < 1e-15 {
          break;
        }
      }
      None => {
        // dependent constraints: the independent ones (Gram-Schmidt) first
        let mut q: Vec<Vec<f64>> = vec![];
        let mut kept: Vec<&dyn SignedDistance> = vec![];
        for (i, gi) in g.iter().enumerate() {
          let mut r = gi.clone();
          for qj in &q {
            let s = vect_sp(qj, &r);
            r.iter_mut().zip(qj).for_each(|(a, b)| *a -= s * b);
          }
          let l = norm(&r);
          if l >= 1e-10 {
            q.push(r.into_iter().map(|v| v / l).collect());
            kept.push(cs[i]);
          }
        }
        if kept.len() < k {
          multi_projection(&kept, x);
        }
        for c in cs {
          try_projection(*c, x, true);
        }
        (d, g) = eval(x);
      }
    }
    iter += 1;
    residual = norm(&d);
  }
  d.iter().all(|v| v.abs() <= SEPS)
}

/// |grad f| / the largest |eigenvalue| of its Hessian at x, projected
/// onto f = 0 first if proj.
pub fn curvature_radius_estimate(f: &dyn SignedDistance, x: &[f64], proj: bool) -> f64 {
  let mut x = x.to_vec();
  if proj {
    try_projection(f, &mut x, true);
  }
  let n = x.len();
  let (_, g) = f.grad(&x);
  let h = f.hess(&x);
  let rows: Vec<&[f64]> = (0..n).map(|i| &h[i * n..(i + 1) * n]).collect();
  let vp = sym_eigenvalues(&Tensor::from_rows(&rows))
    .into_iter()
    .fold(0.0f64, |a, v| a.max(v.abs()));
  vect_sp(&g, &g).sqrt() / vp.max(1e-10)
}
