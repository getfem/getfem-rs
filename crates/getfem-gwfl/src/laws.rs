//! GetFEM's hyperelastic laws as nonlinear operators
//! (getfem_nonlinear_elasticity.cc): `<law>_potential`, `<law>_PK2` (and its
//! alias `<law>_sigma`), with Plane_Strain_ variants. Each law has GetFEM's
//! hand-written strain energy, PK2 stress and its gradient in the
//! Green-Lagrange strain E; derivatives w.r.t. the law parameters are not
//! available (as GetFEM), except for Saint_Venant_Kirchhoff_sigma.

use std::sync::Arc;

use getfem_geom::geotrans::{lu_det, lu_inverse};

use crate::operators::{NonlinearOperator, OpResult, Operators};

#[derive(Clone, Copy, Debug)]
pub enum Law {
  SaintVenantKirchhoff,
  GeneralizedBlatzKo,
  CiarletGeymonat,
  /// C1 (j1 - 3) [+ C2 (j2 - 3)] [+ D1 (sqrt(i3) - 1)^2]
  MooneyRivlin {
    compressible: bool,
    neohookean: bool,
  },
  /// Bonet or Ciarlet (GetFEM: Wriggers) volumetric part
  NeoHookean {
    bonet: bool,
  },
}

/// 4th order tensor index (i, j, k, l), column-major
fn t4(n: usize, i: usize, j: usize, k: usize, l: usize) -> usize {
  i + n * (j + n * (k + n * l))
}

fn trace(a: &[f64], n: usize) -> f64 {
  (0..n).fold(0.0, |s, i| s + a[i * (n + 1)])
}

/// sum of A_ij B_ji
fn frobenius_product_trans(a: &[f64], b: &[f64], n: usize) -> f64 {
  let mut r = 0.0;
  for i in 0..n {
    for j in 0..n {
      r += a[i + n * j] * b[j + n * i];
    }
  }
  r
}

fn identity(n: usize) -> Vec<f64> {
  let mut a = vec![0.0; n * n];
  for i in 0..n {
    a[i * (n + 1)] = 1.0;
  }
  a
}

/// a += s b
fn add_scaled(a: &mut [f64], s: f64, b: &[f64]) {
  for (x, y) in a.iter_mut().zip(b) {
    *x += s * y;
  }
}

/// a += s x (x) y
fn add_dyad(r: &mut [f64], s: f64, x: &[f64], y: &[f64]) {
  let nn = x.len();
  for (kl, yv) in y.iter().enumerate() {
    for (ij, xv) in x.iter().enumerate() {
      r[ij + nn * kl] += s * xv * yv;
    }
  }
}

/// GetFEM's compute_invariants of a matrix C: i1, i2, i3, j1, j2, their
/// gradients and symmetrized second gradients
struct Invariants {
  c: Vec<f64>,
  n: usize,
  /// C^-1 and det C
  cinv: Vec<f64>,
  i3: f64,
}

impl Invariants {
  fn new(c: Vec<f64>, n: usize) -> OpResult<Self> {
    let mut cinv = c.clone();
    let i3 = lu_inverse(&mut cinv, n).map_err(|_| "Non invertible matrix".to_string())?;
    Ok(Invariants { c, n, cinv, i3 })
  }

  fn i1(&self) -> f64 {
    trace(&self.c, self.n)
  }

  fn di1(&self) -> Vec<f64> {
    identity(self.n)
  }

  fn i2(&self) -> f64 {
    let tr = self.i1();
    (tr * tr - frobenius_product_trans(&self.c, &self.c, self.n)) / 2.0
  }

  fn di2(&self) -> Vec<f64> {
    let mut d: Vec<f64> = identity(self.n).iter().map(|x| x * self.i1()).collect();
    add_scaled(&mut d, -1.0, &self.c);
    d
  }

  fn ddi2(&self) -> Vec<f64> {
    let n = self.n;
    let mut r = vec![0.0; n * n * n * n];
    for i in 0..n {
      for k in 0..n {
        r[t4(n, i, i, k, k)] += 1.0;
      }
    }
    for i in 0..n {
      for j in 0..n {
        r[t4(n, i, j, j, i)] -= 0.5;
        r[t4(n, j, i, j, i)] -= 0.5;
      }
    }
    r
  }

  fn di3(&self) -> Vec<f64> {
    self.cinv.iter().map(|x| x * self.i3).collect()
  }

  fn ddi3(&self) -> Vec<f64> {
    let n = self.n;
    let det = self.i3 / 2.0;
    let e = |i: usize, j: usize| self.cinv[i + n * j];
    let mut r = vec![0.0; n * n * n * n];
    for i in 0..n {
      for j in 0..n {
        for k in 0..n {
          for l in 0..n {
            r[t4(n, i, j, k, l)] =
              det * (e(j, i) * e(l, k) - e(j, k) * e(l, i) + e(i, j) * e(l, k) - e(i, k) * e(l, j));
          }
        }
      }
    }
    r
  }

  fn j1(&self) -> f64 {
    self.i1() * self.i3.abs().powf(-1.0 / 3.0)
  }

  fn dj1(&self) -> Vec<f64> {
    let mut d = self.di1();
    add_scaled(&mut d, -self.i1() / (3.0 * self.i3), &self.di3());
    let s = self.i3.abs().powf(-1.0 / 3.0);
    d.iter_mut().for_each(|x| *x *= s);
    d
  }

  fn ddj1(&self) -> Vec<f64> {
    let (di1, di3) = (self.di1(), self.di3());
    let coeff1 = 1.0 / (3.0 * self.i3);
    let coeff2 = 4.0 * coeff1 * coeff1 * self.i1();
    let mut r = self.ddi3();
    r.iter_mut().for_each(|x| *x *= -self.i1() * coeff1);
    add_dyad(&mut r, coeff2, &di3, &di3);
    add_dyad(&mut r, -coeff1, &di1, &di3);
    add_dyad(&mut r, -coeff1, &di3, &di1);
    let s = self.i3.abs().powf(-1.0 / 3.0);
    r.iter_mut().for_each(|x| *x *= s);
    r
  }

  fn j2(&self) -> f64 {
    self.i2() * self.i3.abs().powf(-2.0 / 3.0)
  }

  fn dj2(&self) -> Vec<f64> {
    let mut d = self.di2();
    add_scaled(&mut d, -2.0 * self.i2() / (3.0 * self.i3), &self.di3());
    let s = self.i3.abs().powf(-2.0 / 3.0);
    d.iter_mut().for_each(|x| *x *= s);
    d
  }

  fn ddj2(&self) -> Vec<f64> {
    let (di2, di3) = (self.di2(), self.di3());
    let coeff1 = 2.0 / (3.0 * self.i3);
    let coeff2 = 5.0 * coeff1 * coeff1 * self.i2() / 2.0;
    let mut r = self.ddi2();
    add_scaled(&mut r, -self.i2() * coeff1, &self.ddi3());
    add_dyad(&mut r, coeff2, &di3, &di3);
    add_dyad(&mut r, -coeff1, &di2, &di3);
    add_dyad(&mut r, -coeff1, &di3, &di2);
    let s = self.i3.abs().powf(-2.0 / 3.0);
    r.iter_mut().for_each(|x| *x *= s);
    r
  }
}

/// C = 2 E + I
fn cauchy_green(e: &[f64], n: usize) -> Vec<f64> {
  e.iter()
    .zip(identity(n))
    .map(|(x, i)| 2.0 * x + i)
    .collect()
}

fn dim3(law: &str, n: usize) -> OpResult<()> {
  if n == 3 {
    Ok(())
  } else {
    Err(format!(
      "{law} hyperelastic law only defined on dimension 3, sorry"
    ))
  }
}

impl Law {
  fn nb_params(self) -> usize {
    match self {
      Law::SaintVenantKirchhoff => 2,
      Law::GeneralizedBlatzKo => 5,
      Law::CiarletGeymonat => 3,
      Law::MooneyRivlin {
        compressible,
        neohookean,
      } => 2 + compressible as usize - neohookean as usize,
      Law::NeoHookean { .. } => 2,
    }
  }

  /// only defined in 3D (2D through the plane strain variant)
  fn only_3d(self) -> bool {
    matches!(
      self,
      Law::GeneralizedBlatzKo | Law::MooneyRivlin { .. } | Law::NeoHookean { .. }
    )
  }

  fn strain_energy(self, e: &[f64], n: usize, p: &[f64], det_trans: f64) -> OpResult<f64> {
    Ok(match self {
      Law::SaintVenantKirchhoff => {
        if det_trans <= 0.0 {
          return Ok(1e200);
        }
        let tr = trace(e, n);
        tr * tr * p[0] / 2.0 + e.iter().fold(0.0, |s, x| s + x * x) * p[1]
      }
      Law::MooneyRivlin {
        compressible,
        neohookean,
      } => {
        if compressible && det_trans <= 0.0 {
          return Ok(1e200);
        }
        dim3("Mooney Rivlin", n)?;
        let ci = Invariants::new(cauchy_green(e, n), n)?;
        let mut k = 0;
        let mut w = p[k] * (ci.j1() - 3.0);
        k += 1;
        if !neohookean {
          w += p[k] * (ci.j2() - 3.0);
          k += 1;
        }
        if compressible {
          let s = ci.i3.abs().sqrt() - 1.0;
          w += p[k] * s * s;
        }
        w
      }
      Law::NeoHookean { bonet } => {
        if det_trans <= 0.0 {
          return Ok(1e200);
        }
        dim3("Neo Hookean", n)?;
        let ci = Invariants::new(cauchy_green(e, n), n)?;
        let (lambda, mu) = (p[0], p[1]);
        let logi3 = ci.i3.ln();
        let w = mu / 2.0 * (ci.i1() - 3.0 - logi3);
        if bonet {
          w + lambda / 8.0 * logi3 * logi3
        } else {
          w + lambda / 4.0 * (ci.i3 - 1.0 - logi3)
        }
      }
      Law::GeneralizedBlatzKo => {
        if det_trans <= 0.0 {
          return Ok(1e200);
        }
        dim3("Generalized Blatz Ko", n)?;
        let ci = Invariants::new(cauchy_green(e, n), n)?;
        let (a, b, c, d, nn) = (p[0], p[1], p[2], p[3], p[4]);
        (a * ci.i1() + b * ci.i3.abs().sqrt() + c * ci.i2() / ci.i3 + d).powf(nn)
      }
      Law::CiarletGeymonat => {
        if det_trans <= 0.0 {
          return Ok(1e200);
        }
        let a = p[2];
        let b = p[1] / 2.0 - p[2];
        let c = p[0] / 4.0 - p[1] / 2.0 + p[2];
        let d = p[0] / 2.0 + p[1];
        let ee = -(3.0 * (a + b) + c);
        let cc = cauchy_green(e, n);
        let det = lu_det(&cc, n);
        let tr = trace(&cc, n);
        let nc = cc.iter().fold(0.0, |s, x| s + x * x);
        a * tr + b * (tr * tr - nc) / 2.0 + c * det - d * det.ln() / 2.0 + ee
      }
    })
  }

  /// PK2 stress
  fn sigma(self, e: &[f64], n: usize, p: &[f64], det_trans: f64) -> OpResult<Vec<f64>> {
    let cc = cauchy_green(e, n);
    let big = |r: &mut Vec<f64>, x: &[f64]| {
      if det_trans <= 0.0 {
        add_scaled(r, 1e200, x);
      }
    };
    Ok(match self {
      Law::SaintVenantKirchhoff => {
        let mut r: Vec<f64> = identity(n)
          .iter()
          .map(|x| x * (p[0] * trace(e, n)))
          .collect();
        add_scaled(&mut r, 2.0 * p[1], e);
        big(&mut r, e);
        r
      }
      Law::MooneyRivlin {
        compressible,
        neohookean,
      } => {
        dim3("Mooney Rivlin", n)?;
        let ci = Invariants::new(cc.clone(), n)?;
        let mut k = 0;
        let mut r: Vec<f64> = ci.dj1().iter().map(|x| x * (2.0 * p[k])).collect();
        k += 1;
        if !neohookean {
          add_scaled(&mut r, 2.0 * p[k], &ci.dj2());
          k += 1;
        }
        if compressible {
          let di3 = p[k] - p[k] / ci.i3.abs().sqrt();
          add_scaled(&mut r, 2.0 * di3, &ci.di3());
          big(&mut r, &cc);
        }
        r
      }
      Law::NeoHookean { bonet } => {
        dim3("Neo Hookean", n)?;
        let ci = Invariants::new(cc.clone(), n)?;
        let (lambda, mu) = (p[0], p[1]);
        let mut r: Vec<f64> = ci.di1().iter().map(|x| x * mu).collect();
        let s = if bonet {
          (lambda / 2.0 * ci.i3.ln() - mu) / ci.i3
        } else {
          lambda / 2.0 - lambda / (2.0 * ci.i3) - mu / ci.i3
        };
        add_scaled(&mut r, s, &ci.di3());
        big(&mut r, &cc);
        r
      }
      Law::GeneralizedBlatzKo => {
        dim3("Generalized Blatz Ko", n)?;
        let ci = Invariants::new(cc.clone(), n)?;
        let (a, b, c, d, nn) = (p[0], p[1], p[2], p[3], p[4]);
        let z = a * ci.i1() + b * ci.i3.abs().sqrt() + c * ci.i2() / ci.i3 + d;
        let nz = nn * z.powf(nn - 1.0);
        let di1 = nz * a;
        let di2 = nz * c / ci.i3;
        let di3 = nz * (b / (2.0 * ci.i3.abs().sqrt()) - c * ci.i2() / (ci.i3 * ci.i3));
        let mut r: Vec<f64> = ci.di1().iter().map(|x| x * (di1 * 2.0)).collect();
        add_scaled(&mut r, di2 * 2.0, &ci.di2());
        add_scaled(&mut r, di3 * 2.0, &ci.di3());
        big(&mut r, &cc);
        r
      }
      Law::CiarletGeymonat => {
        let a = p[2];
        let b = p[1] / 2.0 - p[2];
        let c = p[0] / 4.0 - p[1] / 2.0 + p[2];
        let d = p[0] / 2.0 + p[1];
        let mut r: Vec<f64> = identity(n)
          .iter()
          .map(|x| x * (2.0 * (a + b * trace(&cc, n))))
          .collect();
        add_scaled(&mut r, -2.0 * b, &cc);
        if det_trans <= 0.0 {
          add_scaled(&mut r, 1e200, &cc);
        } else {
          let mut ci = cc.clone();
          let det = lu_inverse(&mut ci, n).map_err(|_| "Non invertible matrix".to_string())?;
          add_scaled(&mut r, 2.0 * c * det - d, &ci);
        }
        r
      }
    })
  }

  /// dPK2/dE, symmetrized
  fn grad_sigma(self, e: &[f64], n: usize, p: &[f64]) -> OpResult<Vec<f64>> {
    let cc = cauchy_green(e, n);
    let mut r = vec![0.0; n * n * n * n];
    match self {
      Law::SaintVenantKirchhoff => {
        for i in 0..n {
          for l in 0..n {
            r[t4(n, i, i, l, l)] += p[0];
            r[t4(n, i, l, i, l)] += p[1] / 2.0;
            r[t4(n, i, l, l, i)] += p[1] / 2.0;
            r[t4(n, l, i, i, l)] += p[1] / 2.0;
            r[t4(n, l, i, l, i)] += p[1] / 2.0;
          }
        }
      }
      Law::MooneyRivlin {
        compressible,
        neohookean,
      } => {
        dim3("Mooney Rivlin", n)?;
        let ci = Invariants::new(cc, n)?;
        let mut k = 0;
        r = ci.ddj1().iter().map(|x| x * (4.0 * p[k])).collect();
        k += 1;
        if !neohookean {
          add_scaled(&mut r, 4.0 * p[k], &ci.ddj2());
          k += 1;
        }
        if compressible {
          let d1 = p[k];
          let di3 = d1 - d1 / ci.i3.abs().sqrt();
          add_scaled(&mut r, 4.0 * di3, &ci.ddi3());
          // second derivative of W w.r.t. the third invariant
          let a22 = d1 / (2.0 * ci.i3.abs().powf(1.5));
          let di = ci.di3();
          add_dyad(&mut r, 4.0 * a22, &di, &di);
        }
      }
      Law::NeoHookean { bonet } => {
        dim3("Neo Hookean", n)?;
        let ci = Invariants::new(cc, n)?;
        let (lambda, mu) = (p[0], p[1]);
        let coeff;
        if bonet {
          let logi3 = ci.i3.ln();
          r = ci
            .ddi3()
            .iter()
            .map(|x| x * ((lambda * logi3 - 2.0 * mu) / ci.i3))
            .collect();
          coeff = (lambda + 2.0 * mu - lambda * logi3) / (ci.i3 * ci.i3);
        } else {
          r = ci
            .ddi3()
            .iter()
            .map(|x| x * (lambda - (lambda + 2.0 * mu) / ci.i3))
            .collect();
          coeff = (lambda + 2.0 * mu) / (ci.i3 * ci.i3);
        }
        let di = ci.di3();
        add_dyad(&mut r, coeff, &di, &di);
      }
      Law::GeneralizedBlatzKo => {
        dim3("Generalized Blatz Ko", n)?;
        let ci = Invariants::new(cc, n)?;
        let (a, b, c, d, nn) = (p[0], p[1], p[2], p[3], p[4]);
        let (i2, i3) = (ci.i2(), ci.i3);
        let z = a * ci.i1() + b * i3.abs().sqrt() + c * i2 / i3 + d;
        let nz = nn * z.powf(nn - 1.0);
        let di2 = nz * c / i3;
        let y = b / (2.0 * i3.abs().sqrt()) - c * i2 / (i3 * i3);
        let di3 = nz * y;
        // 4 di1 ddi1 = 0
        add_scaled(&mut r, 4.0 * di2, &ci.ddi2());
        add_scaled(&mut r, 4.0 * di3, &ci.ddi3());
        let nnz = nn * (nn - 1.0) * z.powf(nn - 2.0);
        // second derivatives of W w.r.t. the invariants
        let mut am = [[0.0; 3]; 3];
        am[0][0] = nnz * a * a;
        am[1][0] = nnz * a * c / i3;
        am[0][1] = am[1][0];
        am[2][0] = nnz * a * y;
        am[0][2] = am[2][0];
        am[1][1] = nnz * c * c / (i3 * i3);
        am[2][1] = nnz * y * c / i3 - nz * c / (i3 * i3);
        am[1][2] = am[2][1];
        am[2][2] = nnz * y * y + nz * (2.0 * c * i2 / i3.powi(3) - b / (4.0 * i3.powf(1.5)));
        let di = [ci.di1(), ci.di2(), ci.di3()];
        for j in 0..3 {
          for k in 0..3 {
            add_dyad(&mut r, 4.0 * am[j][k], &di[j], &di[k]);
          }
        }
      }
      Law::CiarletGeymonat => {
        let b2 = p[1] - p[2] * 2.0;
        let c = p[0] / 4.0 - p[1] / 2.0 + p[2];
        let d = p[0] / 2.0 + p[1];
        let mut ci = cc;
        let det = lu_inverse(&mut ci, n).map_err(|_| "Non invertible matrix".to_string())?;
        let m = |i: usize, j: usize| ci[i + n * j];
        for i in 0..n {
          for j in 0..n {
            r[t4(n, i, i, j, j)] += 2.0 * b2;
            r[t4(n, i, j, i, j)] -= b2;
            r[t4(n, i, j, j, i)] -= b2;
            for k in 0..n {
              for l in 0..n {
                r[t4(n, i, j, k, l)] += (m(i, k) * m(l, j) + m(i, l) * m(k, j))
                  * (d - 2.0 * det * c)
                  + (m(i, j) * m(k, l)) * det * c * 4.0;
              }
            }
          }
        }
      }
    }
    Ok(r)
  }
}

#[derive(Clone, Copy, Debug)]
enum Kind {
  Potential,
  Pk2,
}

#[derive(Clone, Copy, Debug)]
struct LawOp {
  law: Law,
  plane_strain: bool,
  kind: Kind,
}

/// E = (Gu' Gu + Gu + Gu')/2, F = I + Gu, det F
fn kinematics(gu: &[f64], n: usize) -> (Vec<f64>, Vec<f64>, f64) {
  let mut e = vec![0.0; n * n];
  for j in 0..n {
    for i in 0..n {
      let mut s = 0.0;
      for k in 0..n {
        s += gu[k + n * i] * gu[k + n * j];
      }
      e[i + n * j] = (s + gu[i + n * j] + gu[j + n * i]) * 0.5;
    }
  }
  let f: Vec<f64> = gu.iter().zip(identity(n)).map(|(x, i)| x + i).collect();
  let det = lu_det(&f, n);
  (e, f, det)
}

impl LawOp {
  fn shape(&self, a: &[&[usize]]) -> Option<Vec<usize>> {
    let [g, p] = a else { return None };
    let n = *g.first()?;
    let dim_ok = if self.plane_strain {
      n == 2
    } else {
      !self.law.only_3d() || n == 3
    };
    if g.len() != 2 || g[1] != n || p.iter().product::<usize>() != self.law.nb_params() || !dim_ok {
      return None;
    }
    Some(match self.kind {
      Kind::Potential => vec![],
      Kind::Pk2 => vec![n, n],
    })
  }

  /// the law on the (embedded, plane strain) strain: energy, PK2, grad PK2
  fn energy(&self, e: &[f64], n: usize, p: &[f64], det: f64) -> OpResult<f64> {
    if !self.plane_strain {
      return self.law.strain_energy(e, n, p, det);
    }
    self.law.strain_energy(&embed(e), 3, p, det)
  }

  fn pk2(&self, e: &[f64], n: usize, p: &[f64], det: f64) -> OpResult<Vec<f64>> {
    if !self.plane_strain {
      return self.law.sigma(e, n, p, det);
    }
    let s = self.law.sigma(&embed(e), 3, p, det)?;
    Ok((0..4).map(|k| s[k % 2 + 3 * (k / 2)]).collect())
  }

  fn grad_pk2(&self, e: &[f64], n: usize, p: &[f64]) -> OpResult<Vec<f64>> {
    if !self.plane_strain {
      return self.law.grad_sigma(e, n, p);
    }
    let g = self.law.grad_sigma(&embed(e), 3, p)?;
    let mut r = vec![0.0; 16];
    for l in 0..2 {
      for k in 0..2 {
        for j in 0..2 {
          for i in 0..2 {
            r[t4(2, i, j, k, l)] = g[t4(3, i, j, k, l)];
          }
        }
      }
    }
    Ok(r)
  }
}

/// a 2 x 2 strain in 3 x 3
fn embed(e: &[f64]) -> Vec<f64> {
  let mut e3 = vec![0.0; 9];
  for j in 0..2 {
    for i in 0..2 {
      e3[i + 3 * j] = e[i + 2 * j];
    }
  }
  e3
}

fn unavailable(what: &str) -> String {
  format!("{what} not yet available")
}

/// GetFEM's AHL_wrapper_potential and AHL_wrapper_sigma
impl NonlinearOperator for LawOp {
  fn result_shape(&self, args: &[&[usize]]) -> Option<Vec<usize>> {
    self.shape(args)
  }

  fn has_derivative(&self, der: (usize, usize)) -> bool {
    match self.kind {
      Kind::Potential => matches!(der, (1, 0) | (1, 1)),
      Kind::Pk2 => der == (1, 0),
    }
  }

  fn value(&self, args: &[&[f64]], shapes: &[&[usize]], out: &mut [f64]) -> OpResult<()> {
    out.copy_from_slice(&self.value_vec(args, shapes)?);
    Ok(())
  }

  fn derivative(
    &self,
    args: &[&[f64]],
    shapes: &[&[usize]],
    i: usize,
    out: &mut [f64],
  ) -> OpResult<()> {
    out.copy_from_slice(&self.derivative_vec(args, shapes, i)?);
    Ok(())
  }

  fn second_derivative(
    &self,
    args: &[&[f64]],
    shapes: &[&[usize]],
    i: usize,
    j: usize,
    out: &mut [f64],
  ) -> OpResult<()> {
    out.copy_from_slice(&self.second_derivative_vec(args, shapes, i, j)?);
    Ok(())
  }
}

/// the values, derivatives (the law allocates its temporaries, as GetFEM's
/// AHL wrappers)
impl LawOp {
  fn value_vec(&self, args: &[&[f64]], shapes: &[&[usize]]) -> OpResult<Vec<f64>> {
    let n = shapes[0][0];
    let (e, _, det) = kinematics(args[0], n);
    match self.kind {
      Kind::Potential => Ok(vec![self.energy(&e, n, args[1], det)?]),
      Kind::Pk2 => self.pk2(&e, n, args[1], det),
    }
  }
  fn derivative_vec(&self, args: &[&[f64]], shapes: &[&[usize]], i: usize) -> OpResult<Vec<f64>> {
    if i != 0 {
      return Err(unavailable(
        "derivative of a hyperelastic law w.r.t. its parameters",
      ));
    }
    let n = shapes[0][0];
    let (e, f, det) = kinematics(args[0], n);
    let fm = |r: usize, c: usize| f[r + n * c];
    match self.kind {
      // F sigma
      Kind::Potential => {
        let s = self.pk2(&e, n, args[1], det)?;
        let mut r = vec![0.0; n * n];
        for j in 0..n {
          for a in 0..n {
            for k in 0..n {
              r[a + n * j] += fm(a, k) * s[k + n * j];
            }
          }
        }
        Ok(r)
      }
      // grad_sigma(i,j,m,l) F(k,m)
      Kind::Pk2 => {
        let g = self.grad_pk2(&e, n, args[1])?;
        let mut r = vec![0.0; n * n * n * n];
        for l in 0..n {
          for k in 0..n {
            for j in 0..n {
              for a in 0..n {
                let mut v = 0.0;
                for m in 0..n {
                  v += g[t4(n, a, j, m, l)] * fm(k, m);
                }
                r[t4(n, a, j, k, l)] = v;
              }
            }
          }
        }
        Ok(r)
      }
    }
  }
  fn second_derivative_vec(
    &self,
    args: &[&[f64]],
    shapes: &[&[usize]],
    i: usize,
    j: usize,
  ) -> OpResult<Vec<f64>> {
    if !self.has_derivative((i + 1, j + 1)) {
      return Err(unavailable("this second derivative of a hyperelastic law"));
    }
    let n = shapes[0][0];
    let (e, f, det) = kinematics(args[0], n);
    let fm = |r: usize, c: usize| f[r + n * c];
    let s = self.pk2(&e, n, args[1], det)?;
    let g = self.grad_pk2(&e, n, args[1])?;
    let mut r = vec![0.0; n * n * n * n];
    for l in 0..n {
      for k in 0..n {
        for jj in 0..n {
          for a in 0..n {
            let mut v = 0.0;
            if a == k {
              v += s[l + n * jj];
            }
            for m in 0..n {
              for q in 0..n {
                v += g[t4(n, q, jj, m, l)] * fm(k, m) * fm(a, q);
              }
            }
            r[t4(n, a, jj, k, l)] = v;
          }
        }
      }
    }
    Ok(r)
  }
}

/// GetFEM's Saint_Venant_Kirchhoff_sigma: lambda Tr(E) Id + 2 mu E, with
/// its derivative w.r.t. the parameters
#[derive(Clone, Copy, Debug)]
struct SvkSigma;

impl NonlinearOperator for SvkSigma {
  fn result_shape(&self, args: &[&[usize]]) -> Option<Vec<usize>> {
    let [g, p] = args else { return None };
    let n = *g.first()?;
    (g.len() == 2 && g[1] == n && p.iter().product::<usize>() == 2).then(|| vec![n, n])
  }

  fn has_derivative(&self, der: (usize, usize)) -> bool {
    der.1 == 0
  }

  fn value(&self, args: &[&[f64]], shapes: &[&[usize]], out: &mut [f64]) -> OpResult<()> {
    out.copy_from_slice(&self.value_vec(args, shapes)?);
    Ok(())
  }

  fn derivative(
    &self,
    args: &[&[f64]],
    shapes: &[&[usize]],
    i: usize,
    out: &mut [f64],
  ) -> OpResult<()> {
    out.copy_from_slice(&self.derivative_vec(args, shapes, i)?);
    Ok(())
  }

  fn second_derivative(
    &self,
    args: &[&[f64]],
    shapes: &[&[usize]],
    i: usize,
    j: usize,
    out: &mut [f64],
  ) -> OpResult<()> {
    out.copy_from_slice(&self.second_derivative_vec(args, shapes, i, j)?);
    Ok(())
  }
}

/// the values, derivatives (the law allocates its temporaries, as GetFEM's
/// AHL wrappers)
impl SvkSigma {
  fn value_vec(&self, args: &[&[f64]], shapes: &[&[usize]]) -> OpResult<Vec<f64>> {
    let n = shapes[0][0];
    let (lambda, mu) = (args[1][0], args[1][1]);
    let (e, _, _) = kinematics(args[0], n);
    let tre = trace(&e, n);
    let mut r = vec![0.0; n * n];
    for j in 0..n {
      for i in 0..n {
        r[i + n * j] = 2.0 * mu * e[i + n * j];
        if i == j {
          r[i + n * j] += lambda * tre;
        }
      }
    }
    Ok(r)
  }
  fn derivative_vec(
    &self,
    args: &[&[f64]],
    shapes: &[&[usize]],
    nder: usize,
  ) -> OpResult<Vec<f64>> {
    let n = shapes[0][0];
    let (lambda, mu) = (args[1][0], args[1][1]);
    let gu = args[0];
    let u = |r: usize, c: usize| gu[r + n * c];
    if nder == 0 {
      let mut r = vec![0.0; n * n * n * n];
      for l in 0..n {
        for k in 0..n {
          for j in 0..n {
            for i in 0..n {
              let mut v = 0.0;
              if i == j && k == l {
                v += lambda;
              }
              if i == j {
                v += lambda * u(k, l);
              }
              if i == k && j == l {
                v += mu;
              }
              if i == l && j == k {
                v += mu;
              }
              if i == l {
                v += mu * u(k, j);
              }
              if l == j {
                v += mu * u(k, i);
              }
              r[t4(n, i, j, k, l)] = v;
            }
          }
        }
      }
      return Ok(r);
    }
    // w.r.t. lambda: Tr(E) Id, w.r.t. mu: 2 E
    let (e, _, _) = kinematics(gu, n);
    let tre = trace(&e, n);
    let mut r = vec![0.0; 2 * n * n];
    for j in 0..n {
      for i in 0..n {
        if i == j {
          r[i + n * j] = tre;
        }
        r[n * n + i + n * j] = 2.0 * e[i + n * j];
      }
    }
    Ok(r)
  }
  fn second_derivative_vec(
    &self,
    _: &[&[f64]],
    _: &[&[usize]],
    _: usize,
    _: usize,
  ) -> OpResult<Vec<f64>> {
    Err(unavailable(
      "Saint_Venant_Kirchhoff_sigma: second derivative",
    ))
  }
}

const LAWS: &[(&str, Law)] = &[
  ("Saint_Venant_Kirchhoff", Law::SaintVenantKirchhoff),
  ("Generalized_Blatz_Ko", Law::GeneralizedBlatzKo),
  ("Ciarlet_Geymonat", Law::CiarletGeymonat),
  (
    "Incompressible_Mooney_Rivlin",
    Law::MooneyRivlin {
      compressible: false,
      neohookean: false,
    },
  ),
  (
    "Compressible_Mooney_Rivlin",
    Law::MooneyRivlin {
      compressible: true,
      neohookean: false,
    },
  ),
  (
    "Incompressible_Neo_Hookean",
    Law::MooneyRivlin {
      compressible: false,
      neohookean: true,
    },
  ),
  (
    "Compressible_Neo_Hookean",
    Law::MooneyRivlin {
      compressible: true,
      neohookean: true,
    },
  ),
  (
    "Compressible_Neo_Hookean_Bonet",
    Law::NeoHookean { bonet: true },
  ),
  (
    "Compressible_Neo_Hookean_Ciarlet",
    Law::NeoHookean { bonet: false },
  ),
];

pub(crate) fn register(ops: &mut Operators) {
  for &(name, law) in LAWS {
    for plane_strain in [false, true] {
      let pre = if plane_strain { "Plane_Strain_" } else { "" };
      for (suffix, kind) in [
        ("potential", Kind::Potential),
        ("PK2", Kind::Pk2),
        ("sigma", Kind::Pk2),
      ] {
        let name = format!("{pre}{name}_{suffix}");
        // GetFEM: SVK's Plane_Strain_ names are the plain operators (any
        // dimension; in 2D, plane strain is the 2D law)
        let plane_strain = plane_strain && !matches!(law, Law::SaintVenantKirchhoff);
        // GetFEM: a dedicated operator for the SVK stress
        if matches!((law, kind), (Law::SaintVenantKirchhoff, Kind::Pk2)) {
          ops.insert(&name, Arc::new(SvkSigma));
          continue;
        }
        ops.insert(
          &name,
          Arc::new(LawOp {
            law,
            plane_strain,
            kind,
          }),
        );
      }
    }
  }
}
