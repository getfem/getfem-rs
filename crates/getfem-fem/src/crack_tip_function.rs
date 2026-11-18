//! Functions of the crack tip coordinates (GetFEM's xy_function; Python
//! GlobalFunction): f(x, y) with x along the crack beyond the tip, y normal
//! to it (the secondary and primary level sets), r and theta the polar
//! coordinates around the tip. The singular functions of the crack tip
//! asymptotics, cutoffs, their products and sums.

use std::sync::Arc;

use crate::{Error, Result};

/// A function of the crack tip coordinates: value, gradient, Hessian
/// (column-major) in (x, y).
pub trait CrackTipFunctionImpl: Send + Sync {
  fn val(&self, x: f64, y: f64) -> f64;
  fn grad(&self, x: f64, y: f64) -> [f64; 2];
  fn hess(&self, x: f64, y: f64) -> [f64; 4];
}

#[derive(Clone)]
pub struct CrackTipFunction(Arc<dyn CrackTipFunctionImpl>);

impl std::fmt::Debug for CrackTipFunction {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "CrackTipFunction")
  }
}

/// The singular functions of crack_singular_xy_function, with s =
/// sin(theta/2), c = cos(theta/2) (GetFEM's index l in brackets; its
/// 10, 11, 13 repeat 0, 1, 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Singular {
  /// sqrt(r) s (0)
  SqrtRSin,
  /// sqrt(r) c (1)
  SqrtRCos,
  /// sqrt(r) s sin(theta) (2)
  SqrtRSinSinTheta,
  /// sqrt(r) c sin(theta) (3)
  SqrtRCosSinTheta,
  /// r^(3/2) s (4)
  R32Sin,
  /// r^(3/2) c (5)
  R32Cos,
  /// r^(3/2) s c^2 (6)
  R32SinCos2,
  /// r^(3/2) c^3 (7)
  R32Cos3,
  /// -s / sqrt(r) (8)
  MinusInvSqrtRSin,
  /// c / sqrt(r) (9)
  InvSqrtRCos,
  /// r s^2 (12)
  RSin2,
  /// s / r (14)
  InvRSin,
  /// c / r (15)
  InvRCos,
  /// sqrt(r) (not GetFEM's: with angular hats, see angular_hat)
  SqrtR,
}

impl Singular {
  /// The displacement enrichment of linear elastic fracture mechanics
  /// (GetFEM's 0 to 3).
  pub const LEFM: [Singular; 4] = [
    Singular::SqrtRSin,
    Singular::SqrtRCos,
    Singular::SqrtRSinSinTheta,
    Singular::SqrtRCosSinTheta,
  ];

  /// With the second order terms (GetFEM's 0 to 7): the radial profile
  /// of the jump A sqrt(r) + B r^(3/2).
  pub const LEFM_SECOND_ORDER: [Singular; 8] = [
    Singular::SqrtRSin,
    Singular::SqrtRCos,
    Singular::SqrtRSinSinTheta,
    Singular::SqrtRCosSinTheta,
    Singular::R32Sin,
    Singular::R32Cos,
    Singular::R32SinCos2,
    Singular::R32Cos3,
  ];

  /// r^a P(s, c): a and the monomials (coefficient, power of s, of c)
  fn parts(self) -> (f64, Vec<(f64, i32, i32)>) {
    use Singular::*;
    match self {
      SqrtRSin => (0.5, vec![(1.0, 1, 0)]),
      SqrtRCos => (0.5, vec![(1.0, 0, 1)]),
      // sin(theta) = 2 s c
      SqrtRSinSinTheta => (0.5, vec![(2.0, 2, 1)]),
      SqrtRCosSinTheta => (0.5, vec![(2.0, 1, 2)]),
      R32Sin => (1.5, vec![(1.0, 1, 0)]),
      R32Cos => (1.5, vec![(1.0, 0, 1)]),
      R32SinCos2 => (1.5, vec![(1.0, 1, 2)]),
      R32Cos3 => (1.5, vec![(1.0, 0, 3)]),
      MinusInvSqrtRSin => (-0.5, vec![(-1.0, 1, 0)]),
      InvSqrtRCos => (-0.5, vec![(1.0, 0, 1)]),
      RSin2 => (1.0, vec![(1.0, 2, 0)]),
      InvRSin => (-1.0, vec![(1.0, 1, 0)]),
      InvRCos => (-1.0, vec![(1.0, 0, 1)]),
      SqrtR => (0.5, vec![(1.0, 0, 0)]),
    }
  }
}

/// The cutoffs of cutoff_xy_function, functions of the distance rho to
/// the tip.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Cutoff {
  /// 1
  None,
  /// exp(-(2.7 rho / radius)^4)
  Exponential { radius: f64 },
  /// 1 for rho <= inner, 0 for rho >= outer, cubic (C1) in between
  Polynomial { inner: f64, outer: f64 },
  /// as Polynomial, quintic (C2) in between
  Polynomial2 { inner: f64, outer: f64 },
  /// a linear hat in rho: 0 at left, 1 at centre, 0 at right (left =
  /// centre: 1 below centre); hats on radial nodes 0, d, 2d, ... without
  /// one at the last node sum to 1 up to the second last node and fall to
  /// 0 at the last (a radial grid around a crack front)
  Hat { left: f64, centre: f64, right: f64 },
}

/// d/dtheta of sum k s^i c^j: s' = c/2, c' = -s/2
fn dtheta(p: &[(f64, i32, i32)]) -> Vec<(f64, i32, i32)> {
  let mut d = vec![];
  for &(k, i, j) in p {
    if i > 0 {
      d.push((0.5 * k * i as f64, i - 1, j + 1));
    }
    if j > 0 {
      d.push((-0.5 * k * j as f64, i + 1, j - 1));
    }
  }
  d
}

fn eval(p: &[(f64, i32, i32)], s: f64, c: f64) -> f64 {
  p.iter().map(|&(k, i, j)| k * s.powi(i) * c.powi(j)).sum()
}

struct SingularFn {
  a: f64,
  /// P, P', P''
  p: [Vec<(f64, i32, i32)>; 3],
}

impl SingularFn {
  /// r, s, c, the unit vectors n (radial), t (tangential) at (x, y); s
  /// has the sign of y, positive on y = 0
  fn polar(x: f64, y: f64) -> (f64, f64, f64, [f64; 2], [f64; 2]) {
    let r = (x * x + y * y).sqrt();
    let sgn = if y < 0.0 { -1.0 } else { 1.0 };
    // abs: no sqrt of -1e-16
    let s = (0.5 - x / (2.0 * r)).abs().sqrt() * sgn;
    let c = (0.5 + x / (2.0 * r)).abs().sqrt();
    let (ct, st) = (c * c - s * s, 2.0 * s * c);
    (r, s, c, [ct, st], [-st, ct])
  }

  /// f_r, f_theta / r, and the second derivatives f_rr, f_rtheta, f_thetatheta
  fn ders(&self, r: f64, s: f64, c: f64) -> [f64; 5] {
    let a = self.a;
    let g: [f64; 3] = std::array::from_fn(|k| eval(&self.p[k], s, c));
    let ra = r.powf(a);
    [
      a * ra / r * g[0],
      ra / r * g[1],
      a * (a - 1.0) * ra / (r * r) * g[0],
      a * ra / r * g[1],
      ra * g[2],
    ]
  }
}

impl CrackTipFunctionImpl for SingularFn {
  fn val(&self, x: f64, y: f64) -> f64 {
    let (r, s, c, _, _) = Self::polar(x, y);
    if r < 1e-10 {
      return 0.0;
    }
    r.powf(self.a) * eval(&self.p[0], s, c)
  }
  fn grad(&self, x: f64, y: f64) -> [f64; 2] {
    let (r, s, c, n, t) = Self::polar(x, y);
    let [fr, ft, ..] = self.ders(r, s, c);
    [fr * n[0] + ft * t[0], fr * n[1] + ft * t[1]]
  }
  fn hess(&self, x: f64, y: f64) -> [f64; 4] {
    let (r, s, c, n, t) = Self::polar(x, y);
    let [fr, ft, frr, frt, ftt] = self.ders(r, s, c);
    // polar components: rr, r theta, theta theta
    let (hrr, hrt, htt) = (frr, frt / r - ft / r, ftt / (r * r) + fr / r);
    let h = |i: usize, j: usize| {
      hrr * n[i] * n[j] + hrt * (n[i] * t[j] + t[i] * n[j]) + htt * t[i] * t[j]
    };
    [h(0, 0), h(1, 0), h(0, 1), h(1, 1)]
  }
}

/// f(rho), f'(rho), f''(rho)
struct Radial(Cutoff);

impl Radial {
  fn f(&self, rho: f64) -> [f64; 3] {
    match self.0 {
      Cutoff::None => [1.0, 0.0, 0.0],
      Cutoff::Exponential { radius } => {
        let a4 = (2.7 / radius).powi(4);
        let e = (-a4 * rho.powi(4)).exp();
        let d = -4.0 * a4 * rho.powi(3);
        [e, d * e, (d * d - 12.0 * a4 * rho * rho) * e]
      }
      Cutoff::Polynomial { inner, outer } => {
        let (r1, r0) = (inner, outer);
        if rho <= r1 {
          [1.0, 0.0, 0.0]
        } else if rho >= r0 {
          [0.0, 0.0, 0.0]
        } else {
          let c = 1.0 / (r0 - r1).powi(3);
          [
            c * (rho * (rho * (2.0 * rho - 3.0 * (r0 + r1)) + 6.0 * r1 * r0)
              + r0 * r0 * (r0 - 3.0 * r1)),
            6.0 * c * (rho - r0) * (rho - r1),
            6.0 * c * (2.0 * rho - r0 - r1),
          ]
        }
      }
      Cutoff::Hat {
        left,
        centre,
        right,
      } => {
        if rho <= left || rho >= right {
          [
            if rho <= left && left == centre {
              1.0
            } else {
              0.0
            },
            0.0,
            0.0,
          ]
        } else if rho <= centre {
          [(rho - left) / (centre - left), 1.0 / (centre - left), 0.0]
        } else {
          [
            (right - rho) / (right - centre),
            -1.0 / (right - centre),
            0.0,
          ]
        }
      }
      Cutoff::Polynomial2 { inner, outer } => {
        let (r1, r0) = (inner, outer);
        if rho <= r1 {
          [1.0, 0.0, 0.0]
        } else if rho >= r0 {
          [0.0, 0.0, 0.0]
        } else {
          let c = 1.0 / (r0 - r1).powi(5);
          let r = rho;
          [
            c * (r
              * (r
                * (r
                  * (r * (-6.0 * r + 15.0 * (r0 + r1))
                    - 10.0 * (r0 * r0 + 4.0 * r1 * r0 + r1 * r1))
                  + 30.0 * r0 * r1 * (r0 + r1))
                - 30.0 * r1 * r1 * r0 * r0)
              + r0 * r0 * r0 * (r0 * r0 - 5.0 * r1 * r0 + 10.0 * r1 * r1)),
            -30.0 * c * (r - r0).powi(2) * (r - r1).powi(2),
            60.0 * c * (r1 - r) * (r0 - r) * (r0 + r1 - 2.0 * r),
          ]
        }
      }
    }
  }
}

/// Hat m of the linear hats in theta = atan2(y, x) on n sectors:
/// periodic, nodes at 2 pi k / n (k < n); or on [-pi, pi], nodes at
/// -pi + 2 pi k / n (k <= n), the end hats on the two lips.
struct AngularHat {
  n: usize,
  m: usize,
  periodic: bool,
}

impl AngularHat {
  /// the hat and its derivative in theta
  fn f(&self, t: f64) -> [f64; 2] {
    let pi = std::f64::consts::PI;
    let d = 2.0 * pi / self.n as f64;
    // the angle from the node (periodic: in (-pi, pi])
    let mut u = if self.periodic {
      t - self.m as f64 * d
    } else {
      t - (-pi + self.m as f64 * d)
    };
    if self.periodic {
      u -= 2.0 * pi * (u / (2.0 * pi)).round();
    }
    if u.abs() >= d {
      [0.0, 0.0]
    } else {
      [1.0 - u.abs() / d, -u.signum() / d]
    }
  }
}

impl CrackTipFunctionImpl for AngularHat {
  fn val(&self, x: f64, y: f64) -> f64 {
    self.f(y.atan2(x))[0]
  }
  /// d theta / dx = -y / r^2, d theta / dy = x / r^2
  fn grad(&self, x: f64, y: f64) -> [f64; 2] {
    let r2 = x * x + y * y;
    if r2 == 0.0 {
      return [0.0; 2];
    }
    let d = self.f(y.atan2(x))[1];
    [-d * y / r2, d * x / r2]
  }
  /// f' times the Hessian of theta (f'' = 0 between the nodes)
  fn hess(&self, x: f64, y: f64) -> [f64; 4] {
    let r2 = x * x + y * y;
    if r2 == 0.0 {
      return [0.0; 4];
    }
    let d = self.f(y.atan2(x))[1];
    let r4 = r2 * r2;
    let (hxx, hyy, hxy) = (2.0 * x * y / r4, -2.0 * x * y / r4, (y * y - x * x) / r4);
    [d * hxx, d * hxy, d * hxy, d * hyy]
  }
}

impl CrackTipFunctionImpl for Radial {
  fn val(&self, x: f64, y: f64) -> f64 {
    self.f((x * x + y * y).sqrt())[0]
  }
  fn grad(&self, x: f64, y: f64) -> [f64; 2] {
    let rho = (x * x + y * y).sqrt();
    if rho == 0.0 {
      return [0.0; 2];
    }
    let d = self.f(rho)[1] / rho;
    [d * x, d * y]
  }
  fn hess(&self, x: f64, y: f64) -> [f64; 4] {
    let rho = (x * x + y * y).sqrt();
    if rho == 0.0 {
      return [0.0; 4];
    }
    let [_, d1, d2] = self.f(rho);
    let n = [x / rho, y / rho];
    let h = |i: usize, j: usize| {
      let id = if i == j { 1.0 } else { 0.0 };
      d2 * n[i] * n[j] + d1 / rho * (id - n[i] * n[j])
    };
    [h(0, 0), h(1, 0), h(0, 1), h(1, 1)]
  }
}

struct Product(CrackTipFunction, CrackTipFunction);

impl CrackTipFunctionImpl for Product {
  fn val(&self, x: f64, y: f64) -> f64 {
    self.0.val(x, y) * self.1.val(x, y)
  }
  fn grad(&self, x: f64, y: f64) -> [f64; 2] {
    let (f, g) = (self.0.val(x, y), self.1.val(x, y));
    let (df, dg) = (self.0.grad(x, y), self.1.grad(x, y));
    [df[0] * g + f * dg[0], df[1] * g + f * dg[1]]
  }
  fn hess(&self, x: f64, y: f64) -> [f64; 4] {
    let (f, g) = (self.0.val(x, y), self.1.val(x, y));
    let (df, dg) = (self.0.grad(x, y), self.1.grad(x, y));
    let (hf, hg) = (self.0.hess(x, y), self.1.hess(x, y));
    std::array::from_fn(|k| {
      let (i, j) = (k % 2, k / 2);
      hf[k] * g + df[i] * dg[j] + dg[i] * df[j] + f * hg[k]
    })
  }
}

struct Sum(Vec<CrackTipFunction>);

impl CrackTipFunctionImpl for Sum {
  fn val(&self, x: f64, y: f64) -> f64 {
    self.0.iter().map(|f| f.val(x, y)).sum()
  }
  fn grad(&self, x: f64, y: f64) -> [f64; 2] {
    self.0.iter().fold([0.0; 2], |a, f| {
      let g = f.grad(x, y);
      [a[0] + g[0], a[1] + g[1]]
    })
  }
  fn hess(&self, x: f64, y: f64) -> [f64; 4] {
    self.0.iter().fold([0.0; 4], |a, f| {
      let h = f.hess(x, y);
      std::array::from_fn(|k| a[k] + h[k])
    })
  }
}

impl CrackTipFunction {
  /// A user's implementation.
  pub fn new(f: impl CrackTipFunctionImpl + 'static) -> Self {
    CrackTipFunction(Arc::new(f))
  }
  pub fn val(&self, x: f64, y: f64) -> f64 {
    self.0.val(x, y)
  }
  pub fn grad(&self, x: f64, y: f64) -> [f64; 2] {
    self.0.grad(x, y)
  }
  pub fn hess(&self, x: f64, y: f64) -> [f64; 4] {
    self.0.hess(x, y)
  }

  /// A singular function of the crack tip asymptotics (0 within 1e-10 of
  /// the tip).
  pub fn crack(f: Singular) -> Self {
    let (a, p0) = f.parts();
    let p1 = dtheta(&p0);
    let p2 = dtheta(&p1);
    CrackTipFunction::new(SingularFn { a, p: [p0, p1, p2] })
  }

  /// A cutoff around the tip.
  pub fn cutoff(c: Cutoff) -> Result<Self> {
    match c {
      Cutoff::Exponential { radius } if radius.is_nan() || radius <= 0.0 => Err(
        Error::InvalidArgument("exponential cutoff: positive radius expected".into()),
      ),
      Cutoff::Polynomial { inner, outer } | Cutoff::Polynomial2 { inner, outer }
        if inner.is_nan() || outer.is_nan() || inner < 0.0 || inner >= outer =>
      {
        Err(Error::InvalidArgument(
          "polynomial cutoff: 0 <= inner < outer expected".into(),
        ))
      }
      Cutoff::Hat {
        left,
        centre,
        right,
      } if !(left >= 0.0 && left <= centre && centre < right) => Err(Error::InvalidArgument(
        "hat cutoff: 0 <= left <= centre < right expected".into(),
      )),
      _ => Ok(CrackTipFunction::new(Radial(c))),
    }
  }

  /// Hat m of the linear hats in theta = atan2(y, x) on n sectors (theta
  /// = +-pi on the crack): an angular grid around a crack front, a
  /// partition of unity in theta. Periodic: n hats, nodes at 2 pi k / n;
  /// otherwise n + 1 hats on [-pi, pi], the end hats on the two lips
  /// (discontinuous across the crack).
  pub fn angular_hat(n: usize, m: usize, periodic: bool) -> Result<Self> {
    if n < 3 || m >= n + usize::from(!periodic) {
      return Err(Error::InvalidArgument(format!(
        "angular hat {m} of {n} sectors"
      )));
    }
    Ok(CrackTipFunction::new(AngularHat { n, m, periodic }))
  }

  /// f g.
  pub fn product(f: &CrackTipFunction, g: &CrackTipFunction) -> Self {
    CrackTipFunction::new(Product(f.clone(), g.clone()))
  }

  /// The sum of the functions.
  pub fn sum(funcs: &[CrackTipFunction]) -> Self {
    CrackTipFunction::new(Sum(funcs.to_vec()))
  }
}
