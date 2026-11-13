//! Nonlinear operators (GetFEM ga_nonlinear_operator): a trait and a registry
//! owned by the definitions. The built-in operators are GetFEM's, with its
//! hand-written derivatives; a derivative GetFEM does not have is reported
//! as not yet available (`has_derivative`).
//! Tensors are column-major; the derivative w.r.t. argument i has the result
//! indices first, then those of argument i.

use std::collections::BTreeMap;
use std::sync::Arc;

use getfem_geom::geotrans::{lu_det, lu_inverse};
use getfem_linalg::{vect_norm2, vect_norm2_sqr};

pub type OpResult<T> = std::result::Result<T, String>;

pub trait NonlinearOperator: Send + Sync {
  /// shape of the result, None if the arguments do not fit
  fn result_shape(&self, args: &[&[usize]]) -> Option<Vec<usize>>;
  /// the value into out (sized by result_shape; GetFEM: value(args, result))
  fn value(&self, args: &[&[f64]], shapes: &[&[usize]], out: &mut [f64]) -> OpResult<()>;
  /// derivative w.r.t. argument i (0-based): result x argument i
  fn derivative(
    &self,
    args: &[&[f64]],
    shapes: &[&[usize]],
    i: usize,
    out: &mut [f64],
  ) -> OpResult<()>;
  /// second derivative w.r.t. arguments i, j: result x argument i x argument j
  fn second_derivative(
    &self,
    args: &[&[f64]],
    shapes: &[&[usize]],
    i: usize,
    j: usize,
    out: &mut [f64],
  ) -> OpResult<()>;
  /// whether the derivative der = (argument, 0) or the second derivative
  /// (argument, argument) is available (1-based, as the derivation marks it)
  fn has_derivative(&self, _der: (usize, usize)) -> bool {
    true
  }
}

/// Operators by name.
#[derive(Clone)]
pub struct Operators(BTreeMap<String, Arc<dyn NonlinearOperator>>);

impl Default for Operators {
  fn default() -> Self {
    let mut m: BTreeMap<String, Arc<dyn NonlinearOperator>> = BTreeMap::new();
    for b in BUILTINS {
      m.insert(b.name().to_string(), Arc::new(*b));
    }
    let mut ops = Operators(m);
    crate::laws::register(&mut ops);
    ops
  }
}

impl std::fmt::Debug for Operators {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_list().entries(self.0.keys()).finish()
  }
}

impl Operators {
  pub fn get(&self, name: &str) -> Option<&Arc<dyn NonlinearOperator>> {
    self.0.get(name)
  }

  pub fn contains(&self, name: &str) -> bool {
    self.0.contains_key(name)
  }

  pub fn insert(&mut self, name: &str, op: Arc<dyn NonlinearOperator>) {
    self.0.insert(name.to_string(), op);
  }
}

// ------------------------------------------------------ f64 helpers (gmm)

fn trace_f(a: &[f64], n: usize) -> f64 {
  (0..n).fold(0.0, |s, i| s + a[i * (n + 1)])
}

/// c = a b, n x n column-major
fn mult(a: &[f64], b: &[f64], n: usize) -> Small {
  let mut c = Small::filled(n * n, 0.0);
  for j in 0..n {
    for k in 0..n {
      let bkj = b[k + n * j];
      for i in 0..n {
        c[i + n * j] += a[i + n * k] * bkj;
      }
    }
  }
  c
}

/// gmm::vect_norm2_sqr, gmm::vect_norm2
/// a small matrix or tensor, on the stack up to C entries (3 x 3 by default;
/// no allocation per point, as GetFEM's static work matrices)
#[derive(Clone)]
enum Small<const C: usize = 9, T: Copy = f64> {
  Stack([T; C], usize),
  Heap(Vec<T>),
}

impl<const C: usize, T: Copy> Small<C, T> {
  fn filled(len: usize, v: T) -> Self {
    if len <= C {
      Small::Stack([v; C], len)
    } else {
      Small::Heap(vec![v; len])
    }
  }
}

impl<const C: usize, T: Copy> From<&[T]> for Small<C, T> {
  fn from(a: &[T]) -> Self {
    match a.first() {
      Some(&v) if a.len() <= C => {
        let mut s = [v; C];
        s[..a.len()].copy_from_slice(a);
        Small::Stack(s, a.len())
      }
      _ => Small::Heap(a.to_vec()),
    }
  }
}

impl<const C: usize, T: Copy> std::ops::Deref for Small<C, T> {
  type Target = [T];
  fn deref(&self) -> &[T] {
    match self {
      Small::Stack(s, l) => &s[..*l],
      Small::Heap(v) => v,
    }
  }
}

impl<const C: usize, T: Copy> std::ops::DerefMut for Small<C, T> {
  fn deref_mut(&mut self) -> &mut [T] {
    match self {
      Small::Stack(s, l) => &mut s[..*l],
      Small::Heap(v) => v,
    }
  }
}

/// out = c t
fn scaled(t: &[f64], c: f64, out: &mut [f64]) {
  for (o, v) in out.iter_mut().zip(t) {
    *o = v * c;
  }
}

/// gmm::mat_norminf: largest row sum of absolute values
fn mat_norminf(a: &[f64], n: usize) -> f64 {
  (0..n)
    .map(|i| (0..n).fold(0.0, |s, j| s + a[i + n * j].abs()))
    .fold(0.0, f64::max)
}

/// exponent of C's frexp (x = m 2^e, 0.5 <= |m| < 1)
fn frexp_exp(x: f64) -> i32 {
  if x == 0.0 || !x.is_finite() {
    return 0;
  }
  let b = ((x.to_bits() >> 52) & 0x7ff) as i32;
  if b == 0 {
    // subnormal
    return frexp_exp(x * 2f64.powi(64)) - 64;
  }
  b - 1022
}

fn inverse_of(a: &[f64], n: usize) -> OpResult<(f64, Small)> {
  let mut m = Small::from(a);
  let det = lu_inverse(&mut m, n).map_err(|_| "Non invertible matrix".to_string())?;
  Ok((det, m))
}

/// Pade approximants of exp (GetFEM expm, ported from Eigen/Unsupported):
/// the largest infinity norm for degrees 3, 5, 7, 9, their coefficients
const PADE: [(f64, &[f64]); 4] = [
  (1.495585217958292e-2, &[120., 60., 12., 1.]),
  (2.53939833006323e-1, &[30240., 15120., 3360., 420., 30., 1.]),
  (
    9.504178996162932e-1,
    &[
      17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.,
    ],
  ),
  (
    2.097847961257068,
    &[
      17643225600.,
      8821612800.,
      2075673600.,
      302702400.,
      30270240.,
      2162160.,
      110880.,
      3960.,
      90.,
      1.,
    ],
  ),
];
/// degree 13, after scaling by 2^-e to this norm
const PADE13_MAXNORM: f64 = 5.371920351148152;
const PADE13: [f64; 14] = [
  64764752532480000.,
  32382376266240000.,
  7771770303897600.,
  1187353796428800.,
  129060195264000.,
  10559470521600.,
  670442572800.,
  33522128640.,
  1323241920.,
  40840800.,
  960960.,
  16380.,
  182.,
  1.,
];

fn axpy(y: &mut [f64], c: f64, x: &[f64]) {
  y.iter_mut().zip(x).for_each(|(a, b)| *a += c * b);
}

fn add_diag(y: &mut [f64], c: f64, n: usize) {
  (0..n).for_each(|i| y[i * (n + 1)] += c);
}

/// The approximant (v + u)/(v - u) of exp(A 2^-e): the scaled A, e, the
/// coefficients, the even powers A2, A4, ... used, and for degree 13 the
/// inner sums v_, u_ (v = A6 v_ + ..., u' = A6 u_ + ...)
struct Pade {
  a: Small,
  e: i32,
  b: &'static [f64],
  pw: [Small; 4],
  /// the number of powers used
  np: usize,
  inner: Option<(Small, Small)>,
  v: Small,
  /// u before the product with A
  u1: Small,
}

fn zeros(n: usize) -> Small {
  Small::filled(n * n, 0.0)
}

/// a + c b
fn plus(a: &[f64], c: f64, b: &[f64]) -> Small {
  let mut r = Small::<9>::from(a);
  axpy(&mut r, c, b);
  r
}

impl Pade {
  fn new(a: &[f64], n: usize) -> Self {
    let norm = mat_norminf(a, n);
    let (mut a, mut e) = (Small::<9>::from(a), 0);
    let deg = PADE.iter().position(|&(m, _)| norm < m);
    let b: &'static [f64] = match deg {
      Some(d) => PADE[d].1,
      None => {
        e = frexp_exp(norm / PADE13_MAXNORM).max(0);
        a.iter_mut().for_each(|x| *x *= 2f64.powi(-e));
        &PADE13
      }
    };
    // A2, A4, A6 = A2 A4, A8 = A4 A4 (GetFEM's products)
    let np = if deg.is_some() { b.len() / 2 - 1 } else { 3 };
    let mut pw: [Small; 4] = std::array::from_fn(|_| zeros(n));
    pw[0] = mult(&a, &a, n);
    for k in 1..np {
      pw[k] = match k {
        1 => mult(&pw[0], &pw[0], n),
        2 => mult(&pw[0], &pw[1], n),
        _ => mult(&pw[1], &pw[1], n),
      };
    }
    let (mut v, mut u1) = (zeros(n), zeros(n));
    let mut inner = None;
    if deg.is_some() {
      for (k, p) in pw[..np].iter().enumerate() {
        axpy(&mut v, b[2 * k + 2], p);
        axpy(&mut u1, b[2 * k + 3], p);
      }
    } else {
      let (mut v_, mut u_) = (zeros(n), zeros(n));
      for (k, p) in pw[..np].iter().enumerate() {
        axpy(&mut v_, b[2 * k + 8], p);
        axpy(&mut u_, b[2 * k + 9], p);
        axpy(&mut v, b[2 * k + 2], p);
        axpy(&mut u1, b[2 * k + 3], p);
      }
      axpy(&mut v, 1.0, &mult(&pw[2], &v_, n));
      axpy(&mut u1, 1.0, &mult(&pw[2], &u_, n));
      inner = Some((v_, u_));
    }
    add_diag(&mut v, b[0], n);
    add_diag(&mut u1, b[1], n);
    Pade {
      a,
      e,
      b,
      pw,
      np,
      inner,
      v,
      u1,
    }
  }

  /// exp of the scaled A and (v - u)^-1
  fn exp(&self, n: usize) -> OpResult<(Small, Small)> {
    let u = mult(&self.a, &self.u1, n);
    let mut den = plus(&self.v, -1.0, &u);
    lu_inverse(&mut den, n).map_err(|_| "Expm: singular Pade denominator".to_string())?;
    let num = plus(&self.v, 1.0, &u);
    Ok((mult(&den, &num, n), den))
  }

  /// d/dA_kl of v and u (A scaled): GetFEM's product rule on the powers
  fn deriv(&self, n: usize, k: usize, l: usize) -> (Small, Small) {
    let (a, pw, b) = (&self.a, &self.pw, self.b);
    // dA2 = E A + A E, E = e_k e_l^T
    let mut dp: [Small; 4] = std::array::from_fn(|_| zeros(n));
    for j in 0..n {
      dp[0][k + n * j] += a[l + n * j];
      dp[0][j + n * l] += a[j + n * k];
    }
    for m in 1..self.np {
      let (x, y) = match m {
        1 => (mult(&pw[0], &dp[0], n), mult(&dp[0], &pw[0], n)),
        2 => (mult(&pw[0], &dp[1], n), mult(&dp[0], &pw[1], n)),
        _ => (mult(&pw[1], &dp[1], n), mult(&dp[1], &pw[1], n)),
      };
      dp[m] = plus(&x, 1.0, &y);
    }
    let dp = &dp[..self.np];
    let (mut dv, mut du1) = (zeros(n), zeros(n));
    for (m, d) in dp.iter().enumerate() {
      axpy(&mut dv, b[2 * m + 2], d);
      axpy(&mut du1, b[2 * m + 3], d);
    }
    if let Some((v_, u_)) = &self.inner {
      let (mut dv_, mut du_) = (zeros(n), zeros(n));
      for (m, d) in dp.iter().enumerate() {
        axpy(&mut dv_, b[2 * m + 8], d);
        axpy(&mut du_, b[2 * m + 9], d);
      }
      for (x, w, dw) in [(&mut dv, v_, &dv_), (&mut du1, u_, &du_)] {
        axpy(x, 1.0, &mult(&pw[2], dw, n));
        axpy(x, 1.0, &mult(&dp[2], w, n));
      }
    }
    // u = A u1: du = E u1 + A du1
    let mut du = mult(a, &du1, n);
    for j in 0..n {
      du[k + n * j] += self.u1[l + n * j];
    }
    (dv, du)
  }
}

/// GetFEM's expm (getfem_plasticity.cc): Pade approximant, then squarings
fn expm(a: &[f64], n: usize) -> OpResult<Small> {
  let p = Pade::new(a, n);
  let (mut x, _) = p.exp(n)?;
  for _ in 0..p.e {
    x = mult(&x, &x, n);
  }
  Ok(x)
}

/// GetFEM's expm_deriv: dexp(A)[i + N(j + N(k + N l))] = d exp(A)_ij / d A_kl
fn expm_deriv(a: &[f64], n: usize) -> OpResult<Small<81>> {
  let n2 = n * n;
  let p = Pade::new(a, n);
  let (mut x, den) = p.exp(n)?;
  let mut d = Small::<81>::filled(n2 * n2, 0.0);
  // d exp = D^-1 (dN - dD exp), N = v + u, D = v - u; times 2^-e (scaling)
  let s = 2f64.powi(-p.e);
  for l in 0..n {
    for k in 0..n {
      let (dv, du) = p.deriv(n, k, l);
      let dd = plus(&dv, -1.0, &du);
      let mut r = plus(&dv, 1.0, &du);
      axpy(&mut r, -1.0, &mult(&dd, &x, n));
      let blk = mult(&den, &r, n);
      let o = n2 * (k + n * l);
      d[o..o + n2]
        .iter_mut()
        .zip(blk.iter())
        .for_each(|(a, b)| *a = s * b);
    }
  }
  // unscale: d(X^2) = dX X + X dX
  for _ in 0..p.e {
    for kl in 0..n2 {
      let blk = &d[n2 * kl..n2 * (kl + 1)];
      let (q, r) = (mult(blk, &x, n), mult(&x, blk, n));
      for (o, (y, z)) in d[n2 * kl..n2 * (kl + 1)]
        .iter_mut()
        .zip(q.iter().zip(r.iter()))
      {
        *o = y + z;
      }
    }
    x = mult(&x, &x, n);
  }
  Ok(d)
}

/// The scalars of gmm's logm templates: f64, and complex for the matrices
/// with complex eigenvalues.
trait LogmScalar:
  Copy + std::ops::Add<Output = Self> + std::ops::Sub<Output = Self> + std::ops::Mul<Output = Self>
{
  fn real(x: f64) -> Self;
  fn re(self) -> f64;
  fn im(self) -> f64;
  fn abs(self) -> f64;
  fn sqrt(self) -> Self;
  /// gmm::safe_divide
  fn safe_divide(self, b: Self) -> Self;
}

impl LogmScalar for f64 {
  fn real(x: f64) -> Self {
    x
  }
  fn re(self) -> f64 {
    self
  }
  fn im(self) -> f64 {
    0.0
  }
  fn abs(self) -> f64 {
    f64::abs(self)
  }
  fn sqrt(self) -> Self {
    f64::sqrt(self)
  }
  fn safe_divide(self, b: Self) -> Self {
    self / b
  }
}

/// a complex number (std::complex<double> of gmm)
#[derive(Clone, Copy, Debug, PartialEq)]
struct Cplx {
  re: f64,
  im: f64,
}

impl std::ops::Add for Cplx {
  type Output = Cplx;
  fn add(self, b: Cplx) -> Cplx {
    Cplx {
      re: self.re + b.re,
      im: self.im + b.im,
    }
  }
}

impl std::ops::Sub for Cplx {
  type Output = Cplx;
  fn sub(self, b: Cplx) -> Cplx {
    Cplx {
      re: self.re - b.re,
      im: self.im - b.im,
    }
  }
}

impl std::ops::Mul for Cplx {
  type Output = Cplx;
  fn mul(self, b: Cplx) -> Cplx {
    Cplx {
      re: self.re * b.re - self.im * b.im,
      im: self.re * b.im + self.im * b.re,
    }
  }
}

impl Cplx {
  fn div(self, b: Cplx) -> Cplx {
    let d = b.re * b.re + b.im * b.im;
    Cplx {
      re: (self.re * b.re + self.im * b.im) / d,
      im: (self.im * b.re - self.re * b.im) / d,
    }
  }
  fn conj(self) -> Cplx {
    Cplx {
      re: self.re,
      im: -self.im,
    }
  }
  fn scale(self, c: f64) -> Cplx {
    Cplx {
      re: self.re * c,
      im: self.im * c,
    }
  }
}

impl LogmScalar for Cplx {
  fn real(x: f64) -> Self {
    Cplx { re: x, im: 0.0 }
  }
  fn re(self) -> f64 {
    self.re
  }
  fn im(self) -> f64 {
    self.im
  }
  fn abs(self) -> f64 {
    self.re.hypot(self.im)
  }
  /// principal root, as std::sqrt (the sign of the imaginary part kept)
  fn sqrt(self) -> Self {
    let r = self.abs();
    Cplx {
      re: ((r + self.re) / 2.0).sqrt(),
      im: ((r - self.re) / 2.0).sqrt().copysign(self.im),
    }
  }
  fn safe_divide(self, b: Self) -> Self {
    let m = b.re.abs().max(b.im.abs());
    self.scale(1.0 / m).div(b.scale(1.0 / m))
  }
}

/// gmm::sqrtm_utri_inplace: square root of an upper triangular matrix
/// (from GNU Octave), n x n column-major
fn sqrtm_utri_inplace<T: LogmScalar>(a: &mut [T], n: usize) {
  for j in 0..n {
    let ajj = a[j + n * j];
    if ajj.abs() > 0.0 {
      a[j + n * j] = ajj.sqrt();
    }
    for i in (0..j).rev() {
      let aji = a[i + n * j].safe_divide(a[i + n * i] + a[j + n * j]);
      a[i + n * j] = aji;
      for k in 0..i {
        a[k + n * j] = a[k + n * j] - a[k + n * i] * aji;
      }
    }
  }
}

/// gmm::logm_utri_inplace: logarithm of an upper triangular matrix,
/// inverse scaling and squaring with Gauss-Legendre Pade approximants
/// (Higham, Functions of matrices, algorithm 11.9)
fn logm_utri_inplace<T: LogmScalar>(s: &mut [T], n: usize) -> OpResult<()> {
  let tol = f64::EPSILON;
  for i in 0..n.saturating_sub(1) {
    if s[i + 1 + n * i].abs() > tol {
      return Err("Logm: an upper triangular matrix is expected".into());
    }
  }
  for i in 0..n.saturating_sub(1) {
    let d = s[i + n * i];
    if d.re() <= -tol && d.im().abs() <= tol {
      return Err(
        "Logm: principal matrix logarithm is not defined for matrices with negative eigenvalues"
          .into(),
      );
    }
  }
  let theta = [
    0.0,
    0.0,
    1.61e-2,
    5.38e-2,
    1.13e-1,
    1.86e-1,
    2.642960831111435e-1,
  ];
  let (mut scaling, mut p, mut m) = (1.0, 0, 6);
  let one = T::real(1.0);
  for _ in 0..100 {
    // tau = |S - I|_1, the largest column sum
    let tau = (0..n)
      .map(|j| {
        (0..n)
          .map(|i| {
            let v = s[i + n * j];
            if i == j { (v - one).abs() } else { v.abs() }
          })
          .sum::<f64>()
      })
      .fold(0.0, f64::max);
    if tau <= theta[6] {
      p += 1;
      let j1 = (0..6).find(|&j| tau <= theta[j]).unwrap_or(6);
      let j2 = (0..j1).find(|&j| tau <= 2.0 * theta[j]).unwrap_or(6);
      // gmm's unsigned j1 - j2 <= 1
      if (j2 <= j1 && j1 - j2 <= 1) || p == 2 {
        m = j1;
        break;
      }
    }
    sqrtm_utri_inplace(s, n);
    scaling *= 2.0;
  }
  for i in 0..n {
    s[i + n * i] = s[i + n * i] - one;
  }
  if m > 0 {
    let (nodes, wts): (&[f64], &[f64]) = match m {
      1 => (&[0.211324865405187, 0.788675134594813], &[0.5, 0.5]),
      2 => (
        &[0.112701665379258, 0.500000000000000, 0.887298334620742],
        &[0.277777777777778, 0.444444444444444, 0.277777777777778],
      ),
      3 => (
        &[
          0.0694318442029737,
          0.3300094782075718,
          0.6699905217924281,
          0.9305681557970263,
        ],
        &[
          0.173927422568727,
          0.326072577431273,
          0.326072577431273,
          0.173927422568727,
        ],
      ),
      4 => (
        &[
          0.0469100770306681,
          0.2307653449471584,
          0.5,
          0.7692346550528415,
          0.9530899229693319,
        ],
        &[
          0.118463442528095,
          0.239314335249683,
          0.284444444444444,
          0.239314335249683,
          0.118463442528094,
        ],
      ),
      5 => (
        &[
          0.0337652428984240,
          0.1693953067668678,
          0.3806904069584015,
          0.6193095930415985,
          0.8306046932331322,
          0.9662347571015761,
        ],
        &[
          0.0856622461895853,
          0.1803807865240693,
          0.2339569672863452,
          0.2339569672863459,
          0.1803807865240693,
          0.0856622461895852,
        ],
      ),
      _ => (
        &[
          0.0254460438286208,
          0.1292344072003028,
          0.2970774243113015,
          0.4999999999999999,
          0.7029225756886985,
          0.8707655927996973,
          0.9745539561713792,
        ],
        &[
          0.0647424830844348,
          0.1398526957446384,
          0.1909150252525594,
          0.2089795918367343,
          0.1909150252525595,
          0.1398526957446383,
          0.0647424830844349,
        ],
      ),
    };
    let aux1 = Small::<9, T>::from(&*s);
    s.iter_mut().for_each(|v| *v = T::real(0.0));
    let mut x = Small::<9, T>::filled(n, T::real(0.0));
    for (node, w) in nodes.iter().zip(wts) {
      // S += w aux1 (I + node aux1)^-1, row by row: x^T (I + node aux1) = row
      let aux2 = |i: usize, j: usize| {
        let v = aux1[i + n * j] * T::real(*node);
        if i == j { v + one } else { v }
      };
      for i in 0..n {
        for (k, xk) in x.iter_mut().enumerate() {
          *xk = aux1[i + n * k];
        }
        // forward substitution with the lower triangular aux2^T
        for j in 0..n {
          x[j] = x[j].safe_divide(aux2(j, j));
          for k in j + 1..n {
            x[k] = x[k] - x[j] * aux2(j, k);
          }
        }
        for k in 0..n {
          s[i + n * k] = s[i + n * k] + x[k] * T::real(*w);
        }
      }
    }
  }
  s.iter_mut().for_each(|v| *v = *v * T::real(scaling));
  Ok(())
}

/// gmm::block2x2_reduction (with complex scalars, as gmm::logm calls it):
/// each 2x2 diagonal block of the quasi-triangular H made triangular by the
/// unitary reflector P = I - 2 v v^H / |v|^2, H <- P H P, Q <- Q P. GetFEM
/// applies P only to the 2x2 block of H, which is not a similarity of the
/// whole matrix (Logm of a rotation: 0 instead of its generator).
fn block2x2_reduction(h: &mut [Cplx], q: &mut [Cplx], n: usize, tol: f64) {
  if n < 2 {
    return;
  }
  let tol = tol * 2.0;
  let mut i = 0;
  while i < n - 1 {
    let tol_i = (h[i + n * i].abs() + h[i + 1 + n * (i + 1)].abs()) * tol;
    if h[i + 1 + n * i].abs() > tol_i {
      let tr = (h[i + 1 + n * (i + 1)] - h[i + n * i]).scale(0.5);
      let delta = tr * tr + h[i + n * (i + 1)] * h[i + 1 + n * i];
      let theta = (tr - delta.sqrt()).div(h[i + 1 + n * i]);
      let a = theta.abs();
      let v1 = if a == 0.0 {
        Cplx::real(-1.0)
      } else {
        theta.conj().scale(1.0 - (a * a + 1.0).sqrt() / a)
      };
      let v = [Cplx::real(1.0), v1];
      let c = -2.0 / (1.0 + v1.abs() * v1.abs());
      // row_house_update on the rows i, i+1 of H: W = A^H V c, A += V W^H
      for col in 0..n {
        let w = (h[i + n * col].conj() * v[0] + h[i + 1 + n * col].conj() * v[1]).scale(c);
        for r in 0..2 {
          h[i + r + n * col] = h[i + r + n * col] + v[r] * w.conj();
        }
      }
      // col_house_update on the columns i, i+1 of H and of Q: W = A V c,
      // A += W V^H
      for m in [&mut *h, &mut *q] {
        for r in 0..n {
          let w = (m[r + n * i] * v[0] + m[r + n * (i + 1)] * v[1]).scale(c);
          m[r + n * i] = m[r + n * i] + w * v[0].conj();
          m[r + n * (i + 1)] = m[r + n * (i + 1)] + w * v[1].conj();
        }
      }
      i += 1;
    }
    i += 1;
  }
}

/// gmm::logm: principal logarithm by the Schur decomposition A = Q S Q^T,
/// then logm_utri_inplace on S, complex if S has 2x2 blocks
fn logm(a: &[f64], n: usize) -> OpResult<Small> {
  let (mut s, mut q) = (Small::<9>::from(&a[..n * n]), zeros(n));
  getfem_linalg::dense_qr::schur(&mut s, &mut q, n).map_err(|e| format!("Logm: {e}"))?;
  let tol = f64::EPSILON;
  let complex = (0..n.saturating_sub(1)).any(|i| s[i + 1 + n * i].abs() > tol);
  if complex {
    let mut cs: Vec<Cplx> = s.iter().map(|&v| Cplx::real(v)).collect();
    let mut cq: Vec<Cplx> = q.iter().map(|&v| Cplx::real(v)).collect();
    block2x2_reduction(&mut cs, &mut cq, n, tol * 3.0);
    for j in 0..n.saturating_sub(1) {
      for i in j + 1..n {
        cs[i + n * j] = Cplx::real(0.0);
      }
    }
    logm_utri_inplace(&mut cs, n)?;
    // Q S Q^H (gmm: Q^T, wrong for the unitary Q of this branch); real part
    let mut qs = vec![Cplx::real(0.0); n * n];
    for j in 0..n {
      for k in 0..n {
        let skj = cs[k + n * j];
        for i in 0..n {
          qs[i + n * j] = qs[i + n * j] + cq[i + n * k] * skj;
        }
      }
    }
    let mut out = zeros(n);
    for j in 0..n {
      for i in 0..n {
        let mut v = Cplx::real(0.0);
        for k in 0..n {
          v = v + qs[i + n * k] * cq[j + n * k].conj();
        }
        out[i + n * j] = v.re;
      }
    }
    Ok(out)
  } else {
    logm_utri_inplace(&mut s, n)?;
    // Q S Q^T
    let qs = mult(&q, &s, n);
    let mut out = zeros(n);
    for j in 0..n {
      for i in 0..n {
        out[i + n * j] = (0..n).map(|k| qs[i + n * k] * q[j + n * k]).sum();
      }
    }
    Ok(out)
  }
}

/// (tr M, i2(M) = (tr(M)^2 - tr(M M))/2)
fn tr_i2(t: &[f64], n: usize) -> (f64, f64) {
  let tr = trace_f(t, n);
  let mut tr2 = 0.0;
  for i in 0..n {
    for j in 0..n {
      tr2 += t[i + j * n] * t[j + i * n];
    }
  }
  (tr, (tr * tr - tr2) / 2.0)
}

fn delta(i: usize, j: usize) -> f64 {
  (i == j) as u8 as f64
}

/// (I+Grad_u)^-T n normalized, with f = (I+Grad_u)^-1 and the norm of
/// (I+Grad_u)^-T n (GetFEM's Transformed_unit_vector)
fn transformed_unit_vector(
  grad_u: &[f64],
  nv: &[f64],
  n: usize,
) -> OpResult<(Vec<f64>, (Small, f64))> {
  let mut f = Small::<9>::from(grad_u);
  add_diag(&mut f, 1.0, n);
  lu_inverse(&mut f, n).map_err(|_| "Non invertible matrix".to_string())?;
  let mut ndef: Vec<f64> = (0..n)
    .map(|a| (0..n).map(|b| f[b + n * a] * nv[b]).sum())
    .collect();
  let no = vect_norm2(&ndef);
  ndef.iter_mut().for_each(|x| *x /= no);
  Ok((ndef, (f, no)))
}

// ------------------------------------------------------------------ built-ins

#[derive(Clone, Copy, Debug)]
pub enum Builtin {
  Norm,
  NormSqr,
  Normalized,
  NormalizedReg,
  BallProjection,
  Det,
  Inv,
  Expm,
  Logm,
  MatrixI2,
  MatrixJ1,
  MatrixJ2,
  RightCauchyGreen,
  LeftCauchyGreen,
  GreenLagrangian,
  TransformedUnitVector,
}

pub static BUILTINS: &[Builtin] = &[
  Builtin::Norm,
  Builtin::NormSqr,
  Builtin::Normalized,
  Builtin::NormalizedReg,
  Builtin::BallProjection,
  Builtin::Det,
  Builtin::Inv,
  Builtin::Expm,
  Builtin::Logm,
  Builtin::MatrixI2,
  Builtin::MatrixJ1,
  Builtin::MatrixJ2,
  Builtin::RightCauchyGreen,
  Builtin::LeftCauchyGreen,
  Builtin::GreenLagrangian,
  Builtin::TransformedUnitVector,
];

impl Builtin {
  pub fn name(self) -> &'static str {
    match self {
      Builtin::Norm => "Norm",
      Builtin::NormSqr => "Norm_sqr",
      Builtin::Normalized => "Normalized",
      Builtin::NormalizedReg => "Normalized_reg",
      Builtin::BallProjection => "Ball_projection",
      Builtin::Det => "Det",
      Builtin::Inv => "Inv",
      Builtin::Expm => "Expm",
      Builtin::Logm => "Logm",
      Builtin::MatrixI2 => "Matrix_i2",
      Builtin::MatrixJ1 => "Matrix_j1",
      Builtin::MatrixJ2 => "Matrix_j2",
      Builtin::RightCauchyGreen => "Right_Cauchy_Green",
      Builtin::LeftCauchyGreen => "Left_Cauchy_Green",
      Builtin::GreenLagrangian => "Green_Lagrangian",
      Builtin::TransformedUnitVector => "Transformed_unit_vector",
    }
  }

  fn shape(self, a: &[&[usize]]) -> Option<Vec<usize>> {
    let square = |s: &[usize]| s.len() == 2 && s[0] == s[1];
    let size1 = |s: &[usize]| s.iter().product::<usize>() == 1;
    match self {
      Builtin::Norm | Builtin::NormSqr if a.len() == 1 && a[0].len() <= 2 => Some(vec![]),
      Builtin::Normalized if a.len() == 1 && (1..=2).contains(&a[0].len()) => Some(a[0].to_vec()),
      Builtin::NormalizedReg if a.len() == 2 && (1..=2).contains(&a[0].len()) && size1(a[1]) => {
        Some(a[0].to_vec())
      }
      Builtin::BallProjection if a.len() == 2 && a[0].len() <= 2 && size1(a[1]) => {
        Some(a[0].to_vec())
      }
      Builtin::Det | Builtin::MatrixI2 | Builtin::MatrixJ1 | Builtin::MatrixJ2
        if a.len() == 1 && square(a[0]) =>
      {
        Some(vec![])
      }
      Builtin::Inv | Builtin::Expm | Builtin::Logm if a.len() == 1 && square(a[0]) => {
        Some(a[0].to_vec())
      }
      Builtin::RightCauchyGreen | Builtin::GreenLagrangian if a.len() == 1 && a[0].len() == 2 => {
        Some(vec![a[0][1], a[0][1]])
      }
      Builtin::LeftCauchyGreen if a.len() == 1 && a[0].len() == 2 => Some(vec![a[0][0], a[0][0]]),
      Builtin::TransformedUnitVector
        if a.len() == 2 && square(a[0]) && a[1].iter().product::<usize>() == a[0][0] =>
      {
        Some(vec![a[0][0]])
      }
      _ => None,
    }
  }

  /// first derivatives only (GetFEM has no second derivative)
  fn first_only(self) -> bool {
    matches!(
      self,
      Builtin::Expm
        | Builtin::Logm
        | Builtin::Normalized
        | Builtin::NormalizedReg
        | Builtin::BallProjection
        | Builtin::TransformedUnitVector
    )
  }
}

/// The built-in operators as GetFEM (getfem_generic_assembly_functions_and_
/// operators.cc, getfem_nonlinear_elasticity.cc, getfem_plasticity.cc).
impl NonlinearOperator for Builtin {
  fn result_shape(&self, args: &[&[usize]]) -> Option<Vec<usize>> {
    self.shape(args)
  }

  fn has_derivative(&self, der: (usize, usize)) -> bool {
    der.1 == 0 || !self.first_only()
  }

  fn value(&self, args: &[&[f64]], shapes: &[&[usize]], out: &mut [f64]) -> OpResult<()> {
    let t = args[0];
    let n = shapes[0].first().copied().unwrap_or(1);
    match self {
      Builtin::Norm => out[0] = vect_norm2(t),
      Builtin::NormSqr => out[0] = vect_norm2_sqr(t),
      Builtin::Normalized | Builtin::NormalizedReg => {
        let eps = if let Builtin::NormalizedReg = self {
          args[1][0]
        } else {
          1e-25
        };
        let no = (vect_norm2_sqr(t) + eps * eps).sqrt();
        scaled(t, 1.0 / no, out)
      }
      // r u/|u| if |u| > r, else u
      Builtin::BallProjection => {
        let (r, no) = (args[1][0], vect_norm2(t));
        if no > r {
          scaled(t, r / no, out)
        } else {
          out.copy_from_slice(t)
        }
      }
      Builtin::Det => out[0] = lu_det(t, n),
      Builtin::Inv => {
        out.copy_from_slice(t);
        lu_inverse(out, n).map_err(|_| "Non invertible matrix".to_string())?;
      }
      // (these allocate their iterates)
      Builtin::Expm => out.copy_from_slice(&expm(t, n)?),
      Builtin::Logm => out.copy_from_slice(&logm(t, n)?),
      Builtin::MatrixI2 => out[0] = tr_i2(t, n).1,
      Builtin::MatrixJ1 | Builtin::MatrixJ2 => {
        let (tr, i2) = tr_i2(t, n);
        let det = lu_det(t, n);
        out[0] = if det <= 0.0 {
          1e200
        } else if let Builtin::MatrixJ1 = self {
          tr / det.powf(1.0 / 3.0)
        } else {
          i2 / det.powf(2.0 / 3.0)
        }
      }
      Builtin::RightCauchyGreen | Builtin::GreenLagrangian => {
        let (m, nn) = (shapes[0][0], shapes[0][1]);
        let gl = matches!(self, Builtin::GreenLagrangian);
        out.fill(0.0);
        for j in 0..nn {
          for i in 0..nn {
            let o = &mut out[i + nn * j];
            for k in 0..m {
              *o += if gl {
                t[i * m + k] * t[j * m + k] * 0.5
              } else {
                t[i * m + k] * t[j * m + k]
              };
            }
            if gl && i == j {
              *o -= 0.5;
            }
          }
        }
      }
      Builtin::LeftCauchyGreen => {
        let (m, nn) = (shapes[0][0], shapes[0][1]);
        out.fill(0.0);
        for j in 0..m {
          for i in 0..m {
            for k in 0..nn {
              out[i + m * j] += t[k * m + i] * t[k * m + j];
            }
          }
        }
      }
      // (I+Grad_u)^-T n / |(I+Grad_u)^-T n|
      Builtin::TransformedUnitVector => {
        let (ndef, _) = transformed_unit_vector(t, args[1], n)?;
        out.copy_from_slice(&ndef);
      }
    }
    Ok(())
  }

  fn derivative(
    &self,
    args: &[&[f64]],
    shapes: &[&[usize]],
    i: usize,
    out: &mut [f64],
  ) -> OpResult<()> {
    let t = args[0];
    let nt = t.len();
    let n = shapes[0].first().copied().unwrap_or(1);
    match self {
      // u/|u|, 0 at the origin
      Builtin::Norm => {
        let no = vect_norm2(t);
        if no == 0.0 {
          out.fill(0.0)
        } else {
          scaled(t, 1.0 / no, out)
        }
      }
      Builtin::NormSqr => scaled(t, 2.0, out),
      // (|u|^2 Id - u x u)/|u|^3, |u| regularized
      Builtin::Normalized | Builtin::NormalizedReg => {
        let eps = if let Builtin::NormalizedReg = self {
          args[1][0]
        } else {
          1e-25
        };
        let no = (vect_norm2_sqr(t) + eps * eps).sqrt();
        let no3 = no * no * no;
        if i == 1 {
          // w.r.t. eps: -eps u/(|u|^2+eps^2)^(3/2)
          scaled(t, -eps / no3, out)
        } else {
          out.fill(0.0);
          for a in 0..nt {
            out[a * nt + a] += 1.0 / no;
            for b in 0..nt {
              out[b * nt + a] -= t[a] * t[b] / no3;
            }
          }
        }
      }
      Builtin::BallProjection => {
        let (r, no) = (args[1][0], vect_norm2(t));
        let rno3 = r / (no * no * no);
        if i == 0 {
          out.fill(0.0);
          if r > 0.0 {
            for a in 0..nt {
              if no <= r {
                out[a * nt + a] += 1.0;
              } else {
                out[a * nt + a] += r / no;
                for b in 0..nt {
                  out[b * nt + a] -= t[a] * t[b] * rno3;
                }
              }
            }
          }
        } else {
          out.fill(0.0);
          if r > 0.0 && no > r {
            for a in 0..nt {
              out[a] = t[a] / no;
            }
          }
        }
      }
      // det(M) M^-T
      Builtin::Det => {
        let (det, inv) = inverse_of(t, n)?;
        out.fill(0.0);
        if det != 0.0 {
          for j in 0..n {
            for a in 0..n {
              out[a + n * j] = inv[j + n * a] * det;
            }
          }
        }
      }
      // -M^-1_ik M^-1_lj
      Builtin::Inv => {
        let inv = inverse_of(t, n)?.1;
        out.fill(0.0);
        for l in 0..n {
          for k in 0..n {
            for j in 0..n {
              for a in 0..n {
                out[a + n * (j + n * (k + n * l))] = -inv[a + n * k] * inv[l + n * j];
              }
            }
          }
        }
      }
      Builtin::Expm => out.copy_from_slice(&expm_deriv(t, n)?),
      // inverse of the derivative of expm at logm(M)
      Builtin::Logm => {
        let l = logm(t, n)?;
        let d = expm_deriv(&l, n)?;
        let mut inv = d;
        let det = lu_inverse(&mut inv, n * n).map_err(|_| "Non invertible matrix".to_string())?;
        if det <= 0.0 {
          out.fill(0.0);
          (0..n * n).for_each(|i| out[i * (n * n + 1)] = 1.0);
        } else {
          out.copy_from_slice(&inv);
        }
      }
      // Trace(M) I - M^T
      Builtin::MatrixI2 => {
        let tr = trace_f(t, n);
        out.fill(0.0);
        for j in 0..n {
          for a in 0..n {
            out[a + n * j] = delta(a, j) * tr - t[a * n + j];
          }
        }
      }
      // (I - Trace(M) M^-T/3)/det(M)^(1/3)
      Builtin::MatrixJ1 => {
        let tr = trace_f(t, n);
        let (det, m) = inverse_of(t, n)?;
        if det > 0.0 {
          out.fill(0.0);
          for j in 0..n {
            for a in 0..n {
              out[a + n * j] = (delta(a, j) - tr * m[j + n * a] / 3.0) / det.powf(1.0 / 3.0);
            }
          }
        } else {
          out.fill(1e200)
        }
      }
      // (Trace(M) I - M^T - 2 i2(M) M^-T/3)/det(M)^(2/3)
      Builtin::MatrixJ2 => {
        let (tr, i2) = tr_i2(t, n);
        let (det, m) = inverse_of(t, n)?;
        out.fill(0.0);
        for j in 0..n {
          for a in 0..n {
            out[a + n * j] = (delta(a, j) * tr - t[j + n * a] - 2.0 * i2 * m[j + n * a] / 3.0)
              / det.powf(2.0 / 3.0);
          }
        }
      }
      // F_kj delta_li + F_ki delta_lj (halves for Green_Lagrangian)
      Builtin::RightCauchyGreen | Builtin::GreenLagrangian => {
        let (m, nn) = (shapes[0][0], shapes[0][1]);
        let h = if let Builtin::GreenLagrangian = self {
          0.5
        } else {
          1.0
        };
        out.fill(0.0);
        for l in 0..nn {
          for k in 0..m {
            for j in 0..nn {
              for a in 0..nn {
                let o = &mut out[a + nn * (j + nn * (k + m * l))];
                if l == a {
                  *o += t[j * m + k] * h;
                }
                if l == j {
                  *o += t[a * m + k] * h;
                }
              }
            }
          }
        }
      }
      // F_jl delta_ik + F_il delta_kj
      Builtin::LeftCauchyGreen => {
        let (m, nn) = (shapes[0][0], shapes[0][1]);
        out.fill(0.0);
        for l in 0..nn {
          for k in 0..m {
            for j in 0..m {
              for a in 0..m {
                let o = &mut out[a + m * (j + m * (k + m * l))];
                if k == a {
                  *o += t[l * m + j];
                }
                if k == j {
                  *o += t[l * m + a];
                }
              }
            }
          }
        }
      }
      // w.r.t. Grad_u: -G_ik ndef_j, G = (I - ndef x ndef)(I+Grad_u)^-T;
      // w.r.t. n: (I - ndef x ndef)(I+Grad_u)^-T/|(I+Grad_u)^-T n|
      Builtin::TransformedUnitVector => {
        let (ndef, (f, no)) = transformed_unit_vector(t, args[1], n)?;
        // f ndef
        let aux: Vec<f64> = (0..n)
          .map(|k| (0..n).map(|j| f[k + n * j] * ndef[j]).sum())
          .collect();
        if i == 0 {
          for k in 0..n {
            for j in 0..n {
              for a in 0..n {
                let g = f[k + n * a] - ndef[a] * aux[k];
                out[a + n * (j + n * k)] = -g * ndef[j];
              }
            }
          }
        } else {
          for j in 0..n {
            for a in 0..n {
              out[a + n * j] = (f[j + n * a] - ndef[a] * aux[j]) / no;
            }
          }
        }
      }
    }
    Ok(())
  }

  fn second_derivative(
    &self,
    args: &[&[f64]],
    shapes: &[&[usize]],
    _i: usize,
    _j: usize,
    out: &mut [f64],
  ) -> OpResult<()> {
    let t = args[0];
    let nt = t.len();
    let n = shapes[0].first().copied().unwrap_or(1);
    if self.first_only() {
      return Err(format!(
        "{}: second derivative not yet available",
        self.name()
      ));
    }
    match self {
      // (|u|^2 Id - u x u)/|u|^3
      Builtin::Norm => {
        let mut no = vect_norm2(t);
        let no3 = no * no * no;
        no = no.max(1e-25);
        out.fill(0.0);
        for b in 0..nt {
          for a in 0..nt {
            let o = &mut out[b * nt + a];
            *o = -t[a] * t[b] / no3;
            if a == b {
              *o += 1.0 / no;
            }
          }
        }
      }
      Builtin::NormSqr => {
        out.fill(0.0);
        for a in 0..nt {
          out[a * nt + a] = 2.0;
        }
      }
      // det(M) (M^-1_ji M^-1_lk - M^-1_jk M^-1_li)
      Builtin::Det => {
        let (det, inv) = inverse_of(t, n)?;
        out.fill(0.0);
        if det != 0.0 {
          for l in 0..n {
            for k in 0..n {
              for j in 0..n {
                for a in 0..n {
                  out[a + n * (j + n * (k + n * l))] =
                    (inv[j + n * a] * inv[l + n * k] - inv[j + n * k] * inv[l + n * a]) * det;
                }
              }
            }
          }
        }
      }
      // M^-1_ik M^-1_lm M^-1_nj + M^-1_im M^-1_nk M^-1_lj
      Builtin::Inv => {
        let v = inverse_of(t, n)?.1;
        let m = |r: usize, c: usize| v[r + n * c];
        out.fill(0.0);
        let mut it = 0;
        for q in 0..n {
          for p in 0..n {
            for l in 0..n {
              for k in 0..n {
                for j in 0..n {
                  for a in 0..n {
                    out[it] = m(a, k) * m(l, p) * m(q, j) + m(a, p) * m(q, k) * m(l, j);
                    it += 1;
                  }
                }
              }
            }
          }
        }
      }
      // I@I - delta_il delta_jk
      Builtin::MatrixI2 => {
        out.fill(0.0);
        for a in 0..n {
          for j in 0..n {
            out[(n + 1) * (a + n * n * j)] += 1.0;
            out[(n + 1) * n * j + a * (n * n * n + 1)] -= 1.0;
          }
        }
      }
      Builtin::MatrixJ1 => {
        let tr = trace_f(t, n);
        let (det, m) = inverse_of(t, n)?;
        if det > 0.0 {
          let mi = |r: usize, c: usize| m[r + n * c];
          out.fill(0.0);
          for l in 0..n {
            for k in 0..n {
              for j in 0..n {
                for a in 0..n {
                  out[a + n * (j + n * (k + n * l))] =
                    (-delta(k, l) * mi(j, a) + tr * mi(j, k) * mi(l, a) - delta(a, j) * mi(l, k)
                      + tr * mi(j, a) * mi(l, k) / 3.0)
                      / (3.0 * det.powf(1.0 / 3.0));
                }
              }
            }
          }
        } else {
          out.fill(1e200)
        }
      }
      Builtin::MatrixJ2 => {
        let (tr, i2) = tr_i2(t, n);
        let (det, m) = inverse_of(t, n)?;
        let mi = |r: usize, c: usize| m[r + n * c];
        out.fill(0.0);
        for l in 0..n {
          for k in 0..n {
            for j in 0..n {
              for a in 0..n {
                out[a + n * (j + n * (k + n * l))] = (delta(a, j) * delta(k, l)
                  - delta(a, l) * delta(k, j)
                  + 10.0 * i2 * mi(j, a) * mi(l, k) / 9.0
                  - 2.0 * (mi(j, a) * (tr * delta(k, l) - t[l + n * k])) / 3.0
                  - 2.0 * (mi(l, k) * (tr * delta(a, j) - t[j + n * a])) / 3.0
                  - 2.0 * i2 * (mi(j, a) * mi(l, k) - mi(j, k) * mi(l, a)) / 3.0)
                  / det.powf(2.0 / 3.0);
              }
            }
          }
        }
      }
      // delta_ok (delta_li delta_pj + delta_pi delta_lj) (halves for GL)
      Builtin::RightCauchyGreen | Builtin::GreenLagrangian => {
        let (m, nn) = (shapes[0][0], shapes[0][1]);
        let h = if let Builtin::GreenLagrangian = self {
          0.5
        } else {
          1.0
        };
        out.fill(0.0);
        let mut it = 0;
        for p in 0..nn {
          for o in 0..m {
            for l in 0..nn {
              for k in 0..m {
                for j in 0..nn {
                  for a in 0..nn {
                    if o == k {
                      out[it] = h * (delta(l, a) * delta(p, j) + delta(p, a) * delta(l, j));
                    }
                    it += 1;
                  }
                }
              }
            }
          }
        }
      }
      // delta_pl (delta_ki delta_oj + delta_oi delta_kj)
      Builtin::LeftCauchyGreen => {
        let (m, nn) = (shapes[0][0], shapes[0][1]);
        out.fill(0.0);
        let mut it = 0;
        for p in 0..nn {
          for o in 0..m {
            for l in 0..nn {
              for k in 0..m {
                for j in 0..m {
                  for a in 0..m {
                    if p == l {
                      out[it] = delta(k, a) * delta(o, j) + delta(o, a) * delta(k, j);
                    }
                    it += 1;
                  }
                }
              }
            }
          }
        }
      }
      _ => unreachable!(),
    }
    Ok(())
  }
}
