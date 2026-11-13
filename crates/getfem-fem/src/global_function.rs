//! Global functions (GetFEM's getfem_global_function): scalar functions on
//! the whole mesh, with value, gradient, Hessian, a support test and a
//! bounding box; the uniform B-spline functions of orders 3 to 5 in 1D to
//! 3D, sums, products and functions bounded to a box; crack tip functions
//! of level sets. They are the basis functions of global-function
//! elements (fem_global_function).

use std::sync::{Arc, Mutex};

use crate::crack_tip_function::CrackTipFunction;
use crate::fem::FemCtx;
use crate::level_set::{ConvexLevelSet, LevelSet};
use crate::{Error, Result};

/// Where a global function is evaluated (GetFEM's
/// fem_interpolation_context): the real point x and, on an element, the
/// convex and the element context (functions of level sets use them).
#[derive(Clone, Copy, Debug)]
pub struct EvalPoint<'a> {
  pub x: &'a [f64],
  /// the convex and the element context
  pub element: Option<(usize, &'a FemCtx<'a>)>,
}

impl<'a> From<&'a [f64]> for EvalPoint<'a> {
  fn from(x: &'a [f64]) -> Self {
    EvalPoint { x, element: None }
  }
}

impl<'a> From<&'a Vec<f64>> for EvalPoint<'a> {
  fn from(x: &'a Vec<f64>) -> Self {
    EvalPoint { x, element: None }
  }
}

impl<'a, const N: usize> From<&'a [f64; N]> for EvalPoint<'a> {
  fn from(x: &'a [f64; N]) -> Self {
    EvalPoint { x, element: None }
  }
}

impl<'a> From<&EvalPoint<'a>> for EvalPoint<'a> {
  fn from(p: &EvalPoint<'a>) -> Self {
    *p
  }
}

/// A scalar function on the whole mesh. Hessians are dim x dim,
/// column-major.
pub trait GlobalFunctionImpl: Send + Sync {
  fn dim(&self) -> usize;
  fn val(&self, p: &EvalPoint) -> f64;
  fn grad(&self, p: &EvalPoint) -> Vec<f64>;
  fn hess(&self, p: &EvalPoint) -> Vec<f64>;
  /// changes when what it depends on besides the point changes (GetFEM
  /// context_dependencies; 0: nothing): values computed before are invalid
  fn version(&self) -> u64 {
    0
  }
  /// whether x is in the support (GetFEM: true when unknown)
  fn is_in_support(&self, _x: &[f64]) -> bool {
    true
  }
  /// a box containing the support (GetFEM: +-1e25 when unknown)
  fn bounding_box(&self) -> (Vec<f64>, Vec<f64>) {
    (vec![-1e25; self.dim()], vec![1e25; self.dim()])
  }
}

/// A global function (GetFEM's pglobal_function).
#[derive(Clone)]
pub struct GlobalFunction(Arc<dyn GlobalFunctionImpl>);

impl std::fmt::Debug for GlobalFunction {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "GlobalFunction(dim {})", self.dim())
  }
}

impl GlobalFunction {
  /// A user's implementation.
  pub fn new(f: impl GlobalFunctionImpl + 'static) -> Self {
    GlobalFunction(Arc::new(f))
  }
  pub fn dim(&self) -> usize {
    self.0.dim()
  }
  /// at a point (a slice) or an EvalPoint
  pub fn val<'a>(&self, p: impl Into<EvalPoint<'a>>) -> f64 {
    self.0.val(&p.into())
  }
  pub fn grad<'a>(&self, p: impl Into<EvalPoint<'a>>) -> Vec<f64> {
    self.0.grad(&p.into())
  }
  pub fn hess<'a>(&self, p: impl Into<EvalPoint<'a>>) -> Vec<f64> {
    self.0.hess(&p.into())
  }
  pub fn version(&self) -> u64 {
    self.0.version()
  }
  pub fn is_in_support(&self, x: &[f64]) -> bool {
    self.0.is_in_support(x)
  }
  pub fn bounding_box(&self) -> (Vec<f64>, Vec<f64>) {
    self.0.bounding_box()
  }

  /// The uniform B-spline of the given order (3 to 5) and types (1 to
  /// order) per direction (GetFEM global_function_bspline): along each
  /// axis, the knot interval [min, max] split into `type` unit intervals;
  /// type = order is the interior B-spline, smaller types are the
  /// functions at a boundary (at min; max < min mirrors them).
  pub fn bspline(min: &[f64], max: &[f64], order: usize, types: &[usize]) -> Result<Self> {
    let n = min.len();
    if !(1..=3).contains(&n) || max.len() != n || types.len() != n {
      return Err(Error::InvalidArgument(
        "bspline: 1 to 3 directions, one min, max and type each".into(),
      ));
    }
    if !(3..=5).contains(&order) {
      return Err(Error::InvalidArgument(
        "bspline: only orders 3 to 5 are supported".into(),
      ));
    }
    if types.iter().any(|&t| t < 1 || t > order) {
      return Err(Error::InvalidArgument(format!(
        "bspline: types 1 to {order} for order {order}"
      )));
    }
    let axes = (0..n)
      .map(|i| Axis {
        min: min[i],
        max: max[i],
        scale: types[i] as f64 / (max[i] - min[i]),
        f: BSP[order - 3][types[i] - 1],
      })
      .collect();
    Ok(GlobalFunction(Arc::new(BSpline { axes })))
  }

  /// The sum of functions of the same dimension (GetFEM
  /// global_function_sum).
  pub fn sum(funcs: &[GlobalFunction]) -> Result<Self> {
    let dim = funcs.first().map_or(0, |f| f.dim());
    if funcs.iter().any(|f| f.dim() != dim) {
      return Err(Error::InvalidArgument(
        "Incompatible dimensions among the provided global functions".into(),
      ));
    }
    Ok(GlobalFunction(Arc::new(Sum(funcs.to_vec(), dim))))
  }

  /// The product of two functions (GetFEM global_function_product).
  pub fn product(f1: &GlobalFunction, f2: &GlobalFunction) -> Result<Self> {
    if f1.dim() != f2.dim() {
      return Err(Error::InvalidArgument(
        "Incompatible dimensions between the provided global functions".into(),
      ));
    }
    Ok(GlobalFunction(Arc::new(Product(f1.clone(), f2.clone()))))
  }

  /// f with the bounding box [min, max] (GetFEM global_function_bounded
  /// without an expression; its values are those of f).
  pub fn bounded(f: &GlobalFunction, min: &[f64], max: &[f64]) -> Result<Self> {
    if min.len() != f.dim() || max.len() != f.dim() {
      return Err(Error::InvalidArgument("Wrong dimensions".into()));
    }
    Ok(GlobalFunction(Arc::new(Bounded {
      f: f.clone(),
      min: min.to_vec(),
      max: max.to_vec(),
    })))
  }
}

// ------------------------------------------------------------- B-splines

/// a 1D B-spline in unit knot intervals: value, first and second
/// derivatives
type Bsp = [fn(f64) -> f64; 3];

const BSP: [[Bsp; 5]; 3] = [
  [
    [bsp3_01, bsp3_01_der, bsp3_01_der2],
    [bsp3_02, bsp3_02_der, bsp3_02_der2],
    [bsp3_03, bsp3_03_der, bsp3_03_der2],
    [zero, zero, zero],
    [zero, zero, zero],
  ],
  [
    [bsp4_01, bsp4_01_der, bsp4_01_der2],
    [bsp4_02, bsp4_02_der, bsp4_02_der2],
    [bsp4_03, bsp4_03_der, bsp4_03_der2],
    [bsp4_04, bsp4_04_der, bsp4_04_der2],
    [zero, zero, zero],
  ],
  [
    [bsp5_01, bsp5_01_der, bsp5_01_der2],
    [bsp5_02, bsp5_02_der, bsp5_02_der2],
    [bsp5_03, bsp5_03_der, bsp5_03_der2],
    [bsp5_04, bsp5_04_der, bsp5_04_der2],
    [bsp5_05, bsp5_05_der, bsp5_05_der2],
  ],
];

fn zero(_: f64) -> f64 {
  0.0
}

struct Axis {
  min: f64,
  max: f64,
  scale: f64,
  f: Bsp,
}

/// The tensor product of 1D B-splines (GetFEM's global_function_x_bspline_,
/// _xy_, _xyz_).
struct BSpline {
  axes: Vec<Axis>,
}

impl BSpline {
  /// per axis: the value and the scaled first and second derivatives
  fn factors(&self, x: &[f64]) -> Vec<[f64; 3]> {
    self
      .axes
      .iter()
      .zip(x)
      .map(|(a, &xi)| {
        let t = a.scale * (xi - a.min);
        [
          (a.f[0])(t),
          a.scale * (a.f[1])(t),
          a.scale * a.scale * (a.f[2])(t),
        ]
      })
      .collect()
  }
}

impl GlobalFunctionImpl for BSpline {
  fn dim(&self) -> usize {
    self.axes.len()
  }
  fn val(&self, p: &EvalPoint) -> f64 {
    self.factors(p.x).iter().map(|f| f[0]).product()
  }
  fn grad(&self, p: &EvalPoint) -> Vec<f64> {
    let f = self.factors(p.x);
    (0..f.len())
      .map(|i| {
        (0..f.len())
          .map(|j| if j == i { f[j][1] } else { f[j][0] })
          .product()
      })
      .collect()
  }
  fn hess(&self, p: &EvalPoint) -> Vec<f64> {
    let f = self.factors(p.x);
    let n = f.len();
    let mut h = vec![0.0; n * n];
    for i in 0..n {
      for j in 0..n {
        h[i + n * j] = (0..n)
          .map(|k| match (k == i, k == j) {
            (true, true) => f[k][2],
            (true, false) | (false, true) => f[k][1],
            _ => f[k][0],
          })
          .product();
      }
    }
    h
  }
  fn is_in_support(&self, x: &[f64]) -> bool {
    self
      .axes
      .iter()
      .zip(x)
      .all(|(a, &xi)| (xi - (a.min + a.max) / 2.0).abs() + 1e-9 < (a.max - a.min).abs() / 2.0)
  }
  fn bounding_box(&self) -> (Vec<f64>, Vec<f64>) {
    (
      self.axes.iter().map(|a| a.min.min(a.max)).collect(),
      self.axes.iter().map(|a| a.min.max(a.max)).collect(),
    )
  }
}

// ------------------------------------------------- sums, products, boxes

struct Sum(Vec<GlobalFunction>, usize);

impl GlobalFunctionImpl for Sum {
  fn dim(&self) -> usize {
    self.1
  }
  fn version(&self) -> u64 {
    self
      .0
      .iter()
      .map(GlobalFunction::version)
      .max()
      .unwrap_or(0)
  }
  fn val(&self, x: &EvalPoint) -> f64 {
    self.0.iter().map(|f| f.val(x)).sum()
  }
  fn grad(&self, x: &EvalPoint) -> Vec<f64> {
    let mut g = vec![0.0; self.1];
    for f in &self.0 {
      g.iter_mut().zip(f.grad(x)).for_each(|(a, b)| *a += b);
    }
    g
  }
  fn hess(&self, x: &EvalPoint) -> Vec<f64> {
    let mut h = vec![0.0; self.1 * self.1];
    for f in &self.0 {
      h.iter_mut().zip(f.hess(x)).for_each(|(a, b)| *a += b);
    }
    h
  }
  fn is_in_support(&self, x: &[f64]) -> bool {
    self.0.iter().any(|f| f.is_in_support(x))
  }
  /// the union of the boxes
  fn bounding_box(&self) -> (Vec<f64>, Vec<f64>) {
    let mut b = self.0.first().map_or_else(
      || (vec![-1e25; self.1], vec![1e25; self.1]),
      |f| f.bounding_box(),
    );
    for f in &self.0 {
      let (lo, hi) = f.bounding_box();
      for i in 0..self.1 {
        b.0[i] = b.0[i].min(lo[i]);
        b.1[i] = b.1[i].max(hi[i]);
      }
    }
    b
  }
}

struct Product(GlobalFunction, GlobalFunction);

impl GlobalFunctionImpl for Product {
  fn dim(&self) -> usize {
    self.0.dim()
  }
  fn version(&self) -> u64 {
    self.0.version().max(self.1.version())
  }

  fn val(&self, x: &EvalPoint) -> f64 {
    self.0.val(x) * self.1.val(x)
  }
  fn grad(&self, x: &EvalPoint) -> Vec<f64> {
    let (v1, v2) = (self.0.val(x), self.1.val(x));
    self
      .0
      .grad(x)
      .iter()
      .zip(self.1.grad(x))
      .map(|(g1, g2)| g1 * v2 + g2 * v1)
      .collect()
  }
  /// h1 f2 + h2 f1 + g1 g2' + g2 g1'
  fn hess(&self, x: &EvalPoint) -> Vec<f64> {
    let n = self.dim();
    let (v1, v2) = (self.0.val(x), self.1.val(x));
    let (g1, g2) = (self.0.grad(x), self.1.grad(x));
    let (h1, h2) = (self.0.hess(x), self.1.hess(x));
    let mut h = vec![0.0; n * n];
    for j in 0..n {
      for i in 0..n {
        h[i + n * j] = h1[i + n * j] * v2 + h2[i + n * j] * v1 + g1[i] * g2[j] + g2[i] * g1[j];
      }
    }
    h
  }
  fn is_in_support(&self, x: &[f64]) -> bool {
    self.0.is_in_support(x) && self.1.is_in_support(x)
  }
  /// the intersection of the boxes
  fn bounding_box(&self) -> (Vec<f64>, Vec<f64>) {
    let (lo0, hi0) = self.0.bounding_box();
    let (mut lo, mut hi) = self.1.bounding_box();
    for i in 0..lo.len() {
      lo[i] = lo[i].max(lo0[i]);
      hi[i] = hi[i].min(hi0[i]);
    }
    (lo, hi)
  }
}

struct Bounded {
  f: GlobalFunction,
  min: Vec<f64>,
  max: Vec<f64>,
}

impl GlobalFunctionImpl for Bounded {
  fn dim(&self) -> usize {
    self.f.dim()
  }
  fn version(&self) -> u64 {
    self.f.version()
  }

  fn val(&self, x: &EvalPoint) -> f64 {
    self.f.val(x)
  }
  fn grad(&self, x: &EvalPoint) -> Vec<f64> {
    self.f.grad(x)
  }
  fn hess(&self, x: &EvalPoint) -> Vec<f64> {
    self.f.hess(x)
  }
  fn bounding_box(&self) -> (Vec<f64>, Vec<f64>) {
    (self.min.clone(), self.max.clone())
  }
}

// --------------------------------------- GetFEM's 1D B-spline polynomials

const EPS: f64 = 1e-12;

// order k = 3
fn bsp3_01(t: f64) -> f64 {
  if (-EPS..1.0).contains(&t) {
    (1.0 - t).powi(2)
  } else {
    0.0
  }
}
fn bsp3_01_der(t: f64) -> f64 {
  if (-EPS..1.0).contains(&t) {
    2.0 * t - 2.0
  } else {
    0.0
  }
}
fn bsp3_01_der2(t: f64) -> f64 {
  if (EPS..=1.0 - EPS).contains(&t) {
    2.0
  } else {
    0.0
  }
}
fn bsp3_02(t: f64) -> f64 {
  if t >= -EPS {
    if t < 1.0 {
      return (-1.5 * t + 2.0) * t;
    } else if t < 2.0 {
      return 0.5 * (2.0 - t).powi(2);
    }
  }
  0.0
}
fn bsp3_02_der(t: f64) -> f64 {
  if t >= -EPS {
    if t < 1.0 {
      return -3.0 * t + 2.0;
    } else if t < 2.0 {
      return t - 2.0;
    }
  }
  0.0
}
fn bsp3_02_der2(t: f64) -> f64 {
  if t >= EPS {
    if t < 1.0 - EPS {
      return -3.0;
    } else if t < 1.0 + EPS {
      return 0.0;
    } else if t <= 2.0 - EPS {
      return 1.0;
    }
  }
  0.0
}
fn bsp3_03(mut t: f64) -> f64 {
  if t >= -EPS {
    if t < 1.0 {
      return 0.5 * t * t;
    } else if t < 2.0 {
      t -= 1.0;
      return t * (1.0 - t) + 0.5;
    } else if t < 3.0 {
      t = 3.0 - t;
      return 0.5 * t * t;
    }
  }
  0.0
}
fn bsp3_03_der(mut t: f64) -> f64 {
  if t >= -EPS {
    if t < 1.0 {
      return t;
    } else if t < 2.0 {
      t -= 1.0;
      return 1.0 - 2.0 * t;
    } else if t < 3.0 {
      return t - 3.0;
    }
  }
  0.0
}
fn bsp3_03_der2(t: f64) -> f64 {
  if t >= -EPS {
    if t < EPS {
      return 0.0;
    } else if t <= 1.0 - EPS {
      return 1.0;
    } else if t < 1.0 + EPS {
      return 0.0;
    } else if t <= 2.0 - EPS {
      return -2.0;
    } else if t < 2.0 + EPS {
      return 0.0;
    } else if t <= 3.0 - EPS {
      return 1.0;
    } else if t < 3.0 + EPS {
      return 0.0;
    }
  }
  0.0
}

// order k = 4
fn bsp4_01(t: f64) -> f64 {
  if (-EPS..1.0).contains(&t) {
    (1.0 - t).powi(3)
  } else {
    0.0
  }
}
fn bsp4_01_der(t: f64) -> f64 {
  if (-EPS..1.0).contains(&t) {
    -3.0 * (1.0 - t).powi(2)
  } else {
    0.0
  }
}
fn bsp4_01_der2(t: f64) -> f64 {
  if (-EPS..1.0).contains(&t) {
    6.0 * (1.0 - t)
  } else {
    0.0
  }
}
fn bsp4_02(t: f64) -> f64 {
  if t >= -EPS {
    if t < 1.0 {
      return ((7.0 / 4.0 * t - 9.0 / 2.0) * t + 3.0) * t;
    } else if t < 2.0 {
      return 1.0 / 4.0 * (2.0 - t).powi(3);
    }
  }
  0.0
}
fn bsp4_02_der(t: f64) -> f64 {
  if t >= -EPS {
    if t < 1.0 {
      return (21.0 / 4.0 * t - 9.0) * t + 3.0;
    } else if t < 2.0 {
      return -3.0 / 4.0 * (2.0 - t).powi(2);
    }
  }
  0.0
}
fn bsp4_02_der2(t: f64) -> f64 {
  if t >= -EPS {
    if t < 1.0 {
      return 21.0 / 2.0 * t - 9.0;
    } else if t < 2.0 {
      return 3.0 / 2.0 * (2.0 - t);
    }
  }
  0.0
}
fn bsp4_03(mut t: f64) -> f64 {
  if t >= -EPS {
    if t < 1.0 {
      return (-11.0 / 12.0 * t + 3.0 / 2.0) * t * t;
    } else if t < 2.0 {
      t -= 1.0;
      return ((7.0 / 12.0 * t - 5.0 / 4.0) * t + 1.0 / 4.0) * t + 7.0 / 12.0;
    } else if t < 3.0 {
      return 1.0 / 6.0 * (3.0 - t).powi(3);
    }
  }
  0.0
}
fn bsp4_03_der(mut t: f64) -> f64 {
  if t >= -EPS {
    if t < 1.0 {
      return (-11.0 / 4.0 * t + 3.0) * t;
    } else if t < 2.0 {
      t -= 1.0;
      return (7.0 / 4.0 * t - 5.0 / 2.0) * t + 1.0 / 4.0;
    } else if t < 3.0 {
      return -1.0 / 2.0 * (3.0 - t).powi(2);
    }
  }
  0.0
}
fn bsp4_03_der2(mut t: f64) -> f64 {
  if t >= -EPS {
    if t < 1.0 {
      return -11.0 / 2.0 * t + 3.0;
    } else if t < 2.0 {
      t -= 1.0;
      return 7.0 / 2.0 * t - 5.0 / 2.0;
    } else if t < 3.0 {
      return 3.0 - t;
    }
  }
  0.0
}
fn bsp4_04(mut t: f64) -> f64 {
  if t > 2.0 {
    t = 4.0 - t;
  }
  if t >= -EPS {
    if t < 1.0 {
      return 1.0 / 6.0 * (t).powi(3);
    } else if t <= 2.0 {
      t -= 1.0;
      return ((-1.0 / 2.0 * t + 1.0 / 2.0) * t + 1.0 / 2.0) * t + 1.0 / 6.0;
    }
  }
  0.0
}
fn bsp4_04_der(mut t: f64) -> f64 {
  let mut sgn = 1.0;
  if t > 2.0 {
    t = 4.0 - t;
    sgn = -1.0;
  }
  if t >= -EPS {
    if t < 1.0 {
      return 1.0 / 2.0 * (t).powi(2) * sgn;
    } else if t < 2.0 {
      t -= 1.0;
      return ((-3.0 / 2.0 * t + 1.0) * t + 1.0 / 2.0) * sgn;
    }
  }
  0.0
}
fn bsp4_04_der2(mut t: f64) -> f64 {
  if t > 2.0 {
    t = 4.0 - t;
  }
  if t >= -EPS {
    if t < 1.0 {
      return t;
    } else if t < 2.0 {
      t -= 1.0;
      return -3.0 * t + 1.0;
    }
  }
  0.0
}

// order k = 5
fn bsp5_01(t: f64) -> f64 {
  if (-EPS..1.0).contains(&t) {
    (1.0 - t).powi(4)
  } else {
    0.0
  }
}
fn bsp5_01_der(t: f64) -> f64 {
  if (-EPS..1.0).contains(&t) {
    -4.0 * (1.0 - t).powi(3)
  } else {
    0.0
  }
}
fn bsp5_01_der2(t: f64) -> f64 {
  if (-EPS..1.0).contains(&t) {
    12.0 * (1.0 - t).powi(2)
  } else {
    0.0
  }
}
fn bsp5_02(t: f64) -> f64 {
  if t >= -EPS {
    if t < 1.0 {
      return (((-15.0 / 8.0 * t + 7.0) * t - 9.0) * t + 4.0) * t;
    } else if t < 2.0 {
      return 1.0 / 8.0 * (2.0 - t).powi(4);
    }
  }
  0.0
}
fn bsp5_02_der(t: f64) -> f64 {
  if t >= -EPS {
    if t < 1.0 {
      return ((-15.0 / 2.0 * t + 21.0) * t - 18.0) * t + 4.0;
    } else if t < 2.0 {
      return -1.0 / 2.0 * (2.0 - t).powi(3);
    }
  }
  0.0
}
fn bsp5_02_der2(t: f64) -> f64 {
  if t >= -EPS {
    if t < 1.0 {
      return (-45.0 / 2.0 * t + 42.0) * t - 18.0;
    } else if t < 2.0 {
      return 3.0 / 2.0 * (2.0 - t).powi(2);
    }
  }
  0.0
}
fn bsp5_03(mut t: f64) -> f64 {
  if t >= -EPS {
    if t < 1.0 {
      return ((85.0 / 72.0 * t - 11.0 / 3.0) * t + 3.0) * t * t;
    } else if t < 2.0 {
      t = 2.0 - t;
      return (((-23.0 / 72.0 * t + 2.0 / 9.0) * t + 1.0 / 3.0) * t + 2.0 / 9.0) * t + 1.0 / 18.0;
    } else if t < 3.0 {
      return 1.0 / 18.0 * (3.0 - t).powi(4);
    }
  }
  0.0
}
fn bsp5_03_der(mut t: f64) -> f64 {
  if t >= -EPS {
    if t < 1.0 {
      return ((85.0 / 18.0 * t - 11.0) * t + 6.0) * t;
    } else if t < 2.0 {
      t = 2.0 - t;
      return -(((-23.0 / 18.0 * t + 2.0 / 3.0) * t + 2.0 / 3.0) * t + 2.0 / 9.0);
    } else if t < 3.0 {
      return -2.0 / 9.0 * (3.0 - t).powi(3);
    }
  }
  0.0
}
fn bsp5_03_der2(mut t: f64) -> f64 {
  if t >= -EPS {
    if t < 1.0 {
      return (85.0 / 6.0 * t - 22.0) * t + 6.0;
    } else if t < 2.0 {
      t = 2.0 - t;
      return (-23.0 / 6.0 * t + 4.0 / 3.0) * t + 2.0 / 3.0;
    } else if t < 3.0 {
      return 2.0 / 3.0 * (3.0 - t).powi(2);
    }
  }
  0.0
}
fn bsp5_04(mut t: f64) -> f64 {
  if t >= -EPS {
    if t < 1.0 {
      return (-25.0 / 72.0 * t + 2.0 / 3.0) * t * t * t;
    } else if t < 2.0 {
      t = 2.0 - t;
      return (((23.0 / 72.0 * t - 5.0 / 9.0) * t - 1.0 / 3.0) * t + 4.0 / 9.0) * t + 4.0 / 9.0;
    } else if t < 3.0 {
      t -= 2.0;
      return (((-13.0 / 72.0 * t + 5.0 / 9.0) * t - 1.0 / 3.0) * t - 4.0 / 9.0) * t + 4.0 / 9.0;
    } else if t < 4.0 {
      return 1.0 / 24.0 * (4.0 - t).powi(4);
    }
  }
  0.0
}
fn bsp5_04_der(mut t: f64) -> f64 {
  if t >= -EPS {
    if t < 1.0 {
      return (-25.0 / 18.0 * t + 2.0) * t * t;
    } else if t < 2.0 {
      t = 2.0 - t;
      return -(((23.0 / 18.0 * t - 5.0 / 3.0) * t - 2.0 / 3.0) * t + 4.0 / 9.0);
    } else if t < 3.0 {
      t -= 2.0;
      return ((-13.0 / 18.0 * t + 5.0 / 3.0) * t - 2.0 / 3.0) * t - 4.0 / 9.0;
    } else if t < 4.0 {
      return -1.0 / 6.0 * (4.0 - t).powi(3);
    }
  }
  0.0
}
fn bsp5_04_der2(mut t: f64) -> f64 {
  if t >= -EPS {
    if t < 1.0 {
      return (-25.0 / 6.0 * t + 4.0) * t;
    } else if t < 2.0 {
      t = 2.0 - t;
      return (23.0 / 6.0 * t - 10.0 / 3.0) * t - 2.0 / 3.0;
    } else if t < 3.0 {
      t -= 2.0;
      return (-13.0 / 6.0 * t + 10.0 / 3.0) * t - 2.0 / 3.0;
    } else if t < 4.0 {
      return 1.0 / 2.0 * (4.0 - t).powi(2);
    }
  }
  0.0
}
fn bsp5_05(mut t: f64) -> f64 {
  if t > 2.5 {
    t = 5.0 - t;
  }
  if t >= -EPS {
    if t < 1.0 {
      return 1.0 / 24.0 * (t).powi(4);
    } else if t < 2.0 {
      t -= 1.0;
      return (((-1.0 / 6.0 * t + 1.0 / 6.0) * t + 1.0 / 4.0) * t + 1.0 / 6.0) * t + 1.0 / 24.0;
    } else if t < 3.0 {
      t = (t - 2.5).powi(2);
      return (1.0 / 4.0 * t - 5.0 / 8.0) * t + 115.0 / 192.0;
    }
  }
  0.0
}
fn bsp5_05_der(mut t: f64) -> f64 {
  let mut sgn = 1.0;
  if t > 2.5 {
    t = 5.0 - t;
    sgn = -1.0;
  }
  if t >= -EPS {
    if t < 1.0 {
      return 1.0 / 6.0 * (t).powi(3) * sgn;
    } else if t < 2.0 {
      t -= 1.0;
      return (((-2.0 / 3.0 * t + 1.0 / 2.0) * t + 1.0 / 2.0) * t + 1.0 / 6.0) * sgn;
    } else if t < 3.0 {
      t -= 2.5;
      return (t * t - 5.0 / 4.0) * t * sgn;
    }
  }
  0.0
}
fn bsp5_05_der2(mut t: f64) -> f64 {
  if t > 2.5 {
    t = 5.0 - t;
  }
  if t >= -EPS {
    if t < 1.0 {
      return 1.0 / 2.0 * (t).powi(2);
    } else if t < 2.0 {
      t -= 1.0;
      return (-2.0 * t + 1.0) * t + 1.0 / 2.0;
    } else if t < 3.0 {
      t -= 2.5;
      return 3.0 * t * t - 5.0 / 4.0;
    }
  }
  0.0
}

/// The (x, y) = (secondary, primary) functions of a level set on a convex.
/// x, y on a convex and the id of their level set
type ConvexXy = Arc<(ConvexLevelSet, ConvexLevelSet, usize)>;

/// The (x, y) functions on the last convex, and their values at its last
/// point (up to derivative order der, before the side shift): val, grad
/// and hess at one point evaluate the level sets once.
struct XyCache {
  cv: usize,
  xy: ConvexXy,
  point: Option<(Vec<f64>, usize, [RealFn; 2])>,
}
/// A value with its real gradient and Hessian (empty if not computed).
type RealFn = (f64, Vec<f64>, Vec<f64>);

/// A crack tip function of (x, y) = (secondary, primary) values of a level
/// set on each convex (GetFEM global_function_on_levelsets_2D_): with
/// several level sets, the one whose (x, y) is the closest to 0 at the
/// reference origin of the convex. In the mesh dimension (GetFEM: 2): in
/// 3D, (x, y) span the plane normal to the crack front.
struct OnLevelSets {
  lss: Vec<LevelSet>,
  f: CrackTipFunction,
  dim: usize,
  /// the convex and the (x, y) functions of the chosen level set on it
  cache: Mutex<Option<XyCache>>,
}

/// A function of the reference coordinates: value, real gradient, real
/// Hessian at the point of c (the curvature of the transformation
/// included, as real_base_values_at)
fn real_of(l: &ConvexLevelSet, c: &FemCtx, der: usize) -> RealFn {
  let (n, p) = (c.g.nrows(), c.xref.len());
  let v = l.val(c.xref).unwrap_or(f64::NAN);
  if der == 0 {
    return (v, vec![], vec![]);
  }
  let gr = l.grad(c.xref).unwrap_or_else(|_| vec![f64::NAN; p]);
  let g: Vec<f64> = (0..n)
    .map(|r| (0..p).map(|j| c.b[r + n * j] * gr[j]).sum())
    .collect();
  if der == 1 {
    return (v, g, vec![]);
  }
  let mut hr = l.hess(c.xref).unwrap_or_else(|_| vec![f64::NAN; p * p]);
  if !c.pgt.is_linear() {
    // htau_l = sum_a G(l, a) hess(phi_a): hr -= htau . grad
    let gh = c.pgt.hess(c.xref);
    for (jk, h) in hr.iter_mut().enumerate() {
      for (l, gl) in g.iter().enumerate() {
        let ht: f64 = (0..c.g.ncols()).map(|a| c.g[(l, a)] * gh[(a, jk)]).sum();
        *h -= ht * gl;
      }
    }
  }
  let mut h = vec![0.0; n * n];
  for s in 0..n {
    for r in 0..n {
      h[r + n * s] = (0..p)
        .flat_map(|j| (0..p).map(move |k| (j, k)))
        .map(|(j, k)| c.b[r + n * j] * c.b[s + n * k] * hr[j + p * k])
        .sum();
    }
  }
  (v, g, h)
}

impl OnLevelSets {
  /// the (x, y) functions on convex cv
  fn of_convex(&self, cv: usize, p: usize) -> ConvexXy {
    let pair =
      |ls: &LevelSet| Ok::<_, Error>((ls.of_convex(cv, 1)?, ls.of_convex(cv, 0)?, ls.id()));
    let origin = vec![0.0; p];
    let mut best: Option<(f64, (ConvexLevelSet, ConvexLevelSet, usize))> = None;
    for ls in &self.lss {
      let Ok(xy) = pair(ls) else { continue };
      if self.lss.len() == 1 {
        best = Some((0.0, xy));
        break;
      }
      let (x, y) = (
        xy.0.val(&origin).unwrap_or(f64::NAN),
        xy.1.val(&origin).unwrap_or(f64::NAN),
      );
      let d2 = x * x + y * y;
      if best.as_ref().is_none_or(|(d, _)| d2 < *d) {
        best = Some((d2, xy));
      }
    }
    Arc::new(
      best
        .expect("global_function_on_level_set: no level set on the convex")
        .1,
    )
  }

  /// x, y with their derivatives up to der at p; y moved off the crack on
  /// the side of c if the point is on this level set (GetFEM: on any;
  /// 1e-13, the value with y = 0 included)
  fn xy(&self, p: &EvalPoint, der: usize) -> Option<[RealFn; 2]> {
    let (cv, c) = p.element?;
    let [x, mut y] = {
      let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
      if cache.as_ref().is_none_or(|k| k.cv != cv) {
        *cache = Some(XyCache {
          cv,
          xy: self.of_convex(cv, c.xref.len()),
          point: None,
        });
      }
      let k = cache.as_mut().unwrap();
      match &k.point {
        Some((xref, d, r)) if *d >= der && xref.as_slice() == c.xref => r.clone(),
        _ => {
          // at least the gradient: val and grad usually come together
          let d = der;
          let r = [real_of(&k.xy.0, c, d), real_of(&k.xy.1, c, d)];
          k.point = Some((c.xref.to_vec(), d, r.clone()));
          r
        }
      }
    };
    let side = {
      let cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
      c.side_on(cache.as_ref().map_or(0, |k| k.xy.2))
    };
    if der == 0 {
      if side > 0 && y.0 <= 1e-13 {
        y.0 = 1e-13;
      }
      if side < 0 && y.0 >= -1e-13 {
        y.0 = -1e-13;
      }
    } else {
      if side > 0 && y.0 <= 0.0 {
        y.0 = 1e-13;
      }
      if side < 0 && y.0 >= 0.0 {
        y.0 = -1e-13;
      }
    }
    Some([x, y])
  }
}

impl GlobalFunctionImpl for OnLevelSets {
  fn dim(&self) -> usize {
    self.dim
  }
  fn version(&self) -> u64 {
    self.lss.iter().map(LevelSet::version).max().unwrap_or(0)
  }

  fn val(&self, p: &EvalPoint) -> f64 {
    match self.xy(p, 0) {
      Some([x, y]) => self.f.val(x.0, y.0),
      None => f64::NAN,
    }
  }
  fn grad(&self, p: &EvalPoint) -> Vec<f64> {
    let Some([x, y]) = self.xy(p, 1) else {
      return vec![f64::NAN; self.dim];
    };
    let gf = self.f.grad(x.0, y.0);
    x.1
      .iter()
      .zip(&y.1)
      .map(|(dx, dy)| gf[0] * dx + gf[1] * dy)
      .collect()
  }
  fn hess(&self, p: &EvalPoint) -> Vec<f64> {
    let n = self.dim;
    let Some([x, y]) = self.xy(p, 2) else {
      return vec![f64::NAN; n * n];
    };
    let (gf, hf) = (self.f.grad(x.0, y.0), self.f.hess(x.0, y.0));
    let d = [&x.1, &y.1];
    let mut h = vec![0.0; n * n];
    for s in 0..n {
      for r in 0..n {
        let mut v = gf[0] * x.2[r + n * s] + gf[1] * y.2[r + n * s];
        for a in 0..2 {
          for b in 0..2 {
            v += hf[a + 2 * b] * d[a][r] * d[b][s];
          }
        }
        h[r + n * s] = v;
      }
    }
    h
  }
}

impl GlobalFunction {
  /// The crack tip function f of (x, y) = (secondary, primary) values of
  /// the level set ls (GetFEM global_function_on_level_set).
  pub fn on_level_set(ls: &LevelSet, f: &CrackTipFunction) -> Result<Self> {
    Self::on_level_sets(std::slice::from_ref(ls), f)
  }

  /// The same with several level sets: on each convex, the one whose
  /// crack tip is the closest (GetFEM global_function_on_level_sets).
  pub fn on_level_sets(lss: &[LevelSet], f: &CrackTipFunction) -> Result<Self> {
    let Some(l0) = lss.first() else {
      return Err(Error::InvalidArgument(
        "The list of level sets should contain at least one level set.".into(),
      ));
    };
    if lss.iter().any(|l| !l.has_secondary()) {
      return Err(Error::InvalidArgument(
        "global_function_on_level_set: a level set with a secondary function".into(),
      ));
    }
    Ok(GlobalFunction::new(OnLevelSets {
      lss: lss.to_vec(),
      f: f.clone(),
      dim: l0.mesh_fem().linked_mesh().dim(),
      cache: Mutex::new(None),
    }))
  }
}
