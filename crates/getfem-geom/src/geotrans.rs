//! Geometric transformations X = G N(xref) (port of bgeot_geometric_trans).
//!
//! K = G grad N is N x P, B = K (K^T K)^-1 its pseudo-inverse (B = K^-T when
//! square), J = sqrt(det(K^T K)) (|det K| when square).

use std::sync::Arc;

use crate::convex_ref::{self as cr, PRef};
use crate::convex_structure::PCvs;
use crate::intern::Interner;
use crate::naming::Descriptor;
use crate::poly::{Polynomial, RationalFraction};
use crate::{Error, Result};
use getfem_linalg::tensor::{Lu, Mat, singular_value_bounds};

pub type PGeoTrans = Arc<GeoTrans>;
type Poly = Polynomial<f64>;
type Shape = RationalFraction<f64>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GtKey {
  PK { n: usize, k: usize },
  Product(Box<GtKey>, Box<GtKey>),
  LinearProduct(Box<GtKey>, Box<GtKey>),
  Pyramid(usize),
  Q2Incomplete(usize),
  PyramidQ2Incomplete,
  PrismIncompleteP2,
}

impl GtKey {
  /// Some(k) if the key is a product chain of degree-k segments of dim n.
  fn qk(&self) -> Option<(usize, usize)> {
    match self {
      GtKey::PK { n: 1, k } => Some((1, *k)),
      GtKey::Product(a, b) => match (a.qk(), b.as_ref()) {
        (Some((n, k)), GtKey::PK { n: 1, k: k2 }) if k == *k2 => Some((n + 1, k)),
        _ => None,
      },
      _ => None,
    }
  }

  /// Some(n) if the key is a linear product chain of P1 segments.
  fn linear_qk(&self) -> Option<usize> {
    match self {
      GtKey::PK { n: 1, k: 1 } => Some(1),
      GtKey::LinearProduct(a, b) if **b == (GtKey::PK { n: 1, k: 1 }) => {
        a.linear_qk().map(|n| n + 1)
      }
      _ => None,
    }
  }

  pub fn name(&self) -> String {
    match self {
      GtKey::PK { n, k } => format!("GT_PK({n},{k})"),
      GtKey::Product(a, b) => {
        if let Some((n, k)) = self.qk() {
          return format!("GT_QK({n},{k})");
        }
        if let (GtKey::PK { n, k }, GtKey::PK { n: 1, k: k2 }) = (a.as_ref(), b.as_ref())
          && k == k2
        {
          return format!("GT_PRISM({},{k})", n + 1);
        }
        format!("GT_PRODUCT({},{})", a.name(), b.name())
      }
      GtKey::LinearProduct(a, b) => match self.linear_qk() {
        Some(n) => format!("GT_LINEAR_QK({n})"),
        None => format!("GT_LINEAR_PRODUCT({},{})", a.name(), b.name()),
      },
      GtKey::Pyramid(k) => format!("GT_PYRAMID({k})"),
      GtKey::Q2Incomplete(n) => format!("GT_Q2_INCOMPLETE({n})"),
      GtKey::PyramidQ2Incomplete => "GT_PYRAMID_Q2_INCOMPLETE".into(),
      GtKey::PrismIncompleteP2 => "GT_PRISM_INCOMPLETE_P2".into(),
    }
  }
}

pub struct GeoTrans {
  key: GtKey,
  cvr: PRef,
  linear: bool,
  complexity: usize,
  vertices: Vec<usize>,
  funcs: Vec<Shape>,
  grad: Vec<Vec<Shape>>,
  hess: Vec<Vec<Shape>>,
}

impl std::fmt::Debug for GeoTrans {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}", self.name())
  }
}

impl PartialEq for GeoTrans {
  fn eq(&self, o: &Self) -> bool {
    self.key == o.key
  }
}

impl GeoTrans {
  fn new(
    key: GtKey,
    cvr: PRef,
    linear: bool,
    complexity: usize,
    vertices: Vec<usize>,
    funcs: Vec<Shape>,
  ) -> Self {
    let p = cvr.dim();
    let grad: Vec<Vec<Shape>> = funcs
      .iter()
      .map(|f| (0..p).map(|j| f.derived(j)).collect())
      .collect();
    let hess = grad
      .iter()
      .map(|g| (0..p * p).map(|jk| g[jk % p].derived(jk / p)).collect())
      .collect();
    GeoTrans {
      key,
      cvr,
      linear,
      complexity,
      vertices,
      funcs,
      grad,
      hess,
    }
  }

  pub fn key(&self) -> &GtKey {
    &self.key
  }

  pub fn name(&self) -> String {
    self.key.name()
  }

  /// Dimension of the reference element.
  pub fn dim(&self) -> usize {
    self.cvr.dim()
  }

  /// Affine: K constant on the element.
  pub fn is_linear(&self) -> bool {
    self.linear
  }

  pub fn complexity(&self) -> usize {
    self.complexity
  }

  pub fn nb_points(&self) -> usize {
    self.cvr.nb_points()
  }

  pub fn convex_ref(&self) -> &PRef {
    &self.cvr
  }

  pub fn structure(&self) -> &PCvs {
    self.cvr.structure()
  }

  pub fn basic_structure(&self) -> PCvs {
    self.cvr.structure().basic()
  }

  pub fn geometric_nodes(&self) -> &[Vec<f64>] {
    self.cvr.points()
  }

  pub fn normals(&self) -> &[Vec<f64>] {
    self.cvr.normals()
  }

  /// Local indices of the nodes that are vertices.
  pub fn vertices(&self) -> &[usize] {
    &self.vertices
  }

  /// Shape functions N_i (polynomials stored as fractions with denominator 1).
  pub fn basis(&self) -> &[RationalFraction<f64>] {
    &self.funcs
  }

  pub fn val(&self, xref: &[f64]) -> Vec<f64> {
    self.funcs.iter().map(|f| f.eval(xref)).collect()
  }

  /// nb_points x P
  pub fn grad(&self, xref: &[f64]) -> Mat {
    let (r, p) = (self.nb_points(), self.dim());
    let mut m = Mat::zeros(&[r, p]);
    for i in 0..r {
      for j in 0..p {
        m[(i, j)] = self.grad[i][j].eval(xref);
      }
    }
    m
  }

  /// nb_points x P^2, column n*P+m holds d2/dx_n dx_m
  pub fn hess(&self, xref: &[f64]) -> Mat {
    let (r, p) = (self.nb_points(), self.dim());
    let mut m = Mat::zeros(&[r, p * p]);
    for i in 0..r {
      for jk in 0..p * p {
        m[(i, jk)] = self.hess[i][jk].eval(xref);
      }
    }
    m
  }

  fn check_g(&self, g: &Mat) -> Result<()> {
    if g.ncols() != self.nb_points() || g.nrows() < self.dim() {
      return Err(Error::InvalidArgument(format!(
        "{}: G is {:?}, expected N x {} with N >= {}",
        self.name(),
        g.shape(),
        self.nb_points(),
        self.dim()
      )));
    }
    Ok(())
  }

  fn check_x(&self, xref: &[f64]) -> Result<()> {
    if xref.len() != self.dim() {
      return Err(Error::InvalidArgument(format!(
        "{}: reference point of dim {}",
        self.name(),
        xref.len()
      )));
    }
    Ok(())
  }

  /// Real point of reference point xref on the element with nodes G (N x nb_points).
  pub fn transform(&self, xref: &[f64], g: &Mat) -> Result<Vec<f64>> {
    self.check_g(g)?;
    self.check_x(xref)?;
    let v = self.val(xref);
    let mut p = vec![0.0; g.nrows()];
    for (l, a) in v.iter().enumerate() {
      p.iter_mut()
        .zip(g.col(l))
        .for_each(|(pi, gi)| *pi += a * gi);
    }
    Ok(p)
  }

  pub fn project_into_reference_convex(&self, xref: &mut [f64]) -> Result<()> {
    self.cvr.project_into(xref)
  }

  /// Geometric quantities at reference point xref.
  pub fn at(&self, g: &Mat, xref: &[f64]) -> Result<PointGeom> {
    self.check_g(g)?;
    self.check_x(xref)?;
    let k = g.matmul(&self.grad(xref));
    let (b, j) = pseudo_inverse(&k)?;
    Ok(PointGeom {
      x: self.transform(xref, g)?,
      k,
      b,
      j,
    })
  }

  /// Real -> reference: damped Newton (Gauss-Newton if codim > 0) on
  /// |X(xref) - x|. Converged: Newton step <= eps in reference space and
  /// |X(xref) - x| <= 1e3 eps times the element extent, so the result depends
  /// neither on the length unit nor on the aspect ratio. `inside` uses is_in(xref) < eps
  /// (reference space). With `project`, iterates stay in the reference convex.
  pub fn invert(&self, g: &Mat, x: &[f64], eps: f64, project: bool) -> Result<Inverse> {
    self.check_g(g)?;
    if x.len() != g.nrows() {
      return Err(Error::InvalidArgument(format!(
        "{}: real point of dim {}",
        self.name(),
        x.len()
      )));
    }
    let len = element_extent(g);
    // nodes and point relative to the first node: the inversion does not
    // depend on the element position (rounding grows with |x|)
    let (grel, xrel) = relative_to_first_node(g, x);
    let resid = |xref: &[f64]| -> Result<(Vec<f64>, f64)> {
      let d: Vec<f64> = self
        .transform_rel(xref, &grel)
        .iter()
        .zip(&xrel)
        .map(|(a, b)| a - b)
        .collect();
      let r = d.iter().map(|v| v * v).sum::<f64>().sqrt() / len;
      Ok((d, r))
    };
    // B^T d: Newton (Gauss-Newton) step in reference space
    let bt = |b: &Mat, d: &[f64]| -> Vec<f64> {
      (0..self.dim())
        .map(|j| (0..grel.nrows()).map(|i| b[(i, j)] * d[i]).sum())
        .collect()
    };
    let norm = |v: &[f64]| v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    // affine: xref = x0ref + B^T xrel (xrel relative to the first node),
    // exact (GetFEM invert_lin; least squares in codimension > 0)
    if self.linear && !project {
      let x0ref = &self.geometric_nodes()[0];
      if let Ok((b, _)) = pseudo_inverse(&self.k_rel(x0ref, &grel)) {
        let xref: Vec<f64> = x0ref
          .iter()
          .zip(bt(&b, &xrel))
          .map(|(a, s)| a + s)
          .collect();
        let (_, res) = resid(&xref)?;
        let inside = self.cvr.is_in(&xref)? <= eps;
        return Ok(Inverse {
          xref,
          converged: res <= 1e3 * eps,
          inside,
          residual: res,
        });
      }
    }
    // start: nearest node, or linearization through the direction points
    let mut xref = self.initial_guess(&grel, &xrel)?;
    if project {
      self.cvr.project_into(&mut xref)?;
    }
    // iterates stay within the reference box enlarged by its size: far outside,
    // high-degree bases lose all accuracy (thin curved elements)
    let (lo, hi): (Vec<f64>, Vec<f64>) = (0..self.dim())
      .map(|k| {
        let (a, b) = self
          .geometric_nodes()
          .iter()
          .fold((f64::MAX, f64::MIN), |(a, b), noderef| {
            (a.min(noderef[k]), b.max(noderef[k]))
          });
        (a - (b - a), b + (b - a))
      })
      .unzip();
    let clamp = |ptref: &mut [f64]| {
      ptref
        .iter_mut()
        .zip(lo.iter().zip(&hi))
        .for_each(|(v, (a, b))| *v = v.clamp(*a, *b))
    };
    clamp(&mut xref);
    let (mut d, mut res) = resid(&xref)?;
    // converged when the Newton step is below eps in reference space (scale
    // and anisotropy free) and the point is on the element (real residual)
    let mut step_norm = f64::INFINITY;
    let max_it = if self.linear { 3 } else { 100 };
    for _ in 0..max_it {
      // at rounding level: converged (below), no new gradient (as GetFEM)
      if res <= 64.0 * f64::EPSILON {
        break;
      }
      let b = match pseudo_inverse(&self.k_rel(&xref, &grel)) {
        Ok((b, _)) => b,
        Err(Error::Singular) => break,
        Err(e) => return Err(e),
      };
      let dx = bt(&b, &d);
      step_norm = norm(&dx);
      if step_norm <= eps {
        // the last (quadratically small) step still improves the point
        let mut xnref: Vec<f64> = xref.iter().zip(&dx).map(|(a, s)| a - s).collect();
        if project {
          self.cvr.project_into(&mut xnref)?;
        }
        clamp(&mut xnref);
        let (_, rn) = resid(&xnref)?;
        if rn <= res {
          (xref, res) = (xnref, rn);
        }
        break;
      }
      // damped Newton: accept when the residual decreases or the simplified
      // step B^T r(x_new) is shorter (natural monotonicity, Deuflhard); the
      // latter, in reference space, is not fooled by anisotropic elements
      let mut factor = 1.0;
      let mut accepted = false;
      while factor > 1e-12 {
        let mut xnref: Vec<f64> = xref.iter().zip(&dx).map(|(a, s)| a - factor * s).collect();
        if project {
          self.cvr.project_into(&mut xnref)?;
        }
        clamp(&mut xnref);
        let (dn, rn) = resid(&xnref)?;
        if rn < res || norm(&bt(&b, &dn)) < (1.0 - 0.25 * factor) * step_norm {
          (xref, d, res, accepted) = (xnref, dn, rn, true);
          break;
        }
        factor *= 0.5;
      }
      if !accepted {
        break;
      }
    }
    // at round-off level nothing more can be gained (thin elements: the
    // reference accuracy is limited to about machine eps * aspect ratio)
    let converged = res <= 1e3 * eps && (step_norm <= eps || res <= 64.0 * f64::EPSILON);
    let inside = self.cvr.is_in(&xref)? <= eps;
    Ok(Inverse {
      xref,
      converged,
      inside,
      residual: res,
    })
  }

  /// X(xref) - g_0 = sum over j >= 1 of N_j(xref) (g_j - g_0) (partition of
  /// unity), with grel the nodes 1.. relative to node 0 (relative_to_first_node)
  fn transform_rel(&self, xref: &[f64], grel: &Mat) -> Vec<f64> {
    let mut prel = vec![0.0; grel.nrows()];
    for (gj, f) in grel.data().chunks(grel.nrows()).zip(&self.funcs[1..]) {
      let a = f.eval(xref);
      prel.iter_mut().zip(gj).for_each(|(pi, gi)| *pi += a * gi);
    }
    prel
  }

  /// K = sum over j >= 1 of (g_j - g_0) grad N_j(xref)^T (the gradients sum
  /// to 0), grel the nodes 1.. relative to node 0 (relative_to_first_node)
  pub(crate) fn k_rel(&self, xref: &[f64], grel: &Mat) -> Mat {
    let mut k = Mat::zeros(&[grel.nrows(), self.dim()]);
    for (gj, dj) in grel.data().chunks(grel.nrows()).zip(&self.grad[1..]) {
      for (c, d) in dj.iter().enumerate() {
        let d = d.eval(xref);
        k.col_mut(c)
          .iter_mut()
          .zip(gj)
          .for_each(|(kc, gc)| *kc += gc * d);
      }
    }
    k
  }

  /// grel, xrel relative to the first node (relative_to_first_node)
  fn initial_guess(&self, grel: &Mat, xrel: &[f64]) -> Result<Vec<f64>> {
    // coordinate r of node j (node 0: the origin)
    let nd = |j: usize, r: usize| if j == 0 { 0.0 } else { grel[(r, j - 1)] };
    let dist = |j: usize| {
      (0..xrel.len())
        .map(|r| (nd(j, r) - xrel[r]).powi(2))
        .sum::<f64>()
    };
    let jn = (0..self.nb_points())
      .min_by(|&a, &b| dist(a).total_cmp(&dist(b)))
      .unwrap();
    let nearestref = self.geometric_nodes()[jn].clone();
    let dir = self.structure().ind_dir_points();
    if dir.len() != self.dim() + 1 {
      return Ok(nearestref);
    }
    // affine map through the direction points
    let (n, p) = (grel.nrows(), self.dim());
    let r0ref = &self.geometric_nodes()[dir[0]];
    let mut kl = Mat::zeros(&[n, p]);
    let mut kr = Mat::zeros(&[p, p]);
    for (c, &i) in dir[1..].iter().enumerate() {
      for r in 0..n {
        kl[(r, c)] = nd(i, r) - nd(dir[0], r);
      }
      for r in 0..p {
        kr[(r, c)] = self.geometric_nodes()[i][r] - r0ref[r];
      }
    }
    let Ok((bl, _)) = pseudo_inverse(&kl) else {
      return Ok(nearestref);
    };
    let y: Vec<f64> = (0..p)
      .map(|c| (0..n).map(|r| bl[(r, c)] * (xrel[r] - nd(dir[0], r))).sum())
      .collect();
    let linref: Vec<f64> = (0..p)
      .map(|r| r0ref[r] + (0..p).map(|c| kr[(r, c)] * y[c]).sum::<f64>())
      .collect();
    let rl = self
      .transform_rel(&linref, grel)
      .iter()
      .zip(xrel)
      .map(|(a, b)| (a - b) * (a - b))
      .sum::<f64>();
    Ok(if rl < dist(jn) { linref } else { nearestref })
  }

  /// For an affine transformation, the largest distance (relative to the
  /// element extent) between a node and the image of its reference node
  /// under the affine map defined by the direction points; 0 otherwise.
  pub fn affinity_error(&self, g: &Mat) -> Result<f64> {
    self.check_g(g)?;
    if !self.linear {
      return Ok(0.0);
    }
    let len = element_extent(g);
    let mut err = 0.0f64;
    for (j, xref) in self.geometric_nodes().iter().enumerate() {
      let x = self.transform(xref, g)?;
      let d = x
        .iter()
        .zip(g.col(j))
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
      err = err.max(d / len);
    }
    Ok(err)
  }

  /// Element size: largest singular value of K over the geometric nodes
  /// times sqrt(P). GetFEM uses sqrt(N), which overestimates codim > 0.
  pub fn element_size(&self, g: &Mat) -> Result<f64> {
    self.check_g(g)?;
    let nodesref = if self.linear {
      &self.geometric_nodes()[..1]
    } else {
      self.geometric_nodes()
    };
    let mut q = 0.0f64;
    for xref in nodesref {
      let k = g.matmul(&self.grad(xref));
      q = q.max(singular_value_bounds(&k).1);
    }
    Ok(q * (self.dim() as f64).sqrt())
  }
}

/// Result of an inversion; `residual` is |X(xref) - x| / element extent.
#[derive(Clone, Debug)]
pub struct Inverse {
  pub xref: Vec<f64>,
  pub converged: bool,
  pub inside: bool,
  pub residual: f64,
}

/// the nodes 1.. of G (N x (nb_points - 1)) and the point x relative to
/// the first node of G, the origin
pub fn relative_to_first_node(g: &Mat, x: &[f64]) -> (Mat, Vec<f64>) {
  let (n, g0) = (g.nrows(), g.col(0));
  let mut grel = Vec::with_capacity(g.len() - n);
  for gj in g.data()[n..].chunks(n) {
    grel.extend(gj.iter().zip(g0).map(|(a, b)| a - b));
  }
  let grel = Mat::from_data(&[n, g.ncols() - 1], grel).unwrap();
  (grel, x.iter().zip(g0).map(|(a, b)| a - b).collect())
}

/// Largest extent of the node cloud (characteristic length).
pub fn element_extent(g: &Mat) -> f64 {
  (0..g.nrows())
    .map(|i| {
      let (lo, hi) = (0..g.ncols()).fold((f64::MAX, f64::MIN), |(l, h), j| {
        (l.min(g[(i, j)]), h.max(g[(i, j)]))
      });
      hi - lo
    })
    .fold(0.0, f64::max)
    .max(f64::MIN_POSITIVE)
}

/// Geometry at one point: real point, K (N x P), B (N x P), J.
#[derive(Clone, Debug)]
pub struct PointGeom {
  pub x: Vec<f64>,
  pub k: Mat,
  pub b: Mat,
  /// det K if square (signed), sqrt(det K^T K) otherwise
  pub j: f64,
}

/// Inverse in place of a column-major n x n matrix, n <= 3, and its
/// determinant (GetFEM bgeot::lu_inverse, closed forms); None if singular.
fn small_inverse(a: &mut [f64]) -> Option<f64> {
  let det = match a.len() {
    1 => {
      let det = a[0];
      a[0] = 1.0 / det;
      det
    }
    4 => {
      let (x, b, c, d) = (a[0], a[2], a[1], a[3]);
      let det = x * d - b * c;
      a[0] = d / det;
      a[1] /= -det;
      a[2] /= -det;
      a[3] = x / det;
      det
    }
    9 => {
      let a0 = a[4] * a[8] - a[5] * a[7];
      let a1 = a[5] * a[6] - a[3] * a[8];
      let a2 = a[3] * a[7] - a[4] * a[6];
      let det = a[0] * a0 + a[1] * a1 + a[2] * a2;
      let a3 = a[2] * a[7] - a[1] * a[8];
      let a6 = a[1] * a[5] - a[2] * a[4];
      let a4 = a[0] * a[8] - a[2] * a[6];
      let a7 = a[2] * a[3] - a[0] * a[5];
      let a5 = a[1] * a[6] - a[0] * a[7];
      let a8 = a[0] * a[4] - a[1] * a[3];
      a.copy_from_slice(&[
        a0 / det,
        a3 / det,
        a6 / det,
        a1 / det,
        a4 / det,
        a7 / det,
        a2 / det,
        a5 / det,
        a8 / det,
      ]);
      det
    }
    _ => return None,
  };
  (det != 0.0).then_some(det)
}

/// Determinant of a column-major n x n matrix (bgeot::lu_det): closed
/// forms up to 3, LU otherwise.
pub fn lu_det(a: &[f64], n: usize) -> f64 {
  match n {
    1 => a[0],
    2 => a[0] * a[3] - a[1] * a[2],
    3 => {
      let a0 = a[4] * a[8] - a[5] * a[7];
      let a3 = a[5] * a[6] - a[3] * a[8];
      let a6 = a[3] * a[7] - a[4] * a[6];
      a[0] * a0 + a[1] * a3 + a[2] * a6
    }
    _ => Mat::from_data(&[n, n], a[..n * n].to_vec())
      .and_then(|m| Lu::new(&m))
      .map_or(0.0, |lu| lu.det()),
  }
}

/// Inverse in place of a column-major n x n matrix and its determinant
/// (bgeot::lu_inverse); Singular if not invertible.
pub fn lu_inverse(a: &mut [f64], n: usize) -> Result<f64> {
  if n <= 3 {
    return small_inverse(&mut a[..n * n]).ok_or(Error::Singular);
  }
  let lu = Lu::new(&Mat::from_data(&[n, n], a[..n * n].to_vec())?)?;
  a[..n * n].copy_from_slice(lu.inverse().data());
  Ok(lu.det())
}

/// c = a b, column-major slices a (m x k), b (k x n), as Tensor::matmul.
pub fn matmul_into(a: &[f64], b: &[f64], (m, k, n): (usize, usize, usize), c: &mut [f64]) {
  // columns of c: sum_l b[l + k j] a[:, l] (the first term written)
  let a = &a[..m * k];
  for (cj, bj) in c[..m * n].chunks_exact_mut(m).zip(b.chunks_exact(k)) {
    for (l, (&blj, al)) in bj.iter().zip(a.chunks_exact(m)).enumerate() {
      let col = cj.iter_mut().zip(al);
      if l == 0 {
        col.for_each(|(o, x)| *o = x * blj);
      } else {
        col.for_each(|(o, x)| *o += x * blj);
      }
    }
  }
}

/// K = G pc (GetFEM geometric_trans::compute_K_matrix): G n x nb (the
/// nodes), pc nb x p (the gradients of the base functions), column-major;
/// fixed sizes for the usual dimensions (unrolled), matmul_into otherwise
#[allow(non_snake_case)]
pub fn compute_K_matrix(g: &[f64], pc: &[f64], (n, nb, p): (usize, usize, usize), k: &mut [f64]) {
  match (n, p) {
    (2, 2) => k_fixed::<2, 2>(g, pc, nb, k),
    (3, 3) => k_fixed::<3, 3>(g, pc, nb, k),
    (3, 2) => k_fixed::<3, 2>(g, pc, nb, k),
    _ => matmul_into(g, pc, (n, nb, p), k),
  }
}

fn k_fixed<const N: usize, const P: usize>(g: &[f64], pc: &[f64], nb: usize, k: &mut [f64]) {
  let mut acc = [[0.0; N]; P];
  for (l, gl) in g[..N * nb].chunks_exact(N).enumerate() {
    for (j, aj) in acc.iter_mut().enumerate() {
      let b = pc[l + nb * j];
      for (a, x) in aj.iter_mut().zip(gl) {
        *a += x * b;
      }
    }
  }
  for (kj, aj) in k[..N * P].chunks_exact_mut(N).zip(&acc) {
    kj.copy_from_slice(aj);
  }
}

/// b = a^T, column-major N x N
fn transpose_into<const N: usize>(a: &[f64], b: &mut [f64]) {
  let (a, b) = (&a[..N * N], &mut b[..N * N]);
  for r in 0..N {
    for c in 0..N {
      b[r + N * c] = a[c + N * r];
    }
  }
}

/// B (n x p, into b) and J of K (n x p, column-major), as pseudo_inverse
/// (closed forms up to 3).
pub fn pseudo_inverse_small(k: &[f64], n: usize, p: usize, b: &mut [f64]) -> Result<f64> {
  let (mut small, mut big) = ([0.0; 9], vec![]);
  let a: &mut [f64] = if p <= 3 {
    &mut small
  } else {
    big.resize(p * p, 0.0);
    &mut big
  };
  if n == p {
    a[..n * n].copy_from_slice(&k[..n * n]);
    let j = lu_inverse(a, n)?;
    // B = K^-T (fixed sizes unrolled)
    match n {
      1 => b[0] = a[0],
      2 => transpose_into::<2>(a, b),
      3 => transpose_into::<3>(a, b),
      _ => {
        for r in 0..n {
          for c in 0..n {
            b[r + n * c] = a[c + n * r];
          }
        }
      }
    }
    return Ok(j);
  }
  // K^T K, sums from -0.0 as Tensor::tmatmul
  for j in 0..p {
    for i in 0..p {
      a[i + p * j] = (0..n).map(|l| k[l + i * n] * k[l + j * n]).sum();
    }
  }
  let d = lu_inverse(a, p)?;
  matmul_into(k, a, (n, p, p), b);
  Ok(d.abs().sqrt())
}

/// (B, J) from K; B = K (K^T K)^-1, or K^-T when square.
pub fn pseudo_inverse(k: &Mat) -> Result<(Mat, f64)> {
  let (n, p) = (k.nrows(), k.ncols());
  // closed forms as GetFEM: square up to 3, K^T K up to 3
  if (n == p && n <= 3) || (n != p && p <= 3) {
    let mut b = vec![0.0; n * p];
    let j = pseudo_inverse_small(k.data(), n, p, &mut b)?;
    return Ok((Mat::from_data(&[n, p], b)?, j));
  }
  if n == p {
    let lu = Lu::new(k)?;
    let j = lu.det();
    Ok((lu.inverse().transposed(), j))
  } else {
    let ktk = k.tmatmul(k);
    let lu = Lu::new(&ktk)?;
    let j = lu.det().abs().sqrt();
    Ok((k.matmul(&lu.inverse()), j))
  }
}

static CACHE: Interner<GtKey, GeoTrans> = Interner::new();

/// Nodes with all coordinates in {-1, 0, 1} (1e-10), as GetFEM.
fn standard_vertices(cvr: &PRef) -> Vec<usize> {
  let is_v = |ptref: &[f64]| {
    ptref
      .iter()
      .all(|&x| x.abs() <= 1e-10 || (x - 1.0).abs() <= 1e-10 || (x + 1.0).abs() <= 1e-10)
  };
  (0..cvr.nb_points())
    .filter(|&i| is_v(&cvr.points()[i]))
    .collect()
}

/// Lagrange basis function of node i of the degree-K simplex.
fn pk_basis(n: usize, k: usize, noderef: &[f64]) -> Poly {
  let one = Poly::one(n);
  let mut l0 = one.clone();
  for j in 0..n {
    l0 -= &Poly::var(n, j);
  }
  let mut w = vec![0usize; n + 1];
  w[0] = k;
  for nn in 1..=n {
    w[nn] = (0.5 + noderef[nn - 1] * k as f64).floor() as usize;
    w[0] -= w[nn];
  }
  let kf = k as f64;
  let mut p = one.clone();
  for (nn, &wn) in w.iter().enumerate() {
    let l = if nn == 0 {
      l0.clone()
    } else {
      Poly::var(n, nn - 1)
    };
    for j in 0..wn {
      let jf = j as f64;
      p *= &(&(&l * (kf / (jf + 1.0))) - &(&one * (jf / (jf + 1.0))));
    }
  }
  p
}

pub fn pk(n: usize, k: usize) -> Result<PGeoTrans> {
  if k == 0 {
    return Err(Error::InvalidArgument(
      "GT_PK: degree 0 has no vertices".into(),
    ));
  }
  Ok(CACHE.get_or(&GtKey::PK { n, k }, || {
    let cvr = cr::simplex_of_reference_k(n, k);
    let funcs = cvr
      .points()
      .iter()
      .map(|noderef| pk_basis(n, k, noderef).into())
      .collect();
    let vertices = standard_vertices(&cvr);
    GeoTrans::new(GtKey::PK { n, k }, cvr, k == 1, k, vertices, funcs)
  }))
}

pub fn product(a: &PGeoTrans, b: &PGeoTrans) -> PGeoTrans {
  let key = GtKey::Product(Box::new(a.key.clone()), Box::new(b.key.clone()));
  CACHE.get_or(&key.clone(), || {
    let cvr = cr::convex_ref_product(&a.cvr, &b.cvr);
    let n1 = a.nb_points();
    let mut funcs = Vec::with_capacity(n1 * b.nb_points());
    for fb in &b.funcs {
      for fa in &a.funcs {
        funcs.push(fa.direct_product(fb));
      }
    }
    let vertices = b
      .vertices
      .iter()
      .flat_map(|&i2| a.vertices.iter().map(move |&i1| i1 + i2 * n1))
      .collect();
    GeoTrans::new(
      key,
      cvr,
      false,
      a.complexity * b.complexity,
      vertices,
      funcs,
    )
  })
}

pub fn qk(n: usize, k: usize) -> Result<PGeoTrans> {
  if n == 0 {
    return Err(Error::InvalidArgument("GT_QK: dimension 0".into()));
  }
  if n == 1 {
    return pk(1, k);
  }
  Ok(product(&qk(n - 1, k)?, &pk(1, k)?))
}

pub fn prism_pk(n: usize, k: usize) -> Result<PGeoTrans> {
  if n == 0 {
    return Err(Error::InvalidArgument("GT_PRISM: dimension 0".into()));
  }
  Ok(product(&pk(n - 1, k)?, &pk(1, k)?))
}

/// Product with an affine map: only the direction points of the product
/// carry the P1 simplex basis (GetFEM GT_LINEAR_PRODUCT).
pub fn linear_product(a: &PGeoTrans, b: &PGeoTrans) -> Result<PGeoTrans> {
  if !a.linear || !b.linear {
    return Err(Error::InvalidArgument(format!(
      "linear product of non-linear {} and {}",
      a.name(),
      b.name()
    )));
  }
  let key = GtKey::LinearProduct(Box::new(a.key.clone()), Box::new(b.key.clone()));
  Ok(CACHE.get_or(&key.clone(), || {
    let cvr = cr::convex_ref_product(&a.cvr, &b.cvr);
    let n = cvr.dim();
    let p1 = pk(n, 1).expect("P1 simplex");
    let mut funcs = vec![Shape::from_poly(Poly::zero(n)); cvr.nb_points()];
    for (i, &d) in cvr.structure().ind_dir_points().iter().enumerate() {
      funcs[d] = p1.funcs[i].clone();
    }
    let n1 = a.nb_points();
    let vertices = b
      .vertices
      .iter()
      .flat_map(|&i2| a.vertices.iter().map(move |&i1| i1 + i2 * n1))
      .collect();
    GeoTrans::new(
      key,
      cvr,
      true,
      a.complexity.max(b.complexity),
      vertices,
      funcs,
    )
  }))
}

/// Affine map of the parallelepiped (GT_LINEAR_QK).
pub fn linear_qk(n: usize) -> Result<PGeoTrans> {
  match n {
    0 => Err(Error::InvalidArgument("GT_LINEAR_QK: dimension 0".into())),
    1 => pk(1, 1),
    _ => linear_product(&linear_qk(n - 1)?, &pk(1, 1)?),
  }
}

fn polys(n: usize, s: &str) -> Vec<Shape> {
  s.split(';')
    .filter(|t| !t.trim().is_empty())
    .map(|t| {
      crate::poly::parse::<f64>(n, t)
        .expect("builtin basis")
        .into()
    })
    .collect()
}

fn p3(s: &str) -> Poly {
  crate::poly::parse::<f64>(3, s).expect("builtin basis")
}

fn rf(p: &Poly) -> Shape {
  Shape::from_poly(p.clone())
}

/// Pyramid of degree 1 or 2 (rational basis, reference apex at (0,0,1)).
pub fn pyramid(k: usize) -> Result<PGeoTrans> {
  let cvr = cr::pyramid_qk_of_reference(k)?;
  Ok(CACHE.get_or(&GtKey::Pyramid(k), || {
    let funcs = if k == 1 {
      let q = Shape::new(p3("x*y"), p3("1-z"));
      vec![
        &(&rf(&p3("1-x-y-z")) + &q) * 0.25,
        &(&rf(&p3("1+x-y-z")) - &q) * 0.25,
        &(&rf(&p3("1-x+y-z")) - &q) * 0.25,
        &(&rf(&p3("1+x+y-z")) + &q) * 0.25,
        rf(&p3("z")),
      ]
    } else {
      let (xi0, xi1, xi2, xi3) = (
        p3("(1-z-x)*0.5"),
        p3("(1-z-y)*0.5"),
        p3("(1-z+x)*0.5"),
        p3("(1-z+y)*0.5"),
      );
      let (xy, z, un_z) = (p3("x*y"), p3("z"), p3("1-z"));
      let q = Shape::new(Poly::one(3), un_z.clone());
      let qq = &q * &q;
      let zu = &z * &un_z;
      let m = |a: &Poly, b: &Poly| a * b;
      let lin = |xi: &Poly| &(xi * 2.0) - &un_z;
      let qqp = |p: Poly| &qq * &rf(&p);
      let qzp = |p: Poly| &(&q * &rf(&z)) * &rf(&(&p * 4.0));
      vec![
        qqp(&m(&xi0, &xi1) * &(&xy - &zu)),
        qqp(&(&m(&m(&xi0, &xi1), &xi2) * &lin(&xi1)) * 4.0),
        qqp(&m(&xi1, &xi2) * &(&(-&xy) - &zu)),
        qqp(&(&m(&m(&xi3, &xi0), &xi1) * &lin(&xi0)) * 4.0),
        qqp(&m(&m(&xi0, &xi1), &m(&xi2, &xi3)) * 16.0),
        qqp(&(&m(&m(&xi1, &xi2), &xi3) * &lin(&xi2)) * 4.0),
        qqp(&m(&xi3, &xi0) * &(&(-&xy) - &zu)),
        qqp(&(&m(&m(&xi2, &xi3), &xi0) * &lin(&xi3)) * 4.0),
        qqp(&m(&xi2, &xi3) * &(&xy - &zu)),
        qzp(m(&xi0, &xi1)),
        qzp(m(&xi1, &xi2)),
        qzp(m(&xi3, &xi0)),
        qzp(m(&xi2, &xi3)),
        rf(&p3("z*(2*z-1)")),
      ]
    };
    let vertices = standard_vertices(&cvr);
    GeoTrans::new(GtKey::Pyramid(k), cvr, false, k, vertices, funcs)
  }))
}

pub fn q2_incomplete(n: usize) -> Result<PGeoTrans> {
  let cvr = cr::q2_incomplete_of_reference(n)?;
  Ok(CACHE.get_or(&GtKey::Q2Incomplete(n), || {
    let funcs = if n == 2 {
      polys(
        2,
        "1 - 2*x^2*y - 2*x*y^2 + 2*x^2 + 5*x*y + 2*y^2 - 3*x - 3*y;
         4*(x^2*y - x^2 - x*y + x);
         2*x*y*y - 2*x*x*y + 2*x*x - x*y - x;
         4*(x*y*y - x*y - y*y + y);
         4*(x*y - x*y*y);
         2*x*x*y - 2*x*y*y - x*y + 2*y*y - y;
         4*(x*y - x*x*y);
         2*x*x*y + 2*x*y*y - 3*x*y;",
      )
    } else {
      polys(
        3,
        "1 + 2*x^2*y*z + 2*x*y^2*z + 2*x*y*z^2 - 2*x^2*y - 2*x^2*z - 2*x*y^2 - 2*y^2*z - 2*y*z^2 - 2*x*z^2 - 7*x*y*z + 2*x^2 + 2*y^2 + 2*z^2 + 5*y*z + 5*x*z + 5*x*y - 3*x - 3*y - 3*z;
         4*( - x^2*y*z + x*y*z + x^2*z - x*z + x^2*y - x*y - x^2 + x);
         2*x^2*y*z - 2*x*y^2*z - 2*x*y*z^2 - 2*x^2*y - 2*x^2*z + 2*x*y^2 + 2*x*z^2 + 3*x*y*z + 2*x^2 - x*y - x*z - x;
         4*( - x*y^2*z + x*y^2 + y^2*z + x*y*z - x*y - y^2 - y*z + y);
         4*(x*y^2*z - x*y^2 - x*y*z + x*y);
         - 2*x^2*y*z + 2*x*y^2*z - 2*x*y*z^2 + 2*x^2*y - 2*x*y^2 - 2*y^2*z + 2*y*z^2 + 3*x*y*z - x*y + 2*y^2 - y*z - y;
         4*(x^2*y*z - x^2*y - x*y*z + x*y);
         - 2*x^2*y*z - 2*x*y^2*z + 2*x*y*z^2 + 2*x^2*y + 2*x*y^2 + x*y*z - 3*x*y;
         4*( - x*y*z^2 + x*z^2 + y*z^2 + x*y*z - x*z - y*z - z^2 + z);
         4*(x*y*z^2 - x*y*z - x*z^2 + x*z);
         4*(x*y*z^2 - x*y*z - y*z^2 + y*z);
         4*( - x*y*z^2 + x*y*z);
         - 2*x^2*y*z - 2*x*y^2*z + 2*x*y*z^2 + 2*x^2*z + 2*y^2*z - 2*x*z^2 - 2*y*z^2 + 3*x*y*z - x*z - y*z + 2*z^2 - z;
         4*(x^2*y*z - x^2*z - x*y*z + x*z);
         - 2*x^2*y*z + 2*x*y^2*z - 2*x*y*z^2 + 2*x^2*z + 2*x*z^2 + x*y*z - 3*x*z;
         4*(x*y^2*z - y^2*z - x*y*z + y*z);
         4*( - x*y^2*z + x*y*z);
         2*x^2*y*z - 2*x*y^2*z - 2*x*y*z^2 + 2*y^2*z + 2*y*z^2 + x*y*z - 3*y*z;
         4*( - x^2*y*z + x*y*z);
         2*x^2*y*z + 2*x*y^2*z + 2*x*y*z^2 - 5*x*y*z;",
      )
    };
    let vertices = standard_vertices(&cvr);
    GeoTrans::new(GtKey::Q2Incomplete(n), cvr, false, 2, vertices, funcs)
  }))
}

pub fn pyramid_q2_incomplete() -> PGeoTrans {
  CACHE.get_or(&GtKey::PyramidQ2Incomplete, || {
    let cvr = cr::pyramid_q2_incomplete_of_reference();
    let (xy, z, p00m) = (p3("x*y"), p3("z"), p3("1-z"));
    let (pmmm, ppmm, pmpm, pppm) = (p3("1-x-y-z"), p3("1+x-y-z"), p3("1-x+y-z"), p3("1+x+y-z"));
    let (mmm0, mpm0, mmp0, mpp0) = (
      &p3("-1-x-y") * 0.25,
      &p3("-1+x-y") * 0.25,
      &p3("-1-x+y") * 0.25,
      &p3("-1+x+y") * 0.25,
    );
    let (pp0m, pm0m, p0mm, p0pm) = (p3("1+x-z"), p3("1-x-z"), p3("1-y-z"), p3("1+y-z"));
    let fr = |p: Poly| Shape::new(p, p00m.clone());
    let m3 = |a: &Poly, b: &Poly, c: &Poly| &(&(a * b) * c) * 0.5;
    let funcs = vec![
      &rf(&(&mmm0 * &pmmm)) + &fr(&mmm0 * &xy),
      fr(m3(&pp0m, &pm0m, &p0mm)),
      &rf(&(&mpm0 * &ppmm)) - &fr(&mpm0 * &xy),
      fr(m3(&p0pm, &p0mm, &pm0m)),
      fr(m3(&p0pm, &p0mm, &pp0m)),
      &rf(&(&mmp0 * &pmpm)) - &fr(&mmp0 * &xy),
      fr(m3(&pp0m, &pm0m, &p0pm)),
      &rf(&(&mpp0 * &pppm)) + &fr(&mpp0 * &xy),
      fr(&(&pm0m * &p0mm) * &z),
      fr(&(&pp0m * &p0mm) * &z),
      fr(&(&pm0m * &p0pm) * &z),
      fr(&(&pp0m * &p0pm) * &z),
      rf(&p3("2*z^2-z")),
    ];
    let vertices = standard_vertices(&cvr);
    GeoTrans::new(GtKey::PyramidQ2Incomplete, cvr, false, 2, vertices, funcs)
  })
}

pub fn prism_incomplete_p2() -> PGeoTrans {
  CACHE.get_or(&GtKey::PrismIncompleteP2, || {
    let cvr = cr::prism_incomplete_p2_of_reference();
    let funcs = polys(
      3,
      "-2*y*z^2-2*x*z^2+2*z^2-2*y^2*z-4*x*y*z+5*y*z-2*x^2*z+5*x*z-3*z+2*y^2+4*x*y-3*y+2*x^2-3*x+1;
       4*(x*y*z+x^2*z-x*z-x*y-x^2+x);
       2*x*z^2-2*x^2*z-x*z+2*x^2-x;
       4*(y^2*z+x*y*z-y*z-y^2-x*y+y);
       4*(x*y-x*y*z);
       2*y*z^2-2*y^2*z-y*z+2*y^2-y;
       4*(y*z^2+x*z^2-z^2-y*z-x*z+z);
       4*(x*z-x*z^2);
       4*(y*z-y*z^2);
       -2*y*z^2-2*x*z^2+2*z^2+2*y^2*z+4*x*y*z-y*z+2*x^2*z-x*z-z;
       4*(-x*y*z-x^2*z+x*z);
       2*x*z^2+2*x^2*z-3*x*z;
       4*(-y^2*z-x*y*z+y*z);
       4*x*y*z;
       2*y*z^2+2*y^2*z-3*y*z;",
    );
    let vertices = standard_vertices(&cvr);
    GeoTrans::new(GtKey::PrismIncompleteP2, cvr, false, 2, vertices, funcs)
  })
}

/// Transformation from its name, e.g. "GT_QK(2,1)".
pub fn geotrans_descriptor(name: &str) -> Result<PGeoTrans> {
  from_descriptor(&Descriptor::parse(name)?)
}

fn from_descriptor(d: &Descriptor) -> Result<PGeoTrans> {
  match d.suffix("GT_") {
    Some("PK") => {
      let a = d.ints(2)?;
      pk(a[0], a[1])
    }
    Some("QK") => {
      let a = d.ints(2)?;
      qk(a[0], a[1])
    }
    Some("PRISM" | "PRISM_PK") => {
      let a = d.ints(2)?;
      prism_pk(a[0], a[1])
    }
    Some("LINEAR_PRODUCT") => {
      let m = d.methods(2)?;
      linear_product(&from_descriptor(m[0])?, &from_descriptor(m[1])?)
    }
    Some("LINEAR_QK") => linear_qk(d.ints(1)?[0]),
    Some("PYRAMID" | "PYRAMID_QK") => pyramid(d.ints(1)?[0]),
    Some("Q2_INCOMPLETE") => q2_incomplete(d.ints(1)?[0]),
    Some("PYRAMID_Q2_INCOMPLETE") => {
      d.ints(0)?;
      Ok(pyramid_q2_incomplete())
    }
    Some("PRISM_INCOMPLETE_P2") => {
      d.ints(0)?;
      Ok(prism_incomplete_p2())
    }
    Some("PRODUCT") => {
      let m = d.methods(2)?;
      Ok(product(&from_descriptor(m[0])?, &from_descriptor(m[1])?))
    }
    _ => Err(Error::InvalidArgument(format!(
      "unknown geometric transformation {d}"
    ))),
  }
}

#[cfg(test)]
mod tests;
