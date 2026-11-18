//! A mesh cut by level sets (GetFEM's getfem_mesh_level_set): for each
//! convex the level sets crossing it, and for each cut convex a mesh of
//! sub-simplices in its reference element (degree K of the level sets,
//! curved onto them), the zones (signs with respect to each level set) of
//! its sub-elements, the crack tip convexes. GetFEM's cut: points of a
//! grid projected onto the level sets, the element faces and their
//! intersections, decimated, triangulated (Delaunay, our own instead of
//! qhull), crossing sub-simplices refined by the crossing points, checked
//! by integrating the monomials of degree 1 (the grid refined otherwise).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use getfem_geom::delaunay::Delaunay;
use getfem_geom::geotrans::{PGeoTrans, geotrans_descriptor};
use getfem_geom::mesh::{Mesh, MeshData};
use getfem_geom::signed_distance::{
  RefElement, SignedDistance, curvature_radius_estimate, multi_projection, try_projection,
};
use getfem_geom::simplexify::simplexified;
use getfem_linalg::tensor::singular_value_bounds;

use crate::integration::{PIntegration, classical_approx_im};
use crate::level_set::{ConvexLevelSet, LevelSet};
use crate::{Error, Result};
use getfem_linalg::vect_dist2;

/// The signs of a sub-element with respect to each level set: '+', '-',
/// or '0' on the positive side of the secondary level set.
pub type SubZone = String;
/// Sub-zones connected through a level set's secondary part.
pub type Zone = BTreeSet<SubZone>;
pub type ZoneSet = BTreeSet<Zone>;

/// A cut convex: the sub-mesh in its reference element, the faces of
/// sub-elements on a level set, the zones.
#[derive(Clone, Debug)]
pub struct CutConvex {
  pub mesh: Arc<MeshData>,
  /// per vertex of mesh: the level sets whose primary ([0], bit i for
  /// level set i) and secondary ([1]) it is on, from the cut (GetFEM: the
  /// faces on a level set, for visualization)
  pub on: Vec<[u32; 2]>,
  pub zones: ZoneSet,
  /// per level set, its front through the element: 1 grazing an element
  /// the crack splits (all but FRONT_GRAZE of its crack surface behind the
  /// front), -1 the crack grazing an element (all but FRONT_GRAZE ahead),
  /// 0 otherwise (no front inside the element, or a front on its boundary)
  pub graze: Vec<i8>,
}

impl CutConvex {
  /// The level set whose primary face f of sub-element i is on (the
  /// first), from its vertices
  pub fn level_set_of_face(&self, i: usize, f: usize) -> Option<usize> {
    let c = self
      .mesh
      .ind_points_of_face_of_convex(i, f)
      .into_iter()
      .filter(|&p| p < self.on.len())
      .fold(u32::MAX, |c, p| c & self.on[p][0]);
    (c != 0).then(|| c.trailing_zeros() as usize)
  }

  /// Whether point p of mesh is on the front of level set l (its primary
  /// and secondary)
  pub fn on_front(&self, p: usize, l: usize) -> bool {
    self.on.get(p).is_some_and(|[a, b]| (a & b) >> l & 1 == 1)
  }
}

/// The share of a crack surface in an element beyond which the front
/// counts as grazing it (not GetFEM's): split by the crack, or not cut by
/// it, without crack tip functions.
pub const FRONT_GRAZE: f64 = 0.05;

/// A mesh with level sets.
#[derive(Clone, Debug)]
pub struct MeshLevelSet {
  mesh: Mesh,
  level_sets: Vec<LevelSet>,
  /// per convex: '+', '-' (not crossed, its sign), '0' (in the positive
  /// part of the secondary level set), '*' (crossed)
  primary_zones: BTreeMap<usize, String>,
  cut: BTreeMap<usize, CutConvex>,
  crack_tips: BTreeSet<usize>,
  /// per level set: the cut convexes containing its crack tips
  crack_tips_of: Vec<BTreeSet<usize>>,
  adapted: bool,
  /// the level sets' versions at the last adapt
  versions: Vec<u64>,
}

/// -f (GetFEM mls_of_convex(cv, i, inverted))
struct Negated<'a>(&'a dyn SignedDistance);

impl SignedDistance for Negated<'_> {
  fn val(&self, xref: &[f64]) -> f64 {
    -self.0.val(xref)
  }
  fn grad(&self, xref: &[f64]) -> (f64, Vec<f64>) {
    let (d, g) = self.0.grad(xref);
    (-d, g.into_iter().map(|v| -v).collect())
  }
  fn hess(&self, xref: &[f64]) -> Vec<f64> {
    self.0.hess(xref).into_iter().map(|v| -v).collect()
  }
}

/// max(f1, f2) (GetFEM mesher_intersection)
struct Intersection<'a>(&'a dyn SignedDistance, &'a dyn SignedDistance);

impl SignedDistance for Intersection<'_> {
  fn val(&self, xref: &[f64]) -> f64 {
    self.0.val(xref).max(self.1.val(xref))
  }
  fn grad(&self, xref: &[f64]) -> (f64, Vec<f64>) {
    if self.0.val(xref) > self.1.val(xref) {
      self.0.grad(xref)
    } else {
      self.1.grad(xref)
    }
  }
  fn hess(&self, xref: &[f64]) -> Vec<f64> {
    if self.0.val(xref) > self.1.val(xref) {
      self.0.hess(xref)
    } else {
      self.1.hess(xref)
    }
  }
}

/// fixed start points instead of GetFEM's random ones (deterministic)
fn start_point(n: usize, scale: f64) -> Vec<f64> {
  [0.37, 0.23, 0.31][..n].iter().map(|v| v * scale).collect()
}

/// The points of a cut: merged within GetFEM's node tolerance, with the
/// constraints they meet and a curvature radius.
struct PointStock<'a> {
  points: Vec<Vec<f64>>,
  constraints: Vec<u64>,
  radius: Vec<f64>,
  list: &'a [&'a dyn SignedDistance],
  radius_cv: f64,
  /// list[nbelt..] are the level sets (before: the faces)
  nbelt: usize,
  max_norm: f64,
  metric: Option<Metric>,
  /// bit k: list[k] is a secondary level set, its primary list[k - 1]
  /// (3D: needed only near the front, where both vanish)
  secondary: u64,
  /// the points by cells of side HASH (larger than the merge tolerance)
  cells: std::collections::HashMap<[i64; 3], Vec<usize>>,
}

const HASH: f64 = 1e-6;

/// growth of the cut's point spacing with the distance from the points on
/// the level sets
const GRADING: f64 = 0.5;

fn cell(xref: &[f64]) -> [i64; 3] {
  let mut c = [0i64; 3];
  for (ci, v) in c.iter_mut().zip(xref) {
    *ci = (v / HASH).floor() as i64;
  }
  c
}

/// The element's B = K^-T at its centroid (row-major) and the largest
/// singular value of K: curvature radii of the real surface, in reference
/// lengths.
#[derive(Clone, Debug)]
struct Metric {
  b: Vec<f64>,
  smax: f64,
}

/// Curvature radius of the zero set of f at xref (projected first if proj),
/// in reference lengths: with a metric (3D), half the smallest principal
/// curvature radius of the real surface over the largest stretch of the
/// element (a reference step h is at most smax h long; the half measured:
/// GetFEM's accuracy with fewer points); else GetFEM's |grad f| / max
/// |eig(Hess f)| in reference coordinates (DEVIATIONS: much smaller on
/// sheared elements, cuts 4-6 times finer per direction).
fn radius_estimate(
  f: &dyn SignedDistance,
  xref: &[f64],
  proj: bool,
  metric: Option<&Metric>,
) -> f64 {
  let Some(m) = metric else {
    return curvature_radius_estimate(f, xref, proj);
  };
  let mut xref = xref.to_vec();
  if proj {
    try_projection(f, &mut xref, true);
  }
  let n = xref.len();
  let (g, h) = real_grad_hess(f, &xref, m);
  let gn = vect_dist2(&g, &vec![0.0; n]);
  if gn <= 0.0 {
    return 1e10;
  }
  // the Hessian on the tangent plane, P H P with P = I - u u^T
  let u: Vec<f64> = g.iter().map(|v| v / gn).collect();
  let hu: Vec<f64> = (0..n)
    .map(|i| (0..n).map(|j| h[i * n + j] * u[j]).sum())
    .collect();
  let uhu: f64 = (0..n).map(|i| u[i] * hu[i]).sum();
  let rows: Vec<Vec<f64>> = (0..n)
    .map(|i| {
      (0..n)
        .map(|j| h[i * n + j] - u[i] * hu[j] - hu[i] * u[j] + u[i] * u[j] * uhu)
        .collect()
    })
    .collect();
  let r: Vec<&[f64]> = rows.iter().map(|v| v.as_slice()).collect();
  let vp = getfem_linalg::tensor::sym_eigenvalues(&getfem_linalg::tensor::Tensor::from_rows(&r))
    .into_iter()
    .fold(0.0f64, |a, v| a.max(v.abs()));
  gn / vp.max(1e-10) / m.smax / 2.0
}

/// Geometric error allowed for the sub-elements on curved level sets, in
/// reference lengths (relative to the element): the volumes and areas of
/// balls cut by the mesh to about 1e-6 (1e-5: errors of 3e-5 to 6e-5).
const GEOMETRY_TOLERANCE: f64 = 1e-6;

/// The largest cell size s whose interpolation of degree k of a surface of
/// curvature radius r is within tau_g of it, as for a circle arc: r/8
/// (s/r)^2 (k = 1), 0.002 r (s/r)^4 (k = 2: the arc's error is of order 4)
fn grid_step(r: f64, k: usize, tau_g: f64) -> f64 {
  if k <= 1 {
    r * (8.0 * tau_g / r).sqrt()
  } else {
    r * (tau_g / (0.002 * r)).powf(0.25)
  }
}

/// The real gradient B g and Hessian B H B^T of f at xref.
fn real_grad_hess(f: &dyn SignedDistance, xref: &[f64], m: &Metric) -> (Vec<f64>, Vec<f64>) {
  let n = xref.len();
  let (_, gr) = f.grad(xref);
  let hr = f.hess(xref);
  let b = |i: usize, j: usize| m.b[i * n + j];
  let g = (0..n)
    .map(|i| (0..n).map(|j| b(i, j) * gr[j]).sum())
    .collect();
  let mut bh = vec![0.0; n * n];
  for i in 0..n {
    for j in 0..n {
      bh[i * n + j] = (0..n).map(|k| b(i, k) * hr[k * n + j]).sum();
    }
  }
  let mut h = vec![0.0; n * n];
  for i in 0..n {
    for j in 0..n {
      h[i * n + j] = (0..n).map(|k| bh[i * n + k] * b(j, k)).sum();
    }
  }
  (g, h)
}

/// Curvature radius of the curve where f1 and f2 vanish (3D), at xref on
/// both, in reference lengths as radius_estimate: its curvature vector k
/// is in the plane of the unit normals u_i, k.u_i = -t.H_i t / |g_i| (t
/// the unit tangent). The secondary level set of a crack matters there
/// only, on its front (not GetFEM's: the radius of its whole surface).
fn curve_radius_estimate(
  f1: &dyn SignedDistance,
  f2: &dyn SignedDistance,
  xref: &[f64],
  m: &Metric,
) -> f64 {
  let (g1, h1) = real_grad_hess(f1, xref, m);
  let (g2, h2) = real_grad_hess(f2, xref, m);
  let (n1, n2) = (vect_dist2(&g1, &[0.0; 3]), vect_dist2(&g2, &[0.0; 3]));
  if n1 <= 0.0 || n2 <= 0.0 {
    return 1e10;
  }
  let u1: Vec<f64> = g1.iter().map(|v| v / n1).collect();
  let u2: Vec<f64> = g2.iter().map(|v| v / n2).collect();
  let mut t = vec![
    u1[1] * u2[2] - u1[2] * u2[1],
    u1[2] * u2[0] - u1[0] * u2[2],
    u1[0] * u2[1] - u1[1] * u2[0],
  ];
  let st = vect_dist2(&t, &[0.0; 3]);
  // tangent surfaces: no curve direction, the smaller surface radius
  if st < 1e-6 {
    return radius_estimate(f1, xref, false, Some(m)).min(radius_estimate(
      f2,
      xref,
      false,
      Some(m),
    ));
  }
  t.iter_mut().for_each(|v| *v /= st);
  let tht = |h: &[f64]| -> f64 {
    (0..3)
      .map(|i| (0..3).map(|j| t[i] * h[i * 3 + j] * t[j]).sum::<f64>())
      .sum()
  };
  let (k1, k2) = (-tht(&h1) / n1, -tht(&h2) / n2);
  let c: f64 = u1.iter().zip(&u2).map(|(a, b)| a * b).sum();
  let kk = ((k1 * k1 - 2.0 * c * k1 * k2 + k2 * k2) / (st * st)).sqrt();
  1.0 / kk.max(1e-10) / m.smax / 2.0
}

impl<'a> PointStock<'a> {
  fn new(
    list: &'a [&'a dyn SignedDistance],
    nbelt: usize,
    radius_cv: f64,
    metric: Option<Metric>,
    secondary: u64,
  ) -> Self {
    PointStock {
      points: vec![],
      constraints: vec![],
      radius: vec![],
      list,
      radius_cv,
      nbelt,
      max_norm: 0.0,
      metric,
      secondary,
      cells: Default::default(),
    }
  }

  fn clear(&mut self) {
    self.points.clear();
    self.constraints.clear();
    self.radius.clear();
    self.max_norm = 0.0;
    self.cells.clear();
  }

  /// bgeot::node_tab(1e7): eps = max |x| 1e7 machine epsilon
  fn search(&self, xref: &[f64]) -> Option<usize> {
    let eps = self.max_norm * 1e7 * f64::EPSILON;
    debug_assert!(eps < HASH);
    let c = cell(xref);
    let n = xref.len().min(3);
    // the 3^n cells around
    (0..3usize.pow(n as u32))
      .filter_map(|k| {
        let mut d = c;
        for (i, di) in d.iter_mut().enumerate().take(n) {
          *di += (k / 3usize.pow(i as u32) % 3) as i64 - 1;
        }
        self.cells.get(&d)
      })
      .flatten()
      .copied()
      .filter(|&j| vect_dist2(xref, &self.points[j]) < eps)
      .min()
  }

  fn met(&self, xref: &[f64]) -> u64 {
    (0..self.list.len())
      .filter(|&i| self.list[i].val(xref).abs() < 1e-8 * self.radius_cv)
      .fold(0, |b, i| b | (1 << i))
  }

  fn push(&mut self, xref: &[f64], c: u64, r: f64) -> usize {
    self.max_norm = self
      .max_norm
      .max(xref.iter().map(|v| v * v).sum::<f64>().sqrt());
    self.points.push(xref.to_vec());
    self.constraints.push(c);
    self.radius.push(r);
    let j = self.points.len() - 1;
    self.cells.entry(cell(xref)).or_default().push(j);
    j
  }

  /// with a given curvature radius
  fn add_r(&mut self, xref: &[f64], r: f64) -> usize {
    match self.search(xref) {
      Some(j) => j,
      None => {
        let c = self.met(xref);
        self.push(xref, c, r)
      }
    }
  }

  /// with the smallest curvature radius of the constraints met
  fn add(&mut self, xref: &[f64]) -> usize {
    match self.search(xref) {
      Some(j) => j,
      None => {
        let c = self.met(xref);
        let r = (0..self.list.len())
          .filter(|i| c & (1 << i) != 0)
          .map(|i| match self.metric.as_ref() {
            // a secondary: the front's radius, on its primary too
            Some(m) if self.secondary >> i & 1 == 1 => {
              if c >> (i - 1) & 1 == 1 {
                curve_radius_estimate(self.list[i - 1], self.list[i], xref, m)
              } else {
                1e10
              }
            }
            _ => radius_estimate(self.list[i], xref, true, self.metric.as_ref()),
          })
          .fold(1e10, f64::min);
        self.push(xref, c, r)
      }
    }
  }

  /// Points of the most constraints first, a point dropped near a
  /// retained one meeting at least its constraints. The
  /// spacing (not GetFEM's): on a level set the step of its curvature
  /// radius (step, at most cap), elsewhere that of the nearest such points
  /// grown by GRADING times the distance to them (graded cut, fine only
  /// where a curved level set needs it)
  fn decimate(
    &self,
    ref_element: &RefElement,
    cap: f64,
    step: &dyn Fn(f64) -> f64,
  ) -> BTreeSet<usize> {
    let mut kept: BTreeSet<usize> = BTreeSet::new();
    let Some(p0ref) = self.points.first() else {
      return kept;
    };
    let n = p0ref.len();
    // the size field: at a point on a level set the step its radius needs,
    // growing by GRADING times the distance from it, at most cap
    let on_ls: Vec<(usize, f64)> = (0..self.points.len())
      .filter(|&i| self.constraints[i] >> self.nbelt != 0)
      .map(|i| (i, step(self.radius[i]).min(cap)))
      .collect();
    let size = |x: &[f64]| {
      on_ls
        .iter()
        .map(|&(p, s)| s + GRADING * vect_dist2(x, &self.points[p]))
        .fold(cap, f64::min)
    };
    for nb_co in (0..=n).rev() {
      for i in 0..self.points.len() {
        let card = self.constraints[i].count_ones() as usize;
        if kept.contains(&i)
          || !(card == nb_co || (nb_co == n && card > n))
          || ref_element.val(&self.points[i]) >= 1e-8
        {
          continue;
        }
        let ci = self.constraints[i];
        let d = if ci >> self.nbelt != 0 {
          step(self.radius[i]).min(cap)
        } else if nb_co == 0 {
          1.5 * size(&self.points[i])
        } else {
          size(&self.points[i])
        };
        let close = kept.iter().any(|&j| {
          self.constraints[j] & ci == ci && vect_dist2(&self.points[i], &self.points[j]) < d
        });
        if !close {
          kept.insert(i);
        }
      }
    }
    kept
  }
}

/// The level set on the convex as a constraint (0: primary, 1: secondary)
fn of_convex(ls: &LevelSet, cv: usize, i: usize) -> Result<ConvexLevelSet> {
  ls.of_convex(cv, i)
}

impl MeshLevelSet {
  pub fn new(mesh: &Mesh) -> Self {
    MeshLevelSet {
      mesh: mesh.clone(),
      level_sets: vec![],
      primary_zones: BTreeMap::new(),
      cut: BTreeMap::new(),
      crack_tips: BTreeSet::new(),
      crack_tips_of: vec![],
      adapted: false,
      versions: vec![],
    }
  }

  pub fn linked_mesh(&self) -> &Mesh {
    &self.mesh
  }
  pub fn nb_level_sets(&self) -> usize {
    self.level_sets.len()
  }
  pub fn level_set(&self, i: usize) -> &LevelSet {
    &self.level_sets[i]
  }

  /// Adds a level set (once; the handle is shared, not copied).
  pub fn add_level_set(&mut self, ls: &LevelSet) {
    if !self.level_sets.iter().any(|l| l.same(ls)) {
      self.level_sets.push(ls.clone());
      self.adapted = false;
    }
  }

  pub fn sup_level_set(&mut self, ls: &LevelSet) {
    self.level_sets.retain(|l| !l.same(ls));
    self.adapted = false;
  }

  fn check_adapted(&self) -> Result<()> {
    if !self.adapted {
      return Err(Error::InvalidArgument(
        "mesh level set: call adapt() after setting the level sets".into(),
      ));
    }
    Ok(())
  }

  pub fn is_convex_cut(&self, cv: usize) -> bool {
    self.cut.contains_key(&cv)
  }

  /// The cut of a cut convex: its sub-mesh, zones, the level sets of its
  /// vertices.
  pub fn cut_convex(&self, cv: usize) -> Result<&CutConvex> {
    self.check_adapted()?;
    self
      .cut
      .get(&cv)
      .ok_or_else(|| Error::InvalidArgument(format!("convex {cv} is not cut")))
  }

  /// The sub-mesh of a cut convex, in its reference element.
  pub fn mesh_of_convex(&self, cv: usize) -> Result<&Arc<MeshData>> {
    Ok(&self.cut_convex(cv)?.mesh)
  }

  pub fn zoneset_of_convex(&self, cv: usize) -> Result<&ZoneSet> {
    Ok(&self.cut_convex(cv)?.zones)
  }

  /// '+', '-', '0' per level set if not crossed, '*' if crossed
  pub fn primary_zone_of_convex(&self, cv: usize) -> Result<&str> {
    self.check_adapted()?;
    self
      .primary_zones
      .get(&cv)
      .map(|s| s.as_str())
      .ok_or_else(|| Error::InvalidArgument(format!("no convex {cv}")))
  }

  /// The cut convexes containing a crack tip (both functions of a level
  /// set with a secondary part vanish at a sub-mesh point).
  pub fn crack_tip_convexes(&self) -> Result<&BTreeSet<usize>> {
    self.check_adapted()?;
    Ok(&self.crack_tips)
  }

  /// The elements the front of the level set i passes through: a point of
  /// their cut on both its zero sets, inside them, less those it only
  /// grazes (CutConvex::graze); a front on the boundary of elements (a
  /// face, an edge) counts in them unless it passes through the inside of
  /// another element there.
  pub fn crack_tip_convexes_of(&self, i: usize) -> Result<&BTreeSet<usize>> {
    self.check_adapted()?;
    self.crack_tips_of.get(i).ok_or_else(|| {
      Error::InvalidArgument(format!(
        "no level set {i} ({} level sets)",
        self.level_sets.len()
      ))
    })
  }

  /// Cuts the convexes with respect to the level sets.
  pub fn adapt(&mut self) -> Result<()> {
    self.primary_zones.clear();
    self.cut.clear();
    let m = self.mesh.snapshot();
    for cv in m.convex_index() {
      let radius = m
        .trans_of_convex(cv)
        .element_size(&m.points_of_convex(cv))?
        / 2.0;
      let (prim, sec, z) = self.find_crossing_level_set(cv, radius)?;
      if !prim.is_empty() {
        // a secondary cuts the element only where its front passes through
        // it (GetFEM: wherever the secondary crosses the element)
        let mut c = self.cut_element(cv, &prim, &BTreeSet::new(), radius)?;
        let sec = self.fronts_in(cv, &c, &sec)?;
        if !sec.is_empty() {
          c = self.cut_element(cv, &prim, &sec, radius)?;
        }
        for &l in &sec {
          if self.front_inside(cv, &c, l)? {
            c.graze[l] = self.graze_of(cv, &c, l)?;
          }
        }
        c.zones = self.zones_of_element(cv, &c.mesh, &z, &c.graze, radius)?;
        self.cut.insert(cv, c);
      }
      self.primary_zones.insert(cv, z);
    }
    self.update_crack_tip_convexes()?;
    self.adapted = true;
    self.versions = self.level_sets.iter().map(|l| l.version()).collect();
    Ok(())
  }

  /// Adapted, and the level sets unchanged since.
  pub fn is_up_to_date(&self) -> bool {
    self.adapted
      && self.versions
        == self
          .level_sets
          .iter()
          .map(|l| l.version())
          .collect::<Vec<_>>()
  }

  /// Of the level sets in sec, those whose front (primary and secondary
  /// zero) passes through convex cv: the secondary takes both signs on
  /// the points of the primary cut c lying on the primary.
  fn fronts_in(&self, cv: usize, c: &CutConvex, sec: &BTreeSet<usize>) -> Result<BTreeSet<usize>> {
    let mut out = BTreeSet::new();
    for &l in sec {
      let ls = &self.level_sets[l];
      let (m0, m1) = (of_convex(ls, cv, 0)?, of_convex(ls, cv, 1)?);
      let (mut neg, mut pos, mut on) = (false, false, false);
      for p in c.mesh.point_index() {
        let xref = c.mesh.point(p);
        if SignedDistance::val(&m0, xref).abs() < 1e-8 {
          on = true;
          let v = SignedDistance::val(&m1, xref);
          // (a front on the element's boundary counts, as the tips)
          neg |= v <= 1e-10;
          pos |= v >= -1e-10;
        }
      }
      // (no point on the primary: kept)
      if !on || (neg && pos) {
        out.insert(l);
      }
    }
    Ok(out)
  }

  /// The front of the level set l passes through the inside of the cut
  /// element cv (a point of its cut on both zero sets, not on the
  /// element's boundary).
  fn front_inside(&self, cv: usize, c: &CutConvex, l: usize) -> Result<bool> {
    let ls = &self.level_sets[l];
    let (m0, m1) = (of_convex(ls, cv, 0)?, of_convex(ls, cv, 1)?);
    let m = self.mesh.snapshot();
    let cvr = m.trans_of_convex(cv).convex_ref().clone();
    for p in c.mesh.point_index() {
      let x = c.mesh.point(p);
      if SignedDistance::val(&m0, x).abs() < 1e-10
        && SignedDistance::val(&m1, x).abs() < 1e-10
        && cvr.is_in(x)? < -1e-8
      {
        return Ok(true);
      }
    }
    Ok(false)
  }

  /// Whether the front of the level set l grazes the cut element cv
  /// (CutConvex::graze): the faces of the sub-mesh on its primary, behind
  /// (secondary < 0 at their centroid) and ahead of the front, their flat
  /// measure in reference coordinates.
  fn graze_of(&self, cv: usize, c: &CutConvex, l: usize) -> Result<i8> {
    let ls = &self.level_sets[l];
    let (m0, m1) = (of_convex(ls, cv, 0)?, of_convex(ls, cv, 1)?);
    let sm = &c.mesh;
    let n = sm.dim();
    let (mut behind, mut ahead) = (0.0, 0.0);
    for i in sm.convex_index() {
      for f in 0..=n {
        let pts: Vec<&[f64]> = sm
          .ind_points_of_face_of_convex(i, f)
          .iter()
          .map(|&p| sm.point(p))
          .collect();
        if !pts.iter().all(|x| SignedDistance::val(&m0, x).abs() < 1e-8) {
          continue;
        }
        // the face's corners: the farthest nodes (the others lie between)
        let d = |a: &[f64], b: &[f64]| {
          a.iter()
            .zip(b)
            .map(|(p, q)| (p - q).powi(2))
            .sum::<f64>()
            .sqrt()
        };
        let a = pts[0];
        let b = *pts
          .iter()
          .max_by(|x, y| d(a, x).total_cmp(&d(a, y)))
          .unwrap();
        let ab: Vec<f64> = b.iter().zip(a).map(|(p, q)| p - q).collect();
        let measure_c = |x: &[f64]| -> f64 {
          // distance of x to the line ab (2D: 0)
          let ax: Vec<f64> = x.iter().zip(a).map(|(p, q)| p - q).collect();
          let l2: f64 = ab.iter().map(|v| v * v).sum();
          let t = ax.iter().zip(&ab).map(|(p, q)| p * q).sum::<f64>() / l2.max(1e-300);
          ax.iter()
            .zip(&ab)
            .map(|(p, q)| (p - t * q).powi(2))
            .sum::<f64>()
            .sqrt()
        };
        let (measure, centroid) = if n == 2 {
          (
            d(a, b),
            a.iter()
              .zip(b)
              .map(|(p, q)| 0.5 * (p + q))
              .collect::<Vec<f64>>(),
          )
        } else {
          let c3 = *pts
            .iter()
            .max_by(|x, y| measure_c(x).total_cmp(&measure_c(y)))
            .unwrap();
          (
            0.5 * d(a, b) * measure_c(c3),
            (0..n).map(|k| (a[k] + b[k] + c3[k]) / 3.0).collect(),
          )
        };
        if SignedDistance::val(&m1, &centroid) < 0.0 {
          behind += measure;
        } else {
          ahead += measure;
        }
      }
    }
    let total = behind + ahead;
    Ok(if total <= 0.0 {
      0
    } else if behind >= (1.0 - FRONT_GRAZE) * total {
      1
    } else if behind <= FRONT_GRAZE * total {
      -1
    } else {
      0
    })
  }

  /// The level sets crossing convex cv (primary; with their secondary
  /// part), and the primary zone.
  fn find_crossing_level_set(
    &self,
    cv: usize,
    radius: f64,
  ) -> Result<(BTreeSet<usize>, BTreeSet<usize>, String)> {
    let (mut prim, mut sec) = (BTreeSet::new(), BTreeSet::new());
    let mut z: Vec<char> = vec!['*'; self.level_sets.len()];
    for (k, ls) in self.level_sets.iter().enumerate() {
      let s = self.is_not_crossed_by(cv, ls, 0, radius)?;
      if s == 0 {
        if ls.has_secondary() {
          let s1 = self.is_not_crossed_by(cv, ls, 1, radius)?;
          if s1 == 0 {
            sec.insert(k);
            prim.insert(k);
          } else if s1 < 0 {
            prim.insert(k);
          } else {
            z[k] = '0';
          }
        } else {
          prim.insert(k);
        }
      } else {
        z[k] = if s < 0 { '-' } else { '+' };
      }
    }
    Ok((prim, sec, z.into_iter().collect()))
  }

  /// 0 if function i of ls crosses convex cv, else its sign
  fn is_not_crossed_by(&self, cv: usize, ls: &LevelSet, i: usize, radius: f64) -> Result<i32> {
    let mf = ls.mesh_fem();
    let v = ls.values(i)?;
    let eps = 1e-8 * radius;
    let mut p = -2;
    for d in mf.ind_basic_dof_of_element(cv)? {
      let x = v[d];
      let p2 = if x < -eps {
        -1
      } else if x > eps {
        1
      } else {
        0
      };
      if p == -2 {
        p = p2;
      }
      if p2 == 0 || p * p2 < 0 {
        return Ok(0);
      }
    }
    let mls = of_convex(ls, cv, i)?;
    let n = mls.dim();
    let xref = start_point(n, 1e-2);
    let (d, g) = SignedDistance::grad(&mls, &xref);
    if g.iter().map(|v| v * v).sum::<f64>().sqrt() * 2.5 < d.abs() {
      return Ok(p);
    }
    let gt = self.mesh.snapshot().trans_of_convex(cv).clone();
    let re = RefElement::of(&gt.convex_ref().basic().structure().clone())?;
    // a point of the convex on each side
    for s in [false, true] {
      let neg = Negated(&mls);
      let f: &dyn SignedDistance = if s { &neg } else { &mls };
      let mi = Intersection(&re, f);
      let mut xref = start_point(n, 1e-2);
      if !try_projection(&mi, &mut xref, false) || re.val(&xref) > 1e-8 {
        return Ok(p);
      }
    }
    Ok(0)
  }

  /// The sub-mesh of convex cv.
  fn cut_element(
    &self,
    cv: usize,
    primary: &BTreeSet<usize>,
    secondary: &BTreeSet<usize>,
    radius_cv: f64,
  ) -> Result<CutConvex> {
    let m = self.mesh.snapshot();
    let gt = m.trans_of_convex(cv).clone();
    let basic = gt.convex_ref().basic();
    let re = RefElement::of(&basic.structure().clone())?;
    let n = basic.dim();
    if primary.len() + secondary.len() > 16 {
      return Err(Error::InvalidArgument(
        "an element is cut by more than 16 level sets".into(),
      ));
    }
    // step 1: the constraints (the faces, then the level sets), the
    // smallest curvature radius, the degree K
    let mut lss: Vec<ConvexLevelSet> = vec![];
    // per constraint of lss: its level set, 0 primary or 1 secondary
    let mut owner = vec![];
    let mut k_deg = 0;
    for &l in primary {
      let ls = &self.level_sets[l];
      k_deg = k_deg.max(ls.degree());
      lss.push(of_convex(ls, cv, 0)?);
      owner.push((l, 0));
      if secondary.contains(&l) {
        lss.push(of_convex(ls, cv, 1)?);
        owner.push((l, 1));
      }
    }
    let mut list: Vec<&dyn SignedDistance> = re
      .faces()
      .iter()
      .map(|f| f as &dyn SignedDistance)
      .collect();
    let nbelt = list.len();
    list.extend(lss.iter().map(|l| l as &dyn SignedDistance));
    // the smallest curvature radius at the vertices and the centroid
    // projected on each level set (GetFEM: at one random point)
    let vertsref = basic.points().to_vec();
    let centroidref: Vec<f64> = (0..n)
      .map(|k| vertsref.iter().map(|v| v[k]).sum::<f64>() / vertsref.len() as f64)
      .collect();
    let metric = {
      let g = m.points_of_convex(cv);
      let pg = gt.at(&g, &centroidref)?;
      // 3D (measured there; 2D keeps GetFEM's estimate)
      if n == 3 && pg.b.nrows() == n && pg.b.ncols() == n {
        let k = getfem_linalg::tensor::singular_value_bounds(&pg.k);
        Some(Metric {
          b: (0..n * n).map(|ij| pg.b.col(ij % n)[ij / n]).collect(),
          smax: k.1,
        })
      } else {
        None
      }
    };
    // the radius of the zero set of constraint k of list (a face or a level
    // set) at qref (on it):
    // of a secondary level set (3D), of the front curve near qref, if any
    let radius_at = |k: usize, qref: &[f64], near: f64| -> f64 {
      match metric.as_ref() {
        Some(m) if k >= nbelt && owner[k - nbelt].1 == 1 => {
          let mut q = qref.to_vec();
          if multi_projection(&[list[k - 1], list[k]], &mut q)
            && re.val(&q) < 1e-9
            && vect_dist2(&q, qref) < near
          {
            curve_radius_estimate(list[k - 1], list[k], &q, m)
          } else {
            1e10
          }
        }
        _ => radius_estimate(list[k], qref, false, metric.as_ref()),
      }
    };
    let mut r0 = 1e10f64;
    for (il, l) in lss.iter().enumerate() {
      for xref in vertsref.iter().chain([&centroidref]) {
        let mut qref = xref.clone();
        if try_projection(l, &mut qref, true) {
          r0 = r0.min(radius_at(nbelt + il, &qref, f64::INFINITY));
        }
      }
    }
    if r0.abs() < 1e-13 {
      return Err(Error::InvalidArgument(format!(
        "level set on convex {cv}: curvature radius {r0}"
      )));
    }
    // exact integrals of 1 and x over the reference convex (its
    // simplexification)
    let start: Vec<Vec<usize>> = simplexified(basic.structure()).ok_or(Error::NotImplemented(
      "level set cut of a convex without simplexification",
    ))?;
    let exact = moments(&vertsref, &start);
    // the grid step, 3D: at most 1/2 (GetFEM 1/(K+1)), the sub-elements'
    // geometric error on curved level sets at most GEOMETRY_TOLERANCE; 2D:
    // GetFEM's, a fifth of the curvature radius
    let step = |r: f64| {
      if metric.is_some() {
        grid_step(r, k_deg, GEOMETRY_TOLERANCE)
      } else {
        0.2 * r
      }
    };
    let mut cap = if metric.is_some() {
      0.5
    } else {
      1.0 / (k_deg + 1) as f64
    };
    let mut h0 = step(r0).min(cap);
    // a sub-mesh failed: the retries also spaced at most h0 (step alone
    // need not refine on the level sets)
    let mut retry = false;
    let secondary: u64 = if metric.is_some() {
      (0..lss.len())
        .filter(|&j| owner[j].1 == 1)
        .fold(0, |b, j| b | 1 << (nbelt + j))
    } else {
      0
    };
    // distance from xref to the zero set of list[k], first order
    let distance = |k: usize, xref: &[f64]| {
      let (v, g) = list[k].grad(xref);
      v.abs() / g.iter().map(|a| a * a).sum::<f64>().sqrt().max(1e-300)
    };
    let mut stock = PointStock::new(&list, nbelt, radius_cv, metric.clone(), secondary);
    for _ in 0..30 {
      // step 2: grid points projected on the constraints
      stock.clear();
      let geps = 0.001 * h0;
      let gridn = 1 + (1.0 / h0) as usize;
      let nbpt = gridn.pow(n as u32);
      let mut h0_ok = true;
      'grid: for i in 0..nbpt {
        let ptref: Vec<f64> = (0..n)
          .map(|k| ((i / gridn.pow(k as u32)) % gridn) as f64 / (gridn - 1) as f64)
          .collect();
        if re.val(&ptref) >= geps {
          continue;
        }
        let mut co: Vec<usize> = vec![];
        let mut kept = false;
        // the level sets before the faces (reversed)
        for k in (0..list.len()).rev() {
          let mut qref = ptref.clone();
          let (d, v) = list[k].grad(&ptref);
          let nv = v.iter().map(|a| a * a).sum::<f64>().sqrt();
          if nv * h0 * 7.0 > d.abs()
            && try_projection(list[k], &mut qref, true)
            && (secondary >> k & 1 == 0 || distance(k - 1, &qref) < h0 * 3.5)
          {
            if k >= nbelt && vect_dist2(&ptref, &qref) < h0 * 3.5 {
              kept = true;
            }
            if vect_dist2(&ptref, &qref) < h0 / 1.5 {
              co.push(k);
              if re.val(&qref) < 1e-8 {
                let r = radius_at(k, &qref, 3.5 * h0);
                r0 = r0.min(r);
                if h0 > 1.25 * step(r0) {
                  h0 = step(r0);
                  h0_ok = false;
                  break;
                }
                if k >= nbelt || kept {
                  stock.add_r(&qref, r);
                }
              }
            }
          }
        }
        if kept {
          stock.add_r(&ptref, 1e10);
        }
        if !h0_ok {
          break 'grid;
        }
        // the intersections of 2 to n of the constraints met
        for count in 0..(1usize << co.len()) {
          let nn: Vec<usize> = (0..co.len())
            .filter(|j| count & (1 << j) != 0)
            .map(|j| co[j])
            .collect();
          if nn.len() > 1 && nn.len() <= n {
            let mut qref = ptref.clone();
            let sub: Vec<&dyn SignedDistance> = nn.iter().map(|&k| list[k]).collect();
            if multi_projection(&sub, &mut qref) && re.val(&qref) < 1e-9 {
              stock.add(&qref);
            }
          }
        }
      }
      if !h0_ok {
        continue;
      }
      // step 3: Delaunay, refined where a sub-simplex is crossed
      let mut retained = stock.decimate(&re, if retry { cap.min(h0) } else { cap }, &step);
      // the vertices of the convex (GetFEM finds them as intersections of
      // faces; qhull triangulates the hull of the points)
      let mut vert_ids = vec![];
      for vref in &vertsref {
        // exact, also when merged with a projected point
        let j = stock.add(vref);
        stock.points[j] = vref.clone();
        retained.insert(j);
        vert_ids.push(j);
      }
      // exactly on the faces they are on (within 1e-12, or beyond): the
      // triangulation sees them on the hull, not just outside
      let snapped = |ptref: &[f64]| {
        let mut ptref = ptref.to_vec();
        for f in re.faces() {
          f.snap(&mut ptref, 1e-12);
        }
        ptref
      };
      let order: Vec<usize> = retained.iter().copied().collect();
      let mut cts: Vec<u64> = order.iter().map(|&i| stock.constraints[i]).collect();
      let local = |g: usize| order.iter().position(|&o| o == g).unwrap();
      let st: Vec<Vec<usize>> = start
        .iter()
        .map(|s| s.iter().map(|&a| local(vert_ids[a])).collect())
        .collect();
      let mut tri = Delaunay::new(
        order.iter().map(|&i| snapped(&stock.points[i])).collect(),
        &st,
      )?;
      for q in 0..order.len() {
        tri.insert(q);
      }
      // the crossing points of crossed sub-simplices inserted (GetFEM:
      // added, then Delaunay again), at most 15 rounds
      let mut nb_delaunay = 0;
      let simplices = loop {
        nb_delaunay += 1;
        if nb_delaunay > 15 {
          h0 /= 2.0;
          cap /= 2.0;
          retry = true;
          h0_ok = false;
          break vec![];
        }
        let ptsref = tri.points();
        let eps = radius_cv * 1e-7;
        // not those with all vertices on one level set: flat on a plane, a
        // lens on a curved surface, on both sides of it (not GetFEM's)
        // (a secondary, not curved onto, only with its primary)
        let on_one = |s: &[usize]| {
          let all_on = |k: usize| s.iter().all(|&a| list[k].val(&ptsref[a]).abs() <= eps);
          (nbelt..list.len()).any(|k| all_on(k) && (secondary >> k & 1 == 0 || all_on(k - 1)))
        };
        let simplices: Vec<Vec<usize>> = tri
          .simplices()
          .into_iter()
          .filter(|s| quality(ptsref, s) >= 1e-10 && !on_one(s))
          .collect();
        let mut added = vec![];
        for s in &simplices {
          // the level sets crossing or touching the simplex (a vertex on it)
          let crossed: Vec<usize> = (nbelt..list.len())
            .filter(|&k| {
              let v: Vec<f64> = s.iter().map(|&a| list[k].val(&ptsref[a])).collect();
              v.iter().any(|x| x.abs() <= eps)
                || (v.iter().any(|&x| x > eps) && v.iter().any(|&x| x < -eps))
            })
            .collect();
          let mut signs = vec![0.0f64; list.len()];
          let mut prev = vec![0usize; list.len()];
          for ii in 0..=n {
            let xiref = &ptsref[s[ii]];
            for jj in 0..list.len() {
              if secondary >> jj & 1 == 1 && !crossed.contains(&(jj - 1)) {
                continue;
              }
              let dd = list[jj].val(xiref);
              if dd.abs() > radius_cv * 1e-7 {
                if dd * signs[jj] < 0.0 {
                  let xcref = crossing(list[jj], xiref, &ptsref[s[prev[jj]]], dd, radius_cv);
                  let kk = stock.add(&xcref);
                  if retained.insert(kk) {
                    added.push(kk);
                    // crossed or touched by another level set too: also the
                    // point on both (else crossing points of each, closer
                    // and closer to their intersection, can follow each
                    // other forever)
                    if jj >= nbelt {
                      for &k in crossed.iter().filter(|&&k| k != jj) {
                        let mut qref = xcref.clone();
                        if multi_projection(&[list[jj], list[k]], &mut qref)
                          && re.val(&qref) < 1e-9
                          && vect_dist2(&qref, &xcref) < h0
                        {
                          let kq = stock.add(&qref);
                          if retained.insert(kq) {
                            added.push(kq);
                          }
                          break;
                        }
                      }
                    }
                    break;
                  }
                }
                if signs[jj] == 0.0 {
                  signs[jj] = dd;
                  prev[jj] = ii;
                }
              }
            }
          }
        }
        if added.is_empty() {
          break simplices;
        }
        for kk in added {
          let q = tri.add_point(snapped(&stock.points[kk]));
          cts.push(stock.constraints[kk]);
          tri.insert(q);
        }
      };
      let ptsref = tri.points().to_vec();
      if !h0_ok {
        continue;
      }
      // the sub-mesh, of degree K
      let mut sm = MeshData::new(n);
      // at their indices (the constraints cts follow them)
      for (i, ptref) in ptsref.iter().enumerate() {
        sm.add_point_at(i, ptref)?;
      }
      let gt1 = geotrans_descriptor(&format!("GT_PK({n},1)"))?;
      let gtk = geotrans_descriptor(&format!("GT_PK({n},{k_deg})"))?;
      for s in &simplices {
        if k_deg > 1 {
          let g1ref = getfem_linalg::tensor::Tensor::from_rows(
            &(0..n)
              .map(|r| s.iter().map(|&a| ptsref[a][r]).collect::<Vec<f64>>())
              .collect::<Vec<_>>()
              .iter()
              .map(|v| v.as_slice())
              .collect::<Vec<_>>(),
          );
          let nodesref: Vec<Vec<f64>> = gtk
            .convex_ref()
            .points()
            .iter()
            .map(|xref| gt1.transform(xref, &g1ref))
            .collect::<std::result::Result<_, _>>()?;
          let r: Vec<&[f64]> = nodesref.iter().map(|v| v.as_slice()).collect();
          sm.add_convex_by_points(&gtk, &r)?;
        } else {
          sm.add_convex(&gt1, s)?;
        }
      }
      // the level sets each vertex is on
      let on: Vec<[u32; 2]> = cts
        .iter()
        .map(|&c| {
          let mut m = [0u32; 2];
          for (j, &(l, k)) in owner.iter().enumerate() {
            if c >> (nbelt + j) & 1 == 1 {
              m[k] |= 1 << l;
            }
          }
          m
        })
        .collect();
      if k_deg > 1 {
        interpolate_faces(&mut sm, ptsref.len(), &cts, &list, &re, secondary)?;
      }
      // step 5: the monomials of degree 1 integrated on the sub-mesh
      let got = sub_moments(&sm, 2 * k_deg)?;
      let error = got
        .iter()
        .zip(&exact)
        .fold(0.0f64, |e, (a, b)| e.max((a - b).abs()));
      if error > 1e-5 {
        h0 /= 2.0;
        cap /= 2.0;
        retry = true;
        continue;
      }
      return Ok(CutConvex {
        mesh: Arc::new(sm),
        on,
        zones: ZoneSet::new(),
        graze: vec![0; self.level_sets.len()],
      });
    }
    Err(Error::InvalidArgument(format!(
      "level set cut of convex {cv}: no valid sub-mesh"
    )))
  }

  /// The zones of the sub-elements of a cut convex.
  fn zones_of_element(
    &self,
    cv: usize,
    sm: &MeshData,
    prezone: &str,
    graze: &[i8],
    radius: f64,
  ) -> Result<ZoneSet> {
    let mut zones = ZoneSet::new();
    // the functions on cv, once
    let pz: Vec<char> = prezone.chars().collect();
    let mut fs = vec![];
    for (j, ls) in self.level_sets.iter().enumerate() {
      fs.push(if pz[j] == '*' || pz[j] == '0' {
        // a grazing front: the crack splits the element (1), or not (-1)
        let m1 = if ls.has_secondary() && graze[j] == 0 {
          Some(of_convex(ls, cv, 1)?)
        } else {
          None
        };
        Some((of_convex(ls, cv, 0)?, m1))
      } else {
        None
      });
    }
    let mut cache: Option<(PGeoTrans, PIntegration, RuleGeometry)> = None;
    for i in sm.convex_index() {
      if area_estimate(sm, i, &mut cache)? <= 1e-8 {
        continue;
      }
      let mut subz = pz.clone();
      for (j, f) in fs.iter().enumerate() {
        if let Some((m0, m1)) = f {
          let s = sub_simplex_is_not_crossed_by(m0, m1.as_ref(), sm, i, radius);
          subz[j] = if graze[j] < 0 {
            '0'
          } else if s < 0 {
            '-'
          } else if s > 0 {
            '+'
          } else {
            '0'
          };
        }
      }
      merge_zoneset(&mut zones, &subz.into_iter().collect::<String>());
    }
    Ok(zones)
  }

  fn update_crack_tip_convexes(&mut self) -> Result<()> {
    self.crack_tips.clear();
    self.crack_tips_of = vec![BTreeSet::new(); self.level_sets.len()];
    let m = self.mesh.snapshot();
    for (i, ls) in self.level_sets.iter().enumerate() {
      if !ls.has_secondary() {
        continue;
      }
      // per cut convex: its front points (real), inside it or not
      let mut inside_pts: Vec<Vec<f64>> = vec![];
      let mut touching: Vec<(usize, Vec<Vec<f64>>)> = vec![];
      let mut size = f64::MAX;
      for (&cv, c) in &self.cut {
        let (m0, m1) = (of_convex(ls, cv, 0)?, of_convex(ls, cv, 1)?);
        let (gt, g) = (m.trans_of_convex(cv), m.points_of_convex(cv));
        let cvr = gt.convex_ref();
        let (mut pts, mut inside) = (vec![], false);
        for p in c.mesh.point_index() {
          let xref = c.mesh.point(p);
          if SignedDistance::val(&m0, xref).abs() < 1e-10
            && SignedDistance::val(&m1, xref).abs() < 1e-10
          {
            inside |= cvr.is_in(xref)? < -1e-8;
            pts.push(gt.transform(xref, &g)?);
          }
        }
        if pts.is_empty() {
          continue;
        }
        self.crack_tips.insert(cv);
        size = size.min(gt.element_size(&g)?);
        if c.graze[i] != 0 {
          // the front grazes it: inside, handled by the Heaviside function
          inside_pts.extend(pts);
        } else if inside {
          self.crack_tips_of[i].insert(cv);
          inside_pts.extend(pts);
        } else {
          touching.push((cv, pts));
        }
      }
      // a front on the boundary of elements: in them unless it passes
      // through the inside of a neighbour (an edge of the mesh the crack
      // and its front cross, e.g.); a front along faces: in the elements on
      // both sides
      let tol = 1e-6 * size;
      let covered = |p: &[f64]| {
        inside_pts
          .iter()
          .any(|q| q.iter().zip(p).all(|(a, b)| (a - b).abs() < tol))
      };
      for (cv, pts) in touching {
        if !pts.iter().all(|p| covered(p)) {
          self.crack_tips_of[i].insert(cv);
        }
      }
    }
    Ok(())
  }

  /// The (non-conforming) mesh of the sub-elements of the cut convexes
  /// and the uncut convexes; region 0: the faces on a level set.
  pub fn global_cut_mesh(&self) -> Result<Mesh> {
    self.check_adapted()?;
    let m = self.mesh.snapshot();
    let out = Mesh::new(m.dim());
    let mut faces = vec![];
    out.modify(|o| -> Result<()> {
      for cv in m.convex_index() {
        let (gt, g) = (m.trans_of_convex(cv), m.points_of_convex(cv));
        match self.cut.get(&cv) {
          Some(c) => {
            let mut ids = BTreeMap::new();
            for p in c.mesh.point_index() {
              ids.insert(p, o.add_point(&gt.transform(c.mesh.point(p), &g)?)?);
            }
            let mut new_ids = BTreeMap::new();
            for i in c.mesh.convex_index() {
              let pts: Vec<usize> = c
                .mesh
                .ind_points_of_convex(i)
                .iter()
                .map(|p| ids[p])
                .collect();
              new_ids.insert(i, o.add_convex(c.mesh.trans_of_convex(i), &pts)?);
            }
            for i in c.mesh.convex_index() {
              for f in 0..c.mesh.structure_of_convex(i).nb_faces() {
                if c.level_set_of_face(i, f).is_some() {
                  faces.push((new_ids[&i], f));
                }
              }
            }
          }
          None => {
            let pts: Vec<Vec<f64>> = (0..g.ncols()).map(|j| g.col(j).to_vec()).collect();
            let r: Vec<&[f64]> = pts.iter().map(|v| v.as_slice()).collect();
            o.add_convex_by_points(gt, &r)?;
          }
        }
      }
      Ok(())
    })?;
    let mut r = getfem_geom::region::RegionSet::new();
    for (c, f) in faces {
      r.add_face(c, f);
    }
    out.modify(|o| o.set_region_set(0, r))?;
    Ok(out)
  }

  /// The convexes where two or more level sets cross (to call after
  /// adapt).
  pub fn find_level_set_potential_intersections(&self) -> Result<Vec<(usize, BTreeSet<usize>)>> {
    self.check_adapted()?;
    let m = self.mesh.snapshot();
    let mut out = vec![];
    for &cv in self.cut.keys() {
      let radius = m
        .trans_of_convex(cv)
        .element_size(&m.points_of_convex(cv))?
        / 2.0;
      let (prim, _, _) = self.find_crossing_level_set(cv, radius)?;
      if prim.len() > 1 {
        out.push((cv, prim));
      }
    }
    Ok(out)
  }
}

/// +1 / -1: the side of sub-element i; 0: crossed in the positive part of
/// the secondary level set.
fn sub_simplex_is_not_crossed_by(
  mls0: &ConvexLevelSet,
  mls1: Option<&ConvexLevelSet>,
  sm: &MeshData,
  i: usize,
  radius: f64,
) -> i32 {
  let eps = 1e-7 * radius;
  let (mut p, mut is_cut) = (0, false);
  let (mut d1, mut d2) = (1.0f64, 0.0f64);
  let gref = sm.points_of_convex(i);
  for c in 0..gref.ncols() {
    let xref: Vec<f64> = gref.col(c).to_vec();
    let d0 = SignedDistance::val(mls0, &xref);
    if let Some(m1) = mls1 {
      d1 = d1.min(SignedDistance::val(m1, &xref));
    }
    let p2 = if d0 < -eps {
      -1
    } else if d0 > eps {
      1
    } else {
      0
    };
    if p == 0 {
      p = p2;
    }
    if d0.abs() > d2.abs() {
      d2 = d0;
    }
    if p2 == 0 || p * p2 < 0 {
      is_cut = true;
    }
  }
  if is_cut && mls1.is_some() && d1 >= -radius * 0.0001 {
    return 0;
  }
  if d2 < 0.0 { -1 } else { 1 }
}

/// The crossing point of f between a (f(a) = da) and b: Newton along
/// b - a, bisection if it fails (GetFEM cut_element)
fn crossing(f: &dyn SignedDistance, a: &[f64], b: &[f64], da: f64, radius_cv: f64) -> Vec<f64> {
  let mut xref = a.to_vec();
  let mut vv: Vec<f64> = b.iter().zip(a).map(|(p, q)| p - q).collect();
  if da > 0.0 {
    vv.iter_mut().for_each(|v| *v = -*v);
  }
  let (mut dd, mut g) = f.grad(&xref);
  let mut nbit = 0;
  while dd.abs() > 1e-16 * radius_cv && {
    nbit += 1;
    nbit < 20
  } {
    let mut ng: f64 = g.iter().zip(&vv).map(|(p, q)| p * q).sum();
    if ng.abs() < 1e-8 {
      ng = 1e-8;
    }
    if ng < 0.0 {
      ng = 1.0;
    }
    xref
      .iter_mut()
      .zip(&vv)
      .for_each(|(p, q)| *p -= dd / ng * q);
    (dd, g) = f.grad(&xref);
  }
  if dd.abs() > 1e-16 * radius_cv {
    let (mut x1ref, mut x2ref) = (a.to_vec(), b.to_vec());
    let (mut d1, mut d2) = (f.val(&x1ref), f.val(&x2ref));
    if d1 > d2 {
      std::mem::swap(&mut d1, &mut d2);
      std::mem::swap(&mut x1ref, &mut x2ref);
    }
    while d1.abs() > 1e-15 * radius_cv {
      let x3ref: Vec<f64> = x1ref
        .iter()
        .zip(&x2ref)
        .map(|(p, q)| (p + q) / 2.0)
        .collect();
      let d3 = f.val(&x3ref);
      if d3 == d1 || d3 == d2 {
        break;
      }
      if d3 > 0.0 {
        d2 = d3;
        x2ref = x3ref;
      } else {
        d1 = d3;
        x1ref = x3ref;
      }
    }
    xref = x1ref;
  }
  xref
}

/// GetFEM convex_quality_estimate of a P1 simplex: 1 / condition number
/// of K
fn quality(pts: &[Vec<f64>], s: &[usize]) -> f64 {
  let n = s.len() - 1;
  let rows: Vec<Vec<f64>> = (0..n)
    .map(|r| (1..=n).map(|c| pts[s[c]][r] - pts[s[0]][r]).collect())
    .collect();
  let r: Vec<&[f64]> = rows.iter().map(|v| v.as_slice()).collect();
  let (lo, hi) = singular_value_bounds(&getfem_linalg::tensor::Tensor::from_rows(&r));
  if hi > 0.0 { lo / hi } else { 0.0 }
}

/// integrals of 1, x_1, ..., x_n over affine simplices
fn moments(pts: &[Vec<f64>], simplices: &[Vec<usize>]) -> Vec<f64> {
  let n = pts[0].len();
  let mut m = vec![0.0; n + 1];
  for s in simplices {
    let rows: Vec<Vec<f64>> = (0..n)
      .map(|r| (1..=n).map(|c| pts[s[c]][r] - pts[s[0]][r]).collect())
      .collect();
    let r: Vec<&[f64]> = rows.iter().map(|v| v.as_slice()).collect();
    let det = getfem_linalg::tensor::Lu::new(&getfem_linalg::tensor::Tensor::from_rows(&r))
      .map(|l| l.det())
      .unwrap_or(0.0)
      .abs();
    let vol = det / (1..=n).product::<usize>() as f64;
    m[0] += vol;
    for k in 0..n {
      m[k + 1] += vol * s.iter().map(|&a| pts[a][k]).sum::<f64>() / (n + 1) as f64;
    }
  }
  m
}

/// integrals of 1, x_1, ..., x_n over the (curved) sub-mesh, rules of the
/// degree
fn sub_moments(sm: &MeshData, degree: usize) -> Result<Vec<f64>> {
  let n = sm.dim();
  let mut m = vec![0.0; n + 1];
  let mut cache: Option<(PGeoTrans, PIntegration, RuleGeometry)> = None;
  for i in sm.convex_index() {
    let gt = sm.trans_of_convex(i);
    if cache.as_ref().is_none_or(|c| !Arc::ptr_eq(&c.0, gt)) {
      let pim = classical_approx_im(gt, degree)?;
      let rg = RuleGeometry::new(gt, &pim)?;
      cache = Some((gt.clone(), pim, rg));
    }
    let (_, pim, rg) = cache.as_ref().unwrap();
    let a = pim.approx()?;
    let gref = sm.points_of_convex(i);
    for k in 0..a.nb_points_on_convex() {
      let (xref, j) = rg.at(&gref, k);
      let w = a.coeff(k) * j.abs();
      m[0] += w;
      for d in 0..n {
        m[d + 1] += w * xref[d];
      }
    }
  }
  Ok(m)
}

/// The shape functions of a geometric transformation and their gradients
/// at the points of a rule, computed once: the real point and det K at
/// rule point k of a convex in a few flops.
pub struct RuleGeometry {
  val: Vec<Vec<f64>>,
  grad: Vec<Vec<f64>>,
  dim: usize,
}

impl RuleGeometry {
  pub fn new(gt: &PGeoTrans, pim: &PIntegration) -> Result<Self> {
    let a = pim.approx()?;
    let dim = gt.dim();
    let ptsref = (0..a.nb_points_on_convex()).map(|k| a.point(k).to_vec());
    let (mut val, mut grad) = (vec![], vec![]);
    for xref in ptsref {
      val.push(gt.val(&xref));
      let gr = gt.grad(&xref);
      // row-major nb_points x dim
      grad.push(
        (0..gr.nrows() * dim)
          .map(|ij| gr.col(ij % dim)[ij / dim])
          .collect(),
      );
    }
    Ok(RuleGeometry { val, grad, dim })
  }

  /// The real point and det K at rule point k (g: the convex's nodes as
  /// columns); det 0 unless the dimensions agree.
  pub fn at(&self, g: &getfem_linalg::tensor::Tensor<f64>, k: usize) -> (Vec<f64>, f64) {
    let (n, p) = (g.nrows(), self.dim);
    let mut x = vec![0.0; n];
    let mut kk = [[0.0f64; 3]; 3];
    for (a, (va, ga)) in self.val[k].iter().zip(self.grad[k].chunks(p)).enumerate() {
      let ca = g.col(a);
      for r in 0..n {
        x[r] += va * ca[r];
        if n == p && n <= 3 {
          for c in 0..p {
            kk[r][c] += ca[r] * ga[c];
          }
        }
      }
    }
    let det = match (n == p, n) {
      (true, 1) => kk[0][0],
      (true, 2) => kk[0][0] * kk[1][1] - kk[0][1] * kk[1][0],
      (true, 3) => {
        kk[0][0] * (kk[1][1] * kk[2][2] - kk[1][2] * kk[2][1])
          - kk[0][1] * (kk[1][0] * kk[2][2] - kk[1][2] * kk[2][0])
          + kk[0][2] * (kk[1][0] * kk[2][1] - kk[1][1] * kk[2][0])
      }
      _ => 0.0,
    };
    (x, det)
  }
}

/// det K at the reference point xref (0 for a degenerate element)
pub fn jacobian(gt: &PGeoTrans, g: &getfem_linalg::tensor::Tensor<f64>, xref: &[f64]) -> f64 {
  let k = g.matmul(&gt.grad(xref));
  getfem_linalg::tensor::Lu::new(&k)
    .map(|l| l.det())
    .unwrap_or(0.0)
}

/// The measure by a rule of degree 2
fn area_estimate(
  sm: &MeshData,
  i: usize,
  cache: &mut Option<(PGeoTrans, PIntegration, RuleGeometry)>,
) -> Result<f64> {
  let gt: &PGeoTrans = sm.trans_of_convex(i);
  if cache.as_ref().is_none_or(|c| !Arc::ptr_eq(&c.0, gt)) {
    let pim = classical_approx_im(gt, 2)?;
    let rg = RuleGeometry::new(gt, &pim)?;
    *cache = Some((gt.clone(), pim, rg));
  }
  let (_, pim, rg) = cache.as_ref().unwrap();
  let a = pim.approx()?;
  let gref = sm.points_of_convex(i);
  Ok(
    (0..a.nb_points_on_convex())
      .map(|k| a.coeff(k) * rg.at(&gref, k).1.abs())
      .sum(),
  )
}

/// The nodes of degree K not at a vertex projected onto the constraints
/// met by all vertices of their face, recursively over the faces;
/// secondary: as PointStock's (curved onto a secondary only on the front)
fn interpolate_faces(
  sm: &mut MeshData,
  nb_vertices: usize,
  cts: &[u64],
  list: &[&dyn SignedDistance],
  re: &RefElement,
  secondary: u64,
) -> Result<()> {
  let mut done: BTreeSet<usize> = BTreeSet::new();
  // the moves applied at the end (one rebuild of the point index); each
  // node is projected once, from its position
  let mut moves = vec![];
  for i in sm.convex_index().collect::<Vec<_>>() {
    let cvs = sm.structure_of_convex(i).clone();
    let ipts = sm.ind_points_of_convex(i).to_vec();
    for f in 0..cvs.nb_faces() {
      let fpts: Vec<usize> = cvs.ind_points_of_face(f).iter().map(|&k| ipts[k]).collect();
      let fs = cvs
        .face_structure(f)
        .cloned()
        .ok_or(Error::NotImplemented("face structure"))?;
      interpolate_face(
        sm,
        &mut done,
        &mut moves,
        &fpts,
        &fs,
        nb_vertices,
        cts,
        list,
        re,
        secondary,
      )?;
    }
  }
  sm.set_points(&moves)?;
  Ok(())
}

#[allow(clippy::too_many_arguments)]
fn interpolate_face(
  sm: &MeshData,
  done: &mut BTreeSet<usize>,
  moves: &mut Vec<(usize, Vec<f64>)>,
  ipts: &[usize],
  cvs: &getfem_geom::convex_structure::PCvs,
  nb_vertices: usize,
  cts: &[u64],
  list: &[&dyn SignedDistance],
  re: &RefElement,
  secondary: u64,
) -> Result<()> {
  if cvs.dim() == 0 {
    return Ok(());
  }
  if cvs.dim() > 1 {
    for f in 0..cvs.nb_faces() {
      let fpts: Vec<usize> = cvs.ind_points_of_face(f).iter().map(|&k| ipts[k]).collect();
      let fs = cvs
        .face_structure(f)
        .cloned()
        .ok_or(Error::NotImplemented("face structure"))?;
      interpolate_face(
        sm,
        done,
        moves,
        &fpts,
        &fs,
        nb_vertices,
        cts,
        list,
        re,
        secondary,
      )?;
    }
  }
  let mut c = u64::MAX;
  let mut cnt = 0;
  for &p in ipts {
    if p < nb_vertices {
      c &= cts[p];
      cnt += 1;
    }
  }
  // a secondary with its primary only
  let c = (c & !secondary) | (c & secondary & (c << 1));
  if cnt == 0 || c == 0 {
    return Ok(());
  }
  let sub: Vec<&dyn SignedDistance> = (0..list.len())
    .filter(|k| c & (1 << k) != 0)
    .map(|k| list[k])
    .collect();
  for &p in ipts {
    if p >= nb_vertices && done.insert(p) {
      let mut xref = sm.point(p).to_vec();
      // as GetFEM: a failed projection, or one outside the convex, is
      // not applied
      if multi_projection(&sub, &mut xref) && re.val(&xref) <= 1e-8 {
        moves.push((p, xref));
      }
    }
  }
  Ok(())
}

/// Each zone of zones2 merged with the zones of zones1 it shares a
/// sub-zone with
pub(crate) fn merge_zonesets(zones1: &mut ZoneSet, zones2: &ZoneSet) {
  for z2 in zones2 {
    let mut z = z2.clone();
    let (meet, rest): (Vec<Zone>, Vec<Zone>) = std::mem::take(zones1)
      .into_iter()
      .partition(|y| y.iter().any(|s| z.contains(s)));
    for y in meet {
      z.extend(y);
    }
    *zones1 = rest.into_iter().collect();
    zones1.insert(z);
  }
}

/// The sub-zone with its '0' replaced by '+' and '-' in all ways, merged
/// with the zones it shares a sub-zone with
pub(crate) fn merge_zoneset(zones: &mut ZoneSet, subz: &str) {
  let mut z = Zone::new();
  let mut stack = vec![subz.to_string()];
  while let Some(s) = stack.pop() {
    match s.find('0') {
      Some(i) => {
        for c in ['+', '-'] {
          let mut t = s.clone();
          t.replace_range(i..i + 1, &c.to_string());
          stack.push(t);
        }
      }
      None => {
        z.insert(s);
      }
    }
  }
  let (meet, mut rest): (Vec<Zone>, Vec<Zone>) = std::mem::take(zones)
    .into_iter()
    .partition(|y| y.iter().any(|s| z.contains(s)));
  for y in meet {
    z.extend(y);
  }
  rest.push(z);
  *zones = rest.into_iter().collect();
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn fronts_of_cut_vertices() {
    // vertex 0 on the front of level set 1, vertex 1 on the primary of 0
    // and the secondaries of 0 and 1 (a front of neither)
    let c = CutConvex {
      mesh: Arc::new(MeshData::new(3)),
      on: vec![[2, 2], [1, 3]],
      zones: ZoneSet::new(),
      graze: vec![],
    };
    assert!(c.on_front(0, 1) && !c.on_front(0, 0));
    assert!(c.on_front(1, 0) && !c.on_front(1, 1));
  }

  #[test]
  fn zonesets() {
    let mut zs = ZoneSet::new();
    merge_zoneset(&mut zs, "+-");
    merge_zoneset(&mut zs, "--");
    assert_eq!(zs.len(), 2);
    // '0': both signs, joining the two zones
    merge_zoneset(&mut zs, "0-");
    assert_eq!(zs.len(), 1);
    assert_eq!(zs.iter().next().unwrap().len(), 2);
  }
}
