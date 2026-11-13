//! Raytracing and projection transformations (port of the contact
//! transformations of GetFEM's getfem_contact_and_friction_common.cc): from
//! a point of a slave boundary to the nearest master boundary or rigid
//! obstacle, in the deformed configuration.

use std::collections::BTreeMap;
use std::sync::Arc;

use getfem_fem::fem::{FemCtx, PFem, Vectorial, real_base_values_at};
use getfem_geom::geotrans::{PGeoTrans, PointGeom};
use getfem_geom::mesh::{Mesh, MeshData};
use getfem_geom::region::{Region, RegionSet};
use getfem_geom::search::RTree;
use getfem_linalg::tensor::{Lu, Mat};

use crate::defs::{Defs, Support, VarDef};
use crate::ga::{CompileCtx, GaFunction};
use crate::semantic::{Ctx, analyse, analyse_str};
use crate::{Error, Result};
use getfem_linalg::{vect_dist2, vect_norm2, vect_sp};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ContactKind {
  Raytracing,
  Projection,
}

impl ContactKind {
  fn name(self) -> &'static str {
    match self {
      ContactKind::Raytracing => "raytracing",
      ContactKind::Projection => "projection",
    }
  }
}

/// A contact boundary: a stored region of faces of a mesh, deformed by a
/// displacement variable.
#[derive(Clone, Debug)]
pub struct ContactBoundary {
  pub mesh: Mesh,
  pub disp: String,
  pub region: Region,
  pub slave: bool,
}

/// A raytracing or projection transformation (GetFEM
/// add_raytracing_transformation, add_projection_transformation).
#[derive(Clone, Debug)]
pub struct Contact {
  pub kind: ContactKind,
  /// distance beyond which a contact is not considered
  pub release: f64,
  pub boundaries: Vec<ContactBoundary>,
  /// level set expressions of rigid obstacles (of X, or x, y, z, w) and the
  /// dimension of X
  pub obstacles: Vec<(String, usize)>,
}

impl Contact {
  /// (variable, transformation) pairs the target depends on: the
  /// displacements of the boundaries ("") and those of the master
  /// boundaries at the target (name)
  pub(crate) fn deps(&self, name: &str) -> Vec<(String, String)> {
    let mut d = vec![];
    for b in &self.boundaries {
      d.push((b.disp.clone(), String::new()));
      if !b.slave {
        d.push((b.disp.clone(), name.to_string()));
      }
    }
    d.sort();
    d.dedup();
    d
  }
}

/// A displacement field on one element.
pub(crate) struct Disp {
  pub pf: PFem,
  pub gt: PGeoTrans,
  pub g: Mat,
  /// coefficients of the basic DOFs of the element (i*N + c)
  pub coef: Vec<f64>,
}

/// The deformed configuration at a reference point.
pub(crate) struct DPoint {
  pub xref: Vec<f64>,
  pub geo: PointGeom,
  /// x + u
  pub phi: Vec<f64>,
  /// I + grad u
  pub f: Mat,
  /// basis values val[i] and gradients grad[i + nd*r]
  pub val: Vec<f64>,
  pub grad: Vec<f64>,
}

/// sign as gmm::sgn (1 at 0)
fn sgn(x: f64) -> f64 {
  if x < 0.0 { -1.0 } else { 1.0 }
}

/// m v
fn mv(m: &Mat, v: &[f64]) -> Vec<f64> {
  (0..m.nrows())
    .map(|r| (0..m.ncols()).map(|c| m[(r, c)] * v[c]).sum())
    .collect()
}

/// m^T v
fn mtv(m: &Mat, v: &[f64]) -> Vec<f64> {
  (0..m.ncols())
    .map(|c| (0..m.nrows()).map(|r| m[(r, c)] * v[r]).sum())
    .collect()
}

fn inverse(m: &Mat) -> Result<(Mat, f64)> {
  let lu = Lu::new(m).map_err(|e| Error::InvalidArgument(e.to_string()))?;
  Ok((lu.inverse(), lu.det()))
}

impl Disp {
  /// The deformed point x + u only (no derivatives), the same operations as
  /// at(): the residual at GetFEM's line-search trial points
  pub fn pos(&self, xref: &[f64]) -> Result<Vec<f64>> {
    let pf = &self.pf;
    if !pf.is_equivalent()
      || pf.real_element_defined()
      || pf.target_dim() != 1
      || !matches!(pf.vectorial(), Vectorial::None)
    {
      return Ok(self.at(xref)?.phi);
    }
    let mut phi = self.gt.transform(xref, &self.g)?;
    let v = pf.base_value(xref)?;
    let (n, val) = (phi.len(), v.data());
    for (vi, ci) in val.iter().zip(self.coef.chunks_exact(n)) {
      for (p, a) in phi.iter_mut().zip(ci) {
        *p += a * vi;
      }
    }
    Ok(phi)
  }

  pub fn at(&self, xref: &[f64]) -> Result<DPoint> {
    let geo = self.gt.at(&self.g, xref)?;
    let (val, grad, _) =
      real_base_values_at(&self.pf, &FemCtx::new(&self.gt, &self.g, xref, &geo), 1)?;
    let n = geo.x.len();
    let nd = val.len();
    let mut phi = geo.x.clone();
    let mut f = Mat::identity(n);
    for i in 0..nd {
      for c in 0..n {
        let a = self.coef[i * n + c];
        phi[c] += a * val[i];
        for r in 0..n {
          f[(c, r)] += a * grad[i + nd * r];
        }
      }
    }
    Ok(DPoint {
      xref: xref.to_vec(),
      geo,
      phi,
      f,
      val,
      grad,
    })
  }
}

impl DPoint {
  /// normal of face f in the reference configuration (B n_ref, not unit)
  /// and in the deformed one (unit), as GetFEM compute_normal
  fn normals(&self, nref: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
    let n0 = mv(&self.geo.b, nref);
    let (finv, j) = inverse(&self.f)?;
    let mut n: Vec<f64> = mtv(&finv, &n0).iter().map(|v| v * sgn(j)).collect();
    let l = vect_norm2(&n);
    n.iter_mut().for_each(|v| *v /= l);
    Ok((n0, n))
  }

  /// (I + grad u) K: derivative of the deformed point w.r.t. xref
  fn tangent(&self) -> Mat {
    self.f.matmul(&self.geo.k)
  }
}

/// Values of expr (of X) and of its gradient at a point.
pub(crate) struct Obstacle {
  f: GaFunction,
  df: Option<GaFunction>,
  n: usize,
}

impl Obstacle {
  /// expr with X a fixed-size variable of size n, macros x, y, z, w (as
  /// GetFEM's obstacle functions)
  pub(crate) fn new(
    expr: &str,
    n: usize,
    defs: &Defs,
    data: &BTreeMap<String, Vec<f64>>,
  ) -> Result<Self> {
    let mut d = defs.clone();
    d.vars.insert(
      "X".into(),
      VarDef {
        is_data: false,
        support: Support::Fixed,
        qdims: vec![n],
        disabled: false,
      },
    );
    for (k, m) in ["x", "y", "z", "w"].iter().enumerate().take(n) {
      d.macros.add(m, &format!("X({})", k + 1))?;
    }
    let ctx = Ctx {
      meshdim: n,
      ref_elt_dim: n,
      option: 1,
      ignore_x: true,
      ..Ctx::default()
    };
    let (tree, _) = analyse_str(expr, &d, &ctx)?;
    let fixed = |v: &str| data.get(v).cloned();
    let no_trans = |_: &str| None;
    let no_der = |_: &str, _: &str, _: &str| None;
    let cx = CompileCtx {
      defs: &d,
      fixed: &fixed,
      inputs: &["X"],
      vectorized: &|_| false,
      trans: &no_trans,
      trans_der: &no_der,
      internal: &|_| false,
    };
    let unsupported = || Error::Unsupported {
      expr: expr.to_string(),
      pos: 0,
      msg: "obstacle: this expression is not supported".into(),
    };
    if tree
      .root
      .is_some_and(|r| tree.n[r].sizes.iter().product::<usize>() != 1)
    {
      return Err(Error::InvalidArgument(format!(
        "obstacle {expr}: the level set function has to be scalar"
      )));
    }
    let f = match tree.root {
      Some(_) => GaFunction::new(&tree, &cx, &[n]).ok_or_else(unsupported)?,
      None => {
        let (zero, _) = analyse_str("0", &d, &ctx)?;
        GaFunction::new(&zero, &cx, &[n]).ok_or_else(unsupported)?
      }
    };
    let df = match crate::derive::derive(&tree, &d, "X", "", 1, None)? {
      Some(dt) => {
        let (dt, _) = analyse(dt, &d, &ctx)?;
        match dt.root {
          Some(_) => Some(GaFunction::new(&dt, &cx, &[n]).ok_or_else(unsupported)?),
          None => None,
        }
      }
      None => None,
    };
    Ok(Obstacle { f, df, n })
  }

  fn eval(&self, x: &[f64]) -> Result<f64> {
    let mut v = vec![];
    self.f.eval(&[x], &mut v)?;
    Ok(v[0])
  }

  fn grad(&self, x: &[f64]) -> Result<Vec<f64>> {
    match &self.df {
      Some(p) => {
        let mut v = vec![];
        p.eval(&[x], &mut v)?;
        Ok(v)
      }
      None => Ok(vec![0.0; self.n]),
    }
  }
}

/// A boundary ready for one assembly.
pub(crate) struct RtBoundary {
  pub mesh: Arc<MeshData>,
  pub disp: String,
  pub region: Arc<RegionSet>,
  pub slave: bool,
}

struct FaceBox {
  ib: usize,
  cv: usize,
  f: usize,
  mean_normal: Vec<f64>,
}

/// A point on a rigid obstacle: point, unit normal, signed distance.
type RigidHit = (Vec<f64>, Vec<f64>, f64);

/// Provides the displacement of a variable on an element.
pub(crate) type DispOf<'a> = dyn Fn(&str, &MeshData, usize) -> Result<Disp> + 'a;

/// A contact transformation ready for one assembly: the deformed master
/// faces in a box tree.
pub(crate) struct InitializedContact {
  kind: ContactKind,
  release: f64,
  pub bounds: Vec<RtBoundary>,
  tree: RTree,
  faces: Vec<FaceBox>,
  obstacles: Vec<Obstacle>,
}

/// The target of a point.
pub(crate) struct Hit {
  /// 0: none, 1: a master face, 2: a rigid obstacle
  pub status: usize,
  /// status 1: target element, face, reference point
  pub target: Option<(usize, usize, Vec<f64>)>,
  /// status 2: the point on the obstacle
  pub x: Option<Vec<f64>>,
  /// derivatives w.r.t. the displacement at the source (disp_x, test
  /// functions of the source element) and at the target (disp_y, test
  /// functions of the target element): d[k + nt*j]
  pub disp_x: String,
  pub der_x: Vec<f64>,
  pub disp_y: String,
  pub der_y: Vec<f64>,
}

/// N-1 orthonormal vectors orthogonal to the unit vector nv (GetFEM uses
/// random vectors; the canonical ones here)
fn tangents(nv: &[f64]) -> Vec<Vec<f64>> {
  let n = nv.len();
  let mut out: Vec<Vec<f64>> = vec![];
  for k in 0..n {
    if out.len() == n - 1 {
      break;
    }
    let mut t = vec![0.0; n];
    t[k] = 1.0;
    let a = vect_sp(&t, nv);
    t.iter_mut().zip(nv).for_each(|(x, y)| *x -= a * y);
    for s in &out {
      let a = vect_sp(&t, s);
      t.iter_mut().zip(s).for_each(|(x, y)| *x -= a * y);
    }
    let l = vect_norm2(&t);
    if l >= 1e-5 {
      t.iter_mut().for_each(|x| *x /= l);
      out.push(t);
    }
  }
  out
}

/// y0ref + sum a_i t_i
/// The centre of face f of the reference convex cr (mean of its points):
/// the start of the searches on the face, whatever the element's numbering
/// (GetFEM: its first node, from which a curved face's extrapolation can
/// attract the Newton to a root outside the face)
fn face_centre(cr: &getfem_geom::convex_ref::PRef, f: usize) -> Vec<f64> {
  let ind = cr.structure().ind_points_of_face(f);
  let mut c = vec![0.0; cr.dim()];
  for &i in ind {
    c.iter_mut().zip(&cr.points()[i]).for_each(|(v, p)| *v += p);
  }
  c.iter_mut().for_each(|v| *v /= ind.len() as f64);
  c
}

fn on_face(y0ref: &[f64], t: &[Vec<f64>], a: &[f64]) -> Vec<f64> {
  let mut xref = y0ref.to_vec();
  for (ai, ti) in a.iter().zip(t) {
    xref.iter_mut().zip(ti).for_each(|(v, w)| *v += ai * w);
  }
  xref
}

/// The size h of the element at the reference point y0 of a face: the
/// root mean square stretch K t over the orthonormal basis ti of the
/// face's reference plane (independent of the basis), relative to the
/// reference face. The Newton iterations on the face scale their
/// tolerances by it: they depend neither on the length unit nor on the
/// element size (GetFEM with ref/0013).
fn face_size(d: &Disp, y0: &[f64], ti: &[Vec<f64>]) -> Result<f64> {
  let k = &d.at(y0)?.geo.k;
  let l2: f64 = ti
    .iter()
    .map(|t| {
      let v = mv(k, t);
      vect_sp(&v, &v)
    })
    .sum();
  Ok((l2 / ti.len() as f64).sqrt())
}

/// the inverse of m, if |det m| > tol
fn small_inverse(m: &Mat, tol: f64) -> Option<Mat> {
  let lu = Lu::new(m).ok()?;
  (lu.det().abs() > tol).then(|| lu.inverse())
}

impl InitializedContact {
  pub(crate) fn new(
    name: &str,
    c: &Contact,
    defs: &Defs,
    data: BTreeMap<String, Vec<f64>>,
    disp: &DispOf,
  ) -> Result<Self> {
    let mut bounds = vec![];
    for b in &c.boundaries {
      let m = b.mesh.snapshot();
      let region = b.region.resolve(&m)?;
      let (n, what) = (m.dim(), c.kind.name());
      for (cv, f) in region.iter() {
        if f.is_none() {
          return Err(Error::InvalidArgument(format!(
            "{what} transformation {name}: the contact boundary {} of {} contains elements, faces expected",
            b.region.id(),
            b.disp
          )));
        }
        let p = m.trans_of_convex(cv).dim();
        if p != n {
          return Err(Error::InvalidArgument(format!(
            "{what} transformation {name}: element {cv} of the contact boundary of {} has dimension {p} in a mesh of dimension {n}; contact boundaries are faces of elements of the mesh dimension",
            b.disp
          )));
        }
      }
      bounds.push(RtBoundary {
        mesh: m,
        disp: b.disp.clone(),
        region,
        slave: b.slave,
      });
    }
    let mut obstacles = vec![];
    for (e, n) in &c.obstacles {
      obstacles.push(Obstacle::new(e, *n, defs, &data)?);
    }
    let mut rt = InitializedContact {
      kind: c.kind,
      release: c.release,
      bounds,
      tree: RTree::new(1, 0.0),
      faces: vec![],
      obstacles,
    };
    rt.face_boxes(disp)?;
    Ok(rt)
  }

  /// bounding boxes of the deformed master faces (GetFEM compute_face_boxes)
  fn face_boxes(&mut self, disp: &DispOf) -> Result<()> {
    let mut tree: Option<RTree> = None;
    for (ib, b) in self.bounds.iter().enumerate() {
      if b.slave {
        continue;
      }
      let m = &b.mesh;
      let n = m.dim();
      let tr = tree.get_or_insert_with(|| RTree::new(n, 0.0));
      for (cv, f) in b.region.iter() {
        let f = f.expect("faces checked");
        let d = disp(&b.disp, m, cv)?;
        let gt = &d.gt;
        let nref = &gt.normals()[f];
        let (mut lo, mut hi) = (vec![f64::INFINITY; n], vec![f64::NEG_INFINITY; n]);
        let mut nm = vec![0.0; n];
        for &ip in gt.structure().ind_points_of_face(f) {
          let p = d.at(&gt.geometric_nodes()[ip])?;
          let (_, nv) = p.normals(nref)?;
          nm.iter_mut().zip(&nv).for_each(|(a, b)| *a += b);
          for k in 0..n {
            lo[k] = lo[k].min(p.phi[k]);
            hi[k] = hi[k].max(p.phi[k]);
          }
        }
        // security coefficient (nonlinear transformations)
        let h = (0..n).map(|k| hi[k] - lo[k]).fold(0.0, f64::max);
        lo.iter_mut().for_each(|v| *v -= 0.15 * h);
        hi.iter_mut().for_each(|v| *v += 0.15 * h);
        tr.add_box(&lo, &hi, self.faces.len())?;
        let l = vect_norm2(&nm);
        nm.iter_mut().for_each(|v| *v /= l);
        self.faces.push(FaceBox {
          ib,
          cv,
          f,
          mean_normal: nm,
        });
      }
    }
    if let Some(t) = tree {
      self.tree = t;
    }
    Ok(())
  }

  /// Raytrace from pt_x along n_x onto the deformed face f of d (GetFEM:
  /// Newton on the tangential residual), the target point if converged
  fn raytrace(d: &Disp, f: usize, pt_x: &[f64], n_x: &[f64]) -> Result<Option<DPoint>> {
    let n = pt_x.len();
    let cr = d.pf.ref_convex();
    let y0ref = face_centre(cr, f);
    let ti = tangents(&cr.normals()[f]);
    let tt = tangents(n_x);
    let h = face_size(d, &y0ref, &ti)?;
    // the residual from the deformed point alone; the full point
    // (its derivatives) only where the Newton matrix needs it, as GetFEM
    let res = |a: &[f64]| -> Result<Vec<f64>> {
      let phi = d.pos(&on_face(&y0ref, &ti, a))?;
      let dxy: Vec<f64> = phi.iter().zip(pt_x).map(|(a, b)| a - b).collect();
      Ok(tt.iter().map(|t| vect_sp(&dxy, t)).collect())
    };
    let full = |a: &[f64]| d.at(&on_face(&y0ref, &ti, a));
    let hess = |p: &DPoint| -> Mat {
      let g = p.tangent();
      let mut m = Mat::zeros(&[n - 1, n - 1]);
      for (i, t) in ti.iter().enumerate() {
        let gt = mv(&g, t);
        for (j, s) in tt.iter().enumerate() {
          m[(j, i)] = vect_sp(&gt, s);
        }
      }
      m
    };
    let mut a = vec![0.0; n - 1];
    let mut r = res(&a)?;
    // the full point at a, when computed
    let mut p: Option<DPoint> = None;
    let mut residual = vect_norm2(&r);
    let (mut nbfail, mut niter) = (0, 0);
    while residual > 2e-12 * h && niter <= 30 {
      let mut inv = None;
      for sub in 0..=4 {
        if p.is_none() {
          p = Some(full(&a)?);
        }
        let m = hess(p.as_ref().expect("computed"));
        inv = small_inverse(&m, 1e-15 * h.powi(n as i32 - 1));
        if inv.is_some() {
          break;
        }
        // GetFEM perturbs a randomly
        a.iter_mut()
          .enumerate()
          .for_each(|(i, v)| *v += 1e-7 * (1.0 + (sub + i) as f64) / 7.0);
        p = None;
      }
      let Some(hinv) = inv else { break };
      let dir: Vec<f64> = mv(&hinv, &r).iter().map(|v| -v).collect();
      if vect_norm2(&dir) > 10.0 {
        nbfail += 1;
      }
      if nbfail >= 4 {
        break;
      }
      let mut lambda = 1.0;
      let mut next = None;
      for j in 0..5 {
        let b: Vec<f64> = a.iter().zip(&dir).map(|(x, y)| x + lambda * y).collect();
        let r2 = res(&b)?;
        let better = vect_norm2(&r2) < residual;
        next = Some((b, r2));
        if better {
          break;
        }
        lambda /= if j < 3 { 2.0 } else { 5.0 };
      }
      let (b, r2) = next.expect("one step");
      residual = vect_norm2(&r2);
      (a, r, p) = (b, r2, None);
      let dist_ref = vect_norm2(&a);
      if (niter > 1 && dist_ref > 15.0) || (niter > 5 && dist_ref > 8.0) {
        break;
      }
      niter += 1;
    }
    if vect_norm2(&r) >= 2e-6 * h {
      return Ok(None);
    }
    Ok(Some(match p {
      Some(p) => p,
      None => full(&a)?,
    }))
  }

  /// Projection of pt_x onto the deformed face f of d (GetFEM: quasi
  /// Newton on the distance), the target point if converged
  fn project(d: &Disp, f: usize, pt_x: &[f64]) -> Result<Option<DPoint>> {
    let n = pt_x.len();
    let cr = d.pf.ref_convex();
    let y0ref = face_centre(cr, f);
    let ti = tangents(&cr.normals()[f]);
    let h = face_size(d, &y0ref, &ti)?;
    // |x(a) - pt_x| / 2 and the gradient of |x(a) - pt_x|^2 / 2 (as GetFEM)
    let cost = |a: &[f64]| -> Result<(f64, Vec<f64>, DPoint)> {
      let p = d.at(&on_face(&y0ref, &ti, a))?;
      let dxy: Vec<f64> = p.phi.iter().zip(pt_x).map(|(a, b)| a - b).collect();
      let g = p.tangent();
      let grad = ti.iter().map(|t| vect_sp(&mv(&g, t), &dxy)).collect();
      Ok((vect_norm2(&dxy) / 2.0, grad, p))
    };
    const EPS: f64 = 1e-10;
    let mut a = vec![0.0; n - 1];
    let (mut dist, mut grad, mut p) = cost(&a)?;
    let mut niter = 0;
    while vect_norm2(&grad) > 1e-7 * h * h && niter <= 50 {
      let mut inv = None;
      for sub in 0..=4 {
        let (_, g0, _) = cost(&a)?;
        let mut m = Mat::zeros(&[n - 1, n - 1]);
        for i in 0..n - 1 {
          let mut b = a.clone();
          b[i] += EPS;
          let (_, gb, _) = cost(&b)?;
          for j in 0..n - 1 {
            m[(j, i)] = (gb[j] - g0[j]) / EPS;
          }
        }
        inv = small_inverse(&m, 1e-15 * h.powi(2 * n as i32 - 2));
        if inv.is_some() {
          break;
        }
        a.iter_mut()
          .enumerate()
          .for_each(|(i, v)| *v += 1e-7 * (1.0 + (sub + i) as f64) / 7.0);
      }
      let Some(hinv) = inv else { break };
      let dir: Vec<f64> = mv(&hinv, &grad).iter().map(|v| -v).collect();
      let mut lambda = 1.0;
      let mut b = a.clone();
      while lambda >= 1e-3 {
        b = a.iter().zip(&dir).map(|(x, y)| x + lambda * y).collect();
        if cost(&b)?.0 < dist {
          break;
        }
        b = a.iter().zip(&dir).map(|(x, y)| x - lambda * y).collect();
        if cost(&b)?.0 < dist {
          break;
        }
        lambda /= 2.0;
      }
      a = b;
      (dist, grad, p) = cost(&a)?;
      niter += 1;
    }
    // GetFEM tries BFGS when this fails; here the pair is not considered
    Ok((vect_norm2(&grad) < 2e-6 * h * h).then_some(p))
  }

  /// The nearest rigid obstacle within the release distance: point, unit
  /// normal, signed distance
  fn obstacle(&self, pt_x: &[f64], n_x: &[f64]) -> Result<Option<RigidHit>> {
    let (mut d0, mut io, mut n_y) = (1e300, None, vec![]);
    for (i, o) in self.obstacles.iter().enumerate() {
      if o.n != pt_x.len() {
        return Err(Error::InvalidArgument(format!(
          "obstacle of dimension {} in a mesh of dimension {}",
          o.n,
          pt_x.len()
        )));
      }
      let d1 = o.eval(pt_x)?;
      if d1.abs() < self.release && d1 < d0 {
        let g = o.grad(pt_x)?;
        if vect_sp(&g, n_x) < 0.0 {
          (d0, io, n_y) = (d1, Some(i), g);
        }
      }
    }
    let Some(i) = io else { return Ok(None) };
    let o = &self.obstacles[i];
    let mut pt = pt_x.to_vec();
    let (mut d1, mut nit, mut nb_fail) = (d0, 0, 0);
    let (mut alpha, mut beta) = (0.0, 0.0);
    while d1.abs() > 1e-13
      && {
        nit += 1;
        nit < 50
      }
      && nb_fail < 3
    {
      if nit != 1 {
        n_y = o.grad(&pt)?;
      }
      let mut lambda = 1.0;
      let mut d2 = d1;
      while lambda >= 1e-3 {
        match self.kind {
          // along n_x (GetFEM)
          ContactKind::Raytracing => {
            alpha = beta - lambda * d1 / vect_sp(&n_y, n_x);
            pt = pt_x.iter().zip(n_x).map(|(x, v)| x + alpha * v).collect();
          }
          // along the gradient (GetFEM's version is broken)
          ContactKind::Projection => {
            let s = lambda * d1 / vect_sp(&n_y, &n_y);
            pt = pt.iter().zip(&n_y).map(|(x, v)| x - s * v).collect();
          }
        }
        d2 = o.eval(&pt)?;
        if d2.abs() < d1.abs() {
          break;
        }
        lambda /= 2.0;
      }
      if self.kind == ContactKind::Raytracing && (beta - d1 / vect_sp(&n_y, n_x)).abs() > 500.0 {
        nb_fail += 1;
      }
      beta = alpha;
      d1 = d2;
    }
    let dist = vect_dist2(&pt, pt_x);
    // GetFEM warns on failure and keeps an undefined point
    if d1.abs() > 1e-8 || dist > self.release {
      return Ok(None);
    }
    let l = vect_norm2(&n_y);
    n_y.iter_mut().for_each(|v| *v /= l);
    Ok(Some((pt, n_y, dist * sgn(d0))))
  }

  /// The target of the point xref of face face_x of element cv_x of m_x
  /// (GetFEM raytracing/projection_interpolate_transformation::transform).
  pub(crate) fn transform(
    &self,
    disp: &DispOf,
    m_x: &MeshData,
    cv_x: usize,
    face_x: usize,
    xref: &[f64],
    ders: bool,
  ) -> Result<Hit> {
    let what = self.kind.name();
    let ib_x = self
      .bounds
      .iter()
      .position(|b| {
        b.mesh.geometry_version() == m_x.geometry_version() && b.region.is_in_face(cv_x, face_x)
      })
      .ok_or_else(|| {
        Error::InvalidArgument(format!(
          "{what} transformation: face {face_x} of element {cv_x} is on no contact boundary"
        ))
      })?;
    let b_x = &self.bounds[ib_x];
    let n = m_x.dim();
    let d_x = disp(&b_x.disp, m_x, cv_x)?;
    let px = d_x.at(xref)?;
    let (_, n_x) = px.normals(&d_x.gt.normals()[face_x])?;
    let pt_x = px.phi.clone();

    // (signed distance, n_y, target: (ib, cv, f, point) or obstacle point)
    #[allow(clippy::large_enum_variant)]
    enum T {
      Face(usize, usize, usize, DPoint),
      Rigid(Vec<f64>),
    }
    let mut best: Option<(f64, Vec<f64>, T)> = self
      .obstacle(&pt_x, &n_x)?
      .map(|(p, ny, g)| (g, ny, T::Rigid(p)));
    let (lo, hi): (Vec<f64>, Vec<f64>) = (
      pt_x.iter().map(|v| v - self.release).collect(),
      pt_x.iter().map(|v| v + self.release).collect(),
    );
    let cand = if self.faces.is_empty() {
      vec![]
    } else {
      self
        .tree
        .find_line_intersecting_boxes(&pt_x, &n_x, Some((&lo, &hi)))?
    };
    for id in cand {
      let fb = &self.faces[id];
      let b_y = &self.bounds[fb.ib];
      let same_mesh = b_y.mesh.geometry_version() == m_x.geometry_version();
      // criterion 1: compatible normals, not the same element
      if vect_sp(&fb.mean_normal, &n_x) >= 0.0 || (fb.cv == cv_x && same_mesh) {
        continue;
      }
      let d_y = disp(&b_y.disp, &b_y.mesh, fb.cv)?;
      let py = match self.kind {
        ContactKind::Raytracing => Self::raytrace(&d_y, fb.f, &pt_x, &n_x)?,
        ContactKind::Projection => Self::project(&d_y, fb.f, &pt_x)?,
      };
      // criteria 2, 3: converged, inside the element
      let Some(py) = py else { continue };
      if d_y.pf.ref_convex().is_in(&py.xref)? >= 1e-6 {
        continue;
      }
      // criterion 4: release distance
      let diff: Vec<f64> = pt_x.iter().zip(&py.phi).map(|(a, b)| a - b).collect();
      let mut g = vect_norm2(&diff);
      if g > self.release {
        continue;
      }
      let (n0_y, n_y) = py.normals(&d_y.gt.normals()[fb.f])?;
      g *= sgn(vect_sp(&diff, &n_y));
      // criteria 5, 7: nearest (signed distance)
      if best.as_ref().is_some_and(|b| b.0 < g) {
        continue;
      }
      // criterion 1 again
      if vect_sp(&n_y, &n_x) >= 0.0 {
        continue;
      }
      // criterion 6 (self contact): the unit normal in the reference
      // configuration (GetFEM: n0 not unit, ref/0012)
      if same_mesh {
        let l0 = vect_norm2(&n0_y);
        let n0_y: Vec<f64> = n0_y.iter().map(|v| v / l0).collect();
        let dr: Vec<f64> = px.geo.x.iter().zip(&py.geo.x).map(|(a, b)| a - b).collect();
        let rd = vect_norm2(&dr);
        if rd < 4.0 * self.release && vect_sp(&dr, &n0_y) < -0.01 * rd {
          continue;
        }
      }
      best = Some((g, n_y, T::Face(fb.ib, fb.cv, fb.f, py)));
    }
    let mut hit = Hit {
      status: 0,
      target: None,
      x: None,
      disp_x: b_x.disp.clone(),
      der_x: vec![],
      disp_y: String::new(),
      der_y: vec![],
    };
    let Some((g, n_y, t)) = best else {
      return Ok(hit);
    };
    let (f_y, val_y) = match &t {
      T::Face(ib, cv, f, py) => {
        hit.status = 1;
        hit.target = Some((*cv, *f, py.xref.clone()));
        hit.disp_y = self.bounds[*ib].disp.clone();
        (py.f.clone(), py.val.clone())
      }
      T::Rigid(p) => {
        hit.status = 2;
        hit.x = Some(p.clone());
        (Mat::identity(n), vec![])
      }
    };
    if !ders {
      return Ok(hit);
    }
    // projector: raytracing I - n_x n_y / n_x.n_y, projection I - n_y n_y
    let mut pr = Mat::identity(n);
    for r in 0..n {
      for c in 0..n {
        pr[(r, c)] -= match self.kind {
          ContactKind::Raytracing => n_x[r] * n_y[c] / vect_sp(&n_x, &n_y),
          ContactKind::Projection => n_y[r] * n_y[c],
        };
      }
    }
    let m1 = inverse(&f_y)?.0.matmul(&pr);
    // raytracing: g Dn_x[Test_u], Dn_x = -(I - n_x n_x) F_x^-T Grad_Test_u^T n_x
    let m2 = m1.matmul(&inverse(&px.f)?.0.transposed());
    let nd = px.val.len();
    let nt = nd * n;
    hit.der_x = vec![0.0; nt * n];
    for i in 0..nd {
      for c in 0..n {
        let k = i * n + c;
        for j in 0..n {
          let mut v = m1[(j, c)] * px.val[i];
          if self.kind == ContactKind::Raytracing {
            v -= g
              * n_x[c]
              * (0..n)
                .map(|r| m2[(j, r)] * px.grad[i + nd * r])
                .sum::<f64>();
          }
          hit.der_x[k + nt * j] = v;
        }
      }
    }
    let ny = val_y.len() * n;
    hit.der_y = vec![0.0; ny * n];
    for (i, v) in val_y.iter().enumerate() {
      for c in 0..n {
        for j in 0..n {
          hit.der_y[i * n + c + ny * j] = -m1[(j, c)] * v;
        }
      }
    }
    Ok(hit)
  }
}
