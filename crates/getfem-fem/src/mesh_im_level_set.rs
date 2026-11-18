//! Integration on the parts of the elements cut by level sets (GetFEM's
//! mesh_im_level_set); 2D not on GetFEM's sub-simplices (DEVIATIONS).
//!
//! 2D: each element's reference convex carries a lattice of triangles (n
//! per direction, `set_lattice`, `set_lattice_of_element`); a lattice
//! triangle cut by a level set becomes a few 6-node P2 triangles
//! (`cut2d`: the rules G1-G6 of the design doc, caps, clipping at the
//! element's boundary, hierarchical splitting of unresolved triangles),
//! each with the symmetric 3-point rule. On the level set: Gauss points on
//! the curve's quadratic pieces. Several level sets: GetFEM's set
//! expressions.
//!
//! 3D, and 2D with cracks (GetFEM's): the sub-simplices of
//! mesh_level_set's cut (curved onto the level sets), the selected ones
//! with a simplex rule (`set_simplex_im`), those touching a crack tip with
//! IM_QUASI_POLAR of `set_simplex_im`; on the level set, the faces between
//! selected and unselected sub-simplices with the rule's face points
//! (cracks: GetFEM's faces with all their nodes on the lips).

use std::ops::Deref;

use crate::cut2d::{Cells, CutFunction, build, quad_at, quad_d};
use crate::integration::{PIntegration, approx_integration, approx_integration_on};
use crate::level_set::{ConvexLevelSet, LevelSet};
use crate::mesh_im::MeshIm;
use crate::mesh_level_set::{CutConvex, MeshLevelSet, jacobian};
use crate::{Error, Result};
use getfem_geom::mesh::Mesh;
use getfem_linalg::tensor::Mat;
use getfem_linalg::vect_norm2;

/// Where to integrate (GetFEM's INTEGRATE_INSIDE, ..._OUTSIDE, ..._ALL,
/// ..._BOUNDARY); inside is where the level set is negative.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntegrateWhere {
  Inside,
  Outside,
  All,
  Boundary,
}

/// Real length of a reference tangent at a reference point, over |det J|.
type ArcLength<'a> = dyn Fn(&[f64], &[f64]) -> Result<f64> + 'a;

/// Gauss points per quadratic piece of the zero curves (integration on the
/// level set).
const NBOUNDARY: usize = 3;

/// A mesh_im on the parts of the elements selected by the level sets.
pub struct MeshImLevelSet {
  mesh: Mesh,
  lss: Vec<LevelSet>,
  where_: IntegrateWhere,
  csg: std::cell::RefCell<String>,
  lattice: std::cell::Cell<usize>,
  lattice_of: std::cell::RefCell<std::collections::BTreeMap<usize, usize>>,
  regular: MeshIm,
  mim: MeshIm,
  /// a copy of the level sets' cut (3D), cut again by `adapt` if they
  /// changed
  mls: std::cell::RefCell<MeshLevelSet>,
  /// the rule on the sub-simplices of the cut (3D, and 2D at cracks)
  simplex_im: std::cell::RefCell<Option<PIntegration>>,
  /// the base of the quasi-polar rules on the sub-simplices at crack tips
  /// (GetFEM base_singular_pim)
  singular_im: std::cell::RefCell<Option<PIntegration>>,
  /// the rule of each sub-simplex by its size (tau of `set_simplex_im`)
  cell_tolerance: std::cell::Cell<Option<f64>>,
}

/// The default tolerance of the rules of the cut cells (GetFEM: none, the
/// simplex rule on every cell).
pub const CELL_TOLERANCE: f64 = 1e-2;

impl MeshImLevelSet {
  /// The level sets of `mls` (primary functions) and where to integrate;
  /// set the regular method with `set_integration_method`, then `adapt`.
  /// With several level sets, the selected part is the intersection of
  /// their insides, or `set_level_set_boolean_operations`.
  pub fn new(mls: &MeshLevelSet, where_: IntegrateWhere) -> Result<Self> {
    let lss: Vec<LevelSet> = (0..mls.nb_level_sets())
      .map(|i| mls.level_set(i).clone())
      .collect();
    if lss.is_empty() {
      return Err(Error::InvalidArgument(
        "mesh_im_level_set: no level set".into(),
      ));
    }
    let mesh = mls.linked_mesh().clone();
    let mim = MeshIm::new(&mesh);
    if where_ == IntegrateWhere::Boundary {
      mim.set_normal_level_sets(lss.clone());
    }
    Ok(MeshImLevelSet {
      lss,
      where_,
      csg: std::cell::RefCell::new(String::new()),
      lattice: std::cell::Cell::new(2),
      lattice_of: std::cell::RefCell::new(Default::default()),
      regular: MeshIm::new(&mesh),
      mim,
      mls: std::cell::RefCell::new(mls.clone()),
      simplex_im: Default::default(),
      singular_im: Default::default(),
      cell_tolerance: std::cell::Cell::new(Some(CELL_TOLERANCE)),
      mesh,
    })
  }

  /// The part selected by several level sets, GetFEM's set expression:
  /// "a*b" intersection of the insides of the first and second level sets
  /// (the default: all intersected), "a+b" union, "a-b" difference, "!a"
  /// complement, parentheses; evaluated right to left as GetFEM's.
  pub fn set_level_set_boolean_operations(&self, description: &str) -> Result<()> {
    csg_eval(description, &vec![false; self.lss.len()])?;
    *self.csg.borrow_mut() = description.to_string();
    Ok(())
  }

  /// The lattice of every 2D element: n triangles per direction (default
  /// 2). Changing it changes the rules.
  pub fn set_lattice(&self, n: usize) -> Result<()> {
    if n == 0 {
      return Err(Error::InvalidArgument("lattice: n must be positive".into()));
    }
    self.lattice.set(n);
    Ok(())
  }

  /// The lattice of element cv (local refinement), overriding
  /// `set_lattice`.
  pub fn set_lattice_of_element(&self, cv: usize, n: usize) -> Result<()> {
    if n == 0 {
      return Err(Error::InvalidArgument("lattice: n must be positive".into()));
    }
    self.lattice_of.borrow_mut().insert(cv, n);
    Ok(())
  }

  /// The regular method (the elements not cut; 3D also the cut's
  /// sub-simplices unless `set_simplex_im`; 2D cells use the symmetric
  /// 3-point rule).
  pub fn set_integration_method(
    &self,
    arg: impl crate::mesh_im::IntegrationMethodArg + Clone,
  ) -> Result<()> {
    self.regular.set_integration_method(arg.clone())?;
    self.mim.set_integration_method(arg)
  }

  /// The rules on the sub-simplices of the cut elements (tau not in
  /// GetFEM):
  /// - reg on each sub-simplex (3D, and 2D at cracks; default: the regular
  ///   method where it is a simplex rule), by the sub-simplex's size s
  ///   (longest edge, the element's reference size 1): the lowest degree
  ///   k with s^(k+1) <= tau, never more points than reg, reg for cells
  ///   of size 1 and more; small cells get 1 point, tau 1e-2 makes the
  ///   cut cells about as accurate as a degree-4 rule on the element.
  ///   Volume integration only (on boundaries reg). tau = 0: reg on every
  ///   sub-simplex (GetFEM); default CELL_TOLERANCE.
  /// - sing, the base of the rules on the sub-simplices touching a crack
  ///   tip (IM_QUASI_POLAR(sing, vertex)): a rule on the square in 2D, on
  ///   the prism in 3D; None: reg there too. On boundaries (not GetFEM's):
  ///   reg's face rule on the faces touching a crack front graded toward
  ///   it, w -> w^2 (w the barycentric distance from the front), for
  ///   integrands like r^(-1/2) (the lip term of crack_sif).
  pub fn set_simplex_im(
    &self,
    reg: &PIntegration,
    sing: Option<&PIntegration>,
    tau: f64,
  ) -> Result<()> {
    use getfem_geom::convex_structure as cs;
    let a = reg.approx()?;
    if a.ref_convex().structure().nb_points() != a.dim() + 1 {
      return Err(Error::InvalidArgument(format!(
        "set_simplex_im: {} is not a simplex rule",
        reg.name()
      )));
    }
    if let Some(pim) = sing {
      let a = pim.approx()?;
      let st = a.structure();
      let ok = match a.dim() {
        2 => *st == *cs::parallelepiped_structure(2, 1),
        3 => *st == *cs::prism_p1_structure(3),
        _ => false,
      };
      if !ok {
        return Err(Error::InvalidArgument(format!(
          "Base integration method for quasi polar integration not convenient: {}",
          pim.name()
        )));
      }
    }
    if !(0.0..1.0).contains(&tau) {
      return Err(Error::InvalidArgument(format!(
        "set_simplex_im: tau {tau}, 0 <= tau < 1 expected"
      )));
    }
    *self.simplex_im.borrow_mut() = Some(reg.clone());
    *self.singular_im.borrow_mut() = sing.cloned();
    self.cell_tolerance.set((tau > 0.0).then_some(tau));
    Ok(())
  }

  /// Build the methods of the cut elements (and drop the elements outside
  /// the selected part).
  pub fn adapt(&self) -> Result<()> {
    let m = self.mesh.snapshot();
    let csg = self.csg.borrow().clone();
    // the expression is checked when it is set
    let sel = |inside: &[bool]| -> bool {
      if csg.is_empty() {
        inside.iter().all(|b| *b)
      } else {
        csg_eval(&csg, inside).unwrap_or(false)
      }
    };
    let ls_ids: Vec<usize> = self.lss.iter().map(|l| l.id()).collect();
    let mut cut3 = false;
    // cracks: GetFEM's sub-simplices in 2D too (decided 21 Oct 2026)
    let crack = self.lss.iter().any(|l| l.has_secondary());
    for cv in self.mesh.convex_index() {
      let reg = self.regular.int_method_of_element(cv)?;
      let n = *self
        .lattice_of
        .borrow()
        .get(&cv)
        .unwrap_or(&self.lattice.get());
      let cls: Vec<ConvexLevelSet> = self
        .lss
        .iter()
        .map(|l| l.of_convex(cv, 0))
        .collect::<Result<_>>()?;
      let cls2: Vec<Option<ConvexLevelSet>> = self
        .lss
        .iter()
        .map(|l| l.has_secondary().then(|| l.of_convex(cv, 1)).transpose())
        .collect::<Result<_>>()?;
      let gt = m.trans_of_convex(cv).clone();
      let g = m.points_of_convex(cv);
      let dim = reg.approx()?.dim();
      if dim == 3 || crack {
        if !cut3 {
          let mut mls = self.mls.borrow_mut();
          if !mls.is_up_to_date() {
            mls.adapt()?;
          }
          cut3 = true;
        }
        let mls = self.mls.borrow();
        let pim = if mls.is_convex_cut(cv) {
          let elem = |xref: &[f64]| -> Result<Mat> { Ok(gt.at(&g, xref)?.b) };
          let simplex = match self.simplex_im.borrow().clone() {
            Some(s) => s,
            None if reg.approx()?.ref_convex().structure().nb_points() == dim + 1 => reg.clone(),
            None => {
              return Err(Error::InvalidArgument(
                "mesh_im_level_set: set_simplex_im (a simplex rule for the cut)".into(),
              ));
            }
          };
          let sing = self
            .singular_im
            .borrow()
            .clone()
            .filter(|_| mls.crack_tip_convexes().is_ok_and(|t| t.contains(&cv)));
          method_of_sub_mesh(
            cv,
            mls.cut_convex(cv)?,
            &ls_ids,
            &cls,
            &cls2,
            &simplex,
            sing.as_ref(),
            self.cell_tolerance.get(),
            &reg,
            self.where_,
            &sel,
            &elem,
          )?
        } else {
          // not crossed: the regular rule or nothing; with cracks, the signs
          // at the centroid (GetFEM)
          let inside: Vec<bool> = if crack {
            let pts = reg.approx()?.ref_convex().points();
            let c: Vec<f64> = (0..dim)
              .map(|k| pts.iter().map(|p| p[k]).sum::<f64>() / pts.len() as f64)
              .collect();
            let mut v = vec![];
            for (l, l2) in cls.iter().zip(&cls2) {
              let in2 = match l2 {
                Some(l2) => l2.val(&c)? < 0.0,
                None => true,
              };
              v.push(l.val(&c)? < 0.0 && in2);
            }
            v
          } else {
            mls
              .primary_zone_of_convex(cv)?
              .chars()
              .map(|z| z == '-')
              .collect()
          };
          let s = sel(&inside);
          let keep = match self.where_ {
            IntegrateWhere::Inside => s,
            IntegrateWhere::Outside => !s,
            IntegrateWhere::All => true,
            IntegrateWhere::Boundary => false,
          };
          if keep {
            reg.clone()
          } else {
            let a = reg.approx()?;
            approx_integration(
              format!("IM_LEVEL_SET(cv {cv}, none)"),
              a.ref_convex(),
              vec![],
              vec![],
              true,
            )
          }
        };
        self.mim.set_integration_method((vec![cv], &pim))?;
        continue;
      }
      let ds = |xref: &[f64], t: &[f64]| -> Result<f64> {
        let pg = gt.at(&g, xref)?;
        let n = pg.k.nrows();
        let mut v = vec![0.0; n];
        for (j, tj) in t.iter().enumerate() {
          for (i, vi) in v.iter_mut().enumerate() {
            *vi += pg.k.col(j)[i] * tj;
          }
        }
        let len = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        Ok(len / pg.j.abs())
      };
      let pim = method_of_convex(cv, &cls, &ls_ids, &reg, self.where_, &sel, &ds, n)?;
      self.mim.set_integration_method((vec![cv], &pim))?;
    }
    Ok(())
  }
}

impl Deref for MeshImLevelSet {
  type Target = MeshIm;
  fn deref(&self) -> &MeshIm {
    &self.mim
  }
}

/// The reference polygon of a 2D convex (triangle or square), ccw.
fn reference_polygon(reg: &PIntegration) -> Result<Vec<[f64; 2]>> {
  let a = reg.approx()?;
  if a.dim() != 2 {
    return Err(Error::NotImplemented("mesh_im_level_set in 3D (step 3c)"));
  }
  match a.ref_convex().structure().nb_points() {
    3 => Ok(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
    4 => Ok(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
    _ => Err(Error::NotImplemented(
      "mesh_im_level_set on this reference convex",
    )),
  }
}

/// The rule of a 2D convex: its lattice of triangles (n per direction, in
/// reference coordinates) cut by the level sets into 6-node P2 cells
/// (`cut2d`), each with the symmetric 3-point rule; on the level set,
/// Gauss points on the curve's quadratic pieces.
#[allow(clippy::too_many_arguments)]
fn method_of_convex(
  cv: usize,
  cls: &[ConvexLevelSet],
  ls_ids: &[usize],
  reg: &PIntegration,
  where_: IntegrateWhere,
  sel: &dyn Fn(&[bool]) -> bool,
  ds: &ArcLength,
  n: usize,
) -> Result<PIntegration> {
  let a = reg.approx()?;
  let w = reference_polygon(reg)?;
  let err = std::cell::RefCell::new(None);
  let fs: Vec<Exact2> = cls.iter().map(|c| Exact2 { c, err: &err }).collect();
  let fr: Vec<&dyn CutFunction> = fs.iter().map(|f| f as &dyn CutFunction).collect();
  let tris = lattice(&w, n);
  let mut out = Cells::default();
  match where_ {
    IntegrateWhere::Inside | IntegrateWhere::Boundary => build(&tris, &fr, sel, 0, &mut out),
    IntegrateWhere::Outside => build(&tris, &fr, &|i: &[bool]| !sel(i), 0, &mut out),
    IntegrateWhere::All => {
      // every part, cut by every level set (GetFEM's INTEGRATE_ALL): one
      // build per pattern of insides
      for pattern in 0..1usize << fr.len() {
        let one = |i: &[bool]| {
          i.iter()
            .enumerate()
            .all(|(k, b)| *b == (pattern >> k & 1 == 1))
        };
        build(&tris, &fr, &one, 0, &mut out);
      }
    }
  }
  if let Some(e) = err.take() {
    return Err(Error::InvalidArgument(e));
  }
  let name = format!("IM_LEVEL_SET(cv {cv}, {:?}, n {n})", where_);
  let mut ptsref = vec![];
  // per point: the id of its level set
  let mut on = vec![];
  if where_ == IntegrateWhere::Boundary {
    let (x, wg) = gauss_legendre(NBOUNDARY);
    for (c, k) in &out.curve {
      for (s, wq) in x.iter().zip(&wg) {
        let (ptref, t) = (quad_at(c, *s), quad_d(c, *s));
        let tn = t[0].hypot(t[1]);
        if tn > 0.0 {
          ptsref.push((
            ptref.to_vec(),
            wq * tn * ds(&ptref, &[t[0] / tn, t[1] / tn])?,
          ));
          on.push(ls_ids[*k]);
        }
      }
    }
  } else {
    let xi = [
      [1.0 / 6.0, 1.0 / 6.0],
      [2.0 / 3.0, 1.0 / 6.0],
      [1.0 / 6.0, 2.0 / 3.0],
    ];
    for c in &out.cells {
      for ptref in xi {
        ptsref.push((c.map(ptref).to_vec(), c.jac(ptref).abs() / 6.0));
      }
    }
  }
  Ok(approx_integration_on(
    name,
    a.ref_convex(),
    ptsref,
    vec![],
    on,
    true,
  ))
}

/// The lattice of the reference triangle or square w: n per direction,
/// triangles (ccw); a square's cells split along a diagonal.
fn lattice(w: &[[f64; 2]], n: usize) -> Vec<[[f64; 2]; 3]> {
  let h = 1.0 / n as f64;
  let mut out = vec![];
  for i in 0..n {
    for j in 0..n {
      let (x, y) = (i as f64 * h, j as f64 * h);
      if w.len() == 4 {
        out.push([[x, y], [x + h, y], [x + h, y + h]]);
        out.push([[x, y], [x + h, y + h], [x, y + h]]);
        continue;
      }
      if i + j < n {
        out.push([[x, y], [x + h, y], [x, y + h]]);
      }
      if i + j + 1 < n {
        out.push([[x + h, y], [x + h, y + h], [x, y + h]]);
      }
    }
  }
  out
}

/// Gauss-Legendre points and weights on [0, 1].
fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
  let (mut x, mut w) = (vec![0.0; n], vec![0.0; n]);
  for i in 0..n {
    // Newton on the Legendre polynomial from the Chebyshev guess
    let mut t = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
    let mut dp = 1.0;
    for _ in 0..100 {
      let (mut p0, mut p1) = (1.0, t);
      for k in 2..=n {
        let p2 = ((2 * k - 1) as f64 * t * p1 - (k - 1) as f64 * p0) / k as f64;
        p0 = p1;
        p1 = p2;
      }
      dp = n as f64 * (t * p1 - p0) / (t * t - 1.0);
      let dt = p1 / dp;
      t -= dt;
      if dt.abs() < 1e-16 {
        break;
      }
    }
    x[i] = 0.5 * (1.0 - t);
    w[i] = 1.0 / ((1.0 - t * t) * dp * dp);
  }
  (x, w)
}

// ------------------------------------------------------------------ 3D

/// The method of a cut convex on mesh_level_set's sub-simplices (GetFEM
/// build_method_of_convex; 3D, and 2D at cracks): the simplex rule on each
/// selected sub-simplex of the cut (curved onto the level sets), selected
/// by the signs at its node farthest from each function (GetFEM: at its
/// centroid); inside a crack's level set: primary and secondary negative.
/// At a crack front vertex, IM_QUASI_POLAR(sing, the front vertices). On
/// the level sets: the faces of the selected sub-simplices on a level set
/// (from the cut) whose other side is not selected, each point with the
/// level set (GetFEM: the faces with all their nodes within 1e-7 of a
/// level set, for cracks of the set's boundary).
#[allow(clippy::too_many_arguments)]
fn method_of_sub_mesh(
  cv: usize,
  cut: &CutConvex,
  ls_ids: &[usize],
  cls: &[ConvexLevelSet],
  cls2: &[Option<ConvexLevelSet>],
  simplex: &PIntegration,
  sing: Option<&PIntegration>,
  cell_tolerance: Option<f64>,
  reg: &PIntegration,
  where_: IntegrateWhere,
  sel: &dyn Fn(&[bool]) -> bool,
  elem: &ElemGeom,
) -> Result<PIntegration> {
  let sm = &cut.mesh;
  let name = format!("IM_LEVEL_SET(cv {cv}, {:?}, {})", where_, simplex.name());
  let ids: Vec<usize> = sm.convex_index().collect();
  let nb = ids.iter().max().map_or(0, |m| m + 1);
  // the sign at the node farthest from the zero set
  let far = |l: &ConvexLevelSet, i: usize| -> Result<f64> {
    let mut far = 0.0f64;
    for &p in sm.ind_points_of_convex(i) {
      let v = l.val(sm.point(p))?;
      if v.abs() > far.abs() {
        far = v;
      }
    }
    Ok(far)
  };
  // per sub-simplex: inside each level set, selected
  let mut inside = vec![vec![]; nb];
  let mut selected = vec![false; nb];
  for &i in &ids {
    for (l, l2) in cls.iter().zip(cls2) {
      let inn = far(l, i)? < 0.0
        && match l2 {
          Some(l2) => far(l2, i)? < 0.0,
          None => true,
        };
      inside[i].push(inn);
    }
    let s = sel(&inside[i]);
    selected[i] = match where_ {
      IntegrateWhere::Outside => !s,
      IntegrateWhere::All => true,
      _ => s,
    };
  }
  let mut ptsref = vec![];
  // per point of ptsref: the id of its level set, 0 none
  let mut on = vec![];
  let cvr = reg.approx()?.ref_convex().clone();
  let mut faces = vec![vec![]; cvr.structure().nb_faces()];
  for &i in ids.iter().filter(|&&i| selected[i]) {
    let (gt, gref) = (sm.trans_of_convex(i), sm.points_of_convex(i));
    let n = gt.dim();
    // the mesh points at the vertices of the sub-simplex
    let vertex = |j: usize| -> Option<usize> {
      let vj: Vec<f64> = (0..n).map(|k| if j == k + 1 { 1.0 } else { 0.0 }).collect();
      gt.convex_ref()
        .points()
        .iter()
        .position(|p| p.iter().zip(&vj).all(|(a, b)| (a - b).abs() < 1e-12))
        .map(|node| sm.ind_points_of_convex(i)[node])
    };
    // the rule by the cell's size s (longest edge, element = 1): the
    // lowest degree k with s^(k+1) <= tau, at most simplex's points; not on
    // boundaries (curved faces need the full rule, and they are cheap)
    let mut rule = simplex.clone();
    if let Some(tau) = cell_tolerance.filter(|_| where_ != IntegrateWhere::Boundary) {
      let verts: Vec<&[f64]> = (0..=n).filter_map(vertex).map(|p| sm.point(p)).collect();
      let mut size = 0.0f64;
      for a in 0..verts.len() {
        for b in a + 1..verts.len() {
          size = size.max(vect_norm2(
            &verts[a]
              .iter()
              .zip(verts[b])
              .map(|(x, y)| x - y)
              .collect::<Vec<_>>(),
          ));
        }
      }
      // (cells of size >= 1: simplex)
      let k = if size >= 1.0 {
        usize::MAX
      } else if size <= 0.0 {
        1
      } else {
        ((tau.ln() / size.ln()).ceil() as usize)
          .saturating_sub(1)
          .max(1)
      };
      let gt1 = getfem_geom::geotrans::pk(n, 1)?;
      // (no rule of that degree: simplex)
      if k < usize::MAX
        && let Ok(low) = crate::integration::classical_approx_im(&gt1, k)
        && low.approx()?.nb_points() < simplex.approx()?.nb_points()
      {
        rule = low;
      }
    }
    // the vertices of the sub-simplex on a crack front (from the cut): the
    // quasi-polar rule (boundary: the graded faces)
    let mut front: Vec<usize> = vec![];
    if let Some(sp) = sing {
      // per crack, its front vertices; the crack with the most (the first
      // if equal): where the fronts of two cracks meet, the other's are
      // ignored (GetFEM: an error)
      for k in (0..cls.len()).filter(|&k| cls2[k].is_some()) {
        let t: Vec<usize> = (0..=n)
          .filter(|&j| vertex(j).is_some_and(|p| cut.on_front(p, k)))
          .collect();
        if t.len() > front.len() {
          front = t;
        }
      }
      if front.len() >= n {
        return Err(Error::InvalidArgument(format!(
          "mesh_im_level_set: {} vertices of a sub-simplex of convex {cv} on one crack front",
          front.len()
        )));
      }
      if where_ != IntegrateWhere::Boundary && !front.is_empty() {
        let ips: Vec<String> = front.iter().map(|t| t.to_string()).collect();
        rule = crate::integration::integ_descriptor(&format!(
          "IM_QUASI_POLAR({},{})",
          sp.name(),
          ips.join(",")
        ))?;
      }
    }
    let a = rule.approx()?;
    if where_ != IntegrateWhere::Boundary {
      for j in 0..a.nb_points_on_convex() {
        let xref = gt.transform(a.point(j), &gref)?;
        ptsref.push((xref, a.coeff(j) * jacobian(gt, &gref, a.point(j)).abs()));
        on.push(0);
      }
      // the sub-simplex's faces on a face of the element: points of that
      // face (GetFEM)
      for f in 0..gt.structure().nb_faces() {
        let on = sm.ind_points_of_face_of_convex(i, f);
        let nb = on.len() as f64;
        let c: Vec<f64> = (0..gt.dim())
          .map(|k| on.iter().map(|&p| sm.point(p)[k]).sum::<f64>() / nb)
          .collect();
        if cvr.is_in(&c)? < -1e-7 {
          continue;
        }
        let Some(ff) = (0..cvr.structure().nb_faces())
          .find(|&ff| cvr.is_in_face(ff, &c).is_ok_and(|v| v.abs() < 2e-6))
        else {
          continue;
        };
        let nf = &gt.normals()[f];
        for j in 0..a.nb_points_on_face(f) {
          let Ok(pg) = gt.at(&gref, a.point_on_face(f, j)) else {
            continue;
          };
          if pg.j.abs() <= 1e-11 {
            continue;
          }
          let nup = vect_norm2(&mat_vec(&pg.b, nf));
          faces[ff].push((pg.x, a.coeff_on_face(f, j) * pg.j.abs() * nup));
        }
      }
      continue;
    }
    for f in 0..gt.structure().nb_faces() {
      // on a level set, the other side not selected
      let Some(l) = cut.level_set_of_face(i, f) else {
        continue;
      };
      let mut other = inside[i].clone();
      other[l] = !other[l];
      if sel(&other) {
        continue;
      }
      // dS / dS_ref = |J| |B n| (sub-simplex), times |B n| (element) for
      // the face's unit normal n in the element's reference coordinates
      let nf = &gt.normals()[f];
      let on_face: Vec<usize> = (0..=gt.dim())
        .filter(|&v| {
          let x: Vec<f64> = (0..gt.dim())
            .map(|k| if v == k + 1 { 1.0 } else { 0.0 })
            .collect();
          gt.convex_ref()
            .is_in_face(f, &x)
            .is_ok_and(|d| d.abs() < 1e-12)
        })
        .collect();
      let graded: Vec<usize> = on_face
        .iter()
        .copied()
        .filter(|v| front.contains(v))
        .collect();
      for j in 0..a.nb_points_on_face(f) {
        let (xf, wf) = if graded.is_empty() {
          (a.point_on_face(f, j).to_vec(), 1.0)
        } else {
          graded_point(a.point_on_face(f, j), &on_face, &graded)
        };
        // GetFEM skips |J| < 1e-11
        let Ok(pg) = gt.at(&gref, &xf) else {
          continue;
        };
        if pg.j.abs() <= 1e-11 {
          continue;
        }
        let mut m = mat_vec(&pg.b, nf);
        let nup = vect_norm2(&m);
        m.iter_mut().for_each(|v| *v /= nup);
        let be = elem(&pg.x)?;
        on.push(ls_ids[l]);
        ptsref.push((
          pg.x,
          a.coeff_on_face(f, j) * wf * pg.j.abs() * nup * vect_norm2(&mat_vec(&be, &m)),
        ));
      }
    }
  }
  Ok(approx_integration_on(name, &cvr, ptsref, faces, on, true))
}

/// A point of a face of the reference simplex graded toward its vertices
/// on a crack front, and the Jacobian: w, the sum of the barycentrics of
/// the face's other vertices, mapped to w^2 (an r^(-1/2) integrand made
/// smooth); J = 2 w^m (1 + w)^(k-1), k the front vertices, m the others.
fn graded_point(x: &[f64], on_face: &[usize], front: &[usize]) -> (Vec<f64>, f64) {
  let n = x.len();
  let mut lam: Vec<f64> = std::iter::once(1.0 - x.iter().sum::<f64>())
    .chain(x.iter().copied())
    .collect();
  let others: Vec<usize> = on_face
    .iter()
    .copied()
    .filter(|v| !front.contains(v))
    .collect();
  let w: f64 = others.iter().map(|&v| lam[v]).sum();
  for &v in &others {
    lam[v] *= w;
  }
  for &v in front {
    lam[v] *= 1.0 + w;
  }
  let (k, m) = (front.len() as i32, others.len() as i32);
  (
    (1..=n).map(|v| lam[v]).collect(),
    2.0 * w.powi(m) * (1.0 + w).powi(k - 1),
  )
}

/// B of the element at a reference point (K^-T).
type ElemGeom<'a> = dyn Fn(&[f64]) -> Result<Mat> + 'a;

fn mat_vec(b: &Mat, v: &[f64]) -> Vec<f64> {
  let mut r = vec![0.0; b.nrows()];
  for (j, vj) in v.iter().enumerate() {
    for (ri, bij) in r.iter_mut().zip(b.col(j)) {
      *ri += bij * vj;
    }
  }
  r
}

/// GetFEM's set expression of the level sets (letters a, b, ... for the
/// first, second, ...; "+" union, "-" difference, "*" intersection, "!"
/// complement, parentheses; the right operand is the rest of the
/// expression): is a point with these insides selected?
fn csg_eval(expr: &str, inside: &[bool]) -> Result<bool> {
  fn term(s: &[u8], i: &mut usize, inside: &[bool]) -> Result<bool> {
    let bad = |i: usize| Error::InvalidArgument(format!("set expression: parse error at {i}"));
    let r = match s.get(*i) {
      Some(b'(') => {
        *i += 1;
        let r = term(s, i, inside)?;
        if s.get(*i) != Some(&b')') {
          return Err(bad(*i));
        }
        *i += 1;
        r
      }
      Some(b'!') => {
        *i += 1;
        !term(s, i, inside)?
      }
      Some(c @ b'a'..=b'z') => {
        let k = (c - b'a') as usize;
        *i += 1;
        *inside.get(k).ok_or_else(|| {
          Error::InvalidArgument(format!("set expression: no level set '{}'", *c as char))
        })?
      }
      _ => return Err(bad(*i)),
    };
    if let Some(&op @ (b'+' | b'-' | b'*')) = s.get(*i) {
      *i += 1;
      let b = term(s, i, inside)?;
      return Ok(match op {
        b'+' => r || b,
        b'-' => r && !b,
        _ => r && b,
      });
    }
    Ok(r)
  }
  let s = expr.as_bytes();
  let mut i = 0;
  let r = term(s, &mut i, inside)?;
  if i != s.len() {
    return Err(Error::InvalidArgument(format!(
      "set expression: parse error at {i}"
    )));
  }
  Ok(r)
}

/// A level set on a 2D convex with its exact derivatives (reference
/// coordinates); an evaluation error is kept in err (reported after the
/// cut).
struct Exact2<'a> {
  c: &'a ConvexLevelSet,
  err: &'a std::cell::RefCell<Option<String>>,
}

impl Exact2<'_> {
  fn keep<T>(&self, r: Result<T>, zero: T) -> T {
    r.unwrap_or_else(|e| {
      *self.err.borrow_mut() = Some(e.to_string());
      zero
    })
  }
}

impl CutFunction for Exact2<'_> {
  fn val(&self, x: [f64; 2]) -> f64 {
    self.keep(self.c.val(&x), 0.0)
  }
  fn grad(&self, x: [f64; 2]) -> [f64; 2] {
    let g = self.keep(self.c.grad(&x), vec![0.0; 2]);
    [g[0], g[1]]
  }
  fn hess(&self, x: [f64; 2]) -> [[f64; 2]; 2] {
    // column-major
    let h = self.keep(self.c.hess(&x), vec![0.0; 4]);
    [[h[0], h[2]], [h[1], h[3]]]
  }
}
