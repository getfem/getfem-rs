//! Elements of global functions (port of getfem_fem_global_function and
//! getfem_mesh_fem_global_function): on each convex, the functions whose
//! bounding box meets the convex (and, with an integration method, whose
//! support contains one of its points). Their DOFs are global: the
//! indices of the functions.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock};

use getfem_geom::convex_ref::PRef;
use getfem_geom::mesh::Mesh;
use getfem_geom::search::RTree;
use getfem_linalg::tensor::Tensor;

use crate::fem::{DofDesc, Fem, FemCtx, PFem, RealValues};
use crate::global_function::{EvalPoint, GlobalFunction};
use crate::integration::PIntegration;
use crate::mesh_fem::MeshFem;
use crate::mesh_im::MeshIm;
use crate::{Error, Result};

/// The element of one convex.
pub struct GlobalFunctionFem {
  name: String,
  cvr: PRef,
  dofs: Vec<DofDesc>,
  nodes: Vec<Vec<f64>>,
  faces: Vec<Vec<usize>>,
  /// the indices of its functions (GetFEM index_of_global_dof)
  global: Vec<usize>,
  /// its convex
  cv: usize,
  funcs: Arc<Vec<GlobalFunction>>,
  /// its mesh, whose geometry the precomputed values are valid for
  mesh: Mesh,
  /// the values at the points of the integration methods met (GetFEM
  /// precomps)
  precomps: Mutex<Vec<Arc<Precomp>>>,
}

/// The values (0), gradients (1) and Hessians (2) of the functions at the
/// points of an integration method, each computed when first asked
/// (GetFEM precomp_data)
struct Precomp {
  pim: PIntegration,
  /// the geometry version of the mesh, the newest version of the
  /// functions
  version: (u64, u64),
  pts: Vec<[OnceLock<Vec<f64>>; 3]>,
}

impl std::fmt::Debug for GlobalFunctionFem {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}", self.name)
  }
}

impl Fem for GlobalFunctionFem {
  fn name(&self) -> &str {
    &self.name
  }
  fn ref_convex(&self) -> &PRef {
    &self.cvr
  }
  fn target_dim(&self) -> usize {
    1
  }
  fn nb_dof(&self) -> usize {
    self.global.len()
  }
  fn dof_types(&self) -> &[DofDesc] {
    &self.dofs
  }
  fn node(&self, i: usize) -> &[f64] {
    &self.nodes[i]
  }
  fn faces_of_dof(&self, i: usize) -> &[usize] {
    &self.faces[i]
  }
  fn is_equivalent(&self) -> bool {
    true
  }
  fn is_lagrange(&self) -> bool {
    false
  }
  fn is_polynomial(&self) -> bool {
    false
  }
  fn estimated_degree(&self) -> usize {
    5
  }
  fn base_value(&self, _: &[f64]) -> Result<Tensor<f64>> {
    Err(self.real_only())
  }
  fn grad_base_value(&self, _: &[f64]) -> Result<Tensor<f64>> {
    Err(self.real_only())
  }
  fn hess_base_value(&self, _: &[f64]) -> Result<Tensor<f64>> {
    Err(self.real_only())
  }
  fn global_dofs(&self) -> Option<&[usize]> {
    Some(&self.global)
  }
  fn real_element_defined(&self) -> bool {
    true
  }
  fn real_values(&self, c: &FemCtx, der: usize) -> Result<RealValues> {
    // at a point of an integration method, from the precomputed values
    // (GetFEM: c.have_pfp() && c.ii() != -1); not for the methods of one
    // element (cut cells), nor on the sides of a level set
    let Some((pim, ip)) = c
      .pim
      .filter(|(pim, _)| c.xfem_side == 0 && !pim.is_transient())
    else {
      let at = |d| {
        if d <= der {
          self.real_base_value(c, d)
        } else {
          vec![]
        }
      };
      return Ok((at(0), at(1), at(2)));
    };
    let pc = self.precomp(pim)?;
    let at = |d: usize| match d <= der {
      true => pc.pts[ip][d]
        .get_or_init(|| self.real_base_value(c, d))
        .clone(),
      false => vec![],
    };
    Ok((at(0), at(1), at(2)))
  }
}

impl GlobalFunctionFem {
  /// the values (d = 0), gradients (1) or Hessians (2) of its functions at
  /// the point of c: [i], [i + nd r], [i + nd (r + n s)] (GetFEM
  /// real_base_value, real_grad_base_value, real_hess_base_value)
  fn real_base_value(&self, c: &FemCtx, d: usize) -> Vec<f64> {
    let (nd, n) = (self.global.len(), c.x.len());
    let p = EvalPoint {
      x: c.x,
      element: Some((self.cv, c)),
    };
    let fs = self.global.iter().map(|&i| &self.funcs[i]);
    if d == 0 {
      return fs.map(|f| f.val(p)).collect();
    }
    let mut v = vec![0.0; nd * n.pow(d as u32)];
    for (i, f) in fs.enumerate() {
      let fv = if d == 1 { f.grad(p) } else { f.hess(p) };
      for (r, x) in fv.into_iter().enumerate() {
        v[i + nd * r] = x;
      }
    }
    v
  }

  /// the precomputed values at the points of pim, for the current geometry
  /// of the mesh and context of the functions (GetFEM: cleared by
  /// update_from_context)
  fn precomp(&self, pim: &PIntegration) -> Result<Arc<Precomp>> {
    let fv = self.global.iter().map(|&i| self.funcs[i].version());
    let version = (self.mesh.geometry_version(), fv.max().unwrap_or(0));
    let mut pcs = self.precomps.lock().unwrap_or_else(|e| e.into_inner());
    pcs.retain(|pc| pc.version == version);
    if let Some(pc) = pcs.iter().find(|pc| Arc::ptr_eq(&pc.pim, pim)) {
      return Ok(pc.clone());
    }
    let pc = Arc::new(Precomp {
      pim: pim.clone(),
      version,
      pts: (0..pim.approx()?.points().len())
        .map(|_| Default::default())
        .collect(),
    });
    pcs.push(pc.clone());
    Ok(pc)
  }

  fn real_only(&self) -> Error {
    Error::InvalidArgument(format!("{}: real element only", self.name))
  }
}

/// The box of the points g (columns), enlarged by 20% for nonlinear
/// transformations (bgeot::bounding_box).
fn convex_box(g: &Tensor<f64>, linear: bool) -> (Vec<f64>, Vec<f64>) {
  let n = g.nrows();
  let (mut lo, mut hi) = (g.col(0).to_vec(), g.col(0).to_vec());
  for j in 1..g.ncols() {
    for (k, &v) in g.col(j).iter().enumerate() {
      lo[k] = lo[k].min(v);
      hi[k] = hi[k].max(v);
    }
  }
  if !linear {
    for k in 0..n {
      let e = (hi[k] - lo[k]) * 0.2;
      lo[k] -= e;
      hi[k] += e;
    }
  }
  (lo, hi)
}

impl MeshFem {
  /// The space of the global functions funcs on mesh (GetFEM
  /// mesh_fem_global_function::set_functions): on each convex, the
  /// functions whose bounding box meets it, and with mim, whose support
  /// contains one of its integration points. Built from the current mesh
  /// and mim: build it again after they change.
  pub fn from_global_functions(
    mesh: &Mesh,
    funcs: &[GlobalFunction],
    mim: Option<&MeshIm>,
  ) -> Result<MeshFem> {
    let Some(f0) = funcs.first() else {
      return Err(Error::InvalidArgument(
        "Empty list of base functions.".into(),
      ));
    };
    let dim = f0.dim();
    if funcs.iter().any(|f| f.dim() != dim) {
      return Err(Error::InvalidArgument(
        "Incompatible function space dimensions.".into(),
      ));
    }
    let ims = match mim {
      Some(mim) => {
        if !mim.linked_mesh().same(mesh) {
          return Err(Error::InvalidArgument(
            "The provided mesh_im has to be linked to the same mesh as this mesh_fem.".into(),
          ));
        }
        Some(mim.ims()?)
      }
      None => None,
    };
    // equal boxes share one entry, whose functions are listed in order:
    // GetFEM's rtree::add_box merges corners closer than max(1e-13,
    // max|x| 1e4 eps) (bgeot::node_tab)
    let mut tree = RTree::new(dim, 1e-13);
    let mut boxes: Vec<(Vec<f64>, Vec<f64>, Vec<usize>)> = vec![];
    // the boxes by their first min coordinate (ordered bits)
    let key = |x: f64| {
      let b = x.to_bits() as i64;
      if b < 0 { b ^ i64::MAX } else { b }
    };
    let mut by_x: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let dist = |a: &[f64], b: &[f64]| {
      a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
    };
    let mut rmax = 0.0f64;
    for (i, f) in funcs.iter().enumerate() {
      let (lo, hi) = f.bounding_box();
      rmax = rmax.max(norm(&lo)).max(norm(&hi));
      let tol = (rmax * 1e4 * f64::EPSILON).max(1e-13);
      let found = by_x
        .range(key(lo[0] - tol)..=key(lo[0] + tol))
        .flat_map(|(_, v)| v)
        .copied()
        .find(|&b| dist(&boxes[b].0, &lo) < tol && dist(&boxes[b].1, &hi) < tol);
      match found {
        Some(b) => boxes[b].2.push(i),
        None => {
          tree.add_box(&lo, &hi, boxes.len())?;
          by_x.entry(key(lo[0])).or_default().push(boxes.len());
          boxes.push((lo, hi, vec![i]));
        }
      }
    }
    let m = mesh.snapshot();
    let funcs = Arc::new(funcs.to_vec());
    let mf = MeshFem::new(mesh, 1);
    let id = crate::mesh_fem::new_stamp();
    for cv in m.convex_index() {
      let (gt, g) = (m.trans_of_convex(cv), m.points_of_convex(cv));
      let cvr = gt.convex_ref().basic();
      if cvr.dim() != dim {
        return Err(Error::InvalidArgument(
          "Convexes of different dimension: to be done".into(),
        ));
      }
      let (lo, hi) = convex_box(&g, gt.is_linear());
      // the boxes in the order of their ids (GetFEM pbox_set)
      let mut global = vec![];
      let pts = match &ims {
        Some(l) => {
          let a = l.int_method_of_element(cv)?.approx()?;
          (0..a.nb_points())
            .map(|k| gt.transform(a.point(k), &g))
            .collect::<std::result::Result<Vec<Vec<f64>>, _>>()?
        }
        None => vec![],
      };
      for b in tree.find_intersecting_boxes(&lo, &hi)? {
        for &i in &boxes[b].2 {
          if ims.is_none() || pts.iter().any(|x| funcs[i].is_in_support(x)) {
            global.push(i);
          }
        }
      }
      let nd = global.len();
      let nf = cvr.structure().nb_faces();
      let pf: PFem = Arc::new(GlobalFunctionFem {
        name: format!("GLOBAL_FEM#{id}({cv})"),
        dofs: vec![DofDesc::global(dim); nd],
        nodes: vec![vec![1.0 / 20.0; dim]; nd],
        faces: vec![(0..nf).collect(); nd],
        cvr,
        global,
        cv,
        funcs: funcs.clone(),
        mesh: mesh.clone(),
        precomps: Mutex::new(vec![]),
      });
      mf.set_finite_element(([cv], &pf))?;
    }
    Ok(mf)
  }
}

/// The boundary conditions of a uniform B-spline basis at one end of a
/// direction (GetFEM bspline_boundary).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BsplineBoundary {
  Free,
  /// the functions wrapped around (on both ends)
  Periodic,
  /// the functions mirrored and summed
  Symmetry,
}

/// One direction of a uniform basis: per function min, max, shift, type
/// (GetFEM params_for_uniform_1d_bspline_basis_functions).
type Params1d = Vec<(f64, f64, f64, usize)>;

fn params_1d(
  x0: f64,
  x1: f64,
  n: usize,
  order: usize,
  low: BsplineBoundary,
  high: BsplineBoundary,
) -> Result<Params1d> {
  use BsplineBoundary::*;
  if (low == Periodic || high == Periodic) && low != high {
    return Err(Error::InvalidArgument(
      "Periodic BC has to be assigned to both matching sides".into(),
    ));
  }
  if n + 1 < order {
    return Err(Error::InvalidArgument(format!(
      "uniform B-splines of order {order}: at least {} intervals",
      order - 1
    )));
  }
  let dx = (x1 - x0) / n as f64;
  let n_low = match low {
    Periodic => 0,
    Symmetry => order / 2,
    Free => order - 1,
  };
  let n_high = match high {
    Periodic | Free => order - 1,
    Symmetry => order / 2,
  };
  let n_mid = n + 1 - order;
  let nf = n_low + n_mid + n_high;
  let mut out = Vec::with_capacity(nf);
  for i in 0..nf {
    let (xmin, xmax, t);
    let mut shift = 0.0;
    if low == Free && i < n_low {
      t = i + 1;
      xmin = x0;
      xmax = xmin + t as f64 * dx;
    } else if high == Free && i >= n_low + n_mid {
      t = nf - i;
      xmin = x1;
      // xmax < xmin
      xmax = xmin - t as f64 * dx;
    } else if low == Symmetry && i < n_low {
      t = order;
      xmin = x0 - (n_low - i) as f64 * dx;
      xmax = xmin + t as f64 * dx;
      shift = -(xmin + xmax - 2.0 * x0);
    } else if high == Symmetry && i >= n_low + n_mid {
      t = order;
      xmin = x0 + (i - n_low) as f64 * dx;
      xmax = xmin + t as f64 * dx;
      shift = 2.0 * x1 - xmin - xmax;
    } else {
      t = order;
      xmin = x0 + (i - n_low) as f64 * dx;
      xmax = xmin + t as f64 * dx;
    }
    if low == Periodic && xmax > x1 {
      shift = -(x1 - x0);
    }
    out.push((xmin, xmax, shift, t));
  }
  Ok(out)
}

impl MeshFem {
  /// The uniform B-spline basis of order 3 to 5 on n[k] intervals in
  /// each direction k of the bounding box of mesh, with the boundary
  /// conditions low[k], high[k] (GetFEM
  /// define_uniform_bspline_basis_functions_for_mesh_fem, 1D to 3D).
  /// Periodic and symmetric functions are sums of the shifted B-splines.
  pub fn uniform_bspline(
    mesh: &Mesh,
    n: &[usize],
    order: usize,
    low: &[BsplineBoundary],
    high: &[BsplineBoundary],
    mim: Option<&MeshIm>,
  ) -> Result<MeshFem> {
    let dim = mesh.dim();
    if !(1..=3).contains(&dim) || n.len() != dim || low.len() != dim || high.len() != dim {
      return Err(Error::InvalidArgument(format!(
        "uniform B-splines on a mesh of dimension {dim}: {dim} interval counts and boundary conditions expected"
      )));
    }
    let Some((p0, p1)) = mesh.snapshot().bounding_box() else {
      return Err(Error::InvalidArgument(
        "uniform B-splines on an empty mesh".into(),
      ));
    };
    let ps = (0..dim)
      .map(|k| params_1d(p0[k], p1[k], n[k], order, low[k], high[k]))
      .collect::<Result<Vec<_>>>()?;
    // all index combinations, the last direction fastest
    let mut idx = vec![vec![]];
    for p in &ps {
      idx = idx
        .into_iter()
        .flat_map(|v: Vec<usize>| {
          (0..p.len()).map(move |i| {
            let mut w = v.clone();
            w.push(i);
            w
          })
        })
        .collect();
    }
    let mut funcs = Vec::with_capacity(idx.len());
    for ix in idx {
      let par: Vec<_> = ix.iter().zip(&ps).map(|(&i, p)| p[i]).collect();
      let types: Vec<usize> = par.iter().map(|p| p.3).collect();
      let shifted: Vec<bool> = par.iter().map(|p| p.2.abs() >= 1e-10).collect();
      // the unshifted function, then each nonempty subset of the shifted
      // directions shifted (GetFEM's order: x, y, z, xy, yz, zx, xyz)
      let mut subsets: Vec<Vec<usize>> = vec![vec![]];
      let s: Vec<usize> = (0..dim).filter(|&k| shifted[k]).collect();
      subsets.extend(s.iter().map(|&k| vec![k]));
      for (a, b) in [(0, 1), (1, 2), (2, 0)] {
        if a < dim && b < dim && shifted[a] && shifted[b] {
          subsets.push(vec![a, b]);
        }
      }
      if s.len() == 3 {
        subsets.push(vec![0, 1, 2]);
      }
      let mut sum = vec![];
      for sub in &subsets {
        let sh = |k: usize| if sub.contains(&k) { par[k].2 } else { 0.0 };
        let lo: Vec<f64> = (0..dim).map(|k| par[k].0 + sh(k)).collect();
        let hi: Vec<f64> = (0..dim).map(|k| par[k].1 + sh(k)).collect();
        sum.push(GlobalFunction::bspline(&lo, &hi, order, &types)?);
      }
      funcs.push(if sum.len() == 1 {
        sum.pop().unwrap()
      } else {
        GlobalFunction::sum(&sum)?
      });
    }
    MeshFem::from_global_functions(mesh, &funcs, mim)
  }
}
