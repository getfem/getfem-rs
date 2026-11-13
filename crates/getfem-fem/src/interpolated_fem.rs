//! Interpolated elements (port of getfem_interpolated_fem): the functions
//! of a space on one mesh, evaluated on the elements of another mesh.
//! Their DOFs are global: the DOFs of the source space met by the points
//! of the integration method on each element.

use std::sync::Arc;

use getfem_geom::convex_ref::PRef;
use getfem_geom::search::RTree;
use getfem_linalg::tensor::Tensor;

use crate::fem::{DofDesc, Fem, FemCtx, PFem, RealValues, real_base_values};
use crate::mesh_fem::{DofNumbering, MeshFem};
use crate::mesh_im::MeshIm;
use crate::{Error, Result};

/// The source space with its point location.
struct Source {
  num: Arc<DofNumbering>,
  tree: RTree,
}

impl Source {
  fn new(num: Arc<DofNumbering>) -> Result<Self> {
    let m = num.linked_mesh();
    let mut tree = RTree::new(m.dim(), 1e-13);
    for cv in num.convex_index() {
      let g = m.points_of_convex(cv);
      let (mut lo, mut hi) = (vec![f64::MAX; m.dim()], vec![f64::MIN; m.dim()]);
      for j in 0..g.ncols() {
        for (k, &v) in g.col(j).iter().enumerate() {
          lo[k] = lo[k].min(v);
          hi[k] = hi[k].max(v);
        }
      }
      if !m.trans_of_convex(cv).is_linear() {
        // curved elements may bulge past their nodes
        let pad = 0.1 * getfem_geom::geotrans::element_extent(&g);
        lo.iter_mut().for_each(|v| *v -= pad);
        hi.iter_mut().for_each(|v| *v += pad);
      }
      tree.add_box(&lo, &hi, cv)?;
    }
    Ok(Source { num, tree })
  }

  /// A source convex containing x (hint first, as GetFEM's find_a_point
  /// with the previous convex; then the lowest), and the reference point.
  fn locate(&self, x: &[f64], hint: Option<usize>) -> Result<Option<Location>> {
    let m = self.num.linked_mesh();
    if let Some(cv) = hint {
      let inv = m
        .trans_of_convex(cv)
        .invert(&m.points_of_convex(cv), x, 1e-12, false)?;
      if inv.converged && inv.inside {
        return Ok(Some((cv, inv.xref)));
      }
    }
    let mut cvs = self.tree.find_boxes_at_point(x)?;
    cvs.sort_unstable();
    for cv in cvs {
      let inv = m
        .trans_of_convex(cv)
        .invert(&m.points_of_convex(cv), x, 1e-12, false)?;
      if inv.converged && inv.inside {
        return Ok(Some((cv, inv.xref)));
      }
    }
    Ok(None)
  }
}

/// A source convex and the reference point in it.
type Location = (usize, Vec<f64>);

/// The interpolated element of one convex of the target mesh.
pub struct InterpolatedFem {
  name: String,
  cvr: PRef,
  dofs: Vec<DofDesc>,
  nodes: Vec<Vec<f64>>,
  faces: Vec<Vec<usize>>,
  /// the source basic DOFs of the element, sorted
  global: Vec<usize>,
  target: usize,
  src: Arc<Source>,
  /// the integration points of the element (reference coordinates) and
  /// where they are in the source (GetFEM gausspt)
  located: Vec<(Vec<f64>, Option<Location>)>,
}

impl std::fmt::Debug for InterpolatedFem {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}", self.name)
  }
}

impl Fem for InterpolatedFem {
  fn name(&self) -> &str {
    &self.name
  }
  fn ref_convex(&self) -> &PRef {
    &self.cvr
  }
  fn target_dim(&self) -> usize {
    self.target
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
    Err(Error::InvalidArgument(format!(
      "{}: real element only",
      self.name
    )))
  }
  fn grad_base_value(&self, _: &[f64]) -> Result<Tensor<f64>> {
    Err(Error::InvalidArgument(format!(
      "{}: real element only",
      self.name
    )))
  }
  fn hess_base_value(&self, _: &[f64]) -> Result<Tensor<f64>> {
    Err(Error::InvalidArgument(format!(
      "{}: no Hessians of interpolated elements",
      self.name
    )))
  }
  fn global_dofs(&self) -> Option<&[usize]> {
    Some(&self.global)
  }
  fn real_element_defined(&self) -> bool {
    true
  }
  fn real_values(&self, c: &FemCtx, der: usize) -> Result<RealValues> {
    if der > 1 {
      return Err(Error::InvalidArgument(format!(
        "{}: no Hessians of interpolated elements",
        self.name
      )));
    }
    let (val, grad) = self.values_at(c.x, c.xref)?;
    Ok((val, grad, vec![]))
  }
}

impl InterpolatedFem {
  /// val[i + nd*a], grad[i + nd*(a + t*r)]; zero outside the source mesh.
  /// At an integration point of the element (xref), its stored location.
  fn values_at(&self, x: &[f64], xref: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
    let (nd, t, n) = (self.global.len(), self.target, x.len());
    let (mut val, mut grad) = (vec![0.0; nd * t], vec![0.0; nd * t * n]);
    let found = match self
      .located
      .iter()
      .find(|(ptref, _)| ptref.as_slice() == xref)
    {
      Some((_, l)) => l.clone(),
      None => self.src.locate(x, None)?,
    };
    let Some((cv, xsrcref)) = found else {
      return Ok((val, grad));
    };
    let num = &self.src.num;
    let m = num.linked_mesh();
    let pf = num.fem_of_element(cv).unwrap();
    let (v, g, _) = real_base_values(
      pf,
      m.trans_of_convex(cv),
      &m.points_of_convex(cv),
      &xsrcref,
      1,
    )?;
    let (ns, q) = (pf.nb_dof(), num.qdim());
    let dofs = num.ind_basic_dof_of_element(cv);
    for i in 0..ns {
      for a in 0..q {
        let Ok(l) = self.global.binary_search(&dofs[i * q + a]) else {
          continue;
        };
        let c = if t == 1 { 0 } else { a };
        val[l + nd * c] = v[i];
        for r in 0..n {
          grad[l + nd * (c + t * r)] = g[i + ns * r];
        }
      }
    }
    Ok((val, grad))
  }
}

/// The space on the mesh of mim whose functions are those of source,
/// evaluated at the points of mim (GetFEM new_interpolated_fem, set on
/// all convexes of mim). The space is built from the current source and
/// mim: build it again after they change.
pub fn new_interpolated_fem(source: &MeshFem, mim: &MeshIm) -> Result<MeshFem> {
  let num = source.numbering()?;
  // as GetFEM (interpolated_fem: !mf.is_reduced())
  if source.is_reduced() {
    return Err(Error::InvalidArgument(
      "interpolated element of a reduced (or partial) space".into(),
    ));
  }
  let q = num.qdim();
  let ims = mim.ims()?;
  let m = ims.linked_mesh();
  if q != 1 && q != m.dim() {
    return Err(Error::InvalidArgument(format!(
      "interpolated element: source qdim {q}, 1 or the mesh dimension expected"
    )));
  }
  if num
    .convex_index()
    .any(|cv| num.fem_of_element(cv).unwrap().target_dim() != 1)
  {
    return Err(Error::InvalidArgument(
      "interpolated element of vector elements".into(),
    ));
  }
  let src = Arc::new(Source::new(num.clone())?);
  let target = MeshFem::new(&mim.linked_mesh(), q);
  let id = crate::mesh_fem::new_stamp();
  let mut hint = None;
  for cv in ims.convex_index() {
    let a = ims.int_method_of_element(cv)?.approx()?;
    let (gt, g) = (m.trans_of_convex(cv), m.points_of_convex(cv));
    let mut global = std::collections::BTreeSet::new();
    let mut located = Vec::with_capacity(a.nb_points());
    for k in 0..a.nb_points() {
      let l = src.locate(&gt.transform(a.point(k), &g)?, hint)?;
      if let Some((c, _)) = &l {
        global.extend(num.ind_basic_dof_of_element(*c).iter().copied());
        hint = Some(*c);
      }
      located.push((a.point(k).to_vec(), l));
    }
    let nd = global.len();
    let cvr = gt.convex_ref().basic();
    let nf = cvr.structure().nb_faces();
    let pf: PFem = Arc::new(InterpolatedFem {
      name: format!("FEM_INTERPOLATED#{id}({cv})"),
      dofs: vec![DofDesc::global(cvr.dim()); nd],
      nodes: vec![vec![1.0 / 20.0; cvr.dim()]; nd],
      faces: vec![(0..nf).collect(); nd],
      cvr,
      global: global.into_iter().collect(),
      target: q,
      src: src.clone(),
      located,
    });
    target.set_finite_element(([cv], &pf))?;
  }
  Ok(target)
}
