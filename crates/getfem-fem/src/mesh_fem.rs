//! Finite element spaces on a mesh (port of getfem_mesh_fem): an element per
//! convex, a qdim, DOF enumeration with GetFEM's numbering.
//!
//! A `MeshFem` is a handle (SPEC 3): clones are the same space, and changes
//! of the space or of its mesh are seen by all its users. Elements set by a
//! rule (classical elements of degree k, or one element everywhere) also
//! apply to convexes added to the mesh later, as GetFEM's auto_add. The DOF
//! numbering is recomputed for the current mesh when needed. Values on the
//! space are tied to its value stamp, which changes when the numbering or
//! the geometry (points, convexes) changes.
//!
//! Reduction and extension matrices (GetFEM set_reduction_matrices): the
//! DOFs of a reduced space are DOFs = R basic, basic = E DOFs (R E = I),
//! general linear maps (a selection, or a square transform of the basic
//! DOFs such as a rotation of the components). A partial space is the
//! reduction by a selection (a pair of matrices of ones), composed with
//! the reduction of its base, as GetFEM's partial_mesh_fem.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use getfem_geom::geotrans::element_extent;
use getfem_geom::mesh::{Mesh, MeshData};
use getfem_geom::region::{Region, RegionArg};
use getfem_geom::shared::Shared;

use getfem_linalg::{Coo, Csr};

use crate::fem::{DdlType, DofDesc, PFem, classical_fem_alpha};
use crate::{Error, Result};
use getfem_linalg::vect_dist2;

/// Relative tolerance for identifying DOF nodes (times the element size).
/// GetFEM uses 1e-6 times the squared size, which depends on the unit.
pub const DOF_MERGE_TOL: f64 = 1e-8;

static STAMP: AtomicU64 = AtomicU64::new(1);

pub(crate) fn new_stamp() -> u64 {
  STAMP.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug, PartialEq)]
struct Dofs {
  /// convex cv: basic DOFs ind[start[cv]..start[cv + 1]]
  start: Vec<usize>,
  ind: Vec<usize>,
  nb: usize,
  /// (convex, local DOF) where each DOF block was first seen
  first: Vec<(usize, usize)>,
}

/// The DOF numbering of a space for one version of its mesh: a consistent
/// view for the duration of an assembly.
#[derive(Debug)]
pub struct DofNumbering {
  mesh: Arc<MeshData>,
  config: u64,
  fems: Vec<Option<PFem>>,
  dofs: Dofs,
  qdim: usize,
  /// qdim > 1 and scalar elements everywhere (GetFEM: is_uniformly_vectorized)
  vectorized: bool,
  version: u64,
  stamp: u64,
}

impl DofNumbering {
  pub fn linked_mesh(&self) -> &Arc<MeshData> {
    &self.mesh
  }

  pub fn qdim(&self) -> usize {
    self.qdim
  }

  pub fn nb_basic_dof(&self) -> usize {
    self.dofs.nb
  }

  pub fn is_uniformly_vectorized(&self) -> bool {
    self.vectorized
  }

  pub fn fem_of_element(&self, cv: usize) -> Option<&PFem> {
    self.fems.get(cv).and_then(|f| f.as_ref())
  }

  /// Convexes with an element.
  pub fn convex_index(&self) -> impl Iterator<Item = usize> + '_ {
    self
      .mesh
      .convex_index()
      .filter(|&cv| self.fems[cv].is_some())
  }

  /// Basic DOFs of convex cv: for each local DOF i, qdim/target_dim
  /// consecutive global DOFs.
  pub fn ind_basic_dof_of_element(&self, cv: usize) -> &[usize] {
    let d = &self.dofs;
    if cv + 1 >= d.start.len() {
      return &[];
    }
    &d.ind[d.start[cv]..d.start[cv + 1]]
  }

  /// Real position of a Lagrange DOF (from the first convex containing it).
  pub fn point_of_basic_dof(&self, dof: usize) -> Result<Vec<f64>> {
    let d = &self.dofs;
    if dof >= d.nb {
      return Err(Error::InvalidArgument(format!("DOF {dof} out of range")));
    }
    let (cv, i) = d.first[dof];
    let pf = self.fems[cv].as_ref().unwrap();
    Ok(
      self
        .mesh
        .trans_of_convex(cv)
        .transform(pf.node(i), &self.mesh.points_of_convex(cv))?,
    )
  }

  /// Basic DOFs of the convexes and faces of a region.
  /// The basic DOFs on a region (its current content): an id of the mesh
  /// (-1 everywhere) or a region.
  pub fn basic_dof_on_region(&self, r: impl Into<RegionArg>) -> Result<BTreeSet<usize>> {
    let r = r.into().content(&self.mesh)?;
    let mut out = BTreeSet::new();
    for (cv, f) in r.iter() {
      let Some(pf) = self.fem_of_element(cv) else {
        continue;
      };
      let ind = self.ind_basic_dof_of_element(cv);
      let q = self.qdim / pf.target_dim();
      for i in 0..pf.nb_dof() {
        if f.is_none_or(|f| pf.faces_of_dof(i).contains(&f)) {
          out.extend(&ind[i * q..(i + 1) * q]);
        }
      }
    }
    Ok(out)
  }

  /// Changes whenever the geometry or the elements change.
  pub fn version(&self) -> u64 {
    self.version
  }

  /// Changes when the numbering or the geometry changes: values on the
  /// space are valid while it is unchanged.
  pub fn value_stamp(&self) -> u64 {
    self.stamp
  }
}

/// Reduction and extension matrices of a space: DOFs = r (basic DOFs),
/// basic = e (DOFs), r e = I (GetFEM reduction_matrix, extension_matrix).
#[derive(Debug)]
pub struct Reduction {
  /// R; None: not stored, reduce by least squares (ls)
  pub r: Option<Csr<f64>>,
  ls: Option<LeastSquares>,
  pub e: Csr<f64>,
  /// E by columns (its transpose; gmm stores E as a col_matrix): the
  /// extension and the folds touch only its nonzeros
  pub et: Csr<f64>,
  /// the numbering they were set for
  num_stamp: u64,
  /// value stamp of the reduced space
  stamp: u64,
}

/// The DOFs of a mesh_fem for one assembly: the numbering, and the
/// reduction matrices of a reduced (or partial) space.
#[derive(Debug)]
pub struct MfDofs {
  pub numbering: Arc<DofNumbering>,
  red: Option<Arc<Reduction>>,
}

impl MfDofs {
  /// The reduction matrices of a reduced (or partial) space.
  pub fn reduction(&self) -> Option<&Reduction> {
    self.red.as_deref()
  }

  pub fn nb_dof(&self) -> usize {
    match &self.red {
      Some(r) => r.e.ncols,
      None => self.numbering.nb_basic_dof(),
    }
  }

  /// Values on the basic DOFs from values on the DOFs (GetFEM
  /// extend_vector: E per component when v has q values per DOF,
  /// interleaved).
  pub fn extend_vector(&self, v: &[f64]) -> Result<Vec<f64>> {
    let (nd, nb) = (self.nb_dof(), self.numbering.nb_basic_dof());
    let q = per_dof(v.len(), nd)?;
    let Some(r) = &self.red else {
      return Ok(v.to_vec());
    };
    // by the columns of E (gmm::mult of a col_matrix)
    let mut out = vec![0.0; nb * q];
    for j in 0..nd {
      for (d, x) in r.et.row(j) {
        for k in 0..q {
          out[d * q + k] += x * v[j * q + k];
        }
      }
    }
    Ok(out)
  }

  /// Values on the DOFs from values on the basic DOFs (GetFEM
  /// reduce_vector: R per component).
  pub fn reduce_vector(&self, v: &[f64]) -> Result<Vec<f64>> {
    let (nd, nb) = (self.nb_dof(), self.numbering.nb_basic_dof());
    let q = per_dof(v.len(), nb)?;
    let Some(red) = &self.red else {
      return Ok(v.to_vec());
    };
    let Some(r) = &red.r else {
      let ls = red.ls.as_ref().expect("R or least squares");
      let mut out = vec![0.0; nd * q];
      for k in 0..q {
        let vk: Vec<f64> = (0..nb).map(|j| v[j * q + k]).collect();
        for (i, x) in ls.reduce(&vk)?.into_iter().enumerate() {
          out[i * q + k] = x;
        }
      }
      return Ok(out);
    };
    // by the rows of R (gmm::mult of a row_matrix)
    let mut out = vec![0.0; nd * q];
    for i in 0..nd {
      for (j, x) in r.row(i) {
        for k in 0..q {
          out[i * q + k] += x * v[j * q + k];
        }
      }
    }
    Ok(out)
  }
}

/// The least squares reduction (E' E)^-1 E' of an extension matrix E (the
/// one set), followed by post (a selection, for a partial space of it)
#[derive(Clone, Debug)]
struct LeastSquares {
  post: Option<Csr<f64>>,
  ne: Arc<NormalEquations>,
}

/// E and E' E's diagonal, for E' E q = E' b
#[derive(Debug)]
struct NormalEquations {
  e: Csr<f64>,
  et: Csr<f64>,
  d: Vec<f64>,
}

impl LeastSquares {
  fn reduce(&self, b: &[f64]) -> Result<Vec<f64>> {
    let ne = &self.ne;
    let rhs = ne.et.matvec(b)?;
    let ete = |p: &[f64]| {
      ne.et
        .matvec(&ne.e.matvec(p).expect("sizes"))
        .expect("sizes")
    };
    let n = rhs.len();
    let (q, _) = getfem_linalg::cg(ete, &ne.d, &rhs, 1e-13, n.max(10)).map_err(|e| {
      Error::InvalidArgument(format!(
        "reduction by least squares (E'E q = E'b): {e}; is E of full column rank?"
      ))
    })?;
    match &self.post {
      Some(p) => Ok(p.matvec(&q)?),
      None => Ok(q),
    }
  }
}

/// values per DOF of a vector of len values on n DOFs
fn per_dof(len: usize, n: usize) -> Result<usize> {
  if n == 0 && len == 0 {
    return Ok(1);
  }
  if n == 0 || len == 0 || !len.is_multiple_of(n) {
    return Err(Error::InvalidArgument(format!("{len} values on {n} DOFs")));
  }
  Ok(len / n)
}

/// r e = I, to 1e-10
fn check_left_inverse(r: &Csr<f64>, e: &Csr<f64>) -> Result<()> {
  let re = r.mul(e)?;
  let mut worst = (0.0, 0, 0);
  for i in 0..re.nrows {
    let mut diag = 0.0;
    for (j, v) in re.row(i) {
      if j == i {
        diag = v;
      } else if v.abs() > worst.0 {
        worst = (v.abs(), i, j);
      }
    }
    if (diag - 1.0).abs() > worst.0 {
      worst = ((diag - 1.0).abs(), i, i);
    }
  }
  if worst.0 > 1e-10 {
    return Err(Error::InvalidArgument(format!(
      "the reduction matrix R is not a left inverse of the extension matrix E: (R E - I)({}, {}) = {:.3e}; give R as \"auto\" (None) to have it computed from E",
      worst.1, worst.2, worst.0
    )));
  }
  Ok(())
}

/// R = (E' E)^-1 E': stored when E' E is diagonal (every row of E at most
/// one nonzero), otherwise the least squares solve
fn least_squares(e: &Csr<f64>) -> Result<(Option<Csr<f64>>, Option<LeastSquares>)> {
  if e.ncols > e.nrows {
    return Err(Error::InvalidArgument(format!(
      "the extension matrix E ({} x {}) has more columns than rows: not of full column rank, no reduction from it",
      e.nrows, e.ncols
    )));
  }
  let et = e.transpose();
  let d: Vec<f64> = (0..et.nrows)
    .map(|j| et.row(j).map(|(_, v)| v * v).sum())
    .collect();
  if let Some(j) = d.iter().position(|&x| x == 0.0) {
    return Err(Error::InvalidArgument(format!(
      "the extension matrix E has an empty column {j}: no reduction from it"
    )));
  }
  if (0..e.nrows).all(|i| e.row(i).count() <= 1) {
    let mut r = Coo::new(et.nrows, et.ncols);
    for (j, &dj) in d.iter().enumerate() {
      for (i, v) in et.row(j) {
        r.push(j, i, v / dj);
      }
    }
    return Ok((Some(r.to_csr()?), None));
  }
  if getfem_mpi::is_master() {
    println!(
      "GetFEM-rs: reduction from the extension matrix E: E'E not diagonal, R not stored; each reduce solves E'E q = E'b (CG, diagonal preconditioner)"
    );
  }
  let ne = NormalEquations {
    e: e.clone(),
    et,
    d,
  };
  Ok((
    None,
    Some(LeastSquares {
      post: None,
      ne: Arc::new(ne),
    }),
  ))
}

/// the selection matrix of kept (rows) among n columns
fn selection(kept: &[usize], n: usize) -> Csr<f64> {
  let mut c = Coo::new(kept.len(), n);
  for (i, &d) in kept.iter().enumerate() {
    c.push(i, d, 1.0);
  }
  c.to_csr().expect("in range")
}

/// Elements for the convexes without an explicit one.
#[derive(Clone, Debug)]
enum Rule {
  /// alpha: discontinuous (with GetFEM's alpha), or continuous
  Classical {
    k: usize,
    alpha: Option<f64>,
  },
  Fem(PFem),
}

/// Which basic DOFs of the base space a partial space keeps.
#[derive(Clone, Debug)]
enum Selection {
  /// an explicit set of DOFs of the base, for the base numbering of this
  /// value stamp and this reduction of the base (its stamp, 0 for none)
  Dofs {
    dofs: BTreeSet<usize>,
    base_stamp: u64,
    base_rs: u64,
  },
  /// the DOFs on a region, for every numbering of the base
  Region(Region),
}

/// The kept DOFs for one numbering of the base space (and one state of
/// the regions of the mesh, one reduction of the base): kept among the
/// DOFs of the base (its basic DOFs if it is not reduced).
#[derive(Debug)]
struct KeptMap {
  base_version: u64,
  mesh_version: u64,
  kept: Vec<usize>,
  /// R = RR R_base, E = E_base RR^T (RR the selection; GetFEM
  /// partial_mesh_fem adapt), for this reduction of the base; its stamp
  /// the value stamp (changes with the kept set or the base numbering)
  red: Arc<Reduction>,
  base_rs: u64,
}

/// A space restricted to some basic DOFs of a base space.
#[derive(Debug)]
struct Kept {
  base: MeshFem,
  sel: Selection,
  cache: Mutex<Option<Arc<KeptMap>>>,
}

impl Clone for Kept {
  fn clone(&self) -> Self {
    Kept {
      base: self.base.clone(),
      sel: self.sel.clone(),
      cache: Mutex::new(self.cache.lock().unwrap_or_else(|e| e.into_inner()).clone()),
    }
  }
}

impl Kept {
  /// The kept DOFs for the current numbering of the base.
  fn map(&self) -> Result<Arc<KeptMap>> {
    let num = self.base.numbering()?;
    let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
    // the regions: of the mesh itself (the numbering keeps the state of
    // its points and convexes)
    let mesh_version = self.base.linked_mesh().version();
    let base_red = self.base.own_reduction()?;
    let base_rs = base_red.as_ref().map_or(0, |r| r.stamp);
    if let Some(k) = cache.as_ref()
      && k.base_version == num.version
      && k.mesh_version == mesh_version
      && k.base_rs == base_rs
    {
      return Ok(k.clone());
    }
    let kept: Vec<usize> = match &self.sel {
      Selection::Dofs {
        dofs,
        base_stamp,
        base_rs: rs,
      } => {
        if *base_stamp != num.stamp || *rs != base_rs {
          return Err(Error::InvalidArgument(
            "a partial space on an explicit DOF set was built on an older numbering or reduction of its base space (the mesh, its elements or its reduction matrices changed); build it again".into(),
          ));
        }
        dofs.iter().copied().collect()
      }
      // the basic DOFs of the base on the region
      Selection::Region(r) => {
        if base_red.is_some() {
          return Err(Error::InvalidArgument(
            "a partial space on a region of a space with reduction matrices: not available (select its DOFs with MeshFem::partial)".into(),
          ));
        }
        num.basic_dof_on_region(r)?.into_iter().collect()
      }
    };
    let stamp = match cache.as_ref() {
      Some(o) if o.kept == kept && o.red.num_stamp == num.stamp && o.base_rs == base_rs => {
        o.red.stamp
      }
      _ => new_stamp(),
    };
    let nb = base_red.as_ref().map_or(num.nb_basic_dof(), |r| r.e.ncols);
    let rr = selection(&kept, nb);
    let (r, ls, e) = match &base_red {
      Some(b) => {
        let e = b.e.mul(&rr.transpose()).expect("sizes");
        match (&b.r, &b.ls) {
          (Some(br), _) => (Some(rr.mul(br).expect("sizes")), None, e),
          // the base's least squares, then the selection
          (None, Some(bls)) => {
            let post = match &bls.post {
              Some(p) => rr.mul(p).expect("sizes"),
              None => rr,
            };
            let ls = LeastSquares {
              post: Some(post),
              ne: bls.ne.clone(),
            };
            (None, Some(ls), e)
          }
          (None, None) => unreachable!("R or least squares"),
        }
      }
      None => (Some(rr.clone()), None, rr.transpose()),
    };
    let red = Arc::new(Reduction {
      et: e.transpose(),
      r,
      ls,
      e,
      num_stamp: num.stamp,
      stamp,
    });
    let k = Arc::new(KeptMap {
      base_version: num.version,
      mesh_version,
      kept,
      red,
      base_rs,
    });
    *cache = Some(k.clone());
    Ok(k)
  }
}

#[derive(Debug)]
struct Inner {
  mesh: Mesh,
  qdims: Vec<usize>,
  /// per convex: its element, or none (overriding the rule)
  fems: BTreeMap<usize, Option<PFem>>,
  rule: Option<Rule>,
  /// per convex: its DOF partition (GetFEM dof_partition: DOFs of
  /// different partitions are not shared), 0 if absent
  partition: BTreeMap<usize, u32>,
  /// changes with every setting
  config: u64,
  kept: Option<Kept>,
  /// reduction matrices, applied or not (GetFEM use_reduction)
  red: Option<Arc<Reduction>>,
  use_red: bool,
  cache: Mutex<Option<Arc<DofNumbering>>>,
}

impl Clone for Inner {
  fn clone(&self) -> Self {
    Inner {
      mesh: self.mesh.clone(),
      qdims: self.qdims.clone(),
      fems: self.fems.clone(),
      rule: self.rule.clone(),
      partition: self.partition.clone(),
      config: self.config,
      kept: self.kept.clone(),
      red: self.red.clone(),
      use_red: self.use_red,
      cache: Mutex::new(self.cache.lock().unwrap_or_else(|e| e.into_inner()).clone()),
    }
  }
}

/// A finite element space on a mesh (GetFEM mesh_fem), as a shared handle.
#[derive(Clone, Debug)]
pub struct MeshFem(Shared<Inner>);

impl MeshFem {
  /// A space of qdim components on the mesh, without elements yet.
  pub fn new(mesh: &Mesh, qdim: usize) -> Self {
    MeshFem(Shared::new(Inner {
      mesh: mesh.clone(),
      qdims: vec![qdim.max(1)],
      fems: BTreeMap::new(),
      rule: None,
      partition: BTreeMap::new(),
      config: new_stamp(),
      kept: None,
      red: None,
      use_red: false,
      cache: Mutex::new(None),
    }))
  }

  pub fn with_classical_finite_element(self, k: usize) -> Result<Self> {
    self.set_classical_finite_element(k)?;
    Ok(self)
  }

  pub fn with_classical_discontinuous_finite_element(
    self,
    arg: impl DiscontinuousArg,
  ) -> Result<Self> {
    self.set_classical_discontinuous_finite_element(arg)?;
    Ok(self)
  }

  /// set_finite_element as a builder (`&pf`, or `(cvs, &pf)`)
  pub fn with_finite_element(self, arg: impl FiniteElementArg) -> Result<Self> {
    self.set_finite_element(arg)?;
    Ok(self)
  }

  /// set_qdim as a builder (`2`, or a shape `[3, 3]`)
  pub fn with_qdim(self, arg: impl QdimArg) -> Self {
    self.set_qdim(arg);
    self
  }

  /// Same space (not only equal)?
  pub fn same(&self, other: &MeshFem) -> bool {
    self.0.same(&other.0)
  }

  pub fn linked_mesh(&self) -> Mesh {
    self.0.snapshot().mesh.clone()
  }

  pub fn qdim(&self) -> usize {
    self.qdims().iter().product()
  }

  /// Shape of the field at a point (`[qdim]` for vector fields).
  pub fn qdims(&self) -> Vec<usize> {
    let s = self.0.snapshot();
    match &s.kept {
      Some(k) => k.base.qdims(),
      None => s.qdims.clone(),
    }
  }

  fn modify(&self, f: impl FnOnce(&mut Inner) -> Result<()>) -> Result<()> {
    self.0.modify(|s| {
      if s.kept.is_some() {
        return Err(Error::InvalidArgument(
          "a partial space is set up through its base space".into(),
        ));
      }
      f(s)?;
      s.config = new_stamp();
      Ok(())
    })
  }

  /// GetFEM's set_qdim: a number of components (`2`) or a tensor shape
  /// (`[3, 3]`, qdim the product).
  pub fn set_qdim(&self, arg: impl QdimArg) {
    self.set_shape(&arg.shape());
  }

  fn set_shape(&self, shape: &[usize]) {
    let _ = self.modify(|s| {
      s.qdims = shape.iter().map(|&d| d.max(1)).collect();
      Ok(())
    });
  }

  /// GetFEM's set_finite_element, its two forms by the argument: the
  /// element everywhere (`&pf`, also on convexes added later) or on given
  /// convexes (`(cvs, &pf)`).
  pub fn set_finite_element(&self, arg: impl FiniteElementArg) -> Result<()> {
    arg.apply(self)
  }

  /// The element pf on the given convexes.
  fn set_fem_on(&self, cvs: impl IntoIterator<Item = usize>, pf: &PFem) -> Result<()> {
    let m = self.linked_mesh().snapshot();
    let qdim = self.qdim();
    let cvs: Vec<usize> = cvs.into_iter().collect();
    for &cv in &cvs {
      check_fit(&m, cv, pf, qdim)?;
    }
    self.modify(|s| {
      for cv in cvs {
        s.fems.insert(cv, Some(pf.clone()));
      }
      Ok(())
    })
  }

  /// No element on the convexes cvs (GetFEM set_finite_element(cv, 0)).
  fn remove_fem_on(&self, cvs: impl IntoIterator<Item = usize>) -> Result<()> {
    let cvs: Vec<usize> = cvs.into_iter().collect();
    self.modify(|s| {
      for cv in cvs {
        s.fems.insert(cv, None);
      }
      Ok(())
    })
  }

  /// The DOF partition of convex cv (GetFEM set_dof_partition): DOFs are
  /// shared only between convexes of the same partition.
  pub fn set_dof_partition(&self, cv: usize, partition: u32) -> Result<()> {
    self.modify(|s| {
      if partition == 0 {
        s.partition.remove(&cv);
      } else {
        s.partition.insert(cv, partition);
      }
      Ok(())
    })
  }

  /// The DOF partition of convex cv (GetFEM get_dof_partition).
  pub fn dof_partition(&self, cv: usize) -> u32 {
    self.0.snapshot().partition.get(&cv).copied().unwrap_or(0)
  }

  /// The element pf everywhere, also on convexes added later.
  fn set_fem_all(&self, pf: &PFem) -> Result<()> {
    let m = self.linked_mesh().snapshot();
    let qdim = self.qdim();
    for cv in m.convex_index() {
      check_fit(&m, cv, pf, qdim)?;
    }
    self.set_rule(Rule::Fem(pf.clone()))
  }

  /// Lagrange elements of degree k on every element, also on convexes
  /// added later.
  pub fn set_classical_finite_element(&self, k: usize) -> Result<()> {
    self.set_classical(k, None)
  }

  /// Discontinuous Lagrange elements of degree k on every element: `k`,
  /// or `(k, alpha)` with the nodes moved toward the barycenter by alpha
  /// (GetFEM's alpha).
  pub fn set_classical_discontinuous_finite_element(
    &self,
    arg: impl DiscontinuousArg,
  ) -> Result<()> {
    let (k, alpha) = arg.degree_alpha();
    self.set_classical(k, Some(alpha))
  }

  fn set_classical(&self, k: usize, alpha: Option<f64>) -> Result<()> {
    let m = self.linked_mesh().snapshot();
    for cv in m.convex_index() {
      classical_fem_alpha(m.trans_of_convex(cv), k, alpha)?;
    }
    self.set_rule(Rule::Classical { k, alpha })
  }

  fn set_rule(&self, r: Rule) -> Result<()> {
    self.modify(|s| {
      s.fems.clear();
      s.rule = Some(r);
      Ok(())
    })
  }

  /// The numbering for the current mesh (recomputed if the mesh or the
  /// elements changed).
  pub fn numbering(&self) -> Result<Arc<DofNumbering>> {
    let s = self.0.snapshot();
    if let Some(k) = &s.kept {
      k.map()?;
      return k.base.numbering();
    }
    let mut cache = s.cache.lock().unwrap_or_else(|e| e.into_inner());
    let mv = s.mesh.geometry_version();
    if let Some(n) = cache.as_ref()
      && n.mesh.geometry_version() == mv
      && n.config == s.config
    {
      return Ok(n.clone());
    }
    let mesh = s.mesh.snapshot();
    let qdim: usize = s.qdims.iter().product();
    let mut fems: Vec<Option<PFem>> = vec![None; mesh.nb_convexes_max()];
    for cv in mesh.convex_index() {
      fems[cv] = match (s.fems.get(&cv), &s.rule) {
        (Some(pf), _) => pf.clone(),
        (None, Some(Rule::Fem(pf))) => {
          check_fit(&mesh, cv, pf, qdim)?;
          Some(pf.clone())
        }
        (None, Some(Rule::Classical { k, alpha })) => {
          Some(classical_fem_alpha(mesh.trans_of_convex(cv), *k, *alpha)?)
        }
        (None, None) => None,
      };
    }
    let dofs = enumerate(&mesh, &fems, qdim, &s.partition)?;
    let same_fems = |o: &DofNumbering| {
      o.fems.len() == fems.len()
        && o.fems.iter().zip(&fems).all(|(a, b)| match (a, b) {
          (Some(a), Some(b)) => Arc::ptr_eq(a, b),
          (None, None) => true,
          _ => false,
        })
    };
    let stamp = match cache.as_ref() {
      Some(o)
        if o.dofs == dofs
          && o.qdim == qdim
          && same_fems(o)
          && o.mesh.geometry_version() == mesh.geometry_version() =>
      {
        o.stamp
      }
      _ => new_stamp(),
    };
    let vectorized = qdim > 1 && fems.iter().flatten().all(|f| f.target_dim() == 1);
    let n = Arc::new(DofNumbering {
      mesh,
      config: s.config,
      fems,
      dofs,
      qdim,
      vectorized,
      version: new_stamp(),
      stamp,
    });
    *cache = Some(n.clone());
    Ok(n)
  }

  /// The numbering and the map to the DOFs (selection or reduction
  /// matrices), consistent for the duration of an assembly.
  pub fn dofs(&self) -> Result<MfDofs> {
    let red = match &self.0.snapshot().kept {
      Some(k) => Some(k.map()?.red.clone()),
      None => self.own_reduction()?,
    };
    Ok(MfDofs {
      numbering: self.numbering()?,
      red,
    })
  }

  /// The reduction matrices set on this space, if applied; an error if
  /// they were set for an older numbering.
  fn own_reduction(&self) -> Result<Option<Arc<Reduction>>> {
    let s = self.0.snapshot();
    let Some(r) = s.red.as_ref().filter(|_| s.use_red) else {
      return Ok(None);
    };
    if self.numbering()?.stamp != r.num_stamp {
      return Err(Error::InvalidArgument(
        "the reduction matrices were set for an older numbering of the space (the mesh or its elements changed); set them again".into(),
      ));
    }
    Ok(Some(r.clone()))
  }

  /// Reduction and extension matrices (GetFEM set_reduction_matrices):
  /// DOFs = r (basic DOFs), basic DOFs = e (DOFs), r e = I (checked).
  /// r None ("auto"): r the least squares (E' E)^-1 E', stored when E' E
  /// is diagonal (no two columns of E share a row: a selection, plain
  /// periodicity), otherwise not stored, each reduce solving E' E q = E' b
  /// (CG, with an info line). Applied from now on; valid for the current
  /// numbering (set them again when the mesh or the elements change).
  pub fn set_reduction_matrices(&self, r: Option<&Csr<f64>>, e: &Csr<f64>) -> Result<()> {
    if self.is_partial() {
      return Err(Error::InvalidArgument(
        "reduction matrices of a partial space: set them on its base".into(),
      ));
    }
    let num = self.numbering()?;
    let nb = num.nb_basic_dof();
    if e.nrows != nb {
      return Err(Error::InvalidArgument(format!(
        "wrong dimensions of the extension matrix ({} x {}) for {nb} basic DOFs",
        e.nrows, e.ncols
      )));
    }
    let (r, ls) = match r {
      Some(r) => {
        if r.ncols != nb || r.nrows != e.ncols {
          return Err(Error::InvalidArgument(format!(
            "wrong dimensions of the reduction ({} x {}) and extension ({} x {}) matrices for {nb} basic DOFs",
            r.nrows, r.ncols, e.nrows, e.ncols
          )));
        }
        check_left_inverse(r, e)?;
        (Some(r.clone()), None)
      }
      None => least_squares(e)?,
    };
    let red = Arc::new(Reduction {
      r,
      ls,
      e: e.clone(),
      et: e.transpose(),
      num_stamp: num.stamp,
      stamp: new_stamp(),
    });
    self.0.modify(|s| {
      s.red = Some(red);
      s.use_red = true;
      Ok(())
    })
  }

  /// The basic DOFs kept (GetFEM reduce_to_basic_dof): R the selection,
  /// E its transpose.
  pub fn reduce_to_basic_dof(&self, kept: &BTreeSet<usize>) -> Result<()> {
    let nb = self.nb_basic_dof()?;
    if let Some(&d) = kept.iter().find(|&&d| d >= nb) {
      return Err(Error::InvalidArgument(format!(
        "DOF {d} out of range ({nb} basic DOFs)"
      )));
    }
    let k: Vec<usize> = kept.iter().copied().collect();
    let r = selection(&k, nb);
    self.set_reduction_matrices(Some(&r), &r.transpose())
  }

  /// Apply or not the reduction matrices set (GetFEM set_reduction).
  pub fn set_reduction(&self, on: bool) -> Result<()> {
    let s = self.0.snapshot();
    if on && s.red.is_none() {
      return Err(Error::InvalidArgument("no reduction matrices set".into()));
    }
    self.0.modify(|s| s.use_red = on);
    self.own_reduction().map(|_| ())
  }

  /// The basic DOFs whose mass diagonal (the integral of the squared basis
  /// function with mim) exceeds fraction times its median: the DOFs with
  /// support in the domain, for `MeshFem::partial` (enriched DOFs whose
  /// support has no volume, in cut elements, are singular). A space of
  /// qdim 1.
  pub fn dofs_with_support(
    &self,
    mim: &crate::mesh_im::MeshIm,
    fraction: f64,
  ) -> Result<BTreeSet<usize>> {
    if self.qdim() != 1 {
      return Err(Error::InvalidArgument(
        "dofs_with_support: a space of qdim 1".into(),
      ));
    }
    let m = self.linked_mesh();
    let md = m.snapshot();
    let mut diag = vec![0.0; self.nb_basic_dof()?];
    for cv in mim.convex_index()? {
      let Some(pf) = self.fem_of_element(cv)? else {
        continue;
      };
      let pim = mim.int_method_of_element(cv)?;
      let Ok(a) = pim.approx() else {
        continue;
      };
      let dofs = self.ind_basic_dof_of_element(cv)?;
      let (gt, g) = (md.trans_of_convex(cv), md.points_of_convex(cv));
      for q in 0..a.nb_points_on_convex() {
        let xref = &a.points()[q];
        let j = gt.at(&g, xref)?.j.abs();
        let v = crate::fem::real_base_values(&pf, gt, &g, xref, 0)?.0;
        for (k, &d) in dofs.iter().enumerate() {
          diag[d] += a.coeffs()[q] * j * v[k] * v[k];
        }
      }
    }
    let mut sorted: Vec<f64> = diag.iter().copied().filter(|v| *v > 0.0).collect();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let median = sorted.get(sorted.len() / 2).copied().unwrap_or(0.0);
    Ok(
      (0..diag.len())
        .filter(|&d| diag[d] > fraction * median)
        .collect(),
    )
  }

  /// The DOFs are not the basic DOFs: reduction matrices applied, or a
  /// partial space (GetFEM is_reduced).
  pub fn is_reduced(&self) -> bool {
    let s = self.0.snapshot();
    s.kept.is_some() || (s.use_red && s.red.is_some())
  }

  /// The reduction matrix R (DOFs x basic DOFs) of a reduced (or
  /// partial) space; an error when R is not stored (set from E alone, E'E
  /// not diagonal).
  pub fn reduction_matrix(&self) -> Result<Option<Csr<f64>>> {
    let dofs = self.dofs()?;
    let Some(red) = &dofs.red else {
      return Ok(None);
    };
    match &red.r {
      Some(r) => Ok(Some(r.clone())),
      None => Err(Error::InvalidArgument(
        "the reduction matrix R is not stored: computed from the extension matrix E by least squares (E'E q = E'b, E'E not diagonal); MeshFem::reduce_vector reduces a vector".into(),
      )),
    }
  }

  /// The extension matrix E (basic DOFs x DOFs) of a reduced space.
  pub fn extension_matrix(&self) -> Result<Option<Csr<f64>>> {
    Ok(self.dofs()?.red.as_ref().map(|r| r.e.clone()))
  }

  /// Changes whenever the geometry or the elements change.
  pub fn version(&self) -> Result<u64> {
    Ok(self.numbering()?.version)
  }

  /// Changes when the DOF numbering, the geometry or the reduction
  /// changes; values on the space are valid while it is unchanged.
  pub fn value_stamp(&self) -> Result<u64> {
    match &self.0.snapshot().kept {
      Some(k) => Ok(k.map()?.red.stamp),
      None => match self.own_reduction()? {
        Some(r) => Ok(r.stamp),
        None => Ok(self.numbering()?.stamp),
      },
    }
  }

  pub fn nb_basic_dof(&self) -> Result<usize> {
    Ok(self.numbering()?.nb_basic_dof())
  }

  /// Number of DOFs (rows of the reduction matrix of a reduced space,
  /// the kept ones of a partial space).
  pub fn nb_dof(&self) -> Result<usize> {
    Ok(self.dofs()?.nb_dof())
  }

  pub fn fem_of_element(&self, cv: usize) -> Result<Option<PFem>> {
    Ok(self.numbering()?.fem_of_element(cv).cloned())
  }

  /// Convexes with an element.
  pub fn convex_index(&self) -> Result<Vec<usize>> {
    Ok(self.numbering()?.convex_index().collect())
  }

  pub fn ind_basic_dof_of_element(&self, cv: usize) -> Result<Vec<usize>> {
    Ok(self.numbering()?.ind_basic_dof_of_element(cv).to_vec())
  }

  pub fn point_of_basic_dof(&self, dof: usize) -> Result<Vec<f64>> {
    self.numbering()?.point_of_basic_dof(dof)
  }

  pub fn basic_dof_on_region(&self, r: impl Into<RegionArg>) -> Result<BTreeSet<usize>> {
    let r = r.into().on(&self.linked_mesh())?;
    self.numbering()?.basic_dof_on_region(&r)
  }

  /// A new space: the DOFs `dofs` of base (its basic DOFs if base is not
  /// reduced; the others are zero). It follows base, and is an error to
  /// use once the numbering of base changed (build it again). base itself
  /// is not changed.
  pub fn partial(base: &MeshFem, dofs: &BTreeSet<usize>) -> Result<MeshFem> {
    let num = base.numbering()?;
    let n = base.nb_dof()?;
    if let Some(&d) = dofs.iter().find(|&&d| d >= n) {
      return Err(Error::InvalidArgument(format!(
        "DOF {d} out of range ({n} DOFs)"
      )));
    }
    Self::new_partial(
      base,
      Selection::Dofs {
        dofs: dofs.clone(),
        base_stamp: num.stamp,
        base_rs: base.own_reduction()?.map_or(0, |r| r.stamp),
      },
    )
  }

  /// A new space: the basic DOFs of base on a region (GetFEM
  /// add_filtered_fem_variable). It follows base and the region: the kept
  /// DOFs are those on the current content of the region for the current
  /// numbering of base, and values on it are invalid when they change.
  /// base itself is not changed. The entire domain or a stored region.
  pub fn partial_on_region(base: &MeshFem, region: impl Into<RegionArg>) -> Result<MeshFem> {
    let region = region.into().on(&base.linked_mesh())?;
    region.lasting("partial_on_region")?;
    let p = Self::new_partial(base, Selection::Region(region.clone()))?;
    p.nb_dof()?;
    Ok(p)
  }

  fn new_partial(base: &MeshFem, sel: Selection) -> Result<MeshFem> {
    let s = base.0.snapshot();
    if s.kept.is_some() {
      return Err(Error::InvalidArgument(
        "partial space of a partial space".into(),
      ));
    }
    Ok(MeshFem(Shared::new(Inner {
      mesh: s.mesh.clone(),
      qdims: s.qdims.clone(),
      fems: BTreeMap::new(),
      rule: None,
      partition: BTreeMap::new(),
      config: new_stamp(),
      kept: Some(Kept {
        base: base.clone(),
        sel,
        cache: Mutex::new(None),
      }),
      red: None,
      use_red: false,
      cache: Mutex::new(None),
    })))
  }

  pub fn is_partial(&self) -> bool {
    self.0.snapshot().kept.is_some()
  }

  /// Values on the basic DOFs from values on the DOFs, E v (GetFEM's;
  /// zero where not kept for a partial space).
  pub fn extend_vector(&self, v: &[f64]) -> Result<Vec<f64>> {
    self.dofs()?.extend_vector(v)
  }

  /// Values on the DOFs from values on the basic DOFs, R v (GetFEM's;
  /// least squares when R is not stored).
  pub fn reduce_vector(&self, v: &[f64]) -> Result<Vec<f64>> {
    self.dofs()?.reduce_vector(v)
  }
}

fn check_fit(m: &MeshData, cv: usize, pf: &PFem, qdim: usize) -> Result<()> {
  if !m.is_convex_valid(cv) {
    return Err(Error::InvalidArgument(format!(
      "convex {cv} does not exist"
    )));
  }
  let gt = m.trans_of_convex(cv);
  if *gt.basic_structure() != *pf.ref_convex().structure().basic() {
    return Err(Error::InvalidArgument(format!(
      "{} does not fit convex {cv} ({})",
      pf.name(),
      gt.name()
    )));
  }
  if !qdim.is_multiple_of(pf.target_dim()) {
    return Err(Error::InvalidArgument(format!(
      "qdim {qdim} not a multiple of the target dim of {}",
      pf.name()
    )));
  }
  Ok(())
}

/// GetFEM's mesh_fem::enumerate_dof.
fn enumerate(
  m: &MeshData,
  fems: &[Option<PFem>],
  qdim: usize,
  partition: &BTreeMap<usize, u32>,
) -> Result<Dofs> {
  let part = |cv: usize| partition.get(&cv).copied().unwrap_or(0);
  {
    let ncv = m.nb_convexes_max();
    let size: Vec<f64> = (0..ncv)
      .map(|cv| {
        if fems[cv].is_some() {
          element_extent(&m.points_of_convex(cv))
        } else {
          0.0
        }
      })
      .collect();
    // DOFs announced to a convex by an already numbered neighbor
    // (node, description, partition, DOF)
    #[allow(clippy::type_complexity)]
    let mut pending: Vec<Vec<(Vec<f64>, DofDesc, u32, usize)>> = vec![vec![]; ncv];
    let mut done = vec![false; ncv];
    let mut start = vec![0usize; ncv + 1];
    let mut ind = vec![];
    let mut first = vec![];
    let mut nb = 0;
    // global DOFs: numbered when first met (GetFEM enumerate_dof)
    let mut global: BTreeMap<usize, usize> = BTreeMap::new();
    for cv in 0..ncv {
      start[cv] = ind.len();
      let Some(pf) = fems
        .get(cv)
        .and_then(|f| f.as_ref())
        .filter(|_| m.is_convex_valid(cv))
      else {
        continue;
      };
      let q = qdim / pf.target_dim();
      let (gt, g) = (m.trans_of_convex(cv), m.points_of_convex(cv));
      for i in 0..pf.nb_dof() {
        let desc = &pf.dof_types()[i];
        let idof = if desc.ddl.first().is_some_and(|d| d.t == DdlType::GlobalDof) {
          let g = pf
            .global_dofs()
            .ok_or(Error::NotImplemented("global DOFs without indices"))?[i];
          match global.get(&g) {
            Some(&d) => Some(d),
            None => {
              global.insert(g, nb);
              None
            }
          }
        } else if !desc.linkable {
          None
        } else {
          let p = gt.transform(pf.node(i), &g)?;
          let tol = DOF_MERGE_TOL * size[cv];
          let pc = part(cv);
          let found = pending[cv]
            .iter()
            .find(|(x, d, q, _)| *q == pc && vect_dist2(x, &p) <= tol && d.compare(desc).is_eq())
            .map(|e| e.3);
          if found.is_none() {
            for n in m.neighbors_of_convex_faces(cv, pf.faces_of_dof(i)) {
              if done[n] || fems.get(n).is_none_or(|f| f.is_none()) {
                continue;
              }
              let tn = DOF_MERGE_TOL * size[n];
              let pn = &mut pending[n];
              match pn
                .iter_mut()
                .find(|(x, d, q, _)| *q == pc && vect_dist2(x, &p) <= tn && d.compare(desc).is_eq())
              {
                Some(e) => e.3 = nb,
                None => pn.push((p.clone(), desc.clone(), pc, nb)),
              }
            }
          }
          found
        };
        let d0 = match idof {
          Some(d) => d,
          None => {
            let d = nb;
            nb += q;
            first.extend((0..q).map(|_| (cv, i)));
            d
          }
        };
        ind.extend(d0..d0 + q);
      }
      done[cv] = true;
      pending[cv].clear();
    }
    start[ncv] = ind.len();
    Ok(Dofs {
      start,
      ind,
      nb,
      first,
    })
  }
}

/// The forms of MeshFem::set_finite_element (GetFEM's overloads).
pub trait FiniteElementArg {
  fn apply(self, mf: &MeshFem) -> Result<()>;
}

impl FiniteElementArg for &PFem {
  fn apply(self, mf: &MeshFem) -> Result<()> {
    mf.set_fem_all(self)
  }
}

impl<I: IntoIterator<Item = usize>> FiniteElementArg for (I, &PFem) {
  fn apply(self, mf: &MeshFem) -> Result<()> {
    mf.set_fem_on(self.0, self.1)
  }
}

/// `(cvs, None)` removes the elements of cvs (GetFEM: a null pfem).
impl<I: IntoIterator<Item = usize>> FiniteElementArg for (I, Option<&PFem>) {
  fn apply(self, mf: &MeshFem) -> Result<()> {
    match self.1 {
      Some(pf) => mf.set_fem_on(self.0, pf),
      None => mf.remove_fem_on(self.0),
    }
  }
}

/// The forms of MeshFem::set_classical_discontinuous_finite_element
/// (GetFEM's overloads): a degree, or a degree and alpha.
pub trait DiscontinuousArg {
  fn degree_alpha(&self) -> (usize, f64);
}

impl DiscontinuousArg for usize {
  fn degree_alpha(&self) -> (usize, f64) {
    (*self, 0.0)
  }
}

impl DiscontinuousArg for (usize, f64) {
  fn degree_alpha(&self) -> (usize, f64) {
    *self
  }
}

/// The forms of MeshFem::set_qdim (GetFEM's overloads): a number of
/// components or a tensor shape.
pub trait QdimArg {
  fn shape(&self) -> Vec<usize>;
}

impl QdimArg for usize {
  fn shape(&self) -> Vec<usize> {
    vec![(*self).max(1)]
  }
}

/// an integer literal (`2`) is an i32
impl QdimArg for i32 {
  fn shape(&self) -> Vec<usize> {
    vec![usize::try_from(*self).unwrap_or(1).max(1)]
  }
}

impl QdimArg for &[usize] {
  fn shape(&self) -> Vec<usize> {
    self.to_vec()
  }
}

impl<const N: usize> QdimArg for [usize; N] {
  fn shape(&self) -> Vec<usize> {
    self.to_vec()
  }
}

impl<const N: usize> QdimArg for &[usize; N] {
  fn shape(&self) -> Vec<usize> {
    self.to_vec()
  }
}

impl QdimArg for &Vec<usize> {
  fn shape(&self) -> Vec<usize> {
    self.to_vec()
  }
}
