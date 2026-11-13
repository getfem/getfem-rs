//! Finite elements (port of getfem_fem): DOF descriptors, the element trait
//! and the Lagrange families (tau-equivalent elements with polynomial or
//! rational bases). Node and basis order identical to GetFEM.

use std::cmp::Ordering;
use std::fmt;
use std::sync::Arc;

use getfem_geom::convex_ref::{self as cr, PRef};
use getfem_geom::geotrans::{self, PGeoTrans, PointGeom};
use getfem_geom::intern::Interner;
use getfem_geom::naming::{Descriptor, Param};
use getfem_geom::poly::{Polynomial, RationalFraction};
use getfem_linalg::tensor::Tensor;

use crate::fem_composite as fc;
use crate::fem_special as fs;
use crate::{Error, Result};

pub type Poly = Polynomial<f64>;
pub(crate) type Shape = RationalFraction<f64>;

// ---------------------------------------------------------- DOF descriptors

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DdlType {
  Lagrange,
  NormalDerivative,
  Derivative,
  MeanValue,
  Bubble1,
  LagrangeNonconforming,
  GlobalDof,
  SecondDerivative,
  NormalComponent,
  EdgeComponent,
  IpkCenter,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DdlElem {
  pub t: DdlType,
  pub hier_degree: i16,
  pub hier_raff: u16,
  pub spec: usize,
}

impl DdlElem {
  fn of(t: DdlType) -> Self {
    DdlElem {
      t,
      hier_degree: -1,
      hier_raff: 0,
      spec: 0,
    }
  }
}

/// A new range of 1000 xfem indices for the DOFs of an enriched space
/// (GetFEM's reserve_xfem_index: 1100, 2100, ...).
pub fn reserve_xfem_index() -> usize {
  static IND: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(100);
  IND.fetch_add(1000, std::sync::atomic::Ordering::Relaxed) + 1000
}

/// What a degree of freedom is (value, derivative, ...), per direction.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DofDesc {
  pub ddl: Vec<DdlElem>,
  pub linkable: bool,
  pub coord_index: usize,
  pub xfem_index: usize,
  pub all_faces: bool,
}

impl DofDesc {
  fn uniform(n: usize, t: DdlType) -> Self {
    DofDesc {
      ddl: vec![DdlElem::of(t); n],
      linkable: true,
      coord_index: 0,
      xfem_index: 0,
      all_faces: false,
    }
  }

  pub fn lagrange(n: usize) -> Self {
    Self::uniform(n, DdlType::Lagrange)
  }

  /// Degree-0 element: one DOF on all faces, not shared.
  pub fn lagrange_0(n: usize) -> Self {
    DofDesc {
      all_faces: true,
      linkable: false,
      ..Self::lagrange(n)
    }
  }

  /// Lagrange DOF not shared between elements (discontinuous elements).
  pub fn lagrange_nonconforming(n: usize) -> Self {
    DofDesc {
      linkable: false,
      ..Self::lagrange(n)
    }
  }

  /// Derivative in direction k (GetFEM derivative_dof).
  pub fn derivative(n: usize, k: usize) -> Self {
    let mut d = Self::lagrange(n);
    d.ddl[k] = DdlElem::of(DdlType::Derivative);
    d
  }

  /// Second derivative in directions k1, k2 (GetFEM second_derivative_dof).
  pub fn second_derivative(n: usize, k1: usize, k2: usize) -> Self {
    let mut d = Self::lagrange(n);
    d.ddl[k1] = DdlElem::of(DdlType::SecondDerivative);
    d.ddl[k2] = DdlElem::of(DdlType::SecondDerivative);
    d
  }

  pub fn normal_derivative(n: usize) -> Self {
    Self::uniform(n, DdlType::NormalDerivative)
  }

  pub fn normal_component(n: usize) -> Self {
    Self::uniform(n, DdlType::NormalComponent)
  }

  pub fn edge_component(n: usize) -> Self {
    Self::uniform(n, DdlType::EdgeComponent)
  }

  /// Component m of a vector Lagrange DOF (GetFEM to_coord_dof).
  pub fn coord(n: usize, m: usize) -> Self {
    DofDesc {
      coord_index: m,
      ..Self::lagrange(n)
    }
  }

  /// Interior DOF k at the element center (GetFEM ipk_center_dof).
  pub fn ipk_center(n: usize, k: usize) -> Self {
    let mut d = Self::uniform(n, DdlType::IpkCenter);
    d.ddl.iter_mut().for_each(|e| e.spec = k);
    d
  }

  /// Bubble DOF (GetFEM bubble1_dof).
  pub fn bubble1(n: usize) -> Self {
    Self::uniform(n, DdlType::Bubble1)
  }

  /// The DOF with a hierarchical degree (GetFEM deg_hierarchical_dof).
  pub(crate) fn with_hier_degree(&self, deg: usize) -> Self {
    let mut d = self.clone();
    d.ddl.iter_mut().for_each(|e| e.hier_degree = deg as i16);
    d
  }

  /// GetFEM's dof_hierarchical_compatibility: same kinds, ignoring the
  /// hierarchical degrees.
  pub(crate) fn hierarchical_compatible(&self, o: &Self) -> bool {
    if self.coord_index != o.coord_index
      || self.linkable != o.linkable
      || self.xfem_index != o.xfem_index
    {
      return false;
    }
    let n = self.ddl.len().min(o.ddl.len());
    self.ddl[..n]
      .iter()
      .zip(&o.ddl[..n])
      .all(|(a, b)| a.t == b.t)
      && self.ddl[n..].iter().all(|d| d.t == DdlType::Lagrange)
      && o.ddl[n..].iter().all(|d| d.t == DdlType::Lagrange)
  }

  pub fn global(n: usize) -> Self {
    DofDesc {
      all_faces: true,
      linkable: false,
      ..Self::uniform(n, DdlType::GlobalDof)
    }
  }

  pub fn product(a: &DofDesc, b: &DofDesc) -> DofDesc {
    let mut ddl: Vec<DdlElem> = a.ddl.iter().chain(&b.ddl).copied().collect();
    let hd = ddl.iter().map(|d| d.hier_degree).max().unwrap_or(-1);
    let hr = ddl.iter().map(|d| d.hier_raff).max().unwrap_or(0);
    for d in &mut ddl {
      d.hier_degree = hd;
      d.hier_raff = hr;
    }
    DofDesc {
      ddl,
      linkable: a.linkable && b.linkable,
      coord_index: a.coord_index.max(b.coord_index),
      xfem_index: a.xfem_index,
      all_faces: a.all_faces || b.all_faces,
    }
  }

  /// The same DOF with the xfem index ind (GetFEM's xfem_dof): distinct
  /// from the DOFs of other indices at the same node.
  pub fn xfem_dof(&self, ind: usize) -> DofDesc {
    DofDesc {
      xfem_index: ind,
      ..self.clone()
    }
  }

  /// GetFEM's dof_weak_compatibility (missing trailing entries are Lagrange).
  fn weak_cmp(&self, o: &Self) -> Ordering {
    for (a, b) in self.ddl.iter().zip(&o.ddl) {
      let c = a.t.cmp(&b.t).then(a.hier_degree.cmp(&b.hier_degree));
      if c != Ordering::Equal {
        return c;
      }
    }
    let n = self.ddl.len().min(o.ddl.len());
    if self.ddl[n..].iter().any(|d| d.t != DdlType::Lagrange) {
      return Ordering::Greater;
    }
    if o.ddl[n..].iter().any(|d| d.t != DdlType::Lagrange) {
      return Ordering::Less;
    }
    Ordering::Equal
  }

  /// GetFEM's dof_description_compare.
  pub fn compare(&self, o: &Self) -> Ordering {
    self
      .coord_index
      .cmp(&o.coord_index)
      .then(self.linkable.cmp(&o.linkable))
      .then(self.xfem_index.cmp(&o.xfem_index))
      .then_with(|| self.weak_cmp(o))
  }
}

// ------------------------------------------------------------ element trait

/// A finite element on a reference convex. Values are tensors
/// [nb_base, target_dim], gradients [nb_base, target_dim, dim], Hessians
/// [nb_base, target_dim, dim * dim] (column-major, GetFEM's base_tensor layout).
pub trait Fem: Send + Sync + fmt::Debug {
  fn name(&self) -> &str;
  fn ref_convex(&self) -> &PRef;
  fn dim(&self) -> usize {
    self.ref_convex().dim()
  }
  fn target_dim(&self) -> usize;
  fn nb_dof(&self) -> usize;
  fn nb_base(&self) -> usize {
    self.nb_dof()
  }
  fn dof_types(&self) -> &[DofDesc];
  /// Reference position of DOF i.
  fn node(&self, i: usize) -> &[f64];
  /// Faces of the reference convex DOF i lies on.
  fn faces_of_dof(&self, i: usize) -> &[usize];
  /// tau-equivalent: phi_i(tau(xref)) = phi_i^ref(xref), no element matrix.
  fn is_equivalent(&self) -> bool;
  fn is_lagrange(&self) -> bool;
  fn is_polynomial(&self) -> bool;
  fn estimated_degree(&self) -> usize;
  fn base_value(&self, xref: &[f64]) -> Result<Tensor<f64>>;
  fn grad_base_value(&self, xref: &[f64]) -> Result<Tensor<f64>>;
  fn hess_base_value(&self, xref: &[f64]) -> Result<Tensor<f64>>;
  /// The element matrix M_T (nb_dof x nb_base) of a real element with
  /// nodes g (N x nb_points of pgt): phi_i = sum_j M_ij phi_ref_j after the
  /// reference-to-real mapping (GetFEM mat_trans). None if tau-equivalent.
  fn mat_trans(&self, _g: &Tensor<f64>, _pgt: &PGeoTrans) -> Result<Option<Tensor<f64>>> {
    Ok(None)
  }
  /// Mapping of the components of vector elements.
  fn vectorial(&self) -> Vectorial {
    Vectorial::None
  }
  /// Global index of each DOF (GetFEM index_of_global_dof), for elements
  /// whose DOFs are global DOFs.
  fn global_dofs(&self) -> Option<&[usize]> {
    None
  }
  /// Defined on the real element only (interpolated elements, global
  /// functions; GetFEM real_element_defined).
  fn real_element_defined(&self) -> bool {
    false
  }
  /// Their values at the point of c:
  /// `val[i]`, and up to der the real gradients `grad[i + nb_dof*r]` and
  /// Hessians `hess[i + nb_dof*(r + n*s)]`.
  fn real_values(&self, _c: &FemCtx, _der: usize) -> Result<RealValues> {
    Err(Error::InvalidArgument(format!(
      "{}: not defined on the real element",
      self.name()
    )))
  }
  /// The interior element of an HHO element (GetFEM
  /// interior_fem_of_hho_method); None for other elements.
  fn hho_interior(&self) -> Option<PFem> {
    None
  }
  /// The base functions as polynomials of the reference coordinates, for
  /// scalar tau-equivalent polynomial elements (GetFEM fem<base_poly>);
  /// None for the others.
  fn base_polynomials(&self) -> Option<Vec<Poly>> {
    None
  }
}

pub type PFem = Arc<dyn Fem>;

/// Values, gradients, Hessians of the functions of an element at a point.
pub type RealValues = (Vec<f64>, Vec<f64>, Vec<f64>);

/// A point of an element where real-only elements are evaluated (GetFEM's
/// fem_interpolation_context): the geometric transformation and nodes g,
/// the reference point xref and the real one x, K and B there (N x P,
/// column-major), and the side forced on the level set xfem_ls
/// (xfem_side: 1 for Xfem_plus, -1 for Xfem_minus, else 0; see side_on).
#[derive(Clone, Copy, Debug)]
pub struct FemCtx<'a> {
  pub pgt: &'a PGeoTrans,
  pub g: &'a Tensor<f64>,
  pub xref: &'a [f64],
  pub x: &'a [f64],
  pub k: &'a [f64],
  pub b: &'a [f64],
  pub xfem_side: i32,
  /// the id of the level set the point is on (`LevelSet::id`), 0 none
  pub xfem_ls: usize,
  /// xref as point ip of an integration method: the base elements' values
  /// from their fem_precomp on it (GetFEM pfp, ii)
  pub pim: Option<(&'a crate::integration::PIntegration, usize)>,
}

impl<'a> FemCtx<'a> {
  /// The side forced on the level set of id ls_id (Xfem_plus,
  /// Xfem_minus): xfem_side if the point is on it, else 0 (not GetFEM's:
  /// the closest level set, and every level set of the global functions)
  pub fn side_on(&self, ls_id: usize) -> i32 {
    if ls_id != 0 && ls_id == self.xfem_ls {
      self.xfem_side
    } else {
      0
    }
  }

  /// The context at xref with the geometry pg = pgt.at(g, xref), side 0.
  pub fn new(pgt: &'a PGeoTrans, g: &'a Tensor<f64>, xref: &'a [f64], pg: &'a PointGeom) -> Self {
    FemCtx {
      pgt,
      g,
      xref,
      x: &pg.x,
      k: pg.k.data(),
      b: pg.b.data(),
      xfem_side: 0,
      xfem_ls: 0,
      pim: None,
    }
  }
}

impl PartialEq for dyn Fem {
  fn eq(&self, o: &Self) -> bool {
    self.name() == o.name()
  }
}

// ------------------------------------------------- polynomial / rational FEM

/// Scalar tau-equivalent element with polynomial or rational shape functions.
pub struct PolyFem {
  name: String,
  pub(crate) cvr: PRef,
  pub(crate) dofs: Vec<DofDesc>,
  pub(crate) nodes: Vec<Vec<f64>>,
  pub(crate) faces: Vec<Vec<usize>>,
  pub(crate) base: Vec<Shape>,
  grad: Vec<Vec<Shape>>,
  hess: Vec<Vec<Shape>>,
  lagrange: bool,
  polynomial: bool,
  pub(crate) degree: usize,
  /// element matrix of a transformed element (Hermite, Argyris, ...)
  pub(crate) trans: Option<crate::fem_transformed::TransKind>,
  /// components of vector elements (base[i + nb*a]), and their mapping
  pub(crate) target: usize,
  pub(crate) vectorial: Vectorial,
}

/// How the components of a vector element map to the real element
/// (GetFEM vectorial_type): unchanged, with K (primal: Raviart-Thomas),
/// with B (dual: Nedelec).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vectorial {
  None,
  Primal,
  Dual,
}

impl fmt::Debug for PolyFem {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}", self.name)
  }
}

/// Faces of each DOF: all faces for all_faces DOFs, else |is_in_face| < 1e-7.
pub(crate) fn faces_of(cvr: &PRef, dofs: &[DofDesc], nodesref: &[Vec<f64>]) -> Vec<Vec<usize>> {
  let nf = cvr.structure().nb_faces();
  dofs
    .iter()
    .zip(nodesref)
    .map(|(d, xref)| {
      (0..nf)
        .filter(|&f| {
          d.all_faces
            || cvr
              .is_in_face(f, xref)
              .map(|v| v.abs() < 1e-7)
              .unwrap_or(false)
        })
        .collect()
    })
    .collect()
}

impl PolyFem {
  pub(crate) fn new(
    name: String,
    cvr: PRef,
    dofs: Vec<DofDesc>,
    nodesref: Vec<Vec<f64>>,
    base: Vec<Shape>,
    degree: usize,
  ) -> Result<Self> {
    let faces = faces_of(&cvr, &dofs, &nodesref);
    Self::with_faces(name, cvr, dofs, nodesref, faces, base, degree)
  }

  pub(crate) fn with_faces(
    name: String,
    cvr: PRef,
    dofs: Vec<DofDesc>,
    nodesref: Vec<Vec<f64>>,
    faces: Vec<Vec<usize>>,
    base: Vec<Shape>,
    degree: usize,
  ) -> Result<Self> {
    let n = cvr.dim();
    let grad: Vec<Vec<Shape>> = base
      .iter()
      .map(|b| (0..n).map(|j| b.derived(j)).collect())
      .collect();
    let hess = grad
      .iter()
      .map(|g| (0..n * n).map(|jk| g[jk % n].derived(jk / n)).collect())
      .collect();
    let polynomial = base.iter().all(|b| b.denominator().real_degree() == 0);
    Ok(PolyFem {
      name,
      cvr,
      dofs,
      nodes: nodesref,
      faces,
      base,
      grad,
      hess,
      lagrange: true,
      polynomial,
      degree,
      trans: None,
      target: 1,
      vectorial: Vectorial::None,
    })
  }

  /// A vector element: base[i + nb*a] is component a of function i.
  pub(crate) fn vector(mut self, target: usize, v: Vectorial) -> Self {
    self.target = target;
    self.vectorial = v;
    self
  }

  /// A transformed element with this element matrix.
  pub(crate) fn transformed(mut self, t: crate::fem_transformed::TransKind) -> Self {
    self.trans = Some(t);
    self.lagrange = false;
    self
  }

  /// Not a Lagrange element (bubbles, hierarchical elements).
  pub(crate) fn not_lagrange(mut self) -> Self {
    self.lagrange = false;
    self
  }

  /// The shape functions.
  pub fn basis(&self) -> &[Shape] {
    &self.base
  }

  fn check(&self, xref: &[f64]) -> Result<()> {
    if xref.len() != self.cvr.dim() {
      return Err(Error::InvalidArgument(format!(
        "{}: point of dim {}",
        self.name,
        xref.len()
      )));
    }
    Ok(())
  }
}

impl Fem for PolyFem {
  fn name(&self) -> &str {
    &self.name
  }
  fn base_polynomials(&self) -> Option<Vec<Poly>> {
    if self.target != 1 || self.trans.is_some() {
      return None;
    }
    self
      .base
      .iter()
      .map(|b| {
        let d = b.denominator();
        (d.real_degree() == 0 && d.coeffs()[0] != 0.0).then(|| b.numerator() / d.coeffs()[0])
      })
      .collect()
  }
  fn ref_convex(&self) -> &PRef {
    &self.cvr
  }
  fn target_dim(&self) -> usize {
    self.target
  }
  fn nb_dof(&self) -> usize {
    self.dofs.len()
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
    self.trans.is_none()
  }
  fn is_lagrange(&self) -> bool {
    self.lagrange
  }
  fn is_polynomial(&self) -> bool {
    self.polynomial
  }
  fn estimated_degree(&self) -> usize {
    self.degree
  }
  fn base_value(&self, xref: &[f64]) -> Result<Tensor<f64>> {
    self.check(xref)?;
    Tensor::from_data(
      &[self.base.len() / self.target, self.target],
      self.base.iter().map(|b| b.eval(xref)).collect(),
    )
    .map_err(Into::into)
  }
  fn grad_base_value(&self, xref: &[f64]) -> Result<Tensor<f64>> {
    self.check(xref)?;
    let (r, n) = (self.base.len(), self.cvr.dim());
    let mut t = Tensor::zeros(&[r / self.target, self.target, n]);
    for j in 0..n {
      for i in 0..r {
        t.data_mut()[i + r * j] = self.grad[i][j].eval(xref);
      }
    }
    Ok(t)
  }
  fn hess_base_value(&self, xref: &[f64]) -> Result<Tensor<f64>> {
    self.check(xref)?;
    let (r, n) = (self.base.len(), self.cvr.dim());
    let mut t = Tensor::zeros(&[r / self.target, self.target, n * n]);
    for jk in 0..n * n {
      for i in 0..r {
        t.data_mut()[i + r * jk] = self.hess[i][jk].eval(xref);
      }
    }
    Ok(t)
  }
  fn vectorial(&self) -> Vectorial {
    self.vectorial
  }
  fn mat_trans(&self, g: &Tensor<f64>, pgt: &PGeoTrans) -> Result<Option<Tensor<f64>>> {
    match self.trans {
      None => Ok(None),
      Some(t) => crate::fem_transformed::mat_trans(t, self, g, pgt).map(Some),
    }
  }
}

// ------------------------------------------------------------ constructors

static CACHE: Interner<String, PolyFem> = Interner::new();

pub(crate) fn cached(name: String, build: impl FnOnce(String) -> Result<PolyFem>) -> Result<PFem> {
  Ok(CACHE.get_or_try(&name.clone(), || build(name))? as PFem)
}

/// Lagrange basis function of reference node noderef of the degree-k simplex (GetFEM PK_fem_).
pub(crate) fn pk_basis(n: usize, k: usize, noderef: &[f64]) -> Poly {
  let one = Poly::one(n);
  if k == 0 {
    return one;
  }
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

/// GetFEM's %g formatting of the alpha parameter.
fn fmt_g(v: f64) -> String {
  let s = format!("{v}");
  if s.len() > 8 {
    format!("{v:.6}")
      .trim_end_matches('0')
      .trim_end_matches('.')
      .to_string()
  } else {
    s
  }
}

/// Contract the element toward its barycenter: xref -> (1 - alpha) xref + alpha gref,
/// basis composed with the inverse map (GetFEM's alpha of discontinuous FEMs).
pub(crate) fn shrink(nodesref: &mut [Vec<f64>], base: &mut [Poly], alpha: f64) {
  if alpha == 0.0 {
    return;
  }
  let n = nodesref[0].len();
  let gref: Vec<f64> = (0..n)
    .map(|d| nodesref.iter().map(|xref| xref[d]).sum::<f64>() / nodesref.len() as f64)
    .collect();
  for xref in nodesref.iter_mut() {
    xref
      .iter_mut()
      .zip(&gref)
      .for_each(|(a, b)| *a = (1.0 - alpha) * *a + alpha * b);
  }
  for (d, &gd) in gref.iter().enumerate() {
    let s = Poly::from_coeffs(
      1,
      2,
      vec![-alpha * gd / (1.0 - alpha), 1.0 / (1.0 - alpha), 0.0],
    )
    .unwrap();
    for b in base.iter_mut() {
      *b = b.substitute_var(&s, d);
    }
  }
}

pub fn pk(n: usize, k: usize, disc: Option<f64>) -> Result<PFem> {
  if n == 0 {
    return Err(Error::InvalidArgument("FEM_PK: dimension 0".into()));
  }
  let name = match disc {
    None => format!("FEM_PK({n},{k})"),
    Some(0.0) => format!("FEM_PK_DISCONTINUOUS({n},{k})"),
    Some(a) => format!("FEM_PK_DISCONTINUOUS({n},{k},{})", fmt_g(a)),
  };
  if let Some(a) = disc
    && !(0.0..1.0).contains(&a)
  {
    return Err(Error::InvalidArgument(format!(
      "{name}: alpha must be in [0, 1)"
    )));
  }
  cached(name, |name| {
    let cvn = cr::simplex_of_reference_k(n, k);
    let mut nodesref = cvn.points().to_vec();
    let mut base: Vec<Poly> = nodesref.iter().map(|xref| pk_basis(n, k, xref)).collect();
    // the faces of the conforming DOFs (P0: on every face), from the
    // undeformed nodes; then the discontinuous DOF type (GetFEM PK_fem_,
    // then PK_discont_ replaces dof_types_)
    let conforming = if k == 0 {
      DofDesc::lagrange_0(n)
    } else {
      DofDesc::lagrange(n)
    };
    let cvr = cr::simplex_of_reference(n);
    let faces = faces_of(&cvr, &vec![conforming.clone(); nodesref.len()], &nodesref);
    let dof = match disc {
      Some(_) => DofDesc::lagrange_nonconforming(n),
      None => conforming,
    };
    let dofs = vec![dof; nodesref.len()];
    if let Some(a) = disc {
      shrink(&mut nodesref, &mut base, a);
    }
    PolyFem::with_faces(
      name,
      cvr,
      dofs,
      nodesref,
      faces,
      base.into_iter().map(Into::into).collect(),
      k,
    )
  })
}

/// Tensor product, first factor fastest (GetFEM tproduct_femi).
/// Canonical names: chains of 1D Lagrange factors are FEM_QK, a simplex
/// times a segment is FEM_PK_PRISM (as GetFEM names them).
fn product_name(a: &str, b: &str) -> String {
  let parse = |s: &str| Descriptor::parse(s).ok();
  if let (Some(da), Some(db)) = (parse(a), parse(b)) {
    let (sa, sb) = (da.name.as_str(), db.name.as_str());
    let pa: Vec<String> = da.params.iter().map(|p| p.to_string()).collect();
    let pb: Vec<String> = db.params.iter().map(|p| p.to_string()).collect();
    let tail = |v: &[String]| v[1..].join(",");
    for (pk, qk, prism) in [
      ("FEM_PK", "FEM_QK", "FEM_PK_PRISM"),
      (
        "FEM_PK_DISCONTINUOUS",
        "FEM_QK_DISCONTINUOUS",
        "FEM_PK_PRISM_DISCONTINUOUS",
      ),
    ] {
      if sb == pk && pb.first().map(String::as_str) == Some("1") {
        let n: usize = pa.first().and_then(|v| v.parse().ok()).unwrap_or(0);
        if (sa == pk && n == 1 || sa == qk) && tail(&pa) == tail(&pb) {
          return format!("{qk}({},{})", n + 1, tail(&pb));
        }
        if sa == pk && n >= 2 && tail(&pa) == tail(&pb) {
          return format!("{prism}({},{})", n + 1, tail(&pb));
        }
      }
    }
  }
  format!("FEM_PRODUCT({a},{b})")
}

pub fn product(a: &PFem, b: &PFem) -> Result<PFem> {
  product_named(a, b, product_name(a.name(), b.name()))
}

/// Tensor product under a given name.
pub(crate) fn product_named(a: &PFem, b: &PFem, name: String) -> Result<PFem> {
  let (pa, pb) = (poly_of(a)?, poly_of(b)?);
  if pa.target != 1 || pb.target != 1 {
    return Err(Error::InvalidArgument(format!(
      "{name}: products of vector elements"
    )));
  }
  cached(name, |name| {
    let cvr = cr::convex_ref_product(&pa.cvr, &pb.cvr);
    let (mut dofs, mut nodesref, mut base) = (vec![], vec![], vec![]);
    for j in 0..pb.nb_dof() {
      for i in 0..pa.nb_dof() {
        dofs.push(DofDesc::product(&pa.dofs[i], &pb.dofs[j]));
        nodesref.push([pa.nodes[i].as_slice(), pb.nodes[j].as_slice()].concat());
        base.push(pa.base[i].direct_product(&pb.base[j]));
      }
    }
    let f = PolyFem::new(name, cvr, dofs, nodesref, base, pa.degree + pb.degree)?;
    Ok(if pa.lagrange && pb.lagrange {
      f
    } else {
      f.not_lagrange()
    })
  })
}

pub(crate) fn poly_of(f: &PFem) -> Result<Arc<PolyFem>> {
  CACHE.get_or_try::<Error>(&f.name().to_string(), || {
    Err(Error::InvalidArgument(format!(
      "{}: not a polynomial element",
      f.name()
    )))
  })
}

pub fn qk(n: usize, k: usize, disc: Option<f64>) -> Result<PFem> {
  match n {
    0 => Err(Error::InvalidArgument("FEM_QK: dimension 0".into())),
    1 => pk(1, k, disc),
    _ => product(&qk(n - 1, k, disc)?, &pk(1, k, disc)?),
  }
}

pub fn prism_pk(n: usize, k: usize, disc: Option<f64>) -> Result<PFem> {
  match n {
    0 | 1 => Err(Error::InvalidArgument(
      "FEM_PRISM_PK: dimension >= 2".into(),
    )),
    // a 2D prism is the product of two segments (GetFEM 66c9a5b9)
    2 => qk(2, k, disc),
    _ => product(&pk(n - 1, k, disc)?, &pk(1, k, disc)?),
  }
}

/// Element whose nodes and shape functions are those of a geometric
/// transformation (incomplete Q2, incomplete prism, pyramids).
fn from_geotrans(
  name: String,
  gt: &PGeoTrans,
  cvr: PRef,
  disc: bool,
  degree: usize,
) -> Result<PolyFem> {
  let n = gt.dim();
  let dof = if disc {
    DofDesc::lagrange_nonconforming(n)
  } else {
    DofDesc::lagrange(n)
  };
  PolyFem::new(
    name,
    cvr,
    vec![dof; gt.nb_points()],
    gt.geometric_nodes().to_vec(),
    gt.basis().to_vec(),
    degree,
  )
}

pub fn q2_incomplete(n: usize, disc: bool) -> Result<PFem> {
  let name = format!(
    "FEM_Q2_INCOMPLETE{}({n})",
    if disc { "_DISCONTINUOUS" } else { "" }
  );
  let gt = geotrans::q2_incomplete(n)?;
  cached(name, |name| {
    from_geotrans(name, &gt, cr::parallelepiped_of_reference(n, 1), disc, 2)
  })
}

pub fn prism_incomplete_p2(disc: bool) -> Result<PFem> {
  let name = format!(
    "FEM_PRISM_INCOMPLETE_P2{}",
    if disc { "_DISCONTINUOUS" } else { "" }
  );
  let gt = geotrans::prism_incomplete_p2();
  cached(name, |name| {
    from_geotrans(name, &gt, cr::prism_of_reference(3), disc, 2)
  })
}

pub fn pyramid_q2_incomplete(disc: bool) -> Result<PFem> {
  let name = format!(
    "FEM_PYRAMID_Q2_INCOMPLETE{}",
    if disc { "_DISCONTINUOUS" } else { "" }
  );
  let gt = geotrans::pyramid_q2_incomplete();
  cached(name, |name| {
    from_geotrans(name, &gt, cr::pyramid_qk_of_reference(1)?, disc, 2)
  })
}

pub fn pyramid_qk(k: usize, disc: Option<f64>) -> Result<PFem> {
  let name = match disc {
    None => format!("FEM_PYRAMID_QK({k})"),
    Some(0.0) => format!("FEM_PYRAMID_QK_DISCONTINUOUS({k})"),
    Some(a) => format!("FEM_PYRAMID_QK_DISCONTINUOUS({k},{})", fmt_g(a)),
  };
  if disc.is_some_and(|a| a != 0.0) {
    return Err(Error::NotImplemented("pyramid element with alpha"));
  }
  let cvr = cr::pyramid_qk_of_reference(1)?;
  cached(name, |name| match k {
    0 => {
      let dof = DofDesc::lagrange_0(3);
      PolyFem::new(
        name,
        cvr,
        vec![dof],
        vec![vec![0.0, 0.0, 0.5]],
        vec![Poly::one(3).into()],
        0,
      )
    }
    1 | 2 => from_geotrans(name, &geotrans::pyramid(k)?, cvr, disc.is_some(), k),
    _ => Err(Error::InvalidArgument(format!(
      "pyramid elements of degree {k}: 0, 1 or 2 only"
    ))),
  })
}

/// Element from its name, e.g. "FEM_PK(2,1)", "FEM_PRODUCT(FEM_PK(2,2),FEM_PK(1,2))".
pub fn fem_descriptor(name: &str) -> Result<PFem> {
  resolve(&Descriptor::parse(name)?)
}

fn alpha(d: &Descriptor) -> Result<(Vec<usize>, Option<f64>)> {
  let (ints, a) = match d.params.len() {
    3 => match d.params[2] {
      Param::Num(a) => (
        Descriptor {
          name: d.name.clone(),
          params: d.params[..2].to_vec(),
        }
        .ints(2)?,
        a,
      ),
      _ => return Err(Error::InvalidArgument(format!("{d}: bad alpha"))),
    },
    _ => (d.ints(2)?, 0.0),
  };
  Ok((ints, Some(a)))
}

fn resolve(d: &Descriptor) -> Result<PFem> {
  let s = d
    .suffix("FEM_")
    .ok_or_else(|| Error::InvalidArgument(format!("not an element: {d}")))?;
  let opt1 = |d: &Descriptor| -> Result<usize> {
    if d.params.is_empty() {
      Ok(2)
    } else {
      Ok(d.ints(1)?[0])
    }
  };
  match s {
    "PK" => {
      let a = d.ints(2)?;
      pk(a[0], a[1], None)
    }
    "PK_DISCONTINUOUS" => {
      let (a, al) = alpha(d)?;
      pk(a[0], a[1], al)
    }
    "QK" => {
      let a = d.ints(2)?;
      qk(a[0], a[1], None)
    }
    "QK_DISCONTINUOUS" => {
      let (a, al) = alpha(d)?;
      qk(a[0], a[1], al)
    }
    "PK_PRISM" | "PRISM_PK" => {
      let a = d.ints(2)?;
      prism_pk(a[0], a[1], None)
    }
    "PRISM_PK_DISCONTINUOUS" | "PK_PRISM_DISCONTINUOUS" => {
      let (a, al) = alpha(d)?;
      prism_pk(a[0], a[1], al)
    }
    "PRODUCT" => {
      let m = d.methods(2)?;
      product(&resolve(m[0])?, &resolve(m[1])?)
    }
    "Q2_INCOMPLETE" => q2_incomplete(opt1(d)?, false),
    "Q2_INCOMPLETE_DISCONTINUOUS" => q2_incomplete(opt1(d)?, true),
    "PRISM_INCOMPLETE_P2" => prism_incomplete_p2(false),
    "PRISM_INCOMPLETE_P2_DISCONTINUOUS" => prism_incomplete_p2(true),
    "PYRAMID_Q2_INCOMPLETE" => pyramid_q2_incomplete(false),
    "PYRAMID_Q2_INCOMPLETE_DISCONTINUOUS" => pyramid_q2_incomplete(true),
    // GetFEM: degree 2 by default
    "PYRAMID_QK" | "PYRAMID_LAGRANGE" if d.params.is_empty() => pyramid_qk(2, None),
    "PYRAMID_QK" | "PYRAMID_LAGRANGE" => pyramid_qk(d.ints(1)?[0], None),
    "PYRAMID_QK_DISCONTINUOUS" | "PYRAMID_DISCONTINUOUS_LAGRANGE" => {
      let k = match d.params.first() {
        Some(Param::Num(k)) => *k as usize,
        None => 2,
        _ => return Err(Error::InvalidArgument(format!("{d}: degree expected"))),
      };
      let a = match d.params.get(1) {
        Some(Param::Num(a)) => *a,
        _ => 0.0,
      };
      pyramid_qk(k, Some(a))
    }
    "SIMPLEX_IPK" | "SIMPLEX_CIPK" | "QUAD_IPK" | "PRISM_IPK" => {
      let a = d.ints(2)?;
      let cvr = match s {
        "QUAD_IPK" => cr::parallelepiped_of_reference(a[0], 1),
        "PRISM_IPK" => cr::prism_of_reference(a[0]),
        _ => cr::simplex_of_reference(a[0]),
      };
      fs::interior_pk(
        format!("FEM_{s}({},{})", a[0], a[1]),
        cvr,
        a[0],
        a[1],
        s == "SIMPLEX_CIPK",
      )
    }
    "PK_WITH_CUBIC_BUBBLE" => {
      let a = d.ints(2)?;
      fs::pk_with_cubic_bubble(a[0], a[1])
    }
    "P1_NONCONFORMING" => fs::p1_nonconforming(),
    "P1_BUBBLE_FACE" => fs::p1_bubble_face(d.ints(1)?[0]),
    "P1_BUBBLE_FACE_LAG" => fs::p1_bubble_face_lagrange(),
    "GEN_HIERARCHICAL" => {
      let m = d.methods(2)?;
      fs::gen_hierarchical(d.to_string(), &resolve(m[0])?, &resolve(m[1])?)
    }
    "PK_HIERARCHICAL" => {
      let a = d.ints(2)?;
      fs::pk_hierarchical(a[0], a[1])
    }
    "QK_HIERARCHICAL" => {
      let a = d.ints(2)?;
      fs::qk_hierarchical(a[0], a[1])
    }
    "PRISM_PK_HIERARCHICAL" | "PK_PRISM_HIERARCHICAL" => {
      let a = d.ints(2)?;
      fs::prism_pk_hierarchical(a[0], a[1])
    }
    "PK_GAUSSLOBATTO1D" => fs::pk_gauss_lobatto(d.ints(1)?[0]),
    "HERMITE" => crate::fem_transformed::hermite(d.ints(1)?[0]),
    "RTK" => {
      let a = d.ints(2)?;
      crate::fem_vector::rtk(a[0], a[1])
    }
    "RT0" => crate::fem_vector::rtk(d.ints(1)?[0], 0),
    "RT0Q" => crate::fem_vector::rt0q(d.ints(1)?[0]),
    "BDMK" => {
      let a = d.ints(2)?;
      crate::fem_vector::bdmk(a[0], a[1])
    }
    "NEDELEC" => crate::fem_vector::nedelec(d.ints(1)?[0]),
    "ARGYRIS" => crate::fem_transformed::argyris(),
    "MORLEY" => crate::fem_transformed::morley(d.ints(1)?[0]),
    "P1_PIECEWISE_LINEAR_BUBBLE" => {
      d.ints(0)?;
      fc::p1_piecewise_linear_bubble()
    }
    "HCT_TRIANGLE" | "REDUCED_HCT_TRIANGLE" => {
      d.ints(0)?;
      fc::hct_triangle(s == "REDUCED_HCT_TRIANGLE")
    }
    "QUADC1_COMPOSITE" | "REDUCED_QUADC1_COMPOSITE" => {
      d.ints(0)?;
      fc::quadc1(s == "REDUCED_QUADC1_COMPOSITE")
    }
    "STRUCTURED_COMPOSITE" => {
      let (m, k) = match &d.params[..] {
        [Param::Method(m), Param::Num(k)] if k.fract() == 0.0 && *k >= 1.0 => (m, *k as usize),
        _ => {
          return Err(Error::InvalidArgument(format!(
            "{d}: (element, k) expected"
          )));
        }
      };
      fc::structured_composite(&resolve(m)?, k)
    }
    "HHO" => {
      if d.params.len() < 2 || d.params.iter().any(|p| !matches!(p, Param::Method(_))) {
        return Err(Error::InvalidArgument(format!("{d}: elements expected")));
      }
      let ms: Vec<PFem> = d
        .params
        .iter()
        .map(|p| match p {
          Param::Method(m) => resolve(m),
          _ => unreachable!(),
        })
        .collect::<Result<_>>()?;
      let nbf = ms[0].ref_convex().structure().nb_faces();
      let faces = if ms.len() == 2 {
        vec![ms[1].clone(); nbf]
      } else {
        ms[1..].to_vec()
      };
      Ok(Arc::new(crate::fem_hho::HhoFem::new(
        d.to_string(),
        ms[0].clone(),
        faces,
      )?) as PFem)
    }
    "PK_HIERARCHICAL_COMPOSITE" | "PK_FULL_HIERARCHICAL_COMPOSITE" => {
      let a = d.ints(3)?;
      fc::pk_hierarchical_composite(a[0], a[1], a[2], s == "PK_FULL_HIERARCHICAL_COMPOSITE")
    }
    _ => Err(Error::InvalidArgument(format!(
      "unknown or not yet ported element {d}"
    ))),
  }
}

/// The Lagrange element of degree k for the elements of pgt (GetFEM
/// classical_fem / classical_discontinuous_fem).
pub fn classical_fem(pgt: &PGeoTrans, k: usize, disc: bool) -> Result<PFem> {
  classical_fem_alpha(pgt, k, disc.then_some(0.0))
}

/// classical_fem, discontinuous with GetFEM's alpha (the nodes moved
/// toward the barycenter) if al is some.
pub(crate) fn classical_fem_alpha(pgt: &PGeoTrans, k: usize, al: Option<f64>) -> Result<PFem> {
  use getfem_geom::convex_structure as cs;
  let basic = pgt.basic_structure();
  let n = basic.dim();
  let disc = al.is_some();
  if *basic == *cs::simplex_structure(n) {
    pk(n, k, al)
  } else if *basic == *cs::parallelepiped_structure(n, 1) {
    if k == 2 && (n == 2 || n == 3) && **pgt.structure() == *cs::q2_incomplete_structure(n)? {
      q2_incomplete(n, disc)
    } else {
      qk(n, k, al)
    }
  } else if n >= 2 && *basic == *cs::prism_p1_structure(n) {
    if k == 2 && n == 3 && **pgt.structure() == *cs::prism_incomplete_p2_structure() {
      prism_incomplete_p2(disc)
    } else {
      prism_pk(n, k, al)
    }
  } else if *basic == *cs::pyramid_qk_structure(1)? {
    if k == 2 && **pgt.structure() == *cs::pyramid_q2_incomplete_structure() {
      pyramid_q2_incomplete(disc)
    } else {
      pyramid_qk(k, al)
    }
  } else {
    Err(Error::InvalidArgument(format!(
      "no classical element for {}",
      pgt.name()
    )))
  }
}

/// Values, gradients and Hessians of the basis functions on a real
/// element with nodes g (N x nb_points of pgt) at reference point xref, as
/// GetFEM's fem_interpolation_context: val[i + nd*a], grad[i + nd*(a +
/// T*r)], hess[i + nd*(a + T*(r + N*s))] (a: component of vector elements,
/// mapped with K or B); the element matrix M_T applied last. As GetFEM, the
/// variation of K within the element is not differentiated, and the
/// curvature correction of Hessians uses the reference components.
/// Derivatives up to order der only (GetFEM computes each on demand); the
/// others empty.
pub fn real_base_values(
  pf: &PFem,
  pgt: &PGeoTrans,
  g: &Tensor<f64>,
  xref: &[f64],
  der: usize,
) -> Result<RealValues> {
  let pg = pgt.at(g, xref)?;
  real_base_values_at(pf, &FemCtx::new(pgt, g, xref, &pg), der)
}

/// real_base_values at the point of an element context.
pub fn real_base_values_at(pf: &PFem, fc: &FemCtx, der: usize) -> Result<RealValues> {
  if pf.real_element_defined() {
    return pf.real_values(fc, der);
  }
  let (pgt, g, xref) = (fc.pgt, fc.g, fc.xref);
  let (n, p) = (g.nrows(), pgt.dim());
  let pfp = fc
    .pim
    .map(|(pim, ip)| Ok::<_, crate::Error>((crate::precomp::fem_precomp(pf, pim)?, ip)))
    .transpose()?;
  let v_own;
  let v = match &pfp {
    Some((t, ip)) => &t.val[*ip],
    None => {
      v_own = pf.base_value(xref)?;
      &v_own
    }
  };
  let (nb, t) = (pf.nb_base(), pf.target_dim());
  // standard elements (GetFEM is_standard: scalar, equivalent): the
  // values, the gradients B g
  if der <= 1 && t == 1 && pf.is_equivalent() && matches!(pf.vectorial(), Vectorial::None) {
    let val = v.data().to_vec();
    if der == 0 {
      return Ok((val, vec![], vec![]));
    }
    let gr_own;
    let gr = match &pfp {
      Some((tb, ip)) => &tb.grad[*ip],
      None => {
        gr_own = pf.grad_base_value(xref)?;
        &gr_own
      }
    };
    let mut grad = vec![0.0; nb * n];
    for r in 0..n {
      for (i, gi) in grad[nb * r..nb * (r + 1)].iter_mut().enumerate() {
        *gi = (0..p)
          .map(|j| fc.b[r + n * j] * gr.data()[i + nb * j])
          .sum();
      }
    }
    return Ok((val, grad, vec![]));
  }
  // component map: identity, K or B (N x P)
  let (tt, cmap): (usize, Option<&[f64]>) = match pf.vectorial() {
    Vectorial::None => (t, None),
    Vectorial::Primal => (n, Some(fc.k)),
    Vectorial::Dual => (n, Some(fc.b)),
  };
  let comp = |src: &dyn Fn(usize, usize) -> f64, i: usize, a: usize| -> f64 {
    match cmap {
      None => src(i, a),
      Some(c) => (0..t).map(|b| c[a + n * b] * src(i, b)).sum(),
    }
  };
  let vr = |i: usize, b: usize| v.data()[i + nb * b];
  let mut val = vec![0.0; nb * tt];
  for i in 0..nb {
    for a in 0..tt {
      val[i + nb * a] = comp(&vr, i, a);
    }
  }
  let mt = pf.mat_trans(g, pgt)?;
  let apply = |t: &[f64], w: usize| -> Vec<f64> {
    let Some(m) = &mt else {
      return t.to_vec();
    };
    let nd = pf.nb_dof();
    let mut out = vec![0.0; nd * w];
    for c in 0..w {
      for i in 0..nd {
        out[i + nd * c] = (0..nb).map(|j| m[(i, j)] * t[j + nb * c]).sum();
      }
    }
    out
  };
  if der == 0 {
    return Ok((apply(&val, tt), vec![], vec![]));
  }
  let gr_own;
  let gr = match &pfp {
    Some((t, ip)) => &t.grad[*ip],
    None => {
      gr_own = pf.grad_base_value(xref)?;
      &gr_own
    }
  };
  // reference gradients mapped by B, components untransformed: g1(i,b,r)
  let g1: Vec<f64> = {
    let mut o = vec![0.0; nb * t * n];
    for i in 0..nb {
      for b in 0..t {
        for r in 0..n {
          o[i + nb * (b + t * r)] = (0..p)
            .map(|j| fc.b[r + n * j] * gr.data()[i + nb * (b + t * j)])
            .sum();
        }
      }
    }
    o
  };
  let mut grad = vec![0.0; nb * tt * n];
  for r in 0..n {
    let src = |i: usize, b: usize| g1[i + nb * (b + t * r)];
    for i in 0..nb {
      for a in 0..tt {
        grad[i + nb * (a + tt * r)] = comp(&src, i, a);
      }
    }
  }
  if der == 1 {
    return Ok((apply(&val, tt), apply(&grad, tt * n), vec![]));
  }
  let hr = pf.hess_base_value(xref)?;
  // second derivatives of the transformation: htau[l][j + p*k]
  let gh = pgt.hess(xref);
  let htau: Vec<Vec<f64>> = (0..n)
    .map(|l| {
      (0..p * p)
        .map(|jk| (0..g.ncols()).map(|a| g[(l, a)] * gh[(a, jk)]).sum())
        .collect()
    })
    .collect();
  let mut hess = vec![0.0; nb * tt * n * n];
  for i in 0..nb {
    for a in 0..tt {
      let href: Vec<f64> = (0..p * p)
        .map(|jk| {
          let src = |i: usize, b: usize| hr.data()[i + nb * (b + t * jk)];
          let h = comp(&src, i, a);
          // curvature correction with the reference component a (GetFEM)
          let ac = a.min(t - 1);
          h - (0..n)
            .map(|l| htau[l][jk] * g1[i + nb * (ac + t * l)])
            .sum::<f64>()
        })
        .collect();
      for r in 0..n {
        for s2 in 0..n {
          let mut h = 0.0;
          for j in 0..p {
            for k in 0..p {
              h += fc.b[r + n * j] * fc.b[s2 + n * k] * href[j + p * k];
            }
          }
          hess[i + nb * (a + tt * (r + n * s2))] = h;
        }
      }
    }
  }
  Ok((
    apply(&val, tt),
    apply(&grad, tt * n),
    apply(&hess, tt * n * n),
  ))
}
