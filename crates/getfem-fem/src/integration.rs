//! Integration methods (GetFEM's getfem_integration): exact polynomial
//! integration and cubature rules with points on the element and on each
//! face. Point order identical to GetFEM (matters for ImData).

use std::sync::Arc;

use getfem_geom::convex_ref::{self as cr, PRef};
use getfem_geom::convex_structure::PCvs;
use getfem_geom::geotrans::{self, PGeoTrans};
use getfem_geom::naming::Descriptor;
use getfem_geom::poly::{Polynomial, PowerIndex};
use getfem_linalg::tensor::{Lu, Mat, det};

use crate::cubature_data::CUBATURES;
use crate::{Error, Result};
use getfem_geom::intern::Interner;
use getfem_linalg::{vect_dist2, vect_norm2};

pub type PIntegration = Arc<IntegrationMethod>;

// ------------------------------------------------------ exact integration

/// Exact integration of polynomials on simplices and their products.
#[derive(Clone, Debug)]
pub enum PolyIntegration {
  Simplex(PCvs),
  Product(Box<PolyIntegration>, Box<PolyIntegration>, PCvs),
}

impl PolyIntegration {
  pub fn structure(&self) -> &PCvs {
    match self {
      PolyIntegration::Simplex(c) | PolyIntegration::Product(_, _, c) => c,
    }
  }

  pub fn dim(&self) -> usize {
    self.structure().dim()
  }

  /// Integral of x^power over the reference element.
  pub fn int_monomial(&self, power: &[usize]) -> f64 {
    match self {
      PolyIntegration::Simplex(c) => {
        let mut res = 1.0;
        let mut fa = 1usize;
        for &p in power {
          for k in 1..=p {
            res *= k as f64 / fa as f64;
            fa += 1;
          }
        }
        for _ in 0..c.dim() {
          res /= fa as f64;
          fa += 1;
        }
        res
      }
      PolyIntegration::Product(a, b, _) => {
        let n1 = a.dim();
        a.int_monomial(&power[..n1]) * b.int_monomial(&power[n1..])
      }
    }
  }

  /// Integral of x^power over face f.
  pub fn int_monomial_on_face(&self, power: &[usize], f: usize) -> f64 {
    match self {
      PolyIntegration::Simplex(c) => {
        if f != 0 && power[f - 1] != 0 {
          return 0.0;
        }
        let mut res = if f == 0 { (c.dim() as f64).sqrt() } else { 1.0 };
        let mut fa = 1usize;
        for &p in power {
          for k in 1..=p {
            res *= k as f64 / fa as f64;
            fa += 1;
          }
        }
        for _ in 1..c.dim() {
          res /= fa as f64;
          fa += 1;
        }
        res
      }
      PolyIntegration::Product(a, b, _) => {
        let n1 = a.dim();
        let nf1 = a.structure().nb_faces();
        let (p1, p2) = power.split_at(n1);
        if f < nf1 {
          a.int_monomial_on_face(p1, f) * b.int_monomial(p2)
        } else {
          a.int_monomial(p1) * b.int_monomial_on_face(p2, f - nf1)
        }
      }
    }
  }

  pub fn int_poly(&self, p: &Polynomial<f64>) -> f64 {
    let mut pi = PowerIndex::new(p.dim());
    let mut s = 0.0;
    for &c in p.coeffs() {
      if c != 0.0 {
        s += c * self.int_monomial(pi.powers());
      }
      pi.next();
    }
    s
  }

  pub fn int_poly_on_face(&self, p: &Polynomial<f64>, f: usize) -> f64 {
    let mut pi = PowerIndex::new(p.dim());
    let mut s = 0.0;
    for &c in p.coeffs() {
      if c != 0.0 {
        s += c * self.int_monomial_on_face(pi.powers(), f);
      }
      pi.next();
    }
    s
  }
}

// -------------------------------------------------- approximate integration

/// Cubature: points on the element first, then the points of each face.
#[derive(Clone, Debug)]
pub struct ApproxIntegration {
  cvr: PRef,
  pts: Vec<Vec<f64>>,
  w: Vec<f64>,
  /// rep[0] = points on the element; face f: rep[f]..rep[f+1]
  rep: Vec<usize>,
  /// per point: the id of the level set it is on (`LevelSet::id`; the lips
  /// of `MeshImLevelSet`), 0 none; empty: none (not GetFEM's)
  on: Vec<usize>,
}

impl ApproxIntegration {
  pub fn ref_convex(&self) -> &PRef {
    &self.cvr
  }

  pub fn dim(&self) -> usize {
    self.cvr.dim()
  }

  pub fn structure(&self) -> PCvs {
    self.cvr.structure().basic()
  }

  pub fn nb_points(&self) -> usize {
    self.w.len()
  }

  pub fn nb_points_on_convex(&self) -> usize {
    self.rep[0]
  }

  /// Number of faces of the reference convex.
  pub fn nb_convex_faces(&self) -> usize {
    self.rep.len() - 1
  }

  pub fn nb_points_on_face(&self, f: usize) -> usize {
    self.rep[f + 1] - self.rep[f]
  }

  pub fn ind_first_point_on_face(&self, f: usize) -> usize {
    self.rep[f]
  }

  pub fn points(&self) -> &[Vec<f64>] {
    &self.pts
  }

  pub fn point(&self, i: usize) -> &[f64] {
    &self.pts[i]
  }

  pub fn point_on_face(&self, f: usize, i: usize) -> &[f64] {
    &self.pts[self.rep[f] + i]
  }

  pub fn coeffs(&self) -> &[f64] {
    &self.w
  }

  pub fn coeff(&self, i: usize) -> f64 {
    self.w[i]
  }

  pub fn coeff_on_face(&self, f: usize, i: usize) -> f64 {
    self.w[self.rep[f] + i]
  }

  /// The id of the level set point i is on, 0 none: the side of
  /// Xfem_plus and Xfem_minus is forced on it only
  pub fn level_set_of_point(&self, i: usize) -> usize {
    self.on.get(i).copied().unwrap_or(0)
  }
}

/// Incremental construction with GetFEM's point merging (per face group).
struct Builder {
  cvr: PRef,
  groups: Vec<Vec<(Vec<f64>, f64)>>,
}

impl Builder {
  fn new(cvr: PRef) -> Self {
    let nf = cvr.structure().nb_faces();
    Builder {
      cvr,
      groups: vec![vec![]; nf + 1],
    }
  }

  fn find(&self, g: usize, ptref: &[f64]) -> Option<usize> {
    let grp = &self.groups[g];
    let r = grp
      .iter()
      .map(|(qref, _)| vect_norm2(qref))
      .fold(vect_norm2(ptref), f64::max)
      .max(1e-60);
    let eps = r * 1e4 * f64::EPSILON;
    grp
      .iter()
      .position(|(qref, _)| vect_dist2(qref, ptref) < eps)
  }

  /// face = None for the element itself; weights below 1e-15 are dropped.
  fn add_point(&mut self, ptref: &[f64], w: f64, face: Option<usize>) {
    if w.abs() <= 1e-15 {
      return;
    }
    let g = face.map_or(0, |f| f + 1);
    match self.find(g, ptref) {
      Some(i) => self.groups[g][i].1 += w,
      None => self.groups[g].push((ptref.to_vec(), w)),
    }
  }

  fn add_point_norepeat(&mut self, ptref: &[f64], w: f64) {
    if self.find(0, ptref).is_none() {
      self.add_point(ptref, w, None);
    }
  }

  /// Orbit of ptref under the symmetries of the simplex or of the cube.
  fn add_point_full_symmetric(&mut self, ptref: &[f64], w: f64) -> Result<()> {
    let s = self.cvr.structure().clone();
    let n = s.dim();
    if n + 1 == s.nb_faces() {
      let mut p3ref = ptref.to_vec();
      p3ref.push(1.0 - ptref.iter().sum::<f64>());
      let mut ind = vec![0usize; n];
      loop {
        let mut used = vec![false; n + 1];
        if ind.iter().all(|&k| !std::mem::replace(&mut used[k], true)) {
          let p2ref: Vec<f64> = ind.iter().map(|&k| p3ref[k]).collect();
          self.add_point_norepeat(&p2ref, w);
        }
        ind[0] += 1;
        let mut k = 0;
        while ind[k] == n + 1 {
          ind[k] = 0;
          k += 1;
          if k == n {
            return Ok(());
          }
          ind[k] += 1;
        }
      }
    } else if s.nb_points() == 1 << n {
      for i in 0..1usize << n {
        let p2ref: Vec<f64> = (0..n)
          .map(|k| {
            if i & (1 << k) != 0 {
              ptref[k]
            } else {
              1.0 - ptref[k]
            }
          })
          .collect();
        self.add_point_norepeat(&p2ref, w);
      }
      Ok(())
    } else {
      Err(Error::InvalidArgument(
        "full symmetric points only on simplices and parallelepipeds".into(),
      ))
    }
  }

  /// Map a (dim-1) method onto face f.
  fn add_method_on_face(&mut self, m: &ApproxIntegration, f: usize) {
    let dir = self.cvr.structure().ind_dir_points_of_face(f);
    let dpref: Vec<Vec<f64>> = dir.iter().map(|&i| self.cvr.points()[i].clone()).collect();
    let n = m.dim();
    let vecs: Vec<Vec<f64>> = (0..n)
      .map(|i| {
        dpref[i + 1]
          .iter()
          .zip(&dpref[0])
          .map(|(a, b)| a - b)
          .collect()
      })
      .collect();
    let mut scale = 1.0;
    if n > 0 {
      let a = Mat::from_data(&[n + 1, n], vecs.concat()).unwrap();
      scale = det(&a.tmatmul(&a)).unwrap().abs().sqrt();
    }
    for i in 0..m.nb_points_on_convex() {
      let mut ptref = dpref[0].clone();
      for (j, v) in vecs.iter().enumerate() {
        ptref
          .iter_mut()
          .zip(v)
          .for_each(|(a, b)| *a += b * m.point(i)[j]);
      }
      self.add_point(&ptref, m.coeff(i) * scale, Some(f));
    }
  }

  fn finish(self) -> ApproxIntegration {
    // rep[0] = interior count, rep[f + 1] = end of face f
    let (mut ptsref, mut w, mut rep) = (vec![], vec![], vec![]);
    for grp in self.groups {
      for (ptref, c) in grp {
        ptsref.push(ptref);
        w.push(c);
      }
      rep.push(ptsref.len());
    }
    ApproxIntegration {
      cvr: self.cvr,
      pts: ptsref,
      w,
      rep,
      on: vec![],
    }
  }
}

// ------------------------------------------------------------ the methods

#[derive(Debug)]
pub enum ImKind {
  None,
  Exact(PolyIntegration),
  Approx(ApproxIntegration),
}

#[derive(Debug)]
pub struct IntegrationMethod {
  name: String,
  kind: ImKind,
  /// point sets of one element (cut elements): not in the caches by
  /// name; the fem_precomp on them kept here instead
  transient: bool,
  pub(crate) fem_precomps: std::sync::Mutex<Vec<(String, Arc<crate::precomp::FemPrecomp>)>>,
}

impl PartialEq for IntegrationMethod {
  fn eq(&self, o: &Self) -> bool {
    self.name == o.name
  }
}

impl IntegrationMethod {
  pub fn name(&self) -> &str {
    &self.name
  }

  pub fn kind(&self) -> &ImKind {
    &self.kind
  }

  pub fn approx(&self) -> Result<&ApproxIntegration> {
    match &self.kind {
      ImKind::Approx(a) => Ok(a),
      _ => Err(Error::InvalidArgument(format!(
        "{} is not an approximate integration method",
        self.name
      ))),
    }
  }

  pub fn exact(&self) -> Result<&PolyIntegration> {
    match &self.kind {
      ImKind::Exact(e) => Ok(e),
      _ => Err(Error::InvalidArgument(format!(
        "{} is not an exact integration method",
        self.name
      ))),
    }
  }

  pub fn is_none(&self) -> bool {
    matches!(self.kind, ImKind::None)
  }

  /// A point set of one evaluation (not to be tabulated).
  pub fn is_transient(&self) -> bool {
    self.transient
  }
}

/// Points with unit weights on a reference convex: `interior`, then the
/// points of each face (GetFEM's interpolation contexts evaluate at such
/// points). Transient sets are not tabulated by the precomputations.
pub fn point_set(
  name: String,
  cvr: &PRef,
  interior: Vec<Vec<f64>>,
  faces: Vec<Vec<Vec<f64>>>,
  transient: bool,
) -> PIntegration {
  let unit = |v: Vec<Vec<f64>>| v.into_iter().map(|ptref| (ptref, 1.0)).collect::<Vec<_>>();
  let faces = faces.into_iter().map(unit).collect();
  approx_integration(name, cvr, unit(interior), faces, transient)
}

/// A cubature given by its points and weights on a reference convex:
/// `interior`, then the points of each face (close points not merged).
/// Transient methods are not tabulated by the precomputations; others are
/// cached by name.
pub fn approx_integration(
  name: String,
  cvr: &PRef,
  interior: Vec<(Vec<f64>, f64)>,
  faces: Vec<Vec<(Vec<f64>, f64)>>,
  transient: bool,
) -> PIntegration {
  approx_integration_on(name, cvr, interior, faces, vec![], transient)
}

/// `approx_integration` with the level set of each point (see
/// `ApproxIntegration::level_set_of_point`; empty: none)
pub fn approx_integration_on(
  name: String,
  cvr: &PRef,
  interior: Vec<(Vec<f64>, f64)>,
  faces: Vec<Vec<(Vec<f64>, f64)>>,
  on: Vec<usize>,
  transient: bool,
) -> PIntegration {
  let key = name.clone();
  let build = move || {
    let mut rep = vec![interior.len()];
    let (mut ptsref, mut w): (Vec<_>, Vec<_>) = interior.into_iter().unzip();
    for f in faces {
      for (ptref, c) in f {
        ptsref.push(ptref);
        w.push(c);
      }
      rep.push(ptsref.len());
    }
    rep.resize(cvr.structure().nb_faces() + 1, ptsref.len());
    IntegrationMethod {
      name,
      kind: ImKind::Approx(ApproxIntegration {
        cvr: cvr.clone(),
        pts: ptsref,
        w,
        rep,
        on,
      }),
      transient,
      fem_precomps: Default::default(),
    }
  };
  if transient {
    return Arc::new(build());
  }
  CACHE.get_or(&key, build)
}

static CACHE: Interner<String, IntegrationMethod> = Interner::new();

/// Integration method from its name, e.g. "IM_TRIANGLE(6)".
pub fn integ_descriptor(name: &str) -> Result<PIntegration> {
  resolve(&Descriptor::parse(name)?)
}

fn named(name: String, build: impl FnOnce() -> Result<ImKind>) -> Result<PIntegration> {
  CACHE.get_or_try(&name.clone(), || {
    Ok(IntegrationMethod {
      name,
      kind: build()?,
      transient: false,
      fem_precomps: Default::default(),
    })
  })
}

fn resolve(d: &Descriptor) -> Result<PIntegration> {
  let s = d
    .suffix("IM_")
    .ok_or_else(|| Error::InvalidArgument(format!("not an integration method: {d}")))?;
  let name = d.to_string();
  // tabulated rules first, as GetFEM (e.g. IM_GAUSS1D(odd) from the table)
  if let Some(c) = CUBATURES.iter().find(|c| c.name == name) {
    return named(name, || tabulated(c).map(ImKind::Approx));
  }
  match s {
    "NONE" => {
      d.ints(0)?;
      named(name, || Ok(ImKind::None))
    }
    "EXACT_SIMPLEX" => {
      let n = d.ints(1)?[0];
      if n == 0 {
        return Err(Error::InvalidArgument("IM_EXACT_SIMPLEX(0)".into()));
      }
      named(name, || {
        Ok(ImKind::Exact(PolyIntegration::Simplex(
          getfem_geom::convex_structure::simplex_structure(n),
        )))
      })
    }
    "EXACT_PARALLELEPIPED" => {
      let n = d.ints(1)?[0];
      match n {
        0 => Err(Error::InvalidArgument("IM_EXACT_PARALLELEPIPED(0)".into())),
        1 => integ_descriptor("IM_EXACT_SIMPLEX(1)"),
        _ => integ_descriptor(&format!(
          "IM_PRODUCT(IM_EXACT_PARALLELEPIPED({}),IM_EXACT_SIMPLEX(1))",
          n - 1
        )),
      }
    }
    "EXACT_PRISM" => {
      let n = d.ints(1)?[0];
      if n < 2 {
        return Err(Error::InvalidArgument("IM_EXACT_PRISM(n), n >= 2".into()));
      }
      integ_descriptor(&format!(
        "IM_PRODUCT(IM_EXACT_SIMPLEX({}),IM_EXACT_SIMPLEX(1))",
        n - 1
      ))
    }
    "PRODUCT" => {
      let m = d.methods(2)?;
      let (a, b) = (resolve(m[0])?, resolve(m[1])?);
      let name = format!("IM_PRODUCT({},{})", a.name, b.name);
      match (&a.kind, &b.kind) {
        (ImKind::Exact(x), ImKind::Exact(y)) => {
          let cvs =
            getfem_geom::convex_structure::convex_product_structure(x.structure(), y.structure());
          named(name, || {
            Ok(ImKind::Exact(PolyIntegration::Product(
              Box::new(x.clone()),
              Box::new(y.clone()),
              cvs,
            )))
          })
        }
        (ImKind::Approx(x), ImKind::Approx(y)) => named(name, || Ok(ImKind::Approx(product(x, y)))),
        _ => Err(Error::InvalidArgument(format!(
          "{name}: cannot mix exact and approximate methods"
        ))),
      }
    }
    "GAUSS1D" => {
      let k = d.ints(1)?[0];
      if k % 2 == 1 {
        return integ_descriptor(&format!("IM_GAUSS1D({})", k - 1));
      }
      named(name, || Ok(ImKind::Approx(gauss1d(k / 2 + 1))))
    }
    "NC" => {
      let a = d.ints(2)?;
      named(name, || newton_cotes(a[0], a[1]).map(ImKind::Approx))
    }
    "NC_PARALLELEPIPED" => {
      let a = d.ints(2)?;
      match a[0] {
        0 => Err(Error::InvalidArgument("IM_NC_PARALLELEPIPED(0,k)".into())),
        1 => integ_descriptor(&format!("IM_NC(1,{})", a[1])),
        n => integ_descriptor(&format!(
          "IM_PRODUCT(IM_NC_PARALLELEPIPED({},{}),IM_NC(1,{}))",
          n - 1,
          a[1],
          a[1]
        )),
      }
    }
    "NC_PRISM" => {
      let a = d.ints(2)?;
      if a[0] < 2 {
        return Err(Error::InvalidArgument("IM_NC_PRISM(n,k), n >= 2".into()));
      }
      integ_descriptor(&format!(
        "IM_PRODUCT(IM_NC({},{}),IM_NC(1,{}))",
        a[0] - 1,
        a[1],
        a[1]
      ))
    }
    "GAUSS_PARALLELEPIPED" => {
      let a = d.ints(2)?;
      match a[0] {
        0 => Err(Error::InvalidArgument(
          "IM_GAUSS_PARALLELEPIPED(0,k)".into(),
        )),
        1 => integ_descriptor(&format!("IM_GAUSS1D({})", a[1])),
        n => integ_descriptor(&format!(
          "IM_PRODUCT(IM_GAUSS_PARALLELEPIPED({},{}),IM_GAUSS1D({}))",
          n - 1,
          a[1],
          a[1]
        )),
      }
    }
    "PYRAMID" => {
      let m = d.methods(1)?;
      let base = resolve(m[0])?;
      let name = format!("IM_PYRAMID({})", base.name);
      named(name, || pyramid(base.approx()?).map(ImKind::Approx))
    }
    "QUASI_POLAR" => {
      use getfem_geom::naming::Param;
      // the vertex numbers: floor(x + 0.01)
      let ip = |x: &f64| (x + 0.01).floor().max(0.0) as usize;
      let (m, ip1, ip2) = match &d.params[..] {
        [Param::Method(m)] => (m, 0, 0),
        [Param::Method(m), Param::Num(a)] => (m, ip(a), ip(a)),
        [Param::Method(m), Param::Num(a), Param::Num(b)] => (m, ip(a), ip(b)),
        _ => {
          return Err(Error::InvalidArgument(format!(
            "{d}: bad parameters for quasi polar integration"
          )));
        }
      };
      let base = resolve(m)?;
      let cube = base.approx()?.dim() == 3
        && *base.approx()?.structure()
          == *getfem_geom::convex_structure::parallelepiped_structure(3, 1);
      if cube != (d.params.len() == 1) {
        return Err(Error::InvalidArgument(format!(
          "{d}: bad number of parameters"
        )));
      }
      let name = format!("IM_QUASI_POLAR({},{ip1},{ip2})", base.name);
      named(name, || {
        quasi_polar(base.approx()?, ip1, ip2).map(ImKind::Approx)
      })
    }
    "STRUCTURED_COMPOSITE" => {
      let (m, k) = match &d.params[..] {
        [
          getfem_geom::naming::Param::Method(m),
          getfem_geom::naming::Param::Num(k),
        ] if k.fract() == 0.0 && *k >= 1.0 => (m, *k as usize),
        _ => return Err(Error::InvalidArgument(format!("{d}: (method, k) expected"))),
      };
      let base = resolve(m)?;
      let name = format!("IM_STRUCTURED_COMPOSITE({},{k})", base.name);
      named(name, || {
        let a = base.approx()?;
        let sub = crate::fem_composite::structured_mesh_precomposite(a.ref_convex(), k)?;
        composite(&sub, a, a.ref_convex().clone()).map(ImKind::Approx)
      })
    }
    "HCT_COMPOSITE" | "QUADC1_COMPOSITE" | "PYRAMID_COMPOSITE" => {
      let base = resolve(d.methods(1)?[0])?;
      let name = format!("IM_{s}({})", base.name);
      named(name, || {
        use crate::fem_composite as fc;
        let a = base.approx()?;
        let (sub, cvr) = match s {
          "HCT_COMPOSITE" => (fc::hct_mesh()?, a.ref_convex().clone()),
          "QUADC1_COMPOSITE" => (fc::quadc1_mesh()?, cr::parallelepiped_of_reference(2, 1)),
          _ => (fc::pyramid_mesh()?, cr::pyramid_qk_of_reference(1)?),
        };
        composite(&sub, a, cvr).map(ImKind::Approx)
      })
    }
    _ => Err(Error::InvalidArgument(format!(
      "unknown integration method {name}"
    ))),
  }
}

/// The method a on each sub-element, faces on the faces of cvr (GetFEM
/// composite_approx_int_method).
fn composite(
  sub: &crate::fem_composite::MeshPrecomposite,
  a: &ApproxIntegration,
  cvr: PRef,
) -> Result<ApproxIntegration> {
  let m = &sub.mesh;
  let mut b = Builder::new(cvr.clone());
  for (i, cv) in m.convex_index().enumerate() {
    let (pgt, g) = (m.trans_of_convex(cv), m.points_of_convex(cv));
    if *pgt.convex_ref().structure().basic() != *a.structure() {
      return Err(Error::InvalidArgument(
        "composite method: the method does not fit the sub-elements".into(),
      ));
    }
    let (bm, det) = (&sub.b[i], sub.det[i].abs());
    for j in 0..a.nb_points_on_convex() {
      b.add_point(&pgt.transform(a.point(j), &g)?, a.coeff(j) * det, None);
    }
    for f in 0..pgt.structure().nb_faces() {
      let ip = m.ind_points_of_face_of_convex(cv, f);
      let bary: Vec<f64> = (0..m.dim())
        .map(|d| ip.iter().map(|&p| m.point(p)[d]).sum::<f64>() / ip.len() as f64)
        .collect();
      let Some(f2) = (0..cvr.structure().nb_faces()).find(|&f3| {
        cvr
          .is_in_face(f3, &bary)
          .map(|v| v.abs() < 1e-7)
          .unwrap_or(false)
      }) else {
        continue;
      };
      let nr = &pgt.normals()[f];
      let w: Vec<f64> = (0..bm.nrows())
        .map(|r| (0..bm.ncols()).map(|c| bm[(r, c)] * nr[c]).sum())
        .collect();
      let mul = vect_norm2(&w) * det;
      for j in 0..a.nb_points_on_face(f) {
        b.add_point(
          &pgt.transform(a.point_on_face(f, j), &g)?,
          a.coeff_on_face(f, j) * mul,
          Some(f2),
        );
      }
    }
  }
  Ok(b.finish())
}

fn tabulated(c: &crate::cubature_data::CubatureData) -> Result<ApproxIntegration> {
  let cvr = if c.geotrans == "GT_PK(0,0)" {
    cr::simplex_of_reference(0)
  } else {
    geotrans::geotrans_descriptor(c.geotrans)?
      .convex_ref()
      .clone()
  };
  let n = cvr.dim();
  let mut b = Builder::new(cvr);
  for (ty, v) in c.nodes {
    let (ptref, w) = (&v[..n], v[n]);
    match ty {
      0 => b.add_point(ptref, w, None),
      1 => b.add_point_full_symmetric(ptref, w)?,
      2 => {
        let mut perm: Vec<usize> = (0..n).collect();
        loop {
          let p2ref: Vec<f64> = perm.iter().map(|&i| ptref[i]).collect();
          b.add_point_full_symmetric(&p2ref, w)?;
          if !next_permutation(&mut perm) {
            break;
          }
        }
      }
      _ => {
        return Err(Error::InvalidArgument(format!(
          "{}: bad node type {ty}",
          c.name
        )));
      }
    }
  }
  for (f, fm) in c.faces.iter().enumerate() {
    if n > 0 {
      b.add_method_on_face(integ_descriptor(fm)?.approx()?, f);
    }
  }
  Ok(b.finish())
}

/// Lexicographic next permutation (bgeot::permutation order).
fn next_permutation(p: &mut [usize]) -> bool {
  let n = p.len();
  if n < 2 {
    return false;
  }
  let Some(i) = (0..n - 1).rev().find(|&i| p[i] < p[i + 1]) else {
    return false;
  };
  let j = (i + 1..n).rev().find(|&j| p[j] > p[i]).unwrap();
  p.swap(i, j);
  p[i + 1..].reverse();
  true
}

/// Legendre P_n and P_{n-1} at x by the three-term recurrence.
fn legendre(n: usize, x: f64) -> (f64, f64) {
  let (mut p0, mut p1) = (1.0, x);
  if n == 0 {
    return (1.0, 0.0);
  }
  for k in 2..=n {
    let kf = k as f64;
    let p2 = ((2.0 * kf - 1.0) * x * p1 - (kf - 1.0) * p0) / kf;
    (p0, p1) = (p1, p2);
  }
  (p1, p0)
}

/// Gauss-Legendre with n points on [0, 1]; face points 1 (face 0), 0 (face 1).
fn gauss1d(n: usize) -> ApproxIntegration {
  // roots by bisection between consecutive roots of P_{n-1}
  let mut roots: Vec<f64> = vec![0.0];
  for m in 2..=n {
    let mut r = vec![0.0; m];
    let mut a = -1.0;
    for k in 0..m / 2 {
      let b = roots[k];
      let (mut c, mut d) = (a, b);
      let mut cv = legendre(m, c).0;
      for _ in 0..200 {
        let e = 0.5 * (c + d);
        if e <= c || e >= d {
          break;
        }
        let ev = legendre(m, e).0;
        if ev * cv < 0.0 {
          d = e;
        } else {
          (c, cv) = (e, ev);
        }
      }
      r[k] = c;
      r[m - k - 1] = -c;
      a = b;
    }
    roots = r;
  }
  let mut ptsref = vec![];
  let mut w = vec![];
  let nf = n as f64;
  for &x0 in roots.iter().take(n) {
    // Newton polish and w = 1/((1-x^2) P_n'^2)
    let mut x = x0;
    for _ in 0..3 {
      let (p, pm1) = legendre(n, x);
      let dp = nf * (pm1 - x * p) / (1.0 - x * x);
      if dp == 0.0 {
        break;
      }
      x -= p / dp;
    }
    let (p, pm1) = legendre(n, x);
    let dp = nf * (pm1 - x * p) / (1.0 - x * x);
    ptsref.push(vec![0.5 + 0.5 * x]);
    w.push(1.0 / ((1.0 - x * x) * dp * dp));
  }
  ptsref.push(vec![1.0]);
  w.push(1.0);
  ptsref.push(vec![0.0]);
  w.push(1.0);
  ApproxIntegration {
    cvr: cr::simplex_of_reference(1),
    pts: ptsref,
    w,
    rep: vec![n, n + 1, n + 2],
    on: vec![],
  }
}

fn newton_cotes(n: usize, k: usize) -> Result<ApproxIntegration> {
  let cvr = cr::simplex_of_reference(n);
  let mut b = Builder::new(cvr);
  if n == 0 {
    b.add_point(&[], 1.0, None);
    return Ok(b.finish());
  }
  let exact = PolyIntegration::Simplex(getfem_geom::convex_structure::simplex_structure(n));
  // nodes: the degree-k lattice (barycenter for k = 0); monomials of degree <= k
  let nodesref: Vec<Vec<f64>> = if k == 0 {
    vec![vec![1.0 / (n + 1) as f64; n]]
  } else {
    cr::simplex_of_reference_k(n, k).points().to_vec()
  };
  let r = nodesref.len();
  let mut pi = PowerIndex::new(n);
  let mut base = vec![];
  for _ in 0..r {
    base.push(pi.clone());
    pi.next();
  }
  let mut m = Mat::zeros(&[r, r]);
  let f: Vec<f64> = base
    .iter()
    .map(|p| exact.int_monomial(p.powers()))
    .collect();
  for (i, p) in base.iter().enumerate() {
    for (q, xref) in nodesref.iter().enumerate() {
      m[(i, q)] = p
        .powers()
        .iter()
        .zip(xref)
        .map(|(&e, &v)| v.powi(e as i32))
        .product();
    }
  }
  let u = Lu::new(&m)?.solve(&f);
  for (xref, &wq) in nodesref.iter().zip(&u) {
    b.add_point(xref, wq, None);
  }
  let face = integ_descriptor(&format!("IM_NC({},{})", n - 1, k))?;
  for fc in 0..n + 1 {
    b.add_method_on_face(face.approx()?, fc);
  }
  Ok(b.finish())
}

/// (points, weights) of the element (face None) or of face f.
fn group(m: &ApproxIntegration, f: Option<usize>) -> (Vec<Vec<f64>>, Vec<f64>) {
  let (a, b) = match f {
    None => (0, m.rep[0]),
    Some(f) => (m.rep[f], m.rep[f + 1]),
  };
  (m.pts[a..b].to_vec(), m.w[a..b].to_vec())
}

/// Tensor product of rules, first factor fastest.
fn tensor(
  out: &mut (Vec<Vec<f64>>, Vec<f64>),
  a: &(Vec<Vec<f64>>, Vec<f64>),
  b: &(Vec<Vec<f64>>, Vec<f64>),
) {
  for (pb, wb) in b.0.iter().zip(&b.1) {
    for (pa, wa) in a.0.iter().zip(&a.1) {
      out.0.push([pa.as_slice(), pb.as_slice()].concat());
      out.1.push(wa * wb);
    }
  }
}

fn product(a: &ApproxIntegration, b: &ApproxIntegration) -> ApproxIntegration {
  let cvr = cr::convex_ref_product(&a.cvr, &b.cvr);
  let (ca, cb) = (group(a, None), group(b, None));
  let mut out = (vec![], vec![]);
  tensor(&mut out, &ca, &cb);
  let mut rep = vec![out.0.len()];
  for f in 0..a.cvr.structure().nb_faces() {
    tensor(&mut out, &group(a, Some(f)), &cb);
    rep.push(out.0.len());
  }
  for f in 0..b.cvr.structure().nb_faces() {
    tensor(&mut out, &ca, &group(b, Some(f)));
    rep.push(out.0.len());
  }
  ApproxIntegration {
    cvr,
    pts: out.0,
    w: out.1,
    rep,
    on: vec![],
  }
}

/// IM_PYRAMID(base on the cube): the cube mapped onto the pyramid by the
/// degenerate Q1 map with the top face collapsed to the apex (the
/// quasi-polar construction, pyramid case).
fn pyramid(base: &ApproxIntegration) -> Result<ApproxIntegration> {
  if *base.structure() != *getfem_geom::convex_structure::parallelepiped_structure(3, 1) {
    return Err(Error::InvalidArgument(
      "IM_PYRAMID needs a method on the cube".into(),
    ));
  }
  let q1: PGeoTrans = geotrans::qk(3, 1)?;
  let apex = [0.0, 0.0, 1.0];
  let n1: Vec<[f64; 3]> = vec![
    [-1., -1., 0.],
    [1., -1., 0.],
    [-1., 1., 0.],
    [1., 1., 0.],
    apex,
    apex,
    apex,
    apex,
  ];
  let gref = Mat::from_data(&[3, 8], n1.concat()).unwrap();
  let pyr = cr::pyramid_qk_of_reference(1)?;
  let mut b = Builder::new(pyr.clone());
  for i in 0..base.nb_points() {
    let fp = (0..base.cvr.structure().nb_faces())
      .find(|&f| i >= base.rep[f] && i < base.rep[f + 1])
      .filter(|_| i >= base.rep[0]);
    let ptref = base.point(i);
    let xref = q1.transform(ptref, &gref)?;
    let k = gref.matmul(&q1.grad(ptref));
    let mut j1 = det(&k)?.abs();
    match fp {
      None => b.add_point(&xref, base.coeff(i) * j1, None),
      Some(f) if j1 > 1e-10 => {
        // |K^-T n| scales the face measure
        let nr = &base.cvr.normals()[f];
        let kinv = getfem_linalg::tensor::inverse(&k)?;
        let n: Vec<f64> = (0..3)
          .map(|r| (0..3).map(|c| kinv[(c, r)] * nr[c]).sum())
          .collect();
        j1 *= vect_norm2(&n);
        let faces: Vec<usize> = (0..5)
          .filter(|&ff| {
            pyr
              .is_in_face(ff, &xref)
              .map(|v| v.abs() < 1e-8)
              .unwrap_or(false)
          })
          .collect();
        if faces.len() > 1 {
          return Err(Error::InvalidArgument(
            "IM_PYRAMID: an integration point is common to two faces".into(),
          ));
        }
        if let Some(&ff) = faces.first() {
          b.add_point(&xref, base.coeff(i) * j1, Some(ff));
        }
      }
      _ => {}
    }
  }
  Ok(b.finish())
}

/// IM_QUASI_POLAR(base, ip1[, ip2]): a rule on the reference simplex
/// singular at its vertex ip1 (2D: base on the square; 3D: on the prism,
/// singular along the edge ip1-ip2 or at the vertex ip1 = ip2, or on the
/// tetrahedron, at ip1); a base on the cube gives the pyramid rule. The
/// base is mapped by a degenerate Q1 transformation onto a reference
/// simplex, then onto the simplex with ip1 (and ip2) at the collapsed
/// nodes.
fn quasi_polar(base: &ApproxIntegration, ip1: usize, ip2: usize) -> Result<ApproxIntegration> {
  use getfem_geom::convex_structure as cs;
  #[derive(PartialEq)]
  enum What {
    Square,
    Prism,
    TetraCyl,
    Prism2,
  }
  let n = base.dim();
  let st = base.structure().clone();
  if n == 3 && *st == *cs::parallelepiped_structure(3, 1) {
    return pyramid(base);
  }
  let what = if n == 2 {
    What::Square
  } else if n == 3 && *st == *cs::prism_p1_structure(3) {
    if ip1 == ip2 {
      What::Prism2
    } else {
      What::Prism
    }
  } else if n == 3 && *st == *cs::simplex_structure(3) {
    What::TetraCyl
  } else {
    return Err(Error::InvalidArgument(
      "IM_QUASI_POLAR: incoherent integration method".into(),
    ));
  };
  if ip1 > n || ip2 > n {
    return Err(Error::InvalidArgument(
      "IM_QUASI_POLAR: bad parameters".into(),
    ));
  }
  // the degenerate Q1 map (nodes1), the simplex (nodes2), and for the
  // tetrahedron the simplex of its two halves (nodes3)
  let q1: PGeoTrans = geotrans::qk(n, 1)?;
  let mut nodes1: Vec<Vec<f64>> = q1.convex_ref().points().to_vec();
  let mut nodes2 = vec![vec![0.0; n]; n + 1];
  let mut nodes3 = vec![vec![0.0; n]; n + 1];
  let mut ip2 = ip2;
  let mut other: Vec<usize> = match what {
    What::Square => {
      nodes1[3] = nodes1[1].clone();
      nodes2[ip1] = nodes1[1].clone();
      ip2 = ip1;
      vec![0, 2]
    }
    What::Prism => {
      nodes1[4] = nodes1[0].clone();
      nodes1[5] = nodes1[1].clone();
      nodes2[ip1] = nodes1[0].clone();
      nodes2[ip2] = nodes1[1].clone();
      vec![2, 6]
    }
    What::TetraCyl => {
      nodes3 = vec![
        nodes1[0].clone(),
        nodes1[1].clone(),
        nodes1[2].clone(),
        nodes1[4].clone(),
      ];
      nodes2[ip1] = nodes1[1].clone();
      ip2 = ip1;
      vec![0, 2, 4]
    }
    What::Prism2 => {
      nodes2[ip1] = nodes1[4].clone();
      vec![0, 1, 2]
    }
  };
  for (i, nd) in nodes2.iter_mut().enumerate() {
    if i != ip1 && i != ip2 {
      *nd = nodes1[other.pop().unwrap()].clone();
    }
  }
  let g1 = Mat::from_data(&[n, nodes1.len()], nodes1.concat())?;
  // the simplex map x = n2_0 + K2 xref
  let mut k2 = Mat::zeros(&[n, n]);
  for k in 0..n {
    for r in 0..n {
      k2[(r, k)] = nodes2[k + 1][r] - nodes2[0][r];
    }
  }
  let (k2inv, j2) = (getfem_linalg::tensor::inverse(&k2)?, det(&k2)?.abs());
  let simplex = cr::simplex_of_reference(n);
  let mut b = Builder::new(simplex.clone());
  let mut normal2 = vec![0.0; n];
  let halves = if what == What::TetraCyl { 2 } else { 1 };
  for nc in 0..halves {
    let (mut k3, mut j3) = (Mat::identity(n), 1.0);
    if what == What::TetraCyl {
      if nc == 1 {
        nodes3[0] = nodes1[3].clone();
      }
      // K3 = (dX3/dP)^-T
      let mut d3 = Mat::zeros(&[n, n]);
      for k in 0..n {
        for r in 0..n {
          d3[(k, r)] = nodes3[k + 1][r] - nodes3[0][r];
        }
      }
      j3 = det(&d3)?.abs();
      k3 = getfem_linalg::tensor::inverse(&d3)?;
    }
    for i in 0..base.nb_points() {
      let mut k4 = Mat::identity(n);
      let mut j4 = 1.0;
      let fp = (0..base.cvr.structure().nb_faces())
        .find(|&f| i >= base.rep[f] && i < base.rep[f + 1])
        .filter(|_| i >= base.rep[0]);
      let mut normal1 = fp
        .map(|f| base.cvr.normals()[f].clone())
        .unwrap_or_default();
      let mut p = base.point(i).to_vec();
      if what == What::TetraCyl {
        let q = p.clone();
        p = (0..n)
          .map(|r| {
            nodes3[0][r]
              + (0..n)
                .map(|k| (nodes3[k + 1][r] - nodes3[0][r]) * q[k])
                .sum::<f64>()
          })
          .collect();
        let (x, y, omz) = (p[0], p[1], 1.0 - p[2]);
        k4[(0, 1)] = -y / omz;
        k4[(1, 1)] = 1.0 - x / omz;
        k4[(2, 1)] = -x * y / (omz * omz);
        j4 = det(&k4)?.abs();
        p[1] *= 1.0 - x / omz;
      }
      if what == What::Prism2 {
        let (x, y, omz) = (p[0], p[1], 1.0 - p[2]);
        k4[(0, 0)] = omz;
        k4[(2, 0)] = -x;
        k4[(1, 1)] = omz;
        k4[(2, 1)] = -y;
        j4 = det(&k4)?.abs();
        p[0] *= omz;
        p[1] *= omz;
      }
      let p1 = q1.transform(&p, &g1)?;
      // K = (dP1/dP)^T
      let k = g1.matmul(&q1.grad(&p)).transposed();
      let mut j1 = det(&k)?.abs() * j3 * j4;
      if fp.is_some() && j1 > 1e-10 && j4 > 1e-10 {
        if what == What::TetraCyl {
          normal1 = mat_vec(&k3, &normal1);
        }
        let n2 = mat_vec(&getfem_linalg::tensor::inverse(&k4)?, &normal1);
        normal2 = mat_vec(&getfem_linalg::tensor::inverse(&k)?, &n2);
        let nn = vect_norm2(&normal2);
        j1 *= nn;
        normal2.iter_mut().for_each(|v| *v /= nn);
      }
      let d: Vec<f64> = (0..n).map(|r| p1[r] - nodes2[0][r]).collect();
      let p2 = mat_vec(&k2inv, &d);
      if simplex.is_in(&p2)? >= 1e-8 {
        return Err(Error::InvalidArgument(format!(
          "IM_QUASI_POLAR: point not in the convex ref: {p2:?}"
        )));
      }
      if i < base.nb_points_on_convex() {
        b.add_point(&p2, base.coeff(i) * j1 / j2, None);
      } else if j1 > 1e-10 {
        let faces: Vec<usize> = (0..n + 1)
          .filter(|&f| simplex.is_in_face(f, &p2).is_ok_and(|v| v.abs() < 1e-8))
          .collect();
        if faces.len() > 1 {
          return Err(Error::InvalidArgument(
            "IM_QUASI_POLAR: an integration point is common to two faces".into(),
          ));
        }
        if let Some(&f) = faces.first() {
          // K2^T normal2
          let n1: Vec<f64> = (0..n)
            .map(|r| (0..n).map(|c| k2[(c, r)] * normal2[c]).sum())
            .collect();
          b.add_point(&p2, base.coeff(i) * j1 * vect_norm2(&n1) / j2, Some(f));
        }
      }
    }
  }
  Ok(b.finish())
}

fn mat_vec(a: &Mat, v: &[f64]) -> Vec<f64> {
  (0..a.nrows())
    .map(|r| (0..a.ncols()).map(|c| a[(r, c)] * v[c]).sum())
    .collect()
}

/// Exact method for the structure of pgt (simplex, parallelepiped, prism).
pub fn classical_exact_im(pgt: &PGeoTrans) -> Result<PIntegration> {
  let cvs = pgt.basic_structure();
  let n = cvs.dim();
  use getfem_geom::convex_structure as cs;
  if *cvs == *cs::simplex_structure(n) {
    integ_descriptor(&format!("IM_EXACT_SIMPLEX({n})"))
  } else if *cvs == *cs::parallelepiped_structure(n, 1) {
    integ_descriptor(&format!("IM_EXACT_PARALLELEPIPED({n})"))
  } else if n >= 2 && *cvs == *cs::prism_p1_structure(n) {
    integ_descriptor(&format!("IM_EXACT_PRISM({n})"))
  } else {
    Err(Error::InvalidArgument(format!(
      "no exact integration for {}",
      pgt.name()
    )))
  }
}

/// A cubature exact at least up to `degree` for the elements of pgt.
pub fn classical_approx_im(pgt: &PGeoTrans, degree: usize) -> Result<PIntegration> {
  // the methods found (a list for mixed meshes):
  // the search builds and parses names
  thread_local! {
    static FOUND: std::cell::RefCell<Vec<(PGeoTrans, usize, PIntegration)>> = Default::default();
  }
  let hit = FOUND.with(|f| {
    f.borrow()
      .iter()
      .find(|(g, d, _)| Arc::ptr_eq(g, pgt) && *d == degree)
      .map(|e| e.2.clone())
  });
  if let Some(pim) = hit {
    return Ok(pim);
  }
  let pim = approx_for(&pgt.structure().clone(), degree.max(1))?;
  FOUND.with(|f| f.borrow_mut().push((pgt.clone(), degree, pim.clone())));
  Ok(pim)
}

fn approx_for(cvs: &PCvs, degree: usize) -> Result<PIntegration> {
  use getfem_geom::convex_structure as cs;
  let basic = cvs.basic();
  let n = basic.dim();
  if *basic == *cs::simplex_structure(n) {
    let fam = match n {
      0 => return integ_descriptor("IM_NC(0,0)"),
      1 => "IM_GAUSS1D",
      2 => "IM_TRIANGLE",
      3 => "IM_TETRAHEDRON",
      4 => "IM_SIMPLEX4D",
      _ => {
        return Err(Error::InvalidArgument(format!(
          "no cubature for simplices of dimension {n}"
        )));
      }
    };
    for k in degree..degree + 10 {
      if let Ok(m) = integ_descriptor(&format!("{fam}({k})")) {
        return Ok(m);
      }
    }
    Err(Error::InvalidArgument(format!(
      "no {fam} rule of degree >= {degree}"
    )))
  } else if *basic == *cs::pyramid_qk_structure(1)? {
    // the cube-to-pyramid map adds a factor 4 (1-z)^2: degree + 2 on the cube
    // (GetFEM uses degree, not even exact for constants at degree 1)
    integ_descriptor(&format!(
      "IM_PYRAMID(IM_GAUSS_PARALLELEPIPED(3,{}))",
      degree + 2
    ))
  } else if let Some((a, b)) = basic.product_factors() {
    let (ma, mb) = (approx_for(a, degree)?, approx_for(b, degree)?);
    integ_descriptor(&format!("IM_PRODUCT({},{})", ma.name(), mb.name()))
  } else {
    Err(Error::InvalidArgument(
      "no classical cubature for this element".into(),
    ))
  }
}
