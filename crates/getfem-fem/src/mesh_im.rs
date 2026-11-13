//! Integration methods on a mesh (port of getfem_mesh_im), as a shared
//! handle (SPEC 3). Methods set by a rule (a degree, or one method
//! everywhere) also apply to convexes added to the mesh later.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use getfem_geom::mesh::{Mesh, MeshData};
use getfem_geom::shared::Shared;

use crate::integration::{PIntegration, classical_approx_im};
use crate::level_set::LevelSet;
use crate::mesh_fem::new_stamp;
use crate::{Error, Result};

/// The methods of every convex for one version of the mesh (GetFEM
/// mesh_im::ims).
#[derive(Debug)]
pub struct Ims {
  mesh: Arc<MeshData>,
  config: u64,
  ims: Vec<Option<PIntegration>>,
  version: u64,
  /// on the zero set of these level sets (mesh_im_level_set, boundary):
  /// Normal is the normal of the one the point is on
  normal_level_sets: Vec<LevelSet>,
}

impl Ims {
  pub fn linked_mesh(&self) -> &Arc<MeshData> {
    &self.mesh
  }

  /// The level sets whose normal Normal is (GetFEM: a mesh_im_level_set
  /// integrating on the boundary); empty for other methods.
  pub fn normal_level_sets(&self) -> &[LevelSet] {
    &self.normal_level_sets
  }

  /// The method of convex cv; an error if none (assembly must not
  /// silently skip elements).
  pub fn int_method_of_element(&self, cv: usize) -> Result<&PIntegration> {
    self
      .ims
      .get(cv)
      .and_then(|m| m.as_ref())
      .ok_or_else(|| Error::InvalidArgument(format!("no integration method on convex {cv}")))
  }

  pub fn convex_index(&self) -> impl Iterator<Item = usize> + '_ {
    self
      .mesh
      .convex_index()
      .filter(|&cv| self.ims[cv].is_some())
  }

  /// Changes whenever the geometry or the methods change.
  pub fn version(&self) -> u64 {
    self.version
  }
}

#[derive(Clone, Debug)]
enum Rule {
  Degree(usize),
  Method(PIntegration),
}

#[derive(Debug)]
struct Inner {
  mesh: Mesh,
  ims: BTreeMap<usize, PIntegration>,
  rule: Option<Rule>,
  config: u64,
  cache: Mutex<Option<Arc<Ims>>>,
  normal_level_sets: Vec<LevelSet>,
}

impl Clone for Inner {
  fn clone(&self) -> Self {
    Inner {
      mesh: self.mesh.clone(),
      ims: self.ims.clone(),
      rule: self.rule.clone(),
      config: self.config,
      cache: Mutex::new(self.cache.lock().unwrap_or_else(|e| e.into_inner()).clone()),
      normal_level_sets: self.normal_level_sets.clone(),
    }
  }
}

/// Integration methods on a mesh (GetFEM mesh_im), as a shared handle.
#[derive(Clone, Debug)]
pub struct MeshIm(Shared<Inner>);

impl MeshIm {
  pub fn new(mesh: &Mesh) -> Self {
    MeshIm(Shared::new(Inner {
      mesh: mesh.clone(),
      ims: BTreeMap::new(),
      rule: None,
      config: new_stamp(),
      cache: Mutex::new(None),
      normal_level_sets: vec![],
    }))
  }

  /// Normal on these level sets (set by MeshImLevelSet on the boundary).
  pub(crate) fn set_normal_level_sets(&self, lss: Vec<LevelSet>) {
    self.modify(|s| s.normal_level_sets = lss)
  }

  /// set_integration_method as a builder: `MeshIm::new(&m)
  /// .with_integration_method(5)` or `.with_integration_method(&pim)`
  pub fn with_integration_method(self, arg: impl IntegrationMethodArg) -> Result<Self> {
    self.set_integration_method(arg)?;
    Ok(self)
  }

  pub fn same(&self, other: &MeshIm) -> bool {
    self.0.same(&other.0)
  }

  pub fn linked_mesh(&self) -> Mesh {
    self.0.snapshot().mesh.clone()
  }

  fn modify(&self, f: impl FnOnce(&mut Inner)) {
    self.0.modify(|s| {
      f(s);
      s.config = new_stamp();
    })
  }

  /// GetFEM's set_integration_method, its four forms by the argument: a
  /// method (`&pim`) or a degree (`5`, the classical method exact for
  /// polynomials of that degree), everywhere (also on convexes added
  /// later) or on given convexes (`(cvs, &pim)`, `(cvs, 5)`).
  pub fn set_integration_method(&self, arg: impl IntegrationMethodArg) -> Result<()> {
    arg.apply(self)
  }

  /// The method pim on the given convexes.
  fn set_method_on(&self, cvs: impl IntoIterator<Item = usize>, pim: &PIntegration) -> Result<()> {
    let m = self.linked_mesh().snapshot();
    let cvs: Vec<usize> = cvs.into_iter().collect();
    for &cv in &cvs {
      check_fit(&m, cv, pim)?;
    }
    self.modify(|s| {
      for cv in cvs {
        s.ims.insert(cv, pim.clone());
      }
    });
    Ok(())
  }

  /// The method pim everywhere, also on convexes added later.
  fn set_method_all(&self, pim: &PIntegration) -> Result<()> {
    let m = self.linked_mesh().snapshot();
    for cv in m.convex_index() {
      check_fit(&m, cv, pim)?;
    }
    self.set_rule(Rule::Method(pim.clone()));
    Ok(())
  }

  /// Integration exact for polynomials of this degree on every element,
  /// also on convexes added later.
  fn set_degree_all(&self, degree: usize) -> Result<()> {
    let m = self.linked_mesh().snapshot();
    for cv in m.convex_index() {
      classical_approx_im(m.trans_of_convex(cv), degree)?;
    }
    self.set_rule(Rule::Degree(degree));
    Ok(())
  }

  /// The classical method of this degree on the given convexes.
  fn set_degree_on(&self, cvs: impl IntoIterator<Item = usize>, degree: usize) -> Result<()> {
    let m = self.linked_mesh().snapshot();
    for cv in cvs {
      let pim = classical_approx_im(m.trans_of_convex(cv), degree)?;
      self.set_method_on([cv], &pim)?;
    }
    Ok(())
  }

  fn set_rule(&self, r: Rule) {
    self.modify(|s| {
      s.ims.clear();
      s.rule = Some(r);
    })
  }

  /// The methods for the current mesh.
  pub fn ims(&self) -> Result<Arc<Ims>> {
    let s = self.0.snapshot();
    let mut cache = s.cache.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(l) = cache.as_ref()
      && l.mesh.geometry_version() == s.mesh.geometry_version()
      && l.config == s.config
    {
      return Ok(l.clone());
    }
    let mesh = s.mesh.snapshot();
    let mut ims = vec![None; mesh.nb_convexes_max()];
    for cv in mesh.convex_index() {
      ims[cv] = match (s.ims.get(&cv), &s.rule) {
        (Some(p), _) => Some(p.clone()),
        (None, Some(Rule::Method(p))) => {
          check_fit(&mesh, cv, p)?;
          Some(p.clone())
        }
        (None, Some(Rule::Degree(d))) => Some(classical_approx_im(mesh.trans_of_convex(cv), *d)?),
        (None, None) => None,
      };
    }
    let l = Arc::new(Ims {
      mesh,
      config: s.config,
      ims,
      version: new_stamp(),
      normal_level_sets: s.normal_level_sets.clone(),
    });
    *cache = Some(l.clone());
    Ok(l)
  }

  /// Changes whenever the geometry or the methods change.
  pub fn version(&self) -> Result<u64> {
    Ok(self.ims()?.version)
  }

  pub fn int_method_of_element(&self, cv: usize) -> Result<PIntegration> {
    Ok(self.ims()?.int_method_of_element(cv)?.clone())
  }

  pub fn convex_index(&self) -> Result<Vec<usize>> {
    Ok(self.ims()?.convex_index().collect())
  }
}

fn check_fit(m: &MeshData, cv: usize, pim: &PIntegration) -> Result<()> {
  if !m.is_convex_valid(cv) {
    return Err(Error::InvalidArgument(format!(
      "convex {cv} does not exist"
    )));
  }
  if let Ok(a) = pim.approx()
    && *a.structure() != *m.trans_of_convex(cv).basic_structure()
  {
    return Err(Error::InvalidArgument(format!(
      "{} does not fit convex {cv}",
      pim.name()
    )));
  }
  Ok(())
}

/// The forms of MeshIm::set_integration_method (GetFEM's overloads).
pub trait IntegrationMethodArg {
  fn apply(self, mim: &MeshIm) -> Result<()>;
}

impl IntegrationMethodArg for &PIntegration {
  fn apply(self, mim: &MeshIm) -> Result<()> {
    mim.set_method_all(self)
  }
}

impl IntegrationMethodArg for usize {
  fn apply(self, mim: &MeshIm) -> Result<()> {
    mim.set_degree_all(self)
  }
}

/// an integer literal (`5`) is an i32
impl IntegrationMethodArg for i32 {
  fn apply(self, mim: &MeshIm) -> Result<()> {
    mim.set_degree_all(degree_of(self)?)
  }
}

impl<I: IntoIterator<Item = usize>> IntegrationMethodArg for (I, &PIntegration) {
  fn apply(self, mim: &MeshIm) -> Result<()> {
    mim.set_method_on(self.0, self.1)
  }
}

impl<I: IntoIterator<Item = usize>> IntegrationMethodArg for (I, usize) {
  fn apply(self, mim: &MeshIm) -> Result<()> {
    mim.set_degree_on(self.0, self.1)
  }
}

impl<I: IntoIterator<Item = usize>> IntegrationMethodArg for (I, i32) {
  fn apply(self, mim: &MeshIm) -> Result<()> {
    mim.set_degree_on(self.0, degree_of(self.1)?)
  }
}

fn degree_of(d: i32) -> Result<usize> {
  usize::try_from(d).map_err(|_| Error::InvalidArgument(format!("integration degree {d}")))
}
