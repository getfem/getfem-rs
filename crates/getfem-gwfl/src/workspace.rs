//! Workspace: definitions plus the spaces and values of variables and data,
//! and assembly of scalars, vectors and matrices from GWFL expressions.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::Arc;

use getfem_fem::fem::{FemCtx, PFem, Vectorial, real_base_values};
use getfem_fem::im_data::{ImData, ImdPoints};
use getfem_fem::integration::PIntegration;
use getfem_fem::mesh_fem::{MeshFem, MfDofs, Reduction};
use getfem_fem::mesh_im::{Ims, MeshIm};
use getfem_fem::precomp::{FemPrecomp, GeoPrecomp, fem_precomp, geo_precomp};
use getfem_geom::geotrans::PGeoTrans;
use getfem_geom::geotrans::PointGeom;
use getfem_geom::region::{ALL, Region, RegionArg};
use getfem_linalg::{ColMatrix, RsVector, add_to_rsvector};

use crate::contact::{Contact, ContactBoundary, ContactKind, Disp, InitializedContact};
use crate::defs::{Defs, Support};
use crate::elementary::{ElemFem, ElementaryTransformation, P0Projection, Rotated2DRt0Projection};
use crate::semantic::{Ctx, Der, Src, Ty, analyse, analyse_str};
use crate::transform::{InitializedTransformation, InterpolateTransformation};
use crate::{Error, Result};

#[derive(Clone, Debug)]
pub enum Space {
  Fem(MeshFem),
  Im(ImData),
  Fixed,
}

impl Space {
  /// Value stamp of the space (None for fixed size) and its size.
  fn stamp_and_size(&self) -> Result<Option<(u64, usize)>> {
    Ok(match self {
      Space::Fem(mf) => Some((mf.value_stamp()?, mf.nb_dof()?)),
      Space::Im(imd) => Some((imd.value_stamp()?, imd.nb_values()?)),
      Space::Fixed => None,
    })
  }
}

#[derive(Clone, Debug)]
struct VarDescription {
  space: Space,
  value: Vec<f64>,
  /// position in the global system (enabled variables only)
  interval: Option<Range<usize>>,
  /// insertion number, for the numbering
  seq: usize,
  /// value stamp of the space the value belongs to
  stamp: u64,
  is_variable: bool,
  is_disabled: bool,
  /// an im variable condensed out of the linear solve; numbered after all
  /// others
  is_internal: bool,
  /// the data version of its last value change (GetFEM v_num_data)
  version: u64,
}

#[derive(Clone, Debug)]
pub struct Workspace {
  defs: Defs,
  variables: BTreeMap<String, VarDescription>,
  /// interpolate transformations (the predefined neighbor_element too)
  transformations: BTreeMap<String, InterpolateTransformation>,
  elem_transformations: BTreeMap<String, Arc<dyn ElementaryTransformation>>,
  nb_dof: usize,
  /// incremented when data (not variables) change
  data_version: u64,
  /// incremented when the variables are numbered again
  numbering: u64,
  /// incremented by every change of the definitions (not of values); the
  /// analysed trees of the expressions, valid for this structure (GetFEM
  /// analyses an expression once, when it is added)
  structure: u64,
  trees_cache: TreesCache,
  /// the expressions added: expression, mim, region
  exprs: Vec<(String, MeshIm, Region)>,
  assembled: Assembled,
}

/// The assembled potential, vector and matrix (GetFEM's V, K): cleared and
/// refilled by each assembly (their storage reused) unless given; a copy
/// starts empty
#[derive(Debug, Default)]
struct Assembled {
  potential: f64,
  matrix: ColMatrix<f64>,
  vector: Vec<f64>,
  /// K'_QJ of an assembly with condensation (GetFEM KQJpr)
  coupling: ColMatrix<f64>,
  /// vector, matrix given by set_assembled_* (added to, not cleared)
  given: (bool, bool),
  /// the outputs of the threads, their storage reused
  thread_outs: std::sync::Mutex<ThreadOuts>,
}

impl Clone for Assembled {
  fn clone(&self) -> Self {
    Self::default()
  }
}

/// expression, order, mesh and reference dimensions, geometry versions of
/// the mesh and of the variables' meshes
type TreesKey = (String, usize, usize, usize, Vec<u64>);

#[derive(Debug, Default)]
struct TreesCache(std::sync::Mutex<CachedTrees>);

#[derive(Debug, Default)]
struct CachedTrees {
  structure: u64,
  trees: std::collections::HashMap<TreesKey, Arc<Vec<crate::semantic::Tree>>>,
}

/// a copy starts empty
impl Clone for TreesCache {
  fn clone(&self) -> Self {
    TreesCache::default()
  }
}

impl Default for Workspace {
  fn default() -> Self {
    let transformations = [(
      crate::defs::NEIGHBOR.to_string(),
      InterpolateTransformation::Neighbor,
    )]
    .into();
    Workspace {
      defs: Defs::default(),
      variables: BTreeMap::new(),
      transformations,
      elem_transformations: BTreeMap::new(),
      nb_dof: 0,
      data_version: 0,
      numbering: 0,
      structure: 0,
      trees_cache: Default::default(),
      exprs: vec![],
      assembled: Assembled::default(),
    }
  }
}

impl Workspace {
  pub fn new() -> Self {
    Self::default()
  }

  #[doc(hidden)]
  pub fn defs(&self) -> &Defs {
    &self.defs
  }

  /// Number of unknowns of the global system (all variables, internal
  /// ones included, last).
  pub fn nb_dof(&self) -> usize {
    self.nb_dof
  }

  /// Number of unknowns of the enabled internal variables: the last ones
  /// of the numbering.
  #[doc(hidden)]
  pub fn nb_internal_dof(&self) -> usize {
    self
      .variables
      .values()
      .filter(|s| s.is_internal)
      .filter_map(|s| s.interval.as_ref().map(|r| r.len()))
      .sum()
  }

  /// Number of unknowns of the other variables: the size of the condensed
  /// system.
  #[doc(hidden)]
  pub fn nb_primary_dof(&self) -> usize {
    self.nb_dof - self.nb_internal_dof()
  }

  /// An enabled internal variable; a disabled one is data, not internal.
  pub fn is_internal_variable(&self, name: &str) -> bool {
    self
      .variables
      .get(name)
      .is_some_and(|s| s.is_internal && s.interval.is_some())
  }

  /// Range of a variable in the global system.
  pub fn interval_of_variable(&self, name: &str) -> Option<Range<usize>> {
    self.variables.get(name).and_then(|s| s.interval.clone())
  }

  /// The value of a variable or of data (as GetFEM, "variable" for both)
  pub fn variable(&self, name: &str) -> Option<&[f64]> {
    self.variables.get(name).map(|s| s.value.as_slice())
  }

  /// The value of a variable or of data (as GetFEM, "variable" for
  /// both): its values, or a GWFL expression interpolated on its space (a
  /// mesh_fem, on its basic DOFs then reduced, said on stdout; or an
  /// im_data), evaluated with the values before the change; a variable
  /// on the same space copied by values.
  pub fn set_variable<'a>(&mut self, name: &str, v: impl ValueArg<'a>) -> Result<()> {
    let v = v.values(self, name)?;
    self.set_values(name, &v)
  }

  /// Several variables or data from GWFL expressions, (name, expr) pairs
  /// (each as set_variable), all evaluated with the values before the
  /// change, then set: the order does not matter (a Newmark update reads
  /// the old values).
  pub fn set_variables(&mut self, pairs: &[(&str, &str)]) -> Result<()> {
    if let Some((_, (n, _))) = pairs
      .iter()
      .enumerate()
      .find(|(i, (n, _))| pairs[..*i].iter().any(|(m, _)| m == n))
    {
      return Err(Error::InvalidArgument(format!("set_variables: {n} twice")));
    }
    let vs: Vec<Vec<f64>> = pairs
      .iter()
      .map(|(n, e)| self.interpolation_on_variable(n, e))
      .collect::<Result<_>>()?;
    for ((n, _), v) in pairs.iter().zip(&vs) {
      self.set_values(n, v)?;
    }
    Ok(())
  }

  /// expr interpolated on the space of the variable name; a variable on
  /// the same space (same mesh_fem or im_data, or the same fixed shape)
  /// copied by values (any element, not only Lagrange)
  fn interpolation_on_variable(&self, name: &str, expr: &str) -> Result<Vec<f64>> {
    let s = self
      .variables
      .get(name)
      .ok_or_else(|| Error::InvalidArgument(format!("unknown variable {name}")))?;
    if let Some(src) = self.variables.get(expr.trim())
      && src.value.len() == s.value.len()
      && match (&s.space, &src.space) {
        (Space::Fem(a), Space::Fem(b)) => a.same(b),
        (Space::Im(a), Space::Im(b)) => a.same(b),
        (Space::Fixed, Space::Fixed) => {
          let q = |n: &str| self.defs.var(n).map(|v| &v.qdims);
          q(name) == q(expr.trim())
        }
        _ => false,
      }
    {
      return Ok(src.value.clone());
    }
    let at = |e: Error| Error::InvalidArgument(format!("{name} = {expr}: {e}"));
    let v = match &s.space {
      Space::Fem(mf) => {
        // on the basic DOFs, then reduced (R, or least squares)
        let b = self.interpolation_basic_dofs(expr, mf, ALL).map_err(at)?;
        mf.reduce_vector(&b)?
      }
      Space::Im(imd) => self.interpolation_im_data(expr, imd, ALL).map_err(at)?,
      Space::Fixed => {
        return Err(Error::InvalidArgument(format!(
          "{name} = {expr}: {name} has a fixed size, no space to interpolate on"
        )));
      }
    };
    if v.len() != s.value.len() {
      return Err(Error::InvalidArgument(format!(
        "{name} = {expr}: {} values, {name} has {}",
        v.len(),
        s.value.len()
      )));
    }
    Ok(v)
  }

  fn set_values(&mut self, name: &str, v: &[f64]) -> Result<()> {
    let s = self
      .variables
      .get_mut(name)
      .ok_or_else(|| Error::InvalidArgument(format!("unknown variable {name}")))?;
    // after a change of the space, the value for the new numbering
    let now = s.space.stamp_and_size()?;
    let resized = match now {
      Some((stamp, n)) if stamp != s.stamp && v.len() == n => {
        s.stamp = stamp;
        s.value.len() != n
      }
      _ => false,
    };
    if !resized && s.value.len() != v.len() {
      return Err(Error::InvalidArgument(format!(
        "{name}: {} values, not {}",
        s.value.len(),
        v.len()
      )));
    }
    let data = s.interval.is_none();
    s.value = v.to_vec();
    if data || resized {
      self.data_version += 1;
      s.version = self.data_version;
    }
    if resized {
      self.renumber();
    }
    Ok(())
  }

  /// Variables and data whose space changed (numbering, points or
  /// convexes) since their value was set: an error listing them (SPEC 12,
  /// nothing is reset or transferred automatically).
  #[doc(hidden)]
  pub fn check_spaces(&self) -> Result<()> {
    let mut stale = vec![];
    for (name, st) in &self.variables {
      if let Some((stamp, n)) = st.space.stamp_and_size()?
        && stamp != st.stamp
      {
        stale.push(format!(
          "{name} ({} values, the space now has {n})",
          st.value.len()
        ));
      }
    }
    if stale.is_empty() {
      return Ok(());
    }
    Err(Error::InvalidArgument(format!(
      "the spaces of {} changed (mesh points, convexes or elements): set their values for the new spaces with set_value, e.g. transferred from the old ones",
      stale.join(", ")
    )))
  }

  fn add(
    &mut self,
    name: &str,
    is_data: bool,
    space: Space,
    qdims: &[usize],
    value: Vec<f64>,
  ) -> Result<()> {
    if crate::names::reserved(name).is_some()
      || self.defs.var(name).is_some()
      || self.defs.macros.exists(name)
    {
      return Err(Error::InvalidArgument(format!(
        "{name}: name already in use or reserved"
      )));
    }
    let support = match &space {
      Space::Fem(mf) => Support::Fem {
        mesh_dim: mf.linked_mesh().dim(),
      },
      Space::Im(_) => Support::ImData,
      Space::Fixed => Support::Fixed,
    };
    let interval = (!is_data).then(|| {
      let r = self.nb_dof..self.nb_dof + value.len();
      self.nb_dof = r.end;
      r
    });
    self.defs.add(name, is_data, support, qdims);
    let seq = self.variables.len();
    let stamp = space.stamp_and_size()?.map_or(0, |x| x.0);
    self.variables.insert(
      name.to_string(),
      VarDescription {
        space,
        value,
        interval,
        seq,
        stamp,
        is_variable: !is_data,
        is_disabled: false,
        is_internal: false,
        version: 0,
      },
    );
    // internal variables stay last
    if self.variables.values().any(|s| s.is_internal) {
      self.renumber();
    }
    Ok(())
  }

  /// A disabled variable keeps its value
  /// but leaves the global system; nothing is derived w.r.t. it and terms
  /// with its test functions are dropped. The enabled variables are
  /// renumbered in insertion order.
  pub fn disable_variable(&mut self, name: &str) -> Result<()> {
    self.structure += 1;
    self.set_disabled(name, true)
  }

  pub fn enable_variable(&mut self, name: &str) -> Result<()> {
    self.structure += 1;
    self.set_disabled(name, false)
  }

  fn set_disabled(&mut self, name: &str, disabled: bool) -> Result<()> {
    match self.variables.get_mut(name) {
      Some(s) if s.is_variable => s.is_disabled = disabled,
      _ => return Err(Error::InvalidArgument(format!("{name} is not a variable"))),
    }
    self.renumber();
    self.data_version += 1;
    Ok(())
  }

  /// Positions of the enabled variables, in insertion order, the internal
  /// ones last (GetFEM resize_global_system: two passes).
  fn renumber(&mut self) {
    self.numbering += 1;
    let mut order: Vec<(bool, usize, String)> = self
      .variables
      .iter()
      .map(|(n, s)| (s.is_internal, s.seq, n.clone()))
      .collect();
    order.sort();
    self.nb_dof = 0;
    for (_, _, n) in order {
      let s = self.variables.get_mut(&n).unwrap();
      s.interval = (s.is_variable && !s.is_disabled).then(|| {
        let r = self.nb_dof..self.nb_dof + s.value.len();
        self.nb_dof = r.end;
        r
      });
    }
  }

  /// All variables and data in insertion order: (name, is variable,
  /// disabled, number of values).
  #[doc(hidden)]
  pub fn entries(&self) -> Vec<(String, bool, bool, usize)> {
    let mut v: Vec<_> = self
      .variables
      .iter()
      .map(|(n, s)| {
        (
          s.seq,
          n.clone(),
          s.is_variable,
          s.is_disabled,
          s.value.len(),
        )
      })
      .collect();
    v.sort();
    v.into_iter()
      .map(|(_, n, var, d, len)| (n, var, d, len))
      .collect()
  }

  pub fn is_disabled_variable(&self, name: &str) -> bool {
    self.variables.get(name).is_some_and(|s| s.is_disabled)
  }

  /// variables of a tree, without the disabled ones
  fn enabled_variables(&self, t: &crate::semantic::Tree) -> Vec<(String, String)> {
    let mut v = crate::derive::variables(t, &self.defs);
    v.retain(|(n, _)| !self.is_disabled_variable(n));
    v
  }

  fn fem_size(mf: &MeshFem) -> Result<usize> {
    Ok(mf.nb_dof()?)
  }

  pub fn add_fem_variable(&mut self, name: &str, mf: &MeshFem) -> Result<()> {
    self.structure += 1;
    let n = Self::fem_size(mf)?;
    let q = mf.qdims();
    self.add(name, false, Space::Fem(mf.clone()), &q, vec![0.0; n])
  }

  /// A variable on the DOFs of mf on a region, for compatibility with
  /// GetFEM; the canonical form is
  /// add_fem_variable(name, &MeshFem::partial_on_region(mf, region)?). mf is
  /// not changed; the new space is mesh_fem_of_variable(name).
  pub fn add_filtered_fem_variable(
    &mut self,
    name: &str,
    mf: &MeshFem,
    region: impl Into<RegionArg>,
  ) -> Result<()> {
    self.structure += 1;
    let p = MeshFem::partial_on_region(mf, region)?;
    self.add_fem_variable(name, &p)
  }

  /// An interpolate transformation whose target point is the value of
  /// expr at the source point, located in the target mesh. expr may read
  /// variables: the derivatives through the transformation follow.
  pub fn add_interpolate_transformation_from_expression(
    &mut self,
    name: &str,
    source: &getfem_geom::mesh::Mesh,
    target: &getfem_geom::mesh::Mesh,
    expr: &str,
  ) -> Result<()> {
    self.structure += 1;
    if self.defs.interpolate_transformations.contains_key(name) {
      return Err(Error::InvalidArgument(format!(
        "an interpolate transformation {name} already exists"
      )));
    }
    let tree = crate::transform::analyse_expression(expr, source.dim(), &self.defs)?;
    let bad = |msg: String| Error::Semantic {
      expr: expr.to_string(),
      pos: 0,
      msg,
    };
    let Some(r) = tree.root else {
      return Err(bad("empty transformation expression".into()));
    };
    let n = &tree.n[r];
    if n.nb_test_functions() > 0 || n.tensor_proper_size() != target.dim() {
      return Err(bad(format!(
        "a transformation expression gives a point of the target mesh (dimension {})",
        target.dim()
      )));
    }
    let deps = crate::derive::variables(&tree, &self.defs);
    // nested transformations: defined before (no cycles)
    if let Some((v, t)) = deps
      .iter()
      .find(|(_, t)| !t.is_empty() && !self.transformations.contains_key(t))
    {
      return Err(bad(format!(
        "{v} read through {t}: unknown transformation {t}"
      )));
    }
    self
      .defs
      .interpolate_transformations
      .insert(name.to_string(), deps);
    self.transformations.insert(
      name.to_string(),
      InterpolateTransformation::Expression {
        source: source.clone(),
        target: target.clone(),
        expr: expr.to_string(),
      },
    );
    Ok(())
  }

  /// A raytracing transformation: from a point of a slave boundary along
  /// its deformed normal to the nearest master boundary or rigid obstacle
  /// within release_distance.
  /// Status 1: a master face, 2: a rigid obstacle (Interpolate(X, name)
  /// only), 0: none.
  pub fn add_raytracing_transformation(&mut self, name: &str, release_distance: f64) -> Result<()> {
    self.structure += 1;
    self.add_contact(name, ContactKind::Raytracing, release_distance)
  }

  /// A projection transformation: as raytracing, the nearest point of the
  /// master boundaries.
  pub fn add_projection_transformation(&mut self, name: &str, release_distance: f64) -> Result<()> {
    self.structure += 1;
    self.add_contact(name, ContactKind::Projection, release_distance)
  }

  /// The faces of region of mesh, deformed by the displacement disp, as a
  /// master (target) boundary of the raytracing transformation name.
  pub fn add_master_contact_boundary_to_raytracing_transformation(
    &mut self,
    name: &str,
    mesh: &getfem_geom::mesh::Mesh,
    disp: &str,
    region: impl Into<RegionArg>,
  ) -> Result<()> {
    self.structure += 1;
    self.add_contact_boundary(name, ContactKind::Raytracing, mesh, disp, region, false)
  }

  /// A slave (source) boundary of the raytracing transformation name.
  pub fn add_slave_contact_boundary_to_raytracing_transformation(
    &mut self,
    name: &str,
    mesh: &getfem_geom::mesh::Mesh,
    disp: &str,
    region: impl Into<RegionArg>,
  ) -> Result<()> {
    self.structure += 1;
    self.add_contact_boundary(name, ContactKind::Raytracing, mesh, disp, region, true)
  }

  /// A master boundary of the projection transformation name.
  pub fn add_master_contact_boundary_to_projection_transformation(
    &mut self,
    name: &str,
    mesh: &getfem_geom::mesh::Mesh,
    disp: &str,
    region: impl Into<RegionArg>,
  ) -> Result<()> {
    self.structure += 1;
    self.add_contact_boundary(name, ContactKind::Projection, mesh, disp, region, false)
  }

  /// A slave boundary of the projection transformation name.
  pub fn add_slave_contact_boundary_to_projection_transformation(
    &mut self,
    name: &str,
    mesh: &getfem_geom::mesh::Mesh,
    disp: &str,
    region: impl Into<RegionArg>,
  ) -> Result<()> {
    self.structure += 1;
    self.add_contact_boundary(name, ContactKind::Projection, mesh, disp, region, true)
  }

  /// A rigid obstacle: the zero level set of expr, a scalar expression of
  /// X (or x, y, z, w) of dimension n, negative inside the obstacle.
  pub fn add_rigid_obstacle_to_raytracing_transformation(
    &mut self,
    name: &str,
    expr: &str,
    n: usize,
  ) -> Result<()> {
    self.structure += 1;
    self.add_obstacle(name, ContactKind::Raytracing, expr, n)
  }

  /// A rigid obstacle of the projection transformation name.
  pub fn add_rigid_obstacle_to_projection_transformation(
    &mut self,
    name: &str,
    expr: &str,
    n: usize,
  ) -> Result<()> {
    self.structure += 1;
    self.add_obstacle(name, ContactKind::Projection, expr, n)
  }

  fn add_contact(&mut self, name: &str, kind: ContactKind, release: f64) -> Result<()> {
    if self.defs.interpolate_transformations.contains_key(name) {
      return Err(Error::InvalidArgument(format!(
        "an interpolate transformation {name} already exists"
      )));
    }
    if release.is_nan() || release <= 0.0 {
      return Err(Error::InvalidArgument(format!(
        "{name}: the release distance has to be positive"
      )));
    }
    self
      .defs
      .interpolate_transformations
      .insert(name.to_string(), vec![]);
    self.transformations.insert(
      name.to_string(),
      InterpolateTransformation::Contact(Contact {
        kind,
        release,
        boundaries: vec![],
        obstacles: vec![],
      }),
    );
    Ok(())
  }

  fn contact(&mut self, name: &str, kind: ContactKind) -> Result<&mut Contact> {
    match self.transformations.get_mut(name) {
      Some(InterpolateTransformation::Contact(c)) if c.kind == kind => Ok(c),
      _ => Err(Error::InvalidArgument(format!(
        "{name} is not a {} transformation",
        match kind {
          ContactKind::Raytracing => "raytracing",
          ContactKind::Projection => "projection",
        }
      ))),
    }
  }

  fn add_contact_boundary(
    &mut self,
    name: &str,
    kind: ContactKind,
    mesh: &getfem_geom::mesh::Mesh,
    disp: &str,
    region: impl Into<RegionArg>,
    slave: bool,
  ) -> Result<()> {
    let region = region.into().on(mesh)?;
    region.lasting("contact boundary")?;
    match self.variables.get(disp).map(|s| &s.space) {
      Some(Space::Fem(mf)) if mf.linked_mesh().same(mesh) => {}
      Some(Space::Fem(_)) => {
        return Err(Error::InvalidArgument(format!(
          "{name}: the displacement {disp} is not on the mesh of its contact boundary"
        )));
      }
      _ => {
        return Err(Error::InvalidArgument(format!(
          "{name}: the displacement {disp} is not a finite element variable"
        )));
      }
    }
    let c = self.contact(name, kind)?;
    c.boundaries.push(ContactBoundary {
      mesh: mesh.clone(),
      disp: disp.to_string(),
      region,
      slave,
    });
    let deps = c.deps(name);
    self
      .defs
      .interpolate_transformations
      .insert(name.to_string(), deps);
    Ok(())
  }

  fn add_obstacle(&mut self, name: &str, kind: ContactKind, expr: &str, n: usize) -> Result<()> {
    let data = self
      .variables
      .iter()
      .filter(|(_, st)| matches!(st.space, Space::Fixed))
      .map(|(k, st)| (k.clone(), st.value.clone()))
      .collect();
    crate::contact::Obstacle::new(expr, n, &self.defs, &data)?;
    self
      .contact(name, kind)?
      .obstacles
      .push((expr.to_string(), n));
    Ok(())
  }

  /// An elementary transformation, used as Elementary_transformation(u,
  /// name) or Elementary_transformation(u, name, v).
  pub fn add_elementary_transformation(
    &mut self,
    name: &str,
    t: impl ElementaryTransformation + 'static,
  ) -> Result<()> {
    self.structure += 1;
    let t: Arc<dyn ElementaryTransformation> = Arc::new(t);
    if self.elem_transformations.contains_key(name) {
      return Err(Error::InvalidArgument(format!(
        "an elementary transformation {name} already exists"
      )));
    }
    self
      .defs
      .elementary_transformations
      .insert(name.to_string());
    self.elem_transformations.insert(name.to_string(), t);
    Ok(())
  }

  /// The L2 projection onto the rotated RT0 element (2D; MITC plates).
  #[allow(non_snake_case)]
  pub fn add_2D_rotated_RT0_projection(&mut self, name: &str) -> Result<()> {
    self.structure += 1;
    self.add_elementary_transformation(name, Rotated2DRt0Projection)
  }

  /// The HHO reconstructed gradient (target: a space of dimension qdim*N).
  #[allow(non_snake_case)]
  pub fn add_HHO_reconstructed_gradient(&mut self, name: &str) -> Result<()> {
    self.structure += 1;
    self.add_elementary_transformation(name, crate::hho::HhoReconstructedGradient { sym: false })
  }

  /// The HHO reconstructed symmetrized gradient.
  #[allow(non_snake_case)]
  pub fn add_HHO_reconstructed_symmetrized_gradient(&mut self, name: &str) -> Result<()> {
    self.structure += 1;
    self.add_elementary_transformation(name, crate::hho::HhoReconstructedGradient { sym: true })
  }

  /// The HHO reconstructed value (target: typically degree k + 1).
  #[allow(non_snake_case)]
  pub fn add_HHO_reconstructed_value(&mut self, name: &str) -> Result<()> {
    self.structure += 1;
    self.add_elementary_transformation(name, crate::hho::HhoReconstructedValue { sym: false })
  }

  /// The HHO reconstructed value, symmetrized gradient.
  #[allow(non_snake_case)]
  pub fn add_HHO_reconstructed_symmetrized_value(&mut self, name: &str) -> Result<()> {
    self.structure += 1;
    self.add_elementary_transformation(name, crate::hho::HhoReconstructedValue { sym: true })
  }

  /// The HHO stabilization operator.
  #[allow(non_snake_case)]
  pub fn add_HHO_stabilization(&mut self, name: &str) -> Result<()> {
    self.structure += 1;
    self.add_elementary_transformation(name, crate::hho::HhoStabilization { sym: false })
  }

  /// The HHO stabilization operator, symmetrized.
  #[allow(non_snake_case)]
  pub fn add_HHO_symmetrized_stabilization(&mut self, name: &str) -> Result<()> {
    self.structure += 1;
    self.add_elementary_transformation(name, crate::hho::HhoStabilization { sym: true })
  }

  /// The element L2 projection of the gradient of a variable onto the target
  /// space, Elementary_transformation(u, name, G) (a P0 space G of qdim
  /// qdim(u)*N gives the element mean of Grad_u, for B-bar and F-bar).
  pub fn add_projected_gradient(&mut self, name: &str) -> Result<()> {
    self.structure += 1;
    self.add_elementary_transformation(name, crate::elementary::ProjectedGradient)
  }

  /// The L2 projection onto piecewise constants.
  #[allow(non_snake_case)]
  pub fn add_P0_projection(&mut self, name: &str) -> Result<()> {
    self.structure += 1;
    self.add_elementary_transformation(name, P0Projection)
  }

  pub fn add_fem_data(&mut self, name: &str, mf: &MeshFem, value: &[f64]) -> Result<()> {
    self.structure += 1;
    let n = Self::fem_size(mf)?;
    if value.len() != n {
      return Err(Error::InvalidArgument(format!(
        "{name}: {} values for {n} DOFs",
        value.len()
      )));
    }
    let q = mf.qdims();
    self.add(name, true, Space::Fem(mf.clone()), &q, value.to_vec())
  }

  pub fn add_fixed_size_variable(&mut self, name: &str, shape: &[usize]) -> Result<()> {
    self.structure += 1;
    let n = shape.iter().product();
    self.add(name, false, Space::Fixed, shape, vec![0.0; n])
  }

  pub fn add_fixed_size_data(&mut self, name: &str, shape: &[usize], value: &[f64]) -> Result<()> {
    self.structure += 1;
    if value.len() != shape.iter().product::<usize>() {
      return Err(Error::InvalidArgument(format!(
        "{name}: {} values for shape {shape:?}",
        value.len()
      )));
    }
    self.add(name, true, Space::Fixed, shape, value.to_vec())
  }

  pub fn add_scalar_data(&mut self, name: &str, v: f64) -> Result<()> {
    self.structure += 1;
    self.add_fixed_size_data(name, &[1], &[v])
  }

  /// A variable at the points of imd: one tensor of imd's shape per point,
  /// zero initially; its test functions are the identity at each point.
  pub fn add_im_variable(&mut self, name: &str, imd: &ImData) -> Result<()> {
    self.structure += 1;
    let n = imd.nb_values()?;
    self.add(
      name,
      false,
      Space::Im(imd.clone()),
      &imd.tensor_shape(),
      vec![0.0; n],
    )
  }

  /// An im variable statically condensed out before the linear solve,
  /// numbered after all others.
  pub fn add_internal_im_variable(&mut self, name: &str, imd: &ImData) -> Result<()> {
    self.add_im_variable(name, imd)?;
    self.variables.get_mut(name).unwrap().is_internal = true;
    self.renumber();
    Ok(())
  }

  pub fn add_im_data(&mut self, name: &str, imd: &ImData, value: &[f64]) -> Result<()> {
    self.structure += 1;
    let n = imd.nb_values()?;
    if value.len() != n {
      return Err(Error::InvalidArgument(format!(
        "{name}: {} values for {n}",
        value.len()
      )));
    }
    let q = imd.tensor_shape();
    self.add(name, true, Space::Im(imd.clone()), &q, value.to_vec())
  }

  pub fn add_macro(&mut self, name: &str, expr: &str) -> Result<()> {
    self.structure += 1;
    self.defs.macros.add(name, expr)
  }

  /// Finite element space of a variable or data (GetFEM: also
  /// associated_mf).
  pub fn mesh_fem_of_variable(&self, name: &str) -> Option<MeshFem> {
    match &self.variables.get(name)?.space {
      Space::Fem(mf) => Some(mf.clone()),
      _ => None,
    }
  }

  /// Changes whenever the numbering of the variables or the value of one
  /// of the named data changes (all data for None), for caches of linear
  /// terms (GetFEM: a brick is computed again when one of its data
  /// changed).
  #[doc(hidden)]
  pub fn data_stamp(&self, names: Option<&[String]>) -> u64 {
    let Some(names) = names else {
      return self.data_version;
    };
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ self.numbering;
    for n in names {
      let v = self.variables.get(n).map_or(u64::MAX, |s| s.version);
      h = (h ^ v).wrapping_mul(0x0100_0000_01b3);
    }
    h
  }

  /// The data an expression reads (None: possibly all, through an
  /// interpolate transformation).
  #[doc(hidden)]
  pub fn term_data(&self, expr: &str, mim: &MeshIm) -> Result<Option<Vec<String>>> {
    let m = mim.linked_mesh();
    let ctx = Ctx {
      meshdim: m.dim(),
      ref_elt_dim: ref_elt_dim_of_mesh(&m, &m.region(ALL)?)?,
      option: 1,
      ..Ctx::default()
    };
    let (tree, _) = analyse_str(expr, &self.defs, &ctx)?;
    Ok(crate::derive::data(&tree, &self.defs))
  }

  /// Changes whenever a space of a variable or data changes, geometry
  /// included (for caches of linear terms).
  #[doc(hidden)]
  pub fn spaces_version(&self) -> Result<u64> {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for st in self.variables.values() {
      let v = match &st.space {
        Space::Fem(mf) => mf.version()?,
        Space::Im(imd) => imd.points()?.value_stamp(),
        Space::Fixed => 0,
      };
      h = (h ^ v).wrapping_mul(0x0000_0100_0000_01b3);
    }
    Ok(h)
  }

  /// Variables in the order of the global system, with their ranges.
  pub fn variables(&self) -> Vec<(String, Range<usize>)> {
    let mut v: Vec<(String, Range<usize>)> = self
      .variables
      .iter()
      .filter_map(|(n, s)| s.interval.clone().map(|r| (n.clone(), r)))
      .collect();
    v.sort_by_key(|x| x.1.start);
    v
  }

  /// Order of an expression (0 potential, 1 residual, 2 bilinear form) and
  /// whether its tangent does not depend on the variables (linear term).
  #[doc(hidden)]
  pub fn term_info(
    &self,
    expr: &str,
    mim: &MeshIm,
    region: impl Into<RegionArg>,
  ) -> Result<(usize, bool)> {
    let region = region.into().on(&mim.linked_mesh())?;
    let m = mim.linked_mesh();
    let region = &region;
    let ctx = Ctx {
      meshdim: m.dim(),
      ref_elt_dim: ref_elt_dim_of_mesh(&m, region)?,
      option: 1,
      ..Ctx::default()
    };
    let (tree, _) = analyse_str(expr, &self.defs, &ctx)?;
    let Some(r) = tree.root else {
      return Ok((0, true));
    };
    let order = tree.n[r].nb_test_functions();
    let linear = self
      .trees(expr, mim, region, 2)?
      .iter()
      .all(|t| self.enabled_variables(t).is_empty());
    Ok((order, linear))
  }

  /// Analysed trees of an expression at the requested order (derived), one
  /// per pair of test functions; cached until the definitions change.
  fn trees(
    &self,
    expr: &str,
    mim: &MeshIm,
    region: &Region,
    order: usize,
  ) -> Result<Arc<Vec<crate::semantic::Tree>>> {
    let m = mim.linked_mesh();
    let md = m.snapshot();
    let first = if region.id() == ALL {
      md.convex_index().next()
    } else {
      region.resolve(&md)?.iter().next().map(|(cv, _)| cv)
    };
    let red = first.map_or(m.dim(), |cv| m.trans_of_convex(cv).dim());
    let mut geo = vec![m.geometry_version()];
    geo.extend(self.variables.values().filter_map(|st| match &st.space {
      Space::Fem(mf) => Some(mf.linked_mesh().geometry_version()),
      _ => None,
    }));
    let key = (expr.to_string(), order, m.dim(), red, geo);
    {
      let mut c = self.trees_cache.0.lock().unwrap();
      if c.structure != self.structure {
        c.trees.clear();
        c.structure = self.structure;
      }
      if let Some(t) = c.trees.get(&key) {
        return Ok(t.clone());
      }
    }
    let t = Arc::new(self.trees_uncached(expr, mim, region, order)?);
    self
      .trees_cache
      .0
      .lock()
      .unwrap()
      .trees
      .insert(key, t.clone());
    Ok(t)
  }

  /// The second semantic analysis, at compilation: the values of the
  /// fixed-size variables and data replace them, and the constant
  /// subexpressions are folded (e.g. lambda*Id(meshdim): no instruction at
  /// the points), the trees folded to zero dropped; then the trees of the
  /// same test functions joined (GetFEM add_tree).
  fn eval_fixed_size(
    &self,
    trees: Arc<Vec<crate::semantic::Tree>>,
    mim: &MeshIm,
    region: &Region,
  ) -> Result<Arc<Vec<crate::semantic::Tree>>> {
    let value = |n: &crate::semantic::Node| -> Option<&[f64]> {
      let plain = Ty::Field {
        src: Src::Plain,
        der: Der::Val,
        test: false,
      };
      match self.variables.get(&n.name) {
        Some(st) if n.ty == plain && matches!(st.space, Space::Fixed) => Some(&st.value),
        _ => None,
      }
    };
    // the terms of the same test functions in one tree (GetFEM add_tree)
    let key = |t: &crate::semantic::Tree| {
      t.root.map(|r| {
        let n = &t.n[r];
        let names = [
          &n.name_test1,
          &n.interp_test1,
          &n.name_test2,
          &n.interp_test2,
        ];
        (n.test, names.map(String::clone))
      })
    };
    let fold = |t: &crate::semantic::Tree| t.n.iter().any(|n| value(n).is_some());
    let keys: Vec<_> = trees.iter().map(key).collect();
    let merge = (1..keys.len()).any(|i| keys[..i].contains(&keys[i]));
    if !merge && !trees.iter().any(fold) {
      return Ok(trees);
    }
    let m = mim.linked_mesh();
    let ctx = Ctx {
      meshdim: m.dim(),
      ref_elt_dim: ref_elt_dim_of_mesh(&m, region)?,
      ..Ctx::default()
    };
    let mut out = Vec::with_capacity(trees.len());
    for t in trees.iter() {
      let mut t = t.clone();
      if fold(&t) {
        for i in 0..t.n.len() {
          if let Some(v) = value(&t.n[i]).map(<[f64]>::to_vec) {
            let n = &mut t.n[i];
            n.ty = Ty::Constant;
            n.data = v;
          }
        }
        t = analyse(t, &self.defs, &ctx)?.0;
      }
      let (k, t) = match out.iter().position(|o| key(o) == key(&t)) {
        Some(k) => (k, analyse(out.remove(k).join(&t), &self.defs, &ctx)?.0),
        None => (out.len(), t),
      };
      // a term folded to zero adds nothing (GetFEM compiles no zero tree)
      if t.root.is_some_and(|r| t.n[r].ty != Ty::Zero) {
        out.insert(k, t);
      }
    }
    Ok(Arc::new(out))
  }

  fn trees_uncached(
    &self,
    expr: &str,
    mim: &MeshIm,
    region: &Region,
    order: usize,
  ) -> Result<Vec<crate::semantic::Tree>> {
    let m = mim.linked_mesh();
    // dimension of the first element (entire domain: without its set)
    let md = m.snapshot();
    let first = if region.id() == ALL {
      md.convex_index().next()
    } else {
      region.resolve(&md)?.iter().next().map(|(cv, _)| cv)
    };
    let red = first.map_or(m.dim(), |cv| m.trans_of_convex(cv).dim());
    let ctx = Ctx {
      meshdim: m.dim(),
      ref_elt_dim: red,
      option: 1,
      ..Ctx::default()
    };
    let (tree, _) = analyse_str(expr, &self.defs, &ctx)?;
    let Some(r) = tree.root else {
      return Ok(vec![]);
    };
    let got = tree.n[r].nb_test_functions();
    if got > order {
      let what = ["a scalar", "a vector", "a matrix"][order];
      return Err(Error::Semantic {
        expr: expr.to_string(),
        pos: 0,
        msg: format!("expression of order {got} cannot be assembled as {what}"),
      });
    }
    // derive w.r.t. each variable up to the requested order (GetFEM add_tree)
    let mut level = vec![tree];
    for k in got..order {
      let mut next = vec![];
      for t in &level {
        // variables of transformations on other meshes (GetFEM
        // extract_variables on the mesh of the term)
        let vars: Vec<_> = self
          .enabled_variables(t)
          .into_iter()
          .filter(|(v, tr)| {
            !tr.is_empty()
              || match self.variables.get(v).map(|s| &s.space) {
                Some(Space::Fem(mf)) => mf.linked_mesh().geometry_version() == m.geometry_version(),
                _ => true,
              }
          })
          .collect();
        for v in derive_all(t, &self.defs, &ctx, k + 1, &vars)? {
          next.push(v);
        }
      }
      level = next;
    }
    let mut trees = vec![];
    let disabled =
      |t: &BTreeSet<(String, String)>| t.iter().any(|(n, _)| self.is_disabled_variable(n));
    for tree in level {
      let (tree, tests) = analyse(tree, &self.defs, &ctx)?;
      if tree.root.is_none() {
        continue;
      }
      if tests.test1.len() > 1 || tests.test2.len() > 1 {
        let t2s: Vec<_> = if tests.test2.is_empty() {
          vec![Default::default()]
        } else {
          tests.test2.iter().cloned().collect()
        };
        for t1 in &tests.test1 {
          for t2 in &t2s {
            let c = Ctx {
              option: 2,
              selected_test1: t1.clone(),
              selected_test2: t2.clone(),
              ..ctx.clone()
            };
            let (t, _) = analyse(tree.clone(), &self.defs, &c)?;
            if t.root.is_some()
              && !self.is_disabled_variable(&t1.0)
              && !self.is_disabled_variable(&t2.0)
            {
              trees.push(t);
            }
          }
        }
      } else if !disabled(&tests.test1) && !disabled(&tests.test2) {
        trees.push(tree);
      }
    }
    Ok(trees)
  }

  /// Add expr on mim and region, assembled with the other expressions by
  /// `assembly`; its order (number of test functions) returned.
  pub fn add_expression(
    &mut self,
    expr: &str,
    mim: &MeshIm,
    region: impl Into<RegionArg>,
  ) -> Result<usize> {
    let m = mim.linked_mesh();
    let region = region.into().on(&m)?;
    let ctx = Ctx {
      meshdim: m.dim(),
      ref_elt_dim: ref_elt_dim_of_mesh(&m, &region)?,
      option: 1,
      ..Ctx::default()
    };
    let (tree, _) = analyse_str(expr, &self.defs, &ctx)?;
    let order = tree.root.map_or(0, |r| tree.n[r].nb_test_functions());
    self.exprs.push((expr.to_string(), mim.clone(), region));
    Ok(order)
  }

  /// Remove all the expressions added.
  pub fn clear_expressions(&mut self) {
    self.exprs.clear();
  }

  /// Assemble the expressions added, at order 0 (the assembled potential),
  /// 1 (vector) or 2 (matrix). The assembled vector and matrix are cleared
  /// first, unless given by set_assembled_vector or set_assembled_matrix
  /// (then added to). The expressions on the same mim and region form one
  /// instruction set, those of the same test functions one tree.
  ///
  /// condensation (order 2): the internal variables condensed out at each
  /// point; on entry the assembled vector holds the residual of all
  /// variables, on exit -K_IQ R'_Q on the primary rows and R'_Q on the
  /// internal ones; K'_QJ in the internal coupling matrix.
  pub fn assembly(&mut self, order: usize, condensation: bool) -> Result<()> {
    let exprs = std::mem::take(&mut self.exprs);
    let terms: Vec<(&str, &MeshIm, &Region)> =
      exprs.iter().map(|(e, m, r)| (e.as_str(), m, r)).collect();
    let (mut v, mut k) = (
      std::mem::take(&mut self.assembled.vector),
      std::mem::take(&mut self.assembled.matrix),
    );
    let res = self.assembly_of(&terms, order, condensation, &mut v, &mut k);
    (self.assembled.vector, self.assembled.matrix) = (v, k);
    self.exprs = exprs;
    res
  }

  fn assembly_of(
    &mut self,
    terms: &[(&str, &MeshIm, &Region)],
    order: usize,
    condensation: bool,
    v: &mut Vec<f64>,
    k: &mut ColMatrix<f64>,
  ) -> Result<()> {
    let n = self.nb_dof;
    let (given_v, given_k) = self.assembled.given;
    if !given_v && order == 1 {
      v.clear();
      v.resize(n, 0.0);
    }
    if !given_k && order == 2 {
      k.resize(n, n);
      k.clear();
    }
    let size = |what: &str, s: (usize, usize)| -> Result<()> {
      match s == (n, if what == "vector" { 1 } else { n }) {
        true => Ok(()),
        false => Err(Error::InvalidArgument(format!(
          "assembled {what} of size {}x{} for {n} DOFs",
          s.0, s.1
        ))),
      }
    };
    match order {
      0 => {
        let mut s = 0.0;
        for (mim, rg, es) in groups(terms) {
          self.assemble_with(
            &es,
            mim,
            rg,
            0,
            &mut EntriesTarget(|_, _, x| s += x, 1),
            None,
          )?;
        }
        let mut s = [s];
        getfem_mpi::sum(&mut s);
        self.assembled.potential = s[0];
      }
      1 => {
        size("vector", (v.len(), 1))?;
        // MPI: this assembly's part apart, summed over the ranks, then
        // added (a given vector's values are not summed)
        let mpi = getfem_mpi::size() > 1;
        let mut w = match mpi {
          true => vec![0.0; n],
          false => std::mem::take(v),
        };
        for (mim, rg, es) in groups(terms) {
          self.assemble_with(
            &es,
            mim,
            rg,
            1,
            &mut EntriesTarget(|i, _, x| w[i] += x, n),
            None,
          )?;
        }
        if mpi {
          getfem_mpi::sum(&mut w);
          v.iter_mut().zip(&w).for_each(|(a, b)| *a += b);
        } else {
          *v = w;
        }
      }
      2 => {
        size("matrix", (k.nrows, k.ncols))?;
        let mut cond = match condensation {
          true => {
            size("vector", (v.len(), 1))?;
            Some(CondOut {
              residual: Arc::new(std::mem::take(v)),
              kqj: ColMatrix::new(n, n),
              rhs: vec![0.0; n],
              rq: vec![0.0; n],
            })
          }
          false => None,
        };
        for (mim, rg, es) in groups(terms) {
          let d = mim.linked_mesh().dim();
          let mut target = MatrixTarget {
            k: &mut *k,
            n: d,
            sort: vec![],
          };
          self.assemble_with(&es, mim, rg, 2, &mut target, cond.as_mut())?;
        }
        if let Some(mut c) = cond {
          // MPI: summed over the ranks (GetFEM: V and unreduced_V); the
          // matrices stay each rank's
          getfem_mpi::sum(&mut c.rhs);
          getfem_mpi::sum(&mut c.rq);
          // -K_IQ R'_Q on the primary rows, R'_Q on the internal ones
          *v = c.rhs.iter().zip(&c.rq).map(|(a, b)| a + b).collect();
          self.assembled.coupling = c.kqj;
        }
      }
      _ => {
        return Err(Error::InvalidArgument(format!(
          "assembly of order {order} (0, 1 or 2)"
        )));
      }
    }
    Ok(())
  }

  /// The assembled potential.
  pub fn assembled_potential(&self) -> f64 {
    self.assembled.potential
  }

  /// The assembled vector, over all variables (interval_of_variable for
  /// the part of one).
  pub fn assembled_vector(&self) -> &[f64] {
    &self.assembled.vector
  }

  /// The assembled matrix.
  pub fn assembled_matrix(&self) -> &ColMatrix<f64> {
    &self.assembled.matrix
  }

  /// K'_QJ of the last assembly with condensation.
  pub fn internal_coupling_matrix(&self) -> &ColMatrix<f64> {
    &self.assembled.coupling
  }

  /// The assembled vector given: assembly adds to v.
  pub fn set_assembled_vector(&mut self, v: Vec<f64>) {
    self.assembled.vector = v;
    self.assembled.given.0 = true;
  }

  /// The assembled matrix given: assembly adds to k.
  pub fn set_assembled_matrix(&mut self, k: ColMatrix<f64>) {
    self.assembled.matrix = k;
    self.assembled.given.1 = true;
  }

  /// The assembled vector, matrix and internal coupling matrix moved out
  /// (Rust ownership: GetFEM's caller owns the storage given by
  /// set_assembled_*); the workspace's own storage again.
  #[doc(hidden)]
  pub fn release_assembled(&mut self) -> (Vec<f64>, ColMatrix<f64>, ColMatrix<f64>) {
    self.assembled.given = (false, false);
    (
      std::mem::take(&mut self.assembled.vector),
      std::mem::take(&mut self.assembled.matrix),
      std::mem::take(&mut self.assembled.coupling),
    )
  }

  /// Values of expr at the points of the methods of mim: f(cv, face,
  /// point index in the method, components). Returns the number of
  /// components (None if expr is zero).
  fn eval_points(
    &self,
    expr: &str,
    mim: &MeshIm,
    region: &Region,
    f: &mut PointTarget,
  ) -> Result<Option<usize>> {
    if self.trees(expr, mim, region, 0)?.is_empty() {
      return Ok(None);
    }
    let mut size = None;
    let mut g = |cv: usize, face: Option<usize>, ip: usize, c: &[f64]| {
      size = Some(c.len());
      f(cv, face, ip, c)
    };
    self.assemble(expr, mim, region, VALUES, &mut Points(&mut g))?;
    Ok(size.or(Some(1)))
  }

  /// Values of expr on a target (GetFEM's interpolation of the Python
  /// interface): the nodes of a Lagrange space mf, the points of an im_data
  /// (as its tensors), or points of a mesh, `(pts, &mesh)`.
  pub fn interpolation(
    &self,
    expr: &str,
    target: impl InterpolationArg,
    region: impl Into<RegionArg>,
  ) -> Result<Vec<f64>> {
    target.interpolate(self, expr, region.into())
  }

  /// Values of expr at the nodes of the Lagrange space mf, averaged over
  /// the elements sharing a node: qdim(mf) * m values per basic DOF when
  /// expr has qdim(mf) * m components.
  fn interpolation_lagrange_fem(
    &self,
    expr: &str,
    mf: &MeshFem,
    region: impl Into<RegionArg>,
  ) -> Result<Vec<f64>> {
    // a reduced space refused, as GetFEM
    if mf.is_reduced() {
      return Err(Error::InvalidArgument(
        "interpolation on a reduced (or partial) space: interpolate on the space without its reduction, then reduce".into(),
      ));
    }
    self.interpolation_basic_dofs(expr, mf, region)
  }

  /// interpolation_lagrange_fem on the basic DOFs of mf, reduced or not
  fn interpolation_basic_dofs(
    &self,
    expr: &str,
    mf: &MeshFem,
    region: impl Into<RegionArg>,
  ) -> Result<Vec<f64>> {
    let region = region.into().on(&mf.linked_mesh())?;
    let num = mf.numbering()?;
    // per element (fem) of mf: its nodes as a point set (GetFEM
    // pf->node_tab), the nodes on each face, its convexes
    type FemNodes<'a> = (&'a PFem, PIntegration, Vec<Vec<usize>>, Vec<usize>);
    let mut nodes: Vec<FemNodes> = vec![];
    for cv in num.convex_index() {
      let pf = num.fem_of_element(cv).unwrap();
      if let Some(n) = nodes.iter_mut().find(|n| Arc::ptr_eq(n.0, pf)) {
        n.3.push(cv);
        continue;
      }
      if !pf.is_lagrange() || pf.target_dim() != 1 {
        return Err(Error::InvalidArgument(format!(
          "interpolation on {}: Lagrange elements only",
          pf.name()
        )));
      }
      let cvr = pf.ref_convex();
      let on_face: Vec<Vec<usize>> = (0..cvr.structure().nb_faces())
        .map(|f| {
          (0..pf.nb_dof())
            .filter(|&i| pf.faces_of_dof(i).contains(&f))
            .collect()
        })
        .collect();
      let node = |i: usize| pf.node(i).to_vec();
      let pim = getfem_fem::integration::point_set(
        format!("IM_NODES({})", pf.name()),
        cvr,
        (0..pf.nb_dof()).map(node).collect(),
        on_face
          .iter()
          .map(|is| is.iter().map(|&i| node(i)).collect())
          .collect(),
        false,
      );
      nodes.push((pf, pim, on_face, vec![cv]));
    }
    let mim = MeshIm::new(&mf.linked_mesh());
    for (_, pim, _, cvs) in &nodes {
      mim.set_integration_method((cvs.iter().copied(), pim))?;
    }
    let q = num.qdim();
    let mut sum = vec![0.0; num.nb_basic_dof()];
    let mut count = vec![0.0; num.nb_basic_dof() / q];
    let mut s = q;
    // the components found on this rank (0: no node here)
    let mut found = 0;
    let size = self.eval_points(expr, &mim, &region, &mut |cv, face, ip, c| {
      if c.len() % q != 0 {
        return Err(Error::InvalidArgument(format!(
          "interpolation of {expr} ({} components) on a space of qdim {q}",
          c.len()
        )));
      }
      if s != c.len() {
        s = c.len();
        sum = vec![0.0; num.nb_basic_dof() / q * s];
      }
      found = s;
      let pf = num.fem_of_element(cv).unwrap();
      let i = match face {
        None => ip,
        Some(f) => {
          let (_, pim, on_face, _) = nodes.iter().find(|n| Arc::ptr_eq(n.0, pf)).unwrap();
          on_face[f][ip - pim.approx()?.ind_first_point_on_face(f)]
        }
      };
      let node = num.ind_basic_dof_of_element(cv)[i * q] / q;
      sum[s * node..s * (node + 1)]
        .iter_mut()
        .zip(c)
        .for_each(|(a, b)| *a += b);
      count[node] += 1.0;
      Ok(())
    })?;
    if size.is_none() {
      return Ok(vec![0.0; num.nb_basic_dof()]);
    }
    // MPI: each rank its elements' nodes, the sums and counts over the
    // ranks (GetFEM's finalize: MPI_MAX of the sizes, MPI_SUM)
    if getfem_mpi::size() > 1 {
      let mut sz = [found as f64];
      getfem_mpi::max(&mut sz);
      if sz[0] as usize != s {
        s = (sz[0] as usize).max(q);
        sum = vec![0.0; num.nb_basic_dof() / q * s];
      }
      getfem_mpi::sum(&mut sum);
      getfem_mpi::sum(&mut count);
    }
    for (node, &k) in count.iter().enumerate() {
      if k > 1.0 {
        sum[s * node..s * (node + 1)]
          .iter_mut()
          .for_each(|a| *a /= k);
      }
    }
    Ok(sum)
  }

  /// Values of expr at points of the mesh, each point evaluated on the
  /// first element containing it; a point outside the mesh (or the region)
  /// is an error.
  fn interpolation_mti(
    &self,
    expr: &str,
    mesh: &getfem_geom::mesh::Mesh,
    points: &[Vec<f64>],
    region: impl Into<RegionArg>,
  ) -> Result<Vec<f64>> {
    let region = region.into().on(mesh)?;
    let m = mesh.snapshot();
    let loc = getfem_geom::locate::Locator::new(&m, 1e-10)?;
    let within = region.resolve(&m)?;
    // points of each convex: (point index, reference coordinates)
    let mut per_cv: BTreeMap<usize, Vec<(usize, Vec<f64>)>> = BTreeMap::new();
    for (p, x) in points.iter().enumerate() {
      let Some((cv, xref)) = loc.find_first(x, |cv| within.is_in(cv))? else {
        return Err(Error::InvalidArgument(format!(
          "interpolation: point {x:?} outside the mesh or the region"
        )));
      };
      per_cv.entry(cv).or_default().push((p, xref));
    }
    let mim = MeshIm::new(mesh);
    for (&cv, pts) in &per_cv {
      let cvr = m.trans_of_convex(cv).convex_ref().clone();
      let set = pts.iter().map(|(_, xref)| xref.clone()).collect();
      let pim =
        getfem_fem::integration::point_set(format!("IM_POINTS({cv})"), &cvr, set, vec![], true);
      mim.set_integration_method(([cv], &pim))?;
    }
    let mut set = getfem_geom::region::RegionSet::new();
    per_cv.keys().for_each(|&cv| set.add(cv));
    let cvs = Region::temporary(&m, set)?;
    let mut out = vec![];
    self.eval_points(expr, &mim, &cvs, &mut |cv, _, ip, c| {
      if out.is_empty() {
        out = vec![0.0; points.len() * c.len()];
      }
      let p = per_cv[&cv][ip].0;
      out[p * c.len()..(p + 1) * c.len()].copy_from_slice(c);
      Ok(())
    })?;
    // MPI: each point on one element, of one rank: the values summed over
    // the ranks (the components agreed first: a rank may have no point)
    if getfem_mpi::size() > 1 {
      let mut c = [if points.is_empty() {
        0.0
      } else {
        (out.len() / points.len()) as f64
      }];
      getfem_mpi::max(&mut c);
      out.resize(points.len() * c[0] as usize, 0.0);
      getfem_mpi::sum(&mut out);
    }
    if out.is_empty() {
      out = vec![0.0; points.len()];
    }
    Ok(out)
  }

  /// Values of expr at the points of an ImData, as its tensors.
  fn interpolation_im_data(
    &self,
    expr: &str,
    imd: &ImData,
    region: impl Into<RegionArg>,
  ) -> Result<Vec<f64>> {
    let region = region.into().on(&imd.linked_mesh_im().linked_mesh())?;
    let pts = imd.points()?;
    let ts = imd.tensor_size();
    let mut out = vec![0.0; pts.nb_points() * ts];
    self.eval_points(
      expr,
      &imd.linked_mesh_im(),
      &region,
      &mut |cv, _face, ip, c| {
        if c.len() != ts {
          return Err(Error::InvalidArgument(format!(
            "interpolation of {expr} ({} components) on ImData tensors of size {ts}",
            c.len()
          )));
        }
        if let Some(k) = pts.index_of_point(cv, ip) {
          out[k * ts..(k + 1) * ts].copy_from_slice(c);
        }
        Ok(())
      },
    )?;
    // MPI: each point of one element, of one rank
    getfem_mpi::sum(&mut out);
    Ok(out)
  }

  /// Element-wise L2 projection of expr on the discontinuous space mf.
  pub fn local_projection(
    &self,
    expr: &str,
    mim: &MeshIm,
    mf: &MeshFem,
    region: impl Into<RegionArg>,
  ) -> Result<Vec<f64>> {
    let region = region.into().on(&mim.linked_mesh())?;
    if mf.is_reduced() {
      return Err(Error::InvalidArgument(
        "local_projection on a reduced (or partial) space".into(),
      ));
    }
    let num = mf.numbering()?;
    let mut owner = vec![usize::MAX; num.nb_basic_dof()];
    for cv in num.convex_index() {
      for &d in num.ind_basic_dof_of_element(cv) {
        if owner[d] != usize::MAX && owner[d] != cv {
          return Err(Error::InvalidArgument(
            "local_projection: the space must be discontinuous (DOFs not shared)".into(),
          ));
        }
        owner[d] = cv;
      }
    }
    let c = "c__local_projection";
    let mut ws = self.clone();
    ws.add_fem_variable(c, mf)?;
    let dot = if mf.qdims().len() > 1 { ":" } else { "." };
    let r = ws.interval_of_variable(c).unwrap();
    ws.clear_expressions();
    ws.add_expression(&format!("({expr}){dot}Test_{c}"), mim, region.clone())?;
    ws.assembly(1, false)?;
    let f = ws.assembled_vector().to_vec();
    ws.clear_expressions();
    ws.add_expression(&format!("Test_{c}{dot}Test2_{c}"), mim, region.clone())?;
    ws.assembly(2, false)?;
    let mm = ws.assembled_matrix().to_csr();
    let mut out = vec![0.0; num.nb_basic_dof()];
    let m = num.linked_mesh();
    let region = &region;
    // MPI: a rank's mass matrix has its elements' blocks
    let mine = (getfem_mpi::size() > 1).then(|| m.mpi_region());
    for cv in region.resolve(m)?.convexes() {
      if num.fem_of_element(cv).is_none() || mine.as_ref().is_some_and(|r| !r.is_in(cv)) {
        continue;
      }
      let dofs = num.ind_basic_dof_of_element(cv);
      let n = dofs.len();
      let mut a = getfem_linalg::tensor::Mat::zeros(&[n, n]);
      for (i, &di) in dofs.iter().enumerate() {
        for (j, &dj) in dofs.iter().enumerate() {
          a[(i, j)] = mm.get(r.start + di, r.start + dj);
        }
      }
      let b: Vec<f64> = dofs.iter().map(|&d| f[r.start + d]).collect();
      let x = getfem_linalg::tensor::Lu::new(&a)?.solve(&b);
      for (i, &d) in dofs.iter().enumerate() {
        out[d] = x[i];
      }
    }
    // MPI: each DOF of one element, of one rank
    getfem_mpi::sum(&mut out);
    Ok(out)
  }

  fn assemble(
    &self,
    expr: &str,
    mim: &MeshIm,
    region: &Region,
    order: usize,
    target: &mut impl AssemblyTarget,
  ) -> Result<()> {
    self.assemble_with(&[expr], mim, region, order, target, None)
  }

  /// One instruction set for the terms exprs (on the same mim and region); cond:
  /// the matrix with the internal variables condensed (outputs in it)
  fn assemble_with(
    &self,
    exprs: &[&str],
    mim: &MeshIm,
    region: &Region,
    order: usize,
    target: &mut impl AssemblyTarget,
    mut cond: Option<&mut CondOut>,
  ) -> Result<()> {
    self.check_spaces()?;
    let mut asm_args = self.asm_args(mim)?;
    let tord = if order == VALUES { 0 } else { order };
    let joined;
    let expr = match exprs {
      [e] => *e,
      _ => {
        joined = exprs.join(" + ");
        joined.as_str()
      }
    };
    let trees = match exprs {
      [e] => self.trees(e, mim, region, tord)?,
      _ => {
        let mut all = vec![];
        for e in exprs {
          all.extend(self.trees(e, mim, region, tord)?.iter().cloned());
        }
        Arc::new(all)
      }
    };
    if trees.is_empty() {
      return Ok(());
    }
    let trees = self.eval_fixed_size(trees, mim, region)?;
    if trees.is_empty() {
      return Ok(());
    }
    // GetFEM's compiled instructions
    let fixed = |v: &str| match self.variables.get(v) {
      Some(st) if matches!(st.space, Space::Fixed) => Some(st.value.clone()),
      _ => None,
    };
    let vectorized = |v: &str| {
      asm_args
        .var_mfs
        .get(v)
        .is_some_and(|w| w.numbering.is_uniformly_vectorized())
    };
    let trans = |name: &str| -> Option<crate::ga::TransKind> {
      use crate::ga::TransKind;
      if matches!(
        self.transformations.get(name)?,
        InterpolateTransformation::Contact(_)
      ) {
        return Some(TransKind::Contact);
      }
      match (
        asm_args.transformations.get(name)?,
        self.transformations.get(name)?,
      ) {
        (InitializedTransformation::Neighbor, _) => Some(TransKind::Neighbor),
        (InitializedTransformation::Contact(_), _) => Some(TransKind::Contact),
        (
          InitializedTransformation::Expression { .. },
          InterpolateTransformation::Expression { source, expr, .. },
        ) => crate::transform::analyse_expression(expr, source.dim(), &self.defs)
          .ok()
          .map(TransKind::Expr),
        _ => None,
      }
    };
    let trans_der = |tr: &str, v: &str, vt: &str| -> Option<Option<crate::semantic::Tree>> {
      let InterpolateTransformation::Expression { source, expr, .. } =
        self.transformations.get(tr)?
      else {
        return None;
      };
      let n = source.dim();
      let tree = crate::transform::analyse_expression(expr, n, &self.defs).ok()?;
      let Some(d) = crate::derive::derive(&tree, &self.defs, v, vt, 1, None).ok()? else {
        return Some(None);
      };
      let ctx = Ctx {
        meshdim: n,
        ref_elt_dim: n,
        option: 1,
        ..Ctx::default()
      };
      let (d, _) = analyse(d, &self.defs, &ctx).ok()?;
      Some(d.root.is_some().then_some(d))
    };
    let cx = crate::ga::CompileCtx {
      defs: &self.defs,
      fixed: &fixed,
      inputs: &[],
      vectorized: &vectorized,
      trans: &trans,
      trans_der: &trans_der,
      internal: &|v: &str| cond.is_some() && self.is_internal_variable(v),
    };
    let gis = match order {
      0 => crate::ga::compile_scalar(&trees, &cx),
      1 => crate::ga::compile_vector(&trees, &cx),
      VALUES => crate::ga::compile_values(&trees, &cx),
      _ => crate::ga::compile_matrix(&trees, &cx),
    };
    #[cfg(test)]
    if let Some(g) = &gis {
      g.record();
    }
    let Some(gis) = gis else {
      if order != VALUES
        && let Some(t) = trees
          .iter()
          .find(|t| t.root.is_some_and(|r| t.n[r].tensor_proper_size() != 1))
      {
        let r = t.root.unwrap();
        return Err(Error::Semantic {
          expr: expr.to_string(),
          pos: 0,
          msg: format!(
            "the integrand must be scalar valued, not of size {}",
            t.n[r].tensor_proper_size()
          ),
        });
      }
      return Err(Error::Unsupported {
        expr: expr.to_string(),
        pos: 0,
        msg: "not supported".into(),
      });
    };
    self.init_contact_transformations(&mut asm_args, &gis.trans)?;
    // variables on spaces with reduction matrices: assembled on their
    // basic DOFs (temporary intervals after the DOFs), then K = P^T K P,
    // V = P^T V (P: identity on the DOFs, E on a temporary interval; GetFEM
    // Kuu, Kur, Vu and the unreduced terms). Only those the instructions read.
    let mut tmp = BTreeMap::new();
    let mut ext = vec![];
    let mut n_ext = self.nb_dof;
    for key in &gis.vars {
      let name = key.split_once('\u{1}').map_or(key.as_str(), |(v, _)| v);
      let (Some(mf_dofs), Some(st)) = (asm_args.var_mfs.get(name), self.variables.get(name)) else {
        continue;
      };
      if let (Some(r), Some(iv), false) =
        (mf_dofs.reduction(), &st.interval, tmp.contains_key(name))
      {
        tmp.insert(name.to_string(), n_ext);
        ext.push((n_ext, iv.start, r));
        n_ext += r.e.nrows;
      }
    }
    if let Some(c) = cond.as_deref_mut() {
      c.size(self.nb_dof, n_ext);
    }
    if tmp.is_empty() {
      return self.assemble_ga(
        &gis,
        &asm_args,
        region,
        expr,
        target,
        (&tmp, &[], None, cond),
      );
    }
    // vectors: a dense temporary vector (GetFEM Vu); matrices: a column
    // matrix on all indices (Kuu, Kur, Kru)
    let vector = order == 1;
    let mut unreduced = UnreducedOut {
      n: self.nb_dof,
      k: ColMatrix::new(
        if vector { 0 } else { n_ext },
        if vector { 0 } else { n_ext },
      ),
      vu: if vector {
        vec![0.0; n_ext - self.nb_dof]
      } else {
        vec![]
      },
      vector,
      dim: asm_args.ims.linked_mesh().dim(),
      sort: vec![],
    };
    self.assemble_ga(
      &gis,
      &asm_args,
      region,
      expr,
      target,
      (&tmp, &ext, Some(&mut unreduced), cond),
    )
  }
}

/// The temporary intervals of an assembly: (start, start of the interval
/// of the variable, its reduction)
type TmpVarIntervals<'a> = [(usize, usize, &'a Reduction)];

/// The unreduced terms of out folded into inner (GetFEM Vu, Kuu, Kur, Kru):
/// V += E^T Vu, K += P^T K_ext P (P: identity on the DOFs, E on a temporary
/// interval)
fn fold_unreduced(ext: &TmpVarIntervals, out: &UnreducedOut, inner: &mut dyn AssemblyTarget) {
  let n = out.n;
  if out.vector {
    // V += E^T Vu, by the columns of E
    for &(t, off, r) in ext {
      let vu = &out.vu[t - n..t - n + r.e.nrows];
      for c in 0..r.et.nrows {
        let x: f64 = r.et.row(c).map(|(d, y)| y * vu[d]).sum();
        if x != 0.0 {
          inner.entry(off + c, 0, x);
        }
      }
    }
    return;
  }
  // K = P^T K_ext P by columns (GetFEM Kuu, Kur, Kru folded): column j
  // of K_ext, and for column c of a temporary interval's E the sum of
  // E(d, c) K_ext(:, t + d); its rows of a temporary interval folded with
  // E^T, into a dense column
  // the dense column: values, rows set (seen, in order of first set)
  let mut dense = (vec![0.0; n], vec![false; n], vec![]);
  let add =
    |(acc, seen, rows): &mut (Vec<f64>, Vec<bool>, Vec<usize>), kcol: &[(usize, f64)], y: f64| {
      for &(i, v) in kcol {
        let mut push = |i: usize, x: f64| {
          acc[i] += x;
          if !std::mem::replace(&mut seen[i], true) {
            rows.push(i);
          }
        };
        match ext
          .iter()
          .find(|(t, _, r)| (*t..t + r.e.nrows).contains(&i))
        {
          Some(&(t, off, r)) => r.e.row(i - t).for_each(|(c, e)| push(off + c, y * v * e)),
          None => push(i, y * v),
        }
      }
    };
  let mut col: Vec<(usize, f64)> = vec![];
  for j in 0..n {
    add(&mut dense, &out.k.cols[j], 1.0);
    if let Some(&(t, off, r)) = ext
      .iter()
      .find(|(_, off, r)| (*off..off + r.e.ncols).contains(&j))
    {
      r.et
        .row(j - off)
        .for_each(|(d, y)| add(&mut dense, &out.k.cols[t + d], y));
    }
    let (acc, seen, rows) = &mut dense;
    rows.sort_unstable();
    col.clear();
    col.extend(rows.drain(..).map(|i| (i, std::mem::take(&mut acc[i]))));
    col.iter().for_each(|&(i, _)| seen[i] = false);
    if !col.is_empty() {
      inner.add_column(j, &col);
    }
  }
}

/// A finite element variable of a compiled instruction set on the current element
/// and point: what the instructions read (GetFEM: the mesh_fem's
/// fem_interpolation_context).
struct GaVar<'a> {
  kind: GaKind<'a>,
  /// on the basic DOFs for a space with reduction matrices (E U)
  value: std::borrow::Cow<'a, [f64]>,
  /// start of its interval
  off: usize,
  /// start of its temporary interval on the basic DOFs (a variable on a
  /// space with reduction matrices, GetFEM temporary_interval_of_variable)
  toff: Option<usize>,
  pf: Option<PFem>,
  pc: Option<Arc<getfem_fem::precomp::FemPrecomp>>,
  /// reference Hessians at the points of the integration method
  hr: Vec<getfem_linalg::tensor::Tensor<f64>>,
  /// element matrix of a transformed element (GetFEM M), on this element
  mt: Option<getfem_linalg::tensor::Tensor<f64>>,
  /// values, real gradients, Hessians at the point (real elements)
  real: Option<getfem_fem::fem::RealValues>,
  /// the same on the sides + and - of the level set the point is on (Xfem_plus,
  /// Xfem_minus), if read
  real_xfem: [Option<getfem_fem::fem::RealValues>; 2],
  /// its element: tau-equivalent, scalar, on the reference element
  equiv: bool,
  std: bool,
  /// Interp: the target element, test functions of the largest element
  tg: GaTarget,
  nbmax: usize,
  name: &'a str,
}

enum GaKind<'a> {
  Fem(&'a MfDofs),
  /// fixed size variable of this size
  Fixed(usize),
  /// im_data: points, tensor size
  Im(&'a ImdPoints, usize),
  /// a variable on the target elements of transformation tr
  Interp(&'a MfDofs, usize),
}

/// A transformation at the current point.
struct GaTr<'a> {
  name: &'a str,
  rt: &'a InitializedTransformation,
  tgt: Option<TPoint>,
  status: usize,
  /// contact: the hit at the point
  hit: Option<crate::contact::Hit>,
  /// calls so far (fields on the target are cached per call)
  stamp: u64,
}

/// A variable on the current target element: DOFs, coefficients, real
/// basis values (val, grad, hess as PtField).
#[derive(Default)]
struct GaTarget {
  stamp: u64,
  f: Option<PtField>,
}

/// The instructions' view of the workspace at an element and point.
struct GaEnv<'a> {
  vars: Vec<GaVar<'a>>,
  geo: crate::ga::GeoCtx,
  cv: usize,
  face: Option<usize>,
  ip: usize,
  /// second derivatives of the transformation at the point:
  /// htau[l + n (j + p k)], empty if linear
  htau: Vec<f64>,
  /// work
  tmp: Vec<f64>,
  /// Hessians used (also of interpolated fields)
  needs_hess: bool,
  trs: Vec<GaTr<'a>>,
  /// elementary transformations: transformation, source, dest
  ets: Vec<(&'a str, &'a Arc<dyn ElementaryTransformation>, usize, usize)>,
  gt: Option<PGeoTrans>,
  /// integration method of the element (contact)
  pim: Option<PIntegration>,
  ws: &'a Workspace,
  asm_args: &'a AssemblyArgs,
  mesh: &'a Arc<getfem_geom::mesh::MeshData>,
  expr: &'a str,
  target: &'a mut dyn AssemblyTarget,
  /// global indices of the test DOFs (work)
  rows: Vec<Option<usize>>,
  cols: Vec<Option<usize>>,
  rows1: Vec<Option<usize>>,
  cols1: Vec<Option<usize>>,
  /// condensation outputs (condensed matrix assembly)
  cond: Option<&'a mut CondOut>,
}

impl GaVar<'_> {
  /// number of functions of the element (after its element matrix)
  fn nd(&self) -> usize {
    let pf = self.pf.as_ref().unwrap();
    if self.mt.is_some() || self.real.is_some() {
      pf.nb_dof()
    } else {
      pf.nb_base()
    }
  }
}

/// real gradients of a scalar element for the reference dimension P:
/// out[i + nb k] = sum_j B[k + n j] gr[i + nb j] (GetFEM mat_tmult)
fn grad_ref<const P: usize>(out: &mut [f64], b: &[f64], gr: &[f64], n: usize, nb: usize) {
  let g: [&[f64]; P] = std::array::from_fn(|j| &gr[nb * j..nb * (j + 1)]);
  for (k, ok) in out.chunks_exact_mut(nb).enumerate() {
    let bk: [f64; P] = std::array::from_fn(|j| b[k + n * j]);
    for (i, o) in ok.iter_mut().enumerate() {
      let mut s = bk[0] * g[0][i];
      for j in 1..P {
        s += bk[j] * g[j][i];
      }
      *o = s;
    }
  }
}

/// phi_i = sum_j M_ij phi_j on each column of t (nb rows), identity if no M
fn trans(m: Option<&getfem_linalg::tensor::Tensor<f64>>, t: &[f64], nb: usize, out: &mut [f64]) {
  let Some(m) = m else {
    out.copy_from_slice(t);
    return;
  };
  let w = t.len() / nb;
  let nd = out.len() / w;
  out.fill(0.0);
  for c in 0..w {
    for j in 0..nb {
      let tj = t[j + nb * c];
      for i in 0..nd {
        out[i + nd * c] += m[(i, j)] * tj;
      }
    }
  }
}

impl GaEnv<'_> {
  /// component map of vector elements: identity, K (primal), B (dual)
  fn cmap(&self, pf: &PFem, a: usize, c: usize) -> f64 {
    let n = self.geo.n;
    match pf.vectorial() {
      Vectorial::None => (a == c) as u8 as f64,
      Vectorial::Primal => self.geo.k[a + n * c],
      Vectorial::Dual => self.geo.b[a + n * c],
    }
  }

  fn populate_dofs(&self, v: usize, out: &mut Vec<Option<usize>>) {
    let var = &self.vars[v];
    out.clear();
    match var.kind {
      GaKind::Fem(mf_dofs) if mf_dofs.reduction().is_some() => out.extend(
        mf_dofs
          .numbering
          .ind_basic_dof_of_element(self.cv)
          .iter()
          .map(|&d| var.toff.map(|t| t + d)),
      ),
      GaKind::Fem(mf_dofs) => out.extend(
        mf_dofs
          .numbering
          .ind_basic_dof_of_element(self.cv)
          .iter()
          .map(|&d| Some(var.off + d)),
      ),
      GaKind::Fixed(n) => out.extend((0..n).map(|k| Some(var.off + k))),
      // the components at the current point (none outside the region)
      GaKind::Im(pts, ts) => {
        let k = pts.index_of_point(self.cv, self.ip);
        out.extend((0..ts).map(|c| k.map(|k| var.off + k * ts + c)));
      }
      GaKind::Interp(_, tr) => {
        if var.tg.stamp == self.trs[tr].stamp
          && let Some(f) = &var.tg.f
        {
          out.extend_from_slice(&f.dofs);
        }
        out.resize(var.nbmax, None);
      }
    }
  }
}

impl crate::ga::Ext for GaEnv<'_> {
  fn xfem_base(&mut self, v: usize, der: u8, side: i8, out: &mut [f64]) -> Result<()> {
    // elements not defined on the real element: no side
    if self.vars[v].real.is_none() {
      match der {
        0 => self.val_base(v, out),
        1 => self.grad_base(v, out),
        _ => self.hess_base(v, out)?,
      }
      return Ok(());
    }
    let (val, grad, hess) = self.vars[v].real_xfem[usize::from(side < 0)]
      .as_ref()
      .ok_or_else(|| {
        Error::InvalidArgument("Xfem_plus/Xfem_minus: no values on this side".into())
      })?;
    out.copy_from_slice(match der {
      0 => val,
      1 => grad,
      _ => hess,
    });
    Ok(())
  }

  fn matrix_opt10(&mut self, v1: usize, v2: usize, q: usize, elem: &[f64]) {
    let (mut ra, mut rb) = (
      std::mem::take(&mut self.rows),
      std::mem::take(&mut self.cols),
    );
    self.populate_dofs(v1, &mut ra);
    if v2 != v1 {
      self.populate_dofs(v2, &mut rb);
    }
    let cols = if v2 == v1 { &ra } else { &rb };
    let (mut r1, mut r2) = (
      std::mem::take(&mut self.rows1),
      std::mem::take(&mut self.cols1),
    );
    // the norm once for the components
    let ninf = elem.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    for c in 0..q {
      r1.clear();
      r1.extend(ra.chunks_exact(q).map(|x| x[c]));
      r2.clear();
      r2.extend(cols.chunks_exact(q).map(|x| x[c]));
      self.target.elem_norm(&r1, &r2, elem, ninf);
    }
    (self.rows, self.cols, self.rows1, self.cols1) = (ra, rb, r1, r2);
  }

  fn matrix(&mut self, v1: usize, v2: usize, elem: &[f64]) {
    let (mut ra, mut rb) = (
      std::mem::take(&mut self.rows),
      std::mem::take(&mut self.cols),
    );
    self.populate_dofs(v1, &mut ra);
    if v2 != v1 {
      self.populate_dofs(v2, &mut rb);
    }
    self
      .target
      .elem(&ra, if v2 == v1 { &ra } else { &rb }, elem);
    (self.rows, self.cols) = (ra, rb);
  }

  fn cached_residual(&mut self, v: usize, out: &mut [f64]) -> Result<()> {
    let mut ra = std::mem::take(&mut self.rows);
    self.populate_dofs(v, &mut ra);
    let r: &[f64] = self.cond.as_ref().map_or(&[], |c| &c.residual[..]);
    for (o, d) in out.iter_mut().zip(&ra) {
      *o = d.and_then(|d| r.get(d).copied()).unwrap_or(0.0);
    }
    self.rows = ra;
    Ok(())
  }

  fn cond_coupling(&mut self, q: usize, j: usize, k: &[f64]) {
    let (mut ra, mut rb) = (
      std::mem::take(&mut self.rows),
      std::mem::take(&mut self.cols),
    );
    self.populate_dofs(q, &mut ra);
    self.populate_dofs(j, &mut rb);
    if let Some(c) = self.cond.as_deref_mut() {
      let nq = ra.len();
      for (b, cb) in rb.iter().enumerate() {
        let Some(cb) = cb else { continue };
        for (a, rq) in ra.iter().enumerate() {
          if let Some(rq) = rq {
            c.kqj.add_entry(*rq, *cb, k[a + nq * b]);
          }
        }
      }
    }
    (self.rows, self.cols) = (ra, rb);
  }

  fn cond_rq(&mut self, q: usize, r: &[f64]) {
    let mut ra = std::mem::take(&mut self.rows);
    self.populate_dofs(q, &mut ra);
    if let Some(c) = self.cond.as_deref_mut() {
      for (d, &x) in ra.iter().zip(r) {
        if let Some(d) = d {
          c.rq[*d] = x;
        }
      }
    }
    self.rows = ra;
  }

  fn cond_vector(&mut self, v: usize, elem: &[f64]) {
    let mut ra = std::mem::take(&mut self.rows);
    self.populate_dofs(v, &mut ra);
    if let Some(c) = self.cond.as_deref_mut() {
      for (d, &x) in ra.iter().zip(elem) {
        if let Some(d) = d {
          c.rhs[*d] += x;
        }
      }
    }
    self.rows = ra;
  }

  fn vector(&mut self, v: usize, elem: &[f64]) {
    let var = &self.vars[v];
    match var.kind {
      GaKind::Fem(mf_dofs) if mf_dofs.reduction().is_some() => {
        if let Some(t) = var.toff {
          for (&e, &d) in elem
            .iter()
            .zip(mf_dofs.numbering.ind_basic_dof_of_element(self.cv))
          {
            self.target.entry(t + d, 0, e);
          }
        }
      }
      GaKind::Fem(mf_dofs) => {
        for (&e, &d) in elem
          .iter()
          .zip(mf_dofs.numbering.ind_basic_dof_of_element(self.cv))
        {
          self.target.entry(var.off + d, 0, e);
        }
      }
      GaKind::Fixed(_) => {
        for (k, &e) in elem.iter().enumerate() {
          self.target.entry(var.off + k, 0, e);
        }
      }
      GaKind::Im(pts, ts) => {
        if let Some(k) = pts.index_of_point(self.cv, self.ip) {
          for (c, &e) in elem.iter().enumerate().take(ts) {
            self.target.entry(var.off + k * ts + c, 0, e);
          }
        }
      }
      GaKind::Interp(..) => {
        let mut ra = std::mem::take(&mut self.rows);
        self.populate_dofs(v, &mut ra);
        for (&e, d) in elem.iter().zip(&ra) {
          if let Some(d) = d {
            self.target.entry(*d, 0, e);
          }
        }
        self.rows = ra;
      }
    }
  }

  fn slice_dofs(&mut self, v: usize, out: &mut [f64]) {
    let var = &self.vars[v];
    let GaKind::Fem(mf_dofs) = var.kind else {
      return;
    };
    // (the values on the basic DOFs for a reduced space)
    for (o, &d) in out
      .iter_mut()
      .zip(mf_dofs.numbering.ind_basic_dof_of_element(self.cv))
    {
      *o = var.value[d];
    }
  }

  fn hess_base(&mut self, v: usize, out: &mut [f64]) -> Result<()> {
    // GetFEM's hess_base_value: B (H_ref - htau.grad) B^T
    let var = &self.vars[v];
    if let Some((_, _, h)) = &var.real {
      out.copy_from_slice(h);
      return Ok(());
    }
    let pf = var.pf.as_ref().unwrap();
    let t = pf.target_dim();
    let gr = var.pc.as_ref().unwrap().grad[self.ip].data();
    let hr = var.hr[self.ip].data();
    let (n, p) = (self.geo.n, self.geo.p);
    let b = &self.geo.b;
    let nb = pf.nb_base();
    let td = out.len() / (n * n * var.nd());
    let mut tmp = std::mem::take(&mut self.tmp);
    tmp.clear();
    tmp.resize(nb * td * n * n, 0.0);
    let (mut href, mut gx) = (vec![0.0; p * p], vec![0.0; n]);
    for i in 0..nb {
      for a in 0..td {
        // curvature correction with the reference component
        let ac = a.min(t - 1);
        if !self.htau.is_empty() {
          for (l, g) in gx[..n].iter_mut().enumerate() {
            *g = (0..p)
              .map(|j| b[l + n * j] * gr[i + nb * (ac + t * j)])
              .sum();
          }
        }
        for (jk, h) in href[..p * p].iter_mut().enumerate() {
          *h = if t == 1 {
            hr[i + nb * jk]
          } else {
            (0..t)
              .map(|c| self.cmap(pf, a, c) * hr[i + nb * (c + t * jk)])
              .sum()
          };
          if !self.htau.is_empty() {
            *h -= (0..n).map(|l| self.htau[l + n * jk] * gx[l]).sum::<f64>();
          }
        }
        for r in 0..n {
          for s2 in 0..n {
            let mut s = 0.0;
            for j in 0..p {
              for k in 0..p {
                s += b[r + n * j] * b[s2 + n * k] * href[j + p * k];
              }
            }
            tmp[i + nb * (a + td * (r + n * s2))] = s;
          }
        }
      }
    }
    trans(self.vars[v].mt.as_ref(), &tmp, nb, out);
    self.tmp = tmp;
    Ok(())
  }

  fn trans_call(
    &mut self,
    tr: usize,
    y: &[f64],
    x: &[f64],
    strict: bool,
    ders: bool,
  ) -> Result<()> {
    let m = self.mesh;
    let n = m.dim();
    let t = &mut self.trs[tr];
    t.stamp += 1;
    t.tgt = None;
    t.hit = None;
    match t.rt {
      InitializedTransformation::Expression {
        source_geometry,
        target,
        boxes,
        ..
      } => {
        if *source_geometry != m.geometry_version() {
          return Err(crate::transform::wrong_mesh(t.name));
        }
        if let Some((cv, xref)) = boxes.locate(target, &y[..target.dim()])? {
          let gt = target.trans_of_convex(cv);
          let geo = gt.at(&target.points_of_convex(cv), &xref)?;
          t.tgt = Some(TPoint {
            mesh: target.clone(),
            cv,
            xref,
            geo,
            normal: None,
          });
        }
      }
      InitializedTransformation::Neighbor => {
        let f = self.face.ok_or_else(|| {
          Error::InvalidArgument(format!(
            "{} can only be applied to faces (term {})",
            t.name, self.expr
          ))
        })?;
        if let Some((cv2, f2)) = m.adjacent_face(self.cv, f) {
          let (gt2, g2) = (m.trans_of_convex(cv2), m.points_of_convex(cv2));
          // GetFEM geotrans_inv_convex default tolerance
          let inv = gt2.invert(&g2, &x[..n], 1e-11, false)?;
          if !inv.converged || gt2.convex_ref().is_in(&inv.xref)? >= 1e-4 {
            return Err(Error::InvalidArgument(format!(
              "inversion of the geometric transformation failed in {} (element {cv2})",
              t.name
            )));
          }
          let geo = gt2.at(&g2, &inv.xref)?;
          let un = unit_normal(&geo, &gt2.normals()[f2]);
          t.tgt = Some(TPoint {
            mesh: m.clone(),
            cv: cv2,
            xref: inv.xref,
            geo,
            normal: Some(un),
          });
        }
      }
      InitializedTransformation::Contact(c) => {
        let f = self.face;
        let ai = self.pim.as_ref().unwrap().approx()?;
        let (ws, var_mfs, ext) = (self.ws, &self.asm_args.var_mfs, &self.asm_args.ext);
        let disp =
          |v: &str, mm: &getfem_geom::mesh::MeshData, cv: usize| ws.disp(var_mfs, ext, v, mm, cv);
        let h = c.transform(&disp, m, self.cv, f, ai.point(self.ip), ders)?;
        if let Some((cv, f2, xref)) = &h.target {
          let ib = c.bounds.iter().position(|b| b.disp == h.disp_y).unwrap();
          let mesh = c.bounds[ib].mesh.clone();
          let gt = mesh.trans_of_convex(*cv);
          let geo = gt.at(&mesh.points_of_convex(*cv), xref)?;
          let un = match f2 {
            Some(f2) => unit_normal(&geo, &gt.normals()[*f2]),
            None => {
              let mut v = vec![0.0; mesh.dim()];
              unit_normal_of_k(geo.k.data(), v.len(), &mut v);
              v
            }
          };
          t.tgt = Some(TPoint {
            mesh,
            cv: *cv,
            xref: xref.clone(),
            geo,
            normal: Some(un),
          });
        }
        t.hit = Some(h);
      }
    }
    t.status = match &t.hit {
      Some(h) => h.status,
      None => t.tgt.is_some() as usize,
    };
    if strict && t.status == 0 {
      return Err(Error::InvalidArgument(format!(
        "no target element for transformation {} at a point of element {} (term {}); points without target need Interpolate_filter",
        t.name, self.cv, self.expr
      )));
    }
    if strict && t.status == 2 {
      return Err(Error::InvalidArgument(format!(
        "transformation {0} gives a rigid obstacle at a point of element {1} (term {2}); these points need Interpolate_filter({0}, ..., 2)",
        t.name, self.cv, self.expr
      )));
    }
    Ok(())
  }

  fn trans_der(&mut self, tr: usize, v: usize, out: &mut [f64]) -> Result<()> {
    out.fill(0.0);
    let t = &self.trs[tr];
    let Some(h) = &t.hit else {
      return Ok(());
    };
    let var = &self.vars[v];
    let (src, source) = match var.kind {
      GaKind::Fem(_) if var.name == h.disp_x => (&h.der_x, true),
      GaKind::Interp(_, t2) if t2 == tr && var.name == h.disp_y => (&h.der_y, false),
      _ => return Ok(()),
    };
    if src.is_empty() {
      return Ok(());
    }
    let n = self.geo.n;
    let (nt, tt) = (src.len() / n, out.len() / n);
    if nt > tt || (source && nt != tt) {
      return Err(Error::Internal(format!(
        "derivative of {} w.r.t. {}: {nt} test functions for {tt}",
        t.name, var.name
      )));
    }
    for j in 0..n {
      out[tt * j..tt * j + nt].copy_from_slice(&src[nt * j..nt * (j + 1)]);
    }
    Ok(())
  }

  fn trans_status(&self, tr: usize) -> usize {
    self.trs[tr].status
  }

  fn interp_x(&self, tr: usize, out: &mut [f64]) -> Result<()> {
    out.fill(0.0);
    let t = &self.trs[tr];
    let x = match (&t.tgt, &t.hit) {
      (Some(p), _) => &p.geo.x,
      (None, Some(crate::contact::Hit { x: Some(x), .. })) => x,
      _ => return Ok(()),
    };
    for (o, x) in out.iter_mut().zip(x) {
      *o = *x;
    }
    Ok(())
  }

  fn interp_elt(&self, tr: usize, b: bool, out: &mut [f64]) {
    out.fill(0.0);
    if let Some(p) = &self.trs[tr].tgt {
      let m = if b { &p.geo.b } else { &p.geo.k };
      let r0 = m.nrows();
      for r in 0..r0 {
        for j in 0..m.ncols() {
          if let Some(o) = out.get_mut(r + r0 * j) {
            *o = m[(r, j)];
          }
        }
      }
    }
  }

  fn interp_normal(&self, tr: usize, out: &mut [f64]) -> Result<()> {
    out.fill(0.0);
    let t = &self.trs[tr];
    if let Some(p) = &t.tgt {
      let nv = p.normal.as_ref().ok_or_else(|| {
        Error::InvalidArgument(format!(
          "Interpolate(Normal, {}): the transformation gives no target face (term {})",
          t.name, self.expr
        ))
      })?;
      out.copy_from_slice(&nv[..out.len()]);
    }
    Ok(())
  }

  fn interp_field(&mut self, v: usize, der: u8, test: bool, out: &mut [f64]) -> Result<()> {
    out.fill(0.0);
    let GaKind::Interp(mf_dofs, tr) = self.vars[v].kind else {
      return Ok(());
    };
    let t = &self.trs[tr];
    let Some(tp) = &t.tgt else {
      return Ok(());
    };
    let var = &mut self.vars[v];
    let num = &mf_dofs.numbering;
    if var.tg.stamp != t.stamp {
      var.tg.stamp = t.stamp;
      var.tg.f = None;
      let name = var.name;
      if num.linked_mesh().geometry_version() != tp.mesh.geometry_version() {
        // a contact target on the boundary of another variable
        if t.hit.is_some() {
          return Ok(());
        }
        return Err(Error::InvalidArgument(format!(
          "Interpolate({name}, {0}): {name} is not on the target mesh of {0}",
          t.name
        )));
      }
      let pf = num.fem_of_element(tp.cv).ok_or_else(|| {
        Error::InvalidArgument(format!("{name} has no finite element on element {}", tp.cv))
      })?;
      let gt = tp.mesh.trans_of_convex(tp.cv);
      let (val, grad, hess) = real_base_values(
        pf,
        gt,
        &tp.mesh.points_of_convex(tp.cv),
        &tp.xref,
        if self.needs_hess { 2 } else { 1 },
      )?;
      let ind = num.ind_basic_dof_of_element(tp.cv);
      let coef = ind.iter().map(|&d| var.value[d]).collect();
      // a reduced space: its temporary interval
      let mut dofs: Vec<Option<usize>> = ind
        .iter()
        .map(|&d| match mf_dofs.reduction() {
          Some(_) => var.toff.map(|t| t + d),
          None => Some(var.off + d),
        })
        .collect();
      dofs.resize(var.nbmax, None);
      var.tg.f = Some(PtField {
        nd: pf.nb_dof(),
        dofs,
        coef,
        val,
        grad,
        hess,
      });
    }
    let Some(f) = var.tg.f.as_ref() else {
      return Ok(());
    };
    let n = self.geo.n;
    let pf = num.fem_of_element(tp.cv).unwrap();
    let ft = pf.target_dim();
    let fq = num.qdim() / ft;
    let comps = fq * ft;
    let (tab, nc) = match der {
      0 => (&f.val, 1),
      1 | 3 => (&f.grad, n),
      _ => (&f.hess, n * n),
    };
    let ts = if der == 3 {
      out.len()
    } else {
      out.len() / (comps * nc)
    };
    let nd = f.nd;
    for i in 0..nd {
      for c in 0..fq {
        let k = i * fq + c;
        let u = if test { 1.0 } else { f.coef[k] };
        if u == 0.0 {
          continue;
        }
        for a in 0..ft {
          let comp = c * ft + a;
          for r in 0..nc {
            let o = match der {
              3 if r == comp => 0,
              3 => continue,
              _ => comp + comps * r,
            };
            let o = if test { k + ts * o } else { o };
            out[o] += u * tab[i + nd * (a + ft * r)];
          }
        }
      }
    }
    Ok(())
  }

  fn elem_trans(&mut self, e: usize, out: &mut [f64]) -> Result<()> {
    let (name, t, vs, vd) = self.ets[e];
    let gt = self.gt.as_ref().unwrap();
    let elem = |v: usize| {
      let var = &self.vars[v];
      let GaKind::Fem(mf_dofs) = var.kind else {
        unreachable!()
      };
      ElemFem {
        pf: var.pf.as_ref().unwrap(),
        qdim: mf_dofs.numbering.qdim(),
        gt,
        g: &self.geo.g,
        cv: self.cv,
      }
    };
    let (es, ed) = (elem(vs), elem(vd));
    let m = t.give_transformation(&es, &ed)?;
    let tdim = |e: &ElemFem| e.pf.nb_dof() * e.qdim / e.pf.target_dim();
    let (nr, nc) = (tdim(&ed), tdim(&es));
    if m.nrows() != nr || m.ncols() != nc || out.len() != nr * nc {
      return Err(Error::InvalidArgument(format!(
        "elementary transformation {name}: a {}x{} matrix for {nr} DOFs of {} and {nc} of {}",
        m.nrows(),
        m.ncols(),
        self.vars[vd].name,
        self.vars[vs].name
      )));
    }
    for k in 0..nc {
      for j in 0..nr {
        out[j + nr * k] = m[(j, k)];
      }
    }
    Ok(())
  }

  fn point_value(&mut self, v: &[f64]) -> Result<()> {
    self.target.point(self.cv, self.face, self.ip, v)
  }

  fn im_val(&mut self, v: usize, out: &mut [f64]) -> Result<()> {
    let var = &self.vars[v];
    let GaKind::Im(pts, ts) = var.kind else {
      return Ok(());
    };
    let k = pts.index_of_point(self.cv, self.ip).ok_or_else(|| {
      Error::InvalidArgument(format!(
        "im_data has no value at point {} of element {}",
        self.ip, self.cv
      ))
    })?;
    out.copy_from_slice(&var.value[k * ts..(k + 1) * ts]);
    Ok(())
  }

  fn val_base(&mut self, v: usize, out: &mut [f64]) {
    let var = &self.vars[v];
    if var.std {
      out.copy_from_slice(&var.pc.as_ref().unwrap().val[self.ip].data()[..out.len()]);
      return;
    }
    if let Some((val, _, _)) = &var.real {
      out.copy_from_slice(val);
      return;
    }
    let pf = var.pf.as_ref().unwrap();
    let t = pf.target_dim();
    let vr = var.pc.as_ref().unwrap().val[self.ip].data();
    let nb = pf.nb_base();
    let td = out.len() / var.nd();
    if t == 1 && var.mt.is_none() {
      out.copy_from_slice(&vr[..out.len()]);
      return;
    }
    // components: a <- sum_c C(a, c) c
    let mut tmp = std::mem::take(&mut self.tmp);
    tmp.clear();
    tmp.resize(nb * td, 0.0);
    for a in 0..td {
      for c in 0..t {
        let cc = self.cmap(pf, a, c);
        if cc != 0.0 {
          for i in 0..nb {
            tmp[i + nb * a] += cc * vr[i + nb * c];
          }
        }
      }
    }
    trans(self.vars[v].mt.as_ref(), &tmp, nb, out);
    self.tmp = tmp;
  }

  fn grad_base(&mut self, v: usize, out: &mut [f64]) {
    // real gradients: sum_j B_rj grad_ref_j
    let var = &self.vars[v];
    if var.std {
      // column k of the gradients: sum_j B_kj (column j of the reference
      // gradients) (GetFEM: mat_tmult)
      let gr = var.pc.as_ref().unwrap().grad[self.ip].data();
      let (n, p) = (self.geo.n, self.geo.p);
      let nb = out.len() / n;
      let b = &self.geo.b[..n * p];
      let gr = &gr[..nb * p];
      match p {
        1 => grad_ref::<1>(out, b, gr, n, nb),
        2 => grad_ref::<2>(out, b, gr, n, nb),
        3 => grad_ref::<3>(out, b, gr, n, nb),
        _ => {
          for (k, ok) in out.chunks_exact_mut(nb).enumerate() {
            let g0 = &gr[..nb];
            let b0 = b[k];
            for (o, &g) in ok.iter_mut().zip(g0) {
              *o = b0 * g;
            }
            for j in 1..p {
              let bj = b[k + n * j];
              for (o, &g) in ok.iter_mut().zip(&gr[nb * j..nb * (j + 1)]) {
                *o += bj * g;
              }
            }
          }
        }
      }
      return;
    }
    if let Some((_, gr, _)) = &var.real {
      out.copy_from_slice(gr);
      return;
    }
    let pf = var.pf.as_ref().unwrap();
    let t = pf.target_dim();
    let gr = var.pc.as_ref().unwrap().grad[self.ip].data();
    let (n, p) = (self.geo.n, self.geo.p);
    let nb = pf.nb_base();
    let td = out.len() / (n * var.nd());
    if t == 1 && var.mt.is_none() {
      for k in 0..n {
        for i in 0..nb {
          let mut s = 0.0;
          for j in 0..p {
            s += self.geo.b[k + n * j] * gr[i + nb * j];
          }
          out[i + nb * k] = s;
        }
      }
      return;
    }
    let mut tmp = std::mem::take(&mut self.tmp);
    tmp.clear();
    tmp.resize(nb * td * n, 0.0);
    for r in 0..n {
      for a in 0..td {
        for c in 0..t {
          let cc = self.cmap(pf, a, c);
          if cc == 0.0 {
            continue;
          }
          for i in 0..nb {
            let g: f64 = (0..p)
              .map(|j| self.geo.b[r + n * j] * gr[i + nb * (c + t * j)])
              .sum();
            tmp[i + nb * (a + td * r)] += cc * g;
          }
        }
      }
    }
    trans(self.vars[v].mt.as_ref(), &tmp, nb, out);
    self.tmp = tmp;
  }
}

impl Workspace {
  /// GetFEM's ga_exec: elements (or faces) of the region, points; the
  /// element instructions at the first point.
  fn assemble_ga(
    &self,
    gis: &crate::ga::InstructionSet,
    asm_args: &AssemblyArgs,
    region: &Region,
    expr: &str,
    target: &mut impl AssemblyTarget,
    (tmp, ext, unreduced, mut cond): (
      &BTreeMap<String, usize>,
      &TmpVarIntervals,
      Option<&mut UnreducedOut>,
      Option<&mut CondOut>,
    ),
  ) -> Result<()> {
    let m = asm_args.ims.linked_mesh();
    let set;
    let mut elems: Vec<(usize, Option<usize>)> = if region.id() == ALL {
      m.convex_index().map(|cv| (cv, None)).collect()
    } else {
      set = region.resolve(m)?;
      set.iter().collect()
    };
    // MPI: this rank's elements (GetFEM's region visitor with
    // intersect_with_mpi); the sums over the ranks in assembly_of
    if getfem_mpi::size() > 1 {
      let part = m.mpi_region();
      elems.retain(|&(cv, _)| part.is_in(cv));
    }
    // threads: equal blocks of the elements, each to its own outputs, added
    // in thread order (GetFEM mesh_region::partition_begin)
    let nt = getfem_linalg::threads::num_threads();
    let shape = target.thread_target_shape();
    let (Some(shape), true) = (shape, nt > 1 && elems.len() >= nt) else {
      return self.assemble_elems(
        gis,
        asm_args,
        &elems,
        expr,
        target,
        (tmp, ext, unreduced, cond),
      );
    };
    // the first block into the outputs themselves, each other block by a
    // thread into its own outputs, emptied by that thread (as GetFEM:
    // thread 0 into the original)
    let mut outs = std::mem::take(&mut *self.assembled.thread_outs.lock().unwrap());
    let blocks: Vec<_> = elems.chunks(elems.len().div_ceil(nt)).collect();
    outs.resize(blocks.len() - 1);
    let unreduced_shape = unreduced.as_deref().map(UnreducedOut::shape);
    let cond_shape = cond.as_deref().map(CondOut::shape);
    let res: Result<()> = std::thread::scope(|scope| {
      let threads: Vec<_> = blocks[1..]
        .iter()
        .zip(outs.targets.iter_mut())
        .zip(outs.unreduced.iter_mut().zip(outs.conds.iter_mut()))
        .map(
          |((block, thread_target), (thread_unreduced, thread_cond))| {
            let cond_shape = &cond_shape;
            scope.spawn(move || {
              thread_target.reset(shape);
              let thread_unreduced = unreduced_shape.map(|s| thread_unreduced.reset(s));
              let thread_cond = cond_shape.as_ref().map(|s| thread_cond.reset(s));
              self.assemble_elems(
                gis,
                asm_args,
                block,
                expr,
                thread_target,
                (tmp, ext, thread_unreduced, thread_cond),
              )
            })
          },
        )
        .collect();
      let r = self.assemble_elems(
        gis,
        asm_args,
        blocks[0],
        expr,
        target,
        (tmp, ext, unreduced, cond.as_deref_mut()),
      );
      threads
        .into_iter()
        .try_for_each(|t| t.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))?;
      r
    });
    let res = res.and_then(|_| target.add_thread_targets(&mut outs.targets));
    if let (Ok(_), Some(c)) = (&res, cond) {
      c.add(&mut outs.conds);
    }
    *self.assembled.thread_outs.lock().unwrap() = outs;
    res
  }

  /// The instructions on the elements (and faces) elems
  fn assemble_elems(
    &self,
    gis: &crate::ga::InstructionSet,
    asm_args: &AssemblyArgs,
    elems: &[(usize, Option<usize>)],
    expr: &str,
    target: &mut dyn AssemblyTarget,
    (tmp, ext, unreduced, mut cond): (
      &BTreeMap<String, usize>,
      &TmpVarIntervals,
      Option<&mut UnreducedOut>,
      Option<&mut CondOut>,
    ),
  ) -> Result<()> {
    // the unreduced terms to out, then folded into target (by each thread
    // its own, as GetFEM)
    if let Some(out) = unreduced {
      let mut unreduced_target = UnreducedTarget {
        inner: &mut *target,
        out: &mut *out,
      };
      let conds = cond.as_deref_mut();
      self.assemble_elems(
        gis,
        asm_args,
        elems,
        expr,
        &mut unreduced_target,
        (tmp, ext, None, conds),
      )?;
      fold_unreduced(ext, out, target);
      if let Some(c) = cond {
        c.fold(out.n, ext);
      }
      return Ok(());
    }
    let m = asm_args.ims.linked_mesh();
    let mut trs = vec![];
    for name in &gis.trans {
      let Some(rt) = asm_args.transformations.get(name) else {
        return Err(Error::InvalidArgument(format!(
          "unknown transformation {name}"
        )));
      };
      trs.push(GaTr {
        name,
        rt,
        tgt: None,
        status: 0,
        hit: None,
        stamp: 0,
      });
    }
    let mut vars = vec![];
    for key in &gis.vars {
      // a variable on the target elements of a transformation
      let (name, tr) = match key.split_once('\u{1}') {
        Some((v, t)) => (v, gis.trans.iter().position(|x| x == t)),
        None => (key.as_str(), None),
      };
      let unknown = || Error::InvalidArgument(format!("unknown variable {name}"));
      let st = self.variables.get(name).ok_or_else(unknown)?;
      let kind = match &st.space {
        Space::Fem(_) => match (asm_args.var_mfs.get(name), tr) {
          (Some(mf_dofs), Some(tr)) => GaKind::Interp(mf_dofs, tr),
          (Some(mf_dofs), None) => GaKind::Fem(mf_dofs),
          _ => return Err(unknown()),
        },
        Space::Fixed => GaKind::Fixed(st.value.len()),
        Space::Im(imd) => GaKind::Im(
          asm_args.var_imds.get(name).ok_or_else(unknown)?,
          imd.tensor_size(),
        ),
      };
      // a space with reduction matrices: on the basic DOFs (GetFEM
      // extend_variable_in_gis)
      let (value, toff) = match &kind {
        GaKind::Fem(mf_dofs) | GaKind::Interp(mf_dofs, _) if mf_dofs.reduction().is_some() => {
          let v = mf_dofs.extend_vector(&st.value)?;
          if v.len() != mf_dofs.numbering.nb_basic_dof() {
            return Err(Error::Unsupported {
              expr: expr.to_string(),
              pos: 0,
              msg: format!(
                "{name}: several values per DOF of a space with reduction matrices: not yet available"
              ),
            });
          }
          (std::borrow::Cow::Owned(v), tmp.get(name).copied())
        }
        _ => (std::borrow::Cow::Borrowed(&st.value[..]), None),
      };
      vars.push(GaVar {
        kind,
        value,
        off: st.interval.as_ref().map_or(0, |r| r.start),
        toff,
        pf: None,
        pc: None,
        hr: vec![],
        mt: None,
        real: None,
        real_xfem: [None, None],
        equiv: true,
        std: true,
        tg: GaTarget::default(),
        nbmax: 0,
        name,
      });
    }
    let mut ets = vec![];
    for (name, vs, vd) in &gis.elts {
      let Some(t) = self.elem_transformations.get(name) else {
        return Err(Error::InvalidArgument(format!(
          "unknown elementary transformation {name}"
        )));
      };
      ets.push((name.as_str(), t, *vs as usize, *vd as usize));
    }
    let mut es = gis.exec_state();
    es.x = vec![0.0; m.dim()];
    es.normal = vec![0.0; m.dim()];
    for (k, v) in vars.iter_mut().enumerate() {
      // fixed-size and im variables (the components at the point)
      if let GaKind::Fixed(nb) | GaKind::Im(_, nb) = v.kind {
        es.dims[k] = crate::ga::VarDims {
          nb,
          td: 1,
          qdim: 1,
          n: m.dim(),
        };
      }
      // tests on target elements: sized by the largest element (padded)
      if let GaKind::Interp(mf_dofs, _) = v.kind {
        let num = &mf_dofs.numbering;
        let fems: Vec<&PFem> = num
          .convex_index()
          .filter_map(|cv| num.fem_of_element(cv))
          .collect();
        let Some(f0) = fems.first() else {
          return Err(Error::InvalidArgument(format!(
            "{}: no finite element on the target mesh",
            v.name
          )));
        };
        let td = f0.target_dim();
        if fems.iter().any(|f| f.target_dim() != td) {
          return Err(Error::Unsupported {
            expr: expr.to_string(),
            pos: 0,
            msg: format!("{}: elements of different target dimensions", v.name),
          });
        }
        let nb = fems.iter().map(|f| f.nb_dof()).max().unwrap();
        v.nbmax = nb * (num.qdim() / td);
        es.dims[k] = crate::ga::VarDims {
          nb,
          td,
          qdim: num.qdim(),
          n: m.dim(),
        };
      }
    }
    es.layout(gis);
    // the elements of the variables: qdim and Hessians supported
    let mut seen: Vec<PFem> = vec![];
    for &(cv, _) in elems {
      for v in &vars {
        let GaKind::Fem(mf_dofs) = v.kind else {
          continue;
        };
        let Some(pf) = mf_dofs.numbering.fem_of_element(cv) else {
          return Err(Error::InvalidArgument(format!(
            "{} has no finite element on element {cv}",
            v.name
          )));
        };
        if seen.iter().any(|s| Arc::ptr_eq(s, pf)) {
          continue;
        }
        let t = pf.target_dim();
        let q = mf_dofs.numbering.qdim();
        // GetFEM: qdim a multiple of the target dimension (mesh_fem)
        let td = if t > 1 && !matches!(pf.vectorial(), Vectorial::None) {
          m.dim()
        } else {
          t
        };
        if q % td != 0 {
          return Err(Error::Unsupported {
            expr: expr.to_string(),
            pos: 0,
            msg: format!(
              "{}: element {} of target dimension {td} on a space of qdim {q}",
              v.name,
              pf.name()
            ),
          });
        }
        seen.push(pf.clone());
      }
    }
    let mut pool = PrecompPool::default();
    let mut env = GaEnv {
      vars,
      geo: crate::ga::GeoCtx {
        n: m.dim(),
        ..Default::default()
      },
      cv: 0,
      face: None,
      ip: 0,
      htau: vec![],
      tmp: vec![],
      needs_hess: gis.needs_hess,
      trs,
      ets,
      gt: None,
      pim: None,
      ws: self,
      asm_args,
      mesh: m,
      expr,
      target,
      rows: vec![],
      cols: vec![],
      rows1: vec![],
      cols1: vec![],
      cond,
    };
    let has_contact = env
      .trs
      .iter()
      .any(|t| matches!(t.rt, InitializedTransformation::Contact(_)));
    elem_loop(
      gis,
      asm_args,
      elems,
      (&mut es, &mut env),
      &mut pool,
      has_contact,
    )?;
    if gis.scalar {
      env.target.entry(0, 0, es.acc);
    }
    Ok(())
  }
}

/// The instructions on the elements: the per-element work only (setup in
/// assemble_elems, rare work out of line: a compact loop)
#[inline(never)]
fn elem_loop(
  gis: &crate::ga::InstructionSet,
  asm_args: &AssemblyArgs,
  elems: &[(usize, Option<usize>)],
  (es, env): (&mut crate::ga::ExecState, &mut GaEnv),
  pool: &mut PrecompPool,
  has_contact: bool,
) -> Result<()> {
  let m = env.mesh;
  let mut old_pim: Option<PIntegration> = None;
  // the geometric precomp of the previous element: looked up only when the
  // transformation or the method changes (no reference count per element:
  // shared by the threads)
  let mut old_gp: Option<(PGeoTrans, PIntegration, Arc<GeoPrecomp>)> = None;
  for &(cv, face) in elems {
    let Ok(pim) = asm_args.ims.int_method_of_element(cv) else {
      continue;
    };
    if pim.is_none() {
      continue;
    }
    let ai = pim.approx()?;
    let gt = m.trans_of_convex(cv);
    m.points_of_convex_into(cv, &mut env.geo.g);
    env.geo.p = gt.dim();
    env.cv = cv;
    env.face = face;
    if !gis.elts.is_empty() {
      env.gt = Some(gt.clone());
    }
    if has_contact {
      env.pim = Some(pim.clone());
    }
    if !old_gp
      .as_ref()
      .is_some_and(|(g, p, _)| Arc::ptr_eq(g, gt) && Arc::ptr_eq(p, pim))
    {
      old_gp = Some((gt.clone(), pim.clone(), pool.geo(gt, pim)?));
    }
    let gp = &old_gp.as_ref().unwrap().2;
    let new_pim = !old_pim.as_ref().is_some_and(|o| Arc::ptr_eq(o, pim));
    // the elements of the variables: tables, layout if they changed
    let mut relayout = false;
    for (k, v) in env.vars.iter_mut().enumerate() {
      let GaKind::Fem(mf_dofs) = v.kind else {
        continue;
      };
      let pf = mf_dofs.numbering.fem_of_element(cv).unwrap();
      if new_pim || !v.pf.as_ref().is_some_and(|o| Arc::ptr_eq(o, pf)) {
        let d = fem_changed(v, pf, (pim, ai), pool, gis.needs_hess, m.dim())?;
        if es.dims[k] != d {
          es.dims[k] = d;
          relayout = true;
        }
      }
      if !v.equiv {
        v.mt = pf.mat_trans(&env.geo.g, gt)?;
      }
    }
    if relayout {
      es.layout(gis);
    }
    if new_pim {
      old_pim = Some(pim.clone());
    }
    let (first, nb) = match face {
      Some(f) => (ai.ind_first_point_on_face(f), ai.nb_points_on_face(f)),
      None => (0, ai.nb_points_on_convex()),
    };
    if gis.needs_elt_size {
      es.elt_size = gt.element_size(&env.geo.g)?;
    }
    let mut jac = 0.0;
    // Normal on the zero sets of a level set mim (GetFEM
    // ga_instruction_level_set_normal_vector)
    let ls_normal = if face.is_none() && gis.needs_normal {
      level_sets_of_convex(asm_args, cv)?
    } else {
      vec![]
    };
    let any_real = env.vars.iter().any(|v| v.real.is_some());
    let lin = gt.is_linear();
    let curved = gis.needs_hess && !lin;
    es.nbpt = nb;
    for ipt in 0..nb {
      let ip = first + ipt;
      env.ip = ip;
      es.ipt = ipt;
      if ipt == 0 || !lin {
        env.geo.at(gp, ip)?;
        jac = env.geo.j.abs();
        if let Some(f) = face {
          jac *= env.geo.face_normal(gt, f, &mut es.normal);
        } else if gis.needs_normal && ls_normal.is_empty() {
          element_unit_normal(&env.geo, cv, &mut es.normal)?;
        }
      }
      if !ls_normal.is_empty() {
        level_set_normal(
          &ls_normal,
          ai.level_set_of_point(ip),
          ai.point(ip),
          &env.geo.b,
          &mut es.normal,
        )?;
      }
      es.coeff = ai.coeff(ip) * jac;
      if gis.needs_x || any_real {
        env.geo.x_at(gp, ip, &mut es.x);
      }
      if any_real {
        let c = FemCtx {
          pgt: gt,
          g: &env.geo.g,
          xref: ai.point(ip),
          x: &es.x[..m.dim()],
          k: &env.geo.k,
          b: &env.geo.b,
          xfem_side: 0,
          xfem_ls: ai.level_set_of_point(ip),
          pim: Some((pim, ip)),
        };
        let der = if gis.needs_hess { 2 } else { 1 };
        real_values_at(&mut env.vars, &c, der, &gis.xfem_sides)?;
      }
      if gis.needs_kb {
        es.k.clone_from(&env.geo.k);
        es.b.clone_from(&env.geo.b);
      }
      if curved {
        trans_hessian(&env.geo, gt, ai.point(ip), &mut env.htau);
      }
      if gis.needs_curvature {
        element_curvature(&env.geo, gt, ai.point(ip), &mut es.curvature)?;
      }
      if ipt == 0 {
        gis.run(&gis.elt_instructions, es, env)?;
      }
      gis.run(&gis.instructions, es, env)?;
    }
  }
  Ok(())
}

/// The terms grouped by mim and region: the same mim and the same stored
/// region (or both everywhere)
fn groups<'a>(
  terms: &[(&'a str, &'a MeshIm, &'a Region)],
) -> Vec<(&'a MeshIm, &'a Region, Vec<&'a str>)> {
  let mut groups: Vec<(&MeshIm, &Region, Vec<&str>)> = vec![];
  for &(e, mim, r) in terms {
    let same =
      |g: &(&MeshIm, &Region, Vec<&str>)| g.0.same(mim) && !r.is_temporary() && g.1.id() == r.id();
    match groups.iter_mut().find(|g| same(g)) {
      Some(g) => g.2.push(e),
      None => groups.push((mim, r, vec![e])),
    }
  }
  groups
}

// the rare per-element work of assemble_elems, out of line (a compact
// element loop: lean by default)

/// A variable's element changed (or the integration method): its kind,
/// precomputation, reference Hessians; its dimensions returned
#[inline(never)]
fn fem_changed(
  v: &mut GaVar,
  pf: &PFem,
  (pim, ai): (&PIntegration, &getfem_fem::integration::ApproxIntegration),
  pool: &mut PrecompPool,
  needs_hess: bool,
  n: usize,
) -> Result<crate::ga::VarDims> {
  let real = pf.real_element_defined();
  v.real = real.then(|| (vec![], vec![], vec![]));
  v.equiv = pf.is_equivalent();
  v.std = v.equiv && !real && pf.target_dim() == 1;
  if !real {
    v.pc = Some(pool.fem(pf, pim)?);
  }
  if needs_hess && !real {
    v.hr.clear();
    for xref in ai.points() {
      v.hr.push(pf.hess_base_value(xref)?);
    }
  }
  let t = pf.target_dim();
  let GaKind::Fem(mf_dofs) = v.kind else {
    unreachable!()
  };
  v.pf = Some(pf.clone());
  Ok(crate::ga::VarDims {
    nb: if pf.is_equivalent() && !real {
      pf.nb_base()
    } else {
      pf.nb_dof()
    },
    td: if t > 1 && !matches!(pf.vectorial(), Vectorial::None) {
      n
    } else {
      t
    },
    qdim: mf_dofs.numbering.qdim(),
    n,
  })
}

/// The level sets of a level set mim on element cv (Normal on their zero
/// sets)
#[inline(never)]
fn level_sets_of_convex(
  asm_args: &AssemblyArgs,
  cv: usize,
) -> Result<Vec<(usize, getfem_fem::level_set::ConvexLevelSet)>> {
  asm_args
    .ims
    .normal_level_sets()
    .iter()
    .map(|l| Ok((l.id(), l.of_convex(cv, 0)?)))
    .collect()
}

/// The values of the real elements at the point (and on the sides of the
/// level set, Xfem_plus/minus, where read)
#[inline(never)]
fn real_values_at(vars: &mut [GaVar], c: &FemCtx, der: usize, xfem_sides: &[u8]) -> Result<()> {
  for (k, v) in vars.iter_mut().enumerate() {
    if let Some(rv) = &mut v.real {
      let pf = v.pf.as_ref().unwrap();
      *rv = pf.real_values(c, der)?;
      let sides = xfem_sides.get(k).copied().unwrap_or(0);
      for (j, side) in [(0, 1), (1, -1)] {
        if sides & (1 << j) != 0 {
          let cs = FemCtx {
            xfem_side: side,
            ..*c
          };
          v.real_xfem[j] = Some(pf.real_values(&cs, der)?);
        }
      }
    }
  }
  Ok(())
}

/// htau = G hess(phi) at xref (a curved transformation)
#[inline(never)]
fn trans_hessian(geo: &crate::ga::GeoCtx, gt: &PGeoTrans, xref: &[f64], htau: &mut Vec<f64>) {
  let gh = gt.hess(xref);
  let (n, p, g) = (geo.n, geo.p, &geo.g);
  htau.clear();
  htau.extend((0..n * p * p).map(|lj| {
    let (l, jk) = (lj % n, lj / n);
    (0..g.ncols()).map(|a| g[(l, a)] * gh[(a, jk)]).sum::<f64>()
  }));
}

/// element_curvature = grad_s n = -B b B^T (n x n) on a curve in 2D or a
/// surface in 3D: n the unit normal oriented by the parametrization (K_1 x
/// K_2 in 3D, (K_21, -K_11) in 2D), b_ac = n . d2x/dxi_a dxi_c; zero for a
/// linear transformation
fn element_curvature(
  geo: &crate::ga::GeoCtx,
  gt: &PGeoTrans,
  xref: &[f64],
  out: &mut Vec<f64>,
) -> Result<()> {
  let (n, p) = (geo.n, geo.p);
  if p + 1 != n || !(n == 2 || n == 3) {
    return Err(Error::InvalidArgument(
      "element_curvature is only defined for curves in 2D and surfaces in 3D".into(),
    ));
  }
  out.clear();
  out.resize(n * n, 0.0);
  if gt.is_linear() {
    return Ok(());
  }
  let b = &geo.b;
  let mut nrm = [0.0; 3];
  unit_normal_of_k(&geo.k, n, &mut nrm[..n]);
  // b_ac = n . G hess(phi)_ac
  let gh = gt.hess(xref);
  let g = &geo.g;
  let mut bac = [0.0; 4];
  for (ac, bv) in bac[..p * p].iter_mut().enumerate() {
    *bv = (0..n)
      .map(|l| nrm[l] * (0..g.ncols()).map(|a| g[(l, a)] * gh[(a, ac)]).sum::<f64>())
      .sum();
  }
  for j in 0..n {
    for i in 0..n {
      let mut s = 0.0;
      for a in 0..p {
        for c in 0..p {
          s += b[i + n * a] * bac[a + p * c] * b[j + n * c];
        }
      }
      out[i + n * j] = -s;
    }
  }
  Ok(())
}

/// unit normal of a curve in 2D or a surface in 3D, oriented by the
/// parametrization: (K_21, -K_11) in 2D, K_1 x K_2 in 3D (K n x p
/// column-major)
pub(crate) fn unit_normal_of_k(k: &[f64], n: usize, out: &mut [f64]) {
  if n == 2 {
    out.copy_from_slice(&[k[1], -k[0]]);
  } else {
    out.copy_from_slice(&[
      k[1] * k[5] - k[2] * k[4],
      k[2] * k[3] - k[0] * k[5],
      k[0] * k[4] - k[1] * k[3],
    ]);
  }
  let nn = out.iter().map(|v| v * v).sum::<f64>().sqrt();
  out.iter_mut().for_each(|v| *v /= nn);
}

/// Normal inside element cv (not on a face): the element's own unit normal
/// on a curve in 2D or a surface in 3D (as element_curvature); undefined,
/// an error, on other element dimensions (GetFEM: an empty vector)
#[inline(never)]
fn element_unit_normal(geo: &crate::ga::GeoCtx, cv: usize, out: &mut [f64]) -> Result<()> {
  if geo.p + 1 != geo.n || !(geo.n == 2 || geo.n == 3) {
    return Err(Error::InvalidArgument(format!(
      "Normal inside element {cv} (of dimension {} in {}D): defined on faces, and inside curves in 2D and surfaces in 3D only",
      geo.p, geo.n
    )));
  }
  unit_normal_of_k(&geo.k, geo.n, out);
  Ok(())
}

/// Derivatives of an analysed tree w.r.t. each of its variables, analysed.
/// unit outward normal B n_ref / |B n_ref| at a point of a face
fn unit_normal(geo: &PointGeom, nr: &[f64]) -> Vec<f64> {
  let mut un: Vec<f64> = (0..geo.b.nrows())
    .map(|r| (0..geo.b.ncols()).map(|j| geo.b[(r, j)] * nr[j]).sum())
    .collect();
  let nn = un.iter().map(|v| v * v).sum::<f64>().sqrt();
  un.iter_mut().for_each(|v| *v /= nn);
  un
}

/// The unit normal of the level set the point is on (`id`, from the cut)
/// at xref: its reference gradient through B (N x P), normalized (GetFEM
/// mesh_im_level_set::compute_normal_vector: of the closest level set)
#[inline(never)]
fn level_set_normal(
  lss: &[(usize, getfem_fem::level_set::ConvexLevelSet)],
  id: usize,
  xref: &[f64],
  b: &[f64],
  out: &mut [f64],
) -> Result<()> {
  let Some((_, l)) = lss.iter().find(|(i, _)| *i == id) else {
    return Err(Error::InvalidArgument(
      "Normal: the point is on no level set".into(),
    ));
  };
  let un = l.grad(xref)?;
  let n = out.len();
  for (r, o) in out.iter_mut().enumerate() {
    *o = un.iter().enumerate().map(|(k, u)| b[r + n * k] * u).sum();
  }
  let no = out.iter().map(|v| v * v).sum::<f64>().sqrt();
  if no != 0.0 {
    out.iter_mut().for_each(|v| *v /= no);
  }
  Ok(())
}

fn derive_all(
  tree: &crate::semantic::Tree,
  defs: &Defs,
  ctx: &Ctx,
  order: usize,
  vars: &[(String, String)],
) -> Result<Vec<crate::semantic::Tree>> {
  let mut out = vec![];
  for (v, tr) in vars {
    if let Some(d) = crate::derive::derive(tree, defs, v, tr, order, None)? {
      let (d, _) = analyse(d, defs, ctx)?;
      if d.root.is_some() {
        out.push(d);
      }
    }
  }
  Ok(out)
}

/// Receiver of values at points: (convex, face, point index, components).
type PointTarget<'a> = dyn FnMut(usize, Option<usize>, usize, &[f64]) -> Result<()> + 'a;

/// Spaces and integration methods as seen by one assembly.
struct AssemblyArgs {
  ims: Arc<Ims>,
  var_mfs: BTreeMap<String, MfDofs>,
  var_imds: BTreeMap<String, Arc<ImdPoints>>,
  transformations: BTreeMap<String, InitializedTransformation>,
  /// the values on the basic DOFs of the contact displacements on spaces
  /// with reduction matrices
  ext: BTreeMap<String, Vec<f64>>,
}

impl Workspace {
  /// The displacement var on element cv of m (contact transformations;
  /// ext: its values on the basic DOFs of a space with reduction matrices).
  fn disp(
    &self,
    var_mfs: &BTreeMap<String, MfDofs>,
    ext: &BTreeMap<String, Vec<f64>>,
    var: &str,
    m: &getfem_geom::mesh::MeshData,
    cv: usize,
  ) -> Result<Disp> {
    let (Some(st), Some(mf_dofs)) = (self.variables.get(var), var_mfs.get(var)) else {
      return Err(Error::InvalidArgument(format!(
        "contact displacement {var} is not a finite element variable"
      )));
    };
    let value = match mf_dofs.reduction() {
      Some(_) => ext
        .get(var)
        .ok_or_else(|| Error::Internal(format!("contact displacement {var}: not extended")))?,
      None => &st.value,
    };
    let num = &mf_dofs.numbering;
    if num.linked_mesh().geometry_version() != m.geometry_version() {
      return Err(Error::InvalidArgument(format!(
        "contact displacement {var} is not on the mesh of its contact boundary"
      )));
    }
    let pf = num.fem_of_element(cv).ok_or_else(|| {
      Error::InvalidArgument(format!("{var} has no finite element on element {cv}"))
    })?;
    if num.qdim() != m.dim() || pf.target_dim() != 1 {
      return Err(Error::InvalidArgument(format!(
        "contact displacement {var}: a scalar element of qdim {} expected",
        m.dim()
      )));
    }
    let coef = num
      .ind_basic_dof_of_element(cv)
      .iter()
      .map(|&d| value[d])
      .collect();
    Ok(Disp {
      pf: pf.clone(),
      gt: m.trans_of_convex(cv).clone(),
      g: m.points_of_convex(cv),
      coef,
    })
  }

  fn asm_args(&self, mim: &MeshIm) -> Result<AssemblyArgs> {
    let mut var_mfs = BTreeMap::new();
    let mut var_imds = BTreeMap::new();
    for (name, st) in &self.variables {
      match &st.space {
        Space::Fem(mf) => {
          var_mfs.insert(name.clone(), mf.dofs()?);
        }
        Space::Im(imd) => {
          var_imds.insert(name.clone(), imd.points()?);
        }
        Space::Fixed => {}
      }
    }
    // contact transformations: by init_contact_transformations, for the
    // instruction sets using them
    let mut transformations = BTreeMap::new();
    for (name, t) in &self.transformations {
      if !matches!(t, InterpolateTransformation::Contact(_)) {
        transformations.insert(name.clone(), InitializedTransformation::new(t, &self.defs)?);
      }
    }
    Ok(AssemblyArgs {
      ims: mim.ims()?,
      var_mfs,
      var_imds,
      transformations,
      ext: BTreeMap::new(),
    })
  }

  /// The contact transformations among names (the deformed face boxes: once
  /// per assembly that uses them, as GetFEM's init)
  fn init_contact_transformations(
    &self,
    asm_args: &mut AssemblyArgs,
    names: &[String],
  ) -> Result<()> {
    let mut built = vec![];
    for name in names {
      let Some(InterpolateTransformation::Contact(c)) = self.transformations.get(name) else {
        continue;
      };
      if asm_args.transformations.contains_key(name) {
        continue;
      }
      let data = self
        .variables
        .iter()
        .filter(|(_, st)| matches!(st.space, Space::Fixed))
        .map(|(k, st)| (k.clone(), st.value.clone()))
        .collect();
      for b in &c.boundaries {
        if let Some(mf_dofs) = asm_args.var_mfs.get(&b.disp)
          && mf_dofs.reduction().is_some()
          && !asm_args.ext.contains_key(&b.disp)
          && let Some(st) = self.variables.get(&b.disp)
        {
          let v = mf_dofs.extend_vector(&st.value)?;
          asm_args.ext.insert(b.disp.clone(), v);
        }
      }
      let (var_mfs, ext) = (&asm_args.var_mfs, &asm_args.ext);
      let disp =
        |v: &str, m: &getfem_geom::mesh::MeshData, cv: usize| self.disp(var_mfs, ext, v, m, cv);
      let rt = InitializedContact::new(name, c, &self.defs, data, &disp)?;
      built.push((
        name.clone(),
        InitializedTransformation::Contact(Box::new(rt)),
      ));
    }
    asm_args.transformations.extend(built);
    Ok(())
  }
}

/// Precomputations of the elements seen so far in an assembly (GetFEM
/// gis.gp_pool, gis.fp_pool; the global caches are keyed by names, slow to
/// look up per element).
#[derive(Default)]
struct PrecompPool {
  geo: Vec<(PGeoTrans, PIntegration, Arc<GeoPrecomp>)>,
  fem: Vec<(PFem, PIntegration, Arc<FemPrecomp>)>,
}

impl PrecompPool {
  fn geo(&mut self, gt: &PGeoTrans, pim: &PIntegration) -> Result<Arc<GeoPrecomp>> {
    if let Some(e) = self
      .geo
      .iter()
      .find(|e| Arc::ptr_eq(&e.0, gt) && Arc::ptr_eq(&e.1, pim))
    {
      return Ok(e.2.clone());
    }
    let gp = geo_precomp(gt, pim)?;
    if !pim.is_transient() {
      self.geo.push((gt.clone(), pim.clone(), gp.clone()));
    }
    Ok(gp)
  }

  fn fem(&mut self, pf: &PFem, pim: &PIntegration) -> Result<Arc<FemPrecomp>> {
    let same = |a: &PFem| std::ptr::addr_eq(Arc::as_ptr(a), Arc::as_ptr(pf));
    if let Some(e) = self
      .fem
      .iter()
      .find(|e| same(&e.0) && Arc::ptr_eq(&e.1, pim))
    {
      return Ok(e.2.clone());
    }
    let pc = fem_precomp(pf, pim)?;
    if !pim.is_transient() {
      self.fem.push((pf.clone(), pim.clone(), pc.clone()));
    }
    Ok(pc)
  }
}

/// A target point of a transformation.
struct TPoint {
  mesh: Arc<getfem_geom::mesh::MeshData>,
  cv: usize,
  xref: Vec<f64>,
  geo: PointGeom,
  /// unit outward normal of the target face (neighbor element)
  normal: Option<Vec<f64>>,
}

struct PtField {
  nd: usize,
  /// global indices in the variable's vector (None: not in a partial space)
  dofs: Vec<Option<usize>>,
  coef: Vec<f64>,
  /// val[i + nd*a], grad[i + nd*(a + t*r)], hess[i + nd*(a + t*(r + N*s))]
  val: Vec<f64>,
  grad: Vec<f64>,
  hess: Vec<f64>,
}

/// Where an assembly goes: entries (scalars, vectors), element matrices
/// m[t1 + na*t2] on the test DOFs ra (rows), rb (columns).
trait AssemblyTarget {
  fn entry(&mut self, i: usize, j: usize, v: f64);
  fn elem(&mut self, ra: &[Option<usize>], rb: &[Option<usize>], m: &[f64]);
  /// elem with its sup norm known
  fn elem_norm(&mut self, ra: &[Option<usize>], rb: &[Option<usize>], m: &[f64], _ninf: f64) {
    self.elem(ra, rb, m)
  }
  /// values at point ip of element cv (face) (interpolation)
  fn point(&mut self, _cv: usize, _face: Option<usize>, _ip: usize, _v: &[f64]) -> Result<()> {
    Ok(())
  }
  /// column j += entries (rows sorted, distinct)
  fn add_column(&mut self, j: usize, col: &[(usize, f64)]) {
    for &(i, v) in col {
      self.entry(i, j, v);
    }
  }
  /// the shape of the targets of the threads; None: not assembled by
  /// threads
  fn thread_target_shape(&self) -> Option<ThreadTargetShape> {
    None
  }
  /// the targets of the threads added, in thread order (their storage may
  /// be taken)
  fn add_thread_targets(&mut self, thread_targets: &mut [ThreadTarget]) -> Result<()> {
    thread_targets.iter().try_for_each(|t| t.add_to(self))
  }
}

/// The target of a thread: a vector (and scalar), a matrix, or the values
/// at the points (cv, face, ip, end of its values), replayed in order
#[derive(Debug)]
enum ThreadTarget {
  Vector(Vec<f64>),
  Matrix(ColMatrix<f64>, usize, Vec<(usize, usize)>),
  Points(Vec<(usize, Option<usize>, usize, usize)>, Vec<f64>),
}

/// The shape of a ThreadTarget: Vector(n entries), Matrix(nrows, ncols,
/// dim of the MatrixTarget), Points
#[derive(Clone, Copy)]
enum ThreadTargetShape {
  Vector(usize),
  Matrix(usize, usize, usize),
  Points,
}

/// The outputs of the threads: their targets, their entries on temporary
/// intervals, their outputs of condensation
#[derive(Debug, Default)]
struct ThreadOuts {
  targets: Vec<ThreadTarget>,
  unreduced: Vec<UnreducedOut>,
  conds: Vec<CondOut>,
}

impl ThreadOuts {
  fn resize(&mut self, n: usize) {
    self.targets.resize_with(n, || ThreadTarget::Vector(vec![]));
    self.unreduced.resize_with(n, UnreducedOut::default);
    self.conds.resize_with(n, CondOut::default);
  }
}

/// cols += tcols, a column of tcols moved into an empty one of cols, not
/// copied
fn add_cols(cols: &mut [RsVector<f64>], tcols: &mut [RsVector<f64>]) {
  for (col, tcol) in cols.iter_mut().zip(tcols) {
    if col.is_empty() {
      std::mem::swap(col, tcol);
    } else if !tcol.is_empty() {
      add_to_rsvector(col, tcol);
    }
  }
}

impl ThreadTarget {
  /// empty, of the shape s (its storage reused)
  fn reset(&mut self, s: ThreadTargetShape) {
    match (s, self) {
      (ThreadTargetShape::Vector(n), ThreadTarget::Vector(v)) => {
        v.clear();
        v.resize(n, 0.0);
      }
      (ThreadTargetShape::Matrix(nrows, ncols, dim), ThreadTarget::Matrix(k, n, _)) => {
        k.resize(nrows, ncols);
        k.clear();
        *n = dim;
      }
      (ThreadTargetShape::Points, ThreadTarget::Points(pts, values)) => {
        pts.clear();
        values.clear();
      }
      (ThreadTargetShape::Vector(n), t) => *t = ThreadTarget::Vector(vec![0.0; n]),
      (ThreadTargetShape::Points, t) => *t = ThreadTarget::Points(vec![], vec![]),
      (ThreadTargetShape::Matrix(nrows, ncols, dim), t) => {
        *t = ThreadTarget::Matrix(ColMatrix::new(nrows, ncols), dim, vec![])
      }
    }
  }

  fn add_to<T: AssemblyTarget + ?Sized>(&self, target: &mut T) -> Result<()> {
    match self {
      ThreadTarget::Vector(v) => {
        for (i, &x) in v.iter().enumerate() {
          if x != 0.0 {
            target.entry(i, 0, x);
          }
        }
      }
      ThreadTarget::Matrix(k, ..) => {
        for (j, col) in k.cols.iter().enumerate() {
          if !col.is_empty() {
            target.add_column(j, col);
          }
        }
      }
      ThreadTarget::Points(pts, values) => {
        let mut start = 0;
        for &(cv, face, ip, end) in pts {
          target.point(cv, face, ip, &values[start..end])?;
          start = end;
        }
      }
    }
    Ok(())
  }
}

impl AssemblyTarget for ThreadTarget {
  fn entry(&mut self, i: usize, j: usize, v: f64) {
    match self {
      ThreadTarget::Vector(w) => w[i] += v,
      ThreadTarget::Matrix(k, ..) => k.add_entry(i, j, v),
      ThreadTarget::Points(..) => {}
    }
  }

  fn elem(&mut self, ra: &[Option<usize>], rb: &[Option<usize>], m: &[f64]) {
    let ninf = m.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    self.elem_norm(ra, rb, m, ninf);
  }

  fn elem_norm(&mut self, ra: &[Option<usize>], rb: &[Option<usize>], m: &[f64], ninf: f64) {
    match self {
      ThreadTarget::Vector(w) => EntriesTarget(|i, _, v| w[i] += v, 0).elem(ra, rb, m),
      ThreadTarget::Matrix(k, n, sort) => {
        if ninf != 0.0 {
          k.add_elem_matrix(ra, rb, m, ninf * 1e-14, *n, sort);
        }
      }
      ThreadTarget::Points(..) => {}
    }
  }

  fn point(&mut self, cv: usize, face: Option<usize>, ip: usize, v: &[f64]) -> Result<()> {
    if let ThreadTarget::Points(pts, values) = self {
      values.extend_from_slice(v);
      pts.push((cv, face, ip, values.len()));
    }
    Ok(())
  }

  fn add_column(&mut self, j: usize, col: &[(usize, f64)]) {
    match self {
      ThreadTarget::Vector(_) => col.iter().for_each(|&(i, v)| self.entry(i, j, v)),
      ThreadTarget::Matrix(k, ..) => k.add_to_column(j, col),
      ThreadTarget::Points(..) => {}
    }
  }
}

/// assemble: the values of the expression at the points
const VALUES: usize = usize::MAX;

/// Outputs of a condensed matrix assembly (GetFEM: the internal coupling
/// matrix and the overwritten vector), in the extended numbering of the
/// spaces with reduction matrices until folded.
#[derive(Debug, Default)]
struct CondOut {
  /// the residual of all variables (internal ones included), shared by the
  /// threads
  residual: Arc<Vec<f64>>,
  /// K'_QJ: rows internal DOFs, columns the primary ones
  kqj: ColMatrix<f64>,
  /// -K_IQ R'_Q on the primary rows
  rhs: Vec<f64>,
  /// R'_Q on the internal rows
  rq: Vec<f64>,
}

impl CondOut {
  /// the residual, n, n_ext
  fn shape(&self) -> (Arc<Vec<f64>>, usize, usize) {
    (self.residual.clone(), self.rq.len(), self.kqj.ncols)
  }

  /// empty, of the shape s (its storage reused)
  fn reset(&mut self, (residual, n, n_ext): &(Arc<Vec<f64>>, usize, usize)) -> &mut Self {
    self.residual = residual.clone();
    self.kqj.resize(*n, *n_ext);
    self.kqj.clear();
    self.rhs.clear();
    self.rq.clear();
    self.rhs.resize(*n_ext, 0.0);
    self.rq.resize(*n, 0.0);
    self
  }

  /// the outputs of the threads added, in thread order
  fn add(&mut self, thread_conds: &mut [CondOut]) {
    for c in thread_conds {
      add_cols(&mut self.kqj.cols, &mut c.kqj.cols);
      self.rhs.iter_mut().zip(&c.rhs).for_each(|(a, b)| *a += b);
      self.rq.iter_mut().zip(&c.rq).for_each(|(a, b)| *a += b);
    }
  }

  /// sizes for an assembly with temporary intervals up to n_ext
  fn size(&mut self, n: usize, n_ext: usize) {
    self.kqj.ncols = n_ext;
    self.kqj.cols.resize(n_ext, vec![]);
    self.rhs.resize(n_ext, 0.0);
    self.rq.resize(n, 0.0);
  }

  /// the temporary intervals of reduced spaces folded with E (columns of
  /// K'_QJ, rows of the right-hand side), exactly once
  fn fold(&mut self, n: usize, ext: &TmpVarIntervals) {
    for &(t, off, r) in ext {
      for b in 0..r.e.nrows {
        let col = std::mem::take(&mut self.kqj.cols[t + b]);
        let x = self.rhs[t + b];
        self.rhs[t + b] = 0.0;
        for (c, e) in r.e.row(b) {
          for &(i, v) in &col {
            self.kqj.add_entry(i, off + c, v * e);
          }
          self.rhs[off + c] += x * e;
        }
      }
    }
    self.kqj.ncols = n;
    self.kqj.cols.truncate(n);
    self.rhs.truncate(n);
  }
}

/// Interpolation: values to a closure
struct Points<'a, 'b>(&'a mut PointTarget<'b>);

impl AssemblyTarget for Points<'_, '_> {
  fn entry(&mut self, _: usize, _: usize, _: f64) {}
  fn elem(&mut self, _: &[Option<usize>], _: &[Option<usize>], _: &[f64]) {}
  fn point(&mut self, cv: usize, face: Option<usize>, ip: usize, v: &[f64]) -> Result<()> {
    (self.0)(cv, face, ip, v)
  }
  fn thread_target_shape(&self) -> Option<ThreadTargetShape> {
    Some(ThreadTargetShape::Points)
  }
}

/// Scalars and vectors: entries to a closure, on the given number of rows
struct EntriesTarget<F>(F, usize);

impl<F: FnMut(usize, usize, f64)> AssemblyTarget for EntriesTarget<F> {
  fn entry(&mut self, i: usize, j: usize, v: f64) {
    (self.0)(i, j, v)
  }

  fn elem(&mut self, ra: &[Option<usize>], rb: &[Option<usize>], m: &[f64]) {
    let na = ra.len();
    for (t2, j) in rb.iter().enumerate() {
      let Some(j) = *j else { continue };
      for (t1, i) in ra.iter().enumerate() {
        if let Some(i) = *i {
          (self.0)(i, j, m[t1 + na * t2]);
        }
      }
    }
  }

  fn thread_target_shape(&self) -> Option<ThreadTargetShape> {
    Some(ThreadTargetShape::Vector(self.1))
  }
}

/// The entries on the temporary intervals (indices from n) of an assembly
/// with reduction matrices: in k (all indices, GetFEM Kuu, Kur, Kru), or
/// in vu for a vector (GetFEM Vu)
#[derive(Debug, Default)]
struct UnreducedOut {
  n: usize,
  k: ColMatrix<f64>,
  /// a vector: its temporary entries (from n)
  vu: Vec<f64>,
  vector: bool,
  dim: usize,
  sort: Vec<(usize, usize)>,
}

impl UnreducedOut {
  /// n, the size of k (or vu), vector, dim
  fn shape(&self) -> (usize, usize, bool, usize) {
    let size = if self.vector {
      self.vu.len()
    } else {
      self.k.ncols
    };
    (self.n, size, self.vector, self.dim)
  }

  /// empty, of the shape s (its storage reused)
  fn reset(&mut self, (n, size, vector, dim): (usize, usize, bool, usize)) -> &mut Self {
    (self.n, self.vector, self.dim) = (n, vector, dim);
    self.vu.clear();
    self.k.clear();
    match vector {
      true => self.vu.resize(size, 0.0),
      false => self.k.resize(size, size),
    }
    self
  }
}

/// Assembly of unreduced terms (GetFEM unreduced_terms): the entries on
/// temporary intervals to out, the others to inner
struct UnreducedTarget<'a, S: ?Sized> {
  inner: &'a mut S,
  out: &'a mut UnreducedOut,
}

impl<S: AssemblyTarget + ?Sized> AssemblyTarget for UnreducedTarget<'_, S> {
  fn entry(&mut self, i: usize, j: usize, v: f64) {
    let t = &mut *self.out;
    if i < t.n && j < t.n {
      self.inner.entry(i, j, v);
    } else if t.vector {
      t.vu[i - t.n] += v;
    } else {
      t.k.add_entry(i, j, v);
    }
  }
  fn elem(&mut self, ra: &[Option<usize>], rb: &[Option<usize>], m: &[f64]) {
    let ninf = m.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    self.elem_norm(ra, rb, m, ninf);
  }
  fn elem_norm(&mut self, ra: &[Option<usize>], rb: &[Option<usize>], m: &[f64], ninf: f64) {
    let t = &mut *self.out;
    if ra.iter().chain(rb).flatten().all(|&i| i < t.n) {
      return self.inner.elem_norm(ra, rb, m, ninf);
    }
    if ninf == 0.0 {
      return;
    }
    t.k
      .add_elem_matrix(ra, rb, m, ninf * 1e-14, t.dim, &mut t.sort);
  }
  fn point(&mut self, cv: usize, face: Option<usize>, ip: usize, v: &[f64]) -> Result<()> {
    self.inner.point(cv, face, ip, v)
  }
}

/// Matrices as GetFEM: nothing for a zero element matrix, else the entries
/// above 1e-14 times its largest, into the columns (add_elem_matrix)
struct MatrixTarget<'a> {
  k: &'a mut ColMatrix<f64>,
  n: usize,
  sort: Vec<(usize, usize)>,
}

impl AssemblyTarget for MatrixTarget<'_> {
  fn entry(&mut self, i: usize, j: usize, v: f64) {
    self.k.add_entry(i, j, v);
  }
  fn add_column(&mut self, j: usize, col: &[(usize, f64)]) {
    self.k.add_to_column(j, col);
  }

  fn thread_target_shape(&self) -> Option<ThreadTargetShape> {
    Some(ThreadTargetShape::Matrix(
      self.k.nrows,
      self.k.ncols,
      self.n,
    ))
  }

  /// by column blocks, one merge thread each: the columns of the thread
  /// targets added in thread order; a column into an empty one moved, not
  /// copied (most columns are of one thread)
  fn add_thread_targets(&mut self, thread_targets: &mut [ThreadTarget]) -> Result<()> {
    let block = self.k.ncols.div_ceil(thread_targets.len() + 1).max(1);
    // thread_cols[b][t]: the columns of block b of thread target t
    let mut thread_cols: Vec<Vec<&mut [RsVector<f64>]>> =
      (0..self.k.ncols.div_ceil(block)).map(|_| vec![]).collect();
    for t in thread_targets.iter_mut() {
      if let ThreadTarget::Matrix(k, ..) = t {
        for (b, cols) in thread_cols.iter_mut().zip(k.cols.chunks_mut(block)) {
          b.push(cols);
        }
      }
    }
    std::thread::scope(|scope| {
      for (cols, thread_cols) in self.k.cols.chunks_mut(block).zip(thread_cols) {
        scope.spawn(move || {
          for tcols in thread_cols {
            add_cols(cols, tcols);
          }
        });
      }
    });
    Ok(())
  }

  fn elem(&mut self, ra: &[Option<usize>], rb: &[Option<usize>], m: &[f64]) {
    let ninf = m.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    self.elem_norm(ra, rb, m, ninf);
  }

  fn elem_norm(&mut self, ra: &[Option<usize>], rb: &[Option<usize>], m: &[f64], ninf: f64) {
    if ninf == 0.0 {
      return;
    }
    self
      .k
      .add_elem_matrix(ra, rb, m, ninf * 1e-14, self.n, &mut self.sort);
  }
}

/// The value of Workspace::set_variable: values, or a GWFL expression
/// interpolated on the variable's space.
pub trait ValueArg<'a> {
  fn values(self, ws: &Workspace, name: &str) -> Result<Cow<'a, [f64]>>;
}

impl<'a> ValueArg<'a> for &'a [f64] {
  fn values(self, _: &Workspace, _: &str) -> Result<Cow<'a, [f64]>> {
    Ok(Cow::Borrowed(self))
  }
}

impl<'a> ValueArg<'a> for &'a Vec<f64> {
  fn values(self, _: &Workspace, _: &str) -> Result<Cow<'a, [f64]>> {
    Ok(Cow::Borrowed(self))
  }
}

impl<'a, const N: usize> ValueArg<'a> for &'a [f64; N] {
  fn values(self, _: &Workspace, _: &str) -> Result<Cow<'a, [f64]>> {
    Ok(Cow::Borrowed(self))
  }
}

impl<'a> ValueArg<'a> for &str {
  fn values(self, ws: &Workspace, name: &str) -> Result<Cow<'a, [f64]>> {
    Ok(Cow::Owned(ws.interpolation_on_variable(name, self)?))
  }
}

/// The target of Workspace::interpolation: a mesh_fem, an im_data or
/// points of a mesh.
pub trait InterpolationArg {
  fn interpolate(self, ws: &Workspace, expr: &str, region: RegionArg) -> Result<Vec<f64>>;
}

impl InterpolationArg for &MeshFem {
  fn interpolate(self, ws: &Workspace, expr: &str, region: RegionArg) -> Result<Vec<f64>> {
    ws.interpolation_lagrange_fem(expr, self, region)
  }
}

impl InterpolationArg for &ImData {
  fn interpolate(self, ws: &Workspace, expr: &str, region: RegionArg) -> Result<Vec<f64>> {
    ws.interpolation_im_data(expr, self, region)
  }
}

impl InterpolationArg for (&[Vec<f64>], &getfem_geom::mesh::Mesh) {
  fn interpolate(self, ws: &Workspace, expr: &str, region: RegionArg) -> Result<Vec<f64>> {
    ws.interpolation_mti(expr, self.1, self.0, region)
  }
}

impl InterpolationArg for (&Vec<Vec<f64>>, &getfem_geom::mesh::Mesh) {
  fn interpolate(self, ws: &Workspace, expr: &str, region: RegionArg) -> Result<Vec<f64>> {
    ws.interpolation_mti(expr, self.1, self.0, region)
  }
}

#[cfg(test)]
mod tests;

/// The dimension of the first element of the region (a 1D mesh in 2D: 1),
/// the mesh dimension if none.
fn ref_elt_dim_of_mesh(m: &getfem_geom::mesh::Mesh, region: &Region) -> Result<usize> {
  let md = m.snapshot();
  let first = if region.id() == ALL {
    md.convex_index().next()
  } else {
    region.resolve(&md)?.iter().next().map(|(cv, _)| cv)
  };
  Ok(first.map_or(m.dim(), |cv| m.trans_of_convex(cv).dim()))
}
