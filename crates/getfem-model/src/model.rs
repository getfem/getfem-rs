//! Model: a workspace with a list of terms and Dirichlet conditions,
//! assembled into one residual and tangent over all variables.

use std::ops::Range;

use getfem_fem::im_data::ImData;
use getfem_fem::mesh_fem::MeshFem;
use getfem_fem::mesh_im::MeshIm;
use getfem_geom::region::{Region, RegionArg};
use getfem_gwfl::workspace::{InterpolationArg, ValueArg, Workspace};
use getfem_linalg::{ColMatrix, Coo, Csr, RsVector};

use crate::newton::{CondensedStep, Linearized, NonlinearSystem};
use crate::{Error, Result};

/// A term of a model, in the order of addition (as listed by summary()).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TermId(pub usize);

fn region_name(r: &Region) -> String {
  match r.id() {
    -1 => "everywhere".into(),
    id => format!("on region {id}"),
  }
}

/// What a cached term matrix depends on: its data (all for None), the
/// spaces, its integration method.
fn stamps(ws: &Workspace, mim: &MeshIm, data: Option<&[String]>) -> Result<(u64, u64, u64)> {
  // the mesh version follows the edits of the regions of the terms; the
  // number of MPI ranks decides which elements a rank's matrix holds
  Ok((
    ws.data_stamp(data),
    ws.spaces_version()?,
    mim.version()? ^ mim.linked_mesh().version().rotate_left(32) ^ getfem_mpi::size() as u64,
  ))
}

/// How a term is treated: linearity detected, asserted, or not cached.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TermKind {
  Auto,
  Linear,
  Nonlinear,
}

/// The contribution of a linear term, kept between assemblies: its matrix
/// and residual at u = 0 (GetFEM: the brick's kept matrices and vectors)
struct LinContrib {
  /// versions of the data, the spaces and the integration methods used
  stamps: (u64, u64, u64),
  nb_dof: usize,
  k: ColMatrix<f64>,
  /// residual at u = 0
  r0: Vec<f64>,
}

struct Term {
  expr: String,
  mim: MeshIm,
  region: Region,
  /// 0 potential, 1 residual, 2 bilinear form
  order: usize,
  linear: bool,
  /// the data it reads (None: possibly all), GetFEM's data list
  data: Option<Vec<String>>,
  lin_contrib: Option<LinContrib>,
}

/// DOFs of a variable fixed to the values of a data (elimination).
struct Elimination {
  var: String,
  region: Region,
  data: Option<String>,
}

#[derive(Default)]
pub struct Model {
  ws: Workspace,
  terms: Vec<Term>,
  elim: Vec<Elimination>,
  explicit: Vec<Explicit>,
  /// constraints of the multipliers added by the helpers, per variable
  mults: std::collections::BTreeMap<String, Vec<Multiplier>>,
  /// the tangent matrix and rhs (-R) of the last assembly; the matrix's
  /// storage reused by the next one
  tangent_matrix: ColMatrix<f64>,
  rhs: Vec<f64>,
  /// with condensation (GetFEM full_rrhs, internal_rTM, internal_sol): the
  /// rhs of all unknowns, K'_QJ, the internal solution
  condensed: Option<(Vec<f64>, Csr<f64>, Vec<f64>)>,
}

/// What an assembly builds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuildVersion {
  /// the rhs only (BUILD_RHS); the tangent matrix stays as it was
  Rhs,
  /// the tangent matrix and the rhs (BUILD_ALL)
  All,
  /// the tangent matrix and the rhs with the internal variables condensed
  /// out (BUILD_ALL_CONDENSED): both on the primary unknowns, the internal
  /// solution in internal_solution
  AllCondensed,
}

/// What a multiplier prescribes on its variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Constraint {
  /// the value (Dirichlet condition)
  Value,
  /// the normal derivative of a scalar variable
  NormalDerivative,
}

/// The multiplier space of a Dirichlet condition: a mesh_fem, or a degree
/// (classical space of the qdim of var on its mesh).
pub trait MultiplierArg {
  fn mesh_fem(self, md: &Model, var: &str) -> Result<MeshFem>;
}

impl MultiplierArg for &MeshFem {
  fn mesh_fem(self, _: &Model, _: &str) -> Result<MeshFem> {
    Ok(self.clone())
  }
}

impl MultiplierArg for usize {
  fn mesh_fem(self, md: &Model, var: &str) -> Result<MeshFem> {
    let mf_u = md.required_mesh_fem(var)?;
    Ok(MeshFem::new(&mf_u.linked_mesh(), mf_u.qdim()).with_classical_finite_element(self)?)
  }
}

/// an integer literal (`1`) is an i32
impl MultiplierArg for i32 {
  fn mesh_fem(self, md: &Model, var: &str) -> Result<MeshFem> {
    let k = usize::try_from(self)
      .map_err(|_| Error::InvalidArgument(format!("multiplier degree {self}")))?;
    k.mesh_fem(md, var)
  }
}

/// The constraints of a multiplier (one column per DOF: rows of the
/// variable).
#[derive(Clone, Debug)]
struct Multiplier {
  name: String,
  cols: Vec<RsVector<f64>>,
}

/// Native contribution (set_explicit_matrix, set_explicit_rhs): tangent
/// block M (rows of var1, columns of var2) with residual M u_var2, or
/// a right-hand side L with residual -L.
enum Explicit {
  Matrix {
    var1: String,
    var2: String,
    m: Csr<f64>,
  },
  Rhs {
    var: String,
    l: Vec<f64>,
  },
}

impl Model {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn workspace(&self) -> &Workspace {
    &self.ws
  }

  pub fn workspace_mut(&mut self) -> &mut Workspace {
    &mut self.ws
  }

  /// A variable on the DOFs of mf on a region (see
  /// Workspace::add_filtered_fem_variable).
  pub fn add_filtered_fem_variable(
    &mut self,
    name: &str,
    mf: &MeshFem,
    region: impl Into<RegionArg>,
  ) -> Result<()> {
    Ok(self.ws.add_filtered_fem_variable(name, mf, region)?)
  }

  /// An interpolate transformation X -> expr from source to target
  /// (Workspace::add_interpolate_transformation_from_expression).
  pub fn add_interpolate_transformation_from_expression(
    &mut self,
    name: &str,
    source: &getfem_geom::mesh::Mesh,
    target: &getfem_geom::mesh::Mesh,
    expr: &str,
  ) -> Result<()> {
    Ok(
      self
        .ws
        .add_interpolate_transformation_from_expression(name, source, target, expr)?,
    )
  }

  /// Workspace::add_raytracing_transformation.
  pub fn add_raytracing_transformation(&mut self, name: &str, release_distance: f64) -> Result<()> {
    Ok(
      self
        .ws
        .add_raytracing_transformation(name, release_distance)?,
    )
  }

  /// Workspace::add_master_contact_boundary_to_raytracing_transformation.
  pub fn add_master_contact_boundary_to_raytracing_transformation(
    &mut self,
    name: &str,
    mesh: &getfem_geom::mesh::Mesh,
    disp: &str,
    region: impl Into<RegionArg>,
  ) -> Result<()> {
    Ok(
      self
        .ws
        .add_master_contact_boundary_to_raytracing_transformation(name, mesh, disp, region)?,
    )
  }

  /// Workspace::add_slave_contact_boundary_to_raytracing_transformation.
  pub fn add_slave_contact_boundary_to_raytracing_transformation(
    &mut self,
    name: &str,
    mesh: &getfem_geom::mesh::Mesh,
    disp: &str,
    region: impl Into<RegionArg>,
  ) -> Result<()> {
    Ok(
      self
        .ws
        .add_slave_contact_boundary_to_raytracing_transformation(name, mesh, disp, region)?,
    )
  }

  /// Workspace::add_rigid_obstacle_to_raytracing_transformation.
  pub fn add_rigid_obstacle_to_raytracing_transformation(
    &mut self,
    name: &str,
    expr: &str,
    n: usize,
  ) -> Result<()> {
    Ok(
      self
        .ws
        .add_rigid_obstacle_to_raytracing_transformation(name, expr, n)?,
    )
  }

  /// Workspace::add_projection_transformation.
  pub fn add_projection_transformation(&mut self, name: &str, release_distance: f64) -> Result<()> {
    Ok(
      self
        .ws
        .add_projection_transformation(name, release_distance)?,
    )
  }

  /// Workspace::add_master_contact_boundary_to_projection_transformation.
  pub fn add_master_contact_boundary_to_projection_transformation(
    &mut self,
    name: &str,
    mesh: &getfem_geom::mesh::Mesh,
    disp: &str,
    region: impl Into<RegionArg>,
  ) -> Result<()> {
    Ok(
      self
        .ws
        .add_master_contact_boundary_to_projection_transformation(name, mesh, disp, region)?,
    )
  }

  /// Workspace::add_slave_contact_boundary_to_projection_transformation.
  pub fn add_slave_contact_boundary_to_projection_transformation(
    &mut self,
    name: &str,
    mesh: &getfem_geom::mesh::Mesh,
    disp: &str,
    region: impl Into<RegionArg>,
  ) -> Result<()> {
    Ok(
      self
        .ws
        .add_slave_contact_boundary_to_projection_transformation(name, mesh, disp, region)?,
    )
  }

  /// Workspace::add_rigid_obstacle_to_projection_transformation.
  pub fn add_rigid_obstacle_to_projection_transformation(
    &mut self,
    name: &str,
    expr: &str,
    n: usize,
  ) -> Result<()> {
    Ok(
      self
        .ws
        .add_rigid_obstacle_to_projection_transformation(name, expr, n)?,
    )
  }

  /// Workspace::add_elementary_transformation.
  pub fn add_elementary_transformation(
    &mut self,
    name: &str,
    t: impl getfem_gwfl::elementary::ElementaryTransformation + 'static,
  ) -> Result<()> {
    Ok(self.ws.add_elementary_transformation(name, t)?)
  }

  /// Workspace::add_2D_rotated_RT0_projection (MITC plates).
  #[allow(non_snake_case)]
  pub fn add_2D_rotated_RT0_projection(&mut self, name: &str) -> Result<()> {
    Ok(self.ws.add_2D_rotated_RT0_projection(name)?)
  }

  /// Workspace::add_projected_gradient (B-bar, F-bar).
  pub fn add_projected_gradient(&mut self, name: &str) -> Result<()> {
    Ok(self.ws.add_projected_gradient(name)?)
  }

  /// Workspace::add_P0_projection.
  #[allow(non_snake_case)]
  pub fn add_P0_projection(&mut self, name: &str) -> Result<()> {
    Ok(self.ws.add_P0_projection(name)?)
  }

  /// Values of expr on a mesh_fem, an im_data or points of a mesh
  /// (Workspace::interpolation).
  pub fn interpolation(
    &self,
    expr: &str,
    target: impl InterpolationArg,
    region: impl Into<RegionArg>,
  ) -> Result<Vec<f64>> {
    Ok(self.ws.interpolation(expr, target, region)?)
  }

  // ---- the workspace's variables, data and assembly (its methods, as the
  // model's own: md.add_fem_variable(...) and so on) ----

  pub fn nb_dof(&self) -> usize {
    self.ws.nb_dof()
  }

  /// The value of a variable or of data (Workspace::variable)
  pub fn variable(&self, name: &str) -> Option<&[f64]> {
    self.ws.variable(name)
  }

  /// Workspace::set_variable (variables and data): values, or a GWFL
  /// expression interpolated on the variable's space
  pub fn set_variable<'a>(&mut self, name: &str, v: impl ValueArg<'a>) -> Result<()> {
    Ok(self.ws.set_variable(name, v)?)
  }

  /// Workspace::set_variables: (name, expr) pairs, all evaluated with
  /// the old values
  pub fn set_variables(&mut self, pairs: &[(&str, &str)]) -> Result<()> {
    Ok(self.ws.set_variables(pairs)?)
  }

  pub fn check_spaces(&self) -> Result<()> {
    Ok(self.ws.check_spaces()?)
  }

  /// The range of an enabled variable in the global system.
  pub fn interval_of_variable(&self, name: &str) -> Option<Range<usize>> {
    self.ws.interval_of_variable(name)
  }

  pub fn mesh_fem_of_variable(&self, name: &str) -> Option<MeshFem> {
    self.ws.mesh_fem_of_variable(name)
  }

  pub fn add_fem_variable(&mut self, name: &str, mf: &MeshFem) -> Result<()> {
    Ok(self.ws.add_fem_variable(name, mf)?)
  }

  pub fn add_fixed_size_variable(&mut self, name: &str, shape: &[usize]) -> Result<()> {
    Ok(self.ws.add_fixed_size_variable(name, shape)?)
  }

  pub fn add_im_variable(&mut self, name: &str, imd: &ImData) -> Result<()> {
    Ok(self.ws.add_im_variable(name, imd)?)
  }

  pub fn add_internal_im_variable(&mut self, name: &str, imd: &ImData) -> Result<()> {
    Ok(self.ws.add_internal_im_variable(name, imd)?)
  }

  pub fn add_fem_data(&mut self, name: &str, mf: &MeshFem, value: &[f64]) -> Result<()> {
    Ok(self.ws.add_fem_data(name, mf, value)?)
  }

  pub fn add_fixed_size_data(&mut self, name: &str, shape: &[usize], value: &[f64]) -> Result<()> {
    Ok(self.ws.add_fixed_size_data(name, shape, value)?)
  }

  pub fn add_scalar_data(&mut self, name: &str, v: f64) -> Result<()> {
    Ok(self.ws.add_scalar_data(name, v)?)
  }

  pub fn add_im_data(&mut self, name: &str, imd: &ImData, value: &[f64]) -> Result<()> {
    Ok(self.ws.add_im_data(name, imd, value)?)
  }

  pub fn add_macro(&mut self, name: &str, expr: &str) -> Result<()> {
    Ok(self.ws.add_macro(name, expr)?)
  }

  #[allow(non_snake_case)]
  pub fn add_HHO_reconstructed_gradient(&mut self, name: &str) -> Result<()> {
    Ok(self.ws.add_HHO_reconstructed_gradient(name)?)
  }

  #[allow(non_snake_case)]
  pub fn add_HHO_reconstructed_symmetrized_gradient(&mut self, name: &str) -> Result<()> {
    Ok(self.ws.add_HHO_reconstructed_symmetrized_gradient(name)?)
  }

  #[allow(non_snake_case)]
  pub fn add_HHO_reconstructed_value(&mut self, name: &str) -> Result<()> {
    Ok(self.ws.add_HHO_reconstructed_value(name)?)
  }

  #[allow(non_snake_case)]
  pub fn add_HHO_reconstructed_symmetrized_value(&mut self, name: &str) -> Result<()> {
    Ok(self.ws.add_HHO_reconstructed_symmetrized_value(name)?)
  }

  #[allow(non_snake_case)]
  pub fn add_HHO_stabilization(&mut self, name: &str) -> Result<()> {
    Ok(self.ws.add_HHO_stabilization(name)?)
  }

  #[allow(non_snake_case)]
  pub fn add_HHO_symmetrized_stabilization(&mut self, name: &str) -> Result<()> {
    Ok(self.ws.add_HHO_symmetrized_stabilization(name)?)
  }

  /// Element-wise L2 projection of expr (Workspace::local_projection).
  pub fn local_projection(
    &self,
    expr: &str,
    mim: &MeshIm,
    mf: &MeshFem,
    region: impl Into<RegionArg>,
  ) -> Result<Vec<f64>> {
    Ok(self.ws.local_projection(expr, mim, mf, region)?)
  }

  fn push_term(
    &mut self,
    mim: &MeshIm,
    region: RegionArg,
    expr: &str,
    kind: TermKind,
  ) -> Result<TermId> {
    let region = region.on(&mim.linked_mesh())?;
    region.lasting(&format!("term {expr}"))?;
    let (order, lin) = self.ws.term_info(expr, mim, &region)?;
    if kind == TermKind::Linear && !lin {
      return Err(Error::InvalidArgument(format!("term {expr} is not linear")));
    }
    let linear = lin && kind != TermKind::Nonlinear;
    let data = self.ws.term_data(expr, mim)?;
    self.terms.push(Term {
      expr: expr.to_string(),
      mim: mim.clone(),
      region,
      order,
      linear,
      data,
      lin_contrib: None,
    });
    Ok(TermId(self.terms.len() - 1))
  }

  /// A GWFL term (potential, residual or bilinear form); linearity is
  /// detected and linear terms are cached while the data do not change.
  pub fn add_term(
    &mut self,
    mim: &MeshIm,
    region: impl Into<RegionArg>,
    expr: &str,
  ) -> Result<TermId> {
    self.push_term(mim, region.into(), expr, TermKind::Auto)
  }

  /// As add_term, with an error if the term is not linear.
  pub fn add_linear_term(
    &mut self,
    mim: &MeshIm,
    region: impl Into<RegionArg>,
    expr: &str,
  ) -> Result<TermId> {
    self.push_term(mim, region.into(), expr, TermKind::Linear)
  }

  /// As add_term, never cached.
  pub fn add_nonlinear_term(
    &mut self,
    mim: &MeshIm,
    region: impl Into<RegionArg>,
    expr: &str,
  ) -> Result<TermId> {
    self.push_term(mim, region.into(), expr, TermKind::Nonlinear)
  }

  /// The explicit tangent block M of the equations of var1 w.r.t. var2 (the
  /// residual gets M u_var2): set, replaced, or with None removed (not
  /// assembled at all). At most one per (var1, var2).
  pub fn set_explicit_matrix(&mut self, var1: &str, var2: &str, m: Option<Csr<f64>>) -> Result<()> {
    match &m {
      Some(m) => self.check_explicit_matrix(var1, var2, m)?,
      None => {
        self.check_variable(var1)?;
        self.check_variable(var2)?;
      }
    }
    let i = self.explicit_matrix(var1, var2);
    match (i, m) {
      (Some(i), Some(m)) => {
        self.explicit[i] = Explicit::Matrix {
          var1: var1.to_string(),
          var2: var2.to_string(),
          m,
        }
      }
      (None, Some(m)) => self.explicit.push(Explicit::Matrix {
        var1: var1.to_string(),
        var2: var2.to_string(),
        m,
      }),
      (Some(i), None) => {
        self.explicit.remove(i);
      }
      (None, None) => {}
    }
    Ok(())
  }

  /// The explicit right-hand side L of the equations of var (residual
  /// -L): set, replaced, or with None removed (not assembled at all). At
  /// most one per variable.
  pub fn set_explicit_rhs(&mut self, var: &str, l: Option<Vec<f64>>) -> Result<()> {
    match &l {
      Some(l) => self.check_explicit_rhs(var, l)?,
      None => {
        self.check_variable(var)?;
      }
    }
    let i = self.explicit_rhs(var);
    match (i, l) {
      (Some(i), Some(l)) => {
        self.explicit[i] = Explicit::Rhs {
          var: var.to_string(),
          l,
        }
      }
      (None, Some(l)) => self.explicit.push(Explicit::Rhs {
        var: var.to_string(),
        l,
      }),
      (Some(i), None) => {
        self.explicit.remove(i);
      }
      (None, None) => {}
    }
    Ok(())
  }

  fn explicit_matrix(&self, v: &str, c: &str) -> Option<usize> {
    self
      .explicit
      .iter()
      .position(|e| matches!(e, Explicit::Matrix { var1, var2, .. } if var1 == v && var2 == c))
  }

  fn explicit_rhs(&self, v: &str) -> Option<usize> {
    self
      .explicit
      .iter()
      .position(|e| matches!(e, Explicit::Rhs { var, .. } if var == v))
  }

  fn check_variable(&self, v: &str) -> Result<usize> {
    self
      .ws
      .variable(v)
      .map(|x| x.len())
      .ok_or_else(|| Error::InvalidArgument(format!("unknown variable {v}")))
  }

  fn check_explicit_matrix(&self, var1: &str, var2: &str, m: &Csr<f64>) -> Result<()> {
    let size = |v: &str| self.check_variable(v);
    if (m.nrows, m.ncols) != (size(var1)?, size(var2)?) {
      return Err(Error::InvalidArgument(format!(
        "explicit matrix of size {}x{} for {var1}, {var2}",
        m.nrows, m.ncols
      )));
    }
    Ok(())
  }

  fn check_explicit_rhs(&self, var: &str, l: &[f64]) -> Result<()> {
    if self.ws.variable(var).map(|x| x.len()) != Some(l.len()) {
      return Err(Error::InvalidArgument(format!(
        "explicit rhs of size {} for {var}",
        l.len()
      )));
    }
    Ok(())
  }

  /// Dirichlet condition var = g on a region with a multiplier on the DOFs
  /// on the region of mf_mult (a mesh_fem, or a degree: a classical space of
  /// the qdim of var on its mesh), or on all DOFs of a reduced (or partial)
  /// space; g None for zero. Returns the multiplier. Constraints that depend
  /// on each other (or on earlier multipliers of var) are an error, which
  /// tells how to select independent DOFs (see select_multiplier_dofs_for_variable).
  /// (GetFEM add_Dirichlet_condition_with_multipliers)
  #[allow(non_snake_case)]
  pub fn add_Dirichlet_condition_with_multiplier(
    &mut self,
    mim: &MeshIm,
    var: &str,
    mf_mult: impl MultiplierArg,
    region: impl Into<RegionArg>,
    g: Option<&str>,
  ) -> Result<String> {
    let mf_mult = mf_mult.mesh_fem(self, var)?;
    let region = region.into().on(&mim.linked_mesh())?;
    let region = &region;
    let name = self.fresh(&format!("mult_on_{var}"));
    self.add_checked_multiplier(&name, mim, var, &mf_mult, region, Constraint::Value)?;
    let c = g.map_or(var.to_string(), |g| format!("({var} - {g})"));
    self.add_term(
      mim,
      Some(region),
      &format!("{name}.Test_{var} + {c}.Test_{name}"),
    )?;
    Ok(name)
  }

  /// Dirichlet condition on the normal derivative of the scalar var by a
  /// multiplier on the DOFs on the region of mf_mult (or on all DOFs of a
  /// reduced space): Grad_var.Normal = g.Normal, g a vector field (GetFEM
  /// add_normal_derivative_Dirichlet_condition_with_multipliers), or zero.
  /// mf_mult as for add_Dirichlet_condition_with_multiplier.
  #[allow(non_snake_case)]
  pub fn add_normal_derivative_Dirichlet_condition_with_multiplier(
    &mut self,
    mim: &MeshIm,
    var: &str,
    mf_mult: impl MultiplierArg,
    region: impl Into<RegionArg>,
    g: Option<&str>,
  ) -> Result<String> {
    let mf_mult = mf_mult.mesh_fem(self, var)?;
    let region = region.into().on(&mim.linked_mesh())?;
    let region = &region;
    let name = self.fresh(&format!("mult_on_{var}"));
    self.add_checked_multiplier(
      &name,
      mim,
      var,
      &mf_mult,
      region,
      Constraint::NormalDerivative,
    )?;
    let dn = format!("(Grad_Test_{var}.Normal)");
    let c = g.map_or(format!("Grad_{var}"), |g| format!("(Grad_{var} - {g})"));
    self.add_term(
      mim,
      Some(region),
      &format!("{name}*{dn} + ({c}.Normal)*Test_{name}"),
    )?;
    Ok(name)
  }

  /// The DOFs of mf_mult (its basic DOFs on the region, or all its DOFs if
  /// it is reduced) whose constraints on var are linearly independent, of
  /// each other and of those of the multipliers of var added before by the
  /// helpers (pivoted Gram-Schmidt on the constraints): pass
  /// MeshFem::partial(&mf_mult, &dofs)? as the multiplier space.
  pub fn select_multiplier_dofs_for_variable(
    &self,
    var: &str,
    mim: &MeshIm,
    mf_mult: &MeshFem,
    region: impl Into<RegionArg>,
    c: Constraint,
  ) -> Result<std::collections::BTreeSet<usize>> {
    let region = region.into().on(&mim.linked_mesh())?;
    let region = &region;
    let (dofs, cols) = self.constraints(mim, var, mf_mult, region, c)?;
    Ok(
      self
        .independent(var, &cols)?
        .into_iter()
        .map(|k| dofs[k])
        .collect(),
    )
  }

  /// The DOFs of mf_mult of the multiplier (sorted: the basic DOFs on the
  /// region, all its DOFs if it is reduced) and their constraints on var.
  fn constraints(
    &self,
    mim: &MeshIm,
    var: &str,
    mf_mult: &MeshFem,
    region: &Region,
    c: Constraint,
  ) -> Result<(Vec<usize>, Vec<RsVector<f64>>)> {
    let (space, dofs) = if mf_mult.is_reduced() {
      (mf_mult.clone(), (0..mf_mult.nb_dof()?).collect())
    } else {
      (
        MeshFem::partial_on_region(mf_mult, region)?,
        mf_mult.basic_dof_on_region(region)?.into_iter().collect(),
      )
    };
    if self.ws.interval_of_variable(var).is_none() {
      return Err(Error::InvalidArgument(format!(
        "{var} is not an enabled variable"
      )));
    }
    // only the two variables of the coupling term
    let m = "m__constraint";
    let mut ws = Workspace::new();
    ws.add_fem_variable(var, &self.required_mesh_fem(var)?)?;
    ws.add_fem_variable(m, &space)?;
    let (rv, rm) = (
      ws.interval_of_variable(var).unwrap(),
      ws.interval_of_variable(m).unwrap(),
    );
    let coupling = match c {
      Constraint::Value => format!("Test_{var}.Test2_{m}"),
      Constraint::NormalDerivative => format!("(Grad_Test_{var}.Normal)*Test2_{m}"),
    };
    ws.add_expression(&coupling, mim, region)?;
    ws.assembly(2, false)?;
    let k = ws.assembled_matrix();
    // the columns of the multiplier, rows of var
    let cols: Vec<RsVector<f64>> = k.cols[rm]
      .iter()
      .map(|c| {
        c.iter()
          .filter(|e| rv.contains(&e.0))
          .map(|&(i, v)| (i - rv.start, v))
          .collect()
      })
      .collect();
    // MPI: the whole constraints (GetFEM MPI_SUM_SPARSE_MATRIX)
    Ok((dofs, summed_columns(cols)))
  }

  /// Indices of the independent columns, after those of the earlier
  /// multipliers of var.
  fn independent(&self, var: &str, cols: &[RsVector<f64>]) -> Result<Vec<usize>> {
    let nrows = self.ws.interval_of_variable(var).map_or(0, |r| r.len());
    let prev: Vec<&RsVector<f64>> = self
      .mults
      .get(var)
      .map_or(vec![], |l| l.iter().flat_map(|m| m.cols.iter()).collect());
    let np = prev.len();
    let mut b = Coo::new(nrows, np + cols.len());
    for (j, c) in prev.into_iter().chain(cols).enumerate() {
      for &(i, v) in c {
        b.push(i, j, v);
      }
    }
    Ok(
      getfem_linalg::range_basis_after(&b.to_csr()?, np, 1e-10)
        .into_iter()
        .filter(|&j| j >= np)
        .map(|j| j - np)
        .collect(),
    )
  }

  /// The multiplier variable on mf_mult (region DOFs, or a reduced space),
  /// if its constraints are independent.
  fn add_checked_multiplier(
    &mut self,
    name: &str,
    mim: &MeshIm,
    var: &str,
    mf_mult: &MeshFem,
    region: &Region,
    c: Constraint,
  ) -> Result<()> {
    let (_, cols) = self.constraints(mim, var, mf_mult, region, c)?;
    let ind = self.independent(var, &cols)?;
    if ind.len() < cols.len() {
      let earlier: Vec<&str> = self
        .mults
        .get(var)
        .map_or(vec![], |l| l.iter().map(|m| m.name.as_str()).collect());
      let on = if earlier.is_empty() {
        String::new()
      } else {
        format!(" or on those of {}", earlier.join(", "))
      };
      return Err(Error::InvalidArgument(format!(
        "{name}: {} of the {} constraints of its DOFs on {var} depend on the others{on}; \
         the system would be singular (typical of non-Lagrange multiplier spaces). \
         Select independent DOFs: let dofs = md.select_multiplier_dofs_for_variable(\"{var}\", &mim, &mf, \
         &region, Constraint::{c:?})?; and pass MeshFem::partial(&mf, &dofs)? as the multiplier space",
        cols.len() - ind.len(),
        cols.len()
      )));
    }
    let space = if mf_mult.is_reduced() {
      mf_mult.clone()
    } else {
      MeshFem::partial_on_region(mf_mult, region)?
    };
    self.ws.add_fem_variable(name, &space)?;
    self
      .mults
      .entry(var.to_string())
      .or_default()
      .push(Multiplier {
        name: name.to_string(),
        cols,
      });
    Ok(())
  }

  /// Dirichlet condition by penalization: coef (u - g).v on a region; g
  /// None for zero.
  #[allow(non_snake_case)]
  pub fn add_Dirichlet_condition_with_penalization(
    &mut self,
    mim: &MeshIm,
    var: &str,
    coef: f64,
    region: impl Into<RegionArg>,
    g: Option<&str>,
  ) -> Result<()> {
    let region = region.into().on(&mim.linked_mesh())?;
    let region = &region;
    let c = self.fresh(&format!("penalty_on_{var}"));
    self.ws.add_scalar_data(&c, coef)?;
    let d = g.map_or(var.to_string(), |g| format!("({var} - {g})"));
    self.add_term(mim, Some(region), &format!("{c}*{d}.Test_{var}"))?;
    Ok(())
  }

  /// Dirichlet condition by elimination of the DOFs of u on a region; g is
  /// finite element data on the same space, or None for zero.
  #[allow(non_snake_case)]
  pub fn add_Dirichlet_condition_with_simplification(
    &mut self,
    var: &str,
    region: impl Into<RegionArg>,
    g: Option<&str>,
  ) -> Result<()> {
    let region = region
      .into()
      .on(&self.required_mesh_fem(var)?.linked_mesh())?;
    let region = &region;
    region.lasting(&format!("elimination of {var}"))?;
    let mf = self.required_mesh_fem(var)?;
    if let Some(g) = g {
      // the same space (GetFEM: mf_data == &mf_u)
      let same = self.ws.mesh_fem_of_variable(g).is_some_and(|m| m.same(&mf));
      if !same {
        return Err(Error::InvalidArgument(format!(
          "{g} must be data on the space of {var}"
        )));
      }
    }
    self.elim.push(Elimination {
      var: var.to_string(),
      region: region.clone(),
      data: g.map(String::from),
    });
    Ok(())
  }

  /// the space of a finite element variable, an error otherwise
  fn required_mesh_fem(&self, var: &str) -> Result<MeshFem> {
    self
      .ws
      .mesh_fem_of_variable(var)
      .ok_or_else(|| Error::InvalidArgument(format!("{var} is not a finite element variable")))
  }

  fn fresh(&self, base: &str) -> String {
    let mut k = 0;
    loop {
      let n = if k == 0 {
        base.to_string()
      } else {
        format!("{base}_{k}")
      };
      if self.ws.defs().var(&n).is_none() {
        return n;
      }
      k += 1;
    }
  }

  /// Disable exactly this variable: it keeps its value, acts as data, and
  /// the terms testing it are dropped. Other variables, such as its
  /// multipliers, are not touched (unlike GetFEM).
  pub fn disable_variable(&mut self, var: &str) -> Result<()> {
    self.ws.disable_variable(var)?;
    self.clear_caches();
    Ok(())
  }

  pub fn enable_variable(&mut self, var: &str) -> Result<()> {
    self.ws.enable_variable(var)?;
    self.clear_caches();
    Ok(())
  }

  fn clear_caches(&mut self) {
    for t in &mut self.terms {
      t.lin_contrib = None;
    }
  }

  /// The values of the enabled variables, in the order of the system,
  /// without the internal ones.
  #[allow(clippy::wrong_self_convention)] // GetFEM's name
  pub fn from_variables(&self) -> Vec<f64> {
    self.values_of_variables(false)
  }

  /// Sets the enabled variables, without the internal ones, from u
  /// (nb_primary_dof values).
  #[allow(clippy::wrong_self_convention)] // GetFEM's name
  pub fn to_variables(&mut self, u: &[f64]) -> Result<()> {
    self.set_values_of_variables(u, false)
  }

  /// from_variables, with_internal: all nb_dof values
  pub(crate) fn values_of_variables(&self, with_internal: bool) -> Vec<f64> {
    let n = if with_internal {
      self.ws.nb_dof()
    } else {
      self.ws.nb_primary_dof()
    };
    let mut u = vec![0.0; n];
    for (v, r) in self.ws.variables() {
      if r.end <= n {
        u[r].copy_from_slice(self.ws.variable(&v).unwrap());
      }
    }
    u
  }

  pub(crate) fn set_values_of_variables(&mut self, u: &[f64], with_internal: bool) -> Result<()> {
    let n = if with_internal {
      self.ws.nb_dof()
    } else {
      self.ws.nb_primary_dof()
    };
    if u.len() != n {
      return Err(Error::InvalidArgument(format!(
        "to_variables: {} values for {n} unknowns",
        u.len()
      )));
    }
    for (v, r) in self.ws.variables() {
      if r.end <= n {
        self.ws.set_variable(&v, &u[r])?;
      }
    }
    Ok(())
  }

  /// The rhs -R and (version All) the tangent matrix of all terms and
  /// conditions at the current values, kept in the model (rhs,
  /// tangent_matrix). With Dirichlet elimination the rhs needs the tangent:
  /// both are built.
  pub fn assembly(&mut self, version: BuildVersion) -> Result<()> {
    self.condensed = None;
    if version == BuildVersion::AllCondensed {
      return self.assembly_condensed();
    }
    let u = self.values_of_variables(true);
    let r = if version == BuildVersion::Rhs && self.elim.is_empty() {
      self.residual_of_terms(&u)?
    } else {
      self.assemble_all(&u)?
    };
    self.rhs.clear();
    self.rhs.extend(r.iter().map(|x| -x));
    Ok(())
  }

  /// the tangent matrix of the last assembly with version All
  pub fn tangent_matrix(&self) -> &ColMatrix<f64> {
    &self.tangent_matrix
  }

  /// the rhs -R of the last assembly (on the primary unknowns, condensed,
  /// with AllCondensed)
  pub fn rhs(&self) -> &[f64] {
    &self.rhs
  }

  /// The internal unknowns' part of the solution at zero primary
  /// increment, of the last assembly with AllCondensed
  pub fn internal_solution(&self) -> Option<&[f64]> {
    self.condensed.as_ref().map(|c| c.2.as_slice())
  }

  /// the residual R, the tangent into self.tangent_matrix (its storage
  /// reused: clear keeps the columns' capacity)
  fn assemble_all(&mut self, u: &[f64]) -> Result<Vec<f64>> {
    let n = self.ws.nb_dof();
    let mut k = std::mem::take(&mut self.tangent_matrix);
    k.resize(n, n);
    k.clear();
    let mut r = vec![0.0; n];
    // the tangents and residuals of the nonlinear terms assembled together
    // directly into the global matrix and vector (no matrix per term to
    // merge), then the linear terms' kept matrices added
    let nl: Vec<_> = self
      .terms
      .iter()
      .filter(|t| !t.linear && t.order != 2)
      .map(|t| (t.expr.as_str(), &t.mim, &t.region))
      .collect();
    (k, r) = assemble_terms(&mut self.ws, &nl, Some(k), Some(r))?;
    for t in self.terms.iter_mut() {
      if !t.linear && t.order != 2 {
        continue;
      }
      let valid = t.linear
        && t.lin_contrib.as_ref().is_some_and(|c| {
          c.stamps == stamps(&self.ws, &t.mim, t.data.as_deref()).unwrap_or_default()
            && c.nb_dof == n
        });
      if valid {
        // linear: K u + (r - K u) of the assembly that filled lin_contrib
        let c = t.lin_contrib.as_ref().unwrap();
        let ku = mult_summed(&c.k, u);
        for i in 0..n {
          r[i] += ku[i] + c.r0[i];
        }
        k.add(&c.k);
        continue;
      }
      // the term's matrix, in the storage of its former lin_contrib (reused)
      let mut kt = t.lin_contrib.take().map(|c| c.k).unwrap_or_default();
      kt.resize(n, n);
      kt.clear();
      let term = [(t.expr.as_str(), &t.mim, &t.region)];
      let rt = (t.order != 2).then(|| vec![0.0; n]);
      let (kt, rt) = assemble_terms(&mut self.ws, &term, Some(kt), rt)?;
      let ku = (t.order == 2 || t.linear).then(|| mult_summed(&kt, u));
      let rt = match t.order {
        2 => ku.clone().unwrap(),
        _ => rt,
      };
      r.iter_mut().zip(&rt).for_each(|(a, b)| *a += b);
      k.add(&kt);
      t.lin_contrib = if t.linear {
        let ku = ku.unwrap();
        Some(LinContrib {
          stamps: stamps(&self.ws, &t.mim, t.data.as_deref())?,
          nb_dof: n,
          r0: rt.iter().zip(&ku).map(|(a, b)| a - b).collect(),
          k: kt,
        })
      } else {
        None
      };
    }
    let own = own_rows(&k);
    for e in &self.explicit {
      match e {
        Explicit::Matrix { var1, var2, m } => {
          let (Some(ri), Some(ci)) = (
            self.ws.interval_of_variable(var1),
            self.ws.interval_of_variable(var2),
          ) else {
            continue; // disabled
          };
          for i in 0..m.nrows {
            let owned = own.as_ref().is_none_or(|o| o[ri.start + i]);
            for (j, v) in m.row(i) {
              if owned {
                k.add_entry(ri.start + i, ci.start + j, v);
              }
              r[ri.start + i] += v * u[ci.start + j];
            }
          }
        }
        Explicit::Rhs { var, l } => {
          if let Some(ri) = self.ws.interval_of_variable(var) {
            r[ri].iter_mut().zip(l).for_each(|(a, b)| *a -= b);
          }
        }
      }
    }
    let checked = self.check_equations(&k);
    let r = checked.and_then(|_| self.eliminate(u, r, &mut k, own.as_deref()));
    self.tangent_matrix = k;
    r
  }

  /// Terms (valid linear caches: K u + r0) and explicit matrices and rhs.
  fn residual_of_terms(&mut self, u: &[f64]) -> Result<Vec<f64>> {
    let n = self.ws.nb_dof();
    let mut r = vec![0.0; n];
    // the terms without a valid lin_contrib: residuals, and matrices (order 2)
    // times u, each kind assembled together
    let (mut res, mut mat) = (vec![], vec![]);
    for t in &self.terms {
      let lin_contrib = t.lin_contrib.as_ref().filter(|c| {
        t.linear
          && c.stamps == stamps(&self.ws, &t.mim, t.data.as_deref()).unwrap_or_default()
          && c.nb_dof == n
      });
      match lin_contrib {
        Some(c) => {
          let ku = mult_summed(&c.k, u);
          for (a, (b, c)) in r.iter_mut().zip(ku.iter().zip(&c.r0)) {
            *a += b + c;
          }
        }
        None if t.order == 2 => mat.push((t.expr.as_str(), &t.mim, &t.region)),
        None => res.push((t.expr.as_str(), &t.mim, &t.region)),
      }
    }
    if !res.is_empty() {
      (_, r) = assemble_terms(&mut self.ws, &res, None, Some(r))?;
    }
    if !mat.is_empty() {
      let (k, _) = assemble_terms(&mut self.ws, &mat, Some(ColMatrix::new(n, n)), None)?;
      r.iter_mut()
        .zip(&mult_summed(&k, u))
        .for_each(|(a, b)| *a += b);
    }
    for e in &self.explicit {
      match e {
        Explicit::Matrix { var1, var2, m } => {
          let (Some(ri), Some(ci)) = (
            self.ws.interval_of_variable(var1),
            self.ws.interval_of_variable(var2),
          ) else {
            continue; // disabled
          };
          for i in 0..m.nrows {
            for (j, v) in m.row(i) {
              r[ri.start + i] += v * u[ci.start + j];
            }
          }
        }
        Explicit::Rhs { var, l } => {
          if let Some(ri) = self.ws.interval_of_variable(var) {
            r[ri].iter_mut().zip(l).for_each(|(a, b)| *a -= b);
          }
        }
      }
    }
    Ok(r)
  }

  /// BUILD_ALL_CONDENSED (GetFEM BUILD_RHS_WITH_INTERNAL, then
  /// BUILD_MATRIX_CONDENSED): the condensed tangent and rhs on the primary
  /// unknowns, the rhs of all, K'_QJ and the internal solution kept. All
  /// terms are assembled at each call (no lin_contrib on this path).
  fn assembly_condensed(&mut self) -> Result<()> {
    if !self.elim.is_empty() {
      return Err(Error::InvalidArgument(
        "Dirichlet elimination with internal variables: not available (use multipliers)".into(),
      ));
    }
    let internal = |v: &str| self.ws.is_internal_variable(v);
    for e in &self.explicit {
      if let Explicit::Matrix { var1, var2, .. } = e
        && (internal(var1) || internal(var2))
      {
        return Err(Error::InvalidArgument(format!(
          "explicit matrix on the internal variable {var1} or {var2}: not available"
        )));
      }
    }
    let (n, np) = (self.ws.nb_dof(), self.ws.nb_primary_dof());
    let r = self.residual_of_terms(&self.values_of_variables(true))?;
    let terms: Vec<(&str, &MeshIm, &Region)> = self
      .terms
      .iter()
      .map(|t| (t.expr.as_str(), &t.mim, &t.region))
      .collect();
    let ws = &mut self.ws;
    ws.clear_expressions();
    for &(e, mim, rg) in &terms {
      ws.add_expression(e, mim, rg)?;
    }
    ws.set_assembled_vector(r.clone());
    ws.set_assembled_matrix(ColMatrix::new(n, n));
    let res = ws.assembly(2, true);
    ws.clear_expressions();
    let (v, mut k, ckqj) = ws.release_assembled();
    res?;
    let own = own_rows(&k);
    for e in &self.explicit {
      if let Explicit::Matrix { var1, var2, m } = e
        && let (Some(ri), Some(ci)) = (
          self.ws.interval_of_variable(var1),
          self.ws.interval_of_variable(var2),
        )
      {
        for i in (0..m.nrows).filter(|&i| own.as_ref().is_none_or(|o| o[ri.start + i])) {
          for (j, v) in m.row(i) {
            k.add_entry(ri.start + i, ci.start + j, v);
          }
        }
      }
    }
    // the primary block (the internal rows and columns are empty)
    k.cols.truncate(np);
    (k.nrows, k.ncols) = (np, np);
    self.check_equations(&k)?;
    let mut kqj = Coo::new(n - np, np);
    for (j, col) in ckqj.cols.iter().enumerate().take(np) {
      for &(i, v) in col {
        if i >= np {
          kqj.push(i - np, j, v);
        }
      }
    }
    // the condensed residual on the primary unknowns, its rhs
    self.tangent_matrix = k;
    self.rhs = r[..np].iter().zip(&v).map(|(a, b)| -a - b).collect();
    self.condensed = Some((
      r.iter().map(|x| -x).collect(),
      kqj.to_csr()?,
      v[np..].iter().map(|x| -x).collect(),
    ));
    Ok(())
  }

  /// An enabled variable whose equations involve no enabled unknown makes
  /// the tangent singular: name it (typically the multiplier of a disabled
  /// variable). Internal variables are not in a condensed tangent.
  fn check_equations(&self, k: &ColMatrix<f64>) -> Result<()> {
    let mut used = vec![false; k.nrows];
    k.cols.iter().flatten().for_each(|&(i, _)| used[i] = true);
    let vars: Vec<_> = self
      .ws
      .variables()
      .into_iter()
      .filter(|(_, r)| r.end <= k.nrows && !r.is_empty())
      .collect();
    // the variables with an entry, on any MPI rank
    let mut has: Vec<f64> = vars
      .iter()
      .map(|(_, r)| f64::from(u8::from(r.clone().any(|i| used[i]))))
      .collect();
    getfem_mpi::sum(&mut has);
    for ((name, _), h) in vars.iter().zip(&has) {
      if *h == 0.0 {
        return Err(Error::InvalidArgument(format!(
          "the equations of {name} do not depend on any enabled variable; if {name} is the multiplier of a disabled variable, disable it too"
        )));
      }
    }
    Ok(())
  }

  /// Variables, data and terms as the model sees them: sizes, positions,
  /// disabled state, and for each term the detected order and linearity.
  pub fn summary(&self) -> String {
    let mut s = String::from("variables and data:\n");
    for (name, var, disabled, len) in self.ws.entries() {
      let what = match (var, disabled, self.ws.interval_of_variable(&name)) {
        (false, _, _) => "data".to_string(),
        (true, true, _) => "variable, disabled".to_string(),
        (true, false, r) => format!("variable, DOFs {:?}", r.unwrap_or(0..0)),
      };
      s += &format!("  {name}: {len} values, {what}\n");
    }
    s += "terms:\n";
    for (i, t) in self.terms.iter().enumerate() {
      let kind = ["potential", "residual", "bilinear form"][t.order];
      let lin = if t.linear {
        "linear, matrix cached"
      } else {
        "nonlinear"
      };
      let rg = region_name(&t.region);
      s += &format!("  {i}: {} ({kind}, {lin}, {rg})\n", t.expr);
    }
    for e in &self.elim {
      s += &format!(
        "  elimination of {} {}, value {}\n",
        e.var,
        region_name(&e.region),
        e.data.as_deref().unwrap_or("0")
      );
    }
    for e in &self.explicit {
      match e {
        Explicit::Matrix { var1, var2, m } => {
          s += &format!(
            "  explicit matrix {var1} x {var2}, {} x {}\n",
            m.nrows, m.ncols
          )
        }
        Explicit::Rhs { var, .. } => s += &format!("  explicit rhs of {var}\n"),
      }
    }
    s
  }

  /// Rows and columns of the eliminated DOFs: du_i = g_i - u_i.
  /// In place, as GetFEM's dof constraints on the tangent matrix.
  /// own: the rows this rank adds the model's entries to (MPI, own_rows)
  fn eliminate(
    &self,
    u: &[f64],
    mut r: Vec<f64>,
    k: &mut ColMatrix<f64>,
    own: Option<&[bool]>,
  ) -> Result<Vec<f64>> {
    if self.elim.is_empty() {
      return Ok(r);
    }
    let n = u.len();
    let mut fixed: Vec<Option<f64>> = vec![None; n];
    for e in &self.elim {
      let mf = self.required_mesh_fem(&e.var)?;
      let Some(off) = self.ws.interval_of_variable(&e.var).map(|r| r.start) else {
        continue; // disabled
      };
      let g = e.data.as_ref().map(|g| self.ws.variable(g).unwrap());
      if mf.is_reduced() {
        return Err(Error::InvalidArgument(format!(
          "elimination of {}: on a reduced (or partial) space not available (use multipliers)",
          e.var
        )));
      }
      for d in mf.basic_dof_on_region(&e.region)? {
        let gv = g.map_or(0.0, |g| g[d]);
        fixed[off + d] = Some(gv - u[off + d]);
      }
    }
    // the fixed columns times their increments to the rhs (summed over the
    // MPI ranks), the unit column by the row's owner
    let mut moved = vec![0.0; n];
    for (j, col) in k.cols.iter_mut().enumerate() {
      if let Some(d) = fixed[j] {
        for &(i, v) in col.iter() {
          if fixed[i].is_none() {
            moved[i] += v * d;
          }
        }
        *col = match own.is_none_or(|o| o[j]) {
          true => vec![(j, 1.0)],
          false => vec![],
        };
      } else {
        col.retain(|&(i, _)| fixed[i].is_none());
      }
    }
    getfem_mpi::sum(&mut moved);
    r.iter_mut().zip(&moved).for_each(|(a, b)| *a += b);
    for (ri, f) in r.iter_mut().zip(&fixed) {
      if let Some(d) = f {
        *ri = -d;
      }
    }
    Ok(r)
  }
}

impl NonlinearSystem for Model {
  fn values(&self) -> Vec<f64> {
    self.values_of_variables(true)
  }

  fn set_values(&mut self, u: &[f64]) -> Result<()> {
    self.set_values_of_variables(u, true)
  }

  fn residual_and_tangent(&mut self) -> Result<Linearized> {
    if self.ws.nb_internal_dof() > 0 {
      self.assembly(BuildVersion::AllCondensed)?;
      let (rhs, kqj, rhs_q) = self.condensed.clone().unwrap();
      return Ok(Linearized {
        rhs,
        condensed: Some(CondensedStep {
          rhs: self.rhs.clone(),
          kqj,
          rhs_q,
        }),
      });
    }
    self.assembly(BuildVersion::All)?;
    Ok(Linearized {
      rhs: self.rhs.clone(),
      condensed: None,
    })
  }

  fn tangent_matrix(&self) -> &ColMatrix<f64> {
    &self.tangent_matrix
  }

  fn variables(&self) -> Vec<(String, Range<usize>)> {
    self.ws.variables()
  }
}

/// The terms (expression, mim, region) assembled together by the workspace
/// (GetFEM: a model's expressions in one ga_workspace): added to k (order
/// 2) and r (order 1), each when given.
fn assemble_terms(
  ws: &mut Workspace,
  terms: &[(&str, &MeshIm, &Region)],
  k: Option<ColMatrix<f64>>,
  r: Option<Vec<f64>>,
) -> Result<(ColMatrix<f64>, Vec<f64>)> {
  ws.clear_expressions();
  let mut res = terms
    .iter()
    .try_for_each(|&(e, mim, rg)| ws.add_expression(e, mim, rg).map(|_| ()));
  if let Some(k) = k {
    ws.set_assembled_matrix(k);
    res = res.and_then(|_| ws.assembly(2, false));
  }
  if let Some(r) = r {
    ws.set_assembled_vector(r);
    res = res.and_then(|_| ws.assembly(1, false));
  }
  ws.clear_expressions();
  let (r, k, _) = ws.release_assembled();
  res?;
  Ok((k, r))
}

/// k u summed over the MPI ranks: each rank's matrices are its elements'
/// part (GetFEM: MPI_SUM_VECTOR of the products); k u alone on one rank
pub(crate) fn mult_summed(k: &ColMatrix<f64>, u: &[f64]) -> Vec<f64> {
  let mut v = k.mult(u);
  getfem_mpi::sum(&mut v);
  v
}

/// as mult_summed, for the condensation's coupling K'_QJ
pub(crate) fn matvec_summed(k: &Csr<f64>, u: &[f64]) -> Result<Vec<f64>> {
  let mut v = k.matvec(u)?;
  getfem_mpi::sum(&mut v);
  Ok(v)
}

/// With MPI on several ranks, the rows this rank adds the model's own
/// entries to (explicit matrices, the eliminated DOFs' 1s): the lowest rank
/// whose part of k has an entry in the row, rank 0 for rows of none; each
/// entry added once, exactly. None on one rank (all rows).
fn own_rows(k: &ColMatrix<f64>) -> Option<Vec<bool>> {
  let (n, r) = (getfem_mpi::size(), getfem_mpi::rank());
  if n < 2 {
    return None;
  }
  // n - (the lowest rank with an entry), 0 for none
  let mut low = vec![0.0; k.nrows];
  k.cols
    .iter()
    .flatten()
    .for_each(|&(i, _)| low[i] = (n - r) as f64);
  getfem_mpi::max(&mut low);
  Some(
    low
      .iter()
      .map(|&x| match x {
        0.0 => r == 0,
        _ => n - x as usize == r,
      })
      .collect(),
  )
}

/// Sparse columns summed over the MPI ranks (each rank's its elements'
/// part), the same on every rank: every rank's entries gathered and added
/// per entry in rank order (GetFEM MPI_SUM_SPARSE_MATRIX); as given on one
/// rank
pub(crate) fn summed_columns(cols: Vec<RsVector<f64>>) -> Vec<RsVector<f64>> {
  if getfem_mpi::size() == 1 {
    return cols;
  }
  let mine: Vec<f64> = cols
    .iter()
    .enumerate()
    .flat_map(|(j, c)| c.iter().flat_map(move |&(i, v)| [j as f64, i as f64, v]))
    .collect();
  let mut whole = std::collections::BTreeMap::new();
  for t in getfem_mpi::all_gather(&mine).chunks(3) {
    *whole.entry((t[0] as usize, t[1] as usize)).or_insert(0.0) += t[2];
  }
  let mut out: Vec<RsVector<f64>> = vec![vec![]; cols.len()];
  for ((j, i), v) in whole {
    out[j].push((i, v));
  }
  out
}
