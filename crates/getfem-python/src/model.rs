//! gf.Model (GetFEM gf_model, gf_model_get, gf_model_set)

use getfem::gwfl::workspace::Workspace;
use getfem::model::{
  BuildVersion, Constraint, ContinuationReport, ContinuationStatus, Method, Model as RsModel,
  Newton, NewtonReport as RsReport, NumericalContinuation, Status, VariableScale,
  VariableTolerance as RsTolerance,
};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyTuple};

use crate::error::ToPy;
use crate::im_data::MeshImData;
use crate::mesh::Mesh;
use crate::mesh_fem::MeshFem;
use crate::mesh_im::MeshIm;
use crate::spmat::{Inner, Spmat, sub_matrix};

#[pyclass(module = "getfem_rs")]
pub struct Model(pub(crate) RsModel);

/// values: a float numpy array (not copied), any sequence of numbers, or
/// a number
#[derive(FromPyObject)]
enum Values<'py> {
  Array(PyReadonlyArray1<'py, f64>),
  Seq(Vec<f64>),
  Num(f64),
}

#[derive(FromPyObject)]
enum Sizes {
  One(usize),
  Shape(Vec<usize>),
}

impl Sizes {
  fn shape(self) -> Vec<usize> {
    match self {
      Sizes::One(n) => vec![n],
      Sizes::Shape(s) => s,
    }
  }
}

/// set_variable's value: a GWFL expression or values
#[derive(FromPyObject)]
enum ValueOrExpr<'py> {
  Expr(String),
  Values(Values<'py>),
}

impl Values<'_> {
  fn as_slice(&self) -> PyResult<&[f64]> {
    match self {
      Values::Array(a) => Ok(a.as_slice()?),
      Values::Seq(v) => Ok(v),
      Values::Num(x) => Ok(std::slice::from_ref(x)),
    }
  }
}

/// md.solve's result: linear solves (all methods), converged, info (a
/// dict of Newton's report)
type Report<'py> = (usize, bool, Bound<'py, PyDict>);
/// numerical_continuation_step's result: ((T_X, T_gamma), h, info)
type ContStep<'py> = ((Bound<'py, PyArray1<f64>>, f64), f64, Bound<'py, PyDict>);

/// GetFEM-rs Newton method <-> its Python string
const METHODS: [(Method, &str); 3] = [
  (Method::Newton, "newton"),
  (Method::NleqErr, "nleq_err"),
  (Method::NleqRes, "nleq_res"),
];

fn method_name(m: Method) -> String {
  METHODS.iter().find(|(k, _)| *k == m).unwrap().1.to_string()
}

fn method(s: &str) -> PyResult<Method> {
  METHODS
    .iter()
    .find(|(_, n)| *n == s)
    .map(|(m, _)| *m)
    .ok_or_else(|| {
      PyValueError::new_err(format!(
        "unknown Newton method {s:?} (newton, nleq_err, nleq_res)"
      ))
    })
}

/// a variable tolerance: (variables, atol), variables a name or a list
/// (variables, value), variables a name or a list
fn variables_and_value(t: &Bound<'_, PyAny>, what: &str) -> PyResult<(Vec<String>, f64)> {
  #[derive(FromPyObject)]
  enum Vars {
    One(String),
    Many(Vec<String>),
  }
  let (v, x): (Vars, f64) = t.extract().map_err(|_| {
    PyValueError::new_err(format!(
      "{what}: (variables, {what}), variables a name or a list"
    ))
  })?;
  let v = match v {
    Vars::One(n) => vec![n],
    Vars::Many(v) => v,
  };
  Ok((v, x))
}

fn variable_tolerance(t: &Bound<'_, PyAny>) -> PyResult<RsTolerance> {
  let (variables, atol) = variables_and_value(t, "atol")?;
  Ok(RsTolerance { variables, atol })
}

/// info: method (that converged; None: not converged, or at the start),
/// message (why not converged), status, tolerance, scale, hint, and per
/// residual evaluation residuals, residuals_d, residuals_per_variable,
/// residuals_per_tolerance; per full Newton step increments
fn report(py: Python<'_>, r: RsReport) -> PyResult<Report<'_>> {
  let status = match r.status {
    Status::Converged => "converged",
    Status::MaxIterations => "max_iterations",
  };
  let dicts = |v: &[Vec<(String, f64)>]| -> PyResult<Vec<Bound<'_, PyDict>>> {
    v.iter().map(|p| p.clone().into_py_dict(py)).collect()
  };
  let info = PyDict::new(py);
  info.set_item("method", r.method.map(method_name))?;
  info.set_item("message", &r.message)?;
  info.set_item("status", status)?;
  info.set_item("tolerance", r.tolerance)?;
  info.set_item("scale", r.scale)?;
  info.set_item("hint", &r.hint)?;
  info.set_item("residuals", &r.residuals)?;
  info.set_item("residuals_d", &r.residuals_d)?;
  info.set_item("residuals_per_variable", dicts(&r.residuals_per_variable)?)?;
  info.set_item(
    "residuals_per_tolerance",
    dicts(&r.residuals_per_tolerance)?,
  )?;
  info.set_item("increments", dicts(&r.increments)?)?;
  Ok((r.iterations, r.converged(), info))
}

/// the info dict of a continuation step
fn continuation_info<'py>(py: Python<'py>, r: ContinuationReport) -> PyResult<Bound<'py, PyDict>> {
  let info = PyDict::new(py);
  let status = match r.status {
    ContinuationStatus::Initialized => "initialized",
    ContinuationStatus::Converged => "converged",
    ContinuationStatus::Failed => "failed",
  };
  info.set_item("status", status)?;
  info.set_item("h0", r.h0)?;
  info.set_item("sing_label", &r.sing_label)?;
  if r.x_sing.is_empty() {
    info.set_item("x_sing", py.None())?;
  } else {
    info.set_item("x_sing", PyArray1::from_vec(py, r.x_sing))?;
  }
  info.set_item("gamma_sing", r.gamma_sing)?;
  let t: Vec<(Bound<'_, PyArray1<f64>>, f64)> = r
    .tx_sing
    .into_iter()
    .zip(r.tgamma_sing)
    .map(|(x, g)| (PyArray1::from_vec(py, x), g))
    .collect();
  info.set_item("t_sing", t)?;
  let [a, b, c] = r.sing_indicators;
  info.set_item("sing_indicators", (a, b, c))?;
  info.set_item("tau_bp_graph", r.tau_bp_graph)?;
  Ok(info)
}

#[pymethods]
impl Model {
  #[new]
  fn new() -> Self {
    Model(RsModel::new())
  }

  fn add_fem_variable(&mut self, name: &str, mf: PyRef<'_, MeshFem>) -> PyResult<()> {
    self.0.add_fem_variable(name, &mf.0).py()
  }

  /// a variable of fixed size (sizes a number or a tensor shape)
  fn add_variable(&mut self, name: &str, sizes: Sizes) -> PyResult<()> {
    self.0.add_fixed_size_variable(name, &sizes.shape()).py()
  }

  /// [first, size] of an enabled variable in the system
  fn interval_of_variable<'py>(
    &self,
    py: Python<'py>,
    name: &str,
  ) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let r = self
      .0
      .interval_of_variable(name)
      .ok_or_else(|| PyValueError::new_err(format!("{name}: not an enabled variable")))?;
    Ok(PyArray1::from_slice(py, &[r.start as i64, r.len() as i64]))
  }

  /// the mesh_fem of a fem variable or data (a multiplier's too)
  fn mesh_fem_of_variable(&self, name: &str) -> PyResult<MeshFem> {
    self
      .0
      .mesh_fem_of_variable(name)
      .map(MeshFem)
      .ok_or_else(|| {
        PyValueError::new_err(format!(
          "mesh_fem_of_variable: {name:?}, no fem variable or data"
        ))
      })
  }

  /// the variable acts as data (keeps its value), the terms testing it
  /// dropped; its multipliers untouched (unlike GetFEM)
  fn disable_variable(&mut self, name: &str) -> PyResult<()> {
    self.0.disable_variable(name).py()
  }

  fn enable_variable(&mut self, name: &str) -> PyResult<()> {
    self.0.enable_variable(name).py()
  }

  /// a variable on the DOFs of mf on region (its basic DOFs there)
  fn add_filtered_fem_variable(
    &mut self,
    name: &str,
    mf: PyRef<'_, MeshFem>,
    region: i64,
  ) -> PyResult<()> {
    self.0.add_filtered_fem_variable(name, &mf.0, region).py()
  }

  /// a variable at the integration points of imd
  fn add_im_variable(&mut self, name: &str, imd: PyRef<'_, MeshImData>) -> PyResult<()> {
    self.0.add_im_variable(name, &imd.0).py()
  }

  /// a variable at the integration points of imd, condensed out of the
  /// tangent system
  fn add_internal_im_variable(&mut self, name: &str, imd: PyRef<'_, MeshImData>) -> PyResult<()> {
    self.0.add_internal_im_variable(name, &imd.0).py()
  }

  /// a scalar, or a vector of values; with sizes (a number or a tensor
  /// shape) a tensor, values in column-major order
  #[pyo3(signature = (name, value, sizes = None))]
  fn add_initialized_data(
    &mut self,
    name: &str,
    value: &Bound<'_, PyAny>,
    sizes: Option<Sizes>,
  ) -> PyResult<()> {
    if sizes.is_none()
      && let Ok(v) = value.extract::<f64>()
    {
      return self.0.add_scalar_data(name, v).py();
    }
    let v: Vec<f64> = match value.extract::<f64>() {
      Ok(x) => vec![x],
      Err(_) => value.extract()?,
    };
    let shape = sizes.map_or(vec![v.len()], Sizes::shape);
    self.0.add_fixed_size_data(name, &shape, &v).py()
  }

  /// fixed size data of sizes (a number or a tensor shape), zero
  fn add_data(&mut self, name: &str, sizes: Sizes) -> PyResult<()> {
    let shape = sizes.shape();
    let n = shape.iter().product();
    self.0.add_fixed_size_data(name, &shape, &vec![0.0; n]).py()
  }

  /// data on the DOFs of mf, zero (without GetFEM's sizes argument)
  fn add_fem_data(&mut self, name: &str, mf: PyRef<'_, MeshFem>) -> PyResult<()> {
    let n = mf.0.nb_dof().py()?;
    self.0.add_fem_data(name, &mf.0, &vec![0.0; n]).py()
  }

  /// data on the DOFs of mf, initialized with value (without GetFEM's
  /// sizes argument)
  fn add_initialized_fem_data(
    &mut self,
    name: &str,
    mf: PyRef<'_, MeshFem>,
    value: Values<'_>,
  ) -> PyResult<()> {
    self.0.add_fem_data(name, &mf.0, value.as_slice()?).py()
  }

  /// data at the integration points of imd, zero (values by
  /// set_variable)
  fn add_im_data(&mut self, name: &str, imd: PyRef<'_, MeshImData>) -> PyResult<()> {
    let n = imd.0.nb_values().py()?;
    self.0.add_im_data(name, &imd.0, &vec![0.0; n]).py()
  }

  /// a raytracing transformation for contact (Interpolate(.., name)),
  /// targets beyond release_distance ignored
  fn add_raytracing_transformation(&mut self, name: &str, release_distance: f64) -> PyResult<()> {
    self
      .0
      .add_raytracing_transformation(name, release_distance)
      .py()
  }

  /// the boundary region of m, displaced by disp, a target of name
  fn add_master_contact_boundary_to_raytracing_transformation(
    &mut self,
    name: &str,
    m: PyRef<'_, Mesh>,
    disp: &str,
    region: i64,
  ) -> PyResult<()> {
    self
      .0
      .add_master_contact_boundary_to_raytracing_transformation(name, &m.0, disp, region)
      .py()
  }

  /// the boundary region of m, displaced by disp, where name raytraces
  /// from
  fn add_slave_contact_boundary_to_raytracing_transformation(
    &mut self,
    name: &str,
    m: PyRef<'_, Mesh>,
    disp: &str,
    region: i64,
  ) -> PyResult<()> {
    self
      .0
      .add_slave_contact_boundary_to_raytracing_transformation(name, &m.0, disp, region)
      .py()
  }

  /// the DOFs of mf_mult (its basic DOFs on region) whose constraints
  /// on var ("value", or "normal_derivative" of a scalar var) are
  /// linearly independent, of each other and of var's multipliers added
  /// before: the multiplier space gf.MeshFem("partial", mf_mult, dofs)
  /// (GetFEM drops dependent ones silently, DEVIATIONS bug 14)
  #[pyo3(signature = (var, mim, mf_mult, region, constraint = "value"))]
  fn select_multiplier_dofs_for_variable<'py>(
    &self,
    py: Python<'py>,
    var: &str,
    mim: PyRef<'_, MeshIm>,
    mf_mult: PyRef<'_, MeshFem>,
    region: i64,
    constraint: &str,
  ) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let c = match constraint {
      "value" => Constraint::Value,
      "normal_derivative" => Constraint::NormalDerivative,
      k => {
        return Err(PyValueError::new_err(format!(
          "constraint: \"value\" or \"normal_derivative\" (not \"{k}\")"
        )));
      }
    };
    let d = self
      .0
      .select_multiplier_dofs_for_variable(var, &mim.0, &mf_mult.0, region, c)
      .py()?;
    Ok(PyArray1::from_vec(
      py,
      d.into_iter().map(|i| i as i64).collect(),
    ))
  }

  /// var = data (zero if None) on region by penalization: the term
  /// coef (var - data).Test_var
  #[allow(non_snake_case)]
  #[pyo3(signature = (mim, var, coef, region, data = None))]
  fn add_Dirichlet_condition_with_penalization(
    &mut self,
    mim: PyRef<'_, MeshIm>,
    var: &str,
    coef: f64,
    region: i64,
    data: Option<&str>,
  ) -> PyResult<()> {
    self
      .0
      .add_Dirichlet_condition_with_penalization(&mim.0, var, coef, region, data)
      .py()
  }

  /// var = data (zero if None) on region by a multiplier on mf_mult (a
  /// gf.MeshFem or a degree); the multiplier's name returned
  #[allow(non_snake_case)]
  #[pyo3(signature = (mim, var, mf_mult, region, data = None))]
  fn add_Dirichlet_condition_with_multiplier(
    &mut self,
    mim: PyRef<'_, MeshIm>,
    var: &str,
    mf_mult: &Bound<'_, PyAny>,
    region: i64,
    data: Option<&str>,
  ) -> PyResult<String> {
    if let Ok(mf) = mf_mult.cast::<MeshFem>() {
      let mf = &mf.get().0;
      return self
        .0
        .add_Dirichlet_condition_with_multiplier(&mim.0, var, mf, region, data)
        .py();
    }
    let k: usize = mf_mult.extract()?;
    self
      .0
      .add_Dirichlet_condition_with_multiplier(&mim.0, var, k, region, data)
      .py()
  }

  /// var = data (zero if None; fem data on var's space) on region, by
  /// elimination of var's DOFs there
  #[allow(non_snake_case)]
  #[pyo3(signature = (var, region, data = None))]
  fn add_Dirichlet_condition_with_simplification(
    &mut self,
    var: &str,
    region: i64,
    data: Option<&str>,
  ) -> PyResult<()> {
    self
      .0
      .add_Dirichlet_condition_with_simplification(var, region, data)
      .py()
  }

  /// an interpolate transformation from source to target, X -> expr
  fn add_interpolate_transformation_from_expression(
    &mut self,
    name: &str,
    source: PyRef<'_, Mesh>,
    target: PyRef<'_, Mesh>,
    expr: &str,
  ) -> PyResult<()> {
    self
      .0
      .add_interpolate_transformation_from_expression(name, &source.0, &target.0, expr)
      .py()
  }

  fn add_macro(&mut self, name: &str, expr: &str) -> PyResult<()> {
    self.0.add_macro(name, expr).py()
  }

  #[pyo3(signature = (mim, expr, region = -1))]
  fn add_linear_term(&mut self, mim: PyRef<'_, MeshIm>, expr: &str, region: i64) -> PyResult<()> {
    self
      .0
      .add_linear_term(&mim.0, region, expr)
      .py()
      .map(|_| ())
  }

  #[pyo3(signature = (mim, expr, region = -1))]
  fn add_nonlinear_term(
    &mut self,
    mim: PyRef<'_, MeshIm>,
    expr: &str,
    region: i64,
  ) -> PyResult<()> {
    self
      .0
      .add_nonlinear_term(&mim.0, region, expr)
      .py()
      .map(|_| ())
  }

  /// B, a gf.Spmat, the tangent block of the equations of var1 w.r.t.
  /// var2 (the residual gets B var2): set, replaced, or with None removed
  #[pyo3(signature = (var1, var2, b))]
  fn set_explicit_matrix(
    &mut self,
    py: Python<'_>,
    var1: &str,
    var2: &str,
    b: Option<PyRef<'_, Spmat>>,
  ) -> PyResult<()> {
    let b = b.map(|b| b.with(py, |k| k.to_csr()));
    self.0.set_explicit_matrix(var1, var2, b).py()
  }

  /// L, the right-hand side of the equations of var: set, replaced, or
  /// with None removed
  #[pyo3(signature = (var, l))]
  fn set_explicit_rhs(&mut self, var: &str, l: Option<Values<'_>>) -> PyResult<()> {
    let l = l.map(|l| l.as_slice().map(|x| x.to_vec())).transpose()?;
    self.0.set_explicit_rhs(var, l).py()
  }

  fn nb_dof(&self) -> usize {
    self.0.nb_dof()
  }

  /// a copy of the values of a variable or of data (as GetFEM, "variable"
  /// for both)
  fn variable<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyArray1<f64>>> {
    match self.0.variable(name) {
      Some(v) => Ok(PyArray1::from_slice(py, v)),
      None => Err(PyKeyError::new_err(format!("no variable or data {name}"))),
    }
  }

  /// the values of a variable or of data (as GetFEM, "variable" for both):
  /// values, or a GWFL expression interpolated on its space (a mesh_fem, on
  /// its basic DOFs then reduced, or an im_data) with the values before the
  /// change
  fn set_variable(&mut self, name: &str, v: ValueOrExpr<'_>) -> PyResult<()> {
    match v {
      ValueOrExpr::Expr(e) => self.0.set_variable(name, e.as_str()).py(),
      ValueOrExpr::Values(v) => self.0.set_variable(name, v.as_slice()?).py(),
    }
  }

  /// several variables or data from GWFL expressions, (name, expr) pairs
  /// (each as set_variable), all evaluated with the values before the
  /// change, then set
  fn set_variables(&mut self, pairs: Vec<(String, String)>) -> PyResult<()> {
    let p: Vec<(&str, &str)> = pairs
      .iter()
      .map(|(n, e)| (n.as_str(), e.as_str()))
      .collect();
    self.0.set_variables(&p).py()
  }

  /// the enabled variables' values, in the order of the system, without
  /// the internal ones
  #[allow(clippy::wrong_self_convention)] // GetFEM's name
  fn from_variables<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, self.0.from_variables())
  }

  /// sets the enabled variables, without the internal ones
  #[allow(clippy::wrong_self_convention)] // GetFEM's name
  fn to_variables(&mut self, u: Values<'_>) -> PyResult<()> {
    self.0.to_variables(u.as_slice()?).py()
  }

  /// md.interpolation(expr, mf | imd | Pts, m[, region]): expr on the
  /// DOFs of a gf.MeshFem, at the points of a gf.MeshImData or at the
  /// points Pts (dim x n) of the mesh m (on the convexes of region)
  #[pyo3(signature = (expr, target, *args))]
  fn interpolation<'py>(
    &self,
    py: Python<'py>,
    expr: &str,
    target: &Bound<'_, PyAny>,
    args: &Bound<'_, PyTuple>,
  ) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (t, rest) = Target::of(target, args)?;
    let region: i64 = match rest.len() {
      0 => -1,
      1 => rest.get_item(0)?.extract()?,
      _ => {
        return Err(PyValueError::new_err(
          "md.interpolation: expr, target[, m][, region]",
        ));
      }
    };
    let v = match &t {
      Target::Mf(mf) => self.0.interpolation(expr, &mf.get().0, region),
      Target::Imd(imd) => self.0.interpolation(expr, &imd.get().0, region),
      Target::Pts(pts, m) => self.0.interpolation(expr, (pts, &m.get().0), region),
    };
    Ok(PyArray1::from_vec(py, v.py()?))
  }

  /// element-wise L2 projection of expr on mf (discontinuous)
  #[pyo3(signature = (mim, expr, mf, region = -1))]
  fn local_projection<'py>(
    &self,
    py: Python<'py>,
    mim: PyRef<'_, MeshIm>,
    expr: &str,
    mf: PyRef<'_, MeshFem>,
    region: i64,
  ) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let v = self.0.local_projection(expr, &mim.0, &mf.0, region).py()?;
    Ok(PyArray1::from_vec(py, v))
  }

  /// the model's assembly, the GIL released: version "all" (the tangent
  /// matrix and the rhs, kept in the model: md.tangent_matrix(),
  /// md.rhs()), "rhs" (the rhs only) or "all_condensed" (internal
  /// variables condensed out)
  #[pyo3(signature = (version = "all"))]
  fn assembly(&mut self, py: Python<'_>, version: &str) -> PyResult<()> {
    let v = match version {
      "all" => BuildVersion::All,
      "rhs" => BuildVersion::Rhs,
      "all_condensed" => BuildVersion::AllCondensed,
      _ => {
        return Err(PyValueError::new_err(format!(
          "assembly: version {version:?}, not all, rhs or all_condensed"
        )));
      }
    };
    let md = &mut self.0;
    py.detach(|| md.assembly(v)).py()
  }

  /// the tangent matrix of the last assembly: a read-only view (a Spmat),
  /// always the current one, no copy
  fn tangent_matrix(slf: Bound<'_, Self>) -> Spmat {
    Spmat(Inner::Tangent(slf.unbind()))
  }

  /// the rhs -R of the last assembly
  fn rhs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_slice(py, self.0.rhs())
  }

  /// Newton (its cascade of methods), the GIL released; converged when the
  /// global and all variable tolerances hold. Options: Rust Newton's
  /// fields; methods as strings ("newton", "nleq_err", "nleq_res"), each
  /// variable tolerance (variables, atol), variables a name or a list.
  /// Returns (iterations, converged, info) (GetFEM: the first two), info a
  /// dict of Newton's report (see report)
  #[allow(clippy::too_many_arguments)]
  #[pyo3(signature = (
    *, methods = None, max_iter = None, atol = None, rtol = None,
    variable_tolerances = vec![], verbose = false
  ))]
  fn solve<'py>(
    &mut self,
    py: Python<'py>,
    methods: Option<Vec<String>>,
    max_iter: Option<usize>,
    atol: Option<f64>,
    rtol: Option<f64>,
    variable_tolerances: Vec<Bound<'_, PyAny>>,
    verbose: bool,
  ) -> PyResult<Report<'py>> {
    let d = Newton::default();
    let methods = match methods {
      Some(v) => v.iter().map(|s| method(s)).collect::<PyResult<_>>()?,
      None => d.methods.clone(),
    };
    let newton = Newton {
      methods,
      max_iter: max_iter.unwrap_or(d.max_iter),
      atol: atol.unwrap_or(d.atol),
      rtol: rtol.unwrap_or(d.rtol),
      variable_tolerances: variable_tolerances
        .iter()
        .map(variable_tolerance)
        .collect::<PyResult<_>>()?,
      verbose,
    };
    let md = &mut self.0;
    let r = py.detach(|| newton.solve(md, None)).py()?;
    report(py, r)
  }

  /// One Moore-Penrose continuation step in the scalar data parameter,
  /// dimensionless from 0 to 1 (GetFEM's ContStruct and
  /// Moore_Penrose_continuation, as one function), its metric over the
  /// variables [(variables, scale), ...] (variables a name or a list,
  /// scale a typical magnitude of their values; GetFEM: sc_fac,
  /// variable_name),
  /// from the model's point along the tangent T = (T_X, T_gamma) with the
  /// step length h; the model holds the new point (the start point if
  /// failed). T None: the tangent at the model's point, oriented by
  /// direction, h = h_init (GetFEM's init_Moore_Penrose_continuation).
  /// sing_indicators: info["sing_indicators"] of the previous step (with
  /// singularities > 0).
  /// Options: GetFEM's, noisy 0, 1, 2, and gamma_max (default inf): the
  /// path ends there (a step from gamma >= gamma_max fails). Returns
  /// (T, h, info), info: status ("initialized", "converged", "failed"), h0
  /// (the step length used), sing_label, x_sing, gamma_sing, t_sing (the
  /// branch tangents), sing_indicators, tau_bp_graph
  #[allow(clippy::too_many_arguments)]
  #[pyo3(signature = (
    parameter, variables, T = None, h = 0.0, *, direction = 1.0, sing_indicators = None,
    parametrised_data = None, gamma_max = None, h_init = None, h_max = None, h_min = None,
    h_inc = None, h_dec = None, max_iter = None, thr_iter = None, max_res = None,
    max_diff = None, min_cos = None, singularities = None, non_smooth = None,
    delta_max = None, delta_min = None, thr_var = None, nb_dir = None, nb_span = None,
    seed = None, noisy = None
  ))]
  #[allow(non_snake_case)]
  fn numerical_continuation_step<'py>(
    &mut self,
    py: Python<'py>,
    parameter: &str,
    variables: Vec<Bound<'_, PyAny>>,
    T: Option<(PyReadonlyArray1<'_, f64>, f64)>,
    h: f64,
    direction: f64,
    sing_indicators: Option<(f64, f64, f64)>,
    parametrised_data: Option<(String, String, String)>,
    gamma_max: Option<f64>,
    h_init: Option<f64>,
    h_max: Option<f64>,
    h_min: Option<f64>,
    h_inc: Option<f64>,
    h_dec: Option<f64>,
    max_iter: Option<usize>,
    thr_iter: Option<usize>,
    max_res: Option<f64>,
    max_diff: Option<f64>,
    min_cos: Option<f64>,
    singularities: Option<u8>,
    non_smooth: Option<bool>,
    delta_max: Option<f64>,
    delta_min: Option<f64>,
    thr_var: Option<f64>,
    nb_dir: Option<usize>,
    nb_span: Option<usize>,
    seed: Option<u32>,
    noisy: Option<usize>,
  ) -> PyResult<ContStep<'py>> {
    let variables = variables
      .iter()
      .map(|t| {
        let (variables, scale) = variables_and_value(t, "scale")?;
        Ok(VariableScale { variables, scale })
      })
      .collect::<PyResult<_>>()?;
    let mut s = NumericalContinuation::new(parameter, variables);
    s.parametrised_data = parametrised_data;
    macro_rules! set {
      ($($f:ident),*) => { $(if let Some(v) = $f { s.$f = v; })* };
    }
    set!(
      gamma_max,
      h_init,
      h_max,
      h_min,
      h_inc,
      h_dec,
      max_iter,
      thr_iter,
      max_res,
      max_diff,
      min_cos,
      singularities,
      non_smooth,
      delta_max,
      delta_min,
      thr_var,
      nb_dir,
      nb_span,
      seed,
      noisy
    );
    let (mut tx, mut tgamma) = match T {
      Some((x, g)) => (x.as_slice()?.to_vec(), g),
      None => (vec![], direction),
    };
    let mut h = h;
    let sing_indicators = sing_indicators.map(|(a, b, c)| [a, b, c]);
    let md = &mut self.0;
    let r = py
      .detach(|| s.step(md, &mut tx, &mut tgamma, &mut h, sing_indicators))
      .py()?;
    Ok((
      (PyArray1::from_vec(py, tx), tgamma),
      h,
      continuation_info(py, r)?,
    ))
  }
}

/// the target of an interpolation: a mesh_fem, an im_data or the points
/// of a mesh
enum Target<'py> {
  Mf(Bound<'py, MeshFem>),
  Imd(Bound<'py, MeshImData>),
  Pts(Vec<Vec<f64>>, Bound<'py, Mesh>),
}

impl<'py> Target<'py> {
  /// the target and the arguments after it (Pts, m: the mesh the first)
  fn of(
    target: &Bound<'py, PyAny>,
    args: &Bound<'py, PyTuple>,
  ) -> PyResult<(Self, Bound<'py, PyTuple>)> {
    if let Ok(mf) = target.cast::<MeshFem>() {
      return Ok((Target::Mf(mf.clone()), args.clone()));
    }
    if let Ok(imd) = target.cast::<MeshImData>() {
      return Ok((Target::Imd(imd.clone()), args.clone()));
    }
    let pts: PyReadonlyArray2<'_, f64> = target.extract().map_err(|_| {
      PyValueError::new_err("interpolation: a gf.MeshFem, a gf.MeshImData or points Pts, m")
    })?;
    let m = match args.len() {
      0 => None,
      _ => args.get_item(0)?.cast::<Mesh>().ok().cloned(),
    }
    .ok_or_else(|| PyValueError::new_err("interpolation: the mesh m after the points"))?;
    let pts = pts
      .as_array()
      .columns()
      .into_iter()
      .map(|c| c.to_vec())
      .collect();
    Ok((Target::Pts(pts, m), args.get_slice(1, args.len())))
  }
}

/// gf.interpolation(expr, mf | imd | Pts, m[, region][, (name, mf | imd,
/// U), ...]): expr on the DOFs of mf, at the points of imd or at the points
/// Pts (dim x n) of the mesh m (on the convexes of region), its fields
/// given as (name, mf | imd, U); no model (md.interpolation with one).
/// gf.interpolation(mf, U, mf2 | sl): U (on mf) onto mf2 or a slice
#[pyfunction]
#[pyo3(signature = (expr, target, *args))]
pub fn interpolation<'py>(
  py: Python<'py>,
  expr: &Bound<'py, PyAny>,
  target: &Bound<'py, PyAny>,
  args: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
  if let Ok(mf) = expr.cast::<MeshFem>() {
    if args.len() != 1 {
      return Err(PyValueError::new_err("gf.interpolation: mf, U, mf2 | sl"));
    }
    return crate::mesh_fem::interpolate_on(py, mf.get(), target.extract()?, &args.get_item(0)?);
  }
  let expr: String = expr.extract()?;
  let expr = expr.as_str();
  let (t, mut rest) = Target::of(target, args)?;
  let mut region = -1;
  if let Some(r) = rest.iter().next().and_then(|r| r.extract::<i64>().ok()) {
    region = r;
    rest = rest.get_slice(1, rest.len());
  }
  let mut ws = Workspace::default();
  for f in rest.iter() {
    let (name, space, u): (String, Bound<'_, PyAny>, PyReadonlyArray1<'_, f64>) = f
      .extract()
      .map_err(|_| PyValueError::new_err("gf.interpolation: fields as (name, mf | imd, U)"))?;
    match space.cast::<MeshFem>() {
      Ok(mf) => ws.add_fem_data(&name, &mf.get().0, u.as_slice()?),
      Err(_) => ws.add_im_data(&name, &space.cast::<MeshImData>()?.get().0, u.as_slice()?),
    }
    .py()?;
  }
  let v = match &t {
    Target::Mf(mf) => ws.interpolation(expr, &mf.get().0, region),
    Target::Imd(imd) => ws.interpolation(expr, &imd.get().0, region),
    Target::Pts(pts, m) => ws.interpolation(expr, (pts, &m.get().0), region),
  };
  Ok(PyArray1::from_vec(py, v.py()?))
}

/// gf.define_gwfl_function(name, nargs, expr, der1="", der2=""): a GWFL
/// function of t (and u) for the whole program, its derivatives der1, der2
/// (GetFEM's gf.asm_define_function); a name in use: an error
#[pyfunction]
#[pyo3(signature = (name, nargs, expr, der1 = "", der2 = ""))]
pub fn define_gwfl_function(
  name: &str,
  nargs: usize,
  expr: &str,
  der1: &str,
  der2: &str,
) -> PyResult<()> {
  getfem::gwfl::define_gwfl_function(name, nargs, expr, der1, der2).py()
}

/// select_output: a variable (order 1) or a pair (order 2)
#[derive(FromPyObject)]
pub enum SelectOutput {
  One(String),
  Two(String, String),
}

/// gf.asm_generic(mim, order, expr, region[, md], select_output=None):
/// expr assembled (with the model's variables and data if md), order 0
/// (a scalar), 1 (a vector over the model's DOFs) or 2 (a gf.Spmat over
/// them); an expression of lower order differentiated (order 2 of a
/// residual: its tangent); select_output: the rows of a variable (order
/// 1), the block of a pair of variables, rows and columns (order 2)
/// (GetFEM's option, as a keyword)
#[pyfunction]
#[pyo3(signature = (mim, order, expr, region, md = None, select_output = None))]
pub fn asm_generic<'py>(
  py: Python<'py>,
  mim: PyRef<'_, MeshIm>,
  order: usize,
  expr: &str,
  region: i64,
  md: Option<PyRefMut<'_, Model>>,
  select_output: Option<SelectOutput>,
) -> PyResult<Bound<'py, PyAny>> {
  if order > 2 {
    return Err(PyValueError::new_err("asm_generic: order 0, 1 or 2"));
  }
  let mut md = md;
  let mut own = Workspace::default();
  let ws = match md.as_mut() {
    Some(md) => md.0.workspace_mut(),
    None => &mut own,
  };
  let interval = |v: &str| {
    ws.interval_of_variable(v)
      .ok_or_else(|| PyValueError::new_err(format!("asm_generic: {v}: not an enabled variable")))
  };
  let (rows, cols) = match (&select_output, order) {
    (None, _) => (None, None),
    (Some(SelectOutput::One(v)), 1) => (Some(interval(v)?), None),
    (Some(SelectOutput::Two(v, w)), 2) => (Some(interval(v)?), Some(interval(w)?)),
    _ => {
      return Err(PyValueError::new_err(
        "asm_generic: select_output a variable at order 1, a pair at order 2",
      ));
    }
  };
  ws.clear_expressions();
  let o = ws.add_expression(expr, &mim.0, region).py()?;
  if o > order {
    ws.clear_expressions();
    return Err(PyValueError::new_err(format!(
      "asm_generic: {expr} is of order {o}, more than {order}"
    )));
  }
  let r = ws.assembly(order, false).py();
  ws.clear_expressions();
  r?;
  match (order, rows, cols) {
    (0, ..) => Ok(ws.assembled_potential().into_pyobject(py)?.into_any()),
    (1, r, _) => {
      let v = ws.assembled_vector();
      let v = r.map_or(v, |r| &v[r]);
      Ok(PyArray1::from_slice(py, v).into_any())
    }
    (_, Some(r), Some(c)) => {
      let k = sub_matrix(
        ws.assembled_matrix(),
        &r.collect::<Vec<_>>(),
        &c.collect::<Vec<_>>(),
      )?;
      Ok(Bound::new(py, Spmat::own(k))?.into_any())
    }
    _ => Ok(Bound::new(py, Spmat::own(ws.assembled_matrix().clone()))?.into_any()),
  }
}
