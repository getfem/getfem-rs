//! gf.MumpsContext: MUMPS on a Spmat, copied into MUMPS' arrays at
//! analyze and factorize

use getfem::linalg::{LinearSolver, MatrixKind, SparseMatrix};
use getfem::mumps::Mumps;
use std::sync::Mutex;

use numpy::{AllowTypeChange, PyArray1, PyArrayDyn, PyArrayLikeDyn, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::error::ToPy;
use crate::spmat::Spmat;

/// MUMPS (GetFEM-rs' sequential or MPI build): set_matrix, analyze,
/// factorize, solve; set_icntl, set_cntl as the MUMPS user guide (1-based)
#[pyclass(module = "getfem_rs")]
pub struct MumpsContext {
  /// a Mutex: a pyclass is Sync, Mumps only Send
  mumps: Mutex<Mumps>,
  kind: MatrixKind,
  a: Option<Py<Spmat>>,
}

impl MumpsContext {
  /// analyze (true) or factorize with the matrix set, the GIL released
  fn run(&mut self, py: Python<'_>, analyze: bool) -> PyResult<()> {
    let (mumps, kind) = (
      self.mumps.get_mut().unwrap_or_else(|e| e.into_inner()),
      self.kind,
    );
    let job = |m: &mut Mumps, a: &dyn SparseMatrix| match analyze {
      true => m.analyze(a, kind),
      false => m.factorize(a),
    };
    let a = self
      .a
      .as_ref()
      .ok_or_else(|| PyValueError::new_err("MumpsContext: set_matrix first"))?;
    a.get().with(py, |k| {
      if k.nrows == 0 {
        return Err(PyValueError::new_err(
          "MumpsContext: the model has no tangent matrix yet (md.assembly or md.solve)",
        ));
      }
      py.detach(|| job(mumps, k)).py()
    })
  }
}

#[pymethods]
impl MumpsContext {
  /// sym as MUMPS' SYM: 0 general, 1 symmetric positive definite, 2
  /// symmetric (then the full matrix or its lower triangle)
  #[new]
  #[pyo3(signature = (sym = 0))]
  fn new(sym: i32) -> PyResult<Self> {
    let kind = match sym {
      0 => MatrixKind::General,
      1 => MatrixKind::SymmetricPositiveDefinite,
      2 => MatrixKind::Symmetric,
      _ => {
        return Err(PyValueError::new_err(format!(
          "MumpsContext: sym = {sym}, not 0, 1 or 2"
        )));
      }
    };
    Ok(MumpsContext {
      mumps: Mutex::new(Mumps::new()),
      kind,
      a: None,
    })
  }

  /// the matrix, a Spmat: ctx.set_matrix(md.tangent_matrix()), read at
  /// analyze and factorize
  fn set_matrix(&mut self, a: Py<Spmat>) {
    self.a = Some(a);
  }

  /// symbolic analysis of the matrix set
  fn analyze(&mut self, py: Python<'_>) -> PyResult<()> {
    self.run(py, true)
  }

  /// numerical factorization of the matrix set (its pattern the analyzed one)
  fn factorize(&mut self, py: Python<'_>) -> PyResult<()> {
    self.run(py, false)
  }

  /// the solution of A x = b with the current factors
  fn solve<'py>(&mut self, py: Python<'py>, b: Vec<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let mut x = b;
    let mumps = self.mumps.get_mut().unwrap_or_else(|e| e.into_inner());
    py.detach(|| mumps.solve(&mut x)).py()?;
    Ok(PyArray1::from_vec(py, x))
  }

  /// ICNTL(i) = value (1-based), from now on, also for later analyses
  fn set_icntl(&mut self, i: usize, value: i32) -> PyResult<()> {
    self
      .mumps
      .get_mut()
      .unwrap_or_else(|e| e.into_inner())
      .set_icntl(i, value)
      .py()
  }

  /// INFOG(i) of the last MUMPS call (1-based, as the MUMPS user guide),
  /// also after a failed one
  fn infog(&mut self, i: usize) -> PyResult<i32> {
    self
      .mumps
      .get_mut()
      .unwrap_or_else(|e| e.into_inner())
      .infog(i)
      .py()
  }

  /// RINFOG(i) of the last MUMPS call (1-based), as infog
  fn rinfog(&mut self, i: usize) -> PyResult<f64> {
    self
      .mumps
      .get_mut()
      .unwrap_or_else(|e| e.into_inner())
      .rinfog(i)
      .py()
  }

  /// CNTL(i) = value (1-based), as set_icntl
  fn set_cntl(&mut self, i: usize, value: f64) -> PyResult<()> {
    self
      .mumps
      .get_mut()
      .unwrap_or_else(|e| e.into_inner())
      .set_cntl(i, value)
      .py()
  }
}

/// gf.linsolve_mumps(M, b[, "sym"]): the solution of M x = b (GetFEM's),
/// b a vector or a matrix of right hand sides (columns); "sym": M
/// symmetric (MUMPS' SYM 2). After gf.mpi_init on several ranks: M each
/// rank's part (as assembled, summed by MUMPS; as GetFEM), b whole, x on
/// all ranks
#[pyfunction]
#[pyo3(signature = (m, b, sym = None))]
pub fn linsolve_mumps<'py>(
  py: Python<'py>,
  m: &Bound<'py, Spmat>,
  b: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
  sym: Option<&str>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
  let kind = match sym {
    None => MatrixKind::General,
    Some("sym") => MatrixKind::Symmetric,
    Some(o) => {
      return Err(PyValueError::new_err(format!(
        "linsolve_mumps: unknown option {o}, not \"sym\""
      )));
    }
  };
  let b = b.as_array();
  let shape = b.shape().to_vec();
  let n = m.get().with(py, |k| k.nrows);
  if shape.is_empty() || shape.len() > 2 || shape[0] != n || m.get().with(py, |k| k.ncols) != n {
    return Err(PyValueError::new_err(format!(
      "linsolve_mumps: M is {n}x{}, b {shape:?}",
      m.get().with(py, |k| k.ncols)
    )));
  }
  let nrhs = shape.get(1).copied().unwrap_or(1);
  // columns, contiguous
  let mut x: Vec<f64> = (0..nrhs)
    .flat_map(|j| (0..n).map(move |i| (i, j)))
    .map(|(i, j)| if shape.len() == 1 { b[[i]] } else { b[[i, j]] })
    .collect();
  let k = m.get().with(py, Clone::clone);
  py.detach(|| -> getfem::linalg::Result<()> {
    let mut lu = Mumps::new();
    lu.analyze(&k, kind)?;
    lu.factorize(&k)?;
    for c in x.chunks_mut(n) {
      lu.solve(c)?;
    }
    Ok(())
  })
  .py()?;
  // back to b's shape (row-major)
  let y: Vec<f64> = (0..n)
    .flat_map(|i| (0..nrhs).map(move |j| (i, j)))
    .map(|(i, j)| x[i + j * n])
    .collect();
  PyArray1::from_vec(py, y).reshape(shape)
}
