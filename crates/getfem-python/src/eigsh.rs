//! gf.eigsh: ARPACK-ng's eigenpairs of K phi = lambda M phi nearest sigma

use getfem::linalg::{ColMatrix, SparseMatrix};
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;

use crate::error::ToPy;
use crate::spmat::Spmat;

/// eigenvalues, modes as rows
type EigenPairs<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>);

/// The nev eigenvalues nearest sigma of K phi = lambda M phi (K
/// symmetric, M symmetric positive semi-definite, none: the identity),
/// ascending, and their modes as rows (M-orthonormal): vals, modes. After
/// gf.mpi_init on several ranks: K, M each rank's part (as assembled)
#[pyfunction]
#[pyo3(signature = (k, m = None, nev = 6, sigma = 0.0))]
pub fn eigsh<'py>(
  py: Python<'py>,
  k: &Bound<'py, Spmat>,
  m: Option<&Bound<'py, Spmat>>,
  nev: usize,
  sigma: f64,
) -> PyResult<EigenPairs<'py>> {
  // copies: the matrices may be views of a model, the GIL released
  let k: ColMatrix<f64> = k.get().with(py, Clone::clone);
  let m: Option<ColMatrix<f64>> = m.map(|m| m.get().with(py, Clone::clone));
  let (vals, modes) = py
    .detach(|| getfem::arpack::eigsh(&k, m.as_ref().map(|m| m as &dyn SparseMatrix), nev, sigma))
    .py()?;
  Ok((
    PyArray1::from_vec(py, vals),
    PyArray2::from_vec2(py, &modes)?,
  ))
}
