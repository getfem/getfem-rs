//! GetFEM-rs' errors as Python exceptions, by kind: what does not exist or
//! does not fit, ValueError; the rest (singular, solvers), RuntimeError

use getfem::{fem, geom, gwfl, io, linalg, model};
use pyo3::PyErr;
use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};

/// a Rust error as a Python exception
pub trait ToPy<T> {
  fn py(self) -> pyo3::PyResult<T>;
}

fn value(e: impl ToString) -> PyErr {
  PyValueError::new_err(e.to_string())
}

fn runtime(e: impl ToString) -> PyErr {
  PyRuntimeError::new_err(e.to_string())
}

fn linalg_err(e: linalg::Error) -> PyErr {
  match e {
    linalg::Error::InvalidArgument(_) => value(e),
    _ => runtime(e),
  }
}

fn geom_err(e: geom::Error) -> PyErr {
  match e {
    geom::Error::InvalidArgument(_) | geom::Error::Parse(_) | geom::Error::NotImplemented(_) => {
      value(e)
    }
    geom::Error::Singular => runtime(e),
    geom::Error::Linalg(e) => linalg_err(e),
  }
}

fn fem_err(e: fem::Error) -> PyErr {
  match e {
    fem::Error::InvalidArgument(_) | fem::Error::NotImplemented(_) => value(e),
    fem::Error::Geom(e) => geom_err(e),
    fem::Error::Linalg(e) => linalg_err(e),
  }
}

impl<T> ToPy<T> for geom::Result<T> {
  fn py(self) -> pyo3::PyResult<T> {
    self.map_err(geom_err)
  }
}

impl<T> ToPy<T> for fem::Result<T> {
  fn py(self) -> pyo3::PyResult<T> {
    self.map_err(fem_err)
  }
}

fn gwfl_err(e: gwfl::Error) -> PyErr {
  match e {
    gwfl::Error::Fem(e) => fem_err(e),
    gwfl::Error::Geom(e) => geom_err(e),
    gwfl::Error::Linalg(e) => linalg_err(e),
    gwfl::Error::Internal(_) => runtime(e),
    _ => value(e),
  }
}

fn model_err(e: model::Error) -> PyErr {
  match e {
    model::Error::NotImplemented(_) | model::Error::InvalidArgument(_) => value(e),
    model::Error::Gwfl(e) => gwfl_err(e),
    model::Error::Fem(e) => fem_err(e),
    model::Error::Geom(e) => geom_err(e),
    model::Error::Linalg(e) => linalg_err(e),
    model::Error::Solve(_) => runtime(e),
  }
}

impl<T> ToPy<T> for model::Result<T> {
  fn py(self) -> pyo3::PyResult<T> {
    self.map_err(model_err)
  }
}

impl<T> ToPy<T> for gwfl::Result<T> {
  fn py(self) -> pyo3::PyResult<T> {
    self.map_err(gwfl_err)
  }
}

fn io_err(e: io::Error) -> PyErr {
  match e {
    io::Error::Io(_) => PyOSError::new_err(e.to_string()),
    io::Error::Format(_) => value(e),
    io::Error::Geom(e) => geom_err(e),
    io::Error::Fem(e) => fem_err(e),
  }
}

impl<T> ToPy<T> for io::Result<T> {
  fn py(self) -> pyo3::PyResult<T> {
    self.map_err(io_err)
  }
}

impl<T> ToPy<T> for linalg::Result<T> {
  fn py(self) -> pyo3::PyResult<T> {
    self.map_err(linalg_err)
  }
}
