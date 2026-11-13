//! gf.GeoTrans, gf.Fem, gf.Integ: by name (GetFEM gf_geotrans, gf_fem,
//! gf_integ)

use getfem::fem::fem::{PFem, fem_descriptor};
use getfem::fem::integration::{PIntegration, integ_descriptor};
use getfem::geom::geotrans::{PGeoTrans, geotrans_descriptor};
use pyo3::prelude::*;

use crate::error::ToPy;

#[pyclass(module = "getfem_rs", frozen)]
pub struct GeoTrans(pub PGeoTrans);

#[pymethods]
impl GeoTrans {
  /// gf.GeoTrans("GT_PK(2,1)")
  #[new]
  fn new(name: &str) -> PyResult<Self> {
    geotrans_descriptor(name).py().map(GeoTrans)
  }

  fn name(&self) -> String {
    self.0.name()
  }

  fn dim(&self) -> usize {
    self.0.dim()
  }

  fn __repr__(&self) -> String {
    format!("gf.GeoTrans(\"{}\")", self.0.name())
  }
}

#[pyclass(module = "getfem_rs", frozen)]
pub struct Fem(pub PFem);

#[pymethods]
impl Fem {
  /// gf.Fem("FEM_PK(2,1)")
  #[new]
  fn new(name: &str) -> PyResult<Self> {
    fem_descriptor(name).py().map(Fem)
  }

  fn name(&self) -> String {
    self.0.name().to_string()
  }

  fn dim(&self) -> usize {
    self.0.dim()
  }

  fn __repr__(&self) -> String {
    format!("gf.Fem(\"{}\")", self.0.name())
  }
}

#[pyclass(module = "getfem_rs", frozen)]
pub struct Integ(pub PIntegration);

#[pymethods]
impl Integ {
  /// gf.Integ("IM_TRIANGLE(6)")
  #[new]
  fn new(name: &str) -> PyResult<Self> {
    integ_descriptor(name).py().map(Integ)
  }

  fn name(&self) -> String {
    self.0.name().to_string()
  }

  fn __repr__(&self) -> String {
    format!("gf.Integ(\"{}\")", self.0.name())
  }
}
