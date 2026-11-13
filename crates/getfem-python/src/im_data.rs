//! gf.MeshImData (GetFEM gf_mesh_im_data)

use getfem::fem::im_data::ImData;
use pyo3::prelude::*;

use crate::error::ToPy;
use crate::mesh_im::MeshIm;

#[pyclass(module = "getfem_rs", frozen)]
pub struct MeshImData(pub ImData);

#[pymethods]
impl MeshImData {
  /// gf.MeshImData(mim[, region[, shape]]): a value of shape (a tensor
  /// shape or a size; scalar if none) per integration point of region
  #[new]
  #[pyo3(signature = (mim, region = -1, shape = None))]
  fn new(mim: PyRef<'_, MeshIm>, region: i64, shape: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
    let shape: Vec<usize> = match shape {
      None => vec![],
      Some(s) => match s.extract::<usize>() {
        Ok(n) => vec![n],
        Err(_) => s.extract()?,
      },
    };
    ImData::new(&mim.0, region, &shape).py().map(MeshImData)
  }

  /// the number of integration points
  fn nb_points(&self) -> PyResult<usize> {
    self.0.nb_points().py()
  }

  /// the number of values (points x tensor size)
  fn nb_values(&self) -> PyResult<usize> {
    self.0.nb_values().py()
  }

  fn tensor_shape(&self) -> Vec<usize> {
    self.0.tensor_shape()
  }
}
