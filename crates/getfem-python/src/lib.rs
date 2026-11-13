//! GetFEM-rs from Python, close to GetFEM's Python interface
//! (docs/design/m16-python.md)

#![forbid(unsafe_code)]

use pyo3::prelude::*;

mod descriptors;
mod eigsh;
mod error;
mod im_data;
mod level_set;
mod mesh;
mod mesh_fem;
mod mesh_im;
mod model;
mod mpi;
mod mumps;
mod slice;
mod spmat;
mod structured;

#[pymodule]
mod getfem_rs {
  use pyo3::prelude::*;

  #[pymodule_export]
  use crate::descriptors::{Fem, GeoTrans, Integ};
  #[pymodule_export]
  use crate::eigsh::eigsh;
  #[pymodule_export]
  use crate::im_data::MeshImData;
  #[pymodule_export]
  use crate::level_set::{GlobalFunction, LevelSet, MeshLevelSet};
  #[pymodule_export]
  use crate::mesh::Mesh;
  #[pymodule_export]
  use crate::mesh_fem::MeshFem;
  #[pymodule_export]
  use crate::mesh_im::MeshIm;
  #[pymodule_export]
  use crate::model::{Model, asm_generic, define_gwfl_function, interpolation};
  #[pymodule_export]
  use crate::mpi::mpi_init;
  #[pymodule_export]
  use crate::mumps::{MumpsContext, linsolve_mumps};
  #[pymodule_export]
  use crate::slice::Slice;
  #[pymodule_export]
  use crate::spmat::Spmat;

  /// the assembly's threads, for the whole program (1 by default; with
  /// MPI on several ranks only 1)
  #[pyfunction]
  fn set_num_threads(n: usize) -> PyResult<()> {
    use crate::error::ToPy;
    getfem::set_num_threads(n).py()
  }

  #[pyfunction]
  fn num_threads() -> usize {
    getfem::num_threads()
  }

  /// without mpi_init, each rank of a launcher runs its own sequential
  /// problem (mpi4py task farms): no warning
  #[pymodule_init]
  fn init(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    getfem::mpi::sequential_ranks_intended();
    Ok(())
  }

  /// GetFEM-rs' version
  #[pyfunction]
  fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
  }
}
