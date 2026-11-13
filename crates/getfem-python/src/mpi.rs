//! gf.mpi_init: the script's problem on all ranks of MPI_COMM_WORLD

use std::sync::Mutex;

use pyo3::prelude::*;

use crate::error::ToPy;

/// MPI joined by mpi_init, released at exit
static MPI: Mutex<Option<getfem::mpi::Mpi>> = Mutex::new(None);

/// The assembly and MUMPS on all ranks of MPI_COMM_WORLD (getfem::mpi::init;
/// the MPI build, under mpirun): MPI started by mpi4py adopted, else
/// started here and finalized at exit. Without it each rank runs its own
/// sequential problem. A second call does nothing; in the sequential
/// build, nothing (a warning under a launcher).
#[pyfunction]
pub fn mpi_init(py: Python<'_>) -> PyResult<()> {
  let mut mpi = MPI.lock().unwrap_or_else(|e| e.into_inner());
  if mpi.is_some() {
    return Ok(());
  }
  *mpi = Some(getfem::mpi::init().py()?);
  // before the objects are freed and mpi4py's own finalization
  let at_exit = wrap_pyfunction!(mpi_release, py)?;
  py.import("atexit")?.call_method1("register", (at_exit,))?;
  Ok(())
}

/// MPI released at exit (finalized if started by mpi_init)
#[pyfunction]
fn mpi_release() {
  MPI.lock().unwrap_or_else(|e| e.into_inner()).take();
}
