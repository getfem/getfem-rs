//! MPI (GetFEM's GETFEM_PARA_LEVEL helpers): the rank, the sums and the
//! broadcasts of data replicated on every rank. With the feature `mpi` (an
//! MPI's `mpicc` at build time); without it one rank, the sums no-ops,
//! inlined away. Started by a launcher, a program calls `init()` first;
//! without it each process runs the sequential code (a warning).
//!
//! On several ranks (as GetFEM): the mesh, the DOFs and every vector are
//! the same on all ranks; each rank assembles its part of the mesh (METIS),
//! so assembled scalars and vectors (residuals, interpolations) are summed
//! and whole, but an assembled matrix (a workspace's, a model's tangent)
//! is the rank's part, whole only summed over the ranks. MUMPS takes such
//! parts. Reading a matrix's entries, or solving one built by hand with
//! MUMPS directly, needs that in mind; an explicit matrix given to a model
//! is whole, the same on all ranks.

use getfem_linalg::Result;

#[cfg_attr(feature = "mpi", path = "par.rs")]
#[cfg_attr(not(feature = "mpi"), path = "seq.rs")]
mod imp;

/// MPI initialized by `init()`, finalized when dropped
pub struct Mpi(());

impl Drop for Mpi {
  fn drop(&mut self) {
    imp::finalize();
  }
}

/// Initializes MPI (once per program; MPI started quietly by an earlier
/// MUMPS solve is taken over, MUMPS instances made before keep solving
/// alone; MPI started by someone else, e.g. mpi4py, is adopted, finalized
/// by them). On several ranks each runs one thread: the assembly's and
/// MUMPS' set to 1 for good, said by rank 0 on stdout (MPI:
/// MPI_THREAD_FUNNELED)
pub fn init() -> Result<Mpi> {
  imp::init()?;
  Ok(Mpi(()))
}

/// MPI finalized (nothing more to call; MUMPS instances dropped later
/// skip their termination)
#[doc(hidden)]
pub fn finalized() -> bool {
  imp::finalized()
}

/// No warning for processes of a launcher without init(): each running
/// its own sequential problem is intended (Python, mpi4py task farms)
#[doc(hidden)]
pub fn sequential_ranks_intended() {
  SEQUENTIAL_INTENDED.store(true, std::sync::atomic::Ordering::Relaxed);
}

pub(crate) static SEQUENTIAL_INTENDED: std::sync::atomic::AtomicBool =
  std::sync::atomic::AtomicBool::new(false);

/// This process' rank, 0 to size() - 1
#[inline]
pub fn rank() -> usize {
  imp::rank()
}

/// The number of ranks
#[inline]
pub fn size() -> usize {
  imp::size()
}

#[inline]
pub fn is_master() -> bool {
  rank() == 0
}

/// v summed over the ranks, the same on every rank (GetFEM MPI_SUM_VECTOR)
#[inline]
pub fn sum(v: &mut [f64]) {
  if size() > 1 {
    imp::sum(v);
  }
}

/// the largest of v over the ranks, the same on every rank
#[inline]
pub fn max(v: &mut [f64]) {
  if size() > 1 {
    imp::max(v);
  }
}

/// every rank's v, concatenated in rank order, on every rank
pub fn all_gather(v: &[f64]) -> Vec<f64> {
  match size() {
    1 => v.to_vec(),
    _ => imp::all_gather(v),
  }
}

/// v of rank root on every rank
#[inline]
pub fn bcast(v: &mut [f64], root: usize) {
  if size() > 1 {
    imp::bcast(v, root);
  }
}

/// MUMPS' communicator (its comm_fortran): with the feature mpi all ranks
/// after init(), else each process alone (MPI initialized for it, finalized
/// at exit); None without the feature
pub fn comm_fortran() -> Option<i32> {
  imp::comm_fortran()
}

/// The parts (0 to nparts - 1) of the vertices of a graph (CSR: the
/// neighbours of vertex i are adjncy[xadj[i]..xadj[i + 1]]), balanced, few
/// edges between parts: METIS k-way, the same on every rank (GetFEM
/// mesh::compute_mpi_region). Without the feature mpi, or one part: all 0.
pub fn partition(xadj: &[usize], adjncy: &[usize], nparts: usize) -> Vec<usize> {
  assert!(
    !xadj.is_empty() && nparts >= 1,
    "partition: a CSR graph, nparts >= 1"
  );
  match nparts {
    1 => vec![0; xadj.len() - 1],
    _ => imp::partition(xadj, adjncy, nparts),
  }
}

/// The number of processes of a launcher (mpirun, mpiexec), if more than 1
/// the processes of a launcher (more than one), intended or not
fn launched_size() -> Option<usize> {
  ["OMPI_COMM_WORLD_SIZE", "PMI_SIZE"]
    .iter()
    .find_map(|v| std::env::var(v).ok()?.parse().ok())
    .filter(|&n| n > 1)
}
