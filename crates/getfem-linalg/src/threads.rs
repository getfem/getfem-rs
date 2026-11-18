//! Number of threads of the assembly, one setting for the whole program,
//! 1 by default; 1 for good with MPI on several ranks (single-threaded
//! ranks: more is an error)

use std::sync::Once;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::{Error, Result};

static NUM_THREADS: AtomicUsize = AtomicUsize::new(1);
static SINGLE: AtomicBool = AtomicBool::new(false);

/// Sets the number of threads (at least 1); with MPI on several ranks
/// only 1, more is an error
pub fn set_num_threads(n: usize) -> Result<()> {
  let n = n.max(1);
  if n > 1 && SINGLE.load(Ordering::Relaxed) {
    return Err(Error::InvalidArgument(format!(
      "set_num_threads({n}): MPI on several ranks runs one thread per rank"
    )));
  }
  NUM_THREADS.store(n, Ordering::Relaxed);
  Ok(())
}

pub fn num_threads() -> usize {
  NUM_THREADS.load(Ordering::Relaxed)
}

/// One thread from now on (getfem::mpi::init on several ranks); a warning
/// if more were set
pub fn force_single_thread() {
  SINGLE.store(true, Ordering::Relaxed);
  if NUM_THREADS.swap(1, Ordering::Relaxed) > 1 {
    warn_single();
  }
}

#[cold]
fn warn_single() {
  static WARN: Once = Once::new();
  WARN.call_once(|| {
    eprintln!(
      "GetFEM-rs warning: MPI on several ranks: one thread per rank (the threads set before ignored)"
    );
  });
}
