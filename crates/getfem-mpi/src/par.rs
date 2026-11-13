//! With the feature mpi: MPI_COMM_WORLD through src/shim.c

use std::ffi::c_int;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Once, OnceLock};

use getfem_linalg::{Error, Result};

unsafe extern "C" {
  fn getfem_mpi_initialized(flag: *mut c_int) -> c_int;
  fn getfem_mpi_init(funneled: *mut c_int) -> c_int;
  fn getfem_mpi_finalize() -> c_int;
  fn getfem_mpi_finalized(flag: *mut c_int) -> c_int;
  fn getfem_mpi_query_funneled(funneled: *mut c_int) -> c_int;
  fn getfem_mpi_rank(rank: *mut c_int) -> c_int;
  fn getfem_mpi_size(size: *mut c_int) -> c_int;
  fn getfem_mpi_sum(v: *mut f64, n: c_int) -> c_int;
  fn getfem_mpi_max(v: *mut f64, n: c_int) -> c_int;
  fn getfem_mpi_allgather_counts(n: c_int, counts: *mut c_int) -> c_int;
  fn getfem_mpi_allgatherv(
    v: *const f64,
    n: c_int,
    out: *mut f64,
    counts: *const c_int,
    displs: *const c_int,
  ) -> c_int;
  fn getfem_mpi_bcast(v: *mut f64, n: c_int, root: c_int) -> c_int;
  fn getfem_mpi_comm_world_fortran() -> c_int;
  fn getfem_mpi_comm_self_fortran(comm: *mut c_int) -> c_int;
  fn getfem_metis_part(
    n: i64,
    xadj: *const i64,
    adjncy: *const i64,
    nparts: i64,
    part: *mut i64,
  ) -> c_int;
}

static RANK: AtomicUsize = AtomicUsize::new(0);
/// MUMPS' communicator of a process alone, MPI started quietly for it
static SELF: OnceLock<i32> = OnceLock::new();
/// 0: not initialized
static SIZE: AtomicUsize = AtomicUsize::new(0);
/// MPI started by someone else (mpi4py), not finalized here
static ADOPTED: AtomicBool = AtomicBool::new(false);

fn check(e: c_int, call: &str) -> Result<()> {
  if e == 0 {
    return Ok(());
  }
  Err(Error::Solver {
    solver: "MPI",
    code: e.into(),
    msg: format!("{call}: error {e}"),
  })
}

pub fn init() -> Result<()> {
  let mut flag = 0;
  // SAFETY: an out-parameter
  check(
    unsafe { getfem_mpi_initialized(&mut flag) },
    "MPI_Initialized",
  )?;
  if SIZE.load(Ordering::Relaxed) != 0 {
    return Err(Error::InvalidArgument(
      "getfem::mpi::init: called twice".into(),
    ));
  }
  // MPI started quietly for MUMPS before (SELF, MPI_THREAD_SERIALIZED):
  // taken over; by someone else (mpi4py): adopted, finalized by them
  let adopted = flag != 0 && SELF.get().is_none();
  let (mut funneled, mut r, mut n) = (1, 0, 0);
  // SAFETY: out-parameters; MPI initialized before rank and size
  unsafe {
    if flag == 0 {
      check(getfem_mpi_init(&mut funneled), "MPI_Init_thread")?;
    } else if adopted {
      check(getfem_mpi_query_funneled(&mut funneled), "MPI_Query_thread")?;
    }
    check(getfem_mpi_rank(&mut r), "MPI_Comm_rank")?;
    check(getfem_mpi_size(&mut n), "MPI_Comm_size")?;
  }
  if funneled == 0 {
    if !adopted {
      // SAFETY: MPI initialized here
      unsafe { getfem_mpi_finalize() };
    }
    return Err(Error::InvalidArgument(
      "getfem::mpi::init: the MPI does not provide MPI_THREAD_FUNNELED".into(),
    ));
  }
  ADOPTED.store(adopted, Ordering::Relaxed);
  RANK.store(r as usize, Ordering::Relaxed);
  SIZE.store(n as usize, Ordering::Relaxed);
  // single-threaded ranks, said once
  if n > 1 {
    getfem_linalg::threads::force_single_thread();
    if r == 0 {
      println!("GetFEM-rs: MPI on {n} ranks, one thread per rank (assembly and MUMPS)");
    }
  }
  Ok(())
}

pub fn finalize() {
  if SIZE.swap(0, Ordering::Relaxed) != 0 {
    RANK.store(0, Ordering::Relaxed);
    if !ADOPTED.load(Ordering::Relaxed) && !finalized() {
      // SAFETY: MPI initialized by init()
      unsafe { getfem_mpi_finalize() };
    }
  }
}

pub fn finalized() -> bool {
  let mut flag = 0;
  // SAFETY: an out-parameter (callable before init and after finalize)
  unsafe { getfem_mpi_finalized(&mut flag) };
  flag != 0
}

#[inline]
pub fn rank() -> usize {
  RANK.load(Ordering::Relaxed)
}

#[inline]
pub fn size() -> usize {
  match SIZE.load(Ordering::Relaxed) {
    0 => not_initialized(),
    n => n,
  }
}

/// one rank; a warning (once) when started by a launcher
#[cold]
fn not_initialized() -> usize {
  static WARN: Once = Once::new();
  WARN.call_once(|| {
    if let Some(n) = super::launched_size()
      && !super::SEQUENTIAL_INTENDED.load(Ordering::Relaxed)
    {
      eprintln!(
        "GetFEM-rs warning: {n} processes without getfem::mpi::init(): \
         each runs the sequential code"
      );
    }
  });
  1
}

/// MPI's counts are ints: blocks of at most c_int::MAX
fn blocks(v: &mut [f64]) -> impl Iterator<Item = &mut [f64]> {
  v.chunks_mut(c_int::MAX as usize)
}

pub fn sum(v: &mut [f64]) {
  for b in blocks(v) {
    // SAFETY: b.len() values at b, MPI initialized (size() > 1)
    let e = unsafe { getfem_mpi_sum(b.as_mut_ptr(), b.len() as c_int) };
    assert_eq!(e, 0, "MPI_Allreduce: error {e}");
  }
}

pub fn max(v: &mut [f64]) {
  for b in blocks(v) {
    // SAFETY: as sum
    let e = unsafe { getfem_mpi_max(b.as_mut_ptr(), b.len() as c_int) };
    assert_eq!(e, 0, "MPI_Allreduce: error {e}");
  }
}

pub fn all_gather(v: &[f64]) -> Vec<f64> {
  let n = c_int::try_from(v.len()).expect("all_gather: more than c_int::MAX values");
  let mut counts = vec![0 as c_int; size()];
  // SAFETY: one count per rank
  let e = unsafe { getfem_mpi_allgather_counts(n, counts.as_mut_ptr()) };
  assert_eq!(e, 0, "MPI_Allgather: error {e}");
  let displs: Vec<c_int> = counts
    .iter()
    .scan(0, |s, &c| {
      let d = *s;
      *s += c;
      Some(d)
    })
    .collect();
  let total = counts.iter().map(|&c| c as usize).sum();
  let mut out = vec![0.0; total];
  // SAFETY: n values at v, total values at out, counts and displs per rank
  let e = unsafe {
    getfem_mpi_allgatherv(
      v.as_ptr(),
      n,
      out.as_mut_ptr(),
      counts.as_ptr(),
      displs.as_ptr(),
    )
  };
  assert_eq!(e, 0, "MPI_Allgatherv: error {e}");
  out
}

pub fn bcast(v: &mut [f64], root: usize) {
  for b in blocks(v) {
    // SAFETY: as sum
    let e = unsafe { getfem_mpi_bcast(b.as_mut_ptr(), b.len() as c_int, root as c_int) };
    assert_eq!(e, 0, "MPI_Bcast: error {e}");
  }
}

pub fn partition(xadj: &[usize], adjncy: &[usize], nparts: usize) -> Vec<usize> {
  let n = xadj.len() - 1;
  let x: Vec<i64> = xadj.iter().map(|&i| i as i64).collect();
  let a: Vec<i64> = adjncy.iter().map(|&i| i as i64).collect();
  let mut part = vec![0i64; n];
  // SAFETY: CSR arrays of n vertices, n outputs
  let e = unsafe {
    getfem_metis_part(
      n as i64,
      x.as_ptr(),
      a.as_ptr(),
      nparts as i64,
      part.as_mut_ptr(),
    )
  };
  assert_eq!(e, 0, "METIS_PartGraphKway: error {e}");
  part.into_iter().map(|p| p as usize).collect()
}

pub fn comm_fortran() -> Option<i32> {
  if SIZE.load(Ordering::Relaxed) != 0 {
    // SAFETY: MPI initialized by init()
    return Some(unsafe { getfem_mpi_comm_world_fortran() });
  }
  // once, whatever the thread
  Some(*SELF.get_or_init(|| {
    let mut c = 0;
    // SAFETY: an out-parameter
    let e = unsafe { getfem_mpi_comm_self_fortran(&mut c) };
    assert_eq!(e, 0, "MPI_Init_thread: error {e}");
    c
  }))
}
