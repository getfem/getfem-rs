//! MUMPS as a LinearSolver (sequential; with the feature mpi the MPI one:
//! on several ranks each gives its matrix entries, summed by MUMPS), on
//! num_threads() threads (its BLAS', its OpenMP's). The only unsafe code is the FFI calls and the
//! zero-initialization of the C struct.

use std::sync::Mutex;

use getfem_linalg::{Error, LinearSolver, MatrixKind, Result, SolverInfo, SparseMatrix};
mod ffi;
mod threads;
use ffi::{DMUMPS_STRUC_C, dmumps_c};
pub use threads::{BlasInfo, blas_info, num_threads, set_num_threads};

/// Sequential MUMPS is not documented as thread safe: one call at a time.
static MUMPS_LOCK: Mutex<()> = Mutex::new(());

/// MUMPS' "use MPI_COMM_WORLD" communicator (ignored by the sequential
/// stub).
const USE_COMM_WORLD: i32 = -987654;

pub struct Mumps {
  s: Box<DMUMPS_STRUC_C>,
  kind: Option<MatrixKind>,
  n: usize,
  // the struct points into these; kept alive and in place between jobs
  irn: Vec<i32>,
  jcn: Vec<i32>,
  /// the entries given to MUMPS (the lower triangle if symmetric), in the
  /// matrix's order; irn, jcn: their 1-based rows and columns
  a: Vec<f64>,
  factorized: bool,
  /// MPI on several ranks: each rank's entries (ICNTL(18) = 3), summed by
  /// MUMPS; the right-hand side of rank 0, the solution to all
  dist: bool,
  /// the user's ICNTL and CNTL values (0-based), set at each analysis
  icntl: Vec<(usize, i32)>,
  cntl: Vec<(usize, f64)>,
}

// Safety: the struct only points into buffers owned by Mumps; all calls are
// serialized by MUMPS_LOCK.
unsafe impl Send for Mumps {}

impl Default for Mumps {
  fn default() -> Self {
    Self::new()
  }
}

impl Mumps {
  pub fn new() -> Self {
    // Safety: DMUMPS_STRUC_C is plain C data (ints, floats, pointers, chars);
    // all zero is a valid value, and MUMPS initializes it at job = -1.
    let s: Box<DMUMPS_STRUC_C> = Box::new(unsafe { std::mem::zeroed() });
    Mumps {
      s,
      kind: None,
      n: 0,
      irn: vec![],
      jcn: vec![],
      a: vec![],
      factorized: false,
      dist: false,
      icntl: vec![],
      cntl: vec![],
    }
  }

  /// ICNTL(i) = v (1-based, as the MUMPS user guide), from the next
  /// analysis on (and now, if analyzed); after GetFEM-rs' defaults (output
  /// off)
  pub fn set_icntl(&mut self, i: usize, v: i32) -> Result<()> {
    if !(1..=60).contains(&i) {
      return Err(Error::InvalidArgument(format!(
        "MUMPS: ICNTL({i}), not 1 to 60"
      )));
    }
    self.icntl.retain(|e| e.0 != i - 1);
    self.icntl.push((i - 1, v));
    if self.kind.is_some() {
      self.s.icntl[i - 1] = v;
    }
    Ok(())
  }

  /// INFOG(i) of the last MUMPS call (1-based, as the MUMPS user guide;
  /// the same on all ranks), also after a failed one
  pub fn infog(&self, i: usize) -> Result<i32> {
    match (1..=80).contains(&i) {
      true => Ok(self.s.infog[i - 1]),
      false => Err(Error::InvalidArgument(format!(
        "MUMPS: INFOG({i}), not 1 to 80"
      ))),
    }
  }

  /// RINFOG(i) of the last MUMPS call (1-based), as infog
  pub fn rinfog(&self, i: usize) -> Result<f64> {
    match (1..=40).contains(&i) {
      true => Ok(self.s.rinfog[i - 1]),
      false => Err(Error::InvalidArgument(format!(
        "MUMPS: RINFOG({i}), not 1 to 40"
      ))),
    }
  }

  /// CNTL(i) = v (1-based, as the MUMPS user guide), as set_icntl
  pub fn set_cntl(&mut self, i: usize, v: f64) -> Result<()> {
    if !(1..=15).contains(&i) {
      return Err(Error::InvalidArgument(format!(
        "MUMPS: CNTL({i}), not 1 to 15"
      )));
    }
    self.cntl.retain(|e| e.0 != i - 1);
    self.cntl.push((i - 1, v));
    if self.kind.is_some() {
      self.s.cntl[i - 1] = v;
    }
    Ok(())
  }

  fn call(&mut self, job: i32) -> Result<()> {
    let _g = MUMPS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    self.s.job = job;
    let (irn, jcn, a) = (
      self.irn.as_mut_ptr(),
      self.jcn.as_mut_ptr(),
      self.a.as_mut_ptr(),
    );
    match self.dist {
      true => (self.s.irn_loc, self.s.jcn_loc, self.s.a_loc) = (irn, jcn, a),
      false => (self.s.irn, self.s.jcn, self.s.a) = (irn, jcn, a),
    }
    threads::apply();
    // Safety: s is initialized (job = -1 first) and points to live buffers.
    unsafe { dmumps_c(&mut *self.s) };
    let (i1, i2) = (self.s.infog[0], self.s.infog[1]);
    if i1 >= 0 {
      return Ok(());
    }
    Err(match i1 {
      -10 => Error::Singular(" (MUMPS: numerically singular)".into()),
      -6 => Error::Singular(format!(" (MUMPS: structurally singular, rank {i2})")),
      _ => Error::Solver {
        solver: "MUMPS",
        code: i1 as i64,
        msg: message(i1, i2),
      },
    })
  }

  fn init(&mut self, kind: MatrixKind) -> Result<()> {
    if self.kind.is_some() {
      self.terminate();
    }
    self.s.par = 1;
    self.s.sym = match kind {
      MatrixKind::General => 0,
      MatrixKind::SymmetricPositiveDefinite => 1,
      MatrixKind::Symmetric => 2,
    };
    self.dist = getfem_mpi::size() > 1;
    self.s.comm_fortran = getfem_mpi::comm_fortran().unwrap_or(USE_COMM_WORLD);
    self.call(-1)?;
    // silence MUMPS output
    for i in 0..3 {
      self.s.icntl[i] = -1;
    }
    self.s.icntl[3] = 0;
    for &(i, v) in &self.icntl {
      self.s.icntl[i] = v;
    }
    for &(i, v) in &self.cntl {
      self.s.cntl[i] = v;
    }
    self.kind = Some(kind);
    Ok(())
  }

  fn terminate(&mut self) {
    // after MPI_Finalize (a Python object freed at exit) MUMPS can't end
    if self.kind.take().is_some() && !getfem_mpi::finalized() {
      let _ = self.call(-2);
    }
    self.factorized = false;
  }

  /// Retry on workspace errors (-8, -9, -14, -15, -17, -20) with more memory.
  fn call_growing(&mut self, job: i32) -> Result<()> {
    for _ in 0..6 {
      match self.call(job) {
        Err(Error::Solver {
          code: -8 | -9 | -14 | -15 | -17 | -20,
          ..
        }) => self.s.icntl[13] = (self.s.icntl[13].max(20)) * 2,
        r => return r,
      }
    }
    self.call(job)
  }
}

fn message(i1: i32, i2: i32) -> String {
  match i1 {
    -1 => format!("error on processor {i2}"),
    -2 => format!("nnz = {i2} out of range"),
    -3 => "invalid job sequence".into(),
    -5 | -7 | -13 => format!("memory allocation failed ({i2})"),
    -8 | -9 | -14 | -15 | -17 | -20 => format!("workspace too small (INFOG(2) = {i2})"),
    -12 => "matrix is not positive definite".into(),
    -16 => format!("N = {i2} out of range"),
    _ => format!("INFOG(1) = {i1}, INFOG(2) = {i2} (see the MUMPS user guide)"),
  }
}

impl LinearSolver for Mumps {
  fn analyze(&mut self, a: &dyn SparseMatrix, kind: MatrixKind) -> Result<()> {
    let (n, nnz) = (a.nrows(), a.nnz());
    if n != a.ncols() {
      return Err(Error::InvalidArgument(format!(
        "MUMPS: matrix is {n}x{}",
        a.ncols()
      )));
    }
    if n > i32::MAX as usize || nnz > i32::MAX as usize {
      return Err(Error::InvalidArgument(
        "MUMPS: matrix too large for 32-bit indices".into(),
      ));
    }
    self.init(kind)?;
    let sym = kind != MatrixKind::General;
    let (irn, jcn, vals) = (&mut self.irn, &mut self.jcn, &mut self.a);
    irn.clear();
    jcn.clear();
    vals.clear();
    a.for_each_entry(&mut |i, j, v| {
      if !sym || j <= i {
        irn.push(i as i32 + 1);
        jcn.push(j as i32 + 1);
        vals.push(v);
      }
    });
    self.n = n;
    self.s.n = n as i32;
    let nnz = self.a.len();
    match self.dist {
      true => (self.s.nnz_loc, self.s.nz_loc, self.s.icntl[17]) = (nnz as i64, nnz as i32, 3),
      false => (self.s.nnz, self.s.nz) = (nnz as i64, nnz as i32),
    }
    self.factorized = false;
    self.call_growing(1)
  }

  fn factorize(&mut self, a: &dyn SparseMatrix) -> Result<()> {
    // the values, the pattern checked against the analyzed one on the way
    let sym = self.kind != Some(MatrixKind::General);
    let (mut k, mut same) = (0, self.kind.is_some() && a.nrows() == self.n);
    let (irn, jcn, vals) = (&self.irn, &self.jcn, &mut self.a);
    a.for_each_entry(&mut |i, j, v| {
      if sym && j > i {
        return;
      }
      if same && k < vals.len() && irn[k] == i as i32 + 1 && jcn[k] == j as i32 + 1 {
        vals[k] = v;
      } else {
        same = false;
      }
      k += 1;
    });
    // on several ranks all fail or none (MUMPS' calls are collective)
    let mut bad = [f64::from(u8::from(!same || k != self.a.len()))];
    if self.dist {
      getfem_mpi::sum(&mut bad);
    }
    if bad[0] > 0.0 {
      return Err(Error::InvalidArgument(
        "MUMPS: factorize needs the analyzed pattern".into(),
      ));
    }
    self.factorized = false;
    self.call_growing(2)?;
    // MUMPS does not always stop on a negative pivot with SYM = 1
    if self.kind == Some(MatrixKind::SymmetricPositiveDefinite) && self.s.infog[11] > 0 {
      return Err(Error::Solver {
        solver: "MUMPS",
        code: -12,
        msg: format!(
          "matrix is not positive definite ({} negative pivots)",
          self.s.infog[11]
        ),
      });
    }
    self.factorized = true;
    Ok(())
  }

  fn solve(&mut self, b: &mut [f64]) -> Result<()> {
    if !self.factorized {
      return Err(Error::InvalidArgument(
        "MUMPS: solve before factorize".into(),
      ));
    }
    if b.len() != self.n {
      return Err(Error::InvalidArgument(format!(
        "MUMPS: right-hand side of size {} for n = {}",
        b.len(),
        self.n
      )));
    }
    // on several ranks rank 0's b (the same on all), its solution to all
    self.s.rhs = b.as_mut_ptr();
    self.s.nrhs = 1;
    self.s.lrhs = self.n as i32;
    let r = self.call(3);
    self.s.rhs = std::ptr::null_mut();
    if self.dist && r.is_ok() {
      getfem_mpi::bcast(b, 0);
    }
    r
  }

  fn info(&self) -> SolverInfo {
    let f = self.s.infog[28] as i64;
    SolverInfo {
      solver: "MUMPS",
      n: self.n,
      nnz: self.a.len(),
      nnz_factors: self.factorized.then_some(if f < 0 {
        (-f as u64) * 1_000_000
      } else {
        f as u64
      }),
      flops: self.factorized.then_some(self.s.rinfog[2]),
      messages: vec![],
    }
  }

  fn release(&mut self) {
    self.terminate();
  }
}

impl Drop for Mumps {
  fn drop(&mut self) {
    self.terminate();
  }
}
