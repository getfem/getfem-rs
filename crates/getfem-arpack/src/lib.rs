//! ARPACK-ng: eigenpairs of K phi = lambda M phi nearest a shift sigma (K
//! symmetric, M symmetric positive semi-definite, none: the identity), by
//! implicitly restarted Lanczos (dsaupd, dseupd) in shift-invert mode on
//! (K - sigma M)^-1 M, K - sigma M factorized by MUMPS. The only unsafe code
//! is the FFI calls.

use std::os::raw::c_int;
use std::sync::Mutex;

use getfem_linalg::{ColMatrix, Error, LinearSolver, MatrixKind, Rand, Result, SparseMatrix};
use getfem_mumps::Mumps;

mod ffi;

/// ARPACK keeps state between calls (Fortran SAVE): one at a time.
static ARPACK_LOCK: Mutex<()> = Mutex::new(());

fn arpack_error(routine: &str, info: c_int) -> Error {
  let msg = match info {
    1 => "maximum number of iterations taken".into(),
    3 => "no shifts could be applied (increase ncv)".into(),
    -9999 => "could not build an Arnoldi factorization".into(),
    _ => format!("{routine} info = {info} (see the ARPACK documentation)"),
  };
  Error::Solver {
    solver: "ARPACK",
    code: info as i64,
    msg,
  }
}

/// The nev eigenvalues nearest sigma, ascending, and their modes
/// (M-orthonormal; the identity's if no M). MPI on several ranks: K and M
/// the ranks' parts (as assembled), called by every rank, the results the
/// same on all.
pub fn eigsh(
  k: &dyn SparseMatrix,
  m: Option<&dyn SparseMatrix>,
  nev: usize,
  sigma: f64,
) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
  let n = k.nrows();
  if k.ncols() != n || m.is_some_and(|m| m.nrows() != n || m.ncols() != n) {
    return Err(Error::InvalidArgument(format!(
      "eigsh: K is {n}x{}, M must be the same size",
      k.ncols()
    )));
  }
  if nev == 0 || nev >= n {
    return Err(Error::InvalidArgument(format!(
      "eigsh: nev = {nev}, needs 0 < nev < n = {n}"
    )));
  }
  // K - sigma M, factorized
  let mut a = ColMatrix::<f64>::new(n, n);
  k.for_each_entry(&mut |i, j, v| a.add_entry(i, j, v));
  match m {
    Some(m) => m.for_each_entry(&mut |i, j, v| a.add_entry(i, j, -sigma * v)),
    // MPI: the parts summed by MUMPS, the shift on one rank
    None if getfem_mpi::is_master() => (0..n).for_each(|i| a.add_entry(i, i, -sigma)),
    None => {}
  }
  let mut lu = Mumps::new();
  lu.analyze(&a, MatrixKind::Symmetric)?;
  lu.factorize(&a)?;
  // MPI: M the ranks' parts, M x summed, the same bits on every rank
  // (each rank runs the same ARPACK iterations)
  let mult_m = |x: &[f64], y: &mut [f64]| match m {
    Some(m) => {
      y.fill(0.0);
      m.for_each_entry(&mut |i, j, v| y[i] += v * x[j]);
      getfem_mpi::sum(y);
      getfem_mpi::bcast(y, 0);
    }
    None => y.copy_from_slice(x),
  };

  // scipy's defaults: ncv, 10 n iterations, machine precision
  let ncv = n.min((2 * nev + 1).max(20));
  let lworkl = ncv * (ncv + 8);
  let bmat = if m.is_some() { c"G" } else { c"I" };
  let which = c"LM";
  let (n_, nev_, ncv_) = (n as c_int, nev as c_int, ncv as c_int);
  let mut resid = vec![0.0; n];
  Rand::new(1).fill_random(&mut resid);
  let mut v = vec![0.0; n * ncv];
  let mut iparam = [0 as c_int; 11];
  iparam[0] = 1; // exact shifts
  iparam[2] = (10 * n).min(c_int::MAX as usize) as c_int;
  iparam[6] = 3; // shift-invert
  let mut ipntr = [0 as c_int; 14];
  let mut workd = vec![0.0; 3 * n];
  let mut workl = vec![0.0; lworkl];
  let (mut ido, mut info) = (0 as c_int, 1 as c_int); // info 1: resid given
  let mut tmp = vec![0.0; n];

  let _lock = ARPACK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
  loop {
    // Safety: the buffers have ARPACK's sizes (resid n, v n ncv, workd 3n,
    // workl lworkl, iparam 11, ipntr 14); the strings are nul terminated.
    unsafe {
      ffi::dsaupd_c(
        &mut ido,
        bmat.as_ptr(),
        n_,
        which.as_ptr(),
        nev_,
        0.0,
        resid.as_mut_ptr(),
        ncv_,
        v.as_mut_ptr(),
        n_,
        iparam.as_mut_ptr(),
        ipntr.as_mut_ptr(),
        workd.as_mut_ptr(),
        workl.as_mut_ptr(),
        lworkl as c_int,
        &mut info,
      );
    }
    // ARPACK's 1-based offsets into workd: x, y, and M x
    let at = |i: usize| ipntr[i].max(1) as usize - 1;
    let (x, y, bx) = (at(0), at(1), at(2));
    match ido {
      // y = (K - sigma M)^-1 M x
      -1 => {
        mult_m(&workd[x..x + n], &mut tmp);
        lu.solve(&mut tmp)?;
        workd[y..y + n].copy_from_slice(&tmp);
      }
      // y = (K - sigma M)^-1 (M x), M x given
      1 => {
        tmp.copy_from_slice(&workd[bx..bx + n]);
        lu.solve(&mut tmp)?;
        workd[y..y + n].copy_from_slice(&tmp);
      }
      // y = M x
      2 => {
        tmp.copy_from_slice(&workd[x..x + n]);
        mult_m(&tmp, &mut workd[y..y + n]);
      }
      99 => break,
      _ => return Err(arpack_error("dsaupd", info.min(-1))),
    }
  }
  if info != 0 {
    return Err(arpack_error("dsaupd", info));
  }

  let mut select = vec![0 as c_int; ncv]; // workspace with howmny "A"
  let mut d = vec![0.0; nev];
  let mut z = vec![0.0; n * nev];
  // Safety: as dsaupd, plus select ncv, d nev, z n nev
  unsafe {
    ffi::dseupd_c(
      1, // rvec
      c"A".as_ptr(),
      select.as_mut_ptr(),
      d.as_mut_ptr(),
      z.as_mut_ptr(),
      n_,
      sigma,
      bmat.as_ptr(),
      n_,
      which.as_ptr(),
      nev_,
      0.0,
      resid.as_mut_ptr(),
      ncv_,
      v.as_mut_ptr(),
      n_,
      iparam.as_mut_ptr(),
      ipntr.as_mut_ptr(),
      workd.as_mut_ptr(),
      workl.as_mut_ptr(),
      lworkl as c_int,
      &mut info,
    );
  }
  if info != 0 {
    return Err(arpack_error("dseupd", info));
  }
  let nconv = (iparam[4] as usize).min(nev);
  let mut order: Vec<usize> = (0..nconv).collect();
  order.sort_by(|&a, &b| d[a].total_cmp(&d[b]));
  let values = order.iter().map(|&i| d[i]).collect();
  let modes = order
    .iter()
    .map(|&i| z[i * n..(i + 1) * n].to_vec())
    .collect();
  Ok((values, modes))
}
