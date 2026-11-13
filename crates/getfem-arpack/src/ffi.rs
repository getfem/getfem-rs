//! ARPACK-ng's ISO C binding (arpack.h, a_int = int: the LP64 build)

use std::os::raw::{c_char, c_int};

unsafe extern "C" {
  pub fn dsaupd_c(
    ido: *mut c_int,
    bmat: *const c_char,
    n: c_int,
    which: *const c_char,
    nev: c_int,
    tol: f64,
    resid: *mut f64,
    ncv: c_int,
    v: *mut f64,
    ldv: c_int,
    iparam: *mut c_int,
    ipntr: *mut c_int,
    workd: *mut f64,
    workl: *mut f64,
    lworkl: c_int,
    info: *mut c_int,
  );

  pub fn dseupd_c(
    rvec: c_int,
    howmny: *const c_char,
    select: *const c_int,
    d: *mut f64,
    z: *mut f64,
    ldz: c_int,
    sigma: f64,
    bmat: *const c_char,
    n: c_int,
    which: *const c_char,
    nev: c_int,
    tol: f64,
    resid: *mut f64,
    ncv: c_int,
    v: *mut f64,
    ldv: c_int,
    iparam: *mut c_int,
    ipntr: *mut c_int,
    workd: *mut f64,
    workl: *mut f64,
    lworkl: c_int,
    info: *mut c_int,
  );
}
