//! The small matrix products: libxsmm kernels (Gemm), no BLAS.

use std::ffi::c_void;
use std::sync::{Once, OnceLock};

#[cfg(xsmm)]
unsafe extern "C" {
  fn getfem_xsmm_dispatch(
    ta: i32,
    tb: i32,
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
    beta0: i32,
  ) -> *mut c_void;
  fn getfem_xsmm_call(kernel: *mut c_void, a: *const f64, b: *const f64, c: *mut f64);
}

/// Shape of C = op(A) op(B) (or C += op(A) op(B)), column-major, op(A)
/// m x k, op(B) k x n, A, B transposed if ta, tb
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmShape {
  pub ta: bool,
  pub tb: bool,
  pub m: usize,
  pub n: usize,
  pub k: usize,
  pub lda: usize,
  pub ldb: usize,
  pub ldc: usize,
  pub accumulate: bool,
}

/// A libxsmm kernel of one GemmShape (no BLAS gemm: libxsmm generates
/// kernels per shape at run time, faster on small products and without the
/// lock of OpenBLAS' buffers between threads). Made once per shape, called
/// per product.
#[derive(Clone, Copy, Debug)]
pub struct Gemm {
  shape: GemmShape,
  kernel: std::ptr::NonNull<c_void>,
}

// SAFETY: a kernel is immutable generated code, callable from any thread
unsafe impl Send for Gemm {}
unsafe impl Sync for Gemm {}

impl Gemm {
  /// The kernel of the shape, None without libxsmm (not built for this
  /// architecture) or without a kernel (CPU, system refusing generated
  /// code): then a warning, once
  pub fn new(shape: GemmShape) -> Option<Gemm> {
    let kernel = Self::dispatch(&shape);
    if kernel.is_none() {
      static WARN: Once = Once::new();
      WARN.call_once(|| {
        eprintln!(
          "GetFEM-rs warning: no libxsmm kernel on this architecture, CPU or \
           system; small matrix products by GetFEM-rs' own loops (slower)"
        );
      });
    }
    kernel.map(|kernel| Gemm { shape, kernel })
  }

  #[cfg(xsmm)]
  fn dispatch(s: &GemmShape) -> Option<std::ptr::NonNull<c_void>> {
    let dim = |x: usize| i32::try_from(x).ok();
    let (m, n, k) = (dim(s.m)?, dim(s.n)?, dim(s.k)?);
    let (lda, ldb, ldc) = (dim(s.lda)?, dim(s.ldb)?, dim(s.ldc)?);
    let (ta, tb, beta0) = (s.ta as i32, s.tb as i32, !s.accumulate as i32);
    // SAFETY: a plain call, the dimensions passed by value
    std::ptr::NonNull::new(unsafe { getfem_xsmm_dispatch(ta, tb, m, n, k, lda, ldb, ldc, beta0) })
  }

  #[cfg(not(xsmm))]
  fn dispatch(_s: &GemmShape) -> Option<std::ptr::NonNull<c_void>> {
    None
  }

  /// Whether libxsmm makes kernels here (warning once if not)
  pub fn available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
      let s = GemmShape {
        ta: false,
        tb: false,
        m: 2,
        n: 2,
        k: 2,
        lda: 2,
        ldb: 2,
        ldc: 2,
        accumulate: false,
      };
      Gemm::new(s).is_some()
    })
  }

  pub fn shape(&self) -> &GemmShape {
    &self.shape
  }

  /// c = op(a) op(b), or c += op(a) op(b)
  #[inline]
  pub fn call(&self, a: &[f64], b: &[f64], c: &mut [f64]) {
    let s = &self.shape;
    // the extents read and written, as BLAS
    let (ra, ca) = if s.ta { (s.k, s.m) } else { (s.m, s.k) };
    let (rb, cb) = if s.tb { (s.n, s.k) } else { (s.k, s.n) };
    assert!(a.len() + s.lda >= s.lda * ca + ra && b.len() + s.ldb >= s.ldb * cb + rb);
    assert!(c.len() + s.ldc >= s.ldc * s.n + s.m);
    #[cfg(xsmm)]
    // SAFETY: the extents above are within the slices; the kernel reads a,
    // b and writes c only there
    unsafe {
      getfem_xsmm_call(self.kernel.as_ptr(), a.as_ptr(), b.as_ptr(), c.as_mut_ptr());
    }
  }
}
