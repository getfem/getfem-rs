/* libxsmm built header-only, one translation unit (build.rs), and the two
   functions GetFEM-rs calls (libxsmm's own are internal in this mode) */
#include <libxsmm_source.h>

/* the kernels use AVX registers (x86 from AVX on): they return without
   vzeroupper, which leaves the caller's SSE code slow (Intel), so the call
   clears the upper halves; set at dispatch, before any call */
static int avx;

/* the kernel of C = op(A) op(B) (beta0) or C += op(A) op(B), column-major
   doubles; NULL if libxsmm has none (CPU, system refusing JIT code) */
void *getfem_xsmm_dispatch(int ta, int tb, int m, int n, int k, int lda, int ldb, int ldc, int beta0) {
  libxsmm_gemm_shape s = libxsmm_create_gemm_shape(m, n, k, lda, ldb, ldc, LIBXSMM_DATATYPE_F64,
                                                   LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64,
                                                   LIBXSMM_DATATYPE_F64);
  libxsmm_bitfield f = (ta ? LIBXSMM_GEMM_FLAG_TRANS_A : 0) | (tb ? LIBXSMM_GEMM_FLAG_TRANS_B : 0) |
                       (beta0 ? LIBXSMM_GEMM_FLAG_BETA_0 : 0);
#if defined(__x86_64__)
  __atomic_store_n(&avx, libxsmm_get_target_archid() >= LIBXSMM_X86_AVX, __ATOMIC_RELAXED);
#endif
  return (void *)libxsmm_dispatch_gemm(s, f, LIBXSMM_GEMM_PREFETCH_NONE);
}

/* the arguments of a call, one per thread, only the matrices set: a zeroed
   one per call costs about 100 ns */
static __thread libxsmm_gemm_param param;

void getfem_xsmm_call(void *kernel, const double *a, const double *b, double *c) {
  param.a.primary = (void *)a;
  param.b.primary = (void *)b;
  param.c.primary = c;
  ((libxsmm_gemmfunction)kernel)(&param);
#if defined(__x86_64__)
  if (__atomic_load_n(&avx, __ATOMIC_RELAXED)) __asm__ volatile("vzeroupper");
#endif
}
