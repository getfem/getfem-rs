use getfem_xsmm::{Gemm, GemmShape};

/// c (+)= op(a) op(b) by the definition
fn naive(s: &GemmShape, a: &[f64], b: &[f64], c: &mut [f64]) {
  let at = |i: usize, l: usize| {
    if s.ta {
      a[l + s.lda * i]
    } else {
      a[i + s.lda * l]
    }
  };
  let bt = |l: usize, j: usize| {
    if s.tb {
      b[j + s.ldb * l]
    } else {
      b[l + s.ldb * j]
    }
  };
  for j in 0..s.n {
    for i in 0..s.m {
      let v: f64 = (0..s.k).map(|l| at(i, l) * bt(l, j)).sum();
      let o = &mut c[i + s.ldc * j];
      *o = if s.accumulate { *o + v } else { v };
    }
  }
}

#[test]
fn gemm_matches_the_definition() {
  // GetFEM-rs' shapes (m x 3 x 3 NN, the NT contractions) and small ones
  let shapes = [
    (90, 3, 3),
    (243, 3, 3),
    (10, 30, 45),
    (27, 81, 81),
    (1, 1, 1),
    (3, 2, 4),
    (7, 5, 9),
  ];
  for (m, n, k) in shapes {
    for (tb, accumulate, pad) in [
      (false, false, 0),
      (true, false, 0),
      (true, true, 0),
      (true, false, 3),
    ] {
      // leading dimensions padded by pad (strided sub-matrices)
      let (lda, ldb, ldc) = (m + pad, if tb { n } else { k } + pad, m + pad);
      let s = GemmShape {
        ta: false,
        tb,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        accumulate,
      };
      let Some(g) = Gemm::new(s) else {
        // libxsmm builds on x86_64 and aarch64; elsewhere GetFEM-rs uses
        // its own loops
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        panic!("{s:?}: no kernel");
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        return;
      };
      let a: Vec<f64> = (0..lda * k).map(|i| (0.3 * i as f64 + 0.1).sin()).collect();
      let b: Vec<f64> = (0..ldb * if tb { k } else { n })
        .map(|i| (0.7 * i as f64 + 0.2).cos())
        .collect();
      let c0: Vec<f64> = (0..ldc * n).map(|i| (0.1 * i as f64).sin()).collect();
      let (mut want, mut c) = (c0.clone(), c0);
      naive(&s, &a, &b, &mut want);
      g.call(&a, &b, &mut c);
      let d = c
        .iter()
        .zip(&want)
        .fold(0.0f64, |x, (p, q)| x.max((p - q).abs()));
      assert!(d < 1e-13 * k as f64, "{s:?}: {d}");
    }
  }
}
