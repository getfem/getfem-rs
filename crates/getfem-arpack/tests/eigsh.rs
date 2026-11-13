//! eigsh on 1D finite differences and P1 elements on (0, 1), Dirichlet:
//! lambda_k known in closed form

use getfem_arpack::eigsh;
use getfem_linalg::{ColMatrix, SparseMatrix};

/// n x n tridiagonal (b, a, b)
fn tridiag(n: usize, a: f64, b: f64) -> ColMatrix<f64> {
  let mut k = ColMatrix::new(n, n);
  for i in 0..n {
    k.add_entry(i, i, a);
    if i + 1 < n {
      k.add_entry(i, i + 1, b);
      k.add_entry(i + 1, i, b);
    }
  }
  k
}

/// max_k |K phi_k - lambda_k M phi_k| / |lambda_k|, max |phi_j' M phi_k - delta_jk|
fn check(k: &ColMatrix<f64>, m: &ColMatrix<f64>, vals: &[f64], modes: &[Vec<f64>]) -> (f64, f64) {
  let (mut res, mut orth) = (0.0f64, 0.0f64);
  for (l, p) in vals.iter().zip(modes) {
    let (kp, mp) = (k.mult(p), m.mult(p));
    let r: f64 = kp
      .iter()
      .zip(&mp)
      .map(|(a, b)| (a - l * b).powi(2))
      .sum::<f64>()
      .sqrt();
    res = res.max(r / l.abs());
  }
  for (i, p) in modes.iter().enumerate() {
    let mp = m.mult(p);
    for (j, q) in modes.iter().enumerate() {
      let d: f64 = q.iter().zip(&mp).map(|(a, b)| a * b).sum();
      orth = orth.max((d - if i == j { 1.0 } else { 0.0 }).abs());
    }
  }
  (res, orth)
}

#[test]
fn laplacian_standard() {
  // K = tridiag(-1, 2, -1): lambda_k = 2 - 2 cos(k pi / (n + 1))
  let n = 200;
  let k = tridiag(n, 2.0, -1.0);
  let (vals, modes) = eigsh(&k, None, 5, 0.0).unwrap();
  assert_eq!(vals.len(), 5);
  for (i, l) in vals.iter().enumerate() {
    let th = (i + 1) as f64 * std::f64::consts::PI / (n + 1) as f64;
    assert!((l - (2.0 - 2.0 * th.cos())).abs() < 1e-12, "{i}: {l}");
  }
  let (res, orth) = check(&k, &tridiag(n, 1.0, 0.0), &vals, &modes);
  assert!(res < 1e-9 && orth < 1e-10, "{res} {orth}");
}

#[test]
fn p1_generalized_shifted() {
  // P1 on n + 1 elements of size h: K = tridiag(-1, 2, -1) / h, M =
  // tridiag(1, 4, 1) h / 6: lambda_k = 6 / h^2 (1 - cos t) / (2 + cos t),
  // t = k pi h; the 4 nearest a shift between lambda_20 and lambda_21
  let n = 300;
  let h = 1.0 / (n + 1) as f64;
  let (k, m) = (
    tridiag(n, 2.0 / h, -1.0 / h),
    tridiag(n, 4.0 * h / 6.0, h / 6.0),
  );
  let lam = |k: usize| {
    let t = k as f64 * std::f64::consts::PI * h;
    6.0 / (h * h) * (1.0 - t.cos()) / (2.0 + t.cos())
  };
  let sigma = 0.5 * (lam(20) + lam(21)) + 1.0;
  let (vals, modes) = eigsh(&k, Some(&m as &dyn SparseMatrix), 4, sigma).unwrap();
  let want = [lam(19), lam(20), lam(21), lam(22)];
  for (l, w) in vals.iter().zip(want) {
    assert!((l - w).abs() < 1e-9 * w, "{l} {w}");
  }
  let (res, orth) = check(&k, &m, &vals, &modes);
  assert!(res < 1e-9 && orth < 1e-10, "{res} {orth}");
}

#[test]
fn bad_arguments() {
  let k = tridiag(10, 2.0, -1.0);
  assert!(eigsh(&k, None, 0, 0.0).is_err());
  assert!(eigsh(&k, None, 10, 0.0).is_err());
  let m = tridiag(9, 1.0, 0.0);
  assert!(eigsh(&k, Some(&m as &dyn SparseMatrix), 2, 0.0).is_err());
}
