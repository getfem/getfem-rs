//! M4 exit: Poisson system solved, factorize once / solve many, singular
//! matrices reported, misuse reported.

use getfem_linalg::{Coo, Csr, Error, LinearSolver, MatrixKind};
use getfem_mumps::Mumps;

/// 5-point Laplacian on an m x m grid (Dirichlet), SPD.
fn laplacian(m: usize) -> Csr<f64> {
  let n = m * m;
  let mut c = Coo::new(n, n);
  for i in 0..m {
    for j in 0..m {
      let k = i * m + j;
      c.push(k, k, 4.0);
      if i > 0 {
        c.push(k, k - m, -1.0);
      }
      if i + 1 < m {
        c.push(k, k + m, -1.0);
      }
      if j > 0 {
        c.push(k, k - 1, -1.0);
      }
      if j + 1 < m {
        c.push(k, k + 1, -1.0);
      }
    }
  }
  c.to_csr().unwrap()
}

fn rel_err(a: &[f64], b: &[f64]) -> f64 {
  let d: f64 = a
    .iter()
    .zip(b)
    .map(|(x, y)| (x - y).powi(2))
    .sum::<f64>()
    .sqrt();
  d / b.iter().map(|y| y * y).sum::<f64>().sqrt()
}

#[test]
fn poisson_all_kinds() {
  let a = laplacian(40);
  let x: Vec<f64> = (0..a.nrows).map(|i| (i as f64 * 0.37).sin()).collect();
  let b = a.matvec(&x).unwrap();
  for kind in [
    MatrixKind::General,
    MatrixKind::Symmetric,
    MatrixKind::SymmetricPositiveDefinite,
  ] {
    let mut s = Mumps::new();
    let mut y = b.clone();
    s.solve_system(&a, kind, &mut y).unwrap();
    assert!(rel_err(&y, &x) < 1e-12, "{kind:?}: {}", rel_err(&y, &x));
    let info = s.info();
    assert_eq!((info.n, info.solver), (1600, "MUMPS"));
    assert!(info.nnz_factors.unwrap() > 0 && info.flops.unwrap() > 0.0);
  }
}

#[test]
fn factorize_once_solve_many_and_refactorize() {
  let a = laplacian(20);
  let mut s = Mumps::new();
  s.analyze(&a, MatrixKind::SymmetricPositiveDefinite)
    .unwrap();
  s.factorize(&a).unwrap();
  for r in 0..10 {
    let x: Vec<f64> = (0..a.nrows)
      .map(|i| ((i * (r + 1)) as f64 * 0.1).cos())
      .collect();
    let mut y = a.matvec(&x).unwrap();
    s.solve(&mut y).unwrap();
    assert!(rel_err(&y, &x) < 1e-12);
  }
  // same pattern, new values: A + 2 I
  let mut a2 = a.clone();
  for i in 0..a2.nrows {
    for p in a2.ptr[i]..a2.ptr[i + 1] {
      if a2.idx[p] == i {
        a2.val[p] += 2.0;
      }
    }
  }
  s.factorize(&a2).unwrap();
  let x = vec![1.0; a.nrows];
  let mut y = a2.matvec(&x).unwrap();
  s.solve(&mut y).unwrap();
  assert!(rel_err(&y, &x) < 1e-12);
}

#[test]
fn unsymmetric_system() {
  // convection-diffusion stencil
  let m = 30;
  let n = m * m;
  let mut c = Coo::new(n, n);
  for k in 0..n {
    c.push(k, k, 4.0);
    if k >= 1 {
      c.push(k, k - 1, -1.5);
    }
    if k + 1 < n {
      c.push(k, k + 1, -0.5);
    }
    if k >= m {
      c.push(k, k - m, -1.0);
    }
    if k + m < n {
      c.push(k, k + m, -1.0);
    }
  }
  let a = c.to_csr().unwrap();
  assert!(a.asymmetry() > 0.1);
  let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
  let mut y = a.matvec(&x).unwrap();
  Mumps::new()
    .solve_system(&a, MatrixKind::General, &mut y)
    .unwrap();
  assert!(rel_err(&y, &x) < 1e-12);
}

#[test]
fn singular_matrices_are_errors() {
  // numerically singular
  let mut c = Coo::new(2, 2);
  c.add_block(&[0, 1], &[0, 1], &[1.0, 1.0, 1.0, 1.0]);
  let a = c.to_csr().unwrap();
  let mut y = vec![1.0, 2.0];
  let e = Mumps::new()
    .solve_system(&a, MatrixKind::General, &mut y)
    .unwrap_err();
  assert!(matches!(e, Error::Singular(_)), "{e}");
  // structurally singular: an empty row
  let mut c = Coo::new(3, 3);
  c.push(0, 0, 1.0);
  c.push(2, 2, 1.0);
  c.push(0, 2, 1.0);
  let e = Mumps::new()
    .solve_system(
      &c.to_csr().unwrap(),
      MatrixKind::General,
      &mut [1.0, 1.0, 1.0],
    )
    .unwrap_err();
  assert!(matches!(e, Error::Singular(_)), "{e}");
  // SPD requested for an indefinite matrix
  let mut c = Coo::new(2, 2);
  c.add_block(&[0, 1], &[0, 1], &[1.0, 2.0, 2.0, 1.0]);
  let e = Mumps::new().solve_system(
    &c.to_csr().unwrap(),
    MatrixKind::SymmetricPositiveDefinite,
    &mut [1.0, 1.0],
  );
  assert!(e.is_err());
}

#[test]
fn misuse_is_reported() {
  let a = laplacian(3);
  let mut s = Mumps::new();
  assert!(s.solve(&mut [0.0; 9]).is_err());
  assert!(s.factorize(&a).is_err());
  s.analyze(&a, MatrixKind::General).unwrap();
  assert!(s.factorize(&laplacian(4)).is_err());
  s.factorize(&a).unwrap();
  assert!(s.solve(&mut [0.0; 3]).is_err());
  let mut c = Coo::new(2, 3);
  c.push(0, 0, 1.0);
  assert!(
    Mumps::new()
      .analyze(&c.to_csr().unwrap(), MatrixKind::General)
      .is_err()
  );
  s.release();
  assert!(s.solve(&mut [0.0; 9]).is_err());
}

#[test]
fn instances_in_threads() {
  let hs: Vec<_> = (0..4)
    .map(|t| {
      std::thread::spawn(move || {
        let a = laplacian(15 + t);
        let x = vec![1.0; a.nrows];
        let mut y = a.matvec(&x).unwrap();
        Mumps::new()
          .solve_system(&a, MatrixKind::Symmetric, &mut y)
          .unwrap();
        rel_err(&y, &x)
      })
    })
    .collect();
  for h in hs {
    assert!(h.join().unwrap() < 1e-12);
  }
}
