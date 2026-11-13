use super::schur;
use crate::random::Rand;

/// Q S Q^T = A, Q^T Q = I (relative to |A|), S quasi-triangular with 2x2
/// blocks of complex pairs only; returns the number of 2x2 blocks
fn check(a: &[f64], n: usize) -> usize {
  let (mut s, mut q) = (a.to_vec(), vec![0.0; n * n]);
  schur(&mut s, &mut q, n).unwrap();
  let at = |m: &[f64], i: usize, j: usize| m[i + n * j];
  let na = a.iter().fold(1.0f64, |m, x| m.max(x.abs()));
  let mut err: f64 = 0.0;
  for i in 0..n {
    for j in 0..n {
      let qsq: f64 = (0..n)
        .flat_map(|k| (0..n).map(move |l| (k, l)))
        .map(|(k, l)| at(&q, i, k) * at(&s, k, l) * at(&q, j, l))
        .sum();
      err = err.max((qsq - at(a, i, j)).abs() / na);
      let qtq: f64 = (0..n).map(|k| at(&q, k, i) * at(&q, k, j)).sum();
      err = err.max((qtq - if i == j { 1.0 } else { 0.0 }).abs());
      if i > j + 1 {
        assert_eq!(at(&s, i, j), 0.0);
      }
    }
  }
  assert!(err < 1e-14 * n as f64, "n {n}: {err}");
  let mut blocks = 0;
  for i in 0..n.saturating_sub(1) {
    if at(&s, i + 1, i) != 0.0 {
      assert!(i == 0 || at(&s, i, i - 1) == 0.0, "adjacent 2x2 blocks");
      let tr = (at(&s, i + 1, i + 1) - at(&s, i, i)) / 2.0;
      let delta = tr * tr + at(&s, i, i + 1) * at(&s, i + 1, i);
      assert!(delta < 0.0, "2x2 block of real eigenvalues");
      blocks += 1;
    }
  }
  blocks
}

#[test]
fn schur_reconstructs() {
  // a rotation part (complex pair) and a stretch, column-major
  let a = [2.0, 1.0, 0.0, -3.0, 1.5, 0.2, 0.5, -0.4, 3.0];
  assert_eq!(check(&a, 3), 1);
}

#[test]
fn schur_special() {
  let i3 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
  assert_eq!(check(&i3, 3), 0);
  assert_eq!(check(&[0.0; 9], 3), 0);
  // Jordan block
  assert_eq!(check(&[2.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0, 2.0], 3), 0);
  // rotation by 90 degrees about z
  assert_eq!(check(&[0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0], 3), 1);
  // 2x2: real, complex; 1x1
  assert_eq!(check(&[1.0, 2.0, 3.0, 4.0], 2), 0);
  assert_eq!(check(&[1.0, -2.0, 3.0, 4.0], 2), 1);
  assert_eq!(check(&[5.0], 1), 0);
  // symmetric, repeated eigenvalue: diag(1, 1, 4) rotated
  let b = [2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0];
  assert_eq!(check(&b, 3), 0);
}

#[test]
fn schur_random() {
  let mut r = Rand::new(7);
  for n in 1..=12 {
    for _ in 0..200 {
      let mut a = vec![0.0; n * n];
      r.fill_random(&mut a);
      check(&a, n);
      // symmetric
      for j in 0..n {
        for i in 0..j {
          a[i + n * j] = a[j + n * i];
        }
      }
      assert_eq!(check(&a, n), 0);
    }
  }
}
