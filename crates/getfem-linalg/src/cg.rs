//! Preconditioned conjugate gradient (gmm::cg with gmm::diagonal_precond),
//! the matrix given by its product.

use crate::error::{Error, Result};
use crate::vect::vect_sp;

/// x with A x = b, A symmetric positive definite given by a(p) = A p,
/// preconditioned by the inverse of its diagonal d; converged when
/// |r| <= tol |b|, at most max_iter iterations. Returns x and the
/// iterations done.
pub fn cg(
  a: impl Fn(&[f64]) -> Vec<f64>,
  d: &[f64],
  b: &[f64],
  tol: f64,
  max_iter: usize,
) -> Result<(Vec<f64>, usize)> {
  let n = b.len();
  let mut x = vec![0.0; n];
  let nb = vect_sp(b, b).sqrt();
  if nb == 0.0 {
    return Ok((x, 0));
  }
  let mut r = b.to_vec();
  let mut z: Vec<f64> = r.iter().zip(d).map(|(r, d)| r / d).collect();
  let mut p = z.clone();
  let mut rz = vect_sp(&r, &z);
  for it in 0..max_iter {
    if vect_sp(&r, &r).sqrt() <= tol * nb {
      return Ok((x, it));
    }
    let ap = a(&p);
    let pap = vect_sp(&p, &ap);
    if pap <= 0.0 || !pap.is_finite() {
      return Err(Error::InvalidArgument(
        "cg: matrix not positive definite".into(),
      ));
    }
    let alpha = rz / pap;
    for i in 0..n {
      x[i] += alpha * p[i];
      r[i] -= alpha * ap[i];
      z[i] = r[i] / d[i];
    }
    let rz_new = vect_sp(&r, &z);
    let beta = rz_new / rz;
    rz = rz_new;
    for i in 0..n {
      p[i] = z[i] + beta * p[i];
    }
  }
  if vect_sp(&r, &r).sqrt() <= tol * nb {
    return Ok((x, max_iter));
  }
  Err(Error::InvalidArgument(format!(
    "cg: no convergence in {max_iter} iterations (residual {:.1e} of the right-hand side)",
    vect_sp(&r, &r).sqrt() / nb
  )))
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn tridiagonal() {
    // A = tridiag(-1, 4, -1), x = 1..n
    let n = 20;
    let a = |p: &[f64]| -> Vec<f64> {
      (0..n)
        .map(|i| {
          4.0 * p[i] - if i > 0 { p[i - 1] } else { 0.0 } - if i + 1 < n { p[i + 1] } else { 0.0 }
        })
        .collect()
    };
    let want: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let b = a(&want);
    let (x, it) = cg(a, &vec![4.0; n], &b, 1e-14, n).unwrap();
    assert!(it <= n);
    assert!(x.iter().zip(&want).all(|(x, y)| (x - y).abs() < 1e-12));
  }
}
