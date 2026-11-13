//! gmm's dense QR algorithm (gmm_dense_qr.h, gmm_dense_Householder.h): the
//! real Schur form A = Q S Q^T by Hessenberg reduction and Wilkinson double
//! shift QR steps, without LAPACK. Matrices n x n, column-major.

use std::ops::Range;

use crate::vect::{vect_norm2, vect_norm2_sqr};
use crate::{Error, Result};

/// gmm::house_vector: v scaled to v[0] = 1, the reflector I - 2 v v^T / v^T v
/// mapping the input to a multiple of e1
fn house_vector(v: &mut [f64]) {
  let (mu, a0) = (vect_norm2(v), v[0].abs());
  if mu != 0.0 {
    let s = if a0 == 0.0 {
      1.0 / mu
    } else {
      (a0 / v[0]) / (a0 + mu)
    };
    v.iter_mut().for_each(|x| *x *= s);
  }
  if !v[v.len() - 1].is_finite() {
    v.fill(0.0);
  }
  v[0] = 1.0;
}

/// gmm::row_house_update: rows r of a (leading dimension n), columns c,
/// multiplied to the left by the reflector of v; w: c.len() temporary
fn row_house_update(
  a: &mut [f64],
  n: usize,
  r: Range<usize>,
  c: Range<usize>,
  v: &[f64],
  w: &mut [f64],
) {
  let f = -2.0 / vect_norm2_sqr(v);
  for (wj, j) in w.iter_mut().zip(c.clone()) {
    *wj = f
      * r
        .clone()
        .zip(v)
        .map(|(i, vi)| a[i + n * j] * vi)
        .sum::<f64>();
  }
  for (wj, j) in w.iter().zip(c) {
    for (i, vi) in r.clone().zip(v) {
      a[i + n * j] += vi * wj;
    }
  }
}

/// gmm::col_house_update: rows r, columns c of a, multiplied to the right
/// by the reflector of v; w: r.len() temporary
fn col_house_update(
  a: &mut [f64],
  n: usize,
  r: Range<usize>,
  c: Range<usize>,
  v: &[f64],
  w: &mut [f64],
) {
  let f = -2.0 / vect_norm2_sqr(v);
  for (wi, i) in w.iter_mut().zip(r.clone()) {
    *wi = f
      * c
        .clone()
        .zip(v)
        .map(|(j, vj)| a[i + n * j] * vj)
        .sum::<f64>();
  }
  for (j, vj) in c.zip(v) {
    for (wi, i) in w.iter().zip(r.clone()) {
      a[i + n * j] += wi * vj;
    }
  }
}

/// gmm::Hessenberg_reduction, with Q
fn hessenberg_reduction(h: &mut [f64], q: &mut [f64], n: usize, v: &mut [f64], w: &mut [f64]) {
  q[..n * n].fill(0.0);
  (0..n).for_each(|i| q[i * (n + 1)] = 1.0);
  for k in 1..n.saturating_sub(1) {
    let v = &mut v[..n - k];
    v.copy_from_slice(&h[k + n * (k - 1)..n + n * (k - 1)]);
    house_vector(v);
    row_house_update(h, n, k..n, k - 1..n, v, w);
    col_house_update(h, n, 0..n, k..n, v, w);
    col_house_update(q, n, 0..n, k..n, v, w);
  }
}

/// gmm::qr_stop_criterion: negligible subdiagonal entries set to zero; q:
/// the size of the converged trailing part, p: the start of the active
/// block
fn qr_stop_criterion(h: &mut [f64], n: usize, p: &mut usize, q: &mut usize, tol: f64) {
  if n <= 2 {
    (*q, *p) = (n, 0);
    return;
  }
  let at = |i: usize, j: usize| i + n * j;
  for i in 1..n - *q {
    let s = h[at(i, i - 1)].abs();
    if s < (h[at(i, i)].abs() + h[at(i - 1, i - 1)].abs()) * tol || s < 2.0 * f64::MIN_POSITIVE {
      h[at(i, i - 1)] = 0.0;
    }
  }
  while (*q < n - 1 && h[at(n - 1 - *q, n - 2 - *q)] == 0.0)
    || (*q < n - 2 && h[at(n - 2 - *q, n - 3 - *q)] == 0.0)
  {
    *q += 1;
  }
  if *q >= n - 2 {
    *q = n;
  }
  *p = (n - *q).saturating_sub(2);
  while *p > 0 && h[at(*p, *p - 1)] != 0.0 {
    *p -= 1;
  }
}

/// gmm::Wilkinson_double_shift_qr_step on the active block b of h (from
/// LAPACK); as LAPACK's Schur form (unlike gmm), the reflectors also update
/// h right of and above the block
#[allow(clippy::too_many_arguments)]
fn wilkinson_double_shift_qr_step(
  h: &mut [f64],
  q: &mut [f64],
  n: usize,
  b: Range<usize>,
  tol: f64,
  exc: bool,
  w: &mut [f64],
) {
  let (o, nb) = (b.start, b.len());
  let at = |i: usize, j: usize| o + i + n * (o + j);
  let (h33, h44, h43h34) = if exc {
    // exceptional shift
    let s = h[at(nb - 1, nb - 2)].abs() + h[at(nb - 2, nb - 3)].abs();
    (0.75 * s, 0.75 * s, -0.4375 * s * s)
  } else {
    (
      h[at(nb - 2, nb - 2)],
      h[at(nb - 1, nb - 1)],
      h[at(nb - 1, nb - 2)] * h[at(nb - 2, nb - 1)],
    )
  };
  // start at row m if this makes h(m-1, m-2) negligible
  let mut m = nb - 2;
  let mut v: [f64; 3];
  loop {
    let (h11, h22) = (h[at(m - 1, m - 1)], h[at(m, m)]);
    let (h21, h12) = (h[at(m, m - 1)], h[at(m - 1, m)]);
    let (h44s, h33s) = (h44 - h11, h33 - h11);
    v = [
      (h33s * h44s - h43h34) / h21 + h12,
      h22 - h11 - h33s - h44s,
      h[at(m + 1, m)],
    ];
    let s = v[0].abs() + v[1].abs() + v[2].abs();
    v.iter_mut().for_each(|x| *x /= s);
    if m == 1 {
      break;
    }
    let (h00, h10) = (h[at(m - 2, m - 2)], h[at(m - 1, m - 2)]);
    let tst1 = v[0].abs() * (h00.abs() + h11.abs() + h22.abs());
    if h10.abs() * (v[1].abs() + v[2].abs()) <= tol * tst1 {
      break;
    }
    m -= 1;
  }
  // double shift QR step
  for k in m - 1..nb - 2 {
    house_vector(&mut v);
    let r = (k + 4).min(nb);
    row_house_update(h, n, o + k..o + k + 3, o + k.saturating_sub(1)..n, &v, w);
    col_house_update(h, n, 0..o + r, o + k..o + k + 3, &v, w);
    if k > m - 1 {
      h[at(k + 1, k - 1)] = 0.0;
      if k < nb - 3 {
        h[at(k + 2, k - 1)] = 0.0;
      }
    }
    col_house_update(q, n, 0..n, o + k..o + k + 3, &v, w);
    v[0] = h[at(k + 1, k)];
    v[1] = h[at(k + 2, k)];
    if k < nb - 3 {
      v[2] = h[at(k + 3, k)];
    }
  }
  let mut v = [v[0], v[1]];
  house_vector(&mut v);
  row_house_update(h, n, o + nb - 2..o + nb, o + nb - 3..n, &v, w);
  col_house_update(h, n, 0..o + nb, o + nb - 2..o + nb, &v, w);
  col_house_update(q, n, 0..n, o + nb - 2..o + nb, &v, w);
}

/// gmm::block2x2_reduction (real): the 2x2 diagonal blocks of real
/// eigenvalues reduced to triangular, the others left
fn block2x2_reduction(h: &mut [f64], q: &mut [f64], n: usize, tol: f64, w: &mut [f64]) {
  let at = |i: usize, j: usize| i + n * j;
  let tol = 2.0 * tol;
  let mut i = 0;
  while i + 1 < n {
    let tol_i = (h[at(i, i)].abs() + h[at(i + 1, i + 1)].abs()) * tol;
    if h[at(i + 1, i)].abs() > tol_i {
      // 2x2 block
      let tr = (h[at(i + 1, i + 1)] - h[at(i, i)]) / 2.0;
      let delta = tr * tr + h[at(i, i + 1)] * h[at(i + 1, i)];
      if delta >= 0.0 {
        // gmm's theta = (tr - sqrt(delta)) / h21 and v1 = theta (1 -
        // sqrt(a^2 + 1) / a), rewritten without cancellation
        let sd = delta.sqrt();
        let theta = if tr > 0.0 {
          -h[at(i, i + 1)] / (tr + sd)
        } else {
          (tr - sd) / h[at(i + 1, i)]
        };
        let a = theta.abs();
        let v1 = if a == 0.0 {
          -1.0
        } else {
          -theta.signum() / (a + (a * a + 1.0).sqrt())
        };
        let v = [1.0, v1];
        row_house_update(h, n, i..i + 2, i..n, &v, w);
        col_house_update(h, n, 0..i + 2, i..i + 2, &v, w);
        col_house_update(q, n, 0..n, i..i + 2, &v, w);
        // zero up to rounding
        h[at(i + 1, i)] = 0.0;
      }
      i += 1;
    }
    i += 1;
  }
}

/// gmm::schur by gmm::implicit_qr_algorithm: A = Q S Q^T, S upper
/// quasi-triangular (2x2 blocks: complex eigenvalue pairs), Q orthogonal;
/// s: A on entry, S on exit. No allocation up to n = 8.
pub fn schur(s: &mut [f64], q: &mut [f64], n: usize) -> Result<()> {
  assert!(s.len() >= n * n && q.len() >= n * n);
  if n == 0 {
    return Ok(());
  }
  let (mut st, mut hp) = ([0.0; 16], vec![]);
  let vw: &mut [f64] = if n <= 8 {
    &mut st
  } else {
    hp.resize(2 * n, 0.0);
    &mut hp
  };
  let (v, w) = vw.split_at_mut(vw.len() / 2);
  let tol = 3.0 * f64::EPSILON;
  hessenberg_reduction(s, q, n, v, w);
  let (mut p, mut qc) = (0, 0);
  qr_stop_criterion(s, n, &mut p, &mut qc, tol);
  let (mut ite, mut its) = (0, 0);
  while qc < n {
    wilkinson_double_shift_qr_step(s, q, n, p..n - qc, tol, its == 10 || its == 20, w);
    let q_old = qc;
    qr_stop_criterion(s, n, &mut p, &mut qc, 2.0 * tol);
    if qc != q_old {
      its = 0;
    }
    its += 1;
    ite += 1;
    if ite >= n * 100 {
      return Err(Error::Solver {
        solver: "QR algorithm",
        code: ite as i64,
        msg: "no convergence".into(),
      });
    }
  }
  block2x2_reduction(s, q, n, tol, w);
  // below the subdiagonal: zero up to rounding
  for j in 0..n {
    for i in j + 2..n {
      s[i + n * j] = 0.0;
    }
  }
  Ok(())
}

#[cfg(test)]
mod tests;
