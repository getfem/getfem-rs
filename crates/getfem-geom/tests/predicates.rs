//! The predicates against exact integer arithmetic: points on integer
//! grids (small ones for many degeneracies, large ones for the filter),
//! scaled by powers of two (exact in f64).

use getfem_geom::predicates::{in_sphere, orient};

/// determinant of an integer matrix (n <= 4)
fn det(m: &[Vec<i128>]) -> i128 {
  if m.len() == 1 {
    return m[0][0];
  }
  (0..m.len())
    .map(|j| {
      let minor: Vec<Vec<i128>> = m[1..]
        .iter()
        .map(|r| {
          r.iter()
            .enumerate()
            .filter(|(k, _)| *k != j)
            .map(|(_, v)| *v)
            .collect()
        })
        .collect();
      let t = m[0][j] * det(&minor);
      if j % 2 == 1 { -t } else { t }
    })
    .sum()
}

fn exact_orient(p: &[Vec<i128>]) -> i8 {
  let n = p.len() - 1;
  let m: Vec<Vec<i128>> = (1..=n)
    .map(|i| (0..n).map(|k| p[i][k] - p[0][k]).collect())
    .collect();
  det(&m).signum() as i8
}

/// q inside the circumsphere: |q - c|^2 < |p0 - c|^2, by the lifted
/// determinant with the orientation (independent of the predicates' code)
fn exact_in_sphere(p: &[Vec<i128>], q: &[i128]) -> i8 {
  let n = p.len() - 1;
  let o = exact_orient(p);
  // lifted orientation of (p_i, |p_i|^2) and (q, |q|^2): det of the
  // differences to q
  let m: Vec<Vec<i128>> = (0..=n)
    .map(|i| {
      let d: Vec<i128> = (0..n).map(|k| p[i][k] - q[k]).collect();
      let l = d.iter().map(|v| v * v).sum();
      d.into_iter().chain([l]).collect()
    })
    .collect();
  let s = det(&m).signum() as i8;
  // sign convention fixed by an explicit inside point below
  s * o
}

struct Lcg(u64);
impl Lcg {
  fn next(&mut self, range: i128) -> i128 {
    self.0 = self
      .0
      .wrapping_mul(6364136223846793005)
      .wrapping_add(1442695040888963407);
    ((self.0 >> 33) as i128) % range
  }
}

#[test]
fn convention() {
  // the unit simplex and a point inside its circumsphere, both
  // orientations, n = 1 to 3
  for n in 1..=3usize {
    let mut p: Vec<Vec<f64>> = vec![vec![0.0; n]];
    for k in 0..n {
      let mut e = vec![0.0; n];
      e[k] = 1.0;
      p.push(e);
    }
    let q = vec![0.25; n];
    let far = vec![5.0; n];
    let mut r: Vec<&[f64]> = p.iter().map(|v| v.as_slice()).collect();
    assert_eq!(orient(&r), 1, "n = {n}");
    assert_eq!(in_sphere(&r, &q), 1, "n = {n}");
    assert_eq!(in_sphere(&r, &far), -1, "n = {n}");
    // a vertex is on the sphere
    assert_eq!(in_sphere(&r, &p[1]), 0);
    r.swap(0, 1);
    if n > 1 {
      assert_eq!(orient(&r), -1);
    }
    assert_eq!(in_sphere(&r, &q), 1, "n = {n}, reversed");
  }
}

#[test]
fn against_exact_arithmetic() {
  let mut g = Lcg(7);
  for n in 1..=3usize {
    for (range, scale) in [
      (4, 1.0),
      (9, 0.125),
      (1 << 18, (2.0f64).powi(-30)),
      (1 << 18, 1024.0),
    ] {
      let (mut zeros, mut sign_inside) = (0, 0);
      for _ in 0..3000 {
        let pts: Vec<Vec<i128>> = (0..=n + 1)
          .map(|_| (0..n).map(|_| g.next(range)).collect())
          .collect();
        let f: Vec<Vec<f64>> = pts
          .iter()
          .map(|v| v.iter().map(|&x| x as f64 * scale).collect())
          .collect();
        let s: Vec<&[f64]> = f[..=n].iter().map(|v| v.as_slice()).collect();
        let o = exact_orient(&pts[..=n]);
        assert_eq!(orient(&s), o, "orient {pts:?}");
        if o == 0 {
          assert_eq!(in_sphere(&s, &f[n + 1]), 0);
          continue;
        }
        let e = exact_in_sphere(&pts[..=n], &pts[n + 1]);
        let got = in_sphere(&s, &f[n + 1]);
        if e == 0 {
          zeros += 1;
          assert_eq!(got, 0, "in_sphere {pts:?}");
        } else {
          // the reference's convention: one factor (-1)^n to calibrate
          sign_inside += 1;
          let parity = if n % 2 == 1 { -1 } else { 1 };
          assert_eq!(got, e * parity, "in_sphere {pts:?}");
        }
      }
      if range < 10 {
        assert!(zeros > 0, "no degenerate cases for n = {n}");
      }
      assert!(sign_inside > 0);
    }
  }
}
