//! Exact geometric predicates in 2D and 3D: the signs of the orientation
//! and in-sphere determinants. A floating-point evaluation with an error
//! bound decides almost always; otherwise the determinant is evaluated
//! exactly with floating-point expansions (non-overlapping sums of
//! doubles, Shewchuk's error-free two-sum and two-product).

/// A floating-point expansion: non-overlapping components, increasing
/// magnitude, no zeros.
#[derive(Clone, Debug, Default)]
struct Expansion(Vec<f64>);

fn two_sum(a: f64, b: f64) -> (f64, f64) {
  let s = a + b;
  let bv = s - a;
  (s, (a - (s - bv)) + (b - bv))
}

fn two_prod(a: f64, b: f64) -> (f64, f64) {
  let p = a * b;
  (p, a.mul_add(b, -p))
}

impl Expansion {
  fn of(a: f64) -> Self {
    Expansion(if a == 0.0 { vec![] } else { vec![a] })
  }

  /// a - b, exact
  fn diff(a: f64, b: f64) -> Self {
    let (s, e) = two_sum(a, -b);
    Expansion([e, s].into_iter().filter(|&v| v != 0.0).collect())
  }

  /// self + b (Shewchuk's fast expansion sum: the components merged by
  /// magnitude, then accumulated with two-sum)
  fn add(&self, b: &Expansion) -> Self {
    let (e, f) = (&self.0, &b.0);
    let mut g = Vec::with_capacity(e.len() + f.len());
    let (mut i, mut j) = (0, 0);
    while i < e.len() || j < f.len() {
      if j == f.len() || (i < e.len() && e[i].abs() < f[j].abs()) {
        g.push(e[i]);
        i += 1;
      } else {
        g.push(f[j]);
        j += 1;
      }
    }
    let mut h = Vec::with_capacity(g.len());
    let Some((&g0, rest)) = g.split_first() else {
      return Expansion(h);
    };
    let mut q = g0;
    for &gi in rest {
      let (s, err) = two_sum(q, gi);
      if err != 0.0 {
        h.push(err);
      }
      q = s;
    }
    if q != 0.0 {
      h.push(q);
    }
    Expansion(h)
  }

  fn neg(&self) -> Self {
    Expansion(self.0.iter().map(|v| -v).collect())
  }

  /// self b (Shewchuk's scale expansion)
  fn scale(&self, b: f64) -> Self {
    let mut h = Vec::with_capacity(2 * self.0.len());
    let Some((&e0, rest)) = self.0.split_first() else {
      return Expansion(h);
    };
    let (mut q, err) = two_prod(e0, b);
    if err != 0.0 {
      h.push(err);
    }
    for &ei in rest {
      let (p1, p0) = two_prod(ei, b);
      let (sum, err) = two_sum(q, p0);
      if err != 0.0 {
        h.push(err);
      }
      let (s, err) = two_sum(p1, sum);
      if err != 0.0 {
        h.push(err);
      }
      q = s;
    }
    if q != 0.0 {
      h.push(q);
    }
    Expansion(h)
  }

  fn mul(&self, b: &Expansion) -> Self {
    b.0
      .iter()
      .fold(Expansion::default(), |r, &y| r.add(&self.scale(y)))
  }

  /// the sign of the largest component
  fn sign(&self) -> i8 {
    match self.0.last() {
      None => 0,
      Some(&v) if v > 0.0 => 1,
      _ => -1,
    }
  }
}

/// Determinant by cofactors along the first row (n <= 4), in a ring.
fn det<T: Clone>(
  m: &[Vec<T>],
  mul: &dyn Fn(&T, &T) -> T,
  add: &dyn Fn(&T, &T) -> T,
  neg: &dyn Fn(&T) -> T,
) -> T {
  if m.len() == 1 {
    return m[0][0].clone();
  }
  let mut acc: Option<T> = None;
  for j in 0..m.len() {
    let minor: Vec<Vec<T>> = m[1..]
      .iter()
      .map(|r| {
        r.iter()
          .enumerate()
          .filter(|(k, _)| *k != j)
          .map(|(_, v)| v.clone())
          .collect()
      })
      .collect();
    let mut t = mul(&m[0][j], &det(&minor, mul, add, neg));
    if j % 2 == 1 {
      t = neg(&t);
    }
    acc = Some(match acc {
      None => t,
      Some(a) => add(&a, &t),
    });
  }
  acc.unwrap()
}

/// Determinant (or with abs, permanent of the absolute values) of the m x
/// m matrix a, rows from r, the columns of the mask (no allocation).
fn det_f(a: &[[f64; 4]; 4], m: usize, r: usize, cols: u8, abs: bool) -> f64 {
  if r == m - 1 {
    let c = cols.trailing_zeros() as usize;
    return if abs { a[r][c].abs() } else { a[r][c] };
  }
  let (mut s, mut k) = (0.0, 0);
  for c in 0..m {
    if cols & (1 << c) == 0 {
      continue;
    }
    let v = if abs { a[r][c].abs() } else { a[r][c] };
    let t = v * det_f(a, m, r + 1, cols & !(1 << c), abs);
    s += if k % 2 == 1 && !abs { -t } else { t };
    k += 1;
  }
  s
}

/// The sign of the determinant of the m x m rows, given in f64 and (on
/// demand) exactly.
fn sign_of(rows_f: &[[f64; 4]; 4], m: usize, rows_e: impl Fn() -> Vec<Vec<Expansion>>) -> i8 {
  let all = (1u8 << m) - 1;
  let d = det_f(rows_f, m, 0, all, false);
  // |error| < 128 eps permanent (a generous bound)
  let perm = det_f(rows_f, m, 0, all, true);
  if d.abs() > 128.0 * f64::EPSILON * perm {
    return if d > 0.0 { 1 } else { -1 };
  }
  det(&rows_e(), &|a, b| a.mul(b), &|a, b| a.add(b), &|a| a.neg()).sign()
}

/// The sign of det[p1 - p0, ..., pn - p0] for n + 1 points of dimension n
/// (n = 1 to 3): positive for a positively oriented simplex.
pub fn orient(p: &[&[f64]]) -> i8 {
  let n = p.len() - 1;
  let mut rows_f = [[0.0; 4]; 4];
  for i in 1..=n {
    for k in 0..n {
      rows_f[i - 1][k] = p[i][k] - p[0][k];
    }
  }
  sign_of(&rows_f, n, || {
    (1..=n)
      .map(|i| (0..n).map(|k| Expansion::diff(p[i][k], p[0][k])).collect())
      .collect()
  })
}

/// Whether q is inside (1), on (0) or outside (-1) the circumsphere of
/// the n + 1 points p (dimension n, 1 to 3), of either orientation; 0 for a
/// degenerate simplex.
pub fn in_sphere(p: &[&[f64]], q: &[f64]) -> i8 {
  match orient(p) {
    0 => 0,
    o => o * in_sphere_oriented(p, q),
  }
}

/// in_sphere for a positively oriented simplex p (not checked).
pub fn in_sphere_oriented(p: &[&[f64]], q: &[f64]) -> i8 {
  let n = p.len() - 1;
  // rows [p_i - q, |p_i - q|^2]
  let mut rows_f = [[0.0; 4]; 4];
  for i in 0..=n {
    let mut l = 0.0;
    for k in 0..n {
      let d = p[i][k] - q[k];
      rows_f[i][k] = d;
      l += d * d;
    }
    rows_f[i][n] = l;
  }
  let s = sign_of(&rows_f, n + 1, || {
    (0..=n)
      .map(|i| {
        let d: Vec<Expansion> = (0..n).map(|k| Expansion::diff(p[i][k], q[k])).collect();
        let l = d.iter().fold(Expansion::of(0.0), |a, v| a.add(&v.mul(v)));
        d.into_iter().chain([l]).collect()
      })
      .collect()
  });
  // the lifted determinant is negative inside for odd n with a positive
  // orientation (sign: (-1)^n)
  let parity = if n % 2 == 1 { -1 } else { 1 };
  s * parity
}
