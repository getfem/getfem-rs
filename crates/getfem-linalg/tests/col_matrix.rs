//! gmm col_matrix<rsvector> and GetFEM's add_elem_matrix against a dense
//! reference.

use getfem_linalg::ColMatrix;

struct Lcg(u64);

impl Lcg {
  fn next(&mut self) -> u64 {
    self.0 = self
      .0
      .wrapping_mul(6364136223846793005)
      .wrapping_add(1442695040888963407);
    self.0 >> 33
  }
  fn f(&mut self) -> f64 {
    self.next() as f64 / (1u64 << 31) as f64 - 0.5
  }
}

/// k distinct DOFs out of n, some None
fn dofs(r: &mut Lcg, n: usize, k: usize) -> Vec<Option<usize>> {
  let mut d: Vec<usize> = vec![];
  while d.len() < k {
    let x = r.next() as usize % n;
    if !d.contains(&x) {
      d.push(x);
    }
  }
  d.into_iter()
    .map(|x| (!r.next().is_multiple_of(5)).then_some(x))
    .collect()
}

#[test]
fn elem_matrices_as_dense() {
  let (nr, nc) = (17, 13);
  let mut r = Lcg(7);
  let mut a = ColMatrix::new(nr, nc);
  let mut dense = vec![vec![0.0; nc]; nr];
  let mut sort = vec![];
  for _ in 0..40 {
    let (d1, d2) = (dofs(&mut r, nr, 6), dofs(&mut r, nc, 4));
    let mut e: Vec<f64> = (0..24).map(|_| r.f()).collect();
    // entries dropped by the threshold
    e[3] = 1e-20;
    e[10] = 0.0;
    let thr = 1e-14 * e.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    a.add_elem_matrix(&d1, &d2, &e, thr, 2, &mut sort);
    for (t2, j) in d2.iter().enumerate() {
      for (t1, i) in d1.iter().enumerate() {
        let v = e[t1 + 6 * t2];
        if let (Some(i), Some(j)) = (i, j)
          && v.abs() > thr
        {
          dense[*i][*j] += v;
        }
      }
    }
  }
  for c in &a.cols {
    assert!(c.windows(2).all(|w| w[0].0 < w[1].0), "sorted rows");
  }
  let k = a.to_csr();
  for (i, row) in dense.iter().enumerate() {
    for (j, v) in row.iter().enumerate() {
      assert!((a.get(i, j) - v).abs() < 1e-14);
      assert!((k.get(i, j) - v).abs() < 1e-14);
    }
  }
  assert_eq!(k.nnz(), a.nnz());
  // no stored entry is below the threshold of its element
  assert!(
    a.cols
      .iter()
      .flatten()
      .all(|e| e.1 != 0.0 && e.1.abs() > 1e-19)
  );
  // A + A, A x
  let mut b = a.clone();
  b.add(&a);
  let x: Vec<f64> = (0..nc).map(|j| j as f64 - 3.0).collect();
  let (y, yb) = (a.mult(&x), b.mult(&x));
  for i in 0..nr {
    let yd: f64 = (0..nc).map(|j| dense[i][j] * x[j]).sum();
    assert!((y[i] - yd).abs() < 1e-12 && (yb[i] - 2.0 * yd).abs() < 1e-12);
  }
  a.add_entry(0, 0, 1.0);
  assert!((a.get(0, 0) - dense[0][0] - 1.0).abs() < 1e-14);
}

#[test]
fn add_to_column_merges() {
  let mut k = ColMatrix::<f64>::new(5, 2);
  k.add_to_column(1, &[(1, 2.0), (3, 1.0)]);
  k.add_to_column(1, &[(0, 1.0), (3, 0.5), (4, -1.0)]);
  assert_eq!(k.cols[1], vec![(0, 1.0), (1, 2.0), (3, 1.5), (4, -1.0)]);
  assert!(k.cols[0].is_empty());
}

#[test]
fn add_to_rsvector_as_dense() {
  let mut r = Lcg(7);
  for _ in 0..200 {
    let n = 1 + (r.next() % 30) as usize;
    let sparse = |r: &mut Lcg| -> Vec<(usize, f64)> {
      (0..n)
        .filter_map(|i| {
          if r.next().is_multiple_of(3) {
            Some((i, r.f()))
          } else {
            None
          }
        })
        .collect()
    };
    let (mut a, b) = (sparse(&mut r), sparse(&mut r));
    let mut dense = vec![0.0; n];
    a.iter().chain(&b).for_each(|&(i, v)| dense[i] += v);
    getfem_linalg::add_to_rsvector(&mut a, &b);
    assert!(a.windows(2).all(|w| w[0].0 < w[1].0));
    let mut d = vec![0.0; n];
    a.iter().for_each(|&(i, v)| d[i] = v);
    assert!(d.iter().zip(&dense).all(|(x, y)| (x - y).abs() < 1e-14));
  }
}
