use getfem_linalg::Coo;

#[test]
fn coo_to_csr_sums_duplicates_and_sorts() {
  let mut c = Coo::new(3, 4);
  for (i, j, v) in [
    (2, 3, 1.0),
    (0, 1, 2.0),
    (2, 0, 3.0),
    (0, 1, 0.5),
    (1, 2, 0.0),
    (2, 3, -1.0),
  ] {
    c.push(i, j, v);
  }
  let a = c.to_csr().unwrap();
  assert_eq!(a.ptr, vec![0, 1, 2, 4]);
  assert_eq!(a.idx, vec![1, 2, 0, 3]);
  assert_eq!(a.val, vec![2.5, 0.0, 3.0, 0.0]);
  assert_eq!(a.get(0, 1), 2.5);
  assert_eq!(a.get(1, 1), 0.0);
  assert_eq!(
    a.matvec(&[1.0, 2.0, 3.0, 4.0]).unwrap(),
    vec![5.0, 0.0, 3.0]
  );
  assert!(a.matvec(&[1.0]).is_err());
  let t = a.transpose();
  assert_eq!(
    (t.nrows, t.ncols, t.get(1, 0), t.get(0, 2)),
    (4, 3, 2.5, 3.0)
  );
  let mut bad = Coo::new(1, 1);
  bad.push(0, 5, 1.0);
  assert!(bad.to_csr().is_err());
}

#[test]
fn blocks_and_symmetry() {
  let mut c = Coo::new(3, 3);
  c.add_block(&[0, 2], &[0, 2], &[2.0, -1.0, -1.0, 2.0]);
  c.add_block(&[1, 2], &[1, 2], &[1.0, 0.5, 0.5, 1.0]);
  let a = c.to_csr().unwrap();
  assert_eq!(
    a.to_dense(),
    vec![
      vec![2.0, 0.0, -1.0],
      vec![0.0, 1.0, 0.5],
      vec![-1.0, 0.5, 3.0]
    ]
  );
  assert_eq!(a.asymmetry(), 0.0);
}

#[test]
fn coo_to_csr_sums_in_entry_order() {
  // pseudo-random entries with many duplicates; reference: sums in entry
  // order per (i, j)
  let (n, m) = (7, 5);
  let mut c = Coo::new(n, m);
  let mut refs = std::collections::BTreeMap::new();
  let mut s: u64 = 1;
  for _ in 0..500 {
    s = s
      .wrapping_mul(6364136223846793005)
      .wrapping_add(1442695040888963407);
    let (i, j) = ((s >> 33) as usize % n, (s >> 45) as usize % m);
    let v = ((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 1e3;
    c.push(i, j, v);
    *refs.entry((i, j)).or_insert(-0.0) += v;
  }
  let a = c.to_csr().unwrap();
  let got: Vec<_> = (0..n)
    .flat_map(|i| {
      a.row(i)
        .map(move |(j, v)| ((i, j), v.to_bits()))
        .collect::<Vec<_>>()
    })
    .collect();
  let want: Vec<_> = refs.into_iter().map(|(k, v)| (k, v.to_bits())).collect();
  assert_eq!(got, want);
}

#[test]
fn csr_mul_matches_dense() {
  // A (3 x 4) B (4 x 2) against the dense product; sizes checked
  let a_d = [
    [1.0, 0.0, 2.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.5, -1.0, 0.0, 3.0],
  ];
  let b_d = [[0.0, 1.0], [2.0, 0.0], [0.0, 0.0], [1.0, -1.0]];
  let csr = |d: &[&[f64]]| {
    let mut c = Coo::new(d.len(), d[0].len());
    for (i, r) in d.iter().enumerate() {
      for (j, &v) in r.iter().enumerate() {
        if v != 0.0 {
          c.push(i, j, v);
        }
      }
    }
    c.to_csr().unwrap()
  };
  let a = csr(&a_d.iter().map(|r| &r[..]).collect::<Vec<_>>());
  let b = csr(&b_d.iter().map(|r| &r[..]).collect::<Vec<_>>());
  let c = a.mul(&b).unwrap();
  assert_eq!((c.nrows, c.ncols), (3, 2));
  for (i, ai) in a_d.iter().enumerate() {
    for j in 0..2 {
      let want: f64 = ai.iter().zip(&b_d).map(|(x, bk)| x * bk[j]).sum();
      assert_eq!(c.get(i, j), want, "({i}, {j})");
    }
  }
  assert!(c.ptr.windows(2).all(|w| w[0] <= w[1]));
  assert!(b.mul(&b).is_err());
}
