use super::*;

fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
  let s = b.iter().fold(1.0f64, |m, x| m.max(x.abs()));
  a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol * s)
}

#[test]
fn codim_quantities() {
  // segment of length 5 in 2D along (3,4): K = (3,4), J = 5, B = K/25, size = 5
  let gt = pk(1, 1).unwrap();
  let g = Mat::from_rows(&[&[1.0, 4.0], &[2.0, 6.0]]);
  let pg = gt.at(&g, &[0.3]).unwrap();
  assert_eq!(pg.k.data(), &[3.0, 4.0]);
  assert!((pg.j - 5.0).abs() < 1e-15);
  assert!(close(pg.b.data(), &[3.0 / 25.0, 4.0 / 25.0], 1e-15));
  assert!((gt.element_size(&g).unwrap() - 5.0).abs() < 1e-14);
  // B^T K = I_P for a triangle in 3D
  let t = pk(2, 1).unwrap();
  let g = Mat::from_rows(&[&[0.0, 2.0, 0.5], &[0.0, 0.1, 1.5], &[1.0, 0.3, 0.2]]);
  let pg = t.at(&g, &[0.2, 0.2]).unwrap();
  let btk = pg.b.tmatmul(&pg.k);
  assert!(close(btk.data(), &[1.0, 0.0, 0.0, 1.0], 1e-14));
}

#[test]
fn scale_invariance() {
  // reference-space results identical under scaling 1e-3, 1, 1e3
  let gt = qk(2, 2).unwrap();
  let base: Vec<f64> = gt
    .geometric_nodes()
    .iter()
    .flat_map(|noderef| {
      [
        noderef[0] + 0.1 * noderef[1] * noderef[1],
        noderef[1] - 0.05 * noderef[0],
      ]
    })
    .collect();
  let xref = [0.3, 0.7];
  let g1 = Mat::from_data(&[2, 9], base.clone()).unwrap();
  let pg1 = gt.at(&g1, &xref).unwrap();
  let s1 = gt.element_size(&g1).unwrap();
  for sc in [1e-3, 1e3] {
    let g = Mat::from_data(&[2, 9], base.iter().map(|v| v * sc).collect()).unwrap();
    let pg = gt.at(&g, &xref).unwrap();
    assert!(close(
      pg.k.data(),
      &pg1.k.data().iter().map(|v| v * sc).collect::<Vec<_>>(),
      1e-14
    ));
    assert!(((pg.j / (sc * sc)) - pg1.j).abs() < 1e-14 * pg1.j);
    assert!((gt.element_size(&g).unwrap() / sc - s1).abs() < 1e-14 * s1);
  }
}

#[test]
fn errors() {
  assert!(geotrans_descriptor("GT_PK(2,0)").is_err());
  assert!(geotrans_descriptor("GT_FOO(2,1)").is_err());
  assert!(geotrans_descriptor("GT_PK(2)").is_err());
  let gt = pk(2, 1).unwrap();
  assert!(gt.at(&Mat::zeros(&[2, 2]), &[0.1, 0.1]).is_err());
  let flat = Mat::from_rows(&[&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0]]);
  assert!(matches!(gt.at(&flat, &[0.1, 0.1]), Err(Error::Singular)));
  assert!(Arc::ptr_eq(
    &geotrans_descriptor("GT_QK(2,1)").unwrap(),
    &geotrans_descriptor("GT_PRODUCT(GT_PK(1,1),GT_PK(1,1))").unwrap()
  ));
}

#[test]
fn apex_gradient_exact() {
  // limits along the axis toward the apex, computed with sympy
  let q = 0.25;
  let expect: [[f64; 3]; 14] = [
    [q, q, q],
    [0., 0., 0.],
    [-q, q, q],
    [0., 0., 0.],
    [0., 0., 0.],
    [0., 0., 0.],
    [q, -q, q],
    [0., 0., 0.],
    [-q, -q, q],
    [-1., -1., -1.],
    [1., -1., -1.],
    [-1., 1., -1.],
    [1., 1., -1.],
    [0., 0., 3.],
  ];
  let gt = pyramid(2).unwrap();
  let g = gt.grad(&[0.0, 0.0, 1.0]);
  for (i, e) in expect.iter().enumerate() {
    for j in 0..3 {
      assert!(
        (g[(i, j)] - e[j]).abs() < 1e-12,
        "N{i} d{j}: {} vs {}",
        g[(i, j)],
        e[j]
      );
    }
  }
  // values are exact at the apex too
  let v = gt.val(&[0.0, 0.0, 1.0]);
  assert!(
    v.iter()
      .enumerate()
      .all(|(i, &x)| (x - if i == 13 { 1.0 } else { 0.0 }).abs() < 1e-14),
    "{v:?}"
  );
}
