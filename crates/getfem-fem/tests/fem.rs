//! Lagrange elements: the Lagrange property and names, the hierarchical
//! Q2 in 3D, the pyramid's default degree.

use getfem_fem::fem::fem_descriptor;

#[test]
fn lagrange_property_and_names() {
  // phi_i(node_j) = delta_ij for every continuous Lagrange element
  for name in [
    "FEM_PK(3,3)",
    "FEM_QK(3,2)",
    "FEM_PK_PRISM(3,2)",
    "FEM_Q2_INCOMPLETE(3)",
    "FEM_PRISM_INCOMPLETE_P2",
    "FEM_PYRAMID_QK(2)",
    "FEM_PYRAMID_Q2_INCOMPLETE",
  ] {
    let f = fem_descriptor(name).unwrap();
    for j in 0..f.nb_dof() {
      let v = f.base_value(f.node(j)).unwrap();
      for i in 0..f.nb_dof() {
        let e = if i == j { 1.0 } else { 0.0 };
        assert!(
          (v.data()[i] - e).abs() < 1e-12,
          "{name}: phi_{i}(node {j}) = {}",
          v.data()[i]
        );
      }
    }
  }
  assert_eq!(fem_descriptor("FEM_QK(2,2)").unwrap().name(), "FEM_QK(2,2)");
  assert_eq!(
    fem_descriptor("FEM_PRODUCT(FEM_PK(1,2),FEM_PK(1,2))")
      .unwrap()
      .name(),
    "FEM_QK(2,2)"
  );
  assert_eq!(
    fem_descriptor("FEM_PRISM_PK(3,1)").unwrap().name(),
    "FEM_PK_PRISM(3,1)"
  );
  // a 2D prism is the quadrilateral: the 2D elements, not the 1D ones
  for (prism, quad) in [
    ("FEM_PRISM_PK(2,2)", "FEM_QK(2,2)"),
    (
      "FEM_PRISM_PK_DISCONTINUOUS(2,1)",
      "FEM_QK_DISCONTINUOUS(2,1)",
    ),
    ("FEM_PRISM_PK_HIERARCHICAL(2,2)", "FEM_QK_HIERARCHICAL(2,2)"),
  ] {
    let (p, q) = (
      fem_descriptor(prism).unwrap(),
      fem_descriptor(quad).unwrap(),
    );
    assert_eq!(p.dim(), 2, "{prism}");
    assert_eq!(p.nb_dof(), q.nb_dof(), "{prism}");
    assert_eq!(p.name(), q.name(), "{prism}");
  }
  for s in [
    "FEM_PK(0,1)",
    "FEM_FOO(1)",
    "FEM_PK_DISCONTINUOUS(2,1,1.5)",
    "FEM_PYRAMID_QK(3)",
    "FEM_Q2_INCOMPLETE(4)",
  ] {
    assert!(fem_descriptor(s).is_err(), "{s}");
  }
}

#[test]
#[allow(clippy::needless_range_loop)]
fn qk_hierarchical_3d_spans_q2() {
  // GetFEM builds FEM_QK_HIERARCHICAL(3,k) on the prism; the port builds the
  // hexahedral element, which must span exactly the Q2 space
  let h = fem_descriptor("FEM_QK_HIERARCHICAL(3,2)").unwrap();
  let q = fem_descriptor("FEM_QK(3,2)").unwrap();
  assert_eq!(
    (h.nb_dof(), h.ref_convex().structure().nb_points()),
    (27, 8)
  );
  // M[i][j] = h_j(node_i of Q2): h_j = sum_i M[i][j] L_i (h_j in Q2)
  let m: Vec<Vec<f64>> = (0..27)
    .map(|i| h.base_value(q.node(i)).unwrap().data().to_vec())
    .collect();
  for xref in [[0.13, 0.52, 0.91], [0.77, 0.08, 0.35]] {
    let (hv, lv) = (h.base_value(&xref).unwrap(), q.base_value(&xref).unwrap());
    for j in 0..27 {
      let interp: f64 = (0..27).map(|i| m[i][j] * lv.data()[i]).sum();
      assert!((interp - hv.data()[j]).abs() < 1e-12, "h_{j} not in Q2");
    }
  }
  // and M is invertible: Gaussian elimination with partial pivoting
  let mut a = m.clone();
  for c in 0..27 {
    let p = (c..27)
      .max_by(|&x, &y| a[x][c].abs().total_cmp(&a[y][c].abs()))
      .unwrap();
    a.swap(c, p);
    assert!(a[c][c].abs() > 1e-8, "singular at column {c}");
    for r in c + 1..27 {
      let f = a[r][c] / a[c][c];
      for k in c..27 {
        a[r][k] -= f * a[c][k];
      }
    }
  }
}

#[test]
fn pyramid_degree_defaults_to_two() {
  // GetFEM pyramid_QK_fem, pyramid_disc_QK_fem: k = 2 without parameter
  for (a, b) in [
    ("FEM_PYRAMID_QK", "FEM_PYRAMID_QK(2)"),
    (
      "FEM_PYRAMID_QK_DISCONTINUOUS",
      "FEM_PYRAMID_QK_DISCONTINUOUS(2)",
    ),
  ] {
    let (fa, fb) = (fem_descriptor(a).unwrap(), fem_descriptor(b).unwrap());
    assert_eq!(fa.name(), fb.name());
  }
}
