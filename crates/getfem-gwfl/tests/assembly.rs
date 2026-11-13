//! Assembly through GWFL: integrals with known values.

use getfem_fem::mesh_fem::MeshFem;
use getfem_fem::mesh_im::MeshIm;
use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::mesh::Mesh;
use getfem_geom::region::outer_faces_of_mesh;
use getfem_geom::regular_mesh::regular_mesh;
use getfem_gwfl::workspace::Workspace;

fn unit_square(gt: &str, n: usize) -> (Mesh, MeshIm) {
  let m = regular_mesh(
    &geotrans_descriptor(gt).unwrap(),
    &[n, n],
    &[0.0, 0.0],
    &[1.0, 1.0],
  )
  .unwrap();
  let mim = MeshIm::new(&m).with_integration_method(4).unwrap();
  (m, mim)
}

#[test]
fn integrals() {
  for gt in ["GT_PK(2,1)", "GT_QK(2,1)", "GT_QK(2,2)"] {
    let (m, mim) = unit_square(gt, 3);
    let mut ws = Workspace::new();
    let mut s = |e: &str, r| {
      ws.clear_expressions();
      ws.add_expression(e, &mim, r).unwrap();
      ws.assembly(0, false).unwrap();
      ws.assembled_potential()
    };
    let bnd = outer_faces_of_mesh(&m);
    assert!((s("1", None) - 1.0).abs() < 1e-13, "{gt}");
    assert!((s("X(1)*X(2)", None) - 0.25).abs() < 1e-13, "{gt}");
    assert!((s("Norm_sqr(X)", None) - 2.0 / 3.0).abs() < 1e-13, "{gt}");
    assert!((s("1", Some(&bnd)) - 4.0).abs() < 1e-13, "{gt}");
    assert!((s("X.Normal", Some(&bnd)) - 2.0).abs() < 1e-13, "{gt}");
    assert!(
      (s("element_size*0+[1,2;3,4](2,1)", None) - 3.0).abs() < 1e-13,
      "{gt}"
    );
  }
}

#[test]
fn laplacian_kernel() {
  let (m, mim) = unit_square("GT_PK(2,1)", 4);
  let mf = MeshFem::new(&m, 1)
    .with_classical_finite_element(2)
    .unwrap();
  let mut ws = Workspace::new();
  ws.add_fem_variable("u", &mf).unwrap();
  let k = {
    ws.clear_expressions();
    ws.add_expression("Grad_Test_u.Grad_Test2_u", &mim, None)
      .unwrap();
    ws.assembly(2, false).unwrap();
    ws.assembled_matrix()
  }
  .to_csr();
  // constants are in the kernel; symmetric
  let ones = vec![1.0; ws.nb_dof()];
  assert!(k.matvec(&ones).unwrap().iter().all(|v| v.abs() < 1e-12));
  assert!(k.asymmetry() < 1e-12);
  // u = x^2 is in P2: int |grad u|^2 = 4/3 three ways
  let u: Vec<f64> = (0..ws.nb_dof())
    .map(|d| mf.point_of_basic_dof(d).unwrap()[0].powi(2))
    .collect();
  ws.set_variable("u", &u).unwrap();
  let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
  let e = {
    ws.clear_expressions();
    ws.add_expression("Grad_u.Grad_u", &mim, None).unwrap();
    ws.assembly(0, false).unwrap();
    ws.assembled_potential()
  };
  let r = {
    ws.clear_expressions();
    ws.add_expression("Grad_u.Grad_Test_u", &mim, None).unwrap();
    ws.assembly(1, false).unwrap();
    ws.assembled_vector()
  };
  let ku = k.matvec(&u).unwrap();
  for v in [e, dot(r, &u), dot(&ku, &u)] {
    assert!((v - 4.0 / 3.0).abs() < 1e-12, "{v}");
  }
}

/// Terms equal up to swapped test functions, as GetFEM detects them
/// (sub_tree_are_equal, version 2): a two-test term and its mirror,
/// (A Grad_Test_u):Grad_Test2_u + (A Grad_Test2_u):Grad_Test_u = K + K^T
/// with K the first term's matrix (A not symmetric), and the single-test
/// mirror Test2_u.Test_u = Test_u.Test2_u.
#[test]
fn mirrored_test_terms() {
  let (m, mim) = unit_square("GT_QK(2,2)", 3);
  let mf = MeshFem::new(&m, 2)
    .with_classical_finite_element(2)
    .unwrap();
  let mut ws = Workspace::new();
  ws.add_fem_variable("u", &mf).unwrap();
  ws.add_fixed_size_data("A", &[2, 2], &[2.0, 0.3, -0.7, 1.5])
    .unwrap();
  let mut k = |e: &str| {
    {
      ws.clear_expressions();
      ws.add_expression(e, &mim, None).unwrap();
      ws.assembly(2, false).unwrap();
      ws.assembled_matrix()
    }
    .clone()
  };
  let k1 = k("(A*Grad_Test_u):Grad_Test2_u");
  let ks = k("(A*Grad_Test_u):Grad_Test2_u + (A*Grad_Test2_u):Grad_Test_u");
  let (m1, m2) = (k("Test_u.Test2_u"), k("Test2_u.Test_u"));
  let n = k1.nrows;
  let scale = (0..n).map(|i| k1.get(i, i).abs()).fold(0.0, f64::max);
  let (mut asym, mut err, mut err_m) = (0.0f64, 0.0f64, 0.0f64);
  for i in 0..n {
    for j in 0..n {
      asym = asym.max((k1.get(i, j) - k1.get(j, i)).abs());
      err = err.max((ks.get(i, j) - k1.get(i, j) - k1.get(j, i)).abs());
      err_m = err_m.max((m1.get(i, j) - m2.get(i, j)).abs());
    }
  }
  // the mirror really differs from the term (else the test is empty)
  assert!(asym > 1e-3 * scale, "K symmetric: {asym:.1e}");
  assert!(err < 1e-13 * scale, "K + K^T: {err:.1e}");
  assert!(err_m < 1e-13 * scale, "Test_u.Test2_u: {err_m:.1e}");
}
