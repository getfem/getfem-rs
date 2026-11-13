//! A matrix field (qdims 2 x 2) as a variable: its values, Grad_c (2 x 2 x
//! N), Hess_c, and residuals and tangents through Test_c, Test2_c. c = [x,
//! y; xy, x^2], exact in Q2, on the unit square.

use getfem_fem::mesh_fem::MeshFem;
use getfem_fem::mesh_im::MeshIm;
use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::regular_mesh::regular_unit_mesh;
use getfem_gwfl::workspace::Workspace;

fn setup() -> (Workspace, MeshIm, Vec<f64>) {
  let m = regular_unit_mesh(&[2, 2], &geotrans_descriptor("GT_QK(2,1)").unwrap()).unwrap();
  let mf = MeshFem::new(&m, 1);
  mf.set_qdim([2, 2]);
  mf.set_classical_finite_element(2).unwrap();
  let mim = MeshIm::new(&m).with_integration_method(4).unwrap();
  let mut ws = Workspace::new();
  ws.add_fem_variable("c", &mf).unwrap();
  let c = ws
    .interpolation("[X(1), X(2); X(1)*X(2), sqr(X(1))]", &mf, -1)
    .unwrap();
  ws.set_variable("c", &c).unwrap();
  (ws, mim, c)
}

fn assemble(ws: &mut Workspace, mim: &MeshIm, e: &str, order: usize) {
  ws.clear_expressions();
  ws.add_expression(e, mim, -1).unwrap();
  ws.assembly(order, false).unwrap();
}

#[test]
fn values_gradients_hessians() {
  let (mut ws, mim, _) = setup();
  // Grad_c(i,j,k): d c(i,j) / d x_k; |Grad_c|^2 = 2 + y^2 + 5 x^2
  for (e, v) in [
    ("c(2,1)", 0.25),
    ("Norm_sqr(c)", 1.0 / 3.0 + 1.0 / 3.0 + 1.0 / 9.0 + 0.2),
    ("qdim(c) + qdims(c)(1) + qdims(c)(2)", 8.0),
    ("Grad_c(2,1,1)", 0.5),
    ("Grad_c(2,1,2)", 0.5),
    ("Grad_c(2,2,1)", 1.0),
    ("Grad_c(1,2,2)", 1.0),
    ("Grad(c)(2,2,1)", 1.0),
    ("Grad(X(2)*c)(2,1,2)", 0.5),
    ("Hess_c(2,1,1,2)", 1.0),
    ("Hess_c(2,2,1,1)", 2.0),
    ("Norm_sqr(Reshape(Grad_c, 8))", 4.0),
    (
      "Grad_c(:,:,1):Grad_c(:,:,1) + Grad_c(:,:,2):Grad_c(:,:,2)",
      4.0,
    ),
    ("Contract(Grad_c, 3, Grad_c, 3)(2,1,2,1)", 2.0 / 3.0),
  ] {
    assemble(&mut ws, &mim, e, 0);
    let a = ws.assembled_potential();
    assert!((a - v).abs() < 1e-12, "{e}: {a} vs {v}");
  }
}

#[test]
fn residual_and_tangent_through_tests() {
  let (mut ws, mim, c) = setup();
  let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
  // P = |Grad_c|^2 / 2, quadratic: R(c).c = 2 P = 4, c.K.c = 4
  let p = "Norm_sqr(Reshape(Grad_c, 8)) / 2";
  assemble(&mut ws, &mim, p, 1);
  let r = ws.assembled_vector().to_vec();
  assert!((dot(&r, &c) - 4.0).abs() < 1e-12, "{}", dot(&r, &c));
  assemble(
    &mut ws,
    &mim,
    "Reshape(Grad_c, 8).Reshape(Grad_Test_c, 8)",
    1,
  );
  let d = ws
    .assembled_vector()
    .iter()
    .zip(&r)
    .map(|(a, b)| (a - b).abs());
  assert!(d.fold(0.0, f64::max) < 1e-13);
  assemble(&mut ws, &mim, p, 2);
  let k = ws.assembled_matrix().clone();
  let kc = |k: &getfem_linalg::ColMatrix<f64>| {
    let mut y = vec![0.0; c.len()];
    for (j, col) in k.cols.iter().enumerate() {
      for &(i, v) in col.iter() {
        y[i] += v * c[j];
      }
    }
    y
  };
  let y = kc(&k);
  assert!((dot(&y, &c) - 4.0).abs() < 1e-12, "{}", dot(&y, &c));
  // the tangent written out: the same matrix
  let t = "Reshape(Grad_Test2_c, 8).Reshape(Grad_Test_c, 8)";
  assemble(&mut ws, &mim, t, 2);
  let yt = kc(ws.assembled_matrix());
  let d = yt.iter().zip(&y).map(|(a, b)| (a - b).abs());
  assert!(d.fold(0.0, f64::max) < 1e-13);
  // mass-like term: c:Test_c, R.c = |c|^2
  assemble(&mut ws, &mim, "c:Test_c", 1);
  let m = 1.0 / 3.0 + 1.0 / 3.0 + 1.0 / 9.0 + 0.2;
  assert!((dot(ws.assembled_vector(), &c) - m).abs() < 1e-12);
}
