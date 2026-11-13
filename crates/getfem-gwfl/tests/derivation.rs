//! Derivation: residuals from potentials and tangents from residuals,
//! checked against the explicit forms and against finite differences.

use getfem_fem::mesh_fem::MeshFem;
use getfem_fem::mesh_im::MeshIm;
use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::regular_mesh::regular_mesh;
use getfem_gwfl::workspace::Workspace;
use getfem_linalg::sparse::Csr;
use std::sync::Arc;

fn setup(gt: &str, q: usize) -> (Workspace, MeshIm) {
  let n = if gt.contains("(3,") { 3 } else { 2 };
  let m = regular_mesh(
    &geotrans_descriptor(gt).unwrap(),
    &vec![2; n],
    &vec![0.0; n],
    &vec![1.0; n],
  )
  .unwrap();
  // curve the mesh a little (non-linear transformations only)
  if gt.contains("QK") {
    for i in m.point_index().into_iter().collect::<Vec<_>>() {
      let p = m.point(i).to_vec();
      let mut x = p.clone();
      x[0] += 0.05 * (2.0 * p[1]).sin();
      m.set_point(i, &x).unwrap();
    }
  }
  let mf = MeshFem::new(&m, q)
    .with_classical_finite_element(2)
    .unwrap();
  let mim = MeshIm::new(&m).with_integration_method(4).unwrap();
  let mut ws = Workspace::new();
  ws.add_fem_variable("u", &mf).unwrap();
  let nd = ws.nb_dof();
  ws.set_variable(
    "u",
    &(0..nd)
      .map(|i| 0.05 * (0.7 * i as f64).sin())
      .collect::<Vec<_>>(),
  )
  .unwrap();
  ws.add_scalar_data("K", 3.0).unwrap();
  ws.add_scalar_data("G", 1.5).unwrap();
  (ws, mim)
}

fn csr(ws: &mut Workspace, e: &str, mim: &MeshIm) -> Csr<f64> {
  ({
    ws.clear_expressions();
    ws.add_expression(e, mim, None).unwrap();
    ws.assembly(2, false).unwrap();
    ws.assembled_matrix()
  })
  .to_csr()
}

fn max_diff(a: &Csr<f64>, b: &Csr<f64>) -> f64 {
  let mut w = 0.0f64;
  for i in 0..a.nrows {
    for (j, v) in a.row(i).chain(b.row(i)) {
      let _ = v;
      w = w.max((a.get(i, j) - b.get(i, j)).abs());
    }
  }
  w
}

#[test]
fn laplace_potential() {
  let (mut ws, mim) = setup("GT_PK(2,1)", 1);
  let k = csr(&mut ws, "Grad_Test_u.Grad_Test2_u", &mim);
  assert!(max_diff(&k, &csr(&mut ws, "Norm_sqr(Grad_u)/2", &mim)) < 1e-12);
  assert!(max_diff(&k, &csr(&mut ws, "Grad_u.Grad_Test_u", &mim)) < 1e-12);
  let r = {
    ws.clear_expressions();
    ws.add_expression("Norm_sqr(Grad_u)/2", &mim, None).unwrap();
    ws.assembly(1, false).unwrap();
    ws.assembled_vector()
  }
  .to_vec();
  let ku = k.matvec(ws.variable("u").unwrap()).unwrap();
  assert!(r.iter().zip(&ku).all(|(a, b)| (a - b).abs() < 1e-12));
}

/// residual = finite differences of the potential, tangent = those of the residual
fn fd_check(gt: &str, q: usize, w: &str) {
  let (mut ws, mim) = setup(gt, q);
  let u0 = ws.variable("u").unwrap().to_vec();
  let r = {
    ws.clear_expressions();
    ws.add_expression(w, &mim, None).unwrap();
    ws.assembly(1, false).unwrap();
    ws.assembled_vector()
  }
  .to_vec();
  let k = csr(&mut ws, w, &mim);
  let h = 1e-6;
  for i in (0..u0.len()).step_by(3) {
    let mut up = u0.clone();
    up[i] += h;
    ws.set_variable("u", &up).unwrap();
    let (ep, rp) = (
      {
        ws.clear_expressions();
        ws.add_expression(w, &mim, None).unwrap();
        ws.assembly(0, false).unwrap();
        ws.assembled_potential()
      },
      {
        ws.clear_expressions();
        ws.add_expression(w, &mim, None).unwrap();
        ws.assembly(1, false).unwrap();
        ws.assembled_vector()
      }
      .to_vec(),
    );
    up[i] -= 2.0 * h;
    ws.set_variable("u", &up).unwrap();
    let (em, rm) = (
      {
        ws.clear_expressions();
        ws.add_expression(w, &mim, None).unwrap();
        ws.assembly(0, false).unwrap();
        ws.assembled_potential()
      },
      {
        ws.clear_expressions();
        ws.add_expression(w, &mim, None).unwrap();
        ws.assembly(1, false).unwrap();
        ws.assembled_vector()
      },
    );
    let fd = (ep - em) / (2.0 * h);
    assert!(
      (fd - r[i]).abs() < 1e-6 * (1.0 + fd.abs()),
      "{w} residual[{i}]: {} vs {fd}",
      r[i]
    );
    for j in 0..u0.len() {
      let fd = (rp[j] - rm[j]) / (2.0 * h);
      assert!(
        (fd - k.get(j, i)).abs() < 1e-6 * (1.0 + fd.abs()),
        "{w} tangent[{j},{i}]: {} vs {fd}",
        k.get(j, i)
      );
    }
  }
  assert!(
    k.asymmetry() < 1e-10,
    "{w}: tangent of a potential is symmetric"
  );
}

#[test]
fn hyperelastic_potentials() {
  // compressible neo-Hookean, 2D plane strain and 3D
  fd_check(
    "GT_QK(2,2)",
    2,
    "K/2*sqr(log(Det(Id(2)+Grad_u))) + G/2*(pow(Det(Id(2)+Grad_u),-2/3)*(Norm_sqr(Id(2)+Grad_u)+1)-3)",
  );
  fd_check(
    "GT_PK(3,1)",
    3,
    "K/2*sqr(log(Det(Id(3)+Grad_u))) + G/2*(pow(Det(Id(3)+Grad_u),-2/3)*Norm_sqr(Id(3)+Grad_u)-3)",
  );
  // Saint Venant-Kirchhoff through Green_Lagrangian, a Hessian and an explicit tensor
  fd_check(
    "GT_PK(2,1)",
    2,
    "K/2*sqr(Trace(Green_Lagrangian(Id(2)+Grad_u))) + G*Norm_sqr(Green_Lagrangian(Id(2)+Grad_u))",
  );
  fd_check(
    "GT_QK(2,2)",
    1,
    "sqr(Hess_u:Id(2))/2 + cos(u)*[1,2].Grad_u + pow(1+u*u, 1.5)",
  );
  // divisions: a constant numerator, both operands of u, ./ (and A:A, Norm_sqr)
  fd_check(
    "GT_PK(2,2)",
    1,
    "1/(2+u*u) + u*u/(1+Norm_sqr(Grad_u)) + (Grad_u./([2,3]+Grad_u.*Grad_u)).[1,2] + (Grad_u.Grad_u)*u",
  );
}

#[test]
fn given_derivatives_of_a_function() {
  // GetFEM ga_define_function(name, n, expr, der1, der2): the given
  // expressions are the derivatives (here deliberately not the true ones)
  let (mut ws, mim) = setup("GT_PK(2,1)", 1);
  getfem_gwfl::define_gwfl_function("d1", 2, "3*t", "", "").unwrap();
  getfem_gwfl::define_gwfl_function("d2", 2, "5*u", "", "").unwrap();
  getfem_gwfl::define_gwfl_function("f", 2, "t*t*u", "d1(t, u)", "d2(t, u)").unwrap();
  getfem_gwfl::define_gwfl_function("g", 2, "t*t*u", "d1(t, u)", "").unwrap();
  let mut v = |e: &str| {
    {
      ws.clear_expressions();
      ws.add_expression(e, &mim, None).unwrap();
      ws.assembly(1, false).unwrap();
      ws.assembled_vector()
    }
    .to_vec()
  };
  let close = |a: Vec<f64>, b: Vec<f64>| a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-13);
  assert!(close(v("f(u, 2*u)"), v("(3*u + 2*5*(2*u))*Test_u")));
  // only one of two given: derived symbolically
  assert!(close(v("g(u, 2*u)"), v("(2*u*(2*u) + 2*u*u)*Test_u")));
}

#[test]
fn log_sqrt_pow_derived_three_times() {
  // derivatives written as expressions (1/t, 0.5/sqrt(t), u*pow(t,u-1)),
  // derivable any number of times
  let (mut ws, mim) = setup("GT_PK(2,1)", 1);
  let mut v = |e: &str| {
    {
      ws.clear_expressions();
      ws.add_expression(e, &mim, None).unwrap();
      ws.assembly(1, false).unwrap();
      ws.assembled_vector()
    }
    .to_vec()
  };
  let d3 = |f: &str| format!("Diff(Diff(Diff({f},u,1),u,1),u)");
  let close = |a: Vec<f64>, b: Vec<f64>| a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-13);
  assert!(close(v(&d3("log(u+2)")), v("2/pow(u+2,3)*Test_u")));
  assert!(close(v(&d3("sqrt(u+2)")), v("0.375*pow(u+2,-2.5)*Test_u")));
  assert!(close(
    v(&d3("pow(u+2,2.5)")),
    v("1.875*pow(u+2,-0.5)*Test_u")
  ));
}

#[test]
fn native_functions_and_differentiability() {
  use getfem_gwfl::defs::NativeFn;
  let (mut ws, mim) = setup("GT_PK(2,1)", 1);
  // softplus as a Rust closure with its derivative; the second derivative is
  // not given, so a tangent of a potential using it must fail clearly
  let f = NativeFn(Arc::new(|t, _| (1.0 + t.exp()).ln()));
  let d = NativeFn(Arc::new(|t, _| 1.0 / (1.0 + (-t).exp())));
  getfem_gwfl::define_gwfl_function_native("softplus", 1, f, vec![d]).unwrap();
  getfem_gwfl::define_gwfl_function("softplus_g", 1, "log(1+exp(t))", "", "").unwrap();
  let ws2 = &mut ws;
  // same residual as the GWFL-defined function
  let r1 = {
    ws2.clear_expressions();
    ws2
      .add_expression("softplus(u)*Test_u + Grad_u.Grad_Test_u", &mim, None)
      .unwrap();
    ws2.assembly(1, false).unwrap();
    ws2.assembled_vector()
  }
  .to_vec();
  let r2 = {
    ws2.clear_expressions();
    ws2
      .add_expression("softplus_g(u)*Test_u + Grad_u.Grad_Test_u", &mim, None)
      .unwrap();
    ws2.assembly(1, false).unwrap();
    ws2.assembled_vector()
  };
  assert!(r1.iter().zip(r2).all(|(a, b)| (a - b).abs() < 1e-13));
  let k1 = {
    ws2.clear_expressions();
    ws2
      .add_expression("softplus(u)*Test_u", &mim, None)
      .unwrap();
    ws2.assembly(2, false).unwrap();
    ws2.assembled_matrix()
  }
  .to_csr();
  let k2 = {
    ws2.clear_expressions();
    ws2
      .add_expression("softplus_g(u)*Test_u", &mim, None)
      .unwrap();
    ws2.assembly(2, false).unwrap();
    ws2.assembled_matrix()
  }
  .to_csr();
  assert!(max_diff(&k1, &k2) < 1e-13);
  ws2.clear_expressions();
  let e = ws2
    .add_expression("softplus(u)", &mim, None)
    .and_then(|_| ws2.assembly(2, false))
    .unwrap_err()
    .to_string();
  assert!(e.contains("DER1_softplus cannot be differentiated"), "{e}");
  // differentiability classes: pos_part once, Heaviside never
  let mut asm = |e: &str, order| {
    ws2.clear_expressions();
    ws2
      .add_expression(e, &mim, None)
      .and_then(|_| ws2.assembly(order, false))
  };
  assert!(asm("sqr_pos_part(u)", 1).is_ok());
  assert!(asm("pos_part(u)*Test_u", 2).is_ok());
  let e = asm("Heaviside(u)*Test_u", 2).unwrap_err().to_string();
  assert!(e.contains("Heaviside cannot be differentiated"), "{e}");
}
