//! Port of the GWFL part of GetFEM's tests/test_assembly.cc (test_new_assembly).
//! References to GetFEM's old assembly are replaced by exact values or by
//! identities between GWFL forms; checks on derivatives of potentials
//! (MAT_TEST_2 of order-0 expressions) and on partial spaces come with M7/M8.

use std::f64::consts::PI;

use getfem_fem::mesh_fem::MeshFem;
use getfem_fem::mesh_im::MeshIm;
use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::region::{outer_faces_of_mesh, select_faces_of_normal};
use getfem_geom::regular_mesh::regular_mesh;
use getfem_gwfl::workspace::Workspace;
use getfem_linalg::sparse::Csr;

fn fill(n: usize, seed: f64) -> Vec<f64> {
  (0..n).map(|i| (seed * (i as f64 + 1.0)).sin()).collect()
}

fn max_diff(a: &Csr<f64>, b: &Csr<f64>) -> f64 {
  let mut w = 0.0f64;
  for i in 0..a.nrows {
    for (j, v) in a.row(i) {
      w = w.max((v - b.get(i, j)).abs());
    }
    for (j, v) in b.row(i) {
      w = w.max((v - a.get(i, j)).abs());
    }
  }
  w
}

fn test_new_assembly(n: usize, nx: usize, pk: usize) {
  let m = regular_mesh(
    &geotrans_descriptor(&format!("GT_PK({n},1)")).unwrap(),
    &vec![nx; n],
    &vec![0.0; n],
    &vec![1.0; n],
  )
  .unwrap();
  let border = outer_faces_of_mesh(&m);
  let mut dir = vec![0.0; n];
  dir[n - 1] = 1.0;
  let neumann = select_faces_of_normal(&m, &border, &dir, 0.1).unwrap();
  let fem = |q| {
    MeshFem::new(&m, q)
      .with_classical_finite_element(pk)
      .unwrap()
  };
  let (mfu, mfp) = (fem(n), fem(1));
  let im = |k| MeshIm::new(&m).with_integration_method(k).unwrap();
  let (mim, mim2) = (im(4), im(2));
  let mut ws = Workspace::new();
  ws.add_fixed_size_data("a", &[1], &[3.0]).unwrap();
  ws.add_fixed_size_data("b", &[2], &[3.0, 6.0]).unwrap();
  ws.add_fem_variable("u", &mfu).unwrap();
  let mfa = {
    let mf = MeshFem::new(&m, 1);
    mf.set_qdim([n, n]);
    mf.set_classical_finite_element(pk).unwrap();
    mf
  };
  let na = mfa.nb_basic_dof().unwrap();
  ws.add_fem_data("A", &mfa, &fill(na, 0.11)).unwrap();
  ws.add_fem_variable("p", &mfp).unwrap();
  let (ndu, ndp) = (mfu.nb_basic_dof().unwrap(), mfp.nb_basic_dof().unwrap());
  let u = fill(ndu, 0.37);
  ws.set_variable("u", &u).unwrap();
  ws.set_variable("p", &fill(ndp, 0.73)).unwrap();
  let (iu, ip) = (
    ws.interval_of_variable("u").unwrap(),
    ws.interval_of_variable("p").unwrap(),
  );

  // function integration
  let scal = |ws: &mut Workspace, e: &str, mim: &MeshIm| {
    ws.clear_expressions();
    ws.add_expression(e, mim, None).unwrap();
    ws.assembly(0, false).unwrap();
    ws.assembled_potential()
  };
  let v = 2.0 * n as f64 / PI;
  for (e, val) in [
    ("1", 1.0),
    ("cos(pi*X(1))", 0.0),
    ("cos(pi*X).exp(X*0)", 0.0),
    ("-Derivative_sin(pi*X).exp(X*0)", 0.0),
    ("sin(pi*X).exp(X*0)", v),
    ("Derivative_cos(pi*X).exp(X*0)", -v),
    ("cos(pi*X).Id(meshdim)(:,1)", 0.0),
  ] {
    let r = scal(&mut ws, e, &mim);
    assert!((r - val).abs() < 1e-8, "{n}D {e}: {r} vs {val}");
  }
  if n == 2 {
    getfem_gwfl::define_gwfl_function("dummyfunc", 1, "sin(pi*t/2)+2*sqr(t)-[t;t].[t;t]", "", "")
      .unwrap();
    assert!((scal(&mut ws, "dummyfunc(5)", &mim) - 1.0).abs() < 1e-8);
    getfem_gwfl::define_gwfl_function("dummyfunc2", 1, "cos(pi*t)", "", "").unwrap();
    assert!(scal(&mut ws, "dummyfunc2(X(1))", &mim).abs() < 1e-8);
  }

  // groups of equivalent expressions (SCAL_TEST_2)
  let same = |ws: &mut Workspace, es: &[String], mim: &MeshIm| {
    let e1 = scal(ws, &es[0], mim);
    for e in &es[1..] {
      let e2 = scal(ws, e, mim);
      assert!((e1 - e2).abs() / (e1 + e2) < 1e-7, "{n}D {e}: {e2} vs {e1}");
    }
  };
  let comps = |f: &dyn Fn(usize) -> String, sep: &str| (1..=n).map(f).collect::<Vec<_>>().join(sep);
  let mut l2 = vec![
    "u.u".to_string(),
    "Norm_sqr(u)".into(),
    comps(&|i| format!("u({i})*u({i})"), "+"),
  ];
  let mut perm: Vec<usize> = (1..=n).collect();
  perm.swap(0, 1);
  let pv = perm
    .iter()
    .map(|i| format!("u({i})"))
    .collect::<Vec<_>>()
    .join(";");
  l2.push(format!("[{pv}].[{pv}]"));
  if n == 2 {
    l2.push("sqr(u(1)) + sqr(u(2))".into());
  }
  same(&mut ws, &l2, &mim);
  let h1 = vec![
    "Grad_u:Grad_u".to_string(),
    "Id(meshdim)*Grad_u:Grad_u".into(),
    comps(&|i| format!("Grad_u({i},:).Grad_u({i},:)"), "+"),
    comps(&|i| format!("Grad_u(:,{i}).Grad_u(:,{i})"), "+"),
  ];
  same(&mut ws, &h1, &mim2);
  // L2 norm against the mass matrix: u.u = U^T M U
  let mass = {
    ws.clear_expressions();
    ws.add_expression("Test_u.Test2_u", &mim, None).unwrap();
    ws.assembly(2, false).unwrap();
    ws.assembled_matrix()
  }
  .to_csr();
  let mu_full: Vec<f64> = {
    let mut x = vec![0.0; ws.nb_dof()];
    x[iu.clone()].copy_from_slice(&u);
    mass.matvec(&x).unwrap()
  };
  let utmu: f64 = u.iter().zip(&mu_full[iu.clone()]).map(|(a, b)| a * b).sum();
  assert!((scal(&mut ws, "u.u", &mim) - utmu).abs() < 1e-10 * utmu);

  // source and Neumann terms: u.Test_u = M U (on the region)
  let check_vec =
    |ws: &mut Workspace, e: &str, rg: Option<&getfem_geom::region::Region>, reference: &[f64]| {
      let r = {
        ws.clear_expressions();
        ws.add_expression(e, &mim, rg).unwrap();
        ws.assembly(1, false).unwrap();
        ws.assembled_vector()
      };
      let err = r
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
      assert!(err < 1e-10, "{n}D {e}: {err}");
    };
  check_vec(&mut ws, "u.Test_u", None, &mu_full);
  let mass_n = {
    ws.clear_expressions();
    ws.add_expression("Test_u.Test2_u", &mim, Some(&neumann))
      .unwrap();
    ws.assembly(2, false).unwrap();
    ws.assembled_matrix()
  }
  .to_csr();
  let mut x = vec![0.0; ws.nb_dof()];
  x[iu.clone()].copy_from_slice(&u);
  check_vec(
    &mut ws,
    "u.Test_u",
    Some(&neumann),
    &mass_n.matvec(&x).unwrap(),
  );
  // normal source term with a matrix field A: two spellings
  let r1 = {
    ws.clear_expressions();
    ws.add_expression(
      "(((Reshape(A,meshdim,meshdim))')*Normal).Test_u",
      &mim,
      Some(&neumann),
    )
    .unwrap();
    ws.assembly(1, false).unwrap();
    ws.assembled_vector()
  }
  .to_vec();
  check_vec(&mut ws, "(A'*Normal).Test_u", Some(&neumann), &r1);
  check_vec(&mut ws, "Test_u.(A'*Normal)", Some(&neumann), &r1);

  // stiffness matrices (MAT_TEST_1/2 with explicit test functions)
  let mat = |ws: &mut Workspace, e: &str, mim: &MeshIm| {
    {
      ws.clear_expressions();
      ws.add_expression(e, mim, None).unwrap();
      ws.assembly(2, false).unwrap();
      ws.assembled_matrix()
    }
    .to_csr()
  };
  let lap = mat(&mut ws, "Grad_Test_p:Grad_Test2_p", &mim2);
  let ones: Vec<f64> = (0..ws.nb_dof())
    .map(|i| if ip.contains(&i) { 1.0 } else { 0.0 })
    .collect();
  assert!(lap.matvec(&ones).unwrap().iter().all(|v| v.abs() < 1e-10));
  ws.add_scalar_data("lambda", 3.0).unwrap();
  ws.add_scalar_data("mu", 2.0).unwrap();
  let el = [
    "(lambda*Trace(Grad_Test_u)*Id(qdim(u)) + mu*(Grad_Test_u'+Grad_Test_u)):Grad_Test2_u".to_string(),
    "lambda*Div_Test_u*Div_Test2_u + mu*(Grad_Test_u'+Grad_Test_u):Grad_Test2_u".into(),
    "lambda*((Grad_Test2_u@Grad_Test_u):Id(meshdim)):Id(meshdim) + mu*(Grad_Test_u'+Grad_Test_u):Grad_Test2_u".into(),
    "lambda*Id(meshdim)@Id(meshdim)*Grad_Test_u:Grad_Test2_u + mu*(Grad_Test_u'+Grad_Test_u):Grad_Test2_u".into(),
    "lambda*(Id(meshdim)*Id(meshdim))@Id(meshdim)*Grad_Test_u:Grad_Test2_u + mu*(Grad_Test_u'+Grad_Test_u):Grad_Test2_u"
      .into(),
    format!(
      "lambda*Trace(Grad_Test_u)*Trace(Grad_Test2_u) + {}",
      comps(&|i| format!("mu*(Grad_Test_u'(:,{i})+Grad_Test_u(:,{i})):Grad_Test2_u(:,{i})"), "+")
    ),
    format!(
      "lambda*Trace(Grad_Test_u)*Trace(Grad_Test2_u) + {}",
      comps(&|i| format!("mu*(Grad_Test_u'({i},:)+Grad_Test_u({i},:)):Grad_Test2_u({i},:)"), "+")
    ),
  ];
  let k0 = mat(&mut ws, &el[0], &mim2);
  for e in &el[1..] {
    let err = max_diff(&k0, &mat(&mut ws, e, &mim2));
    assert!(err < 1e-10, "{n}D {e}: {err}");
  }
  // non homogeneous coefficients as fields
  ws.add_fem_data("lambda2", &mfp, &vec![3.0; ndp]).unwrap();
  ws.add_fem_data("mu2", &mfp, &vec![2.0; ndp]).unwrap();
  let k2 = mat(
    &mut ws,
    "(lambda2*Trace(Grad_Test_u)*Id(meshdim) + mu2*(Grad_Test_u'+Grad_Test_u)):Grad_Test2_u",
    &mim2,
  );
  assert!(max_diff(&k0, &k2) < 1e-10);
  let _ = (ip, a_b_used(&mut ws, &mim));
}

/// the fixed-size data a, b of the original test
fn a_b_used(ws: &mut Workspace, mim: &MeshIm) -> f64 {
  let v = {
    ws.clear_expressions();
    ws.add_expression("a*b(2)", mim, None).unwrap();
    ws.assembly(0, false).unwrap();
    ws.assembled_potential()
  };
  // summed point by point (as GetFEM): rounding relative to the value
  assert!((v - 18.0).abs() < 1e-12 * 18.0, "{v} {:e}", v - 18.0);
  v
}

#[test]
fn new_assembly_2d() {
  test_new_assembly(2, 25, 2);
}

#[test]
fn new_assembly_3d() {
  test_new_assembly(3, 7, 2);
}
