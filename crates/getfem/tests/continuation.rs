//! GetFEM's test_continuation: -u'' + u = lambda exp(u) on (0, 1), Neumann,
//! P1, 10 elements, continued in lambda from u = 0: the path ends at
//! gamma_max.

use getfem::fem::fem::fem_descriptor;
use getfem::fem::integration::integ_descriptor;
use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im::MeshIm;
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::regular_mesh::regular_unit_mesh;
use getfem::model::{ContinuationStatus, Model, NumericalContinuation, VariableScale};

#[test]
fn path_ends_at_gamma_max() {
  // Bratu near its limit point 1/e (u constant, lambda = u exp(-u)): the
  // path bends, the last point exactly at gamma_max and on the path
  let m = regular_unit_mesh(&[10], &geotrans_descriptor("GT_PK(1,1)").unwrap()).unwrap();
  let mf = MeshFem::new(&m, 1)
    .with_finite_element(&fem_descriptor("FEM_PK(1,1)").unwrap())
    .unwrap();
  let mim = MeshIm::new(&m)
    .with_integration_method(&integ_descriptor("IM_GAUSS1D(3)").unwrap())
    .unwrap();
  let mut md = Model::new();
  md.add_fem_variable("u", &mf).unwrap();
  md.add_scalar_data("lambda", 0.0).unwrap();
  md.add_term(&mim, None, "Grad_u.Grad_Test_u").unwrap();
  md.add_term(&mim, None, "(u-lambda*exp(u))*Test_u").unwrap();
  let mut s = NumericalContinuation::new("lambda", vec![VariableScale::new(&["u"], 1.0)]);
  (s.h_init, s.h_max, s.gamma_max) = (2e-2, 2e-1, 0.36);
  let (mut tx, mut tgamma, mut h) = (vec![], 1.0, 0.0);
  s.step(&mut md, &mut tx, &mut tgamma, &mut h, None).unwrap();
  let mut steps = 0;
  while md.variable("lambda").unwrap()[0] < s.gamma_max {
    let rep = s.step(&mut md, &mut tx, &mut tgamma, &mut h, None).unwrap();
    assert_eq!(rep.status, ContinuationStatus::Converged);
    steps += 1;
    assert!(steps < 100);
  }
  assert_eq!(md.variable("lambda").unwrap()[0], 0.36);
  let u = md.variable("u").unwrap().to_vec();
  for v in &u {
    assert!((v * (-v).exp() - 0.36).abs() < 1e-8, "u {v}");
  }
  let rep = s.step(&mut md, &mut tx, &mut tgamma, &mut h, None).unwrap();
  assert_eq!(rep.status, ContinuationStatus::Failed);
  assert_eq!(md.variable("u").unwrap(), &u[..]);
}
