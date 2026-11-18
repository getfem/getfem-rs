//! The same Newton solve on any number of ranks: a nonlinear model with a
//! linear term, a source, Dirichlet multipliers, a Dirichlet elimination
//! and an explicit matrix and rhs, solved before getfem::mpi::init (each
//! process alone) and after (each rank its elements' part of the tangent);
//! the residuals of every iteration and the solution agree to round-off.
//! ci/mpi.sh runs it on 1, 2 and 4 ranks. Process-wide MPI: a test binary
//! of its own, one test.

use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im::MeshIm;
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::region::{outer_faces_of_mesh, select_faces_in_box};
use getfem::geom::regular_mesh::regular_unit_mesh;
use getfem::linalg::Coo;
use getfem::model::{Model, Newton};

/// the model, its start values
fn model() -> (Model, Vec<f64>) {
  let m = regular_unit_mesh(&[8, 8], &geotrans_descriptor("GT_PK(2,1)").unwrap()).unwrap();
  let border = outer_faces_of_mesh(&m);
  let left = select_faces_in_box(&m, &border, &[-0.1, -0.1], &[0.1, 1.1]).unwrap();
  let right = select_faces_in_box(&m, &border, &[0.9, -0.1], &[1.1, 1.1]).unwrap();
  m.set_region(1, &left).unwrap();
  m.set_region(2, &right).unwrap();
  let mf = MeshFem::new(&m, 2)
    .with_classical_finite_element(2)
    .unwrap();
  let mim = MeshIm::new(&m).with_integration_method(4).unwrap();
  let n = mf.nb_dof().unwrap();
  let mut md = Model::new();
  md.add_fem_variable("u", &mf).unwrap();
  md.add_term(&mim, None, "(1+Norm_sqr(u))*Grad_u:Grad_Test_u")
    .unwrap();
  md.add_term(&mim, None, "0.5*u.Test_u - [1;2].Test_u")
    .unwrap();
  md.add_fem_data("g", &mf, &vec![0.0; n]).unwrap();
  let g = md.interpolation("[0.1*X(2);-0.2*X(2)]", &mf, -1).unwrap();
  md.set_variable("g", &g).unwrap();
  md.add_Dirichlet_condition_with_multiplier(&mim, "u", &mf, 1, Some("g"))
    .unwrap();
  md.add_Dirichlet_condition_with_simplification("u", 2, Some("g"))
    .unwrap();
  // a few couplings and loads not from GWFL
  let mut k = Coo::new(n, n);
  for i in (0..n).step_by(7) {
    k.push(i, i, 0.3);
    k.push(i, (i + 3) % n, 0.1);
    k.push((i + 3) % n, i, 0.1);
  }
  md.set_explicit_matrix("u", "u", Some(k.to_csr().unwrap()))
    .unwrap();
  md.set_explicit_rhs("u", Some((0..n).map(|i| 0.01 * (i % 5) as f64).collect()))
    .unwrap();
  let x0 = md.from_variables();
  (md, x0)
}

#[test]
fn same_newton_on_any_ranks() {
  let newton = Newton {
    rtol: 1e-12,
    ..Default::default()
  };
  let (mut md, x0) = model();
  let r0 = newton.solve(&mut md, None).unwrap();
  assert!(r0.converged(), "{}", r0.message);
  let u0 = md.from_variables();
  let mpi = getfem::mpi::init().unwrap();
  md.to_variables(&x0).unwrap();
  let r = newton.solve(&mut md, None).unwrap();
  assert!(r.converged(), "{}", r.message);
  assert_eq!((r.method, r.iterations), (r0.method, r0.iterations));
  for (a, b) in r.residuals.iter().zip(&r0.residuals) {
    assert!(
      (a - b).abs() <= 1e-10 * r0.residuals[0],
      "{:?} {:?}",
      r.residuals,
      r0.residuals
    );
  }
  let u = md.from_variables();
  let nu = u0.iter().fold(0.0f64, |a, x| a.max(x.abs()));
  let d = u
    .iter()
    .zip(&u0)
    .fold(0.0f64, |a, (x, y)| a.max((x - y).abs()));
  assert!(d <= 1e-12 * nu, "{d} {nu}");
  drop(mpi);
}
