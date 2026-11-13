//! A nonlinear model solved on meshes of mixed element types (2D triangles
//! and quadrilaterals; 3D hexahedra, prisms, pyramids, tetrahedra; degree
//! 1 and 2): (1 + u^2) Grad_u.Grad_Test_u + u Test_u = the same of p,
//! p quadratic, in the space: Newton finds u = p on the unperturbed meshes;
//! on threads the solution of one thread.

#[path = "../../getfem-gwfl/tests/common/mod.rs"]
mod common;

use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im::MeshIm;
use getfem::model::{Model, Newton};

/// p and its gradient in dimension n
fn quadratic(n: usize) -> (&'static str, &'static str) {
  match n {
    2 => (
      "(1 + X(1) - 2*X(2) + X(1)*X(2) - 0.5*X(2)*X(2))",
      "[1 + X(2); -2 + X(1) - X(2)]",
    ),
    _ => (
      "(1 + X(1) - 2*X(2) + X(1)*X(3) - 0.5*X(2)*X(2) + X(3)*X(3))",
      "[1 + X(3); -2 - X(2); X(1) + 2*X(3)]",
    ),
  }
}

/// the solution on mixed mesh mi with nt threads, its errors to p (values,
/// gradients, squared integrals)
fn solve(mi: usize, perturbed: bool, nt: usize) -> (Vec<f64>, f64, f64) {
  getfem::set_num_threads(nt).unwrap();
  let m = common::mixed::mixed_mesh(mi, perturbed);
  let (p, gp) = quadratic(m.dim());
  let mf = MeshFem::new(&m, 1)
    .with_classical_finite_element(2)
    .unwrap();
  let mim = MeshIm::new(&m)
    .with_integration_method(if mi == 1 { 6 } else { 4 })
    .unwrap();
  let mut md = Model::new();
  md.add_fem_variable("u", &mf).unwrap();
  let e =
    format!("(1+u*u)*Grad_u.Grad_Test_u + u*Test_u - ((1+{p}*{p})*{gp}.Grad_Test_u + {p}*Test_u)");
  md.add_term(&mim, None, &e).unwrap();
  // to round-off (quadratic convergence)
  let newton = Newton {
    rtol: 1e-13,
    ..Default::default()
  };
  let rep = newton.solve(&mut md, None).unwrap();
  assert!(rep.converged(), "mesh {mi} on {nt} threads: not converged");
  let mut s = |e: &str| {
    let ws = md.workspace_mut();
    ws.clear_expressions();
    ws.add_expression(e, &mim, None).unwrap();
    ws.assembly(0, false).unwrap();
    ws.assembled_potential()
  };
  let (e0, e1) = (
    s(&format!("sqr(u - {p})")),
    s(&format!("Norm_sqr(Grad_u - {gp})")),
  );
  getfem::set_num_threads(1).unwrap();
  (md.variable("u").unwrap().to_vec(), e0, e1)
}

#[test]
fn nonlinear_model_on_mixed_meshes() {
  for mi in 0..4 {
    // p reproduced by the space (affine elements); gradients: round-off of
    // the rational pyramid basis next to the apex
    let (_, e0, e1) = solve(mi, false, 1);
    assert!(e0 < 1e-24 && e1 < 1e-20, "mesh {mi}: errors {e0:e} {e1:e}");
    // perturbed: the same solution on threads
    let (one, ..) = solve(mi, true, 1);
    for nt in [2, 8] {
      let (u, ..) = solve(mi, true, nt);
      let worst = u
        .iter()
        .zip(&one)
        .fold(0.0f64, |a, (x, y)| a.max((x - y).abs()));
      assert!(
        worst < 1e-12,
        "mesh {mi} on {nt} threads: difference {worst:e}"
      );
    }
  }
}
