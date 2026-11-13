//! 3D cantilever under its own weight, linear elasticity with Q2 hexahedra:
//! L = 10, square section 1 x 1, E = 1e4, nu = 0.3, clamped at x = 0.
//! Tip deflection vs Timoshenko beam theory. Writes target/showcase/cantilever.vtu.

use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im::MeshIm;
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::region::{outer_faces_of_mesh, select_faces_in_box};
use getfem::geom::regular_mesh::regular_mesh;
use getfem::gwfl::workspace::Workspace;
use getfem::io::vtk::{Format, VtuExport};
use getfem::linalg::{LinearSolver, MatrixKind};
use getfem::mumps::Mumps;

fn main() {
  let out = format!("{}/../../target/showcase", env!("CARGO_MANIFEST_DIR"));
  std::fs::create_dir_all(&out).unwrap();
  let (len, e, nu) = (10.0, 1e4, 0.3);
  let (lambda, mu) = (
    e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)),
    e / (2.0 * (1.0 + nu)),
  );
  let mesh = regular_mesh(
    &geotrans_descriptor("GT_QK(3,1)").unwrap(),
    &[20, 2, 2],
    &[0.0; 3],
    &[len, 1.0, 1.0],
  )
  .unwrap();
  let mf = MeshFem::new(&mesh, 3)
    .with_classical_finite_element(2)
    .unwrap();
  let mim = MeshIm::new(&mesh).with_integration_method(4).unwrap();
  // u = 0 on the clamp: the unknown on the other DOFs
  let clamp = select_faces_in_box(
    &mesh,
    outer_faces_of_mesh(&mesh),
    &[0.0, 0.0, 0.0],
    &[0.0, 1.0, 1.0],
  )
  .unwrap();
  let clamped = mf.basic_dof_on_region(&clamp).unwrap();
  let kept = (0..mf.nb_dof().unwrap())
    .filter(|d| !clamped.contains(d))
    .collect();
  let mfu = MeshFem::partial(&mf, &kept).unwrap();
  let mut ws = Workspace::new();
  ws.add_fem_variable("u", &mfu).unwrap();
  ws.add_scalar_data("lambda", lambda).unwrap();
  ws.add_scalar_data("mu", mu).unwrap();
  let k = {
    ws.clear_expressions();
    ws.add_expression(
      "lambda*Div_Test_u*Div_Test2_u + mu*(Grad_Test_u + Grad_Test_u'):Grad_Test2_u",
      &mim,
      None,
    )
    .unwrap();
    ws.assembly(2, false).unwrap();
    ws.assembled_matrix()
  }
  .to_csr();
  let mut b = {
    ws.clear_expressions();
    ws.add_expression("[0,0,-1].Test_u", &mim, None).unwrap();
    ws.assembly(1, false).unwrap();
    ws.assembled_vector()
  }
  .to_vec();
  Mumps::new()
    .solve_system(&k, MatrixKind::SymmetricPositiveDefinite, &mut b)
    .unwrap();
  let u = mfu.extend_vector(&b).unwrap();
  // tip: mean vertical displacement of the nodes at x = L
  let tip: Vec<f64> = (2..u.len())
    .step_by(3)
    .filter(|&d| (mf.point_of_basic_dof(d).unwrap()[0] - len).abs() < 1e-12)
    .map(|d| u[d])
    .collect();
  let tip = tip.iter().sum::<f64>() / tip.len() as f64;
  let (i2, area) = (1.0 / 12.0, 1.0);
  let bending = len.powi(4) / (8.0 * e * i2);
  let shear = len * len / (2.0 * (5.0 / 6.0) * mu * area);
  println!(
    "{} DOFs, tip deflection {:.4}, beam theory {:.4} (bending {bending:.4} + shear {shear:.4})",
    mf.nb_basic_dof().unwrap(),
    -tip,
    bending + shear
  );
  let mut vtu = VtuExport::new(&format!("{out}/cantilever.vtu"), Format::Ascii);
  vtu.exporting(&mf).unwrap();
  vtu.write_point_data(&mfu, &b, "u").unwrap();
  vtu.write().unwrap();
}
