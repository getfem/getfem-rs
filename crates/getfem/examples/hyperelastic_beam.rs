//! Large deflection of a compressible neo-Hookean cantilever (Q2 hexahedra),
//! clamped at x = 0, loaded by 10 times its own weight in 10 steps. Only the
//! stored energy is written: GWFL derives the residual and the tangent.
//! Newton's three phases (full Newton where it converges). Writes
//! target/showcase/hyperelastic.vtu.

use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im::MeshIm;
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::region::{outer_faces_of_mesh, select_faces_in_box};
use getfem::geom::regular_mesh::regular_mesh;
use getfem::io::vtk::{Format, VtuExport};
use getfem::model::{Model, Newton};

const ENERGY: &str = "K/2*sqr(log(Det(Id(3)+Grad_u))) \
  + G/2*(pow(Det(Id(3)+Grad_u),-2/3)*Norm_sqr(Id(3)+Grad_u) - 3) \
  - load*[0,0,-1].u";

fn main() {
  let out = format!("{}/../../target/showcase", env!("CARGO_MANIFEST_DIR"));
  std::fs::create_dir_all(&out).unwrap();
  let (len, e, nu) = (10.0, 1e4, 0.3);
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
  let mut md = Model::new();
  md.add_fem_variable("u", &mf).unwrap();
  md.add_scalar_data("K", e / (3.0 * (1.0 - 2.0 * nu)))
    .unwrap();
  md.add_scalar_data("G", e / (2.0 * (1.0 + nu))).unwrap();
  md.add_scalar_data("load", 0.0).unwrap();
  md.add_term(&mim, None, ENERGY).unwrap();
  // the clamped end x = 0, region 1
  let clamp = select_faces_in_box(
    &mesh,
    outer_faces_of_mesh(&mesh),
    &[0.0; 3],
    &[0.0, 1.0, 1.0],
  );
  mesh.set_region(1, &clamp.unwrap()).unwrap();
  md.add_Dirichlet_condition_with_simplification("u", 1, None)
    .unwrap();
  let n = md.nb_dof();
  let newton = Newton::default();
  // ten times the gravity of the linear cantilever example
  let steps = 10;
  for step in 1..=steps {
    let load = 10.0 * step as f64 / steps as f64;
    md.set_variable("load", &[load]).unwrap();
    let rep = newton.solve(&mut md, None).unwrap();
    assert!(rep.converged(), "no convergence at load {load}");
    let res: Vec<String> = rep.residuals.iter().map(|r| format!("{r:.1e}")).collect();
    println!(
      "load {load:5.2}: {} iterations, |R| {}",
      rep.iterations,
      res.join(", ")
    );
  }
  let u = md.variable("u").unwrap().to_vec();
  // tip: mean displacement of the nodes at x = L
  let mean_at_tip = |c: usize| {
    let v: Vec<f64> = (c..u.len())
      .step_by(3)
      .filter(|&d| (mf.point_of_basic_dof(d).unwrap()[0] - len).abs() < 1e-12)
      .map(|d| u[d])
      .collect();
    v.iter().sum::<f64>() / v.len() as f64
  };
  let (tip, horiz) = (mean_at_tip(2), mean_at_tip(0));
  println!(
    "{n} DOFs, tip displacement: vertical {tip:.3}, horizontal {horiz:.3} (linear elasticity: 14.97, 0)"
  );
  let mut vtu = VtuExport::new(&format!("{out}/hyperelastic.vtu"), Format::Ascii);
  vtu.exporting(&mf).unwrap();
  vtu.write_point_data(&mf, &u, "u").unwrap();
  vtu.write().unwrap();
}
