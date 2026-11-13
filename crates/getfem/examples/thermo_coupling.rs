//! Thermo-electro-elastic plate with three holes (GetFEM's
//! thermo_elasticity_electrical_coupling): electric potential with a
//! temperature dependent resistivity, Joule heating, thermal expansion.
//! Mesh from GetFEM's mesher (data/meshes). Writes target/showcase/thermo.vtu.

use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im::MeshIm;
use getfem::geom::mesh_file::read_mesh_file;
use getfem::io::vtk::{Format, VtuExport};
use getfem::model::{Model, Newton};

// regions of the mesh file, and the union of top and bottom
const RIGHT: usize = 1;
const LEFT: usize = 2;
const TOP: usize = 3;
const BOTTOM: usize = 4;
const TOP_BOTTOM: usize = 5;

fn main() {
  let out = format!("{}/../../target/showcase", env!("CARGO_MANIFEST_DIR"));
  std::fs::create_dir_all(&out).unwrap();
  let file = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../data/meshes/thermo_elasticity_electrical_coupling.mesh"
  );
  let mesh = read_mesh_file(file).unwrap();
  let space = |q: usize, k: usize| {
    MeshFem::new(&mesh, q)
      .with_classical_finite_element(k)
      .unwrap()
  };
  let (mfu, mft) = (space(2, 2), space(1, 2));
  let mim = MeshIm::new(&mesh).with_integration_method(4).unwrap();

  let mut md = Model::new();
  md.add_fem_variable("u", &mfu).unwrap();
  md.add_fem_variable("T", &mft).unwrap();
  md.add_fem_variable("V", &mft).unwrap();
  for (n, v) in [
    ("t", 1.0),
    ("E", 21e6),
    ("nu", 0.3),
    ("alpha_th", 16.6e-6),
    ("T0", 20.0),
    ("F", 100e2),
    ("rho_0", 1.754e-8),
    ("alpha", 0.0039),
    ("V_left", 0.1),
    ("kappa", 4.0),
    ("D", 10.0),
    ("T_air", 20.0),
  ] {
    md.add_scalar_data(n, v).unwrap();
  }
  md.add_macro(
    "sigma",
    "E/(1+nu)*(nu/(1-nu)*(Div_u-2*alpha_th*T)*Id(2) + Sym(Grad_u)-alpha_th*T*Id(2))",
  )
  .unwrap();
  md.add_macro("rho", "rho_0*(1+alpha*(T-T0))").unwrap();
  // elasticity with thermal expansion, clamped left, traction right
  md.add_term(&mim, None, "t*sigma:Grad_Test_u").unwrap();
  md.add_term(&mim, RIGHT, "-t*F*Test_u(1)").unwrap();
  md.add_Dirichlet_condition_with_multiplier(&mim, "u", &space(2, 1), LEFT, None)
    .unwrap();
  // current with temperature dependent resistivity, 0.1 V across
  md.add_term(&mim, None, "t/rho*Grad_V.Grad_Test_V").unwrap();
  md.add_Dirichlet_condition_with_multiplier(&mim, "V", &space(1, 1), RIGHT, None)
    .unwrap();
  md.add_Dirichlet_condition_with_multiplier(&mim, "V", &space(1, 1), LEFT, Some("V_left"))
    .unwrap();
  // heat: conduction, convection on the faces and edges, Joule heating
  md.add_term(
    &mim,
    None,
    "t*kappa*Grad_T.Grad_Test_T + 2*D*(T-T_air)*Test_T - t/rho*Norm_sqr(Grad_V)*Test_T",
  )
  .unwrap();
  // convection on top and bottom
  let top_bottom = mesh
    .region(TOP)
    .unwrap()
    .merge(&mesh.region(BOTTOM).unwrap())
    .unwrap();
  mesh.set_region(TOP_BOTTOM, &top_bottom).unwrap();
  md.add_term(&mim, TOP_BOTTOM, "t*D*(T-T_air)*Test_T")
    .unwrap();

  // one Newton over all fields (the coupling is one-way, T -> u); see
  // newton_tolerances.rs for tolerances per variable and stall reports
  let newton = Newton {
    rtol: 1e-10,
    ..Default::default()
  };
  let rep = newton.solve(&mut md, None).unwrap();
  assert!(rep.converged(), "{}", rep.message);
  let l2 = {
    let ws = md.workspace_mut();
    ws.clear_expressions();
    ws.add_expression("sqr(T)", &mim, None).unwrap();
    ws.assembly(0, false).unwrap();
    ws.assembled_potential()
  }
  .sqrt();
  println!("{} DOFs, L2 norm of the temperature {l2:.6}", md.nb_dof());

  let mut vtu = VtuExport::new(&format!("{out}/thermo.vtu"), Format::Ascii);
  vtu.exporting(&mfu).unwrap();
  for (mf, v) in [(&mfu, "u"), (&mft, "T"), (&mft, "V")] {
    vtu
      .write_point_data(mf, md.variable(v).unwrap(), v)
      .unwrap();
  }
  vtu.write().unwrap();
}
