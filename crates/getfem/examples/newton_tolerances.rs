//! Convergence tests of the Newton driver, on the plate of
//! thermo_coupling.rs: the residual of each variable's equations, the hint
//! when some equations are solved less well than the global test suggests,
//! a tolerance below the rounding floor (converged there, with a hint), and
//! tolerances per variable with explicit scales.

use getfem::fem::mesh_fem::MeshFem;
use getfem::model::{Newton, VariableTolerance};

/// The plate of thermo_coupling.rs (the same model): model, integration
/// method, multiplier names.
fn thermo_plate() -> (
  getfem::model::Model,
  getfem::fem::mesh_im::MeshIm,
  [String; 3],
) {
  use getfem::fem::mesh_im::MeshIm;
  use getfem::geom::mesh_file::read_mesh_file;
  use getfem::model::Model;
  let file = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../data/meshes/thermo_elasticity_electrical_coupling.mesh"
  );
  let mesh = read_mesh_file(file).unwrap();
  // regions of the mesh file: 1 right, 2 left, 3 top, 4 bottom
  let (right, left) = (1, 2);
  let space = |q: usize, k: usize| {
    MeshFem::new(&mesh, q)
      .with_classical_finite_element(k)
      .unwrap()
  };
  let mim = MeshIm::new(&mesh).with_integration_method(4).unwrap();
  let mut md = Model::new();
  md.add_fem_variable("u", &space(2, 2)).unwrap();
  md.add_fem_variable("T", &space(1, 2)).unwrap();
  md.add_fem_variable("V", &space(1, 2)).unwrap();
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
  md.add_term(&mim, None, "t*sigma:Grad_Test_u").unwrap();
  md.add_term(&mim, right, "-t*F*Test_u(1)").unwrap();
  let mu = md
    .add_Dirichlet_condition_with_multiplier(&mim, "u", &space(2, 1), left, None)
    .unwrap();
  md.add_term(&mim, None, "t/rho*Grad_V.Grad_Test_V").unwrap();
  let mv1 = md
    .add_Dirichlet_condition_with_multiplier(&mim, "V", &space(1, 1), right, None)
    .unwrap();
  let mv2 = md
    .add_Dirichlet_condition_with_multiplier(&mim, "V", &space(1, 1), left, Some("V_left"))
    .unwrap();
  md.add_term(
    &mim,
    None,
    "t*kappa*Grad_T.Grad_Test_T + 2*D*(T-T_air)*Test_T - t/rho*Norm_sqr(Grad_V)*Test_T",
  )
  .unwrap();
  // convection on top and bottom, region 5
  let top_bottom = mesh
    .region(3)
    .unwrap()
    .merge(&mesh.region(4).unwrap())
    .unwrap();
  mesh.set_region(5, &top_bottom).unwrap();
  md.add_term(&mim, 5, "t*D*(T-T_air)*Test_T").unwrap();
  (md, mim, [mu, mv1, mv2])
}

fn main() {
  let temperature = |md: &mut getfem::model::Model, mim: &getfem::fem::mesh_im::MeshIm| {
    ({
      let ws = md.workspace_mut();
      ws.clear_expressions();
      ws.add_expression("sqr(T)", mim, None).unwrap();
      ws.assembly(0, false).unwrap();
      ws.assembled_potential()
    })
    .sqrt()
  };

  println!("1. global test at the default rtol = 1e-8, verbose:");
  let (mut md, mim, _) = thermo_plate();
  let rep = Newton {
    verbose: true,
    ..Default::default()
  }
  .solve(&mut md, None)
  .unwrap();
  println!(
    "   converged: {}, L2 norm of T {:.6}\n",
    rep.converged(),
    temperature(&mut md, &mim)
  );

  println!("2. global rtol = 1e-20, below the rounding floor of this problem:");
  let (mut md, _, _) = thermo_plate();
  let rep = Newton {
    rtol: 1e-20,
    ..Default::default()
  }
  .solve(&mut md, None)
  .unwrap();
  println!(
    "   {:?} in {} solves: {}\n",
    rep.status,
    rep.iterations,
    rep.hint.unwrap_or_default()
  );

  println!("3. a tolerance per physics, scales from the residuals of run 1:");
  let (mut md, mim, [mu, mv1, mv2]) = thermo_plate();
  let newton = Newton {
    // only the variable tolerances count
    atol: f64::INFINITY,
    variable_tolerances: vec![
      VariableTolerance::new(&["u", &mu], 1e-10 * 5e4),
      VariableTolerance::new(&["T"], 1e-10 * 2e4),
      VariableTolerance::new(&["V", &mv1, &mv2], 1e-8 * 2e4),
    ],
    ..Default::default()
  };
  let rep = newton.solve(&mut md, None).unwrap();
  for (step, r) in rep.residuals_per_tolerance.iter().enumerate() {
    let r: Vec<String> = r.iter().map(|(n, x)| format!("{n} {x:.1e}")).collect();
    println!("   after {step} steps: {}", r.join(", "));
  }
  println!(
    "   converged: {}, L2 norm of T {:.6}",
    rep.converged(),
    temperature(&mut md, &mim)
  );
}
