//! The gas-filled crack pair in 3D (the 2D example gas_filled_crack
//! extruded): a 40 x 10 x 20 mm block, clamped left and right, with two
//! curved cracks, y = 2.5 tanh(x/7) (extruded along z) and
//! y = -2.5 tanh((x - 0.3 z)/7) + 0.15 z (oblique), inside the
//! ellipsoid (x/16)^2 + (y/6)^2 + (z/8)^2 < 1: each crack is inside the
//! block, its front a closed curve. The gas in the cracks (ideal gas,
//! mass mR, temperature T, pressure P) pushes their lips apart and
//! exchanges heat with them; the block exchanges heat with its top and
//! bottom faces. The displacement and the temperature are Q2 on
//! hexahedra, enriched by the Heaviside functions of the cracks and, on
//! the elements the fronts pass through, by the crack tip functions of
//! the plane normal to the front (functions of the level sets); the DOFs
//! without support dropped. The crack is first opened with
//! the pressure fixed, then the top face is heated in five steps from 30
//! C to 130 C. Writes the displacement and temperature on the cut mesh of
//! each step to target/showcase/gas_filled_crack_3d_<step>.vtu. Optional
//! argument: element size [mm] (default 2).

use getfem::fem::crack_front::front_element_dofs;
use getfem::fem::crack_tip_function::{CrackTipFunction, Singular};
use getfem::fem::fem::fem_descriptor;
use getfem::fem::global_function::GlobalFunction;
use getfem::fem::integration::integ_descriptor;
use getfem::fem::interpolation::interpolation;
use getfem::fem::level_set::LevelSet;
use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im_level_set::{CELL_TOLERANCE, IntegrateWhere, MeshImLevelSet};
use getfem::fem::mesh_level_set::MeshLevelSet;
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::mesh::Mesh;
use getfem::geom::region::{outer_faces_of_mesh, select_faces_of_normal};
use getfem::geom::regular_mesh::regular_mesh;
use getfem::gwfl::workspace::Workspace;
use getfem::io::vtk::{Format, VtuExport};
use getfem::model::{Model, Newton};

// parameters
const E0: f64 = 5e2; // [N/mm^2] Young modulus at the reference temperature
const NU: f64 = 0.45; // Poisson ratio
const BETA: f64 = 8e-5; // [1/K] expansion coefficient
const GAMMA: f64 = 0.1; // [1/K] thermal weakening
const THETA0: f64 = 30.0; // [C] reference temperature
const ALPHA: f64 = 0.1; // [mm^2/s] thermal diffusivity
const ALPHA_ENV: f64 = 0.02; // [N/mm/s/K] heat transfer to the environment
const ALPHA_CRACK: f64 = 0.01; // [N/mm/s/K] heat transfer to the gas
const MR: f64 = 287e3 * 5e-9; // [N mm/K] gas constant times mass
const ROOM_TEMP: f64 = 30.0; // [C]
const JUMP_TEMP: f64 = 100.0; // [C] final temperature difference
const L: f64 = 40.0; // [mm]
const H: f64 = 10.0; // [mm]
const W: f64 = 20.0; // [mm]
const R1: f64 = 2.5;
const TILT: f64 = 0.15;
const SHEAR: f64 = 0.3;
// semi-axes of the ellipsoid bounding the cracks
const AX: f64 = 16.0;
const AY: f64 = 6.0;
const AZ: f64 = 8.0;

const LEFT: usize = 101;
const RIGHT: usize = 102;
const BOTTOM: usize = 103;
const TOP: usize = 104;

/// The state after a temperature step.
pub struct Step<'a> {
  /// gas temperature [C] and pressure [MPa]
  pub t: f64,
  pub p: f64,
  /// L2 norms of u and theta
  pub norm_u: f64,
  pub norm_theta: f64,
  pub mf_u: &'a MeshFem,
  pub u: &'a [f64],
  pub mf_theta: &'a MeshFem,
  pub theta: &'a [f64],
  pub mls: &'a MeshLevelSet,
}

/// The values of a GWFL expression in x, y, z at the nodes of the level
/// set ls (as Python's ls.set_values with a string).
fn values(ls: &LevelSet, expr: &str) -> Vec<f64> {
  let mf = ls.mesh_fem();
  let mut ws = Workspace::new();
  for (k, x) in ["x", "y", "z"]
    .iter()
    .enumerate()
    .take(mf.linked_mesh().dim())
  {
    ws.add_macro(x, &format!("X({})", k + 1)).unwrap();
  }
  ws.interpolation(expr, &mf, -1).unwrap()
}

/// the crack surface (primary): extruded for sign > 0, oblique otherwise;
/// the ellipsoid (secondary), scaled to a distance near the front (the
/// crack tip functions take the level sets as coordinates around the
/// front)
fn crack(m: &Mesh, sign: f64) -> LevelSet {
  let (shear, tilt) = if sign > 0.0 {
    (0.0, 0.0)
  } else {
    (SHEAR, TILT)
  };
  let ls = LevelSet::new(m, 2, true).unwrap();
  let p = values(
    &ls,
    &format!("y-({})*tanh((x-{shear}*z)/7)-{tilt}*z", sign * R1),
  );
  let s = values(
    &ls,
    &format!("(sqrt(sqr(x/{AX})+sqr(y/{AY})+sqr(z/{AZ}))-1)*{AX}"),
  );
  ls.set_values(0, &p).unwrap();
  ls.set_values(1, &s).unwrap();
  ls
}

/// Runs the example with elements of size h [mm]; step(k, state) after
/// each temperature step.
pub fn simulate(h: f64, step: &mut dyn FnMut(usize, &Step)) {
  let n = |len: f64| (len / h).round() as usize;
  let mesh = regular_mesh(
    &geotrans_descriptor("GT_QK(3,1)").unwrap(),
    &[n(L), n(H), n(W)],
    &[-L / 2.0, -H / 2.0, -W / 2.0],
    &[L, H, W],
  )
  .unwrap();
  let border = outer_faces_of_mesh(&mesh);
  for (rg, n) in [
    (LEFT, [-1.0, 0.0, 0.0]),
    (RIGHT, [1.0, 0.0, 0.0]),
    (BOTTOM, [0.0, -1.0, 0.0]),
    (TOP, [0.0, 1.0, 0.0]),
  ] {
    mesh
      .set_region(
        rg,
        &select_faces_of_normal(&mesh, &border, &n, 0.5).unwrap(),
      )
      .unwrap();
  }
  let (ls1, ls2) = (crack(&mesh, 1.0), crack(&mesh, -1.0));
  let mut mls = MeshLevelSet::new(&mesh);
  mls.add_level_set(&ls1);
  mls.add_level_set(&ls2);
  mls.adapt().unwrap();

  let mf_pre = MeshFem::new(&mesh, 1)
    .with_finite_element(&fem_descriptor("FEM_QK(3,2)").unwrap())
    .unwrap();
  let mfls = MeshFem::from_level_set(&mls, &mf_pre).unwrap();
  // the singular functions near the tips, on the Q1 partition of unity
  let pu_mf = MeshFem::new(&mesh, 1)
    .with_classical_finite_element(1)
    .unwrap();
  let singular = |ls: &LevelSet| {
    let funcs: Vec<GlobalFunction> = Singular::LEFM
      .iter()
      .map(|&s| GlobalFunction::on_level_set(ls, &CrackTipFunction::crack(s)).unwrap())
      .collect();
    MeshFem::from_global_functions(&mesh, &funcs, None).unwrap()
  };
  // the crack tip functions on the Lagrange DOFs of the front elements
  let mf_sing: Vec<MeshFem> = [&ls1, &ls2]
    .iter()
    .enumerate()
    .map(|(i, ls)| {
      let pu_dofs = front_element_dofs(&mls, i, &pu_mf).unwrap();
      MeshFem::from_product(&pu_mf, &singular(ls), Some(&pu_dofs)).unwrap()
    })
    .collect();
  let base_u = MeshFem::from_sum(&[&mf_sing[0], &mf_sing[1], &mfls]).unwrap();
  let base_theta = MeshFem::from_sum(&[&mf_sing[0], &mf_sing[1], &mfls]).unwrap();

  let mim = MeshImLevelSet::new(&mls, IntegrateWhere::All).unwrap();
  mim
    .set_simplex_im(
      &integ_descriptor("IM_TETRAHEDRON(5)").unwrap(),
      Some(&integ_descriptor("IM_PRODUCT(IM_TRIANGLE(3),IM_GAUSS1D(3))").unwrap()),
      CELL_TOLERANCE,
    )
    .unwrap();
  mim
    .set_integration_method(&integ_descriptor("IM_GAUSS_PARALLELEPIPED(3,4)").unwrap())
    .unwrap();
  mim.adapt().unwrap();
  // the DOFs with support (enriched DOFs without are singular)
  let kept = base_u.dofs_with_support(&mim, 1e-8).unwrap();
  let mf_theta = MeshFem::partial(&base_theta, &kept).unwrap();
  base_u.set_qdim(3);
  let mf_u = MeshFem::partial(
    &base_u,
    &kept
      .iter()
      .flat_map(|&i| (0..3).map(move |c| 3 * i + c))
      .collect(),
  )
  .unwrap();
  let mim_bound: Vec<MeshImLevelSet> = ["a", "b"]
    .iter()
    .map(|c| {
      let mb = MeshImLevelSet::new(&mls, IntegrateWhere::Boundary).unwrap();
      mb.set_level_set_boolean_operations(c).unwrap();
      mb.set_simplex_im(
        &integ_descriptor("IM_TETRAHEDRON(3)").unwrap(),
        None,
        CELL_TOLERANCE,
      )
      .unwrap();
      mb.set_integration_method(1usize).unwrap();
      mb.adapt().unwrap();
      mb
    })
    .collect();
  let mut ws = Workspace::new();
  let surf_crack: f64 = mim_bound
    .iter()
    .map(|mb| {
      ws.clear_expressions();
      ws.add_expression("1", mb, None).unwrap();
      ws.assembly(0, false).unwrap();
      ws.assembled_potential()
    })
    .sum();

  let mut md = Model::new();
  md.add_fem_variable("u", &mf_u).unwrap();
  md.add_fem_variable("theta", &mf_theta).unwrap();
  md.add_fixed_size_variable("P", &[1]).unwrap();
  md.add_fixed_size_variable("T", &[1]).unwrap();
  md.set_variable("T", &[273.0 + ROOM_TEMP]).unwrap();
  for (name, v) in [
    ("surf_crack", surf_crack),
    ("mu0", E0 / (2.0 * (1.0 + NU))),
    ("la0", E0 * NU / ((1.0 - 2.0 * NU) * (1.0 + NU))),
    ("ka0", E0 / (3.0 * (1.0 - 2.0 * NU))),
    ("alpha", ALPHA),
    ("alpha_env", ALPHA_ENV),
    ("alpha_crack", ALPHA_CRACK),
    ("beta", BETA),
    ("gamma", GAMMA),
    ("mR", MR),
    ("theta0", THETA0),
  ] {
    md.add_scalar_data(name, v).unwrap();
  }
  // Lamé coefficients and bulk modulus, weakened by the temperature
  for (p1, p2) in [("la", "la0"), ("mu", "mu0"), ("ka", "ka0")] {
    md.add_macro(p1, &format!("{p2}/(1+gamma*(theta-theta0))"))
      .unwrap();
  }
  // deformation gradients, in the block and on both lips
  md.add_macro("F", "Id(3)+Grad_u").unwrap();
  md.add_macro("Fm", "Id(3)+Xfem_minus(Grad_u)").unwrap();
  md.add_macro("Fp", "Id(3)+Xfem_plus(Grad_u)").unwrap();
  md.add_macro("Nanson(FF)", "Det(FF)*Norm(Inv(FF')*Normal)")
    .unwrap();
  md.add_macro("th", "exp(beta*(theta-theta0))").unwrap();
  // Green-Lagrange strain, second Piola-Kirchhoff stress (elastic and
  // thermal expansion parts)
  md.add_macro("E", "(Grad_u+Grad_u'+Grad_u'*Grad_u)*0.5")
    .unwrap();
  md.add_macro(
    "S",
    "((la*Trace(E)*Id(3)+2*mu*E)/th-1.5*(th-1/th)*ka*Id(3))",
  )
  .unwrap();
  // elasticity, heat diffusion in the deformed body
  md.add_nonlinear_term(&mim, None, "(F*S):Grad_Test_u")
    .unwrap();
  md.add_nonlinear_term(
    &mim,
    None,
    "alpha*Det(F)*(Inv(F')*Grad_theta).(Inv(F')*Grad_Test_theta)",
  )
  .unwrap();
  for mb in &mim_bound {
    // ideal gas law
    md.add_nonlinear_term(
      mb,
      None,
      "(P*((Xfem_plus(u)-Xfem_minus(u)).Normalized(Inv(Fm')*Normal)+1e-6)-mR*T/surf_crack)*Test_P",
    )
    .unwrap();
    // heat flux equilibrium of the gas
    md.add_nonlinear_term(
      mb,
      None,
      "(Nanson(Fm)*(T-273-Xfem_minus(theta))+Nanson(Fp)*(T-273-Xfem_plus(theta)))*Test_T",
    )
    .unwrap();
    // heat exchange of the lips with the gas
    md.add_nonlinear_term(
      mb,
      None,
      "-alpha_crack*(Nanson(Fm)*(T-273-Xfem_minus(theta))*Xfem_minus(Test_theta)\
       +Nanson(Fp)*(T-273-Xfem_plus(theta))*Xfem_plus(Test_theta))",
    )
    .unwrap();
  }
  // heat exchange at the bottom and the top
  for (var, rg) in [("theta_B", BOTTOM), ("theta_T", TOP)] {
    md.add_scalar_data(var, ROOM_TEMP).unwrap();
    md.add_nonlinear_term(
      &mim,
      Some(&mesh.region(rg).unwrap()),
      &format!("-alpha_env*Nanson(F)*({var}-theta)*Test_theta"),
    )
    .unwrap();
  }
  // follower pressure on the lips
  for mb in &mim_bound {
    md.add_nonlinear_term(mb, None, "(P*Det(Fm)*Inv(Fm')*Normal).Xfem_minus(Test_u)")
      .unwrap();
    md.add_nonlinear_term(mb, None, "(-P*Det(Fp)*Inv(Fp')*Normal).Xfem_plus(Test_u)")
      .unwrap();
  }
  // clamped left and right
  md.add_Dirichlet_condition_with_multiplier(&mim, "u", 1usize, mesh.region(LEFT).unwrap(), None)
    .unwrap();
  md.add_Dirichlet_condition_with_multiplier(&mim, "u", 1usize, mesh.region(RIGHT).unwrap(), None)
    .unwrap();

  // open the lips with the pressure fixed
  md.disable_variable("P").unwrap();
  md.set_variable("P", &[1e-2]).unwrap();
  let newton = |atol| Newton {
    atol,
    max_iter: 100,
    ..Default::default()
  };
  newton(5e-4).solve(&mut md, None).unwrap();
  md.enable_variable("P").unwrap();

  // heat the top face
  for (k, fact) in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0].iter().enumerate() {
    md.set_variable("theta_T", &[ROOM_TEMP + fact * JUMP_TEMP])
      .unwrap();
    newton(5e-8).solve(&mut md, None).unwrap();
    let (u, theta) = (md.variable("u").unwrap(), md.variable("theta").unwrap());
    let mut ws = Workspace::new();
    ws.add_fem_data("u", &mf_u, u).unwrap();
    ws.add_fem_data("theta", &mf_theta, theta).unwrap();
    let s = Step {
      t: md.variable("T").unwrap()[0] - 273.0,
      p: md.variable("P").unwrap()[0],
      norm_u: {
        ws.clear_expressions();
        ws.add_expression("Norm_sqr(u)", &mim, None).unwrap();
        ws.assembly(0, false).unwrap();
        ws.assembled_potential()
      }
      .sqrt(),
      norm_theta: {
        ws.clear_expressions();
        ws.add_expression("Norm_sqr(theta)", &mim, None).unwrap();
        ws.assembly(0, false).unwrap();
        ws.assembled_potential()
      }
      .sqrt(),
      mf_u: &mf_u,
      u,
      mf_theta: &mf_theta,
      theta,
      mls: &mls,
    };
    step(k, &s);
  }
}

fn main() {
  let h: f64 = std::env::args().nth(1).map_or(2.0, |a| a.parse().unwrap());
  let out = format!("{}/../../target/showcase", env!("CARGO_MANIFEST_DIR"));
  std::fs::create_dir_all(&out).unwrap();
  simulate(h, &mut |k, s| {
    println!(
      "step {k}: gas temperature {:.4} C, pressure {:.6} MPa",
      s.t, s.p
    );
    // the solution on the cut mesh, discontinuous across the cracks
    // (alpha 0.001: the nodes just off the cracks, each side its value)
    let cut = s.mls.global_cut_mesh().unwrap();
    let mfv = MeshFem::new(&cut, 3)
      .with_classical_discontinuous_finite_element((2, 0.001))
      .unwrap();
    let mfw = MeshFem::new(&cut, 1)
      .with_classical_discontinuous_finite_element((2, 0.001))
      .unwrap();
    let v = interpolation(s.mf_u, &mfv, s.u).unwrap();
    let w = interpolation(s.mf_theta, &mfw, s.theta).unwrap();
    let mut vtu = VtuExport::new(
      &format!("{out}/gas_filled_crack_3d_{k}.vtu"),
      Format::Compressed,
    );
    vtu.exporting(&mfv).unwrap();
    vtu.write_point_data(&mfv, &v, "u").unwrap();
    vtu.write_point_data(&mfw, &w, "temperature").unwrap();
    vtu.write().unwrap();
  });
}
