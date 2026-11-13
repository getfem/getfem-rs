//! Viscoelastic ball: the disc of the example bouncing_ball (centre
//! (0, 20), radius 20, Q2, lifted and dropped under a vertical load 0.1
//! on a rigid ground, Newmark with beta = 1/4, gamma = 1/2, the penalty
//! potential r neg_part(y + u_y)^2), made of the large strain quasilinear
//! viscoelastic material of GetFEM's contrib/continuum_mechanics/
//! QLV_viscoelasticity.py (K. Poulios, QLV notes, 2021): the Horgan-Murphy
//! law (gamma = 1/6), Prony relaxation of the hydrostatic and deviatoric
//! stresses to rH = rD = 0.8 of the instantaneous moduli in tauH = tauD =
//! 1, the elastic and viscous stresses of the previous step stored at the
//! integration points (all four evaluated, then stored). Plane strain: F =
//! E (Id(2) + Grad_u) E' + e3 x e3 with E the embedding of the plane, the
//! virtual work E' (F S) E : Grad_Test_u. Instantaneous moduli lambda = mu
//! = 200 (ten times bouncing_ball's: a softer disc does not lift off once
//! the viscous stresses relax), penalty 1000.
//!
//! Writes the deformed disc as compressed .vtu slices with a .pvd
//! collection (target/showcase/viscoelastic_ball/) and prints the kinetic
//! energy, the lowest point, the mean vertical displacement and the
//! contact force: each bounce loses height to the viscous dissipation.

use getfem::fem::im_data::ImData;
use getfem::fem::integration::integ_descriptor;
use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im::MeshIm;
use getfem::fem::slice::{SliceField, SlicerApplyDeformation, StoredMeshSlice};
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::region::{outer_faces_of_mesh, select_faces_of_normal};
use getfem::geom::regular_mesh::regular_ball_mesh;
use getfem::io::vtk::{Format, VtuExport, export_to_pvd};
use getfem::model::{Model, Newton};

const CONTACT: usize = 1;

pub struct Params {
  /// regular_ball_mesh subdivisions: core, peel
  pub nsubdiv: [usize; 2],
  /// initial height of the lowest point
  pub lift: f64,
  pub dt: f64,
  pub t_end: f64,
}

pub struct Step {
  pub t: f64,
  pub kinetic: f64,
  /// min of y + u_y over the nodes
  pub ymin: f64,
  /// mean vertical displacement
  pub uc: f64,
  /// total contact force
  pub force: f64,
  pub iterations: usize,
}

/// Runs the drop; on_step after each step (and at t = 0).
pub fn simulate(p: &Params, on_step: &mut dyn FnMut(&Step, &Model)) {
  let (lambda, mu, load, penalty) = (200.0, 200.0, 0.1, 1000.0);
  let kappa = lambda + 2.0 * mu / 3.0;
  let (tau_h, tau_d, r_h, r_d, gamma_hm) = (1.0, 1.0, 0.8, 0.8, 1.0 / 6.0);
  let (beta, gamma, dt) = (0.25, 0.5, p.dt);
  let gt = geotrans_descriptor("GT_QK(2,2)").unwrap();
  let m = regular_ball_mesh(&gt, &[0.0, 20.0], 20.0, p.nsubdiv, 0, false).unwrap();
  let border = outer_faces_of_mesh(&m);
  // the faces of n_y < -0.01
  let angle = (1.0f64 - 1e-4).sqrt().atan2(0.01);
  m.set_region(
    CONTACT,
    &select_faces_of_normal(&m, &border, &[0.0, -1.0], angle).unwrap(),
  )
  .unwrap();
  let mf = MeshFem::new(&m, 2)
    .with_classical_finite_element(2)
    .unwrap();
  let mim = MeshIm::new(&m).with_integration_method(4).unwrap();
  // the contact on a refined rule
  let mimc = MeshIm::new(&m)
    .with_integration_method(
      &integ_descriptor("IM_STRUCTURED_COMPOSITE(IM_GAUSS_PARALLELEPIPED(2,4),4)").unwrap(),
    )
    .unwrap();
  let imd = ImData::new(&mim, -1, &[3, 3]).unwrap();
  let n = mf.nb_dof().unwrap();
  let y: Vec<f64> = (1..n)
    .step_by(2)
    .map(|i| mf.point_of_basic_dof(i).unwrap()[1])
    .collect();

  let mut md = Model::new();
  md.add_fem_variable("u", &mf).unwrap();
  let u0: Vec<f64> = (0..n)
    .map(|i| if i % 2 == 1 { p.lift } else { 0.0 })
    .collect();
  // at rest, the rigid lift: the acceleration of the load alone
  let a0: Vec<f64> = (0..n)
    .map(|i| if i % 2 == 1 { -load } else { 0.0 })
    .collect();
  md.add_fem_data("u0", &mf, &u0).unwrap();
  md.add_fem_data("v0", &mf, &vec![0.0; n]).unwrap();
  md.add_fem_data("a0", &mf, &a0).unwrap();
  md.set_variable("u", &u0).unwrap();
  for (k, v) in [
    ("kappa", kappa),
    ("mu", mu),
    ("gamma", gamma_hm),
    ("tauH", tau_h),
    ("tauD", tau_d),
    ("rH", r_h),
    ("rD", r_d),
    ("r", penalty),
    ("beta", beta),
    ("dt", dt),
  ] {
    md.add_scalar_data(k, v).unwrap();
  }
  md.add_fixed_size_data("load", &[2], &[0.0, -load]).unwrap();
  // the stresses of the previous step, at the integration points
  let zero = vec![0.0; imd.nb_points().unwrap() * 9];
  for k in ["SeHprev", "SeDprev", "SvHprev", "SvDprev"] {
    md.add_im_data(k, &imd, &zero).unwrap();
  }
  for (k, v) in [
    ("E", "[1,0;0,1;0,0]"),
    ("F", "E*(Id(2)+Grad_u)*E' + [0,0,0;0,0,0;0,0,1]"),
    ("Sqr(a)", "a*a"),
    ("WW1", "mu*(0.5+gamma)"),
    ("WW2", "mu*(0.5-gamma)"),
    (
      "WW3",
      "kappa*(1-1/Det(F)) - 2*WW2*pow(Det(F),-2/3) - WW1*pow(Det(F),-4/3)",
    ),
    ("I1", "Norm_sqr(F)"),
    ("I2", "Sqr(Det(F))*Norm_sqr(Inv(F))"),
    ("I3", "Sqr(Det(F))"),
    (
      "SeH",
      "(1/3*WW1*I1 + 2/3*WW2*I2 + WW3*I3)*Inv(Right_Cauchy_Green(F))",
    ),
    (
      "SeD",
      "WW1*Id(3) + 1/3*(WW2*I2-WW1*I1)*Inv(Right_Cauchy_Green(F)) - WW2*I3*Sqr(Inv(Right_Cauchy_Green(F)))",
    ),
    (
      "SvH",
      "exp(-dt/tauH)*SvHprev - (1-rH)*(tauH/dt-(dt+tauH)/dt*exp(-dt/tauH))*SeHprev - (1-rH)*((dt-tauH)/dt+tauH/dt*exp(-dt/tauH))*SeH",
    ),
    (
      "SvD",
      "exp(-dt/tauD)*SvDprev - (1-rD)*(tauD/dt-(dt+tauD)/dt*exp(-dt/tauD))*SeDprev - (1-rD)*((dt-tauD)/dt+tauD/dt*exp(-dt/tauD))*SeD",
    ),
    ("pred", "(u0 + dt*v0 + dt*dt*(0.5-beta)*a0)"),
  ] {
    md.add_macro(k, v).unwrap();
  }
  md.add_term(
    &mim,
    None,
    "(u - pred)/(beta*dt*dt).Test_u + (E'*(F*(SeH+SvH+SeD+SvD))*E):Grad_Test_u - load.Test_u",
  )
  .unwrap();
  md.add_term(&mimc, CONTACT, "r*sqr(neg_part(X(2) + u(2)))")
    .unwrap();

  let record =
    |md: &mut Model, t: f64, iterations: usize, on_step: &mut dyn FnMut(&Step, &Model)| {
      let u = md.variable("u").unwrap().to_vec();
      let s = Step {
        t,
        kinetic: {
          let ws = md.workspace_mut();
          ws.clear_expressions();
          ws.add_expression("0.5*v0.v0", &mim, None).unwrap();
          ws.assembly(0, false).unwrap();
          ws.assembled_potential()
        },
        ymin: y
          .iter()
          .enumerate()
          .map(|(k, y)| y + u[2 * k + 1])
          .fold(f64::MAX, f64::min),
        uc: {
          let ws = md.workspace_mut();
          ws.clear_expressions();
          ws.add_expression("u(2)", &mim, None).unwrap();
          ws.assembly(0, false).unwrap();
          ws.assembled_potential()
        } / {
          let ws = md.workspace_mut();
          ws.clear_expressions();
          ws.add_expression("1", &mim, None).unwrap();
          ws.assembly(0, false).unwrap();
          ws.assembled_potential()
        },
        force: {
          let ws = md.workspace_mut();
          ws.clear_expressions();
          ws.add_expression("2*r*neg_part(X(2) + u(2))", &mimc, CONTACT)
            .unwrap();
          ws.assembly(0, false).unwrap();
          ws.assembled_potential()
        },
        iterations,
      };
      on_step(&s, md);
    };
  record(&mut md, 0.0, 0, on_step);
  // |R| < 1e-8 (a relative test stops the impact steps at 1e-8 of a large
  // initial residual)
  let newton = Newton {
    atol: 1e-8,
    rtol: 0.0,
    ..Default::default()
  };
  let nt = (p.t_end / dt).round() as usize;
  for k in 1..=nt {
    let rep = newton.solve(&mut md, None).unwrap();
    assert!(
      rep.converged(),
      "step {k}: {:?} {}",
      rep.status,
      rep.message
    );
    // the stresses of this step, then stored as the previous ones
    let h: Vec<Vec<f64>> = ["SeH", "SeD", "SvH", "SvD"]
      .iter()
      .map(|e| md.interpolation(e, &imd, -1).unwrap())
      .collect();
    for (k, v) in ["SeHprev", "SeDprev", "SvHprev", "SvDprev"].iter().zip(&h) {
      md.set_variable(k, v).unwrap();
    }
    let var = |v: &str| md.variable(v).unwrap().to_vec();
    let (u, u0, v0, a0) = (var("u"), var("u0"), var("v0"), var("a0"));
    let a: Vec<f64> = (0..n)
      .map(|i| (u[i] - u0[i] - dt * v0[i] - dt * dt * (0.5 - beta) * a0[i]) / (beta * dt * dt))
      .collect();
    let v: Vec<f64> = (0..n)
      .map(|i| v0[i] + dt * ((1.0 - gamma) * a0[i] + gamma * a[i]))
      .collect();
    md.set_variable("u0", &u).unwrap();
    md.set_variable("v0", &v).unwrap();
    md.set_variable("a0", &a).unwrap();
    record(&mut md, k as f64 * dt, rep.iterations, on_step);
  }
}

fn main() {
  let out = format!(
    "{}/../../target/showcase/viscoelastic_ball",
    env!("CARGO_MANIFEST_DIR")
  );
  std::fs::create_dir_all(&out).unwrap();
  let p = Params {
    nsubdiv: [6, 3],
    lift: 5.0,
    dt: 0.02,
    t_end: 60.0,
  };
  let mut pvd = vec![];
  let mut next_frame = 0.0;
  println!("     t   kinetic       lowest point  mean u_y   contact force  Newton");
  simulate(&p, &mut |s, md| {
    if s.t < next_frame - 1e-9 {
      return;
    }
    next_frame += 0.25;
    println!(
      "{:6.2}  {:.6e}  {:10.5}  {:10.5}  {:12.5e}  {:4}",
      s.t, s.kinetic, s.ymin, s.uc, s.force, s.iterations
    );
    // the deformed disc, sliced (refined 4 times)
    let mf = md.mesh_fem_of_variable("u").unwrap();
    let u = md.variable("u").unwrap();
    let field = SliceField::new(&mf, u).unwrap();
    let mut def = SlicerApplyDeformation { f: field };
    let m = mf.linked_mesh().snapshot();
    let sl = StoredMeshSlice::build(&m, &mut [&mut def], 4, None).unwrap();
    let file = format!("ball_{}.vtu", pvd.len());
    let mut vtu = VtuExport::new(&format!("{out}/{file}"), Format::Compressed);
    vtu.exporting(&sl).unwrap();
    vtu.write_point_data(&mf, u, "u").unwrap();
    vtu.write().unwrap();
    pvd.push((s.t, file));
  });
  let steps: Vec<(f64, &str)> = pvd.iter().map(|(t, f)| (*t, f.as_str())).collect();
  export_to_pvd(&format!("{out}/ball.pvd"), &steps).unwrap();
  println!("written: {out}/ball.pvd");
}
