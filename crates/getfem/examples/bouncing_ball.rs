//! Bouncing ball: an elastic disc (centre (0, 20), radius 20, Q2) lifted
//! above a rigid ground y = 0 and dropped under a vertical load, the 2D
//! case of GetFEM's demo_dynamic_contact.m (its parameters: lambda = mu =
//! 20, the load 0.1, dt = 0.01, penalty 100). Newmark (beta = 1/4,
//! gamma = 1/2) written out: with the previous u0, v0, a0 as data, the
//! mass term on the prediction,
//!   (u - u0 - dt v0 - dt^2 (1/2 - beta) a0)/(beta dt^2).Test_u,
//! then a = (u - u0 - dt v0 - dt^2 (1/2 - beta) a0)/(beta dt^2) and
//! v = v0 + dt ((1 - gamma) a0 + gamma a); the contact by the penalty
//! potential r neg_part(y + u_y)^2 on the lower half of the boundary.
//!
//! Writes the deformed disc as compressed .vtu slices with a .pvd
//! collection (target/showcase/bouncing_ball/) and prints the energy, the
//! lowest point, the mean vertical displacement and the contact force.

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
  /// kinetic + elastic energy - work of the load
  pub energy: f64,
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
  let (clambda, cmu, load, penalty) = (20.0, 20.0, 0.1, 100.0);
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
  // the contact on a refined rule (the demo's)
  let mimc = MeshIm::new(&m)
    .with_integration_method(
      &integ_descriptor("IM_STRUCTURED_COMPOSITE(IM_GAUSS_PARALLELEPIPED(2,4),4)").unwrap(),
    )
    .unwrap();
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
    ("clambda", clambda),
    ("cmu", cmu),
    ("r", penalty),
    ("beta", beta),
    ("dt", dt),
  ] {
    md.add_scalar_data(k, v).unwrap();
  }
  md.add_fixed_size_data("load", &[2], &[0.0, -load]).unwrap();
  md.add_macro("pred", "(u0 + dt*v0 + dt*dt*(0.5-beta)*a0)")
    .unwrap();
  md.add_term(
    &mim,
    None,
    "(u - pred)/(beta*dt*dt).Test_u + (clambda*Div_u*Id(2) + 2*cmu*Sym(Grad_u)):Grad_Test_u - load.Test_u",
  )
  .unwrap();
  md.add_term(&mimc, CONTACT, "r*sqr_neg_part(X(2) + u(2))")
    .unwrap();

  let record =
    |md: &mut Model, t: f64, iterations: usize, on_step: &mut dyn FnMut(&Step, &Model)| {
      let u = md.variable("u").unwrap().to_vec();
      md.set_variable("u0", &u).unwrap();
      let s = Step {
        t,
        energy: {
          let ws = md.workspace_mut();
          ws.clear_expressions();
          ws.add_expression(
            "0.5*v0.v0 + 0.5*(clambda*Div_u0*Id(2) + 2*cmu*Sym(Grad_u0)):Grad_u0 - load.u0",
            &mim,
            None,
          )
          .unwrap();
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
  // |R| < 1e-8, the demo's (the default relative test stops each impact
  // step at 1e-8 of a large initial residual)
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
    let var = |v: &str| md.variable(v).unwrap().to_vec();
    let (u, u0, v0, a0) = (var("u"), var("u0"), var("v0"), var("a0"));
    let a: Vec<f64> = (0..n)
      .map(|i| (u[i] - u0[i] - dt * v0[i] - dt * dt * (0.5 - beta) * a0[i]) / (beta * dt * dt))
      .collect();
    let v: Vec<f64> = (0..n)
      .map(|i| v0[i] + dt * ((1.0 - gamma) * a0[i] + gamma * a[i]))
      .collect();
    md.set_variable("v0", &v).unwrap();
    md.set_variable("a0", &a).unwrap();
    record(&mut md, k as f64 * dt, rep.iterations, on_step);
  }
}

fn main() {
  let out = format!(
    "{}/../../target/showcase/bouncing_ball",
    env!("CARGO_MANIFEST_DIR")
  );
  std::fs::create_dir_all(&out).unwrap();
  let p = Params {
    nsubdiv: [6, 3],
    lift: 5.0,
    dt: 0.01,
    t_end: 40.0,
  };
  let mut pvd = vec![];
  let mut next_frame = 0.0;
  println!("     t   energy      lowest point  mean u_y   contact force  Newton");
  simulate(&p, &mut |s, md| {
    if s.t < next_frame - 1e-9 {
      return;
    }
    next_frame += 0.5;
    println!(
      "{:6.2}  {:.6e}  {:10.5}  {:10.5}  {:12.5e}  {:4}",
      s.t, s.energy, s.ymin, s.uc, s.force, s.iterations
    );
    let mf = md.mesh_fem_of_variable("u").unwrap();
    let field = SliceField::new(&mf, md.variable("u").unwrap()).unwrap();
    let mut def = SlicerApplyDeformation { f: field };
    let m = mf.linked_mesh().snapshot();
    let sl = StoredMeshSlice::build(&m, &mut [&mut def], 4, None).unwrap();
    let file = format!("ball_{}.vtu", pvd.len());
    let mut vtu = VtuExport::new(&format!("{out}/{file}"), Format::Compressed);
    vtu.exporting(&sl).unwrap();
    vtu
      .write_point_data(&mf, md.variable("u").unwrap(), "u")
      .unwrap();
    vtu.write().unwrap();
    pvd.push((s.t, file));
  });
  let steps: Vec<(f64, &str)> = pvd.iter().map(|(t, f)| (*t, f.as_str())).collect();
  export_to_pvd(&format!("{out}/ball.pvd"), &steps).unwrap();
  println!("written: {out}/ball.pvd");
}
