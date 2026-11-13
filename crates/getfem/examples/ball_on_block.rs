//! Ball on a block: the elastic disc of the example bouncing_ball (centre
//! (0, 20), radius 20, Q2, lifted and dropped under a vertical load)
//! falls on an elastic block [-40, 40] x [-20, 0] (Q2, fixed at its
//! bottom, no load), both lambda = mu = 20. Newmark (beta = 1/4, gamma =
//! 1/2) written out for both bodies, with the previous u0, v0, a0 as data:
//!   (u - u0 - dt v0 - dt^2 (1/2 - beta) a0)/(beta dt^2).Test_u,
//! then a = (u - u0 - dt v0 - dt^2 (1/2 - beta) a0)/(beta dt^2) and
//! v = v0 + dt ((1 - gamma) a0 + gamma a). The contact between the lower
//! half of the disc (slave) and the top of the block (master) by a
//! raytracing transformation along the deformed normal of the disc,
//! n1 = Transformed_unit_vector(Grad_u1, Normal) (Poulios, Renard, Comput.
//! Struct. 153, 2015), the gap
//!   g = (Interpolate(X,ray) + Interpolate(u2,ray) - X - u1).n1,
//! and the penalty force r neg_part(g) n1.
//!
//! Writes both bodies as compressed .vtu slices with .pvd collections
//! (target/showcase/ball_on_block/) and prints the energy, the lowest
//! point of the disc, the deepest point of the block's top and the
//! contact force.

use getfem::fem::integration::integ_descriptor;
use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im::MeshIm;
use getfem::fem::slice::{SliceField, SlicerApplyDeformation, StoredMeshSlice};
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::region::{outer_faces_of_mesh, select_faces_of_normal};
use getfem::geom::regular_mesh::{regular_ball_mesh, regular_mesh};
use getfem::io::vtk::{Format, VtuExport, export_to_pvd};
use getfem::model::{Model, Newton};

// regions: of the disc, of the block
const CONTACT: usize = 1;
const TOP: usize = 1;
const BOTTOM: usize = 2;

pub struct Params {
  /// regular_ball_mesh subdivisions of the disc: core, peel
  pub nsubdiv: [usize; 2],
  /// cells of the block
  pub block: [usize; 2],
  /// initial height of the disc's lowest point above the block
  pub lift: f64,
  pub dt: f64,
  pub t_end: f64,
}

pub struct Step {
  pub t: f64,
  /// kinetic + elastic energy of both bodies + penalty energy - work of
  /// the load
  pub energy: f64,
  /// min of y + u_y over the nodes of the disc
  pub ymin: f64,
  /// min of u_y over the nodes of the block's top
  pub dent: f64,
  /// total contact force
  pub force: f64,
  pub iterations: usize,
}

/// Runs the drop; on_step after each step (and at t = 0) with the model
/// and the two spaces (disc, block).
pub fn simulate(p: &Params, on_step: &mut dyn FnMut(&Step, &Model, &MeshFem, &MeshFem)) {
  let (clambda, cmu, load, penalty) = (20.0, 20.0, 0.1, 200.0);
  let (beta, gamma, dt) = (0.25, 0.5, p.dt);
  let ball = regular_ball_mesh(
    &geotrans_descriptor("GT_QK(2,2)").unwrap(),
    &[0.0, 20.0],
    20.0,
    p.nsubdiv,
    0,
    false,
  )
  .unwrap();
  // the faces of n_y < -0.01
  let lower = (1.0f64 - 1e-4).sqrt().atan2(0.01);
  let outer = outer_faces_of_mesh(&ball);
  ball
    .set_region(
      CONTACT,
      &select_faces_of_normal(&ball, &outer, &[0.0, -1.0], lower).unwrap(),
    )
    .unwrap();
  let block = regular_mesh(
    &geotrans_descriptor("GT_QK(2,1)").unwrap(),
    &p.block,
    &[-40.0, -20.0],
    &[80.0, 20.0],
  )
  .unwrap();
  let outer = outer_faces_of_mesh(&block);
  for (n, id) in [([0.0, 1.0], TOP), ([0.0, -1.0], BOTTOM)] {
    block
      .set_region(
        id,
        &select_faces_of_normal(&block, &outer, &n, 0.01).unwrap(),
      )
      .unwrap();
  }
  let mf1 = MeshFem::new(&ball, 2)
    .with_classical_finite_element(2)
    .unwrap();
  let mf2 = MeshFem::new(&block, 2)
    .with_classical_finite_element(2)
    .unwrap();
  let mim1 = MeshIm::new(&ball).with_integration_method(4).unwrap();
  let mim2 = MeshIm::new(&block).with_integration_method(4).unwrap();
  // the contact on a refined rule
  let mimc = MeshIm::new(&ball)
    .with_integration_method(
      &integ_descriptor("IM_STRUCTURED_COMPOSITE(IM_GAUSS_PARALLELEPIPED(2,4),4)").unwrap(),
    )
    .unwrap();
  let (n1, n2) = (mf1.nb_dof().unwrap(), mf2.nb_dof().unwrap());
  let y1: Vec<f64> = (1..n1)
    .step_by(2)
    .map(|i| mf1.point_of_basic_dof(i).unwrap()[1])
    .collect();
  let top: Vec<usize> = (1..n2)
    .step_by(2)
    .filter(|&i| mf2.point_of_basic_dof(i).unwrap()[1].abs() < 1e-9)
    .collect();

  let mut md = Model::new();
  md.add_fem_variable("u1", &mf1).unwrap();
  md.add_fem_variable("u2", &mf2).unwrap();
  // the disc at rest, lifted: the acceleration of the load alone
  let u10: Vec<f64> = (0..n1)
    .map(|i| if i % 2 == 1 { p.lift } else { 0.0 })
    .collect();
  let a10: Vec<f64> = (0..n1)
    .map(|i| if i % 2 == 1 { -load } else { 0.0 })
    .collect();
  for (b, mf, u0, a0) in [
    ("1", &mf1, u10.clone(), a10),
    ("2", &mf2, vec![0.0; n2], vec![0.0; n2]),
  ] {
    let n = u0.len();
    md.add_fem_data(&format!("u0_{b}"), mf, &u0).unwrap();
    md.add_fem_data(&format!("v0_{b}"), mf, &vec![0.0; n])
      .unwrap();
    md.add_fem_data(&format!("a0_{b}"), mf, &a0).unwrap();
  }
  md.set_variable("u1", &u10).unwrap();
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
  for (b, mim, f) in [("1", &mim1, " - load.Test_u1"), ("2", &mim2, "")] {
    md.add_term(
      mim,
      None,
      &format!(
        "(u{b} - (u0_{b} + dt*v0_{b} + dt*dt*(0.5-beta)*a0_{b}))/(beta*dt*dt).Test_u{b} \
         + (clambda*Div_u{b}*Id(2) + 2*cmu*Sym(Grad_u{b})):Grad_Test_u{b}{f}"
      ),
    )
    .unwrap();
  }
  md.add_Dirichlet_condition_with_multiplier(&mim2, "u2", &mf2, BOTTOM, None)
    .unwrap();
  md.add_raytracing_transformation("ray", 2.0).unwrap();
  md.add_slave_contact_boundary_to_raytracing_transformation("ray", &ball, "u1", CONTACT)
    .unwrap();
  md.add_master_contact_boundary_to_raytracing_transformation("ray", &block, "u2", TOP)
    .unwrap();
  // the deformed normal of the disc
  md.add_macro("n1", "Transformed_unit_vector(Grad_u1, Normal)")
    .unwrap();
  md.add_macro("g", "((Interpolate(X,ray)+Interpolate(u2,ray)-X-u1).n1)")
    .unwrap();
  md.add_term(
    &mimc,
    CONTACT,
    "Interpolate_filter(ray, r*neg_part(g)*(n1.Test_u1 - n1.Interpolate(Test_u2,ray)), 1)",
  )
  .unwrap();

  let record = |md: &mut Model,
                t: f64,
                iterations: usize,
                on_step: &mut dyn FnMut(&Step, &Model, &MeshFem, &MeshFem)| {
    let (u1, u2) = (
      md.variable("u1").unwrap().to_vec(),
      md.variable("u2").unwrap().to_vec(),
    );
    md.set_variable("u0_1", &u1).unwrap();
    md.set_variable("u0_2", &u2).unwrap();
    let mut e =
      |b: &str, mim: &MeshIm| {
        let ws = md.workspace_mut();
        ws.clear_expressions();
        ws.add_expression(&format!(
          "0.5*v0_{b}.v0_{b} + 0.5*(clambda*Div_u0_{b}*Id(2) + 2*cmu*Sym(Grad_u0_{b})):Grad_u0_{b}"
        ),
        mim,
        None,).unwrap();
        ws.assembly(0, false).unwrap();
        ws.assembled_potential()
      };
    let s = Step {
      t,
      energy: e("1", &mim1) + e("2", &mim2) + {
        let ws = md.workspace_mut();
        ws.clear_expressions();
        ws.add_expression(
          "Interpolate_filter(ray, 0.5*r*sqr(neg_part(g)), 1)",
          &mimc,
          CONTACT,
        )
        .unwrap();
        ws.assembly(0, false).unwrap();
        ws.assembled_potential()
      } - {
        let ws = md.workspace_mut();
        ws.clear_expressions();
        ws.add_expression("load.u0_1", &mim1, None).unwrap();
        ws.assembly(0, false).unwrap();
        ws.assembled_potential()
      },
      ymin: y1
        .iter()
        .enumerate()
        .map(|(k, y)| y + u1[2 * k + 1])
        .fold(f64::MAX, f64::min),
      dent: top.iter().map(|&i| u2[i]).fold(0.0, f64::min),
      force: {
        let ws = md.workspace_mut();
        ws.clear_expressions();
        ws.add_expression("Interpolate_filter(ray, r*neg_part(g), 1)", &mimc, CONTACT)
          .unwrap();
        ws.assembly(0, false).unwrap();
        ws.assembled_potential()
      },
      iterations,
    };
    on_step(&s, md, &mf1, &mf2);
  };
  record(&mut md, 0.0, 0, on_step);
  // |R| < 1e-8 (absolute: a relative test stops the impact steps at 1e-8
  // of a large initial residual)
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
    for b in ["1", "2"] {
      let var = |v: &str| md.variable(v).unwrap().to_vec();
      let (u, u0, v0, a0) = (
        var(&format!("u{b}")),
        var(&format!("u0_{b}")),
        var(&format!("v0_{b}")),
        var(&format!("a0_{b}")),
      );
      let a: Vec<f64> = (0..u.len())
        .map(|i| (u[i] - u0[i] - dt * v0[i] - dt * dt * (0.5 - beta) * a0[i]) / (beta * dt * dt))
        .collect();
      let v: Vec<f64> = (0..u.len())
        .map(|i| v0[i] + dt * ((1.0 - gamma) * a0[i] + gamma * a[i]))
        .collect();
      md.set_variable(&format!("v0_{b}"), &v).unwrap();
      md.set_variable(&format!("a0_{b}"), &a).unwrap();
    }
    record(&mut md, k as f64 * dt, rep.iterations, on_step);
  }
}

fn main() {
  let out = format!(
    "{}/../../target/showcase/ball_on_block",
    env!("CARGO_MANIFEST_DIR")
  );
  std::fs::create_dir_all(&out).unwrap();
  let p = Params {
    nsubdiv: [6, 3],
    block: [32, 8],
    lift: 5.0,
    dt: 0.01,
    t_end: 40.0,
  };
  let (mut pvd1, mut pvd2) = (vec![], vec![]);
  let mut next_frame = 0.0;
  println!("     t   energy        disc lowest  block top  contact force  Newton");
  simulate(&p, &mut |s, md, mf1, mf2| {
    if s.t < next_frame - 1e-9 {
      return;
    }
    next_frame += 0.25;
    println!(
      "{:6.2}  {:.6e}  {:10.5}  {:10.5}  {:12.5e}  {:4}",
      s.t, s.energy, s.ymin, s.dent, s.force, s.iterations
    );
    let k = pvd1.len();
    for (name, mf, u, pvd) in [
      ("ball", mf1, "u1", &mut pvd1),
      ("block", mf2, "u2", &mut pvd2),
    ] {
      // the deformed body, sliced (refined 4 times)
      let file = format!("{name}_{k}.vtu");
      let u = md.variable(u).unwrap();
      let field = SliceField::new(mf, u).unwrap();
      let mut def = SlicerApplyDeformation { f: field };
      let m = mf.linked_mesh().snapshot();
      let sl = StoredMeshSlice::build(&m, &mut [&mut def], 4, None).unwrap();
      let mut vtu = VtuExport::new(&format!("{out}/{file}"), Format::Compressed);
      vtu.exporting(&sl).unwrap();
      vtu.write_point_data(mf, u, "u").unwrap();
      vtu.write().unwrap();
      pvd.push((s.t, file));
    }
  });
  for (name, pvd) in [("ball", &pvd1), ("block", &pvd2)] {
    let steps: Vec<(f64, &str)> = pvd.iter().map(|(t, f)| (*t, f.as_str())).collect();
    export_to_pvd(&format!("{out}/{name}.pvd"), &steps).unwrap();
  }
  println!("written: {out}/ball.pvd, {out}/block.pvd");
}
