//! Two rubber rings (a GetFEM user's script): rings of radius R = 50
//! and cross section radius r = 5, a structured disc (GT_QK(2,2))
//! extruded into 26 layers of second order hexahedra around the
//! circumference, FEM_Q2_INCOMPLETE(3), a compressible neo-Hookean
//! material (E = 1e5, nu = 0.45):
//!   tau = K log(J) Id + G J^(-2/3) Deviator(F F'),  P = tau F^-T.
//! Each ring has two handles (the outer faces at x = -R and x = R), held
//! by multipliers q (u = p on the handle) and by elastic supports. Ring 1
//! is the master, ring 2 the slave of a raytracing contact along the
//! deformed normal of ring 2, with the penalty c = 1e3:
//!   g = (Interpolate(X,raytrace) + Interpolate(u1,raytrace) - X - u2).n2.
//! Phases: ring 2 is pulled through ring 1 by both handles (30 steps), the
//! handle 2 of ring 1 and the handle 1 of ring 2 are released (8 steps, the
//! supports decreasing to 0), then ring 2 is pulled by its handle 2 alone
//! (16 steps; the script's later steps make Newton cycle between contact
//! states, in GetFEM too).
//!
//! Writes both rings per step (u and the von Mises stress of tau) as .vtu
//! with .pvd collections (target/showcase/two_rings/) and prints the Newton
//! iterations, the largest displacements and the contact force.

use getfem::fem::fem::fem_descriptor;
use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im::MeshIm;
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::mesh::{Mesh, extrude};
use getfem::geom::region::{outer_faces_of_mesh, select_faces_in_box};
use getfem::geom::regular_mesh::regular_ball_mesh;
use getfem::io::vtk::{Format, VtuExport, export_to_pvd};
use getfem::model::{Model, Newton};
use std::f64::consts::PI;

const EXTERIOR: usize = 1;
const HANDLE1: usize = 2;
const HANDLE2: usize = 3;

fn main() {
  let (rr, r) = (50.0, 5.0);
  let (nc, nr1, nr2) = (26, 2, 2);
  let (e, nu) = (1e5, 0.45);
  let c_penal = 1e-2 * e;

  // ring 1: the extruded disc wrapped around the y axis, its ends merged
  let disc = regular_ball_mesh(
    &geotrans_descriptor("GT_QK(2,2)").unwrap(),
    &[0.0, 0.0],
    r,
    [nr1, nr2],
    0,
    false,
  )
  .unwrap();
  let m1 = extrude(&disc, nc, 2).unwrap();
  let moves: Vec<(usize, Vec<f64>)> = m1
    .point_index()
    .into_iter()
    .map(|i| {
      let p = m1.point(i);
      let (s, c) = (2.0 * PI * p[2]).sin_cos();
      (i, vec![(rr + p[0]) * c, p[1], (rr + p[0]) * s])
    })
    .collect();
  m1.set_points(&moves).unwrap();
  m1.merge_points(0.0, -1).unwrap();
  let outer = outer_faces_of_mesh(&m1);
  m1.set_region(EXTERIOR, &outer).unwrap();
  for (id, x) in [(HANDLE1, -rr), (HANDLE2, rr)] {
    let lo = [x - 1.1 * r, -1.1 * r, -0.3 * rr];
    let hi = [x + 1.1 * r, 1.1 * r, 0.3 * rr];
    m1.set_region(id, &select_faces_in_box(&m1, &outer, &lo, &hi).unwrap())
      .unwrap();
  }
  m1.optimize_structure();
  // ring 2: a copy, shifted to interlock
  let m2 = Mesh::from((*m1.snapshot()).clone());
  m2.translate(&[-rr, 2.5 * r, 0.0]).unwrap();

  let pf = fem_descriptor("FEM_Q2_INCOMPLETE(3)").unwrap();
  let mf1 = MeshFem::new(&m1, 3).with_finite_element(&pf).unwrap();
  let mf2 = MeshFem::new(&m2, 3).with_finite_element(&pf).unwrap();
  // von Mises stress (of tau) at the nodes, for the output
  let mfs1 = MeshFem::new(&m1, 1).with_finite_element(&pf).unwrap();
  let mfs2 = MeshFem::new(&m2, 1).with_finite_element(&pf).unwrap();
  let mim1 = MeshIm::new(&m1).with_integration_method(5).unwrap();
  let mim2 = MeshIm::new(&m2).with_integration_method(5).unwrap();
  let mimc2 = MeshIm::new(&m2).with_integration_method(7).unwrap();

  let mut md = Model::new();
  md.add_fem_variable("u1", &mf1).unwrap();
  md.add_fem_variable("u2", &mf2).unwrap();
  md.add_scalar_data("K", e / (3.0 * (1.0 - 2.0 * nu)))
    .unwrap();
  md.add_scalar_data("G", e / (2.0 * (1.0 + nu))).unwrap();
  md.add_macro("F(u)", "Id(3)+Grad_u").unwrap();
  md.add_macro(
    "tau(u)",
    "K*log(Det(F(u)))*Id(3)+G*pow(Det(F(u)),-2/3)*Deviator(F(u)*F(u)')",
  )
  .unwrap();
  md.add_nonlinear_term(&mim1, None, "(tau(u1)*Inv(F(u1)')):Grad_Test_u1")
    .unwrap();
  md.add_nonlinear_term(&mim2, None, "(tau(u2)*Inv(F(u2)')):Grad_Test_u2")
    .unwrap();

  // handles: multipliers q and elastic supports (against twisting)
  for q in ["q11", "q12", "q21", "q22"] {
    md.add_fixed_size_variable(q, &[3]).unwrap();
    md.add_fixed_size_data(&q.replace('q', "p"), &[3], &[0.0; 3])
      .unwrap();
  }
  md.add_scalar_data("penal", 0.0).unwrap();
  for (mim, u, h, q, support) in [
    (&mim1, "u1", HANDLE1, "11", "1e4"),
    (&mim1, "u1", HANDLE2, "12", "penal"),
    (&mim2, "u2", HANDLE1, "21", "penal"),
    (&mim2, "u2", HANDLE2, "22", "1e4"),
  ] {
    md.add_linear_term(mim, h, &format!("({u}-p{q}).q{q}"))
      .unwrap();
    md.add_linear_term(mim, h, &format!("{support}*({u}-p{q}).Test_{u}"))
      .unwrap();
  }

  // contact: ring 2 (slave) on ring 1 (master)
  md.add_scalar_data("c", c_penal).unwrap();
  md.add_raytracing_transformation("raytrace", r).unwrap();
  md.add_master_contact_boundary_to_raytracing_transformation("raytrace", &m1, "u1", EXTERIOR)
    .unwrap();
  md.add_slave_contact_boundary_to_raytracing_transformation("raytrace", &m2, "u2", EXTERIOR)
    .unwrap();
  md.add_macro("normal2", "Normalized(Inv(F(u2)').Normal)")
    .unwrap();
  md.add_macro(
    "gap2",
    "(Interpolate(X,raytrace)+Interpolate(u1,raytrace)-X-u2).normal2",
  )
  .unwrap();
  md.add_nonlinear_term(
    &mimc2,
    EXTERIOR,
    "Interpolate_filter(raytrace, -c*neg_part(gap2)*(Interpolate(Test_u1,raytrace)-Test_u2).normal2, 1)",
  )
  .unwrap();
  println!(
    "{} hexahedra per ring, {} dofs",
    m1.nb_convexes(),
    md.nb_dof()
  );

  let out = format!(
    "{}/../../target/showcase/two_rings",
    env!("CARGO_MANIFEST_DIR")
  );
  std::fs::create_dir_all(&out).unwrap();
  let (mut pvd1, mut pvd2) = (vec![], vec![]);
  // |R| <= 1e-8 of its start, or of the force E r^2 (GetFEM: of the
  // external load; the release steps start at a residual too small for
  // a relative test)
  let newton = Newton {
    atol: 1e-8 * e * r * r,
    ..Default::default()
  };
  let ramp = |t: f64| 1.8 * rr * (0.0f64).max((t - 0.4) / 0.6).powi(2);
  let mut t = 0.0;
  println!("step      t  Newton  max|u1|  max|u2|  contact force");
  for step in 0..55 {
    if step <= 30 {
      t = step as f64 / 30.0;
      let y = -5.0 * r * (t / 0.4).min(1.0);
      md.set_variable("p21", &[1.8 * rr * t.max((4.0 * t).min(0.2)), y, 0.0])
        .unwrap();
      md.set_variable("p22", &[ramp(t), y, 0.0]).unwrap();
    } else if step <= 38 {
      if step == 31 {
        for q in ["q12", "q21"] {
          md.disable_variable(q).unwrap();
          md.set_variable(q, &[0.0; 3]).unwrap();
        }
      }
      let penal = [1e2, 3e1, 1e1, 3e0, 1e0, 1e-1, 1e-3, 0.0][step - 31];
      md.set_variable("penal", &[penal]).unwrap();
    } else {
      t = 1.0 + (step - 38) as f64 / 100.0;
      md.set_variable("p22", &[ramp(t), -5.0 * r, 0.0]).unwrap();
    }
    let rep = newton.solve(&mut md, None).unwrap();
    assert!(
      rep.converged(),
      "step {step}: {:?} {}",
      rep.status,
      rep.message
    );
    let force = {
      let ws = md.workspace_mut();
      ws.clear_expressions();
      ws.add_expression(
        "Interpolate_filter(raytrace, c*neg_part(gap2), 1)",
        &mimc2,
        EXTERIOR,
      )
      .unwrap();
      ws.assembly(0, false).unwrap();
      ws.assembled_potential()
    };
    let umax = |u: &str| {
      md.variable(u)
        .unwrap()
        .iter()
        .fold(0.0f64, |a, v| a.max(v.abs()))
    };
    println!(
      "{step:4}  {t:5.3}  {:6}  {:7.3}  {:7.3}  {force:13.5e}",
      rep.iterations,
      umax("u1"),
      umax("u2")
    );
    for (name, mf, mfs, u, pvd) in [
      ("ring1", &mf1, &mfs1, "u1", &mut pvd1),
      ("ring2", &mf2, &mfs2, "u2", &mut pvd2),
    ] {
      let file = format!("{name}_{step}.vtu");
      let vm = md
        .interpolation(&format!("sqrt(3/2)*Norm(Deviator(tau({u})))"), mfs, None)
        .unwrap();
      let mut vtu = VtuExport::new(&format!("{out}/{file}"), Format::Compressed);
      vtu.exporting(mf).unwrap();
      vtu
        .write_point_data(mf, md.variable(u).unwrap(), "u")
        .unwrap();
      vtu.write_point_data(mfs, &vm, "von Mises").unwrap();
      vtu.write().unwrap();
      pvd.push((step as f64, file));
    }
  }
  for (name, pvd) in [("ring1", &pvd1), ("ring2", &pvd2)] {
    let steps: Vec<(f64, &str)> = pvd.iter().map(|(t, f)| (*t, f.as_str())).collect();
    export_to_pvd(&format!("{out}/{name}.pvd"), &steps).unwrap();
  }
  println!("written: {out}/ring1.pvd, {out}/ring2.pvd");
}
