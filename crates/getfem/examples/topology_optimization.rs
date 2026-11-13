//! Topology optimization of a cantilever (a user's GetFEM script, gf_topopt):
//! a beam LX x LY clamped on the left, a vertical traction on the middle
//! of its right end, large strains (a compressible neo-Hookean law, plane
//! strain), the material density rho(x) = 1/(1+exp(-x)) of a design
//! variable x on uniform order-3 B-splines (one interval per element),
//! stiffness RAMP-interpolated. The compliance (the load's work) is
//! minimized at the volume fraction 0.5 (multiplier multvol), with the
//! adjoint displacements ADJu and a penalty on steep design gradients. The
//! optimality conditions are solved together in pseudo-time: a viscous
//! filter term on x of step dt, dt doubled after fast Newton solves and
//! halved after failed ones, then removed (inv_dt = 0) for the final step.
//!
//! Writes the density, the displacements and the von Mises stress of each
//! design iteration as .vtu files with a .pvd collection
//! (target/showcase/topology_optimization/) and prints the volume
//! fraction, the measure of non-discreteness MND and the objective.

use std::collections::BTreeSet;

use getfem::fem::fem::fem_descriptor;
use getfem::fem::fem_global_function::BsplineBoundary;
use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im::MeshIm;
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::region::{outer_faces_of_mesh, select_faces_in_box};
use getfem::geom::regular_mesh::regular_mesh;
use getfem::gwfl::define_gwfl_function;
use getfem::io::vtk::{Format, VtuExport, export_to_pvd};
use getfem::linalg::{LinearSolver, MatrixKind};
use getfem::model::{BuildVersion, Model, Newton};
use getfem::mumps::Mumps;

const LEFT: usize = 3;
const RIGHT: usize = 5;

pub struct Params {
  /// elements (and B-spline intervals) along x and y
  pub nx: usize,
  pub ny: usize,
  pub itermax: usize,
}

pub struct Iteration {
  pub i: usize,
  /// pseudo-time
  pub t: f64,
  /// the step of the next damped solve
  pub dt: f64,
  pub volume: f64,
  /// measure of non-discreteness
  pub mnd: f64,
  pub objective: f64,
}

/// Runs the optimization; on_iteration after each design iteration.
pub fn optimize(p: &Params, on_iteration: &mut dyn FnMut(&Iteration, &Model)) {
  let (lx, ly) = (100.0, 25.0);
  let (e, nu) = (1e3, 0.4);
  let vol = 0.5 * lx * ly;
  let (q_ramp, contrast) = (4.0, 1e-10);
  let (li, rfilt) = (0.03 * lx, 0.06 * lx);
  let (dy, traction) = (ly / 20.0, -e / 30.0);
  let k_l = 1e-4;
  let eps = 1e-8;

  let gt = geotrans_descriptor("GT_QK(2,2)").unwrap();
  let m = regular_mesh(&gt, &[p.nx, p.ny], &[0.0, 0.0], &[lx, ly]).unwrap();
  let border = outer_faces_of_mesh(&m);
  let (a, b) = (1e-5, ly + 1e-5);
  m.set_region(
    LEFT,
    &select_faces_in_box(&m, &border, &[-a, -a], &[a, b]).unwrap(),
  )
  .unwrap();
  m.set_region(
    RIGHT,
    &select_faces_in_box(&m, &border, &[lx - a, -a], &[lx + a, b]).unwrap(),
  )
  .unwrap();

  // displacements: incomplete Q2, clamped on the left
  let mfn = MeshFem::new(&m, 2)
    .with_finite_element(&fem_descriptor("FEM_Q2_INCOMPLETE(2)").unwrap())
    .unwrap();
  let clamped = mfn.basic_dof_on_region(LEFT).unwrap();
  let kept: BTreeSet<usize> = (0..mfn.nb_dof().unwrap())
    .filter(|d| !clamped.contains(d))
    .collect();
  let mfu = MeshFem::partial(&mfn, &kept).unwrap();
  // design variable: uniform B-splines of order 3
  let free = [BsplineBoundary::Free; 2];
  let mfx = MeshFem::uniform_bspline(&m, &[p.nx, p.ny], 3, &free, &free, None).unwrap();
  let mim = MeshIm::new(&m).with_integration_method(5).unwrap();

  let mut md = Model::new();
  define_gwfl_function(
    "rho_der3",
    1,
    "(exp(t)*(exp(2*t) - 4*exp(t) + 1))/pow(exp(t)+1,4)",
    "",
    "",
  )
  .unwrap();
  define_gwfl_function(
    "rho_der2",
    1,
    "exp(t)*(1-exp(t))/pow(1+exp(t),3)",
    "rho_der3(t)",
    "",
  )
  .unwrap();
  define_gwfl_function("rho_der", 1, "exp(-t)/sqr(1+exp(-t))", "rho_der2(t)", "").unwrap();
  define_gwfl_function("rho", 1, "1/(1+exp(-t))", "rho_der(t)", "").unwrap();
  define_gwfl_function(
    "RAMP",
    1,
    &format!("{contrast}+(1-{contrast})*t/(1+{q_ramp}*(1-t))"),
    "",
    "",
  )
  .unwrap();

  let nx = mfx.nb_dof().unwrap();
  md.add_fem_variable("x", &mfx).unwrap();
  // the L2 projection of 0.5 (the basis is a partition of unity)
  md.set_variable("x", &vec![0.5; nx]).unwrap();
  md.add_fem_data("x0", &mfx, &vec![0.5; nx]).unwrap();
  let mut dt = 1e2;
  md.add_scalar_data("inv_dt", 1.0 / dt).unwrap();
  let area = {
    let ws = md.workspace_mut();
    ws.clear_expressions();
    ws.add_expression("1", &mim, None).unwrap();
    ws.assembly(0, false).unwrap();
    ws.assembled_potential()
  };
  for (k, v) in [
    ("Li", li),
    ("Rfilt", rfilt),
    ("k_l", k_l),
    ("K", e / (3.0 * (1.0 - 2.0 * nu))),
    ("G", e / (2.0 * (1.0 + nu))),
    ("rhobar", vol / area),
    ("LY", ly),
    ("DY", dy),
    ("load", traction),
  ] {
    md.add_scalar_data(k, v).unwrap();
  }
  md.add_fixed_size_variable("multvol", &[1]).unwrap();
  md.add_fem_variable("u", &mfu).unwrap();
  md.add_fem_variable("ADJu", &mfu).unwrap();

  for (k, v) in [
    ("load_weight", "pos_part(1-abs(X(2)-LY/2)/DY)"),
    // the objective
    ("OBJu", "-load_weight*u(2)"),
    ("Dev33(A)", "A-Id(2)/3*(Trace(A)+1)"),
    ("F(u)", "Id(2)+Grad_u"),
    ("F3d(u)", "Id(3)+[1,0;0,1;0,0]*Grad_u*[1,0,0;0,1,0]"),
    (
      "tau(u)",
      "K*log(Det(F(u)))*Id(2)+G*pow(Det(F(u)),-2/3)*Dev33(F(u)*F(u)')",
    ),
    (
      "tau3d(u)",
      "K*log(Det(F(u)))*Id(3)+G*pow(Det(F(u)),-2/3)*Deviator(F3d(u)*F3d(u)')",
    ),
    ("sigma3d(u)", "tau3d(u)/Det(F(u))"),
  ] {
    md.add_macro(k, v).unwrap();
  }
  md.add_nonlinear_term(&mim, RIGHT, "-load_weight*Test_ADJu(2)")
    .unwrap();
  md.add_linear_term(&mim, RIGHT, "load_weight*[0;-load].Test_u")
    .unwrap();
  for t in [
    "RAMP(rho(x))*((tau(u)*Inv(F(u)')):Grad(Test_u))",
    "RAMP(rho(x))*Diff((tau(u)*Inv(F(u)')):Grad(ADJu),u,Test_ADJu)",
    "Diff(RAMP(rho(x)),x,Test_x)*((tau(u)*Inv(F(u)')):Grad(ADJu))",
    // the volume constraint
    "(rho(x)-rhobar)*Test_multvol",
    "Diff(rho(x),x,Test_x)*multvol",
    // the penalty on steep design gradients
    "k_l*pow(pos_part(Norm(Grad(x))-8/Li),5)*(Normalized(Grad(x)).Grad(Test_x))",
  ] {
    md.add_nonlinear_term(&mim, None, t).unwrap();
  }
  // the viscous filter of the pseudo-time steps
  md.add_linear_term(
    &mim,
    None,
    "inv_dt*((x-x0)*Test_x+sqr(Rfilt)*(Grad_x-Grad_x0).Grad_Test_x)",
  )
  .unwrap();
  println!(
    "Displacement dofs: {}\nTotal model dofs: {}",
    mfu.nb_dof().unwrap(),
    md.nb_dof()
  );

  // the equilibrium and its adjoint for the initial design
  let newton = Newton {
    atol: 1e-8,
    rtol: 0.0,
    ..Default::default()
  };
  for v in ["x", "multvol", "ADJu"] {
    md.disable_variable(v).unwrap();
  }
  let rep = newton.solve(&mut md, None).unwrap();
  assert!(rep.converged(), "equilibrium: {}", rep.message);
  md.enable_variable("ADJu").unwrap();
  let rep = newton.solve(&mut md, None).unwrap();
  assert!(rep.converged(), "adjoint: {}", rep.message);
  for v in ["x", "multvol"] {
    md.enable_variable(v).unwrap();
  }
  println!("Mechanical equilibrium solved for the initial design\n\nStarting optimization...");

  let mut final_dt = 1e14;
  // 0: damped steps, 1: the final undamped step tried, 2: done
  let mut fin = 0;
  let mut t = 0.0;
  let mut solver = Mumps::new();
  for i in 0..p.itermax {
    println!("\nDesign iteration {i}");
    md.set_variable("x0", "x").unwrap();
    let saved = md.from_variables();
    loop {
      let mut restot = f64::NAN;
      let mut j = 0;
      while j < 40 {
        let restot_prev = restot;
        md.assembly(BuildVersion::All).unwrap();
        let mut dx = md.rhs().to_vec();
        solver
          .solve_system(md.tangent_matrix(), MatrixKind::General, &mut dx)
          .unwrap();
        let s: Vec<f64> = md
          .from_variables()
          .iter()
          .zip(&dx)
          .map(|(a, b)| a + b)
          .collect();
        md.to_variables(&s).unwrap();
        md.assembly(BuildVersion::Rhs).unwrap();
        let r = md.rhs();
        let norm = |r: &[f64]| r.iter().map(|v| v * v).sum::<f64>().sqrt();
        restot = norm(r);
        let parts: Vec<String> = ["x", "multvol", "u", "ADJu"]
          .iter()
          .map(|n| {
            let iv = md.interval_of_variable(n).unwrap();
            format!("{n}: {:4.1e}", norm(&r[iv]))
          })
          .collect();
        let dts = if fin == 1 {
          "inf".to_string()
        } else {
          format!("{dt}")
        };
        println!(
          "{i} (dt={dts}):{}: res={restot:8.2e} = {}",
          j + 1,
          parts.join(" + ")
        );
        if restot.is_nan() || restot < eps || (j > 5 && restot > restot_prev) {
          break;
        }
        j += 1;
      }
      if !restot.is_nan() && restot < eps {
        t += dt;
        if fin == 1 {
          fin = 2;
        } else if j < 5 {
          if dt > final_dt {
            fin = 1;
            md.set_variable("inv_dt", &[0.0]).unwrap();
            println!("Disabling damping of optimality and intending final step");
          } else {
            dt *= 2.0;
            md.set_variable("inv_dt", &[1.0 / dt]).unwrap();
          }
        }
        break;
      }
      md.to_variables(&saved).unwrap();
      if fin == 1 {
        final_dt *= 10.0;
        fin = 0;
        println!("Reenabling damping of optimality");
      } else {
        dt *= 0.5;
      }
      md.set_variable("inv_dt", &[1.0 / dt]).unwrap();
    }
    let it = Iteration {
      i,
      t,
      dt,
      volume: {
        let ws = md.workspace_mut();
        ws.clear_expressions();
        ws.add_expression("rho(x)", &mim, None).unwrap();
        ws.assembly(0, false).unwrap();
        ws.assembled_potential()
      } / (lx * ly),
      mnd: {
        let ws = md.workspace_mut();
        ws.clear_expressions();
        ws.add_expression("4*rho(x)*(1-rho(x))", &mim, None)
          .unwrap();
        ws.assembly(0, false).unwrap();
        ws.assembled_potential()
      } / (lx * ly),
      objective: {
        let ws = md.workspace_mut();
        ws.clear_expressions();
        ws.add_expression("OBJu", &mim, RIGHT).unwrap();
        ws.assembly(0, false).unwrap();
        ws.assembled_potential()
      },
    };
    println!(
      "{i} (t={t:e}): Volume fraction = {:.6}, MND = {:.6}, Objective value = {:.6}",
      it.volume, it.mnd, it.objective
    );
    on_iteration(&it, &md);
    if fin == 2 {
      break;
    }
  }
}

fn main() {
  let out = format!(
    "{}/../../target/showcase/topology_optimization",
    env!("CARGO_MANIFEST_DIR")
  );
  std::fs::create_dir_all(&out).unwrap();
  let p = Params {
    nx: 80,
    ny: 20,
    itermax: 150,
  };
  let mut pvd = vec![];
  optimize(&p, &mut |it, md| {
    let mfx = md.mesh_fem_of_variable("x").unwrap();
    let mfu = md.mesh_fem_of_variable("u").unwrap();
    let mfout = MeshFem::new(&mfx.linked_mesh(), 1)
      .with_classical_discontinuous_finite_element(2)
      .unwrap();
    let file = format!("topopt_{}.vtu", it.i);
    let mut vtu = VtuExport::new(&format!("{out}/{file}"), Format::Compressed);
    vtu.exporting(&mfout).unwrap();
    vtu
      .write_point_data(&mfx, md.variable("x").unwrap(), "x")
      .unwrap();
    for (e, name) in [
      ("rho(x)", "rho"),
      ("RAMP(rho(x))", "ramp"),
      ("Norm(Deviator(sigma3d(u)))", "Von Mises stress"),
    ] {
      let v = md.interpolation(e, &mfout, None).unwrap();
      vtu.write_point_data(&mfout, &v, name).unwrap();
    }
    vtu
      .write_point_data(&mfu, md.variable("u").unwrap(), "Displacements")
      .unwrap();
    vtu.write().unwrap();
    pvd.push((it.t, file));
  });
  let steps: Vec<(f64, &str)> = pvd.iter().map(|(t, f)| (*t, f.as_str())).collect();
  export_to_pvd(&format!("{out}/topopt.pvd"), &steps).unwrap();
  println!("written: {out}/topopt.pvd");
}
