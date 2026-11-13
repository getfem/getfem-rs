//! The GWFL paper's example 2 (crates/getfem-tests/generators/paper/getfem_gwfl_example2.py):
//! phase field fracture of a plate with hydrogen. u Q2, the damage d Q2, the
//! pressure p Q1 and the hydrogen concentration C Q1 (Gc falls with C),
//! inertia written out (u_prev, v_prev data), an initial crack in the
//! history field psi0_max, the top and bottom faces pulled apart. The
//! script's time step control (Dt x 1.5 after a solve in at most 4 linear
//! solves, / 2 after a failed one), with Dt never above the initial time
//! step: Newton (with its damped methods) rarely fails, so the
//! script's halving alone would not keep Dt small. v_prev from the old
//! u_prev (the paper's script sets u_prev first, so its v_prev was always
//! 0).
//! Writes the reaction force per step to
//! target/showcase/phase_field_hydrogen_forces.dat and the final state to
//! phase_field_hydrogen.vtu. Optional arguments: elements in x and y
//! (default 80 x 60, as the paper), steps (default 250).

use std::io::Write;

use getfem::fem::im_data::ImData;
use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im::MeshIm;
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::mesh::Mesh;
use getfem::geom::region::{outer_faces_of_mesh, select_faces_in_box};
use getfem::io::vtk::{Format, VtuExport};
use getfem::model::{Model, Newton};

const T_RG: usize = 6;
const TB_RG: usize = 7;
const EXT_RG: usize = 8;

/// numpy.linspace
fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
  let step = (b - a) / (n - 1) as f64;
  let mut v: Vec<f64> = (0..n).map(|i| a + i as f64 * step).collect();
  v[n - 1] = b;
  v
}

/// the script's graded Cartesian mesh of [-1/2, 1/2]^2 (finer near x < 0
/// and y = 0), rectangles
fn mesh(nx: usize, ny: usize) -> Mesh {
  let (lx, ly) = (1.0, 1.0);
  let (nx, ny) = (nx.div_ceil(2) * 2, ny.div_ceil(2) * 2);
  let grade = |s: f64, a: f64, b: f64| a * s + b * s.signum() * s.abs().powf(1.5);
  let mut xs: Vec<f64> = linspace(-1.0, 0.0, nx / 4 + 1)
    .into_iter()
    .map(|s| lx / 2.0 * grade(s, 0.2, 0.8))
    .collect();
  xs.extend(
    linspace(0.0, 1.0, nx - nx / 4 + 1)[1..]
      .iter()
      .map(|&s| lx / 2.0 * grade(s, 1.0, 0.0)),
  );
  let ys: Vec<f64> = linspace(-1.0, 1.0, ny + 1)
    .into_iter()
    .map(|s| ly / 2.0 * grade(s, 0.2, 0.8))
    .collect();
  let m = Mesh::new(2);
  let gt = geotrans_descriptor("GT_QK(2,1)").unwrap();
  for j in 0..ys.len() - 1 {
    for i in 0..xs.len() - 1 {
      let q = [
        [xs[i], ys[j]],
        [xs[i + 1], ys[j]],
        [xs[i], ys[j + 1]],
        [xs[i + 1], ys[j + 1]],
      ];
      let r: Vec<&[f64]> = q.iter().map(|v| v.as_slice()).collect();
      m.add_convex_by_points(&gt, &r).unwrap();
    }
  }
  let outer = outer_faces_of_mesh(&m);
  let e = 1e-5;
  let bottom = select_faces_in_box(
    &m,
    &outer,
    &[-lx / 2.0 - e, -ly / 2.0 - e],
    &[lx / 2.0 + e, -ly / 2.0 + e],
  )
  .unwrap();
  let top = select_faces_in_box(
    &m,
    &outer,
    &[-lx / 2.0 - e, ly / 2.0 - e],
    &[lx / 2.0 + e, ly / 2.0 + e],
  )
  .unwrap();
  m.set_region(T_RG, &top).unwrap();
  m.set_region(TB_RG, &bottom.merge(&top).unwrap()).unwrap();
  m.set_region(EXT_RG, &outer).unwrap();
  m
}

fn main() {
  let arg = |k: usize, d: usize| std::env::args().nth(k).map_or(d, |s| s.parse().unwrap());
  let (nx, ny, steps) = (arg(1, 80), arg(2, 60), arg(3, 250));
  let init_time_step = 0.4;
  let (e, nu, rho) = (210e3, 0.3, 8e-9);
  let (gc0, chi, dgb0, c0, vh, r, t, ll): (f64, f64, f64, f64, f64, f64, f64, f64) =
    (2.7, 0.89, 30e6, 0.5, 2000.0, 8.314e3, 300.0, 0.015);
  let strain_rate = 2e-4;
  let m = mesh(nx, ny);
  let space = |q: usize, k: usize| {
    MeshFem::new(&m, q)
      .with_classical_finite_element(k)
      .unwrap()
  };
  let (mfp, mfu, mfd, mfc) = (space(1, 1), space(2, 2), space(1, 2), space(1, 1));
  let mim4 = MeshIm::new(&m).with_integration_method(3).unwrap();
  let mim9 = MeshIm::new(&m).with_integration_method(5).unwrap();
  let mimd9 = ImData::new(&mim9, -1, &[]).unwrap();
  let mut md = Model::new();
  md.add_fem_variable("u", &mfu).unwrap();
  md.add_fem_variable("d", &mfd).unwrap();
  md.add_fem_variable("p", &mfp).unwrap();
  md.add_fem_variable("C", &mfc).unwrap();
  let (nu_, nc) = (mfu.nb_dof().unwrap(), mfc.nb_dof().unwrap());
  md.set_variable("C", &vec![c0; nc]).unwrap();
  md.add_fem_data("u_prev", &mfu, &vec![0.0; nu_]).unwrap();
  md.add_fem_data("v_prev", &mfu, &vec![0.0; nu_]).unwrap();
  md.add_im_data("psi0_max", &mimd9, &vec![0.0; mimd9.nb_values().unwrap()])
    .unwrap();
  for (k, v) in [
    ("Dt", init_time_step),
    ("kappa", e / (3.0 * (1.0 - 2.0 * nu))),
    ("mu", e / (2.0 * (1.0 + nu))),
    ("rho", rho),
    ("Gc0", gc0),
    ("l", ll),
    ("C0", c0),
    ("chi", chi),
    ("c1", (-dgb0 / (r * t)).exp() / 5.5e-5),
    ("c2", vh / (r * t)),
  ] {
    md.add_scalar_data(k, v).unwrap();
  }
  // the initial crack
  let crack = "1e4*Gc0*min(1,max(0,1-(abs(X(2))+pos_part(X(1)))/(l/2)))";
  md.set_variable("psi0_max", &md.interpolation(crack, &mimd9, -1).unwrap())
    .unwrap();
  md.add_linear_term(&mim9, None, "rho/Dt*((u-u_prev)/Dt-v_prev).Test_u")
    .unwrap();
  for (k, v) in [
    ("degradation", "sqr(1-d)+1e-7"),
    ("deveps", "Sym(Grad(u))-Div(u)/3*Id(2)"),
    ("psi0", "(0.5*kappa*sqr(Div(u))+mu*Norm_sqr(deveps))"),
    ("Gc", "(1-chi*C/(C+c1))*Gc0"),
  ] {
    md.add_macro(k, v).unwrap();
  }
  for (mim, expr) in [
    (
      &mim9,
      "degradation*(kappa*Div(u)*Id(2)+2*mu*deveps):Grad(Test_u)",
    ),
    (
      &mim9,
      "(-2*(1-d)*max(psi0_max,psi0)*Test_d+Gc*(d/l*Test_d+l*Grad(d).Grad(Test_d)))",
    ),
    (&mim4, "(p+degradation*kappa*Div(u))*Test_p"),
    (
      &mim4,
      "(Grad(C)+c2*C*Grad(p)).Grad(Test_C)+1e3*pos_part(2*d-1)*(C-C0)*Test_C",
    ),
  ] {
    md.add_nonlinear_term(mim, None, expr).unwrap();
  }
  md.add_fem_data("dirichlet_data", &mfu, &vec![0.0; nu_])
    .unwrap();
  let mult = md
    .add_Dirichlet_condition_with_multiplier(&mim9, "u", &mfu, TB_RG, Some("dirichlet_data"))
    .unwrap();
  md.add_Dirichlet_condition_with_multiplier(&mim4, "C", &mfc, EXT_RG, Some("C0"))
    .unwrap();
  let newton = Newton {
    atol: 1e-9,
    ..Default::default()
  };
  println!("{} displacement DOFs, {} model DOFs", nu_, md.nb_dof());
  let out = format!("{}/../../target/showcase", env!("CARGO_MANIFEST_DIR"));
  std::fs::create_dir_all(&out).unwrap();
  let mut file = std::fs::File::create(format!("{out}/phase_field_hydrogen_forces.dat")).unwrap();
  let mut eps = 0.0;
  for step in 0..steps {
    let eps_old = eps;
    // until a solve converges, Dt halved
    let rep = loop {
      let dt = md.variable("Dt").unwrap()[0];
      eps = eps_old + if step > 0 { strain_rate * dt } else { 0.0 };
      let x = md.from_variables();
      let d = md
        .interpolation(&format!("{eps:e}*[0;X(2)]"), &mfu, -1)
        .unwrap();
      md.set_variable("dirichlet_data", &d).unwrap();
      match newton.solve(&mut md, None) {
        Ok(rep) if rep.converged() => break rep,
        _ => {
          md.to_variables(&x).unwrap();
          md.set_variable("Dt", &[0.5 * dt]).unwrap();
        }
      }
    };
    let dt = md.variable("Dt").unwrap()[0];
    md.set_variables(&[
      ("v_prev", "(u-u_prev)/Dt"),
      ("u_prev", "u"),
      ("psi0_max", "max(psi0_max,psi0)"),
    ])
    .unwrap();
    if rep.iterations <= 4 {
      md.set_variable("Dt", &[(1.5 * dt).min(init_time_step)])
        .unwrap();
    }
    let f = {
      let ws = md.workspace_mut();
      ws.clear_expressions();
      ws.add_expression(&format!("{mult}(2)"), &mim9, T_RG)
        .unwrap();
      ws.assembly(0, false).unwrap();
      ws.assembled_potential()
    };
    let dmax = md
      .variable("d")
      .unwrap()
      .iter()
      .cloned()
      .fold(f64::MIN, f64::max);
    writeln!(
      file,
      "step={step} eps={eps:e} dt={dt:e} force={f:e} dmax={dmax:e} iterations={}",
      rep.iterations
    )
    .unwrap();
    if step % 10 == 0 || step + 1 == steps {
      println!(
        "step {step}: eps {eps:.4e}, dt {dt:.3e}, force {f:.3}, max d {dmax:.4}, {} iterations",
        rep.iterations
      );
    }
  }
  let mut vtu = VtuExport::new(&format!("{out}/phase_field_hydrogen.vtu"), Format::Ascii);
  vtu.exporting(&mfu).unwrap();
  for (mf, n) in [(&mfu, "u"), (&mfd, "d"), (&mfp, "p"), (&mfc, "C")] {
    vtu
      .write_point_data(mf, md.variable(n).unwrap(), n)
      .unwrap();
  }
  vtu.write().unwrap();
  println!("wrote {out}/phase_field_hydrogen.vtu and phase_field_hydrogen_forces.dat");
}
