//! Phase field fracture of a notched plate (GetFEM's
//! interface/tests/python/demo_phase_field.py): u and phi Q2, the history
//! field psi0_max at the integration points, pulled by a Dirichlet
//! multiplier on the top and bottom faces. The demo's step control (load
//! step halving, then a hand-tuned pseudo-dynamic term) is replaced by
//! pseudo-viscous steps on phi where Newton fails: every load step is
//! reached (the port of the demo itself is tests/phase_field.rs). The mesh is
//! bilinear (the demo's affine GT_LINEAR_QK(2) ignores the fourth point of
//! the cells the notch distorts). Writes the reaction force per step (and
//! its pseudo-viscous steps) to target/showcase/phase_field_forces.dat and
//! the final state to phase_field.vtu. Optional arguments: elements per
//! side (default 60, as the demo), load steps (default 100), load case:
//! tension (the demo: top and bottom faces pulled apart, u = eps [0; y]) or
//! shear (moved along the faces, u = eps [y; 0]).

use std::io::Write;

use getfem::fem::im_data::ImData;
use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im::MeshIm;
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::mesh::Mesh;
use getfem::geom::region::{outer_faces_of_mesh, select_faces_in_box};
use getfem::io::vtk::{Format, VtuExport};
use getfem::model::{Method, Model, Newton};

const T_BOUNDARY: usize = 6;
const B_BOUNDARY: usize = 4;
const TB_BOUNDARY: usize = 7;

/// numpy.linspace
fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
  let step = (b - a) / (n - 1) as f64;
  let mut v: Vec<f64> = (0..n).map(|i| a + i as f64 * step).collect();
  v[n - 1] = b;
  v
}

/// the demo's graded, notched mesh of [-1/2, 1/2]^2; NX, NY rounded up to
/// even (y = 0, the notch, a grid line), as the demo does (since GetFEM
/// 9fb90f9)
fn mesh(nx: usize, ny: usize) -> Mesh {
  let (lx, ly) = (1.0, 1.0);
  let even = |n: usize| n.div_ceil(2) * 2;
  let (nx, ny) = (even(nx), even(ny));
  let grade = |s: f64, a: f64, b: f64| a * s + b * s.signum() * s.abs().powf(1.5);
  let n3 = nx as f64 / 3.0;
  let mut xs: Vec<f64> = linspace(-1.0, 0.0, (n3 + 1.0) as usize)
    .into_iter()
    .map(|s| lx / 2.0 * grade(s, 0.2, 0.8))
    .collect();
  xs.extend(
    linspace(0.0, 1.0, (nx as f64 - n3 + 1.0) as usize)[1..]
      .iter()
      .map(|&s| lx / 2.0 * grade(s, 0.6, 0.4)),
  );
  let ys: Vec<f64> = linspace(-1.0, 1.0, ny + 1)
    .into_iter()
    .map(|s| ly / 2.0 * grade(s, 0.2, 0.8))
    .collect();
  let gap = 1e-5;
  let m = Mesh::new(2);
  // bilinear: the notch makes some cells non-parallelograms (the demo's
  // affine GT_LINEAR_QK(2) ignores their fourth point)
  let gt = geotrans_descriptor("GT_QK(2,1)").unwrap();
  // lower half, then the upper half; the points of the notch (x < 0 on
  // y = 0) moved apart by the gap, the others merged
  let half = ny as f64 / 2.0;
  for (rows, sign) in [
    (&ys[..(half + 1.0) as usize], -1.0),
    (&ys[half as usize..], 1.0),
  ] {
    let p = |i: usize, j: usize| -> Vec<f64> {
      let (x, y) = (xs[i], rows[j]);
      let y = if x < -1e-10 {
        y + sign * gap / 2.0 * (1.0 - sign * y / (ly / 2.0))
      } else {
        y
      };
      vec![x, y]
    };
    for j in 0..rows.len() - 1 {
      for i in 0..xs.len() - 1 {
        let q = [p(i, j), p(i + 1, j), p(i, j + 1), p(i + 1, j + 1)];
        let r: Vec<&[f64]> = q.iter().map(|v| v.as_slice()).collect();
        m.add_convex_by_points(&gt, &r).unwrap();
      }
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
  m.set_region(B_BOUNDARY, &bottom).unwrap();
  m.set_region(T_BOUNDARY, &top).unwrap();
  m.set_region(TB_BOUNDARY, &top.merge(&bottom).unwrap())
    .unwrap();
  m
}

/// A load step: Newton; where it fails, pseudo-viscous steps on phi from
/// the start values (implicit Euler, visc = scale / dtau, phi_n the last
/// accepted step), each solved by full Newton (at most 30 solves): dtau x8
/// after at most 3 solves, x2 after at most 6, / 4 after a failure; above
/// 1e4, Newton without the term. Returns the pseudo-viscous steps and the
/// linear solves.
fn solve_step(md: &mut Model, newton: &Newton, scale: f64) -> (usize, usize) {
  let x0 = md.from_variables();
  let rep = newton.solve(md, None).unwrap();
  let mut solves = rep.iterations;
  if rep.converged() {
    return (0, solves);
  }
  let inner = Newton {
    methods: vec![Method::Newton],
    max_iter: 30,
    ..newton.clone()
  };
  md.to_variables(&x0).unwrap();
  let (mut xn, mut dtau, mut k) = (x0, 1.0f64, 0);
  while solves < 10000 {
    let last = dtau > 1e4;
    let visc = if last { 0.0 } else { scale / dtau };
    md.set_variable("visc", &[visc]).unwrap();
    let phi = md.variable("phi").unwrap().to_vec();
    md.set_variable("phi_n", &phi).unwrap();
    let rep = inner.solve(md, None).unwrap();
    solves += rep.iterations;
    k += 1;
    if rep.converged() && last {
      return (k, solves);
    }
    if rep.converged() {
      xn = md.from_variables();
      dtau *= match rep.iterations {
        0..=3 => 8.0,
        4..=6 => 2.0,
        _ => 1.0,
      };
    } else {
      md.to_variables(&xn).unwrap();
      dtau = if last { 1e4 } else { dtau / 4.0 };
    }
  }
  panic!("no convergence in {solves} linear solves");
}

/// The results of a load step.
pub struct Step {
  pub eps: f64,
  /// the reaction on the top face
  pub force: [f64; 2],
  pub phimax: f64,
  /// the pseudo-viscous steps (0: Newton alone), the linear solves
  pub viscous: usize,
  pub solves: usize,
}

/// The demo on nx x nx elements for the given strains (solve_step per
/// load step), tension or shear; on_step after
/// each step. Returns the mesh, the spaces of u and phi and the model.
pub fn simulate(
  nx: usize,
  strains: &[f64],
  shear: bool,
  on_step: &mut dyn FnMut(usize, &Step),
) -> (MeshFem, MeshFem, Model) {
  // imposed displacement of the top and bottom faces
  let load = if shear {
    "eps*[X(2);0]"
  } else {
    "eps*[0;X(2)]"
  };
  let (e, nu, gc0, ll) = (210e3, 0.3, 2.7, 0.015);
  let m = mesh(nx, nx);
  let mfu = MeshFem::new(&m, 2)
    .with_classical_finite_element(2)
    .unwrap();
  let mfphi = MeshFem::new(&m, 1)
    .with_classical_finite_element(2)
    .unwrap();
  let mim = MeshIm::new(&m).with_integration_method(5).unwrap();
  let mimd1 = ImData::new(&mim, -1, &[]).unwrap();
  let mut md = Model::new();
  md.add_fem_variable("u", &mfu).unwrap();
  md.add_fem_variable("phi", &mfphi).unwrap();
  let nu_ = mfu.nb_dof().unwrap();
  md.add_im_data("psi0_max", &mimd1, &vec![0.0; mimd1.nb_points().unwrap()])
    .unwrap();
  for (k, v) in [
    ("kappa", e / (3.0 * (1.0 - 2.0 * nu))),
    ("mu", e / (2.0 * (1.0 + nu))),
    ("Gc0", gc0),
    ("l", ll),
    ("eps", 0.0),
  ] {
    md.add_scalar_data(k, v).unwrap();
  }
  md.add_fem_data("dirichlet_data", &mfu, &vec![0.0; nu_])
    .unwrap();
  for (k, v) in [
    ("damage", "max(1e-7,sqr(1-phi))"),
    (
      "psi0",
      "(0.5*kappa*sqr(Trace(Grad_u))+mu*Norm_sqr((Sym(Grad_u)-Trace(Grad_u)/3*Id(2))))",
    ),
    ("Gc", "Gc0"),
    (
      "sigma",
      "damage*(kappa*Trace(Grad_u)*Id(2)+2*mu*(Sym(Grad_u)-Trace(Grad_u)/3*Id(2)))",
    ),
  ] {
    md.add_macro(k, v).unwrap();
  }
  md.add_term(&mim, None, "sigma:Grad_Test_u").unwrap();
  md.add_term(
    &mim,
    None,
    "(-2*(1-phi)*max(psi0_max,psi0)*Test_phi+Gc*(phi/l*Test_phi+l*Grad_phi.Grad_Test_phi))",
  )
  .unwrap();
  let mult = md
    .add_Dirichlet_condition_with_multiplier(&mim, "u", &mfu, TB_BOUNDARY, Some("dirichlet_data"))
    .unwrap();
  // the pseudo-viscous term on the damage only (u stays in equilibrium),
  // off while visc = 0
  md.add_scalar_data("visc", 0.0).unwrap();
  md.add_fem_data("phi_n", &mfphi, &vec![0.0; mfphi.nb_dof().unwrap()])
    .unwrap();
  md.add_term(&mim, None, "visc*(phi-phi_n)*Test_phi")
    .unwrap();
  let newton = Newton {
    atol: 1e-9,
    ..Default::default()
  };
  println!("{} displacement DOFs, {} model DOFs", nu_, md.nb_dof());
  for (step, &eps) in strains.iter().enumerate() {
    md.set_variable("eps", &[eps]).unwrap();
    let d = md.interpolation(load, &mfu, -1).unwrap();
    md.set_variable("dirichlet_data", &d).unwrap();
    let (viscous, solves) = solve_step(&mut md, &newton, gc0 / ll);
    let h = md.interpolation("max(psi0_max,psi0)", &mimd1, -1).unwrap();
    md.set_variable("psi0_max", &h).unwrap();
    let phimax = md
      .variable("phi")
      .unwrap()
      .iter()
      .cloned()
      .fold(f64::MIN, f64::max);
    let mut f = |c: usize| -{
      let ws = md.workspace_mut();
      ws.clear_expressions();
      ws.add_expression(&format!("{mult}({c})"), &mim, T_BOUNDARY)
        .unwrap();
      ws.assembly(0, false).unwrap();
      ws.assembled_potential()
    };
    let s = Step {
      eps,
      force: [f(1), f(2)],
      phimax,
      viscous,
      solves,
    };
    on_step(step, &s);
  }
  (mfu, mfphi, md)
}

fn main() {
  let arg = |k: usize, d: usize| std::env::args().nth(k).map_or(d, |s| s.parse().unwrap());
  let (nx, steps) = (arg(1, 60), arg(2, 100));
  let shear = match std::env::args().nth(3).as_deref() {
    None | Some("tension") => false,
    Some("shear") => true,
    Some(c) => panic!("load case {c}: tension or shear"),
  };
  // the reaction shown
  let comp = if shear { 0 } else { 1 };
  let out = format!("{}/../../target/showcase", env!("CARGO_MANIFEST_DIR"));
  std::fs::create_dir_all(&out).unwrap();
  let mut file = std::fs::File::create(format!("{out}/phase_field_forces.dat")).unwrap();
  let strains: Vec<f64> = (0..steps).map(|k| 2e-4 * k as f64).collect();
  let (mfu, mfphi, md) = simulate(nx, &strains, shear, &mut |step, s| {
    writeln!(
      file,
      "step={step} eps={:e} fR=({:e},{:e}) phimax={:e} viscous={} solves={}",
      s.eps, s.force[0], s.force[1], s.phimax, s.viscous, s.solves
    )
    .unwrap();
    if s.viscous > 0 {
      println!(
        "step {step}: {} pseudo-viscous steps ({} linear solves)",
        s.viscous, s.solves
      );
    }
    if step % 10 == 0 || step + 1 == steps {
      println!(
        "step {step}: eps {:.4e}, force {:.3}, max phi {:.4}",
        s.eps, s.force[comp], s.phimax
      );
    }
  });
  let mut vtu = VtuExport::new(&format!("{out}/phase_field.vtu"), Format::Ascii);
  vtu.exporting(&mfu).unwrap();
  vtu
    .write_point_data(&mfu, md.variable("u").unwrap(), "u")
    .unwrap();
  vtu
    .write_point_data(&mfphi, md.variable("phi").unwrap(), "phi")
    .unwrap();
  vtu.write().unwrap();
  println!("wrote {out}/phase_field.vtu and phase_field_forces.dat");
}
