//! The GWFL paper's example 1 (crates/getfem-tests/generators/paper/getfem_gwfl_example1.py): an
//! inflated hyperelastic membrane. A disc of diameter 100 and thickness 2
//! (regular_ball_mesh, Q2, embedded in 3D), the displacement u and the
//! thickness vector n as unknowns (F = I2 + Grad u + n x e3), a compressible
//! neo-Hookean energy, a follower pressure gamma * pmax, the rim clamped;
//! Moore-Penrose continuation in gamma past the limit load. Writes gamma,
//! the step length and max |u| per step to target/showcase/membrane.dat
//! and the final u to membrane.vtu. Optional arguments: the core and peel
//! subdivisions (default 10, 3, as the paper), steps (default 300).

use std::io::Write;

use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im::MeshIm;
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::region::outer_faces_of_mesh;
use getfem::geom::regular_mesh::regular_ball_mesh;
use getfem::io::vtk::{Format, VtuExport};
use getfem::linalg::tensor::Mat;
use getfem::model::{ContinuationStatus, Model, Newton, NumericalContinuation, VariableScale};

const DIR_BOUNDARY: usize = 6;

fn main() {
  let arg = |k: usize, d: usize| std::env::args().nth(k).map_or(d, |s| s.parse().unwrap());
  let (nsub, steps) = ([arg(1, 10), arg(2, 3)], arg(3, 300));
  let (d, h_thick, e, nu, pmax) = (100.0, 2.0, 5e3, 0.4, 1.5e2);
  let m = regular_ball_mesh(
    &geotrans_descriptor("GT_QK(2,2)").unwrap(),
    &[0.0, 0.0],
    d / 2.0,
    nsub,
    0,
    false,
  )
  .unwrap();
  // [[1,0],[0,1],[0,0]], column-major
  m.transform(&Mat::from_data(&[3, 2], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap())
    .unwrap();
  m.set_region(DIR_BOUNDARY, &outer_faces_of_mesh(&m))
    .unwrap();
  let mfu = MeshFem::new(&m, 3)
    .with_classical_finite_element(2)
    .unwrap();
  let mfn = MeshFem::new(&m, 3)
    .with_classical_finite_element(2)
    .unwrap();
  let mim = MeshIm::new(&m).with_integration_method(5).unwrap();
  let mut md = Model::new();
  md.add_fem_variable("u", &mfu).unwrap();
  md.add_fem_variable("n", &mfn).unwrap();
  md.set_variable("n", &md.interpolation("[0,0,1]", &mfn, -1).unwrap())
    .unwrap();
  for (k, v) in [
    ("gamma", 0.0),
    ("pmax", pmax),
    ("kappa", e / (3.0 * (1.0 - 2.0 * nu))),
    ("mu", e / (2.0 * (1.0 + nu))),
    ("H", h_thick),
  ] {
    md.add_scalar_data(k, v).unwrap();
  }
  md.add_macro("F", "[1,0,0;0,1,0;0,0,0]+Grad(u)+n@[0,0,1]")
    .unwrap();
  md.add_nonlinear_term(
    &mim,
    None,
    "H*0.5*kappa*sqr(log(Det(F)))+H*0.5*mu*(pow(Det(F),-2/3)*Norm_sqr(F)-3)",
  )
  .unwrap();
  md.add_nonlinear_term(&mim, None, "gamma*pmax*Det(F)*((Inv(F)'*[0,0,-1]).Test_u)")
    .unwrap();
  md.add_Dirichlet_condition_with_multiplier(&mim, "u", &mfu, DIR_BOUNDARY, None)
    .unwrap();
  let nu_ = mfu.nb_dof().unwrap();
  println!("{} displacement DOFs, {} model DOFs", nu_, md.nb_dof());
  let out = format!("{}/../../target/showcase", env!("CARGO_MANIFEST_DIR"));
  std::fs::create_dir_all(&out).unwrap();
  let mut file = std::fs::File::create(format!("{out}/membrane.dat")).unwrap();
  let mut limit = (0.0, 0, 0.0);
  let mut s = NumericalContinuation::new("gamma", vec![VariableScale::new(&["u"], 1.0)]);
  (s.h_init, s.h_max, s.h_min, s.h_dec, s.h_inc) =
    (1e-1, 1.0, 1e-4, 1.0 / 10f64.sqrt(), 10f64.sqrt());
  (s.max_iter, s.thr_iter, s.max_res, s.max_diff, s.min_cos) = (10, 6, 1e-5, 1e-8, 0.999);
  s.singularities = 0;
  let newton = Newton {
    atol: 1e-7,
    ..Default::default()
  };
  let (mut tx, mut tgamma, mut h) = (vec![], 1.0, 0.0);
  for step in 0..=steps {
    if md.variable("gamma").unwrap()[0] >= 1.0 {
      break;
    }
    if step == 0 {
      assert!(newton.solve(&mut md, None).unwrap().converged());
      s.step(&mut md, &mut tx, &mut tgamma, &mut h, None).unwrap();
    } else {
      let rep = s.step(&mut md, &mut tx, &mut tgamma, &mut h, None).unwrap();
      assert_eq!(
        rep.status,
        ContinuationStatus::Converged,
        "continuation failed at step {step}"
      );
    }
    let umax = md
      .variable("u")
      .unwrap()
      .iter()
      .fold(0.0f64, |a, v| a.max(v.abs()));
    let g = md.variable("gamma").unwrap()[0];
    if g > limit.0 {
      limit = (g, step, umax);
    }
    writeln!(file, "step={step} gamma={g:e} h={h:e} umax={umax:e}").unwrap();
    if step % 10 == 0 {
      println!("step {step}: gamma {g:.6}, max |u| {umax:.3}, h {h:.3e}");
    }
  }
  println!(
    "limit load gamma {:.6} at step {} (max |u| {:.3})",
    limit.0, limit.1, limit.2
  );
  let mut vtu = VtuExport::new(&format!("{out}/membrane.vtu"), Format::Ascii);
  vtu.exporting(&mfu).unwrap();
  vtu
    .write_point_data(&mfu, md.variable("u").unwrap(), "u")
    .unwrap();
  vtu.write().unwrap();
  println!("wrote {out}/membrane.vtu and membrane.dat");
}
