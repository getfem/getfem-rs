//! Shell to solid contact: two hemispheres of midsurface radius R = 10
//! and thickness T = 0.25, neo-Hookean (E, nu), pressed together off
//! their poles, half models (symmetry y = 0):
//! - solid: dome up, its equator clamped in x and z; 27-node hexahedra
//!   (Hexa27), one through the thickness, mixed u-p;
//! - shell: dome down above it, shifted sideways by sx, its equator held
//!   in x and pushed down by delta; 9-node quadrilaterals (Quad9) on the
//!   midsurface, a 7-parameter shell (displacement u, director change d,
//!   thickness stretch gradient w), with Normal and element_curvature.
//!
//! The shell at xi across its thickness, K = element_curvature,
//! integrated through the thickness by 3-point Gauss:
//!   x(xi) = X + u + xi (1 + xi w) (Normal + d),
//!   F(xi) = (I + Grad_u + xi (1 + xi w) (K + Grad_d) + xi^2 (Normal + d)@Grad_w
//!            + (d + 2 xi w (Normal + d))@Normal) (I + xi K)^-1.
//! Contact: the shell (slave) on the solid's outer surface (master), rays
//! along the deformed shell normal nx, the gap of the shell's surface
//! facing the solid
//!   gap = (Interpolate(X,ray) + Interpolate(ub,ray) - X - us - T/2 (1 + T/2 w) (Normal + d)).nx,
//! the contact pressure lc (Q1, one order below the displacements) by the
//! augmented Lagrangian lc = neg_part(r gap - lc), as a weak form; the
//! augmentation r low, of the order of the shells' own stiffness E T/R^2.
//!
//! Arguments: shell and solid elements per 45 deg (8, 3), final delta
//! (6), steps (120). Writes both bodies per step as .vtu with .pvd
//! collections (time: delta; target/showcase/shell_solid_contact/) and
//! prints the Newton iterations and the contact force.

use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im::MeshIm;
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::region::RegionSet;
use getfem::geom::regular_mesh::regular_ball_shell_mesh;
use getfem::io::vtk::{Format, VtuExport, export_to_pvd};
use getfem::linalg::tensor::Tensor;
use getfem::model::{Model, Newton};
use std::collections::BTreeSet;
use std::f64::consts::PI;

const YM: usize = 4;
const ZM: usize = 5;
const EXT: usize = 6;
const SHELL: usize = 7;

fn main() {
  let arg = |i: usize, d: f64| std::env::args().nth(i).map_or(d, |a| a.parse().unwrap());
  let (nc, ncb) = (arg(1, 8.0) as usize, arg(2, 3.0) as usize);
  let (delta_max, steps) = (arg(3, 6.0), arg(4, 120.0));
  let (th, rad, gap0, sx): (f64, f64, f64, f64) = (0.25, 10.0, 0.02, 6.0);
  let (e, nu) = (6.825e7, 0.3);
  // centre of the shell sphere: the surfaces gap0 apart along the line of
  // the centres
  let zc = ((2.0 * rad + th + gap0) * (2.0 * rad + th + gap0) - sx * sx).sqrt();

  // solid: dome up, radii R -/+ T/2, the half y >= 0
  let mb = regular_ball_shell_mesh(
    &geotrans_descriptor("GT_QK(3,2)").unwrap(),
    &[0.0, 0.0, 0.0],
    rad + th / 2.0,
    th,
    [ncb, 1],
    2,
    false,
  )
  .unwrap();
  let b = 2.0 * rad;
  mb.set_region(
    YM,
    &mb
      .outer_faces_in_box(&[-b, -1e-5, -1e-5], &[b, 1e-5, b])
      .unwrap(),
  )
  .unwrap();
  mb.set_region(
    ZM,
    &mb
      .outer_faces_in_box(&[-b, -1e-5, -1e-5], &[b, b, 1e-5])
      .unwrap(),
  )
  .unwrap();
  // upper outer faces
  mb.set_region(
    EXT,
    &mb
      .outer_faces_with_direction(&[0.0, 0.0, 1.0], 0.49 * PI)
      .unwrap(),
  )
  .unwrap();

  // shell: dome down above the solid, the half y >= 0 rotated by
  // (x,y,z) -> (-x,y,-z) (a rotation: Normal stays outwards, down,
  // towards the solid)
  let ms = regular_ball_shell_mesh(
    &geotrans_descriptor("GT_QK(2,2)").unwrap(),
    &[0.0, 0.0, 0.0],
    rad,
    0.0,
    [nc, 1],
    2,
    false,
  )
  .unwrap();
  let rot = vec![-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
  ms.transform(&Tensor::from_data(&[3, 3], rot).unwrap())
    .unwrap();
  ms.translate(&[sx, 0.0, zc]).unwrap();
  let (lo, hi) = ([sx - b, -1e-5, zc - b], [sx + b, 1e-5, zc + 1e-5]);
  ms.set_region(YM, &ms.outer_faces_in_box(&lo, &hi).unwrap())
    .unwrap();
  let (lo, hi) = ([sx - b, -1e-5, zc - 1e-5], [sx + b, b, zc + 1e-5]);
  ms.set_region(ZM, &ms.outer_faces_in_box(&lo, &hi).unwrap())
    .unwrap();
  let mut all = RegionSet::new();
  for cv in ms.convex_index() {
    all.add(cv);
  }
  ms.set_region_set(SHELL, all).unwrap();

  let mfb3 = MeshFem::new(&mb, 3)
    .with_classical_finite_element(2)
    .unwrap();
  let fixed: BTreeSet<usize> = mfb3
    .basic_dof_on_region(YM)
    .unwrap()
    .into_iter()
    .filter(|d| d % 3 == 1) // y on y = 0
    .chain(
      mfb3
        .basic_dof_on_region(ZM)
        .unwrap()
        .into_iter()
        .filter(|d| d % 3 != 1),
    ) // x and z on the equator
    .collect();
  let kept = (0..mfb3.nb_dof().unwrap())
    .filter(|d| !fixed.contains(d))
    .collect();
  let mfub = MeshFem::partial(&mfb3, &kept).unwrap();
  let mfp = MeshFem::new(&mb, 1)
    .with_classical_finite_element(1)
    .unwrap();
  let mfs3 = MeshFem::new(&ms, 3)
    .with_classical_finite_element(2)
    .unwrap();
  let fixed = mfs3.basic_dof_on_region(YM).unwrap(); // y on y = 0
  let kept = (0..mfs3.nb_dof().unwrap())
    .filter(|d| d % 3 != 1 || !fixed.contains(d))
    .collect();
  let mfus = MeshFem::partial(&mfs3, &kept).unwrap();
  let mfw = MeshFem::new(&ms, 1)
    .with_classical_finite_element(2)
    .unwrap();
  let mimb = MeshIm::new(&mb).with_integration_method(5).unwrap();
  let mims = MeshIm::new(&ms).with_integration_method(5).unwrap();

  let mut md = Model::new();
  md.add_scalar_data("K", e / (3.0 * (1.0 - 2.0 * nu)))
    .unwrap();
  md.add_scalar_data("G", e / (2.0 * (1.0 + nu))).unwrap();
  md.add_scalar_data("T", th).unwrap();
  // augmentation, low for thin shells
  md.add_scalar_data("r", e * th / (rad * rad)).unwrap();
  md.add_scalar_data("delta", 0.0).unwrap();
  // solid
  md.add_fem_variable("ub", &mfub).unwrap();
  md.add_fem_variable("p", &mfp).unwrap();
  md.add_macro("Fb", "Id(3)+Grad_ub").unwrap();
  md.add_macro("tau_b", "p*Id(3)+G*pow(Det(Fb),-2/3)*Deviator(Fb*Fb')")
    .unwrap();
  md.add_nonlinear_term(&mimb, None, "(tau_b*Inv(Fb)'):Grad_Test_ub")
    .unwrap();
  md.add_nonlinear_term(&mimb, None, "(p-K*log(Det(Fb)))*Test_p")
    .unwrap();
  // shell
  md.add_fem_variable("us", &mfus).unwrap();
  md.add_fem_variable("d", &mfus).unwrap();
  md.add_fem_variable("w", &mfw).unwrap();
  md.add_macro("F0", "Id(3)+Grad_us+d@Normal").unwrap();
  md.add_macro(
    "W0",
    "(K/2*sqr(log(Det(F0)))+G/2*(pow(Det(F0),-2/3)*Norm_sqr(F0)-3))",
  )
  .unwrap();
  md.add_macro("A(xx)", "(Id(3)+xx*element_curvature)")
    .unwrap();
  md.add_macro(
    "F(xx)",
    "(Id(3)+Grad_us+xx*(1+xx*w)*(element_curvature+Grad_d)\
     +xx*xx*(Normal+d)@Grad_w\
     +(d+2*xx*w*(Normal+d))@Normal)*Inv(A(xx))",
  )
  .unwrap();
  md.add_macro("J(xx)", "Det(F(xx))").unwrap();
  md.add_macro(
    "W(xx)",
    "Det(A(xx))*(K/2*sqr(log(J(xx)))+G/2*(pow(J(xx),-2/3)*Norm_sqr(F(xx))-3))",
  )
  .unwrap();
  md.add_macro(
    "tau(xx)",
    "K*log(J(xx))*Id(3)+G*pow(J(xx),-2/3)*Deviator(F(xx)*F(xx)')",
  )
  .unwrap();
  md.add_scalar_data("c0", th / 2.0 * 8.0 / 9.0).unwrap();
  md.add_scalar_data("c", th / 2.0 * 5.0 / 9.0).unwrap();
  md.add_scalar_data("xi", th / 2.0 * (3.0f64 / 5.0).sqrt())
    .unwrap();
  md.add_nonlinear_term(&mims, None, "c0*W0+c*(W(-xi)+W(xi))")
    .unwrap();
  // the shell's equator held in x and pushed down: us(1) = 0,
  // us(3) = -delta
  let mfl = MeshFem::new(&ms, 2)
    .with_classical_finite_element(2)
    .unwrap();
  md.add_filtered_fem_variable("lam", &mfl, ZM).unwrap();
  md.add_nonlinear_term(&mims, ZM, "lam(1)*us(1)+lam(2)*(us(3)+delta)")
    .unwrap();
  // contact: rays from the shell along its deformed normal
  md.add_raytracing_transformation("ray", 0.5).unwrap();
  md.add_master_contact_boundary_to_raytracing_transformation("ray", &mb, "ub", EXT)
    .unwrap();
  md.add_slave_contact_boundary_to_raytracing_transformation("ray", &ms, "us", SHELL)
    .unwrap();
  md.add_macro("nx", "Normalized(Inv(Id(3)+Grad_us)'*Normal)")
    .unwrap();
  md.add_macro(
    "gap",
    "(Interpolate(X,ray)+Interpolate(ub,ray)-X-us-T/2*(1+T/2*w)*(Normal+d)).nx",
  )
  .unwrap();
  md.add_macro(
    "dgap",
    "(Interpolate(Test_ub,ray)-Test_us-T/2*(1+T/2*w)*Test_d-T/2*T/2*Test_w*(Normal+d)).nx",
  )
  .unwrap();
  let mflc = MeshFem::new(&ms, 1)
    .with_classical_finite_element(1)
    .unwrap();
  md.add_fem_variable("lc", &mflc).unwrap();
  md.add_nonlinear_term(
    &mims,
    SHELL,
    "-lc/r*Test_lc+Interpolate_filter(ray, neg_part(r*gap-lc)*(Test_lc/r-dgap), 1)",
  )
  .unwrap();
  println!(
    "{} hexahedra, {} shell elements, {} dofs",
    mb.nb_convexes(),
    ms.nb_convexes(),
    md.nb_dof()
  );

  let out = format!(
    "{}/../../target/showcase/shell_solid_contact",
    env!("CARGO_MANIFEST_DIR")
  );
  std::fs::create_dir_all(&out).unwrap();
  let mfob = MeshFem::new(&mb, 1)
    .with_classical_discontinuous_finite_element(2)
    .unwrap();
  let mfos = MeshFem::new(&ms, 1)
    .with_classical_discontinuous_finite_element(2)
    .unwrap();
  let newton = Newton {
    max_iter: 20,
    ..Default::default()
  };
  let names = ["ub", "p", "us", "d", "w", "lam", "lc"];
  let (mut t, mut dt) = (0.0, 1.0 / steps);
  let (mut pvd_shell, mut pvd_solid) = (vec![], vec![]);
  println!("  delta  Newton  contact force");
  while t < 1.0 - 1e-12 {
    dt = dt.min(1.0 - t);
    let saved: Vec<Vec<f64>> = names
      .iter()
      .map(|v| md.variable(v).unwrap().to_vec())
      .collect();
    md.set_variable("delta", &[(t + dt) * delta_max]).unwrap();
    let rep = newton.solve(&mut md, None).unwrap();
    if !rep.converged() {
      // back, half the step
      for (v, x) in names.iter().zip(&saved) {
        md.set_variable(v, x).unwrap();
      }
      dt /= 2.0;
      assert!(dt > 1e-4 / steps, "{}", rep.message);
      continue;
    }
    t += dt;
    dt = (2.0 * dt).min(1.0 / steps); // back to the nominal step
    let force = {
      let ws = md.workspace_mut();
      ws.clear_expressions();
      ws.add_expression("lc", &mims, SHELL).unwrap();
      ws.assembly(0, false).unwrap();
      2.0 * ws.assembled_potential()
    };
    println!("{:7.4}  {:6}  {force:13.5e}", t * delta_max, rep.iterations);
    let step = pvd_shell.len() + 1;
    let file = format!("solid_{step}.vtu");
    let vm = md
      .interpolation("sqrt(3/2)*Norm(Deviator(tau_b))/Det(Fb)", &mfob, None)
      .unwrap();
    let mut vtu = VtuExport::new(&format!("{out}/{file}"), Format::Compressed);
    vtu.exporting(&mfub).unwrap();
    vtu
      .write_point_data(&mfub, md.variable("ub").unwrap(), "u")
      .unwrap();
    vtu.write_point_data(&mfob, &vm, "von Mises").unwrap();
    vtu.write().unwrap();
    pvd_solid.push((t * delta_max, file));
    let file = format!("shell_{step}.vtu");
    let vm = md
      .interpolation("sqrt(3/2)*Norm(Deviator(tau(0)))/J(0)", &mfos, None)
      .unwrap();
    let mut vtu = VtuExport::new(&format!("{out}/{file}"), Format::Compressed);
    vtu.exporting(&mfus).unwrap();
    vtu
      .write_point_data(&mfus, md.variable("us").unwrap(), "u")
      .unwrap();
    vtu
      .write_point_data(&mfus, md.variable("d").unwrap(), "d")
      .unwrap();
    vtu
      .write_point_data(&mflc, md.variable("lc").unwrap(), "contact pressure")
      .unwrap();
    vtu
      .write_point_data(&mfos, &vm, "von Mises (midsurface)")
      .unwrap();
    vtu.write().unwrap();
    pvd_shell.push((t * delta_max, file));
  }
  for (name, pvd) in [("shell", &pvd_shell), ("solid", &pvd_solid)] {
    let steps: Vec<(f64, &str)> = pvd.iter().map(|(t, f)| (*t, f.as_str())).collect();
    export_to_pvd(&format!("{out}/{name}.pvd"), &steps).unwrap();
  }
  println!("written: {out}/shell.pvd, {out}/solid.pvd");
}
