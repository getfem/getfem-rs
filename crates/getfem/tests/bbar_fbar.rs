//! B-bar and F-bar with the projected gradient
//! elementary transformation: G0 = Elementary_transformation(u, mean, G), the
//! element mean of Grad_u on a P0 space G.
//! - exact for affine fields on distorted quadrilaterals;
//! - B-bar Q1 does not lock for nearly incompressible plane strain (plain
//!   Q1 does), compared with a fine Q2 solution;
//! - F-bar: tangent by finite differences, F-bar = F for homogeneous states.

use getfem::fem::fem::fem_descriptor;
use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im::MeshIm;
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::mesh::Mesh;
use getfem::geom::regular_mesh::{noise_unit_mesh, regular_unit_mesh};
use getfem::gwfl::workspace::Workspace;
use getfem::model::{Model, Newton};

/// Q1 (or Qk) displacement, element mean of its gradient on P0 (2x2)
fn spaces(mesh: &Mesh, k: usize) -> (MeshFem, MeshFem, MeshIm) {
  let mf_u = MeshFem::new(mesh, 2)
    .with_classical_finite_element(k)
    .unwrap();
  let mf_g = MeshFem::new(mesh, 1);
  mf_g.set_qdim([2, 2]);
  mf_g
    .set_finite_element(&fem_descriptor("FEM_QK(2,0)").unwrap())
    .unwrap();
  let mim = MeshIm::new(mesh)
    .with_integration_method(2 * k + 2)
    .unwrap();
  (mf_u, mf_g, mim)
}

fn quads(n: usize, noised: bool) -> Mesh {
  let m = regular_unit_mesh(&[n, n], &geotrans_descriptor("GT_QK(2,1)").unwrap()).unwrap();
  if noised {
    noise_unit_mesh(&m).unwrap();
  }
  m
}

#[test]
fn projected_gradient_is_exact_for_affine_fields() {
  let mesh = quads(4, true);
  let (mf_u, mf_g, mim) = spaces(&mesh, 1);
  let mut ws = Workspace::new();
  ws.add_fem_variable("u", &mf_u).unwrap();
  ws.add_fem_data("G", &mf_g, &vec![0.0; mf_g.nb_dof().unwrap()])
    .unwrap();
  ws.add_projected_gradient("mean").unwrap();
  // u = A x + b
  let u = ws
    .interpolation(
      "[0.3*X(1) - 0.2*X(2) + 0.1, 0.05*X(1) + 0.4*X(2) - 0.2]",
      &mf_u,
      -1,
    )
    .unwrap();
  ws.set_variable("u", &u).unwrap();
  let e = {
    ws.clear_expressions();
    ws.add_expression(
      "Norm_sqr(Elementary_transformation(u, mean, G) - Grad_u)",
      &mim,
      None,
    )
    .unwrap();
    ws.assembly(0, false).unwrap();
    ws.assembled_potential()
  };
  let g = {
    ws.clear_expressions();
    ws.add_expression("Norm_sqr(Grad_u)", &mim, None).unwrap();
    ws.assembly(0, false).unwrap();
    ws.assembled_potential()
  };
  assert!(e <= 1e-28 * g, "{e} vs {g}");
}

/// mean vertical displacement of the right edge of a plane strain square
/// clamped on the left, sheared on the right (E = 1, nu = 0.4999)
fn tip(n: usize, k: usize, bbar: bool) -> f64 {
  let (e, nu) = (1.0, 0.4999);
  let mesh = quads(n, false);
  let (mf_u, mf_g, mim) = spaces(&mesh, k);
  let right = mesh.outer_faces_with_direction(&[1.0, 0.0], 0.0).unwrap();
  mesh.set_region(1, &right).unwrap();
  let left = mesh.outer_faces_with_direction(&[-1.0, 0.0], 0.0).unwrap();
  mesh.set_region(3, &left).unwrap();
  let clamped = 3;
  let mut md = Model::new();
  md.add_fem_variable("u", &mf_u).unwrap();
  md.add_fem_data("G", &mf_g, &vec![0.0; mf_g.nb_dof().unwrap()])
    .unwrap();
  md.add_scalar_data("mu", e / (2.0 * (1.0 + nu))).unwrap();
  md.add_scalar_data("lambda", e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))
    .unwrap();
  if bbar {
    md.add_projected_gradient("mean").unwrap();
    // B-bar: the volumetric part of the gradient replaced by its element mean
    md.add_macro(
      "Gb",
      "Grad_u + ((Trace(Elementary_transformation(u, mean, G)) - Div_u)/2)*Id(2)",
    )
    .unwrap();
    md.add_macro(
      "Gbt",
      "Grad_Test_u + ((Trace(Elementary_transformation(Test_u, mean, G)) - Div_Test_u)/2)*Id(2)",
    )
    .unwrap();
  } else {
    md.add_macro("Gb", "Grad_u").unwrap();
    md.add_macro("Gbt", "Grad_Test_u").unwrap();
  }
  md.add_linear_term(
    &mim,
    None,
    "lambda*Trace(Gb)*Trace(Gbt) + 2*mu*Sym(Gb):Sym(Gbt)",
  )
  .unwrap();
  md.add_linear_term(&mim, 1, "0.001*Test_u(2)").unwrap();
  md.add_Dirichlet_condition_with_multiplier(&mim, "u", &mf_u, clamped, None)
    .unwrap();
  let rep = Newton::default().solve(&mut md, None).unwrap();
  assert!(rep.converged());
  let v = {
    let ws = md.workspace_mut();
    ws.clear_expressions();
    ws.add_expression("u(2)", &mim, 1).unwrap();
    ws.assembly(0, false).unwrap();
    ws.assembled_potential()
  };
  let l = {
    let ws = md.workspace_mut();
    ws.clear_expressions();
    ws.add_expression("1", &mim, 1).unwrap();
    ws.assembly(0, false).unwrap();
    ws.assembled_potential()
  };
  v / l
}

#[test]
fn bbar_does_not_lock() {
  let reference = tip(16, 2, false);
  let (q1, bbar) = (tip(8, 1, false), tip(8, 1, true));
  println!("tip: Q2 {reference:.4e}, Q1 {q1:.4e}, Q1 B-bar {bbar:.4e}");
  assert!(q1 / reference < 0.6, "Q1 should lock: {q1} vs {reference}");
  assert!(
    (bbar / reference - 1.0).abs() < 0.05,
    "B-bar: {bbar} vs {reference}"
  );
}

#[test]
fn fbar_tangent_and_homogeneous_states() {
  let mesh = quads(3, true);
  let (mf_u, mf_g, mim) = spaces(&mesh, 1);
  let mut ws = Workspace::new();
  ws.add_fem_variable("u", &mf_u).unwrap();
  ws.add_fem_data("G", &mf_g, &vec![0.0; mf_g.nb_dof().unwrap()])
    .unwrap();
  ws.add_projected_gradient("mean").unwrap();
  // F-bar: F scaled to the element mean volume change
  ws.add_macro("F", "(Id(2)+Grad_u)").unwrap();
  ws.add_macro("F0", "(Id(2)+Elementary_transformation(u, mean, G))")
    .unwrap();
  ws.add_macro("Fb", "(sqrt(Det(F0)/Det(F))*F)").unwrap();
  // homogeneous: Fb = F
  let homogeneous = ws
    .interpolation("[0.1*X(1) + 0.05*X(2), -0.08*X(2)]", &mf_u, -1)
    .unwrap();
  ws.set_variable("u", &homogeneous).unwrap();
  let d = {
    ws.clear_expressions();
    ws.add_expression("Norm_sqr(Fb - F)", &mim, None).unwrap();
    ws.assembly(0, false).unwrap();
    ws.assembled_potential()
  };
  assert!(d < 1e-28, "{d}");
  // tangent of a neo-Hookean-like energy in Fb against finite differences
  let u0 = ws
    .interpolation(
      "[0.1*sin(2*X(2)) + 0.05*X(1), 0.08*cos(1.5*X(1))*X(2)]",
      &mf_u,
      -1,
    )
    .unwrap();
  let e = "(Norm_sqr(Fb) - 2*log(Det(Fb)) + sqr(Det(Fb) - 1))";
  let h = 1e-6;
  ws.set_variable("u", &u0).unwrap();
  let k = {
    ws.clear_expressions();
    ws.add_expression(e, &mim, None).unwrap();
    ws.assembly(2, false).unwrap();
    ws.assembled_matrix()
  }
  .to_csr();
  let n = u0.len();
  let (mut worst, mut scale) = (0.0f64, 0.0f64);
  for j in 0..n {
    let (mut up, mut um) = (u0.clone(), u0.clone());
    up[j] += h;
    um[j] -= h;
    ws.set_variable("u", &up).unwrap();
    let rp = {
      ws.clear_expressions();
      ws.add_expression(e, &mim, None).unwrap();
      ws.assembly(1, false).unwrap();
      ws.assembled_vector()
    }
    .to_vec();
    ws.set_variable("u", &um).unwrap();
    let rm = {
      ws.clear_expressions();
      ws.add_expression(e, &mim, None).unwrap();
      ws.assembly(1, false).unwrap();
      ws.assembled_vector()
    };
    for i in 0..n {
      scale = scale.max(k.get(i, j).abs());
      worst = worst.max(((rp[i] - rm[i]) / (2.0 * h) - k.get(i, j)).abs());
    }
  }
  println!("F-bar tangent vs finite differences {worst:.1e} (scale {scale:.1e})");
  assert!(worst < 1e-6 * scale, "{worst} vs {scale}");
}
