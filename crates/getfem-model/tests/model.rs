//! Model: Laplace with Dirichlet by elimination, multipliers and penalty
//! (P2, exact quadratic solution), and a nonlinear diffusion with Newton.

use getfem_fem::im_data::ImData;
use getfem_fem::mesh_fem::MeshFem;
use getfem_fem::mesh_im::MeshIm;
use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::region::outer_faces_of_mesh;
use getfem_geom::regular_mesh::regular_mesh;
use getfem_model::{Model, Newton, VariableTolerance};

const EXACT: &str = "(sqr(X(1)) + X(1)*X(2) - 2*sqr(X(2)))";

fn exact(x: &[f64]) -> f64 {
  x[0] * x[0] + x[0] * x[1] - 2.0 * x[1] * x[1]
}

fn setup(k: usize) -> (Model, MeshFem, MeshIm, usize) {
  let mesh = regular_mesh(
    &geotrans_descriptor("GT_PK(2,1)").unwrap(),
    &[4, 4],
    &[0.0; 2],
    &[1.0; 2],
  )
  .unwrap();
  let mf = MeshFem::new(&mesh, 1)
    .with_classical_finite_element(k)
    .unwrap();
  let mim = MeshIm::new(&mesh).with_integration_method(2 * k).unwrap();
  let mut md = Model::new();
  md.add_fem_variable("u", &mf).unwrap();
  // the boundary, stored as region 1
  mesh.set_region(1, &outer_faces_of_mesh(&mesh)).unwrap();
  (md, mf, mim, 1)
}

fn error(md: &mut Model, mim: &MeshIm) -> f64 {
  ({
    let ws = md.workspace_mut();
    ws.clear_expressions();
    ws.add_expression(&format!("sqr(u - {EXACT})"), mim, None)
      .unwrap();
    ws.assembly(0, false).unwrap();
    ws.assembled_potential()
  })
  .sqrt()
}

#[test]
fn laplace_dirichlet() {
  for method in ["elimination", "multiplier", "multiplier degree", "penalty"] {
    let (mut md, mf, mim, bnd) = setup(2);
    // -Laplace(EXACT) = 2
    md.add_term(&mim, None, "Grad_u.Grad_Test_u - 2*Test_u")
      .unwrap();
    match method {
      "elimination" => {
        let g: Vec<f64> = (0..mf.nb_basic_dof().unwrap())
          .map(|d| exact(&mf.point_of_basic_dof(d).unwrap()))
          .collect();
        md.add_fem_data("g", &mf, &g).unwrap();
        md.add_Dirichlet_condition_with_simplification("u", bnd, Some("g"))
          .unwrap();
      }
      "multiplier" => {
        md.add_Dirichlet_condition_with_multiplier(&mim, "u", &mf, bnd, Some(EXACT))
          .unwrap();
      }
      "multiplier degree" => {
        // GetFEM's overload: a classical space of degree 2 on the mesh of u
        md.add_Dirichlet_condition_with_multiplier(&mim, "u", 2, bnd, Some(EXACT))
          .unwrap();
      }
      _ => md
        .add_Dirichlet_condition_with_penalization(&mim, "u", 1e10, bnd, Some(EXACT))
        .unwrap(),
    }
    let rep = Newton::default().solve(&mut md, None).unwrap();
    let e = error(&mut md, &mim);
    println!(
      "{method}: {} iteration(s), L2 error {e:.2e}",
      rep.iterations
    );
    assert!(rep.converged() && rep.iterations == 1);
    assert!(e < 1e-8, "{method}: {e}");
  }
}

#[test]
fn linear_term_added_as_nonlinear() {
  let (mut md, _, mim, _) = setup(1);
  md.add_term(&mim, None, "Grad_u.Grad_Test_u").unwrap();
  md.add_nonlinear_term(&mim, None, "u*Test_u - Test_u")
    .unwrap();
  md.add_nonlinear_term(&mim, None, "Test2_u*Test_u").unwrap();
  let s = md.summary();
  assert_eq!(s.matches("linear, matrix cached").count(), 1, "{s}");
  assert_eq!(s.matches(", nonlinear,").count(), 2, "{s}");
}

#[test]
fn nonlinear_diffusion() {
  // -div((1+u^2) grad u) = f with u = EXACT (Laplace EXACT = -2),
  // Dirichlet by multipliers
  let (mut md, mf, mim, bnd) = setup(2);
  let e = EXACT;
  let f = format!("-(2*{e}*Norm_sqr(Grad({e})) + (1+sqr({e}))*(-2))");
  md.add_term(
    &mim,
    None,
    &format!("(1+sqr(u))*Grad_u.Grad_Test_u - {f}*Test_u"),
  )
  .unwrap();
  md.add_Dirichlet_condition_with_multiplier(&mim, "u", &mf, bnd, Some(e))
    .unwrap();
  let newton = Newton {
    verbose: true,
    ..Default::default()
  };
  let rep = newton.solve(&mut md, None).unwrap();
  let err = error(&mut md, &mim);
  println!("{} iterations, L2 error {err:.2e}", rep.iterations);
  assert!(rep.converged() && rep.iterations < 10);
  // P2 cannot represent the solution of the discrete nonlinear problem
  // exactly: only a discretization error
  assert!(err < 1e-3, "{err}");
}

#[test]
fn explicit_matrix_and_rhs() {
  // the Laplace problem with stiffness and source as explicit contributions
  let (mut md, mf, mim, bnd) = setup(2);
  let k = {
    let ws = md.workspace_mut();
    ws.clear_expressions();
    ws.add_expression("Grad_Test_u.Grad_Test2_u", &mim, None)
      .unwrap();
    ws.assembly(2, false).unwrap();
    ws.assembled_matrix()
  }
  .to_csr();
  let f = {
    let ws = md.workspace_mut();
    ws.clear_expressions();
    ws.add_expression("2*Test_u", &mim, None).unwrap();
    ws.assembly(1, false).unwrap();
    ws.assembled_vector()
  }
  .to_vec();
  md.add_explicit_matrix("u", "u", k).unwrap();
  md.add_explicit_rhs("u", f).unwrap();
  let g: Vec<f64> = (0..mf.nb_basic_dof().unwrap())
    .map(|d| exact(&mf.point_of_basic_dof(d).unwrap()))
    .collect();
  md.add_fem_data("g", &mf, &g).unwrap();
  md.add_Dirichlet_condition_with_simplification("u", bnd, Some("g"))
    .unwrap();
  let rep = Newton::default().solve(&mut md, None).unwrap();
  let e = error(&mut md, &mim);
  println!(
    "explicit: {} iteration(s), L2 error {e:.2e}",
    rep.iterations
  );
  assert!(rep.converged() && e < 1e-8);
}

#[test]
fn newton_tolerances_and_stall() {
  let build = || {
    let (mut md, mf, mim, bnd) = setup(2);
    let f = format!("-(2*{EXACT}*Norm_sqr(Grad({EXACT})) + (1+sqr({EXACT}))*(-2))");
    md.add_term(
      &mim,
      None,
      &format!("(1+sqr(u))*Grad_u.Grad_Test_u - {f}*Test_u"),
    )
    .unwrap();
    md.add_Dirichlet_condition_with_multiplier(&mim, "u", &mf, bnd, Some(EXACT))
      .unwrap();
    md
  };
  // absolute tolerance
  let mut md = build();
  let newton = Newton {
    atol: 1e-9,
    rtol: 0.0,
    ..Default::default()
  };
  let rep = newton.solve(&mut md, None).unwrap();
  assert!(rep.converged() && *rep.residuals.last().unwrap() <= 1e-9 && rep.tolerance == 0.0);
  // below the rounding floor: converged there, with the reason
  let mut md = build();
  let newton = Newton {
    atol: 0.0,
    rtol: 1e-30,
    ..Default::default()
  };
  let rep = newton.solve(&mut md, None).unwrap();
  let hint = rep.hint.clone().unwrap_or_default();
  println!("{:?}: {hint}", rep.status);
  assert!(rep.converged() && rep.iterations < 15 && hint.contains("round-off level"));
  // again from the solution
  let rep = newton.solve(&mut md, None).unwrap();
  println!("{:?} in {}: {:?}", rep.status, rep.iterations, rep.hint);
  assert!(rep.converged() && rep.iterations <= 2);
}

#[test]
fn singular_not_converged() {
  // pure Neumann, incompatible load: no solution; the singular step
  // (MUMPS: values ~1e15) is no rounding floor
  let (mut md, _, mim, _) = setup(1);
  md.add_term(&mim, None, "Grad_u.Grad_Test_u - Test_u")
    .unwrap();
  let rep = Newton {
    max_iter: 30,
    ..Default::default()
  }
  .solve(&mut md, None)
  .unwrap();
  println!("{:?} in {}: {}", rep.status, rep.iterations, rep.message);
  assert!(!rep.converged() && rep.iterations <= 30);
  // the methods by their readable names
  assert!(
    rep
      .message
      .contains(": full Newton (the scaled residual increased")
  );
  assert!(rep.message.contains(", then damped Newton NLEQ-ERR ("));
}

#[test]
fn newton_variable_tolerances() {
  let build = || {
    let (mut md, mf, mim, bnd) = setup(2);
    let f = format!("-(2*{EXACT}*Norm_sqr(Grad({EXACT})) + (1+sqr({EXACT}))*(-2))");
    md.add_term(
      &mim,
      None,
      &format!("(1+sqr(u))*Grad_u.Grad_Test_u - {f}*Test_u"),
    )
    .unwrap();
    md.add_Dirichlet_condition_with_multiplier(&mim, "u", &mf, bnd, Some(EXACT))
      .unwrap();
    md
  };
  // an unknown variable is an error
  let newton = Newton {
    variable_tolerances: vec![VariableTolerance::new(&["v"], 1e-10)],
    ..Default::default()
  };
  assert!(newton.solve(&mut build(), None).is_err());
  // only the variable tolerances (global test off)
  let newton = Newton {
    atol: f64::INFINITY,
    variable_tolerances: vec![
      VariableTolerance::new(&["u"], 1e-12),
      VariableTolerance::new(&["mult_on_u"], 1e-9),
    ],
    ..Default::default()
  };
  let mut md = build();
  let rep = newton.solve(&mut md, None).unwrap();
  let last = rep.residuals_per_tolerance.last().unwrap();
  println!(
    "variable tolerances: {last:?} after {} steps",
    rep.iterations
  );
  assert!(rep.converged() && last[0].1 <= 1e-12 && last[1].1 <= 1e-9 && rep.hint.is_none());
}

/// rtol = 0: no relative test, so no hint on relative reduction
#[test]
fn newton_rtol_zero_no_hint() {
  let (mut md, mf, mim, bnd) = setup(2);
  md.add_term(&mim, None, "(1+sqr(u))*Grad_u.Grad_Test_u - Test_u")
    .unwrap();
  md.add_Dirichlet_condition_with_multiplier(&mim, "u", &mf, bnd, None)
    .unwrap();
  let newton = Newton {
    atol: 1e-10,
    rtol: 0.0,
    ..Default::default()
  };
  let rep = newton.solve(&mut md, None).unwrap();
  assert!(rep.converged() && rep.hint.is_none(), "{:?}", rep.hint);
}

/// elimination data must be on the space of the variable (GetFEM:
/// mf_data == &mf_u), not on another space of the same size
#[test]
fn elimination_data_on_the_same_space() {
  use getfem_geom::regular_mesh::regular_unit_mesh;
  let mesh = regular_unit_mesh(&[2, 2], &geotrans_descriptor("GT_PK(2,1)").unwrap()).unwrap();
  mesh.set_region(1, &outer_faces_of_mesh(&mesh)).unwrap();
  let space = || {
    MeshFem::new(&mesh, 1)
      .with_classical_finite_element(1)
      .unwrap()
  };
  let (mf, other) = (space(), space());
  let mut md = Model::new();
  md.add_fem_variable("u", &mf).unwrap();
  md.add_fem_data("g", &other, &[1.0; 9]).unwrap();
  md.add_fem_data("h", &mf, &[1.0; 9]).unwrap();
  assert!(
    md.add_Dirichlet_condition_with_simplification("u", 1, Some("g"))
      .is_err()
  );
  md.add_Dirichlet_condition_with_simplification("u", 1, Some("h"))
    .unwrap();
}

/// Linear terms are assembled again when the data they read change, also
/// through a macro, and kept otherwise (GetFEM's data list of a brick):
/// -div(a grad u) = f, u = 0 on the boundary, so u scales with f / a.
#[test]
fn linear_terms_follow_their_data() {
  let (mut md, _, mim, bnd) = setup(1);
  md.add_scalar_data("a", 1.0).unwrap();
  md.add_scalar_data("f", 1.0).unwrap();
  md.add_macro("flux", "a*Grad_u").unwrap();
  md.add_linear_term(&mim, None, "flux.Grad_Test_u").unwrap();
  md.add_linear_term(&mim, None, "-f*Test_u").unwrap();
  md.add_linear_term(&mim, bnd, "1e8*a*u*Test_u").unwrap();
  let solve = |md: &mut Model, a: f64, f: f64| {
    md.set_variable("a", &[a]).unwrap();
    md.set_variable("f", &[f]).unwrap();
    Newton::default().solve(md, None).unwrap();
    md.variable("u").unwrap().to_vec()
  };
  let u1 = solve(&mut md, 1.0, 1.0);
  assert!(u1.iter().any(|v| v.abs() > 1e-3));
  for (a, f) in [(1.0, 3.0), (2.0, 3.0), (2.0, 1.0)] {
    let u = solve(&mut md, a, f);
    for (x, y) in u.iter().zip(&u1) {
      assert!(
        (x - y * f / a).abs() < 1e-10,
        "a {a}, f {f}: {x} vs {}",
        y * f / a
      );
    }
  }
}

/// set_variable from a GWFL expression: interpolated on the variable's
/// space with the values before the change, several at once; errors
/// where interpolation has no target; a variable on the same space copied
#[test]
fn set_variable_from_expression() {
  let (mut md, mf, mim, _) = setup(2);
  md.set_variable("u", EXACT).unwrap();
  let u = md.variable("u").unwrap().to_vec();
  assert_eq!(u, md.interpolation(EXACT, &mf, -1).unwrap());
  md.set_variable("u", "2*u").unwrap();
  let twice: Vec<f64> = u.iter().map(|x| 2.0 * x).collect();
  assert_eq!(md.variable("u").unwrap(), &twice[..]);
  // an im_data
  let imd = ImData::new(&mim, -1, &[]).unwrap();
  let n = imd.nb_values().unwrap();
  md.add_im_data("g", &imd, &vec![0.0; n]).unwrap();
  md.set_variable("g", "u*u").unwrap();
  assert_eq!(
    md.variable("g").unwrap(),
    &md.interpolation("u*u", &imd, -1).unwrap()[..]
  );
  // errors: components, fixed size, reduced space
  let e = md.set_variable("u", "[1;2]").unwrap_err().to_string();
  assert!(e.contains("u = [1;2]"), "{e}");
  md.add_fixed_size_data("c", &[1], &[1.0]).unwrap();
  let e = md.set_variable("c", "2").unwrap_err().to_string();
  assert!(e.contains("fixed size"), "{e}");
  // a partial space: on the basic DOFs, then reduced
  let p = MeshFem::partial(&mf, &[3, 5].into()).unwrap();
  md.add_fem_data("p", &p, &[0.0; 2]).unwrap();
  md.set_variable("p", "u").unwrap();
  assert_eq!(md.variable("p").unwrap(), &[twice[3], twice[5]]);
  // several at once, all from the old values: a swap
  md.add_fem_data("w", &mf, &vec![1.0; u.len()]).unwrap();
  md.set_variables(&[("u", "w"), ("w", "u")]).unwrap();
  assert_eq!(md.variable("w").unwrap(), &twice[..]);
  assert!(md.variable("u").unwrap().iter().all(|&x| x == 1.0));
  assert!(md.set_variables(&[("u", "1"), ("u", "2")]).is_err());
  // a variable on the same space: its values copied, any element
  // (Hermite: no interpolation), im_data and fixed size too
  let h = MeshFem::new(&mf.linked_mesh(), 1)
    .with_finite_element(&getfem_fem::fem::fem_descriptor("FEM_HERMITE(2)").unwrap())
    .unwrap();
  let n = h.nb_dof().unwrap();
  let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
  md.add_fem_data("x", &h, &x).unwrap();
  md.add_fem_data("x0", &h, &vec![0.0; n]).unwrap();
  assert!(md.set_variable("x0", "2*x").is_err());
  md.set_variable("x0", " x ").unwrap();
  assert_eq!(md.variable("x0").unwrap(), &x[..]);
  // all from the old values: a swap
  md.set_variable("x0", &vec![1.0; n][..]).unwrap();
  md.set_variables(&[("x", "x0"), ("x0", "x")]).unwrap();
  assert_eq!(md.variable("x0").unwrap(), &x[..]);
  // im_data
  md.add_im_data("g0", &imd, &vec![0.0; imd.nb_values().unwrap()])
    .unwrap();
  md.set_variable("g0", "g").unwrap();
  assert_eq!(md.variable("g0").unwrap(), md.variable("g").unwrap());
  // fixed size: the same shape only
  md.add_fixed_size_data("fa", &[2], &[1.0, 2.0]).unwrap();
  md.add_fixed_size_data("fb", &[2], &[0.0; 2]).unwrap();
  md.add_fixed_size_data("fc", &[1, 2], &[0.0; 2]).unwrap();
  md.set_variable("fb", "fa").unwrap();
  assert_eq!(md.variable("fb").unwrap(), &[1.0, 2.0]);
  assert!(md.set_variable("fc", "fa").is_err());
}
