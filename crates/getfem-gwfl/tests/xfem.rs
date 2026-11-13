//! XFEM spaces: plus/minus in one integrand, graded lips at a crack tip,
//! lips with flat sub-simplices, sides on the lips of crossing cracks.

use getfem_fem::crack_tip_function::{CrackTipFunction, Singular};
use getfem_fem::fem::fem_descriptor;
use getfem_fem::global_function::GlobalFunction;
use getfem_fem::integration::integ_descriptor;
use getfem_fem::level_set::LevelSet;
use getfem_fem::mesh_fem::MeshFem;
use getfem_fem::mesh_im_level_set::{IntegrateWhere, MeshImLevelSet};
use getfem_fem::mesh_level_set::MeshLevelSet;
use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::mesh::Mesh;
use getfem_geom::regular_mesh::regular_mesh;
use getfem_gwfl::workspace::Workspace;

type F = fn(&[f64]) -> f64;

fn level_set(m: &Mesh, k: usize, f: F) -> LevelSet {
  let ls = LevelSet::new(m, k, false).unwrap();
  let mf = ls.mesh_fem();
  let v: Vec<f64> = (0..mf.nb_dof().unwrap())
    .map(|i| f(&mf.point_of_basic_dof(i).unwrap()))
    .collect();
  ls.set_values(0, &v).unwrap();
  ls
}

fn square(gt: &str, n: usize) -> Mesh {
  regular_mesh(
    &geotrans_descriptor(gt).unwrap(),
    &[n, n],
    &[-1.0, -1.0],
    &[2.0, 2.0],
  )
  .unwrap()
}

fn crack_ls(m: &Mesh, k: usize, f0: F, f1: F) -> LevelSet {
  let ls = LevelSet::new(m, k, true).unwrap();
  let mf = ls.mesh_fem();
  let pts: Vec<Vec<f64>> = (0..mf.nb_dof().unwrap())
    .map(|i| mf.point_of_basic_dof(i).unwrap())
    .collect();
  ls.set_values(0, &pts.iter().map(|p| f0(p)).collect::<Vec<_>>())
    .unwrap();
  ls.set_values(1, &pts.iter().map(|p| f1(p)).collect::<Vec<_>>())
    .unwrap();
  ls
}

/// A difference of terms on both sides of the level set (Xfem_minus and
/// Xfem_plus of the test function in one integrand): the difference of the
/// terms, and zero when the data folds both to zero (-0 compiled before).
#[test]
fn xfem_plus_minus_in_one_integrand() {
  let m = square("GT_PK(2,1)", 6);
  let line = level_set(&m, 1, |x| x[0] - 0.13 + 0.07 * x[1]);
  let mut mls = MeshLevelSet::new(&m);
  mls.add_level_set(&line);
  mls.adapt().unwrap();
  let mf = MeshFem::new(&m, 1)
    .with_finite_element(&fem_descriptor("FEM_PK(2,1)").unwrap())
    .unwrap();
  let mfls = MeshFem::from_level_set(&mls, &mf).unwrap();
  mfls.set_qdim(2);
  let mim = MeshImLevelSet::new(&mls, IntegrateWhere::Boundary).unwrap();
  mim.set_integration_method(6usize).unwrap();
  mim.adapt().unwrap();
  let expr = "p*Normal.Xfem_minus(Test_u) - p*Normal.Xfem_plus(Test_u)";
  for p in [2.0, 0.0] {
    let mut ws = Workspace::new();
    ws.add_fem_variable("u", &mfls).unwrap();
    ws.add_scalar_data("p", p).unwrap();
    let mut vec_of = |e: &str| {
      {
        ws.clear_expressions();
        ws.add_expression(e, &mim, None).unwrap();
        ws.assembly(1, false).unwrap();
        ws.assembled_vector()
      }
      .to_vec()
    };
    let a = vec_of("p*Normal.Xfem_minus(Test_u)");
    let b = vec_of("p*Normal.Xfem_plus(Test_u)");
    let c = vec_of(expr);
    // Grad(u) inside Xfem_plus, Xfem_minus: Grad_u
    for side in ["plus", "minus"] {
      let f = vec_of(&format!("p*Normal.(Xfem_{side}(Grad(Test_u))*Normal)"));
      let g = vec_of(&format!("p*Normal.(Xfem_{side}(Grad_Test_u)*Normal)"));
      assert_eq!(f, g, "{side}");
    }
    assert_eq!(a.iter().any(|v| v.abs() > 1e-3), p != 0.0);
    for i in 0..c.len() {
      assert!(
        (c[i] - a[i] + b[i]).abs() < 1e-13,
        "p {p}, {i}: {} vs {}",
        c[i],
        a[i] - b[i]
      );
    }
  }
}

/// The lips of a crack, with a sing rule: graded toward the tip, an
/// r^(-1/2) integrand to 1e-5 (the tip segment exact); without: 1e-3 off
/// at least
#[test]
fn graded_lips_at_a_crack_tip() {
  let m = square("GT_PK(2,1)", 8);
  let ls = LevelSet::new(&m, 1, true).unwrap();
  let mf = ls.mesh_fem();
  let pts: Vec<Vec<f64>> = (0..mf.nb_dof().unwrap())
    .map(|i| mf.point_of_basic_dof(i).unwrap())
    .collect();
  ls.set_values(0, &pts.iter().map(|x| x[1] - 0.47).collect::<Vec<_>>())
    .unwrap();
  ls.set_values(1, &pts.iter().map(|x| x[0] - 0.53).collect::<Vec<_>>())
    .unwrap();
  let mut mls = MeshLevelSet::new(&m);
  mls.add_level_set(&ls);
  mls.adapt().unwrap();
  // the crack (once), x in [-1, 0.53]: int_0^1.53 r^(-1/2) dr
  let exact = 2.0 * 1.53f64.sqrt();
  let mut ws = Workspace::new();
  for (sing, tol) in [(true, 1e-5), (false, 1e-1)] {
    let mim = MeshImLevelSet::new(&mls, IntegrateWhere::Boundary).unwrap();
    let base = integ_descriptor("IM_GAUSS_PARALLELEPIPED(2,6)").unwrap();
    mim
      .set_simplex_im(
        &integ_descriptor("IM_TRIANGLE(6)").unwrap(),
        sing.then_some(&base),
        0.0,
      )
      .unwrap();
    mim.set_integration_method(1usize).unwrap();
    mim.adapt().unwrap();
    let v = {
      ws.clear_expressions();
      ws.add_expression("1/sqrt(abs(0.53 - X(1)))", &mim, None)
        .unwrap();
      ws.assembly(0, false).unwrap();
      ws.assembled_potential()
    };
    let e = (v / exact - 1.0).abs();
    assert!(e < tol && (sing || e > 1e-3), "sing {sing}: {v} vs {exact}");
  }
}

/// The lips of a crack where the cut leaves flat sub-tetrahedra in the
/// crack surface (all their nodes on it): counted once. One hexahedron of
/// the example crossing_cracks_3d at h = 10/7 (the crack y = 2.5
/// tanh(x/7) inside an ellipsoid): 3.35 before, the exact area 1.198518
/// (the level set's Q2 interpolant 2.4e-4 off)
#[test]
fn lips_with_flat_sub_simplices() {
  let (lo, hi) = (
    [10.0 / 7.0, 5.0 / 7.0, 0.0],
    [20.0 / 7.0, 15.0 / 7.0, 10.0 / 7.0],
  );
  let size: Vec<f64> = (0..3).map(|k| hi[k] - lo[k]).collect();
  let m = regular_mesh(
    &geotrans_descriptor("GT_QK(3,1)").unwrap(),
    &[1, 1, 1],
    &lo,
    &size,
  )
  .unwrap();
  let ls = LevelSet::new(&m, 2, true).unwrap();
  let mf = ls.mesh_fem();
  let pts: Vec<Vec<f64>> = (0..mf.nb_dof().unwrap())
    .map(|i| mf.point_of_basic_dof(i).unwrap())
    .collect();
  ls.set_values(
    0,
    &pts
      .iter()
      .map(|x| x[1] - 2.5 * (x[0] / 7.0).tanh())
      .collect::<Vec<_>>(),
  )
  .unwrap();
  let q = |x: &[f64]| (x[0] / 16.0).powi(2) + (x[1] / 6.0).powi(2) + (x[2] / 8.0).powi(2);
  ls.set_values(
    1,
    &pts
      .iter()
      .map(|x| (q(x).sqrt() - 1.0) * 16.0)
      .collect::<Vec<_>>(),
  )
  .unwrap();
  let mut mls = MeshLevelSet::new(&m);
  mls.add_level_set(&ls);
  mls.adapt().unwrap();
  let mim = MeshImLevelSet::new(&mls, IntegrateWhere::Boundary).unwrap();
  mim
    .set_simplex_im(&integ_descriptor("IM_TETRAHEDRON(3)").unwrap(), None, 0.0)
    .unwrap();
  mim.set_integration_method(1usize).unwrap();
  mim.adapt().unwrap();
  let mut w = Workspace::new();
  w.add_expression("1", &mim, None).unwrap();
  w.assembly(0, false).unwrap();
  let area = w.assembled_potential();
  assert!((area / 1.198518 - 1.0).abs() < 1e-3, "lip area {area}");
}

/// Two crossing cracks: on the lips of one, the crack tip function of the
/// other has no side (it is continuous there); on its own lips it jumps
/// (GetFEM forces its side at any point: a jump on both)
#[test]
fn sides_on_the_lips_of_crossing_cracks() {
  let m = square("GT_QK(2,1)", 8);
  let a = crack_ls(&m, 1, |x| x[1] - 0.13, |x| x[0].abs() - 0.6);
  let b = crack_ls(&m, 1, |x| x[0] - 0.07 - 0.2 * x[1], |x| x[1].abs() - 0.6);
  let mut mls = MeshLevelSet::new(&m);
  mls.add_level_set(&a);
  mls.add_level_set(&b);
  mls.adapt().unwrap();
  let f = GlobalFunction::on_level_set(&b, &CrackTipFunction::crack(Singular::SqrtRSin)).unwrap();
  // as the examples: the function times the Q1 hats, all of them (u = f)
  let g = MeshFem::from_global_functions(&m, &[f], None).unwrap();
  let q1 = MeshFem::new(&m, 1)
    .with_classical_finite_element(1)
    .unwrap();
  let mf = MeshFem::from_product(&q1, &g, None).unwrap();
  let mut ws = Workspace::new();
  ws.add_fem_variable("u", &mf).unwrap();
  ws.set_variable("u", &vec![1.0; mf.nb_dof().unwrap()])
    .unwrap();
  let mut jump = |lips: &str| {
    let mim = MeshImLevelSet::new(&mls, IntegrateWhere::Boundary).unwrap();
    mim.set_level_set_boolean_operations(lips).unwrap();
    mim
      .set_simplex_im(&integ_descriptor("IM_TRIANGLE(6)").unwrap(), None, 0.0)
      .unwrap();
    mim.set_integration_method(6usize).unwrap();
    mim.adapt().unwrap();
    {
      ws.clear_expressions();
      ws.add_expression("sqr(Xfem_plus(u)-Xfem_minus(u))", &mim, None)
        .unwrap();
      ws.assembly(0, false).unwrap();
      ws.assembled_potential()
    }
  };
  // its own lips: the jump 2 sqrt(r)
  assert!(jump("b") > 0.1, "{}", jump("b"));
  assert!(jump("a") < 1e-24, "{}", jump("a"));
}
