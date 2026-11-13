//! Port of GetFEM's tests/test_condensation: the same model with its
//! integration-point variables retained (md1) and condensed (md2) gives
//! the same u, eps and p (l1 distance below 1e-9), over the difficulty
//! levels of test_condensation.pl: the variables eps and p retained or
//! internal, a linear or cubic p equation, a displacement space reduced
//! by reduce_to_basic_dof or a multiplier. (The levels from 1000 differ in
//! GetFEM's line search only; our Newton has none.)

use getfem::fem::im_data::ImData;
use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im::MeshIm;
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::region::{outer_faces_of_mesh, select_faces_of_normal};
use getfem::geom::regular_mesh::regular_unit_mesh;
use getfem::model::{Model, Newton};

const LEVELS: [usize; 16] = [
  0, 1, 10, 11, 20, 21, 30, 31, 100, 101, 110, 111, 120, 121, 130, 131,
];

fn l1(a: &[f64], b: &[f64]) -> f64 {
  a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

/// u, eps (empty below level 10), p and the residuals of md1 (retained)
/// and md2
type Sol = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

fn run(level: usize) -> [Sol; 2] {
  let (nx, ny, fem_order, im_order) = (3, 1, 1, 1);
  let m = regular_unit_mesh(&[nx, ny], &geotrans_descriptor("GT_QK(2, 2)").unwrap()).unwrap();
  let outer = outer_faces_of_mesh(&m);
  m.set_region(
    98,
    &select_faces_of_normal(&m, &outer, &[-1.0, 0.0], 0.001).unwrap(),
  )
  .unwrap();
  m.set_region(
    99,
    &select_faces_of_normal(&m, &outer, &[1.0, 0.0], 0.001).unwrap(),
  )
  .unwrap();
  let mf = MeshFem::new(&m, 2)
    .with_classical_finite_element(fem_order)
    .unwrap();
  if level % 1000 <= 99 {
    let left = mf.basic_dof_on_region(98).unwrap();
    let kept = (0..mf.nb_basic_dof().unwrap())
      .filter(|d| !left.contains(d))
      .collect();
    mf.reduce_to_basic_dof(&kept).unwrap();
  }
  let mim = MeshIm::new(&m)
    .with_integration_method(2 * im_order + 1)
    .unwrap();
  let (mimd1, mimd3) = (
    ImData::new(&mim, -1, &[]).unwrap(),
    ImData::new(&mim, -1, &[3]).unwrap(),
  );
  let with_eps = level % 100 > 9;
  let (expr_a, expr_b) = if with_eps {
    (
      "(-p*Id(2)+2*G*([[2/3,0;0,-1/3],[-1/3,0;0,2/3],[0,1;1,0]].eps)):Grad_Test_u",
      "(eps-Grad_u:[[1,0;0,0],[0,0;0,1],[0,0.5;0.5,0]]).Test_eps",
    )
  } else {
    ("(-p*Id(2)+2*G*(Sym(Grad_u)-Div_u*Id(2)/3)):Grad_Test_u", "")
  };
  let expr_c = if level.is_multiple_of(10) {
    "(p+K*Div_u)*Test_p"
  } else {
    "((p+1e3*pow(p,3))+K*Div_u)*Test_p"
  };
  let model = |second: bool| {
    let mut md = Model::new();
    md.add_fem_variable("u", &mf).unwrap();
    let eps_internal = second && level % 100 > 19;
    let p_internal = second && !(20..=29).contains(&(level % 100));
    if with_eps {
      if eps_internal {
        md.add_internal_im_variable("eps", &mimd3).unwrap();
      } else {
        md.add_im_variable("eps", &mimd3).unwrap();
      }
    }
    if p_internal {
      md.add_internal_im_variable("p", &mimd1).unwrap();
    } else {
      md.add_im_variable("p", &mimd1).unwrap();
    }
    md.add_scalar_data("G", 1.0).unwrap();
    md.add_scalar_data("K", 1.0).unwrap();
    md.add_term(&mim, None, expr_a).unwrap();
    if with_eps {
      md.add_term(&mim, None, expr_b).unwrap();
    }
    md.add_term(&mim, None, expr_c).unwrap();
    if level % 1000 > 99 {
      md.add_filtered_fem_variable("dirmult", &mf, 98).unwrap();
      md.add_term(&mim, 98, "u.dirmult").unwrap();
    }
    md.add_term(&mim, 99, "1e-3*Test_u(2)").unwrap();
    let newton = Newton {
      atol: 1e-14,
      rtol: 1e-13,
      ..Default::default()
    };
    let rep = newton.solve(&mut md, None).unwrap();
    assert!(
      rep.converged(),
      "level {level}, model {}: {}",
      1 + second as usize,
      rep.message
    );
    let v = |n: &str| md.variable(n).map_or(vec![], |x| x.to_vec());
    (v("u"), v("eps"), v("p"), rep.residuals)
  };
  [model(false), model(true)]
}

#[test]
fn condensed_equals_retained() {
  for level in LEVELS {
    let [(u1, e1, p1, r1), (u2, e2, p2, r2)] = run(level);
    assert!(
      u1.iter().any(|x| x.abs() > 1e-6),
      "level {level}: no displacement"
    );
    let (du, de, dp) = (l1(&u1, &u2), l1(&e1, &e2), l1(&p1, &p2));
    assert!(
      du < 1e-9 && de < 1e-9 && dp < 1e-9,
      "level {level}: u {du:e} eps {de:e} p {dp:e}"
    );
    // the same iterates (the draft, 4.5): the same residuals until
    // rounding; quadratic convergence (a weight error would make it
    // linear, SPEC 9.4)
    assert_eq!(r1.len(), r2.len(), "level {level}: {r1:?} {r2:?}");
    for k in (0..r1.len()).filter(|&k| r1[k] > 1e-12 * r1[0]) {
      assert!(
        (r1[k] - r2[k]).abs() <= 1e-6 * r1[k],
        "level {level}: {r1:?} {r2:?}"
      );
    }
    if level % 2 == 1 {
      assert!(
        r2.len() >= 3 && r2[2] < 1e2 * r2[1] * r2[1],
        "level {level}: {r2:?}"
      );
    }
  }
}

#[test]
fn linear_with_internal_variable_one_step() {
  // the internal rows count by their own block: converged after one
  // step (they had no weight and needed residuals of exactly 0)
  let m = regular_unit_mesh(&[4, 4], &geotrans_descriptor("GT_QK(2,1)").unwrap()).unwrap();
  m.set_region(1, &outer_faces_of_mesh(&m)).unwrap();
  let mf = MeshFem::new(&m, 1)
    .with_classical_finite_element(1)
    .unwrap();
  let mim = MeshIm::new(&m).with_integration_method(2).unwrap();
  let imd = ImData::new(&mim, -1, &[]).unwrap();
  let mut md = Model::new();
  md.add_fem_variable("u", &mf).unwrap();
  md.add_internal_im_variable("q", &imd).unwrap();
  md.add_term(
    &mim,
    None,
    "Grad_u.Grad_Test_u + q*Test_u - Test_u + (q - 2*u)*Test_q",
  )
  .unwrap();
  md.add_Dirichlet_condition_with_multiplier(&mim, "u", &mf, 1, None)
    .unwrap();
  let rep = Newton::default().solve(&mut md, None).unwrap();
  assert!(rep.converged() && rep.iterations == 1, "{rep:?}");
}
