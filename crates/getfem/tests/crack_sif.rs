//! Stress intensity factors in 2D: a centre crack of half length 1 at 30
//! degrees in [-8, 8]^2, a unit traction on y = +-8 (plane strain), Q1
//! h = 0.2, Heaviside and the crack tip functions on the tip elements;
//! against sqrt(pi a) (cos^2, sin cos) of the infinite plate. The same
//! crack loaded on its lips by the tractions of the uncracked plate (sigma
//! = e_y e_y, normal and shear on the crack) instead: by superposition the
//! same factors.

use getfem::fem::crack_front::crack_fronts;
use getfem::fem::crack_tip_function::{CrackTipFunction, Singular};
use getfem::fem::global_function::GlobalFunction;
use getfem::fem::integration::integ_descriptor;
use getfem::fem::level_set::LevelSet;
use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im_level_set::{CELL_TOLERANCE, IntegrateWhere, MeshImLevelSet};
use getfem::fem::mesh_level_set::MeshLevelSet;
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::region::{outer_faces_of_mesh, select_faces_of_normal};
use getfem::geom::regular_mesh::regular_mesh;
use getfem::gwfl::crack_sif::{CrackedBody, LipTractions, crack_sifs};
use getfem::model::{Model, Newton};

#[test]
fn inclined_centre_crack() {
  let (l, h, a) = (8.0, 0.2, 1.0);
  let (c, s) = (3f64.sqrt() / 2.0, 0.5);
  let n = (2.0 * l / h) as usize;
  let mesh = regular_mesh(
    &geotrans_descriptor("GT_QK(2,1)").unwrap(),
    &[n, n],
    &[-l, -l],
    &[2.0 * l, 2.0 * l],
  )
  .unwrap();
  let border = outer_faces_of_mesh(&mesh);
  for (rg, nn) in [(1, [0.0, 1.0]), (2, [0.0, -1.0])] {
    mesh
      .set_region(
        rg,
        &select_faces_of_normal(&mesh, &border, &nn, 0.5).unwrap(),
      )
      .unwrap();
  }
  let ls = LevelSet::new(&mesh, 2, true).unwrap();
  let mf = ls.mesh_fem();
  let pts: Vec<Vec<f64>> = (0..mf.nb_dof().unwrap())
    .map(|i| mf.point_of_basic_dof(i).unwrap())
    .collect();
  ls.set_values(
    0,
    &pts.iter().map(|x| -s * x[0] + c * x[1]).collect::<Vec<_>>(),
  )
  .unwrap();
  ls.set_values(
    1,
    &pts
      .iter()
      .map(|x| (c * x[0] + s * x[1]).abs() - a)
      .collect::<Vec<_>>(),
  )
  .unwrap();
  let mut mls = MeshLevelSet::new(&mesh);
  mls.add_level_set(&ls);
  mls.adapt().unwrap();
  let pu_mf = MeshFem::new(&mesh, 1)
    .with_classical_finite_element(1)
    .unwrap();
  let mfls = MeshFem::from_level_set(&mls, &pu_mf).unwrap();
  let funcs: Vec<GlobalFunction> = Singular::LEFM
    .iter()
    .map(|&f| GlobalFunction::on_level_set(&ls, &CrackTipFunction::crack(f)).unwrap())
    .collect();
  let global_mf = MeshFem::from_global_functions(&mesh, &funcs, None).unwrap();
  let mut pu_dofs = std::collections::BTreeSet::new();
  for &cv in mls.crack_tip_convexes().unwrap() {
    pu_dofs.extend(pu_mf.ind_basic_dof_of_element(cv).unwrap());
  }
  let sing = MeshFem::from_product(&pu_mf, &global_mf, Some(&pu_dofs)).unwrap();
  let base = MeshFem::from_sum(&[&sing, &mfls]).unwrap();
  let mim = MeshImLevelSet::new(&mls, IntegrateWhere::All).unwrap();
  let quasi_polar = integ_descriptor("IM_GAUSS_PARALLELEPIPED(2,6)").unwrap();
  mim
    .set_simplex_im(
      &integ_descriptor("IM_TRIANGLE(6)").unwrap(),
      Some(&quasi_polar),
      CELL_TOLERANCE,
    )
    .unwrap();
  mim
    .set_integration_method(&integ_descriptor("IM_GAUSS_PARALLELEPIPED(2,4)").unwrap())
    .unwrap();
  mim.adapt().unwrap();
  // the lips, graded toward the tips
  let lips = MeshImLevelSet::new(&mls, IntegrateWhere::Boundary).unwrap();
  lips
    .set_simplex_im(
      &integ_descriptor("IM_TRIANGLE(6)").unwrap(),
      Some(&quasi_polar),
      CELL_TOLERANCE,
    )
    .unwrap();
  lips.set_integration_method(1usize).unwrap();
  lips.adapt().unwrap();
  let kept = base.dofs_with_support(&mim, 1e-8).unwrap();
  base.set_qdim(2);
  let mf_u = MeshFem::partial(
    &base,
    &kept.iter().flat_map(|&i| [2 * i, 2 * i + 1]).collect(),
  )
  .unwrap();
  let (e, nu) = (1.0, 0.3);
  let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
  let mu = e / (2.0 * (1.0 + nu));
  let tractions = LipTractions {
    mim: &lips,
    plus: "-[0,0;0,1]*Normal",
    minus: "[0,0;0,1]*Normal",
  };
  let mut ks = vec![];
  for on_lips in [false, true] {
    let mut md = Model::new();
    md.add_fem_variable("u", &mf_u).unwrap();
    md.add_scalar_data("lambda", lambda).unwrap();
    md.add_scalar_data("mu", mu).unwrap();
    // a weak spring against the rigid motions
    md.add_linear_term(
      &mim,
      None,
      "lambda*Div_u*Div_Test_u + mu*(Grad_u + Grad_u'):Grad_Test_u + 1e-9*u.Test_u",
    )
    .unwrap();
    if on_lips {
      md.add_linear_term(
        &lips,
        None,
        &format!(
          "({}).Xfem_plus(Test_u) + ({}).Xfem_minus(Test_u)",
          tractions.plus, tractions.minus
        ),
      )
      .unwrap();
    } else {
      for rg in [1, 2] {
        md.add_linear_term(&mim, Some(&mesh.region(rg).unwrap()), "-Normal.Test_u")
          .unwrap();
      }
    }
    Newton::default().solve(&mut md, None).unwrap();
    let u = md.variable("u").unwrap();
    let body = CrackedBody {
      mim: &mim,
      mf_u: &mf_u,
      u,
      ls: &ls,
      young: e,
      poisson: nu,
      lips: on_lips.then_some(tractions),
    };
    let mut sifs: Vec<_> = crack_fronts(&ls, None)
      .unwrap()
      .iter()
      .flat_map(|f| crack_sifs(&body, 0.5, f).unwrap())
      .collect();
    sifs.sort_by(|p, q| p.point[0].total_cmp(&q.point[0]));
    assert_eq!(sifs.len(), 2);
    ks.push(sifs);
  }
  let k0 = (std::f64::consts::PI * a).sqrt();
  for (sifs, k) in ks.iter().zip(["remote", "lips"]) {
    for (t, sg) in sifs.iter().zip([-1.0, 1.0]) {
      // K_II relative to e2 = grad phi: opposite at the two tips
      let (ki, kii) = (t.k[0], sg * t.k[1]);
      // measured: 1.6 % and 1.2 % above, partly the finite plate
      assert!((ki / (k0 * c * c) - 1.0).abs() < 0.025, "{k}: K_I {ki}");
      assert!((kii / (k0 * s * c) - 1.0).abs() < 0.025, "{k}: K_II {kii}");
    }
  }
  // measured: 5e-5 apart
  for (p, q) in ks[0].iter().zip(&ks[1]) {
    for m in 0..2 {
      assert!(
        (p.k[m] - q.k[m]).abs() < 2e-3 * k0,
        "remote {:?} vs lips {:?}",
        p.k,
        q.k
      );
    }
  }
}
