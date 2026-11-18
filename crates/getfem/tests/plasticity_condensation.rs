//! The finite strain plasticity model of Section 6 of the condensation
//! draft (ref/internal-variable-condensation-draft-9.pdf): Hencky law,
//! logarithmic J2 flow with the non-smooth yield residual (14), the data
//! Cpi and gam_p updated after each step (15). Its Table 2: the internal
//! variables condensed (Option 3) against retained (Option 2), in 3D with
//! the multiplier xi, and in plane stress with xi and the out of plane
//! strain e33 in one cluster of two.

use getfem::fem::im_data::ImData;
use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im::MeshIm;
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::region::{outer_faces_of_mesh, select_faces_of_normal};
use getfem::geom::regular_mesh::regular_unit_mesh;
use getfem::model::{Model, Newton};

const K: f64 = 10.0;
const G: f64 = 5.0;
const SY: f64 = 0.05;
const H: f64 = 0.5;

fn l1(a: &[f64], b: &[f64]) -> f64 {
  a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

/// per load step: u, xi, e33 (empty in 3D), Newton iterations
type Steps = Vec<(Vec<f64>, Vec<f64>, Vec<f64>, usize)>;

/// plane: plane stress (2D, e33), else 3D; internal: condensed
fn run(plane: bool, internal: bool, loads: &[f64]) -> Steps {
  let n = if plane { 2 } else { 3 };
  let m = if plane {
    regular_unit_mesh(&[3, 3], &geotrans_descriptor("GT_QK(2,1)").unwrap()).unwrap()
  } else {
    regular_unit_mesh(&[2, 2, 2], &geotrans_descriptor("GT_QK(3,1)").unwrap()).unwrap()
  };
  // clamped at x(n) = 0, pulled along e_n at x(n) = 1
  let outer = outer_faces_of_mesh(&m);
  let mut dir = vec![0.0; n];
  dir[n - 1] = -1.0;
  m.set_region(1, &select_faces_of_normal(&m, &outer, &dir, 0.01).unwrap())
    .unwrap();
  dir[n - 1] = 1.0;
  m.set_region(2, &select_faces_of_normal(&m, &outer, &dir, 0.01).unwrap())
    .unwrap();
  let mf = MeshFem::new(&m, n)
    .with_classical_finite_element(1)
    .unwrap();
  let mim = MeshIm::new(&m).with_integration_method(3).unwrap();
  let mimd1 = ImData::new(&mim, -1, &[]).unwrap();
  let mimd33 = ImData::new(&mim, -1, &[3, 3]).unwrap();
  let np = mimd1.nb_points().unwrap();

  let mut md = Model::new();
  md.add_fem_variable("u", &mf).unwrap();
  let add = |md: &mut Model, v: &str, imd: &ImData| {
    if internal {
      md.add_internal_im_variable(v, imd).unwrap();
    } else {
      md.add_im_variable(v, imd).unwrap();
    }
  };
  add(&mut md, "xi", &mimd1);
  if plane {
    add(&mut md, "e33", &mimd1);
  }
  let id: Vec<f64> = (0..np)
    .flat_map(|_| [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    .collect();
  md.add_im_data("Cpi", &mimd33, &id).unwrap();
  md.add_im_data("gam_p", &mimd1, &vec![0.0; np]).unwrap();
  for (k, v) in [("K", K), ("G", G), ("sy", SY), ("H", H), ("d", 0.0)] {
    md.add_scalar_data(k, v).unwrap();
  }
  if plane {
    md.add_macro(
      "F",
      "[1+Grad_u(1,1), Grad_u(1,2), 0; Grad_u(2,1), 1+Grad_u(2,2), 0; 0, 0, exp(e33)]",
    )
    .unwrap();
  } else {
    md.add_macro("F", "Id(3)+Grad(u)").unwrap();
  }
  for (k, e) in [
    ("h_tr", "Logm(F*Cpi*F')"),
    ("n_tr", "Deviator(h_tr)"),
    ("h_e", "h_tr-xi*n_tr"),
    ("tau", "K*log(Det(F))*Id(3)+G*Deviator(h_e)"),
    ("gam", "gam_p+xi*Norm(n_tr)"),
  ] {
    md.add_macro(k, e).unwrap();
  }
  // the global equation (in plane stress on the in-plane block of P)
  let p = if plane {
    "([1,0,0;0,1,0]*(tau*Inv(F)')*[1,0;0,1;0,0])"
  } else {
    "(tau*Inv(F)')"
  };
  md.add_term(&mim, None, &format!("{p}:Grad(Test_u)"))
    .unwrap();
  md.add_term(
    &mim,
    None,
    "(Norm(G*Deviator(h_e)/Det(F))-min(sqrt(2/3)*(sy+H*gam),Norm(G*n_tr/Det(F))))*Test_xi",
  )
  .unwrap();
  if plane {
    md.add_term(&mim, None, "tau(3,3)*Test_e33").unwrap();
  }
  // displacements by multipliers: 0 at region 1, d e_n at region 2
  let target = if plane { "[0;d]" } else { "[0;0;d]" };
  md.add_filtered_fem_variable("m1", &mf, 1).unwrap();
  md.add_term(&mim, 1, "u.m1").unwrap();
  md.add_filtered_fem_variable("m2", &mf, 2).unwrap();
  md.add_term(&mim, 2, &format!("(u-{target}).m2")).unwrap();

  let newton = Newton {
    atol: 1e-13,
    rtol: 1e-12,
    ..Default::default()
  };
  let mut out = vec![];
  for (k, &d) in loads.iter().enumerate() {
    md.set_variable("d", &[d]).unwrap();
    // this GWFL formulation has no derivative of the yield residual
    // w.r.t. xi at u = 0 (n_tr = 0): the first step from an affine stretch
    if k == 0 {
      let stretch = if plane { "[0;d*X(2)]" } else { "[0;0;d*X(3)]" };
      let u0 = md.interpolation(stretch, &mf, -1).unwrap();
      md.set_variable("u", &u0).unwrap();
    }
    let rep = newton.solve(&mut md, None).unwrap();
    assert!(
      rep.converged(),
      "plane {plane}, internal {internal}, d {d}: {}",
      rep.message
    );
    // from_variables: without the internal variables (the default)
    let nint = if internal {
      np * (1 + plane as usize)
    } else {
      0
    };
    assert_eq!(md.from_variables().len(), md.nb_dof() - nint);
    // the data of the next step (15)
    let cpi = "Inv(F)*Expm(h_e)*Inv(F)'";
    md.set_variable("Cpi", &md.interpolation(cpi, &mimd33, -1).unwrap())
      .unwrap();
    md.set_variable("gam_p", &md.interpolation("gam", &mimd1, -1).unwrap())
      .unwrap();
    let v = |n: &str| md.variable(n).map_or(vec![], |x| x.to_vec());
    out.push((v("u"), v("xi"), v("e33"), rep.iterations));
  }
  out
}

fn table_row(plane: bool) {
  let loads = [0.005, 0.01, 0.015, 0.02];
  let (ret, el) = (run(plane, false, &loads), run(plane, true, &loads));
  for (k, (r, e)) in ret.iter().zip(&el).enumerate() {
    let (du, dxi, de) = (l1(&r.0, &e.0), l1(&r.1, &e.1), l1(&r.2, &e.2));
    eprintln!(
      "{} step {k}: |u| {:.3e} |xi| {:.3e}; l1 u {du:.2e} xi {dxi:.2e} e33 {de:.2e}; Newton {} / {}",
      if plane { "plane stress" } else { "3D" },
      r.0.iter().map(|x| x.abs()).sum::<f64>(),
      r.1.iter().map(|x| x.abs()).sum::<f64>(),
      r.3,
      e.3
    );
    // the same discrete problem: the same solution, the same iterations
    assert!(
      du < 1e-9 && dxi < 1e-9 && de < 1e-9,
      "step {k}: {du:e} {dxi:e} {de:e}"
    );
    assert_eq!(r.3, e.3, "step {k}: iterations");
  }
  // plastic flow happened
  assert!(ret.last().unwrap().1.iter().any(|&x| x > 1e-6));
}

#[test]
fn table2_three_dimensional_one_internal_variable() {
  table_row(false);
}

#[test]
fn table2_plane_stress_cluster_of_two() {
  table_row(true);
}
