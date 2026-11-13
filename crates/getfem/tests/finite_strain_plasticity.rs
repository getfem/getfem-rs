//! Finite strain plasticity in GWFL, after GetFEM's
//! contrib/continuum_mechanics (plane strain necking with linear hardening,
//! the plastic multiplier in closed form: no internal variable) and the ksi
//! formulation of the plasticity scripts (ksi an internal variable,
//! condensed): the closed form against ksi condensed, ksi condensed
//! against retained, and continuation with both.

use std::collections::BTreeSet;

use getfem::fem::fem::fem_descriptor;
use getfem::fem::im_data::ImData;
use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im::MeshIm;
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::mesh::Mesh;
use getfem::geom::region::{RegionSet, outer_faces_of_mesh};
use getfem::geom::regular_mesh::regular_unit_mesh;
use getfem::linalg::tensor::Tensor;
use getfem::model::{
  ContinuationStatus, Method, Model, Newton, NumericalContinuation, VariableScale,
  VariableTolerance,
};

const L: f64 = 2.0 * 26.667;
const H: f64 = 2.0 * 6.413;

/// regions 1, 2: x = 0, x = max; 3: y = 0; 4: z = 0 (3D)
fn boundaries(m: &Mesh) {
  let n = m.dim();
  let mut rs = vec![RegionSet::new(); 2 * n];
  for (cv, f) in outer_faces_of_mesh(m)
    .resolve(&m.snapshot())
    .unwrap()
    .iter()
  {
    let f = f.unwrap();
    let nr = m.mean_normal_of_face_of_convex(cv, f).unwrap();
    for c in 0..n {
      if nr[c] < -0.95 {
        rs[2 * c].add_face(cv, f);
      } else if nr[c] > 0.95 {
        rs[2 * c + 1].add_face(cv, f);
      }
    }
  }
  m.set_region_set(1, rs[0].clone()).unwrap();
  m.set_region_set(2, rs[1].clone()).unwrap();
  m.set_region_set(3, rs[2].clone()).unwrap();
  if n == 3 {
    m.set_region_set(4, rs[4].clone()).unwrap();
  }
}

/// the scripts' grading and the reduced height at the center (2D)
fn thin_center(m: &Mesh) {
  let (dy, y0) = (0.2, 0.04);
  let x0 = y0 / dy;
  let b2 = (1.0 - dy) / (x0 - 1.0f64).powi(2);
  let b1 = dy - 2.0 * b2 * x0;
  let b0 = 1.0 - b1 - b2;
  let dh = 0.018 * H;
  let moves: Vec<(usize, Vec<f64>)> = (0..m.nb_points_max())
    .filter(|&i| m.is_point_valid(i))
    .map(|i| {
      let p = m.point(i);
      let t = 2.0 * p[0].abs() / L;
      let x = if t < x0 {
        p[0] * dy
      } else {
        (b0 + b1 * t + b2 * t * t) * p[0].signum() * L / 2.0
      };
      let y = p[1] - (p[1] * dh) / (2.0 * H) * (1.0 + (2.0 * std::f64::consts::PI * x / L).cos());
      (i, vec![x, y])
    })
    .collect();
  m.set_points(&moves).unwrap();
}

fn mesh(dim: usize) -> Mesh {
  let m = if dim == 2 {
    regular_unit_mesh(&[10, 5], &geotrans_descriptor("GT_QK(2,2)").unwrap()).unwrap()
  } else {
    regular_unit_mesh(&[6, 2, 2], &geotrans_descriptor("GT_QK(3,1)").unwrap()).unwrap()
  };
  let mut d = vec![0.0; dim * dim];
  d[0] = L / 2.0;
  for k in 1..dim {
    d[k * (dim + 1)] = H / 2.0;
  }
  m.transform(&Tensor::from_data(&[dim, dim], d).unwrap())
    .unwrap();
  boundaries(&m);
  if dim == 2 {
    thin_center(&m);
  }
  m
}

/// mfN without the normal components on the symmetry faces
fn symmetric_space(mf: &MeshFem, n: usize) -> MeshFem {
  let mut removed = BTreeSet::new();
  for (rg, c) in [(1, 0), (3, 1), (4, 2)].into_iter().take(n) {
    removed.extend(
      mf.basic_dof_on_region(rg)
        .unwrap()
        .into_iter()
        .filter(|d| d % n == c),
    );
  }
  let kept = (0..mf.nb_basic_dof().unwrap())
    .filter(|d| !removed.contains(d))
    .collect();
  MeshFem::partial(mf, &kept).unwrap()
}

/// tolerances in the units of each equation (forces, grip displacement),
/// also the weights of the line search merit; non-smooth: NLEQ-RES after
/// full Newton, no NLEQ-ERR
fn newton(ksi: bool) -> Newton {
  let forces: &[&str] = if ksi { &["u", "ksi"] } else { &["u"] };
  Newton {
    atol: f64::INFINITY,
    methods: vec![Method::Newton, Method::NleqRes],
    variable_tolerances: vec![
      VariableTolerance::new(forces, 1e-6),
      VariableTolerance::new(&["dirmult"], 1e-10),
    ],
    ..Default::default()
  }
}

type Steps = std::collections::BTreeMap<String, Vec<f64>>;

/// the start of a load step: u extrapolated from the two last converged
/// ones (the converged solution does not depend on it)
fn predict(md: &mut Model, prev: &mut Option<Vec<f64>>) {
  let u = md.variable("u").unwrap().to_vec();
  if let Some(p) = prev.as_ref() {
    let guess: Vec<f64> = u.iter().zip(p).map(|(a, b)| 2.0 * a - b).collect();
    md.set_variable("u", &guess).unwrap();
  }
  *prev = Some(u);
}

fn close(name: &str, key: &str, got: &[f64], want: &[f64], tol: f64) {
  assert_eq!(got.len(), want.len(), "{name} {key}");
  for (k, (a, b)) in got.iter().zip(want).enumerate() {
    assert!(
      (a - b).abs() <= tol * b.abs().max(1e-12),
      "{name} {key} step {}: {a} vs GetFEM {b}",
      k + 1
    );
  }
}

/// contrib plane strain: linear hardening, ksi in closed form; or (ksi
/// Some(internal)) ksi a variable solving the yield condition in Cauchy
/// stresses, the same solution
fn plane_strain(ksi: Option<bool>, steps: usize, nsteps: usize) -> Steps {
  let (e, nu, s0, hp, disp) = (210e3, 0.3, 5e2, 21e1, 4.0);
  let m = mesh(2);
  let mf_n = MeshFem::new(&m, 2)
    .with_classical_finite_element(2)
    .unwrap();
  let mfu = symmetric_space(&mf_n, 2);
  let mfmult = MeshFem::new(&m, 1)
    .with_classical_finite_element(2)
    .unwrap();
  let mim = MeshIm::new(&m).with_integration_method(5).unwrap();
  let mim_red = MeshIm::new(&m).with_integration_method(3).unwrap();
  let mimd1 = ImData::new(&mim, -1, &[]).unwrap();
  let mimd4 = ImData::new(&mim, -1, &[4]).unwrap();
  let np = mimd1.nb_points().unwrap();
  let mut md = Model::new();
  md.add_fem_variable("u", &mfu).unwrap();
  match ksi {
    Some(true) => md.add_internal_im_variable("ksi", &mimd1).unwrap(),
    Some(false) => md.add_im_variable("ksi", &mimd1).unwrap(),
    None => {}
  }
  md.add_scalar_data("disp", 0.0).unwrap();
  md.add_scalar_data("K", e / (3.0 * (1.0 - 2.0 * nu)))
    .unwrap();
  md.add_scalar_data("mu", e / (2.0 * (1.0 + nu))).unwrap();
  md.add_im_data("gamma0", &mimd1, &vec![0.0; np]).unwrap();
  let id: Vec<f64> = (0..np).flat_map(|_| [1.0, 1.0, 1.0, 0.0]).collect();
  md.add_im_data("invCp0vec", &mimd4, &id).unwrap();
  let (a, b) = ((2.0f64 / 3.0).sqrt() * s0, 2.0 / 3.0 * hp);
  let mut macros = vec![
    ("F", "Id(2)+Grad_u".to_string()),
    ("F3d", "Id(3)+[1,0;0,1;0,0]*Grad_u*[1,0,0;0,1,0]".into()),
    ("J", "Det(F)".into()),
    ("tauH", "K*log(J)".into()),
    (
      "invCp0",
      "[[[1,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,1,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,1]],[[0,1,0],[1,0,0],[0,0,0]]].invCp0vec".into(),
    ),
    ("devlogbetr", "Deviator(Logm(F3d*invCp0*F3d'))".into()),
    ("Y0", format!("{a}+{b}*gamma0")),
  ];
  if ksi.is_none() {
    macros.push((
      "ksi",
      format!("(1-1/max(1,mu/J*Norm(devlogbetr)/Y0))/(2+{b}/(mu/J))"),
    ));
  }
  macros.extend([
    ("gamma", "gamma0+ksi*Norm(devlogbetr)".to_string()),
    ("devlogbe", "(1-2*ksi)*devlogbetr".into()),
    ("tauD2d", "mu*[1,0,0;0,1,0]*devlogbe*[1,0;0,1;0,0]".into()),
    ("sigma", "tauH/J*Id(3)+mu/J*devlogbe".into()),
    (
      "invCp",
      "(Inv(F3d)*Expm(-ksi*devlogbetr)*(F3d))*invCp0*(Inv(F3d)*Expm(-ksi*devlogbetr)*(F3d))'"
        .into(),
    ),
  ]);
  for (k, v) in &macros {
    md.add_macro(k, v).unwrap();
  }
  md.add_term(&mim_red, None, "(tauH*Inv(F')):Grad(Test_u)")
    .unwrap();
  md.add_term(&mim, None, "(tauD2d*Inv(F')):Grad(Test_u)")
    .unwrap();
  if ksi.is_some() {
    // the yield condition in Cauchy stresses, Y linear: solved by the
    // closed form above
    md.add_term(
      &mim,
      None,
      &format!("(Norm(mu/J*devlogbe)-min({a}+{b}*gamma,Norm(mu/J*devlogbetr)+1e-12*ksi))*Test_ksi"),
    )
    .unwrap();
  }
  md.add_filtered_fem_variable("dirmult", &mfmult, 2).unwrap();
  md.add_term(&mim, 2, "(disp-u(1))*dirmult").unwrap();
  let mut res = Steps::new();
  let v = {
    let ws = md.workspace_mut();
    ws.clear_expressions();
    ws.add_expression("1", &mim, -1).unwrap();
    ws.assembly(0, false).unwrap();
    ws.assembled_potential()
  };
  let mut prev = None;
  for step in 1..=nsteps {
    md.set_variable("disp", &[disp * step as f64 / steps as f64])
      .unwrap();
    predict(&mut md, &mut prev);
    let rep = newton(ksi.is_some()).solve(&mut md, None).unwrap();
    assert!(rep.converged(), "plane strain step {step}: {}", rep.message);
    let mut s = |e: &str, r: i32| {
      let ws = md.workspace_mut();
      ws.clear_expressions();
      ws.add_expression(e, &mim, r).unwrap();
      ws.assembly(0, false).unwrap();
      ws.assembled_potential()
    };
    for (k, x) in [
      ("F", s("dirmult", 2)),
      ("gamma", s("gamma", -1) / v),
      ("s11", s("sigma(1,1)", -1) / v),
    ] {
      res.entry(k.into()).or_default().push(x);
    }
    let g = md.interpolation("gamma", &mimd1, -1).unwrap();
    let c = md.interpolation(
        "[[[1,0,0,0],[0,0,0,0.5],[0,0,0,0]],[[0,0,0,0.5],[0,1,0,0],[0,0,0,0]],[[0,0,0,0],[0,0,0,0],[0,0,1,0]]]:invCp",
        &mimd4,
        -1,
      )
      .unwrap();
    md.set_variable("gamma0", &g).unwrap();
    md.set_variable("invCp0vec", &c).unwrap();
    // ksi of the last step is the start of the next one: from ksi = 0
    // Newton fails at step 2, also with the line search
  }
  res
}

/// the ksi scripts: Eterovic-Bathe or Simo-Miehe, linear and exponential
/// hardening, ksi internal (or retained)
fn ksi_build(dim: usize, simo_miehe: bool, internal: bool) -> (Model, MeshIm, ImData, ImData) {
  let (e, nu) = (206.9e3, 0.29);
  let (s0, sinf, hp, delta) = (0.45e3, 0.715e3, 0.12924e3, 16.93);
  let n = dim;
  let m = mesh(dim);
  let fem = if n == 2 {
    "FEM_Q2_INCOMPLETE(2)"
  } else {
    "FEM_QK(3,1)"
  };
  let pf = fem_descriptor(fem).unwrap();
  let mf_n = MeshFem::new(&m, n).with_finite_element(&pf).unwrap();
  let mfu = symmetric_space(&mf_n, n);
  let mfmult = MeshFem::new(&m, 1).with_finite_element(&pf).unwrap();
  let mim = MeshIm::new(&m)
    .with_integration_method(if n == 2 { 5 } else { 3 })
    .unwrap();
  let mim_vol = MeshIm::new(&m)
    .with_integration_method(if n == 2 { 3 } else { 1 })
    .unwrap();
  let nv = if n == 2 { 4 } else { 6 };
  let mimd1 = ImData::new(&mim, -1, &[]).unwrap();
  let mimdv = ImData::new(&mim, -1, &[nv]).unwrap();
  let np = mimd1.nb_points().unwrap();
  let mut md = Model::new();
  md.add_fem_variable("u", &mfu).unwrap();
  if internal {
    md.add_internal_im_variable("ksi", &mimd1).unwrap();
  } else {
    md.add_im_variable("ksi", &mimd1).unwrap();
  }
  md.add_im_data("gamma0", &mimd1, &vec![0.0; np]).unwrap();
  let id: Vec<f64> = (0..np)
    .flat_map(|_| [1.0, 1.0, 1.0, 0.0, 0.0, 0.0][..nv].to_vec())
    .collect();
  md.add_im_data("invCp0vec", &mimdv, &id).unwrap();
  md.add_scalar_data("disp", 0.0).unwrap();
  md.add_scalar_data("K", e / (3.0 * (1.0 - 2.0 * nu)))
    .unwrap();
  md.add_scalar_data("G", e / (2.0 * (1.0 + nu))).unwrap();
  let mut macros: Vec<(&str, String)> = if n == 2 {
    vec![
      ("ToVec", "[[[1,0,0,0],[0,0,0,0.5],[0,0,0,0]],[[0,0,0,0.5],[0,1,0,0],[0,0,0,0]],[[0,0,0,0],[0,0,0,0],[0,0,1,0]]]".into()),
      ("ToMat", "[[[1,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,1,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,1]],[[0,1,0],[1,0,0],[0,0,0]]]".into()),
      ("F", "Id(2)+Grad(u)".into()),
      ("F3d", "Id(3)+[1,0;0,1;0,0]*Grad(u)*[1,0,0;0,1,0]".into()),
    ]
  } else {
    vec![
      ("ToVec", "[[[1,0,0,0,0,0],[0,0,0,0.5,0,0],[0,0,0,0,0.5,0]],[[0,0,0,0.5,0,0],[0,1,0,0,0,0],[0,0,0,0,0,0.5]],[[0,0,0,0,0.5,0],[0,0,0,0,0,0.5],[0,0,1,0,0,0]]]".into()),
      ("ToMat", "[[[1,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,1,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,1]],[[0,1,0],[1,0,0],[0,0,0]],[[0,0,1],[0,0,0],[1,0,0]],[[0,0,0],[0,0,1],[0,1,0]]]".into()),
      ("F", "Id(3)+Grad(u)".into()),
      ("F3d", "F".into()),
    ]
  };
  macros.extend([
    ("J", "Det(F)".to_string()),
    ("invCp0", "ToMat.invCp0vec".into()),
  ]);
  if simo_miehe {
    macros.extend([
      ("betr", "F3d*invCp0*F3d'".to_string()),
      ("devbetr", "Deviator(betr)".into()),
      ("tauD", "G*pow(J,-2/3)*(1-2/3*Trace(betr)*ksi)*devbetr".into()),
      ("tauDtr", "G*pow(J,-2/3)*devbetr".into()),
      ("tauD_over_G", "pow(J,-2/3)*(1-2/3*Trace(betr)*ksi)*devbetr".into()),
      ("qq", "(1-Det(tauD_over_G))/2".into()),
      ("dd", "sqr(qq)-pow(Norm_sqr(tauD_over_G)/6,3)".into()),
      (
        "invCp",
        "pow(J,2/3)*Inv(F3d)*(tauD_over_G+(pow(qq+sqrt(dd),1/3)+pow(qq-sqrt(dd),1/3))*Id(3))*Inv(F3d)'".into(),
      ),
    ]);
  } else {
    macros.extend([
      ("devbetr", "Deviator(Logm(F3d*invCp0*F3d'))".to_string()),
      ("tauD", "G*(1-2*ksi)*devbetr".into()),
      ("tauDtr", "G*devbetr".into()),
      (
        "invCp",
        "(Inv(F3d)*Expm(-ksi*devbetr)*F3d)*invCp0*(Inv(F3d)*Expm(-ksi*devbetr)*F3d)'".into(),
      ),
    ]);
  }
  let c = (2.0f64 / 3.0).sqrt();
  macros.extend([
    ("tauH", "K*log(J)".to_string()),
    (
      "tauDN",
      if n == 2 {
        "([1,0,0;0,1,0]*tauD*[1,0;0,1;0,0])".into()
      } else {
        "tauD".into()
      },
    ),
    (
      "Y(gamma)",
      format!(
        "{}+{}*(1-exp(-{}*gamma))+{}*gamma",
        c * s0,
        c * (sinf - s0),
        c * delta,
        2.0 / 3.0 * hp
      ),
    ),
    ("gamma", "gamma0+ksi*Norm(devbetr)".into()),
    ("sigma", "(tauH*Id(3)+tauD)/J".into()),
  ]);
  for (k, v) in &macros {
    md.add_macro(k, v).unwrap();
  }
  md.add_term(&mim, None, "(tauDN*Inv(F)'):Grad(Test_u)")
    .unwrap();
  md.add_term(&mim_vol, None, "tauH*Inv(F)':Grad(Test_u)")
    .unwrap();
  // +1e-12 ksi: the derivative in ksi does not vanish at u = 0
  md.add_term(
    &mim,
    None,
    "(Norm(tauD)-min(Y(gamma),Norm(tauDtr)+1e-12*ksi))*Test_ksi",
  )
  .unwrap();
  md.add_filtered_fem_variable("dirmult", &mfmult, 2).unwrap();
  md.add_term(&mim, 2, "(disp-u(1))*dirmult").unwrap();
  (md, mim, mimd1, mimdv)
}

/// the ksi model pulled in load steps (finite_strain_plasticity_ksi.py)
fn ksi_model(
  dim: usize,
  simo_miehe: bool,
  internal: bool,
  steps: usize,
  nsteps: usize,
  stretch: f64,
) -> Steps {
  let (mut md, mim, mimd1, mimdv) = ksi_build(dim, simo_miehe, internal);
  let mut res = Steps::new();
  let v = {
    let ws = md.workspace_mut();
    ws.clear_expressions();
    ws.add_expression("1", &mim, -1).unwrap();
    ws.assembly(0, false).unwrap();
    ws.assembled_potential()
  };
  let mut prev = None;
  for step in 1..=nsteps {
    md.set_variable("disp", &[stretch * step as f64 / steps as f64])
      .unwrap();
    predict(&mut md, &mut prev);
    let rep = newton(true).solve(&mut md, None).unwrap();
    assert!(rep.converged(), "dim {dim} step {step}: {}", rep.message);
    res
      .entry("iterations".into())
      .or_default()
      .push(rep.iterations as f64);
    let mut s = |e: &str, r: i32| {
      let ws = md.workspace_mut();
      ws.clear_expressions();
      ws.add_expression(e, &mim, r).unwrap();
      ws.assembly(0, false).unwrap();
      ws.assembled_potential()
    };
    for (k, x) in [
      ("F", s("dirmult", 2)),
      ("gamma", s("gamma", -1) / v),
      ("s11", s("sigma(1,1)", -1) / v),
    ] {
      res.entry(k.into()).or_default().push(x);
    }
    res
      .entry("ksi".into())
      .or_default()
      .extend(md.variable("ksi").unwrap());
    let g = md.interpolation("gamma", &mimd1, -1).unwrap();
    let c = md.interpolation("ToVec:invCp", &mimdv, -1).unwrap();
    md.set_variable("gamma0", &g).unwrap();
    md.set_variable("invCp0vec", &c).unwrap();
    let nk = md.variable("ksi").unwrap().len();
    md.set_variable("ksi", &vec![0.0; nk]).unwrap();
    let d = {
      let ws = md.workspace_mut();
      ws.clear_expressions();
      ws.add_expression("Det(invCp0)", &mim, -1).unwrap();
      ws.assembly(0, false).unwrap();
      ws.assembled_potential()
    } / v;
    res.entry("detCp".into()).or_default().push(d);
  }
  res
}

#[test]
fn plane_strain_ksi_equals_closed_form() {
  // ksi as an internal variable (and retained) solves the same yield
  // condition: the same solution as the closed form
  let closed = plane_strain(None, 25, 4);
  for internal in [true, false] {
    let r = plane_strain(Some(internal), 25, 4);
    for k in ["F", "gamma", "s11"] {
      close(
        &format!("ksi internal {internal}"),
        k,
        &r[k],
        &closed[k],
        1e-8,
      );
    }
  }
}

#[test]
fn simo_miehe_condensed_equals_retained() {
  let (c, r) = (
    ksi_model(3, true, true, 20, 3, 4.0),
    ksi_model(3, true, false, 20, 3, 4.0),
  );
  for k in ["F", "gamma", "ksi", "iterations"] {
    close("condensed vs retained", k, &c[k], &r[k], 1e-9);
  }
}

/// the ksi model by Moore-Penrose continuation in disp (the settings of
/// finite_strain_plasticity_ksi.py and its non-smooth option; u scaled by
/// 1/sqrt(n)), the history updated after each step
fn ksi_continuation(internal: bool, steps: usize) -> Steps {
  let (mut md, mim, mimd1, mimdv) = ksi_build(2, false, internal);
  let nu = md.mesh_fem_of_variable("u").unwrap().nb_dof().unwrap();
  let sq = (nu as f64).sqrt();
  let mut s = NumericalContinuation::new("disp", vec![VariableScale::new(&["u"], 1.0 / sq)]);
  (s.h_init, s.h_max, s.h_min, s.h_dec, s.h_inc) = (
    1e-1 * sq,
    1.0 * sq,
    1e-5 * sq,
    1.0 / 10f64.sqrt(),
    10f64.sqrt(),
  );
  (s.max_iter, s.thr_iter, s.max_res, s.max_diff, s.min_cos) = (12, 6, 1e-8, 1e-8, 0.995);
  (s.singularities, s.non_smooth) = (0, true);
  let v = {
    let ws = md.workspace_mut();
    ws.clear_expressions();
    ws.add_expression("1", &mim, -1).unwrap();
    ws.assembly(0, false).unwrap();
    ws.assembled_potential()
  };
  let (mut tx, mut tdisp, mut h) = (vec![], 1.0, 0.0);
  let mut res = Steps::new();
  for step in 0..=steps {
    if step == 0 {
      s.step(&mut md, &mut tx, &mut tdisp, &mut h, None).unwrap();
    } else {
      let rep = s.step(&mut md, &mut tx, &mut tdisp, &mut h, None).unwrap();
      assert_eq!(
        rep.status,
        ContinuationStatus::Converged,
        "continuation failed at step {step}"
      );
    }
    let disp = md.variable("disp").unwrap()[0];
    let mut sc = |e: &str, r: i32| {
      let ws = md.workspace_mut();
      ws.clear_expressions();
      ws.add_expression(e, &mim, r).unwrap();
      ws.assembly(0, false).unwrap();
      ws.assembled_potential()
    };
    for (k, x) in [
      ("disp", disp),
      ("h", h),
      ("F", sc("dirmult", 2)),
      ("gamma", sc("gamma", -1) / v),
    ] {
      res.entry(k.into()).or_default().push(x);
    }
    let g = md.interpolation("gamma", &mimd1, -1).unwrap();
    let c = md.interpolation("ToVec:invCp", &mimdv, -1).unwrap();
    md.set_variable("gamma0", &g).unwrap();
    md.set_variable("invCp0vec", &c).unwrap();
    let nk = md.variable("ksi").unwrap().len();
    md.set_variable("ksi", &vec![0.0; nk]).unwrap();
  }
  res
}

#[test]
fn continuation_with_ksi_retained_and_condensed() {
  // the points on the load stepping curve (steps of 0.05 in disp; past the
  // yield kink, which the linear interpolation cuts)
  let retained = ksi_continuation(false, 30);
  let ls = ksi_model(2, false, true, 100, 60, 5.0);
  let (d, f) = (&retained["disp"], &ls["F"]);
  assert!(d[30] > 2.5 && d[30] < 3.0, "disp {} after 30 steps", d[30]);
  let fmax = f.iter().fold(0.0f64, |m, v| m.max(v.abs()));
  for (i, (&di, &fi)) in d.iter().zip(&retained["F"]).enumerate() {
    if di < 0.2 {
      continue;
    }
    let k = (di / 0.05) as usize;
    let t = di / 0.05 - k as f64;
    let fl = (1.0 - t) * f[k - 1] + t * f[k];
    assert!(
      (fi - fl).abs() <= 1e-4 * fmax,
      "step {i}: F {fi} vs load stepping {fl}"
    );
  }
  let condensed = ksi_continuation(true, 30);
  for k in ["disp", "h", "F", "gamma"] {
    let scale = retained[k].iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let d = retained[k]
      .iter()
      .zip(&condensed[k])
      .map(|(a, b)| (a - b).abs())
      .fold(0.0, f64::max);
    println!("{k}: retained vs condensed {:.1e} of the scale", d / scale);
    assert!(d <= 1e-6 * scale, "{k}: condensed differs by {d:e}");
  }
}
