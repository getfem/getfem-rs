//! M10: derivatives through an interpolate transformation that depends on
//! a variable, checked by finite differences (GetFEM crashes there).

mod common;

use getfem_fem::mesh_fem::MeshFem;
use getfem_fem::mesh_im::MeshIm;
use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::mesh::Mesh;
use getfem_geom::regular_mesh::regular_mesh;
use getfem_gwfl::workspace::Workspace;

/// the setup of gwfl_trans.cc
fn setup() -> (Mesh, Workspace, MeshIm) {
  let m = regular_mesh(
    &geotrans_descriptor("GT_PK(2,1)").unwrap(),
    &[3, 2],
    &[0.0; 2],
    &[1.0; 2],
  )
  .unwrap();
  for i in m.point_index() {
    let p = m.point(i).to_vec();
    let q = [
      p[0] + 0.05 * (2.1 * p[1] + 0.3 * p[0]).sin(),
      p[1] + 0.05 * (1.7 * p[0] + 0.2 * p[1]).cos() - 0.05,
    ];
    m.set_point(i, &q).unwrap();
  }
  let mfu = MeshFem::new(&m, 1)
    .with_classical_finite_element(2)
    .unwrap();
  let mfv = MeshFem::new(&m, 2)
    .with_classical_finite_element(1)
    .unwrap();
  let mfd = MeshFem::new(&m, 1)
    .with_classical_discontinuous_finite_element(1)
    .unwrap();
  let mim = MeshIm::new(&m).with_integration_method(4).unwrap();
  let mut ws = Workspace::new();
  ws.add_fem_variable("u", &mfu).unwrap();
  ws.add_fem_variable("v", &mfv).unwrap();
  ws.add_fem_variable("d", &mfd).unwrap();
  let nu = mfu.nb_dof().unwrap();
  let u: Vec<f64> = (0..nu).map(|i| 0.1 * (i as f64 + 1.0).sin()).collect();
  let nv = mfv.nb_dof().unwrap();
  let v: Vec<f64> = (0..nv)
    .map(|i| 0.05 * (2.0 * i as f64 + 1.0).cos())
    .collect();
  ws.set_variable("u", &u).unwrap();
  ws.set_variable("v", &v).unwrap();
  let nd = mfd.nb_dof().unwrap();
  let d: Vec<f64> = (0..nd)
    .map(|i| 0.2 * (0.7 * i as f64 + 0.3).sin())
    .collect();
  ws.set_variable("d", &d).unwrap();
  let c = "[0.5;0.45]+0.7*(X-[0.5;0.45])";
  ws.add_interpolate_transformation_from_expression("contract", &m, &m, c)
    .unwrap();
  ws.add_interpolate_transformation_from_expression("shift", &m, &m, "X+[0.13;0.07]")
    .unwrap();
  ws.add_interpolate_transformation_from_expression("move", &m, &m, &format!("{c}+0.1*v"))
    .unwrap();
  // nested: reads v through move (which depends on v)
  ws.add_interpolate_transformation_from_expression(
    "nest",
    &m,
    &m,
    &format!("{c}+0.05*Interpolate(v,move)"),
  )
  .unwrap();
  (m, ws, mim)
}

/// Derivatives through a transformation that depends on a variable (the
/// target point X' = c + 0.7 (X - c) + 0.1 v) and through a nested one
/// (reading v through it): order 1 against central differences of order 0,
/// order 2 against those of order 1.
#[test]
fn derivatives_through_a_moving_transformation() {
  let (_, mut ws, mim) = setup();
  let n = ws.nb_dof();
  let state: Vec<f64> = ["u", "v", "d"]
    .iter()
    .flat_map(|x| ws.variable(x).unwrap().to_vec())
    .collect();
  let set = |ws: &mut Workspace, x: &[f64]| {
    for name in ["u", "v", "d"] {
      let r = ws.interval_of_variable(name).unwrap();
      ws.set_variable(name, &x[r]).unwrap();
    }
  };
  let h = 1e-6;
  let mut bad = vec![];
  let potentials = [
    "Interpolate(u,move)*u",
    "Norm_sqr(Interpolate(v,move) - v)",
    "Interpolate(X,move).v",
    "Interpolate(Grad_u,move).Interpolate(v,move)",
    "sqr(Interpolate(u,move))*Interpolate(Div_v,move)",
    "Interpolate(u,nest)*u",
    "Interpolate(X,nest).Interpolate(v,nest)",
  ];
  for e in potentials {
    set(&mut ws, &state);
    let g = {
      ws.clear_expressions();
      ws.add_expression(e, &mim, None).unwrap();
      ws.assembly(1, false).unwrap();
      ws.assembled_vector()
    }
    .to_vec();
    let scale = g.iter().fold(0.0f64, |a, x| a.max(x.abs()));
    let mut worst = 0.0f64;
    for i in 0..n {
      let (mut xp, mut xm) = (state.clone(), state.clone());
      xp[i] += h;
      xm[i] -= h;
      set(&mut ws, &xp);
      let fp = {
        ws.clear_expressions();
        ws.add_expression(e, &mim, None).unwrap();
        ws.assembly(0, false).unwrap();
        ws.assembled_potential()
      };
      set(&mut ws, &xm);
      let fm = {
        ws.clear_expressions();
        ws.add_expression(e, &mim, None).unwrap();
        ws.assembly(0, false).unwrap();
        ws.assembled_potential()
      };
      worst = worst.max(((fp - fm) / (2.0 * h) - g[i]).abs());
    }
    println!("{e}: gradient vs finite differences {worst:.1e} (scale {scale:.1e})");
    if worst > 1e-6 * scale.max(1e-12) {
      bad.push(format!("{e}: gradient {worst:e} (scale {scale:e})"));
    }
  }
  let residuals = [
    "Interpolate(u,move)*Test_u",
    "Interpolate(u,move)*u*Test_u + Interpolate(v,move).Test_v",
    "Norm_sqr(Interpolate(v,move) - v)*Test_u",
    "Interpolate(X,move).v*Test_u",
    "Interpolate(u,nest)*u*Test_u + Interpolate(v,nest).Test_v",
    "Interpolate(X,nest).Interpolate(Test_v,nest)",
  ];
  for e in residuals {
    set(&mut ws, &state);
    let k = {
      ws.clear_expressions();
      ws.add_expression(e, &mim, None).unwrap();
      ws.assembly(2, false).unwrap();
      ws.assembled_matrix()
    }
    .to_csr();
    let mut worst = 0.0f64;
    let mut scale = 0.0f64;
    for j in 0..n {
      let (mut xp, mut xm) = (state.clone(), state.clone());
      xp[j] += h;
      xm[j] -= h;
      set(&mut ws, &xp);
      let rp = {
        ws.clear_expressions();
        ws.add_expression(e, &mim, None).unwrap();
        ws.assembly(1, false).unwrap();
        ws.assembled_vector()
      }
      .to_vec();
      set(&mut ws, &xm);
      let rm = {
        ws.clear_expressions();
        ws.add_expression(e, &mim, None).unwrap();
        ws.assembly(1, false).unwrap();
        ws.assembled_vector()
      };
      for i in 0..n {
        let fd = (rp[i] - rm[i]) / (2.0 * h);
        scale = scale.max(k.get(i, j).abs());
        worst = worst.max((fd - k.get(i, j)).abs());
      }
    }
    println!("{e}: tangent vs finite differences {worst:.1e} (scale {scale:.1e})");
    if worst > 1e-6 * scale.max(1e-12) {
      bad.push(format!("{e}: tangent {worst:e} (scale {scale:e})"));
    }
  }
  // as GetFEM: no derivative of a Hessian
  set(&mut ws, &state);
  ws.clear_expressions();
  let err = ws
    .add_expression("Interpolate(Grad_u,move).Interpolate(v,move)", &mim, None)
    .and_then(|_| ws.assembly(2, false))
    .unwrap_err()
    .to_string();
  assert!(err.contains("hessian"), "{err}");
  assert!(bad.is_empty(), "{}", bad.join("\n"));
}
