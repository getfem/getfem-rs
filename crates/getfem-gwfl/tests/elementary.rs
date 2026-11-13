//! M10: GWFL elementary transformations: the projection property on
//! triangles (GetFEM fails there).

use getfem_fem::mesh_fem::MeshFem;
use getfem_fem::mesh_im::MeshIm;
use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::regular_mesh::regular_mesh;
use getfem_gwfl::workspace::Workspace;

/// the setup of gwfl_elem.cc
fn setup(gt: &str, ktheta: usize, perturb: bool) -> (Workspace, MeshIm) {
  let m = regular_mesh(
    &geotrans_descriptor(gt).unwrap(),
    &[3, 2],
    &[0.0; 2],
    &[1.0; 2],
  )
  .unwrap();
  for i in m.point_index().into_iter().filter(|_| perturb) {
    let p = m.point(i).to_vec();
    let q = [
      p[0] + 0.05 * (2.1 * p[1] + 0.3 * p[0]).sin(),
      p[1] + 0.05 * (1.7 * p[0] + 0.2 * p[1]).cos() - 0.05,
    ];
    m.set_point(i, &q).unwrap();
  }
  let mft = MeshFem::new(&m, 2)
    .with_classical_finite_element(ktheta)
    .unwrap();
  let mfu = MeshFem::new(&m, 1)
    .with_classical_finite_element(2)
    .unwrap();
  let mim = MeshIm::new(&m).with_integration_method(4).unwrap();
  let mut ws = Workspace::new();
  ws.add_fem_variable("theta", &mft).unwrap();
  ws.add_fem_variable("u", &mfu).unwrap();
  let nt = mft.nb_dof().unwrap();
  let t: Vec<f64> = (0..nt).map(|i| 0.1 * (i as f64 + 1.0).sin()).collect();
  let nu = mfu.nb_dof().unwrap();
  let u: Vec<f64> = (0..nu)
    .map(|i| 0.05 * (2.0 * i as f64 + 1.0).cos())
    .collect();
  ws.set_variable("theta", &t).unwrap();
  ws.set_variable("u", &u).unwrap();
  ws.add_2D_rotated_RT0_projection("rt0").unwrap();
  ws.add_P0_projection("p0").unwrap();
  (ws, mim)
}

/// L2 projections on affine triangles: (P theta, P theta) = (P theta, theta).
#[test]
fn projections_on_triangles() {
  for k in [1, 2] {
    let (mut ws, mim) = setup("GT_PK(2,1)", k, true);
    for t in ["rt0", "p0"] {
      let e = format!("Elementary_transformation(theta, {t})");
      let a = {
        ws.clear_expressions();
        ws.add_expression(&format!("Norm_sqr({e})"), &mim, None)
          .unwrap();
        ws.assembly(0, false).unwrap();
        ws.assembled_potential()
      };
      let b = {
        ws.clear_expressions();
        ws.add_expression(&format!("{e}.theta"), &mim, None)
          .unwrap();
        ws.assembly(0, false).unwrap();
        ws.assembled_potential()
      };
      println!("k={k} {t}: {a} {b}");
      assert!((a - b).abs() <= 1e-12 * a.abs(), "k={k} {t}: {a} vs {b}");
    }
  }
}
