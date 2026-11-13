//! Assembly with threads: the same scalars, vectors and matrices as with
//! one thread (to round-off), identical from run to run.

use getfem_fem::mesh_fem::MeshFem;
use getfem_fem::mesh_im::MeshIm;
use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::regular_mesh::regular_mesh;
use getfem_gwfl::workspace::Workspace;
use getfem_linalg::threads::set_num_threads;

#[test]
fn threads_as_one_thread() {
  let m = regular_mesh(
    &geotrans_descriptor("GT_PK(2,1)").unwrap(),
    &[7, 5],
    &[0.0, 0.0],
    &[1.0, 1.0],
  )
  .unwrap();
  let mim = MeshIm::new(&m).with_integration_method(4).unwrap();
  let mf = MeshFem::new(&m, 2)
    .with_classical_finite_element(2)
    .unwrap();
  let mut ws = Workspace::new();
  ws.add_fem_variable("u", &mf).unwrap();
  let u: Vec<f64> = (0..ws.nb_dof()).map(|d| (d as f64 * 0.37).sin()).collect();
  ws.set_variable("u", &u).unwrap();
  let bottom = m.outer_faces_with_direction(&[0.0, -1.0], 0.1).unwrap();
  let e = "Norm_sqr(Grad_u)*Norm_sqr(u) + X(1)*u(2)";
  let r = "(Norm_sqr(u)*Grad_u):Grad_Test_u + X(2)*Test_u(1)";
  let k = "(Norm_sqr(u)*Grad_Test2_u):Grad_Test_u + 2*(u.Test2_u)*(Grad_u:Grad_Test_u)";
  let mut run = |nt: usize| {
    set_num_threads(nt).unwrap();
    let s = {
      ws.clear_expressions();
      ws.add_expression(e, &mim, None).unwrap();
      ws.assembly(0, false).unwrap();
      ws.assembled_potential()
    } + {
      ws.clear_expressions();
      ws.add_expression("u(1)", &mim, &bottom).unwrap();
      ws.assembly(0, false).unwrap();
      ws.assembled_potential()
    };
    let v = {
      ws.clear_expressions();
      ws.add_expression(r, &mim, None).unwrap();
      ws.assembly(1, false).unwrap();
      ws.assembled_vector()
    }
    .to_vec();
    let vb = {
      ws.clear_expressions();
      ws.add_expression("u.Test_u", &mim, &bottom).unwrap();
      ws.assembly(1, false).unwrap();
      ws.assembled_vector()
    }
    .to_vec();
    let a = {
      ws.clear_expressions();
      ws.add_expression(k, &mim, None).unwrap();
      ws.assembly(2, false).unwrap();
      ws.assembled_matrix()
    }
    .to_csr()
    .to_dense();
    (s, v, vb, a)
  };
  let one = run(1);
  for nt in [2, 3, 4, 8, 100] {
    let (s, v, vb, a) = run(nt);
    assert!((s - one.0).abs() < 1e-13 * one.0.abs(), "{nt}");
    for (x, y) in [(&v, &one.1), (&vb, &one.2)] {
      let n = y.iter().fold(0.0f64, |a, b| a.max(b.abs()));
      assert!(
        x.iter().zip(y).all(|(p, q)| (p - q).abs() < 1e-13 * n),
        "{nt}"
      );
    }
    let (a1, b1) = (a.concat(), one.3.concat());
    let n = b1.iter().fold(0.0f64, |a, b| a.max(b.abs()));
    assert!(
      a1.iter().zip(&b1).all(|(p, q)| (p - q).abs() < 1e-13 * n),
      "{nt}"
    );
    // identical from run to run
    let again = run(nt);
    assert!(s == again.0 && v == again.1 && vb == again.2, "{nt}");
    assert!(a == again.3, "{nt}");
  }
  set_num_threads(1).unwrap();
}
