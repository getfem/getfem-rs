//! Port of GetFEM's tests/test_assembly_assignment: Norm(X) assigned to an
//! im_data at the integration points, then d*Test_u assembled; the norm of
//! the vector is GetFEM's expected value.

use getfem::fem::im_data::ImData;
use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im::MeshIm;
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::regular_mesh::regular_unit_mesh;
use getfem::gwfl::workspace::Workspace;

/// |v| on an n x n mesh
fn run(n: usize) -> f64 {
  let mesh = regular_unit_mesh(&[n, n], &geotrans_descriptor("GT_QK(2,2)").unwrap()).unwrap();
  let mf = MeshFem::new(&mesh, 1)
    .with_classical_finite_element(2)
    .unwrap();
  let mim = MeshIm::new(&mesh).with_integration_method(3).unwrap();
  let imd = ImData::new(&mim, None, &[]).unwrap();
  let mut ws = Workspace::new();
  ws.add_fem_variable("u", &mf).unwrap();
  ws.add_im_data("d", &imd, &vec![0.0; imd.nb_values().unwrap()])
    .unwrap();
  ws.set_variable("d", &ws.interpolation("Norm(X)", &imd, None).unwrap())
    .unwrap();
  let v = {
    ws.clear_expressions();
    ws.add_expression("d * Test_u", &mim, None).unwrap();
    ws.assembly(1, false).unwrap();
    ws.assembled_vector()
  };
  v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[test]
fn test_assembly_assignment() {
  // tests/test_assembly_assignment.pl
  for (n, expected) in [(10, 0.04478308527), (15, 0.02998533866)] {
    let r = run(n);
    println!("N = {n}: |v| = {r} (GetFEM expects {expected})");
    assert!((r - expected).abs() < 1e-10);
  }
}
