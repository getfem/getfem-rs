//! Laplace equation on a plate with three holes (gmsh, curved P2 triangles):
//! u = 0 on the outer boundary, u = 1 on the holes. Writes
//! target/showcase/plate.vtu.

use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im::MeshIm;
use getfem::gwfl::workspace::Workspace;
use getfem::io::gmsh::{GmshOptions, import_from_gmsh};
use getfem::io::vtk::{Format, VtuExport};
use getfem::linalg::tensor::Mat;
use getfem::linalg::{LinearSolver, MatrixKind};
use getfem::mumps::Mumps;

fn main() {
  let dir = env!("CARGO_MANIFEST_DIR");
  let out = format!("{dir}/../../target/showcase");
  std::fs::create_dir_all(&out).unwrap();
  let g = import_from_gmsh(
    &format!("{dir}/../../data/examples/plate.msh"),
    &GmshOptions::default(),
  )
  .unwrap();
  let mesh = g.mesh;
  // the file's 3D mesh in the plane z = 0, made 2D
  mesh
    .modify(|m| m.transform(&Mat::from_rows(&[&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]])))
    .unwrap();
  let mf = MeshFem::new(&mesh, 1)
    .with_classical_finite_element(2)
    .unwrap();
  let mim = MeshIm::new(&mesh).with_integration_method(6).unwrap();
  // u = ud + v: ud the prescribed values (0 outside, 1 on the holes), v
  // the unknown on the other DOFs; regions: the physical groups "outer"
  // 10, "holes" 20
  let dofs_on = |r: usize| mf.basic_dof_on_region(mesh.region(r).unwrap()).unwrap();
  let (outer, holes) = (dofs_on(10), dofs_on(20));
  let n = mf.nb_dof().unwrap();
  let ud: Vec<f64> = (0..n)
    .map(|d| if holes.contains(&d) { 1.0 } else { 0.0 })
    .collect();
  let kept = (0..n)
    .filter(|d| !outer.contains(d) && !holes.contains(d))
    .collect();
  let mfv = MeshFem::partial(&mf, &kept).unwrap();
  let mut ws = Workspace::new();
  ws.add_fem_data("ud", &mf, &ud).unwrap();
  ws.add_fem_variable("v", &mfv).unwrap();
  let k = {
    ws.clear_expressions();
    ws.add_expression("Grad_Test_v.Grad_Test2_v", &mim, None)
      .unwrap();
    ws.assembly(2, false).unwrap();
    ws.assembled_matrix()
  }
  .to_csr();
  let mut b = {
    ws.clear_expressions();
    ws.add_expression("-Grad_ud.Grad_Test_v", &mim, None)
      .unwrap();
    ws.assembly(1, false).unwrap();
    ws.assembled_vector()
  }
  .to_vec();
  let mut s = Mumps::new();
  s.solve_system(&k, MatrixKind::SymmetricPositiveDefinite, &mut b)
    .unwrap();
  let v = mfv.extend_vector(&b).unwrap();
  let u: Vec<f64> = ud.iter().zip(&v).map(|(a, b)| a + b).collect();
  let (lo, hi) = u
    .iter()
    .fold((f64::MAX, f64::MIN), |(l, h), &v| (l.min(v), h.max(v)));
  println!(
    "{} elements, {} DOFs, u in [{lo:.4}, {hi:.4}], MUMPS factors {} entries",
    mesh.nb_convexes(),
    mf.nb_basic_dof().unwrap(),
    s.info().nnz_factors.unwrap()
  );
  let mut vtu = VtuExport::new(&format!("{out}/plate.vtu"), Format::Ascii);
  vtu.exporting(&mf).unwrap();
  vtu.write_point_data(&mf, &u, "u").unwrap();
  vtu.write().unwrap();
}
