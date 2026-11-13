//! The same assembled scalar and vector on any number of ranks: before
//! getfem::mpi::init (each process all elements) and after (each its part,
//! summed); interpolation on Lagrange nodes, ImData points and mesh points
//! and the local projection, the same; a system of each rank's matrix part
//! solved by MUMPS; ci/mpi.sh runs it on 1, 2 and 4 ranks. Process-wide MPI: a test
//! binary of its own, one test.

use getfem::fem::im_data::ImData;
use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im::MeshIm;
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::mesh::Mesh;
use getfem::geom::region::outer_faces_of_mesh;
use getfem::geom::regular_mesh::regular_unit_mesh;
use getfem::gwfl::workspace::Workspace;
use getfem::linalg::{LinearSolver, MatrixKind};
use getfem::mumps::Mumps;

/// the potential and the residual vector of a nonlinear expression, with a
/// boundary term
fn assemble(m: &Mesh) -> (f64, Vec<f64>) {
  let mf = MeshFem::new(m, 3).with_classical_finite_element(2).unwrap();
  let mim = MeshIm::new(m).with_integration_method(4).unwrap();
  let u: Vec<f64> = (0..mf.nb_dof().unwrap())
    .map(|i| (0.3 * i as f64).sin())
    .collect();
  let mut ws = Workspace::new();
  ws.add_fem_variable("u", &mf).unwrap();
  ws.set_variable("u", &u).unwrap();
  ws.add_expression("Norm_sqr(Grad_u) + 1 + u.u", &mim, -1)
    .unwrap();
  ws.add_expression("u.u", &mim, 1).unwrap();
  ws.assembly(0, false).unwrap();
  let s = ws.assembled_potential();
  let mut ws = Workspace::new();
  ws.add_fem_variable("u", &mf).unwrap();
  ws.set_variable("u", &u).unwrap();
  ws.add_expression(
    "(1 + Norm_sqr(u)) * Grad_u:Grad_Test_u + u.Test_u",
    &mim,
    -1,
  )
  .unwrap();
  ws.add_expression("u.Test_u", &mim, 1).unwrap();
  ws.assembly(1, false).unwrap();
  (s, ws.assembled_vector().to_vec())
}

/// interpolations of a nonlinear expression: on Lagrange nodes (averaged
/// over the elements sharing them), ImData points, mesh points, and the
/// local projection on a discontinuous space, concatenated
fn interpolate(m: &Mesh) -> Vec<f64> {
  let mf = MeshFem::new(m, 3).with_classical_finite_element(2).unwrap();
  let mim = MeshIm::new(m).with_integration_method(4).unwrap();
  let u: Vec<f64> = (0..mf.nb_dof().unwrap())
    .map(|i| (0.3 * i as f64).sin())
    .collect();
  let mut ws = Workspace::new();
  ws.add_fem_variable("u", &mf).unwrap();
  ws.set_variable("u", &u).unwrap();
  let e = "Norm_sqr(u)*Grad_u";
  let mf1 = MeshFem::new(m, 9).with_classical_finite_element(1).unwrap();
  let mut out = ws.interpolation(e, &mf1, -1).unwrap();
  let imd = ImData::new(&mim, -1, &[3, 3]).unwrap();
  out.extend(ws.interpolation(e, &imd, -1).unwrap());
  let pts: Vec<Vec<f64>> = (0..7)
    .map(|k| vec![0.13 * k as f64 + 0.05, 0.4, 0.97 - 0.12 * k as f64])
    .collect();
  out.extend(ws.interpolation(e, (&pts, m), -1).unwrap());
  let dc = MeshFem::new(m, 3)
    .with_finite_element(&getfem::fem::fem::fem_descriptor("FEM_PK_DISCONTINUOUS(3,1)").unwrap())
    .unwrap();
  out.extend(ws.local_projection("Norm_sqr(u)*u", &mim, &dc, -1).unwrap());
  out
}

#[test]
fn same_assembly_on_any_ranks() {
  let m = regular_unit_mesh(&[4, 4, 4], &geotrans_descriptor("GT_PK(3,1)").unwrap()).unwrap();
  m.set_region(1, &outer_faces_of_mesh(&m)).unwrap();
  let (s0, v0) = assemble(&m);
  let i0 = interpolate(&m);
  let mpi = getfem::mpi::init().unwrap();
  // the parts cover the mesh once
  let rg = m.snapshot().mpi_region();
  let mut nb = vec![0.0; getfem::mpi::size() + m.nb_convexes()];
  nb[getfem::mpi::rank()] = rg.size() as f64;
  rg.convexes()
    .for_each(|cv| nb[getfem::mpi::size() + cv] += 1.0);
  getfem::mpi::sum(&mut nb);
  let (sizes, cover) = nb.split_at(getfem::mpi::size());
  assert!(
    cover.iter().all(|&c| c == 1.0),
    "{sizes:?} {:?}",
    cover
      .iter()
      .enumerate()
      .filter(|(_, c)| **c != 1.0)
      .collect::<Vec<_>>()
  );
  // single-threaded ranks
  let one = getfem::mpi::size() > 1;
  assert_eq!(getfem::set_num_threads(4).is_err(), one);
  assert_eq!(getfem::mumps::set_num_threads(4).is_err(), one);
  assert_eq!(getfem::num_threads() == 1, one);
  let (s, v) = assemble(&m);
  let i = interpolate(&m);
  let ni = i0.iter().fold(0.0f64, |a, x| a.max(x.abs()));
  assert_eq!(i.len(), i0.len());
  let di = i
    .iter()
    .zip(&i0)
    .fold(0.0f64, |a, (x, y)| a.max((x - y).abs()));
  assert!(di <= 1e-13 * ni, "interpolation {di} {ni}");
  assert!((s - s0).abs() <= 1e-13 * s0.abs(), "{s} {s0}");
  let nv = v0.iter().fold(0.0f64, |a, x| a.max(x.abs()));
  let d = v
    .iter()
    .zip(&v0)
    .fold(0.0f64, |a, (x, y)| a.max((x - y).abs()));
  assert!(d <= 1e-13 * nv, "{d} {nv}");
  // K x = b, K of each rank's elements: the residual summed over the ranks
  let mf = MeshFem::new(&m, 3)
    .with_classical_finite_element(2)
    .unwrap();
  let mim = MeshIm::new(&m).with_integration_method(4).unwrap();
  let mut ws = Workspace::new();
  ws.add_fem_variable("u", &mf).unwrap();
  ws.add_expression("Grad_u:Grad_Test_u + u.Test_u + [1,2,3].Test_u", &mim, -1)
    .unwrap();
  ws.assembly(2, false).unwrap();
  ws.assembly(1, false).unwrap();
  let k = ws.assembled_matrix().clone();
  let b: Vec<f64> = ws.assembled_vector().iter().map(|x| -x).collect();
  let mut x = b.clone();
  Mumps::new()
    .solve_system(&k, MatrixKind::SymmetricPositiveDefinite, &mut x)
    .unwrap();
  let mut kx = k.mult(&x);
  getfem::mpi::sum(&mut kx);
  let nb = b.iter().fold(0.0f64, |a, x| a.max(x.abs()));
  let d = kx
    .iter()
    .zip(&b)
    .fold(0.0f64, |a, (x, y)| a.max((x - y).abs()));
  assert!(d <= 1e-12 * nb, "{d} {nb}");
  drop(mpi);
}
