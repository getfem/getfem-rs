//! Interpolation: a discontinuous vector source averaged, a vector element
//! with qdim.

use getfem_fem::fem::fem_descriptor;
use getfem_fem::interpolation::{interpolation, interpolation_at_points};
use getfem_fem::mesh_fem::MeshFem;
use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::regular_mesh::regular_mesh;

fn values(n: usize) -> Vec<f64> {
  (0..n)
    .map(|i| 0.1 * (i as f64 + 1.0).sin() + 0.3 * (0.7 * i as f64).cos())
    .collect()
}

fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
  a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
}

/// a linear vector field (exact in P1) from a discontinuous P1 space of
/// qdim 2 onto a continuous one: the averages of every component
#[test]
fn discontinuous_vector_source_averaged() {
  let m = regular_mesh(
    &geotrans_descriptor("GT_PK(2,1)").unwrap(),
    &[2, 2],
    &[0.0, 0.0],
    &[1.0, 1.0],
  )
  .unwrap();
  let src = MeshFem::new(&m, 2)
    .with_classical_discontinuous_finite_element(1)
    .unwrap();
  let tgt = MeshFem::new(&m, 2)
    .with_classical_finite_element(1)
    .unwrap();
  let f = |x: &[f64]| [x[0] + 2.0 * x[1], 3.0 * x[0] - x[1]];
  let field = |mf: &MeshFem| -> Vec<f64> {
    (0..mf.nb_basic_dof().unwrap() / 2)
      .flat_map(|i| f(&mf.point_of_basic_dof(2 * i).unwrap()))
      .collect()
  };
  let got = interpolation(&src, &tgt, &field(&src)).unwrap();
  assert!(close(&got, &field(&tgt), 1e-12), "{got:?}");
}

/// a vector element on a space of qdim Q target_dim (Q = 2): each copy
/// interpolated as on its own space
#[test]
fn vector_element_with_qmult() {
  let m = regular_mesh(
    &geotrans_descriptor("GT_PK(2,1)").unwrap(),
    &[2, 2],
    &[0.0, 0.0],
    &[1.0, 1.0],
  )
  .unwrap();
  let rt = fem_descriptor("FEM_RT0(2)").unwrap();
  let s4 = MeshFem::new(&m, 4).with_finite_element(&rt).unwrap();
  let s2 = MeshFem::new(&m, 2).with_finite_element(&rt).unwrap();
  let n = s2.nb_dof().unwrap();
  assert_eq!(s4.nb_dof().unwrap(), 2 * n);
  let u4 = values(2 * n);
  let copy = |qq: usize| -> Vec<f64> { (0..n).map(|i| u4[2 * i + qq]).collect() };
  let pts = vec![vec![0.3, 0.2], vec![0.7, 0.55], vec![0.15, 0.8]];
  let r4 = interpolation_at_points(&s4, &pts, &u4).unwrap();
  let (r0, r1) = (
    interpolation_at_points(&s2, &pts, &copy(0)).unwrap(),
    interpolation_at_points(&s2, &pts, &copy(1)).unwrap(),
  );
  for p in 0..pts.len() {
    let want = [r0[2 * p], r0[2 * p + 1], r1[2 * p], r1[2 * p + 1]];
    assert!(
      close(&r4[4 * p..4 * p + 4], &want, 1e-12),
      "point {p}: {:?} vs {want:?}",
      &r4[4 * p..4 * p + 4]
    );
  }
}
