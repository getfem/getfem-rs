//! Meshes cut by level sets: two level sets crossing in a face.

use getfem_fem::integration::classical_approx_im;
use getfem_fem::level_set::LevelSet;
use getfem_fem::mesh_level_set::{MeshLevelSet, jacobian};
use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::mesh::{Mesh, MeshData};
use getfem_geom::regular_mesh::regular_mesh;

type F = fn(&[f64]) -> f64;

fn level_set(m: &Mesh, k: usize, f0: F, f1: Option<F>) -> LevelSet {
  let ls = LevelSet::new(m, k, f1.is_some()).unwrap();
  let mf = ls.mesh_fem();
  let pts: Vec<Vec<f64>> = (0..mf.nb_dof().unwrap())
    .map(|i| mf.point_of_basic_dof(i).unwrap())
    .collect();
  ls.set_values(0, &pts.iter().map(|p| f0(p)).collect::<Vec<_>>())
    .unwrap();
  if let Some(f1) = f1 {
    ls.set_values(1, &pts.iter().map(|p| f1(p)).collect::<Vec<_>>())
      .unwrap();
  }
  ls
}

/// measure of a sub-element (rule of degree 6) and its centroid
fn sub_measure(sm: &MeshData, i: usize) -> (f64, Vec<f64>) {
  let gt = sm.trans_of_convex(i);
  let gref = sm.points_of_convex(i);
  let a = classical_approx_im(gt, 6).unwrap();
  let a = a.approx().unwrap();
  let s = (0..a.nb_points_on_convex())
    .map(|k| a.coeff(k) * jacobian(gt, &gref, a.point(k)).abs())
    .sum();
  let n = sm.dim();
  let cref = gt.transform(&vec![1.0 / (n + 1) as f64; n], &gref).unwrap();
  (s, cref)
}

/// Two curved level sets crossing in a face of the element (the crossing
/// line y = 0.15 z on x = 0): crossing points on each, closer and closer
/// to the line, followed each other until the sub-cells crossed or
/// touched by both got the point on both. The cut fills the element.
#[test]
fn level_sets_crossing_in_a_face() {
  let m = regular_mesh(
    &geotrans_descriptor("GT_QK(3,1)").unwrap(),
    &[1, 1, 1],
    &[-4.0, -5.0 / 3.0, -6.0],
    &[4.0, 10.0 / 3.0, 4.0],
  )
  .unwrap();
  let a = level_set(
    &m,
    2,
    |x| x[1] - 2.5 * (x[0] / 7.0).tanh() - 0.15 * x[2],
    None,
  );
  let b = level_set(
    &m,
    2,
    |x| x[1] + 2.5 * (x[0] / 7.0).tanh() - 0.15 * x[2],
    None,
  );
  let mut mls = MeshLevelSet::new(&m);
  mls.add_level_set(&a);
  mls.add_level_set(&b);
  mls.adapt().unwrap();
  assert!(mls.is_convex_cut(0));
  let sm = mls.mesh_of_convex(0).unwrap();
  let v: f64 = sm.convex_index().map(|i| sub_measure(sm, i).0).sum();
  assert!(
    (v - 1.0).abs() < 1e-6,
    "sub-cells measure {v} of the reference cube"
  );
}
