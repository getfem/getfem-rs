//! M10: mesh slices on a level set.

use getfem_fem::slice::{SlicerNone, StoredMeshSlice};
use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::mesh::Mesh;
use getfem_geom::region::outer_faces_of_mesh;
use getfem_geom::regular_mesh::regular_mesh;

fn mesh(gt: &str, n: &[usize]) -> Mesh {
  let d = n.len();
  regular_mesh(
    &geotrans_descriptor(gt).unwrap(),
    n,
    &vec![0.0; d],
    &vec![1.0; d],
  )
  .unwrap()
}

/// sum of the measures of the d-simplexes of a slice (d 1 or 2)
fn measure(sl: &StoredMeshSlice, d: usize) -> f64 {
  let mut s = 0.0;
  for c in &sl.cvlst {
    for x in c.simplexes.iter().filter(|x| x.dim() == d) {
      let p: Vec<&[f64]> = x.inodes.iter().map(|&i| c.nodes[i].pt.as_slice()).collect();
      let e = |k: usize| [p[k][0] - p[0][0], p[k][1] - p[0][1]];
      s += match d {
        1 => e(1)[0].hypot(e(1)[1]),
        _ => 0.5 * (e(1)[0] * e(2)[1] - e(1)[1] * e(2)[0]).abs(),
      };
    }
  }
  s
}

#[test]
fn slice_on_level_set() {
  use getfem_fem::level_set::LevelSet;
  use getfem_fem::mesh_level_set::MeshLevelSet;
  let m = mesh("GT_PK(2,1)", &[4, 4]);
  let f = |p: &[f64]| p[1] - 0.43 - 0.2 * p[0];
  let ls = LevelSet::new(&m, 1, false).unwrap();
  let mf = ls.mesh_fem();
  let v: Vec<f64> = (0..mf.nb_dof().unwrap())
    .map(|i| f(&mf.point_of_basic_dof(i).unwrap()))
    .collect();
  ls.set_values(0, &v).unwrap();
  let mut mls = MeshLevelSet::new(&m);
  mls.add_level_set(&ls);
  mls.adapt().unwrap();
  let md = m.snapshot();
  let sl = StoredMeshSlice::build_on_level_set(&mls, &mut [&mut SlicerNone], 2, None).unwrap();
  assert!((measure(&sl, 2) - 1.0).abs() < 1e-12);
  let mut ncut = 0;
  for c in &sl.cvlst {
    if mls.is_convex_cut(c.cv_num) {
      ncut += 1;
      // nodes of their own, no simplex across the level set
      assert_eq!(c.nodes.len(), 3 * c.simplexes.len());
      for x in &c.simplexes {
        let s: Vec<f64> = x.inodes.iter().map(|&i| f(&c.nodes[i].pt)).collect();
        assert!(
          s.iter().all(|v| *v > -1e-12) || s.iter().all(|v| *v < 1e-12),
          "{s:?}"
        );
      }
    }
  }
  assert!(ncut > 0);
  // the boundary faces: the outer boundary, length 4
  let b = outer_faces_of_mesh(&m).resolve(&md).unwrap();
  let sb = StoredMeshSlice::build_on_level_set(&mls, &mut [], 2, Some(&b)).unwrap();
  assert!((measure(&sb, 1) - 4.0).abs() < 1e-12);
}
