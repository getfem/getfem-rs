use super::*;
use crate::geotrans::geotrans_descriptor;
use crate::region::{self, RegionSet, outer_faces_of_mesh};
use crate::regular_mesh::regular_mesh;

#[test]
fn point_merge_is_relative() {
  for sc in [1e-3, 1.0, 1e3] {
    let mut m = MeshData::new(2);
    let a = m.add_point(&[0.0, 0.0]).unwrap();
    let b = m.add_point(&[sc, 0.0]).unwrap();
    assert_eq!(m.add_point(&[sc * (1.0 + 1e-14), 1e-14 * sc]).unwrap(), b);
    let c = m.add_point(&[sc * (1.0 + 1e-9), 0.0]).unwrap();
    assert!(c != a && c != b);
    assert_eq!(m.nb_points(), 3);
  }
}

#[test]
fn sup_and_optimize() {
  let gt = geotrans_descriptor("GT_PK(2,1)").unwrap();
  let m = crate::regular_mesh::regular_unit_mesh(&[2, 2], &gt).unwrap();
  let n0 = m.nb_convexes();
  let mut r = RegionSet::new();
  r.add(n0 - 1);
  r.add_face(0, 1);
  m.set_region_set(3, r).unwrap();
  let v = m.version();
  m.sup_convex(0, true);
  assert!(m.version() > v);
  assert_eq!(m.nb_convexes(), n0 - 1);
  assert!(!m.region_set(3).unwrap().is_in_face(0, 1));
  m.optimize_structure();
  assert_eq!(m.nb_convexes_max(), n0 - 1);
  assert_eq!(m.nb_points_max(), m.nb_points());
  // last convex moved to 0, region followed
  assert!(m.region_set(3).unwrap().is_in(0));
  for cv in m.convex_index() {
    for p in m.ind_points_of_convex(cv) {
      assert!(m.convexes_of_point(p).contains(&cv));
    }
  }
  let bad = {
    let mut r = RegionSet::new();
    r.add(99);
    r
  };
  assert!(m.set_region_set(4, bad).is_err());
  assert!(m.add_convex(&gt, &[0, 1]).is_err());
  assert!(m.add_point(&[0.0]).is_err());
}

#[test]
fn region_set_ops() {
  let mut a = RegionSet::new();
  a.add(1);
  a.add_face(2, 0);
  a.add_face(2, 1);
  let mut b = RegionSet::new();
  b.add_face(1, 2);
  b.add_face(2, 1);
  b.add(3);
  // whole convex 1 intersected with its face 2 keeps the face
  let i = a.intersect(&b);
  assert_eq!(
    i.iter().collect::<Vec<_>>(),
    vec![(1, Some(2)), (2, Some(1))]
  );
  assert_eq!(a.merge(&b).size(), 5);
  assert_eq!(
    a.subtract(&b).iter().collect::<Vec<_>>(),
    vec![(1, None), (2, Some(0))]
  );
  assert!(a.merge(&b).is_in(3) && !a.is_only_faces() && b.subtract(&a).is_in(3));
}

#[test]
fn faces_in_box_tolerant() {
  // 0.1*3 != 0.3 in floating point: box selection must still get x = 0.3 faces
  let gt = geotrans_descriptor("GT_QK(2,1)").unwrap();
  let m = regular_mesh(&gt, &[3, 1], &[0.0, 0.0], &[0.3, 1.0]).unwrap();
  let of = outer_faces_of_mesh(&m);
  let right = region::select_faces_in_box(&m, &of, &[0.3, 0.0], &[0.3, 1.0]).unwrap();
  assert_eq!(right.resolve(&m.snapshot()).unwrap().size(), 1);
}

#[test]
fn outer_faces_with_direction_and_in_box() {
  let gt = geotrans_descriptor("GT_QK(2,1)").unwrap();
  let m = regular_mesh(&gt, &[3, 2], &[0.0, 0.0], &[0.3, 1.0]).unwrap();
  let size = |r: Region| r.resolve(&m.snapshot()).unwrap().size();
  assert_eq!(
    size(m.outer_faces_with_direction(&[0.0, 1.0], 0.0).unwrap()),
    3
  );
  assert_eq!(
    size(m.outer_faces_with_direction(&[1.0, 1.0], 1.0).unwrap()),
    5
  );
  assert_eq!(
    size(m.outer_faces_in_box(&[0.3, 0.0], &[0.3, 1.0]).unwrap()),
    2
  );
}

#[test]
fn extrude_layers() {
  let gt = geotrans_descriptor("GT_QK(2,1)").unwrap();
  let sq = regular_mesh(&gt, &[1, 1], &[0.0, 0.0], &[1.0, 1.0]).unwrap();
  let m = extrude(&sq, 2, 1).unwrap();
  assert_eq!((m.dim(), m.nb_points(), m.nb_convexes()), (3, 12, 2));
  assert_eq!(m.trans_of_convex(1).name(), "GT_QK(3,1)");
  // nodes of the upper layer: the square at z = 0.5, then at z = 1
  let g = m.points_of_convex(1);
  let z: Vec<f64> = (0..8).map(|k| g[(2, k)]).collect();
  assert_eq!(z, [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]);
  for k in 0..4 {
    assert_eq!((g[(0, k)], g[(1, k)]), (g[(0, k + 4)], g[(1, k + 4)]));
  }
  // degree 2: the middle nodes of each layer
  let gt = geotrans_descriptor("GT_QK(2,2)").unwrap();
  let sq = regular_mesh(&gt, &[1, 1], &[0.0, 0.0], &[1.0, 1.0]).unwrap();
  let m = extrude(&sq, 2, 2).unwrap();
  assert_eq!((m.nb_points(), m.nb_convexes()), (45, 2));
  assert_eq!(m.trans_of_convex(0).name(), "GT_QK(3,2)");
  assert_eq!(m.ind_points_of_convex(0).len(), 27);
  assert!(extrude(&sq, 0, 2).is_err());
}
