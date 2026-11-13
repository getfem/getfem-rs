//! Port of GetFEM tests/test_mesh.cc (parts in M1 scope: no file I/O,
//! refinement or mesh_fem).

use getfem_geom::convex_ref::{parallelepiped_of_reference, simplex_of_reference_k};
use getfem_geom::geotrans::{geotrans_descriptor, pk};
use getfem_geom::mesh::MeshData;
use getfem_geom::region::{RegionSet, outer_faces_of_mesh};
use getfem_geom::regular_mesh::parallelepiped_regular_simplex_mesh;

#[test]
fn test_mesh() {
  let mut m = MeshData::new(3);
  let i1 = m.add_point(&[0.0, 0.0, 0.0]).unwrap();
  let i3 = m.add_point(&[1.0, 0.0, 0.0]).unwrap();
  let i2 = m.add_point(&[1e-13, 0.0, 0.0]).unwrap();
  assert_eq!(
    m.nb_points(),
    2,
    "1e-13 merges with the origin (relative to the box)"
  );
  assert_eq!(i1, i2);
  m.sup_point(i1).unwrap();
  assert_eq!(m.nb_points(), 1);
  let i2 = m.add_point(&[1e-13, 0.0, 0.0]).unwrap();
  m.add_convex(&pk(1, 1).unwrap(), &[i2, i3]).unwrap();
  let pts: [&[f64]; 3] = [&[2.0, 0.0, 0.0], &[0.0, 2.0, 0.0], &[0.0, 0.0, 2.0]];
  let t = pk(2, 1).unwrap();
  let ic2 = m.add_convex_by_points(&t, &pts).unwrap();
  let ic3 = m.add_convex_by_points(&t, &pts).unwrap();
  // GetFEM returns the existing convex; we allow duplicates but share points
  assert_eq!(m.ind_points_of_convex(ic2), m.ind_points_of_convex(ic3));
  assert_eq!(m.point(m.ind_points_of_convex(ic2)[1]), &[0.0, 2.0, 0.0]);

  let ivect: Vec<Vec<f64>> = (0..3)
    .map(|i| (0..3).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
    .collect();
  parallelepiped_regular_simplex_mesh(&mut m, 3, &[0.0; 3], &ivect, &[3, 3, 3]).unwrap();

  let mut r3 = RegionSet::new();
  r3.add(0);
  r3.add_face(3, 1);
  r3.add(2);
  m.set_region_set(3, r3).unwrap();
  let mut r4 = RegionSet::new();
  r4.add(0);
  r4.add(5);
  m.set_region_set(4, r4).unwrap();
  assert!(m.region_set(3).unwrap().is_in(0));
  m.sup_convex(0, false);
  m.sup_convex(1, false);
  assert!(!m.region_set(3).unwrap().is_in(0));
  let n = m.nb_convexes();
  m.optimize_structure();
  assert_eq!(m.nb_convexes(), n);
  // convexes 0, 1 removed: last two moved to 0, 1; regions follow
  let r4 = m.region_set(4).unwrap();
  assert_eq!(r4.size(), 1);
  assert!(r4.is_in(5));
  assert!(!r4.is_in_face(0, 0));
  let r3 = m.region_set(3).unwrap();
  assert_eq!(r3.size(), 2);
  assert!(r3.is_in_face(3, 1));
  assert!(r3.is_in(2));
  assert!(!r3.is_in(0));
}

#[test]
fn test_region() {
  let mut a = RegionSet::new();
  for cv in [4, 9, 2, 5] {
    a.add(cv);
  }
  a.add_face(3, 7);
  a.add_face(3, 3);
  let mut b = RegionSet::new();
  b.add(2);
  b.add_face(3, 2);
  b.add_face(3, 7);
  b.add_face(9, 1);
  b.add_face(9, 5);
  b.add(8);
  let r = a.intersect(&b);
  assert_eq!(
    r.iter().collect::<Vec<_>>(),
    vec![(2, None), (3, Some(7)), (9, Some(1)), (9, Some(5))]
  );
}

#[test]
fn test_convex_ref() {
  for k in 1..=2 {
    let c = simplex_of_reference_k(1, k);
    assert!((c.is_in(&[0.5]).unwrap() + 0.5).abs() < 1e-6);
    assert!((c.is_in(&[-0.1]).unwrap() - 0.1).abs() < 1e-6);
    assert!((c.is_in(&[1.1]).unwrap() - 0.1).abs() < 1e-6);
    let c = simplex_of_reference_k(2, k);
    assert!(c.is_in(&[0.0, 0.0]).unwrap().abs() < 1e-6);
    assert!(c.is_in(&[0.5, 0.5]).unwrap().abs() < 1e-6);
    assert!(c.is_in(&[0.25, 0.25]).unwrap() < -0.2);
    assert!(c.is_in(&[0.85, 0.85]).unwrap() > 0.2);
    assert!(c.is_in(&[-0.25, 0.05]).unwrap() > 0.2);
    assert!(c.is_in(&[0.05, -0.25]).unwrap() > 0.2);
    let c = parallelepiped_of_reference(3, 1);
    assert!((c.is_in(&[0.5, 0.5, 0.5]).unwrap() + 0.5).abs() < 1e-6);
    assert!((c.is_in(&[-0.5, -0.5, -0.5]).unwrap() - 0.5).abs() < 1e-6);
  }
}

#[test]
fn test_search_point() {
  let pts: &[(usize, [f64; 3])] = &[
    (1, [-4., 6., 2.]),
    (2, [0., 6., 0.]),
    (3, [0., 2., 0.]),
    (4, [-2., 6., 2.]),
    (5, [0., 4., 0.]),
    (6, [-1.5, 4.5, 0.5]),
    (7, [1., 2., 0.]),
    (8, [1.5, 1.5, 0.]),
    (9, [5., 5., 0.]),
    (10, [2., 1., 0.]),
    (11, [6., 3., 0.]),
    (12, [2., 0., 0.]),
    (13, [6., 0., 0.]),
    (14, [2., 4., 0.]),
    (15, [4., 2., 0.]),
    (17, [3., 6., 0.]),
    (18, [2., -2., 2.]),
    (19, [2., -2., -2.]),
    (20, [6., -2., 2.]),
    (21, [6., -2., -2.]),
    (22, [2., -1., 1.]),
    (23, [2., -2.5, 0.]),
    (24, [2., -1., -1.]),
    (25, [6., -1., 1.]),
    (26, [6., -2.5, 0.]),
    (27, [6., -1., -1.]),
    (28, [-1., 6., -1.]),
    (29, [-1., 2., -1.]),
    (30, [1., 6., -2.]),
    (31, [1., 2., -2.]),
    (32, [0., 6., -3.]),
    (33, [0., 2., -3.]),
    (34, [2., -5., -2.]),
    (35, [2., -4., 0.]),
    (36, [4., -5., 2.]),
    (37, [6., -5., -2.]),
    (38, [6., -5., 0.]),
    (46, [4., 4., 0.]),
    (49, [6., -5., 2.]),
  ];
  let cvs: &[(&str, &[usize])] = &[
    ("GT_PK(2,2)", &[1, 4, 2, 6, 5, 3]),
    ("GT_QK(2,1)", &[2, 17, 3, 7]),
    ("GT_QK(2,2)", &[7, 8, 10, 14, 46, 15, 17, 9, 11]),
    ("GT_QK(2,1)", &[10, 12, 11, 13]),
    (
      "GT_PRODUCT(GT_PK(2,2),GT_PK(1,1))",
      &[12, 22, 18, 24, 23, 19, 13, 25, 20, 27, 26, 21],
    ),
    (
      "GT_PRODUCT(GT_PK(1,1),GT_PK(1,3))",
      &[2, 3, 28, 29, 30, 31, 32, 33],
    ),
    (
      "GT_PRODUCT(GT_PK(1,2),GT_PRODUCT(GT_PK(1,1),GT_PK(1,1)))",
      &[19, 23, 18, 21, 26, 20, 34, 35, 36, 37, 38, 49],
    ),
  ];
  let mut m = MeshData::new(3);
  let mut id = std::collections::HashMap::new();
  for (k, p) in pts {
    id.insert(*k, m.add_point(p).unwrap());
  }
  for (g, ip) in cvs {
    let ip: Vec<usize> = ip.iter().map(|k| id[k]).collect();
    m.add_convex(&geotrans_descriptor(g).unwrap(), &ip).unwrap();
  }
  assert_eq!(m.nb_points(), pts.len());
  for i in m.point_index() {
    assert_eq!(m.search_point(m.point(i), 0.0), Some(i));
  }
  assert_eq!(m.search_point(&[0.123, 0.456, 0.789], 0.0), None);
}

#[test]
fn test_mesh_matching() {
  // conformity of the simplexified regular mesh, scales 1e-3, 1, 1e3
  for dim in 1..=4 {
    for sc in [1e-3, 1.0, 1e3] {
      let ns = 2;
      let ivect: Vec<Vec<f64>> = (0..dim)
        .map(|i| {
          (0..dim)
            .map(|j| if i == j { sc / ns as f64 } else { 0.0 })
            .collect()
        })
        .collect();
      let mut m = MeshData::new(dim);
      parallelepiped_regular_simplex_mesh(&mut m, dim, &vec![0.0; dim], &ivect, &vec![ns; dim])
        .unwrap();
      let of = outer_faces_of_mesh(&m);
      let mut nb = 0;
      for (cv, f) in of.resolve(&m).unwrap().iter() {
        nb += 1;
        for p in m.ind_points_of_face_of_convex(cv, f.unwrap()) {
          let x = m.point(p);
          assert!(
            x.iter()
              .any(|&c| c.abs() < 1e-10 * sc || (c - sc).abs() < 1e-10 * sc),
            "dim {dim} scale {sc}: not conforming at {x:?}"
          );
        }
      }
      // boundary face counts printed by GetFEM's test_mesh
      assert_eq!(nb, [2, 8, 48, 320][dim - 1], "dim {dim}");
    }
  }
}

/// intersection with the entire domain is the other region (GetFEM
/// mesh_region::intersection); union and difference with it are errors
#[test]
fn intersection_with_the_entire_domain() {
  use getfem_geom::regular_mesh::regular_unit_mesh;
  let m = regular_unit_mesh(&[2, 2], &geotrans_descriptor("GT_PK(2,1)").unwrap()).unwrap();
  m.set_region(1, &outer_faces_of_mesh(&m)).unwrap();
  let (all, r1) = (m.region(-1).unwrap(), m.region(1).unwrap());
  let d = m.snapshot();
  let want: Vec<_> = r1.resolve(&d).unwrap().iter().collect();
  for r in [all.intersect(&r1).unwrap(), r1.intersect(&all).unwrap()] {
    assert_eq!(r.resolve(&d).unwrap().iter().collect::<Vec<_>>(), want);
  }
  assert!(all.intersect(&all).is_err());
  let e = all.merge(&r1).unwrap_err().to_string();
  assert!(e.contains("all_convexes"), "{e}");
}

/// a sheared prism box keeps all the components of its vectors (GetFEM
/// parallelepiped_regular_prism_mesh_: the auxiliary simplices in
/// dimension N)
#[test]
fn sheared_prism_mesh() {
  use getfem_geom::regular_mesh::parallelepiped_regular_prism_mesh;
  let mut m = MeshData::new(3);
  let iv = vec![
    vec![1.0, 0.0, 0.5],
    vec![0.0, 1.0, 0.0],
    vec![0.0, 0.0, 1.0],
  ];
  parallelepiped_regular_prism_mesh(&mut m, 3, &[0.0; 3], &iv, &[1, 1, 1]).unwrap();
  let pts: Vec<Vec<f64>> = m.point_index().map(|i| m.point(i).to_vec()).collect();
  assert_eq!(pts.len(), 8);
  for x in [[1.0, 0.0, 0.5], [1.0, 1.0, 1.5], [0.0, 1.0, 1.0]] {
    assert!(
      pts
        .iter()
        .any(|p| p.iter().zip(&x).all(|(a, b)| (a - b).abs() < 1e-14)),
      "{x:?}"
    );
  }
}

#[test]
fn merge_points_closes_a_seam() {
  use getfem_geom::geotrans::qk;
  use getfem_geom::regular_mesh::regular_unit_mesh;
  // unit square, 8 x 2 quads, wrapped into an annulus: x = 0, 1 coincide
  let m = regular_unit_mesh(&[8, 2], &qk(2, 1).unwrap()).unwrap();
  let moves: Vec<(usize, Vec<f64>)> = m
    .point_index()
    .into_iter()
    .map(|i| {
      let p = m.point(i);
      let a = 2.0 * std::f64::consts::PI * p[0];
      (i, vec![(1.0 + p[1]) * a.cos(), (1.0 + p[1]) * a.sin()])
    })
    .collect();
  m.set_points(&moves).unwrap();
  let nfaces = |m: &getfem_geom::mesh::Mesh| {
    outer_faces_of_mesh(m)
      .resolve(&m.snapshot())
      .unwrap()
      .size()
  };
  assert_eq!((m.nb_points(), nfaces(&m)), (27, 20));
  // the seam's points are not in convex 1 alone
  let mut r = RegionSet::new();
  r.add(1);
  m.set_region_set(1, r).unwrap();
  assert_eq!(m.merge_points(0.0, 1).unwrap(), 0);
  // too large a tolerance: a convex would get a point twice, mesh unchanged
  assert!(m.merge_points(0.5, -1).is_err());
  assert_eq!(m.nb_points(), 27);
  assert_eq!(m.merge_points(0.0, -1).unwrap(), 3);
  assert_eq!((m.nb_points(), m.nb_points_max(), nfaces(&m)), (24, 27, 16));
  m.optimize_structure();
  assert_eq!((m.nb_points_max(), nfaces(&m)), (24, 16));
}
