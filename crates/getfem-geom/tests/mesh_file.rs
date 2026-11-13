//! GetFEM native .mesh files: write/read round trip, regions, errors.

use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::mesh::MeshData;
use getfem_geom::mesh_file::{read_mesh, write_mesh};
use getfem_geom::region::RegionSet;

fn same(a: &MeshData, b: &MeshData) {
  assert_eq!(a.dim(), b.dim());
  assert_eq!(
    a.point_index().collect::<Vec<_>>(),
    b.point_index().collect::<Vec<_>>()
  );
  for i in a.point_index() {
    assert_eq!(a.point(i), b.point(i));
  }
  assert_eq!(
    a.convex_index().collect::<Vec<_>>(),
    b.convex_index().collect::<Vec<_>>()
  );
  for cv in a.convex_index() {
    assert_eq!(a.trans_of_convex(cv), b.trans_of_convex(cv));
    assert_eq!(a.ind_points_of_convex(cv), b.ind_points_of_convex(cv));
  }
  assert_eq!(
    a.region_ids().collect::<Vec<_>>(),
    b.region_ids().collect::<Vec<_>>()
  );
  for r in a.region_ids() {
    assert_eq!(a.region_set(r), b.region_set(r));
  }
}

#[test]
fn test_mesh_file_part() {
  // the file I/O part of GetFEM's test_mesh.cc
  let t = geotrans_descriptor("GT_PK(2,1)").unwrap();
  let m = getfem_geom::regular_mesh::regular_unit_mesh(&[3, 3], &t).unwrap();
  let mut r3 = RegionSet::new();
  r3.add(0);
  r3.add_face(3, 1);
  r3.add(2);
  m.set_region_set(3, r3).unwrap();
  let mut r4 = RegionSet::new();
  r4.add(0);
  r4.add(5);
  m.set_region_set(4, r4).unwrap();
  m.sup_convex(0, false);
  m.sup_convex(1, false);
  // holes are kept by the file format
  let m2 = read_mesh(&write_mesh(&m)).unwrap();
  same(&m.snapshot(), &m2.snapshot());
  assert!(!m2.is_convex_valid(0) && m2.region_set(4).unwrap().is_in(5));
  m.optimize_structure();
  let m3 = read_mesh(&write_mesh(&m)).unwrap();
  same(&m.snapshot(), &m3.snapshot());
  assert_eq!(m3.region_set(3).unwrap().size(), 2);
}

#[test]
fn errors() {
  assert!(read_mesh("nothing").is_err());
  let dup = "BEGIN POINTS LIST POINT 0 0 0 POINT 1 0 0 END POINTS LIST BEGIN MESH STRUCTURE DESCRIPTION END MESH STRUCTURE DESCRIPTION";
  assert!(
    read_mesh(dup)
      .unwrap_err()
      .to_string()
      .contains("same coordinates")
  );
  let bad_gt = "BEGIN POINTS LIST POINT 0 0 0 POINT 1 1 0 END POINTS LIST BEGIN MESH STRUCTURE DESCRIPTION CONVEX 0 'GT_XX(1,1)' 0 1 END MESH STRUCTURE DESCRIPTION";
  assert!(read_mesh(bad_gt).is_err());
  let bad_pt = "BEGIN POINTS LIST POINT 0 0 0 POINT 1 1 0 END POINTS LIST BEGIN MESH STRUCTURE DESCRIPTION CONVEX 0 'GT_PK(1,1)' 0 7 END MESH STRUCTURE DESCRIPTION";
  assert!(read_mesh(bad_pt).is_err());
  // blanks inside quoted names (old files)
  let ok = "BEGIN POINTS LIST POINT 0 0 0 POINT 1 1 0 END POINTS LIST BEGIN MESH STRUCTURE DESCRIPTION CONVEX 0 'GT_PK(1, 1)' 0 1 END MESH STRUCTURE DESCRIPTION";
  assert_eq!(read_mesh(ok).unwrap().nb_convexes(), 1);
}

#[test]
fn region_entries_with_blanks() {
  // GetFEM's tokenizer: '/' is a token, blanks around it allowed
  let head = "BEGIN POINTS LIST POINT 0 0 0 POINT 1 1 0 POINT 2 2 0 END POINTS LIST BEGIN MESH STRUCTURE DESCRIPTION CONVEX 0 'GT_PK(1,1)' 0 1 CONVEX 1 'GT_PK(1,1)' 1 2 END MESH STRUCTURE DESCRIPTION";
  let m = read_mesh(&format!(
    "{head} BEGIN REGION 2 0 /1 1/ 0 1 / 1 END REGION 2"
  ))
  .unwrap();
  let r = m.region_set(2).unwrap();
  let got: Vec<_> = r.iter().collect();
  assert_eq!(got, [(0, Some(1)), (1, Some(0)), (1, Some(1))]);
  // a face number out of range: an error, not a panic
  assert!(read_mesh(&format!("{head} BEGIN REGION 2 0/99 END REGION 2")).is_err());
}
