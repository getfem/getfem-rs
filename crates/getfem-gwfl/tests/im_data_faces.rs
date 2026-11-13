//! im_data on regions with faces (GetFEM im_data: the interior points of
//! the convexes of the region, then the points of its faces), assigned and
//! read on the faces as on the elements.

use getfem_fem::im_data::ImData;
use getfem_fem::mesh_im::MeshIm;
use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::region::{RegionSet, outer_faces_of_mesh};
use getfem_geom::regular_mesh::regular_unit_mesh;
use getfem_gwfl::workspace::Workspace;

fn setup() -> MeshIm {
  let m = regular_unit_mesh(&[2, 2], &geotrans_descriptor("GT_PK(2,1)").unwrap()).unwrap();
  m.set_region(1, &outer_faces_of_mesh(&m)).unwrap();
  // convex 0 and the outer faces: interior and face points
  let mut mixed: RegionSet = m.region_set(1).unwrap();
  mixed.add(0);
  m.set_region_set(2, mixed).unwrap();
  MeshIm::new(&m).with_integration_method(4).unwrap()
}

fn close(a: f64, b: f64) -> bool {
  (a - b).abs() < 1e-13 * b.abs().max(1.0)
}

#[test]
fn numbering_interior_then_faces() {
  let mim = setup();
  let m = mim.linked_mesh();
  let ims = mim.ims().unwrap();
  let ai = |cv: usize| ims.int_method_of_element(cv).unwrap().approx().unwrap();
  let faces: Vec<(usize, usize)> = m
    .region_set(1)
    .unwrap()
    .iter()
    .map(|(cv, f)| (cv, f.unwrap()))
    .collect();
  let on_faces: usize = faces
    .iter()
    .map(|&(cv, f)| ai(cv).nb_points_on_face(f))
    .sum();
  let interior0 = ai(0).nb_points_on_convex();
  // faces only, and convex 0 with the faces
  let p1 = ImData::new(&mim, 1, &[]).unwrap().points().unwrap();
  let p2 = ImData::new(&mim, 2, &[]).unwrap().points().unwrap();
  assert_eq!(p1.nb_points(), on_faces);
  assert_eq!(p2.nb_points(), interior0 + on_faces);
  // the points of convex 0 first, then those of the faces in order
  assert_eq!(p1.index_of_point(0, 0), None);
  assert_eq!(p2.index_of_point(0, interior0 - 1), Some(interior0 - 1));
  let (cv, f) = faces[0];
  let a = ai(cv).ind_first_point_on_face(f);
  assert_eq!(p1.index_of_point(cv, a), Some(0));
  assert_eq!(p2.index_of_point(cv, a), Some(interior0));
  // a face of the convex not in the region: no point
  let fin = (0..ai(cv).nb_convex_faces())
    .find(|g| !faces.contains(&(cv, *g)))
    .unwrap();
  assert_eq!(
    p1.index_of_point(cv, ai(cv).ind_first_point_on_face(fin)),
    None
  );
}

#[test]
fn values_assigned_and_read_on_faces() {
  let mim = setup();
  let mut ws = Workspace::new();
  let (s1, v1) = (
    ImData::new(&mim, 1, &[]).unwrap(),
    ImData::new(&mim, 1, &[2]).unwrap(),
  );
  ws.add_im_data("q", &s1, &vec![0.0; s1.nb_values().unwrap()])
    .unwrap();
  ws.add_im_data("n", &v1, &vec![0.0; v1.nb_values().unwrap()])
    .unwrap();
  ws.set_variable("q", &ws.interpolation("X(1)+2*X(2)", &s1, 1).unwrap())
    .unwrap();
  ws.set_variable("n", &ws.interpolation("Normal", &v1, 1).unwrap())
    .unwrap();
  let mut s = |e: &str, r: i32| {
    ws.clear_expressions();
    ws.add_expression(e, &mim, r).unwrap();
    ws.assembly(0, false).unwrap();
    ws.assembled_potential()
  };
  // the boundary integral of X(1)+2*X(2) on the unit square: 6
  assert!(close(s("q", 1), s("X(1)+2*X(2)", 1)) && close(s("q", 1), 6.0));
  // the stored normals: n.Normal = 1 on the perimeter (4)
  assert!(close(s("n.Normal", 1), 4.0));
  // interpolation on the face points gives the assigned values
  let q = ws.variable("q").unwrap().to_vec();
  let i = ws.interpolation("X(1)+2*X(2)", &s1, 1).unwrap();
  assert!(q.iter().zip(&i).all(|(a, b)| close(*a, *b)));
}

#[test]
fn convex_and_faces_together() {
  let mim = setup();
  let mut ws = Workspace::new();
  let imd = ImData::new(&mim, 2, &[]).unwrap();
  ws.add_im_data("q", &imd, &vec![0.0; imd.nb_values().unwrap()])
    .unwrap();
  ws.set_variable("q", &ws.interpolation("sin(X(1))*X(2)", &imd, 2).unwrap())
    .unwrap();
  // on the mixed region: the integral over convex 0 plus the perimeter
  let mut s = |e: &str| {
    ws.clear_expressions();
    ws.add_expression(e, &mim, 2).unwrap();
    ws.assembly(0, false).unwrap();
    ws.assembled_potential()
  };
  assert!(close(s("q"), s("sin(X(1))*X(2)")));
}
