//! Shared handles (SPEC 3, 12): changes of a mesh or space are seen by the
//! workspace and the model; values invalidated by a new numbering are an
//! error until the user sets them, also after point motion; cached term
//! matrices follow the geometry.

use getfem_fem::mesh_fem::MeshFem;
use getfem_fem::mesh_im::MeshIm;
use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::region::{outer_faces_of_mesh, select_faces_of_normal};
use getfem_geom::regular_mesh::regular_unit_mesh;
use getfem_model::{BuildVersion, Model, Newton};

#[test]
fn changes_propagate_and_invalidate_values() {
  let mesh = regular_unit_mesh(&[2, 2], &geotrans_descriptor("GT_PK(2,1)").unwrap()).unwrap();
  let mf = MeshFem::new(&mesh, 1)
    .with_classical_finite_element(1)
    .unwrap();
  let mim = MeshIm::new(&mesh).with_integration_method(2).unwrap();
  let mut md = Model::new();
  md.add_fem_variable("u", &mf).unwrap();
  md.add_fem_data("g", &mf, &[1.0; 9]).unwrap();
  md.add_term(&mim, None, "Grad_u.Grad_Test_u - g*Test_u")
    .unwrap();
  mesh.set_region(1, &outer_faces_of_mesh(&mesh)).unwrap();
  md.add_Dirichlet_condition_with_simplification("u", 1, None)
    .unwrap();
  assert!(Newton::default().solve(&mut md, None).unwrap().converged());

  // P2 through the same handle: u and g are invalidated, with a message
  mf.set_classical_finite_element(2).unwrap();
  let e = md.assembly(BuildVersion::All).unwrap_err().to_string();
  println!("{e}");
  assert!(e.contains("u (9 values, the space now has 25)") && e.contains("g (9 values"));
  // the user sets new values (here by hand), then the model works again
  md.set_variable("u", &[0.0; 25]).unwrap();
  md.set_variable("g", &[1.0; 25]).unwrap();
  assert_eq!(md.nb_dof(), 25);
  assert!(Newton::default().solve(&mut md, None).unwrap().converged());

  // point motion invalidates the values too; after they are set again,
  // the cached matrix follows the geometry
  let c = mesh
    .point_index()
    .into_iter()
    .find(|&i| mesh.point(i) == [0.5, 0.5])
    .unwrap();
  mesh.set_point(c, &[0.6, 0.45]).unwrap();
  let e = md.assembly(BuildVersion::All).unwrap_err().to_string();
  assert!(e.contains("u (25 values, the space now has 25)"), "{e}");
  let u = md.variable("u").unwrap().to_vec();
  md.set_variable("u", &u).unwrap();
  md.set_variable("g", &[1.0; 25]).unwrap();
  // regions do not invalidate values
  mesh.set_region(7, &mesh.all_convexes()).unwrap();
  md.assembly(BuildVersion::All).unwrap();
  md.assembly(BuildVersion::All).unwrap();
  let r: Vec<f64> = md.rhs().iter().map(|v| -v).collect();
  let direct = {
    let ws = md.workspace_mut();
    ws.clear_expressions();
    ws.add_expression("Grad_u.Grad_Test_u - g*Test_u", &mim, None)
      .unwrap();
    ws.assembly(1, false).unwrap();
    ws.assembled_vector()
  };
  let interior: Vec<usize> = (0..25)
    .filter(|&i| r[i].abs() > 0.0 || direct[i].abs() > 0.0)
    .collect();
  let fixed = mf.basic_dof_on_region(outer_faces_of_mesh(&mesh)).unwrap();
  let d = interior
    .iter()
    .filter(|i| !fixed.contains(i))
    .map(|&i| (r[i] - direct[i]).abs())
    .fold(0.0, f64::max);
  assert!(d < 1e-12, "cached matrix not updated: {d}");
}

#[test]
fn partial_space_of_an_old_numbering() {
  let mesh = regular_unit_mesh(&[2, 2], &geotrans_descriptor("GT_PK(2,1)").unwrap()).unwrap();
  let mf = MeshFem::new(&mesh, 1)
    .with_classical_finite_element(1)
    .unwrap();
  let part = MeshFem::partial(
    &mf,
    &mf.basic_dof_on_region(outer_faces_of_mesh(&mesh)).unwrap(),
  )
  .unwrap();
  assert_eq!(part.nb_dof().unwrap(), 8);
  mf.set_classical_finite_element(2).unwrap();
  let e = part.nb_dof().unwrap_err().to_string();
  assert!(e.contains("build it again"), "{e}");
}

#[test]
fn filtered_variable_beside_a_variable_on_the_same_space() {
  let mesh = regular_unit_mesh(&[2, 2], &geotrans_descriptor("GT_PK(2,1)").unwrap()).unwrap();
  let mf = MeshFem::new(&mesh, 1)
    .with_classical_finite_element(1)
    .unwrap();
  mesh.set_region(1, &outer_faces_of_mesh(&mesh)).unwrap();
  let border = mesh.region(1).unwrap();
  let mut md = Model::new();
  md.add_fem_variable("u", &mf).unwrap();
  let u: Vec<f64> = (0..9).map(|i| i as f64).collect();
  md.set_variable("u", &u).unwrap();
  let stamp = mf.value_stamp().unwrap();
  // the canonical form and GetFEM's one-line form give the same space;
  // mf and the variable on it are unchanged
  let p = MeshFem::partial_on_region(&mf, &border).unwrap();
  md.add_fem_variable("a", &p).unwrap();
  md.add_filtered_fem_variable("b", &mf, &border).unwrap();
  let b = md.mesh_fem_of_variable("b").unwrap();
  assert!(b.is_partial() && !b.same(&mf));
  assert_eq!((p.nb_dof().unwrap(), b.nb_dof().unwrap()), (8, 8));
  assert_eq!(mf.nb_dof().unwrap(), 9);
  assert_eq!(mf.value_stamp().unwrap(), stamp);
  assert_eq!(md.variable("u").unwrap(), &u[..]);
  assert!(md.check_spaces().is_ok());
}

#[test]
fn region_partial_space_follows_its_base() {
  let mesh = regular_unit_mesh(&[2, 2], &geotrans_descriptor("GT_PK(2,1)").unwrap()).unwrap();
  let mf = MeshFem::new(&mesh, 1)
    .with_classical_finite_element(1)
    .unwrap();
  // temporary regions cannot be kept by a space
  let temporary = outer_faces_of_mesh(&mesh);
  let e = MeshFem::partial_on_region(&mf, &temporary)
    .unwrap_err()
    .to_string();
  assert!(e.contains("mesh.set_region"), "{e}");
  mesh.set_region(1, &temporary).unwrap();
  let border = mesh.region(1).unwrap();
  let explicit = MeshFem::partial(&mf, &mf.basic_dof_on_region(&border).unwrap()).unwrap();
  let on_region = MeshFem::partial_on_region(&mf, &border).unwrap();
  let mut md = Model::new();
  md.add_filtered_fem_variable("l", &mf, &border).unwrap();
  let stamp = on_region.value_stamp().unwrap();
  // P2 on the base: the region space follows (16 DOFs on the border), the
  // explicit set must be built again, the value is invalid until set
  mf.set_classical_finite_element(2).unwrap();
  assert_eq!(on_region.nb_dof().unwrap(), 16);
  assert_ne!(on_region.value_stamp().unwrap(), stamp);
  assert!(
    explicit
      .nb_dof()
      .unwrap_err()
      .to_string()
      .contains("build it again")
  );
  let e = md.check_spaces().unwrap_err().to_string();
  assert!(e.contains("l (8 values, the space now has 16)"), "{e}");
  md.set_variable("l", &[0.0; 16]).unwrap();
  assert!(md.check_spaces().is_ok());
  // an edit of region 1 that keeps its DOFs keeps the values valid; one
  // that changes them invalidates them
  let stamp = on_region.value_stamp().unwrap();
  mesh.set_region(1, &border).unwrap();
  assert_eq!(on_region.value_stamp().unwrap(), stamp);
  let bottom = select_faces_of_normal(&mesh, &border, &[0.0, -1.0], 0.01).unwrap();
  mesh.set_region(1, &bottom).unwrap();
  assert_eq!(on_region.nb_dof().unwrap(), 5);
  let e = md.check_spaces().unwrap_err().to_string();
  assert!(e.contains("l (16 values, the space now has 5)"), "{e}");
  // removing the region id is an error at use
  mesh.sup_region(1);
  let e = on_region.nb_dof().unwrap_err().to_string();
  assert!(e.contains("region 1 does not exist"), "{e}");
}
