//! M10: GWFL contact transformations: errors, variables first met inside
//! a filter.

mod common;

use getfem_fem::mesh_fem::MeshFem;
use getfem_fem::mesh_im::MeshIm;
use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::mesh::Mesh;
use getfem_geom::region::{outer_faces_of_mesh, select_faces_of_normal};
use getfem_geom::regular_mesh::regular_mesh;
use getfem_gwfl::workspace::Workspace;

fn values(mf: &MeshFem, f: impl Fn(&[f64], usize) -> f64) -> Vec<f64> {
  (0..mf.nb_dof().unwrap())
    .map(|i| f(&mf.point_of_basic_dof(i).unwrap(), i % 2))
    .collect()
}

/// the setup of gwfl_contact.cc
fn setup() -> (Workspace, MeshIm) {
  let m1 = regular_mesh(
    &geotrans_descriptor("GT_PK(2,1)").unwrap(),
    &[4, 2],
    &[0.0, 0.0],
    &[1.0, 0.5],
  )
  .unwrap();
  let m2 = regular_mesh(
    &geotrans_descriptor("GT_QK(2,1)").unwrap(),
    &[3, 2],
    &[-0.3, -0.4],
    &[0.9, 0.38],
  )
  .unwrap();
  let r1 = select_faces_of_normal(&m1, outer_faces_of_mesh(&m1), &[0.0, -1.0], 0.01).unwrap();
  let r2 = select_faces_of_normal(&m2, outer_faces_of_mesh(&m2), &[0.0, 1.0], 0.01).unwrap();
  m1.set_region(1, &r1).unwrap();
  m2.set_region(2, &r2).unwrap();
  let mf1 = MeshFem::new(&m1, 2)
    .with_classical_finite_element(2)
    .unwrap();
  let mf2 = MeshFem::new(&m2, 2)
    .with_classical_finite_element(1)
    .unwrap();
  let mim = MeshIm::new(&m1).with_integration_method(4).unwrap();
  let mut ws = Workspace::new();
  ws.add_fem_variable("u1", &mf1).unwrap();
  ws.add_fem_variable("u2", &mf2).unwrap();
  let u1 = values(&mf1, |p, c| {
    if c == 0 {
      0.01 * (3.0 * p[0] + p[1]).sin()
    } else {
      -0.045 + 0.015 * (2.0 * p[0]).cos() + 0.01 * p[1]
    }
  });
  let u2 = values(&mf2, |p, c| {
    if c == 0 {
      0.005 * (2.0 * p[1] + p[0]).cos()
    } else {
      0.01 * (2.0 * p[0]).sin()
    }
  });
  ws.set_variable("u1", &u1).unwrap();
  ws.set_variable("u2", &u2).unwrap();
  ws.add_raytracing_transformation("ray", 0.05).unwrap();
  ws.add_slave_contact_boundary_to_raytracing_transformation("ray", &m1, "u1", 1)
    .unwrap();
  ws.add_master_contact_boundary_to_raytracing_transformation("ray", &m2, "u2", 2)
    .unwrap();
  ws.add_rigid_obstacle_to_raytracing_transformation("ray", "y+0.06-0.1*x", 2)
    .unwrap();
  ws.add_projection_transformation("proj", 0.05).unwrap();
  ws.add_slave_contact_boundary_to_projection_transformation("proj", &m1, "u1", 1)
    .unwrap();
  ws.add_master_contact_boundary_to_projection_transformation("proj", &m2, "u2", 2)
    .unwrap();
  (ws, mim)
}

/// A variable first met inside Interpolate_filter, then another one
/// outside it (the compiled caches restored after the filter keep one
/// entry per variable; the contact example crashed).
#[test]
fn variables_first_met_inside_a_filter() {
  let (mut ws, mim) = setup();
  let mut s = |e: &str| {
    ws.clear_expressions();
    ws.add_expression(e, &mim, 1).unwrap();
    ws.assembly(0, false).unwrap();
    ws.assembled_potential()
  };
  let f = "Interpolate_filter(ray, Interpolate(u2,ray)(2), 1)";
  let both = s(&format!("{f} + u1(2)"));
  assert!((both - (s(f) + s("u1(2)"))).abs() <= 1e-12 * both.abs().max(1.0));
}

/// Wrong uses give errors: points without target outside
/// Interpolate_filter, rigid obstacles outside Interpolate_filter(..., 2),
/// volume integrals, faces on no contact boundary, surface meshes.
#[test]
fn contact_errors() {
  let (mut ws, mim) = setup();
  let mut err = |e: &str, rg: i32| {
    ws.clear_expressions();
    let r = ws
      .add_expression(e, &mim, rg)
      .and_then(|_| ws.assembly(0, false));
    r.unwrap_err().to_string()
  };
  let e = err("Interpolate(u2,ray).Normal", 1);
  assert!(e.contains("Interpolate_filter"), "{e}");
  let e = err(
    "Interpolate_filter(ray, Interpolate(X,ray)(1), 1) + Interpolate(X,ray)(2)",
    1,
  );
  assert!(e.contains("Interpolate_filter"), "{e}");
  let e = err("Interpolate_filter(ray, 1, 1)", -1);
  assert!(e.contains("faces"), "{e}");
  // all boundary faces: some on no contact boundary
  let m1 = mim.linked_mesh();
  m1.set_region(7, &outer_faces_of_mesh(&m1)).unwrap();
  let e = err("Interpolate_filter(ray, 1, 1)", 7);
  assert!(e.contains("no contact boundary"), "{e}");

  // triangles in 3D: faces are edges
  let gt = geotrans_descriptor("GT_PK(2,1)").unwrap();
  let surface = |z: f64| {
    let m = Mesh::new(3);
    m.modify(|d| {
      d.add_convex_by_points(&gt, &[&[0.0, 0.0, z], &[1.0, 0.0, z], &[0.0, 1.0, z]])
        .unwrap();
      d.add_convex_by_points(&gt, &[&[1.0, 0.0, z], &[1.0, 1.0, z], &[0.0, 1.0, z]])
        .unwrap();
    });
    m.set_region(1, &outer_faces_of_mesh(&m)).unwrap();
    m
  };
  let (s1, s2) = (surface(0.0), surface(-0.01));
  let mf1 = MeshFem::new(&s1, 3)
    .with_classical_finite_element(1)
    .unwrap();
  let mf2 = MeshFem::new(&s2, 3)
    .with_classical_finite_element(1)
    .unwrap();
  let mim = MeshIm::new(&s1).with_integration_method(2).unwrap();
  let mut ws = Workspace::new();
  ws.add_fem_variable("u1", &mf1).unwrap();
  ws.add_fem_variable("u2", &mf2).unwrap();
  ws.add_raytracing_transformation("ray", 0.1).unwrap();
  ws.add_slave_contact_boundary_to_raytracing_transformation("ray", &s1, "u1", 1)
    .unwrap();
  ws.add_master_contact_boundary_to_raytracing_transformation("ray", &s2, "u2", 1)
    .unwrap();
  let e = ws
    .add_expression("Interpolate_filter(ray, 1, 1)", &mim, 1)
    .and_then(|_| ws.assembly(0, false))
    .unwrap_err()
    .to_string();
  assert!(e.contains("dimension 2 in a mesh of dimension 3"), "{e}");
  // a displacement on another mesh
  let e = ws
    .add_slave_contact_boundary_to_raytracing_transformation("ray", &s1, "u2", 1)
    .unwrap_err()
    .to_string();
  assert!(e.contains("not on the mesh"), "{e}");
}
