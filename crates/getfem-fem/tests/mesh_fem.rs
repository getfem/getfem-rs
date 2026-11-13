//! MeshFem DOF numbering: scale invariance, tensor fields, P0 DOFs on
//! faces, errors.

use getfem_fem::fem::fem_descriptor;
use getfem_fem::mesh_fem::MeshFem;
use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::region::outer_faces_of_mesh;
use getfem_geom::regular_mesh::regular_mesh;

#[test]
fn scale_invariant_numbering() {
  // same numbering for a mesh in mm, m and km
  let gt = geotrans_descriptor("GT_PK(2,2)").unwrap();
  let reference: Vec<Vec<usize>> = {
    let m = regular_mesh(&gt, &[3, 3], &[0.0; 2], &[1.0; 2]).unwrap();
    let mf = MeshFem::new(&m, 1)
      .with_classical_finite_element(3)
      .unwrap();
    m.convex_index()
      .into_iter()
      .map(|cv| mf.ind_basic_dof_of_element(cv).unwrap().to_vec())
      .collect()
  };
  for sc in [1e-3, 1e3] {
    let m = regular_mesh(&gt, &[3, 3], &[0.0; 2], &[sc; 2]).unwrap();
    let mf = MeshFem::new(&m, 1)
      .with_classical_finite_element(3)
      .unwrap();
    let got: Vec<Vec<usize>> = m
      .convex_index()
      .into_iter()
      .map(|cv| mf.ind_basic_dof_of_element(cv).unwrap().to_vec())
      .collect();
    assert_eq!(got, reference, "scale {sc}");
  }
}

#[test]
fn errors() {
  let m = regular_mesh(
    &geotrans_descriptor("GT_QK(2,1)").unwrap(),
    &[2, 2],
    &[0.0; 2],
    &[1.0; 2],
  )
  .unwrap();
  let mf = MeshFem::new(&m, 1);
  assert!(
    mf.set_finite_element(([0], &fem_descriptor("FEM_PK(2,1)").unwrap()))
      .is_err()
  );
  assert!(
    mf.set_finite_element(([99], &fem_descriptor("FEM_QK(2,1)").unwrap()))
      .is_err()
  );
  assert_eq!(mf.nb_basic_dof().unwrap(), 0);
}

#[test]
fn tensor_field() {
  let m = regular_mesh(
    &geotrans_descriptor("GT_PK(2,1)").unwrap(),
    &[1, 1],
    &[0.0; 2],
    &[1.0; 2],
  )
  .unwrap();
  let mf = MeshFem::new(&m, 1)
    .with_classical_finite_element(1)
    .unwrap();
  mf.set_qdim([2, 2]);
  assert_eq!((mf.qdim(), mf.qdims()), (4, vec![2usize, 2]));
  assert_eq!(mf.nb_basic_dof().unwrap(), 16);
  assert_eq!(
    &mf.ind_basic_dof_of_element(0).unwrap()[..5],
    &[0, 1, 2, 3, 4]
  );
}

/// a P0 discontinuous DOF lies on every face of its element (GetFEM
/// PK_fem_: lagrange_0_dof, then the discontinuous type)
#[test]
fn p0_discontinuous_dofs_on_faces() {
  use getfem_geom::regular_mesh::regular_unit_mesh;
  let m = regular_unit_mesh(&[2, 2], &geotrans_descriptor("GT_PK(2,1)").unwrap()).unwrap();
  let mf = MeshFem::new(&m, 1)
    .with_finite_element(&fem_descriptor("FEM_PK_DISCONTINUOUS(2,0)").unwrap())
    .unwrap();
  let border = outer_faces_of_mesh(&m);
  let on: Vec<usize> = mf
    .basic_dof_on_region(&border)
    .unwrap()
    .into_iter()
    .collect();
  let mut cvs: Vec<usize> = border
    .resolve(&m.snapshot())
    .unwrap()
    .iter()
    .map(|(cv, _)| cv)
    .collect();
  cvs.sort();
  cvs.dedup();
  assert_eq!(on.len(), cvs.len());
  assert!(!on.is_empty());
}
