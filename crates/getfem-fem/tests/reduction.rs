//! Reduction and extension matrices of a space (set_reduction_matrices,
//! reduce_to_basic_dof, extend_vector, reduce_vector; partial spaces of a
//! reduced space).

use std::collections::BTreeSet;

use getfem_fem::mesh_fem::MeshFem;
use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::regular_mesh::regular_unit_mesh;
use getfem_linalg::{Coo, Csr};

fn space(qdim: usize, k: usize) -> MeshFem {
  let m = regular_unit_mesh(&[2, 2], &geotrans_descriptor("GT_PK(2,1)").unwrap()).unwrap();
  MeshFem::new(&m, qdim)
    .with_classical_finite_element(k)
    .unwrap()
}

fn dense(a: &Csr<f64>) -> Vec<Vec<f64>> {
  a.to_dense()
}

fn mat(rows: usize, cols: usize, entries: &[(usize, usize, f64)]) -> Csr<f64> {
  let mut c = Coo::new(rows, cols);
  for &(i, j, v) in entries {
    c.push(i, j, v);
  }
  c.to_csr().unwrap()
}

/// the rotation by angle t of the two components at each node of a qdim 2
/// space (basic DOF 2 i + c): square, orthogonal (R = E^T, R E = I)
fn rotation(nb: usize, t: f64) -> (Csr<f64>, Csr<f64>) {
  let (c, s) = (t.cos(), t.sin());
  let mut e = vec![];
  for i in 0..nb / 2 {
    let (a, b) = (2 * i, 2 * i + 1);
    e.extend([(a, a, c), (a, b, -s), (b, a, s), (b, b, c)]);
  }
  let e = mat(nb, nb, &e);
  (e.transpose(), e)
}

fn seq(n: usize) -> Vec<f64> {
  (0..n).map(|i| (0.7 * (i as f64 + 1.0)).sin()).collect()
}

fn close(a: &[f64], b: &[f64]) -> bool {
  a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-14)
}

#[test]
fn reduce_to_basic_dof_selects() {
  let mf = space(1, 1);
  let nb = mf.nb_basic_dof().unwrap();
  assert!(!mf.is_reduced());
  assert_eq!(mf.reduction_matrix().unwrap(), None);
  let kept: BTreeSet<usize> = [0, 2, 3, 7].into();
  mf.reduce_to_basic_dof(&kept).unwrap();
  assert!(mf.is_reduced());
  assert_eq!((mf.nb_dof().unwrap(), mf.nb_basic_dof().unwrap()), (4, nb));
  let r = mf.reduction_matrix().unwrap().unwrap();
  let e = mf.extension_matrix().unwrap().unwrap();
  assert_eq!((r.nrows, r.ncols, e.nrows, e.ncols), (4, nb, nb, 4));
  // extension: the kept basic DOFs, zero elsewhere; reduce(extend) = id
  let v = [1.0, 2.0, 3.0, 4.0];
  let x = mf.extend_vector(&v).unwrap();
  for (d, &xd) in x.iter().enumerate() {
    let want = kept.iter().position(|&k| k == d).map_or(0.0, |i| v[i]);
    assert_eq!(xd, want, "basic DOF {d}");
  }
  assert_eq!(mf.reduce_vector(&x).unwrap(), v);
  // as a partial space on the same DOFs
  let full = space(1, 1);
  let p = MeshFem::partial(&full, &kept).unwrap();
  assert_eq!(p.extend_vector(&v).unwrap(), x);
  assert_eq!(dense(&p.reduction_matrix().unwrap().unwrap()), dense(&r));
}

#[test]
fn square_rotation() {
  // a square reduction: the components rotated at each node, same number
  // of DOFs
  let mf = space(2, 1);
  let nb = mf.nb_basic_dof().unwrap();
  let (r, e) = rotation(nb, 0.3);
  mf.set_reduction_matrices(Some(&r), &e).unwrap();
  assert!(mf.is_reduced());
  assert_eq!(mf.nb_dof().unwrap(), nb);
  let v = seq(nb);
  let x = mf.extend_vector(&v).unwrap();
  assert!(close(&x, &e.matvec(&v).unwrap()));
  assert!(close(&mf.reduce_vector(&x).unwrap(), &v));
  // two values per DOF (interleaved): E on each
  let w: Vec<f64> = v.iter().flat_map(|&a| [a, -2.0 * a]).collect();
  let xw = mf.extend_vector(&w).unwrap();
  let x2 = e
    .matvec(&v.iter().map(|a| -2.0 * a).collect::<Vec<_>>())
    .unwrap();
  for d in 0..nb {
    assert!((xw[2 * d] - x[d]).abs() < 1e-14 && (xw[2 * d + 1] - x2[d]).abs() < 1e-14);
  }
}

#[test]
fn reduction_switched_off_and_on() {
  let mf = space(1, 1);
  let nb = mf.nb_basic_dof().unwrap();
  let s0 = mf.value_stamp().unwrap();
  assert!(mf.set_reduction(true).is_err(), "no matrices set");
  mf.reduce_to_basic_dof(&[1, 4].into()).unwrap();
  let s1 = mf.value_stamp().unwrap();
  assert_ne!(s0, s1);
  assert_eq!(mf.nb_dof().unwrap(), 2);
  mf.set_reduction(false).unwrap();
  assert!(!mf.is_reduced());
  assert_eq!(mf.nb_dof().unwrap(), nb);
  assert_eq!(mf.value_stamp().unwrap(), s0);
  mf.set_reduction(true).unwrap();
  assert_eq!((mf.nb_dof().unwrap(), mf.value_stamp().unwrap()), (2, s1));
}

#[test]
fn wrong_dimensions_and_stale_matrices() {
  let mf = space(1, 1);
  let nb = mf.nb_basic_dof().unwrap();
  let r = mat(2, nb, &[(0, 0, 1.0), (1, 1, 1.0)]);
  assert!(
    mf.set_reduction_matrices(Some(&r), &mat(nb, 3, &[]))
      .is_err()
  );
  assert!(
    mf.set_reduction_matrices(Some(&r), &mat(nb + 1, 2, &[]))
      .is_err()
  );
  assert!(mf.reduce_to_basic_dof(&[nb].into()).is_err());
  assert!(!mf.is_reduced());
  // set for this numbering; another one (degree 2) needs new matrices
  mf.set_reduction_matrices(Some(&r), &r.transpose()).unwrap();
  mf.set_classical_finite_element(2).unwrap();
  let err = mf.nb_dof().unwrap_err().to_string();
  assert!(err.contains("older numbering"), "{err}");
  // a partial space is reduced through its base
  let base = space(1, 1);
  let p = MeshFem::partial(&base, &[0, 1].into()).unwrap();
  assert!(p.is_reduced());
  assert!(p.set_reduction_matrices(Some(&r), &r.transpose()).is_err());
}

#[test]
fn partial_space_of_a_reduced_space() {
  // GetFEM partial_mesh_fem: R = RR R_base, E = E_base RR^T, the kept
  // DOFs among the DOFs of the reduced base
  let base = space(2, 1);
  let nb = base.nb_basic_dof().unwrap();
  let (r, e) = rotation(nb, 0.5);
  base.set_reduction_matrices(Some(&r), &e).unwrap();
  let kept: BTreeSet<usize> = [1, 2, 5].into();
  let p = MeshFem::partial(&base, &kept).unwrap();
  assert_eq!(p.nb_dof().unwrap(), 3);
  let v = [0.3, -1.0, 2.0];
  // on the base DOFs: v on the kept ones, zero elsewhere; then E_base
  let mut vb = vec![0.0; nb];
  for (i, &k) in kept.iter().enumerate() {
    vb[k] = v[i];
  }
  assert!(close(
    &p.extend_vector(&v).unwrap(),
    &e.matvec(&vb).unwrap()
  ));
  assert!(close(
    &p.reduce_vector(&p.extend_vector(&v).unwrap()).unwrap(),
    &v
  ));
  let pe = p.extension_matrix().unwrap().unwrap();
  assert_eq!((pe.nrows, pe.ncols), (nb, 3));
  // the DOFs were those of this reduction of the base: an error once it
  // changed (as for a new numbering)
  base.set_reduction(false).unwrap();
  let err = p.extend_vector(&v).unwrap_err().to_string();
  assert!(err.contains("build it again"), "{err}");
  base.set_reduction(true).unwrap();
  assert!(close(
    &p.extend_vector(&v).unwrap(),
    &e.matvec(&vb).unwrap()
  ));
  let mut fewer = Coo::new(2, nb);
  fewer.push(0, 0, 1.0);
  fewer.push(1, 1, 1.0);
  let fewer = fewer.to_csr().unwrap();
  base
    .set_reduction_matrices(Some(&fewer), &fewer.transpose())
    .unwrap();
  assert!(p.nb_dof().is_err());
}

#[test]
fn reduction_from_the_extension_matrix() {
  // R "auto" (None): E' E diagonal (DOF 0 copied to DOF 1, a periodicity,
  // the others kept): R = (E'E)^-1 E' stored, the mean of the copies
  let mf = space(1, 1);
  let nb = mf.nb_basic_dof().unwrap();
  let mut ee = vec![(0, 0, 1.0), (1, 0, 1.0)];
  ee.extend((2..nb).map(|i| (i, i - 1, 1.0)));
  let e = mat(nb, nb - 1, &ee);
  mf.set_reduction_matrices(None, &e).unwrap();
  let r = mf.reduction_matrix().unwrap().unwrap();
  assert_eq!(r.get(0, 0), 0.5);
  assert_eq!(r.get(0, 1), 0.5);
  let v = seq(nb - 1);
  assert!(close(
    &mf.reduce_vector(&mf.extend_vector(&v).unwrap()).unwrap(),
    &v
  ));
  // E' E not diagonal (a column on all DOFs): R not stored, least squares
  let mut ee: Vec<_> = (0..nb).map(|i| (i, nb, i as f64)).collect();
  ee.extend((0..nb).map(|i| (i, i, 1.0)));
  let e = mat(nb, nb + 1, &ee);
  // more columns than rows: not of full column rank
  assert!(mf.set_reduction_matrices(None, &e).is_err());
  let mut ee: Vec<_> = (0..nb).map(|i| (i, nb - 1, i as f64)).collect();
  ee.extend((0..nb - 1).map(|i| (i, i, 1.0)));
  let e = mat(nb, nb, &ee);
  mf.set_reduction_matrices(None, &e).unwrap();
  let err = mf.reduction_matrix().unwrap_err().to_string();
  assert!(err.contains("not stored"), "{err}");
  let v = seq(nb);
  let back = mf.reduce_vector(&mf.extend_vector(&v).unwrap()).unwrap();
  assert!(back.iter().zip(&v).all(|(a, b)| (a - b).abs() < 1e-12));
  // an explicit R that is not a left inverse of E
  let r = e.transpose();
  let err = mf
    .set_reduction_matrices(Some(&r), &e)
    .unwrap_err()
    .to_string();
  assert!(err.contains("not a left inverse"), "{err}");
}
