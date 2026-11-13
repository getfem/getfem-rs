//! Level sets: values on a Lagrange space, the function of each convex in
//! its reference coordinates (exact for polynomials of the degree),
//! simplify, errors.

use getfem_fem::level_set::LevelSet;
use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::regular_mesh::regular_mesh;
use getfem_geom::signed_distance::{SignedDistance, try_projection};

#[test]
fn level_set_of_convex() {
  // [0, 2] x [0, 1], 4 x 2 quadrilaterals: element size 0.5
  let m = regular_mesh(
    &geotrans_descriptor("GT_QK(2,1)").unwrap(),
    &[4, 2],
    &[0.0, 0.0],
    &[2.0, 1.0],
  )
  .unwrap();
  let ls = LevelSet::new(&m, 2, true).unwrap();
  let mf = ls.mesh_fem();
  let n = mf.nb_dof().unwrap();
  let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1] - 0.64;
  let v: Vec<f64> = (0..n)
    .map(|i| f(&mf.point_of_basic_dof(i).unwrap()))
    .collect();
  let v1: Vec<f64> = (0..n)
    .map(|i| mf.point_of_basic_dof(i).unwrap()[0] - 1.1)
    .collect();
  let version = ls.version();
  ls.set_values(0, &v).unwrap();
  ls.set_values(1, &v1).unwrap();
  assert_ne!(ls.version(), version);
  // convex 3: its reference point (0.3, 0.6)
  let m0 = m.snapshot();
  let cv = 3;
  let g = m0.points_of_convex(cv);
  let xref = [0.3, 0.6];
  let x = m0.trans_of_convex(cv).transform(&xref, &g).unwrap();
  let c = ls.of_convex(cv, 0).unwrap();
  assert!((c.val(&xref).unwrap() - f(&x)).abs() < 1e-14);
  // reference gradient: the element is 0.5 x 0.5
  let gr = c.grad(&xref).unwrap();
  assert!((gr[0] - 0.5 * 2.0 * x[0]).abs() < 1e-13 && (gr[1] - 0.5 * 2.0 * x[1]).abs() < 1e-13);
  let h = c.hess(&xref).unwrap();
  assert!((h[0] - 0.5).abs() < 1e-13 && h[1].abs() < 1e-13 && (h[3] - 0.5).abs() < 1e-13);
  let s = ls.of_convex(cv, 1).unwrap();
  assert!((s.val(&xref).unwrap() - (x[0] - 1.1)).abs() < 1e-14);
  // as a constraint: projection onto the circle in reference coordinates
  let mut ptref = vec![0.5, 0.5];
  assert!(try_projection(&c, &mut ptref, true));
  let xp = m0.trans_of_convex(cv).transform(&ptref, &g).unwrap();
  assert!(f(&xp).abs() < 1e-13 && SignedDistance::val(&c, &ptref).abs() < 1e-14);
  // simplify: values below eps times the radius estimate (0.5 sqrt(2)/2)
  let small: Vec<f64> = v1.iter().map(|_| 0.01).collect();
  ls.set_values(1, &small).unwrap();
  ls.simplify(0.03).unwrap();
  assert!(ls.values(1).unwrap().iter().all(|&v| v == 0.0));
  ls.set_values(1, &small).unwrap();
  ls.simplify(0.02).unwrap();
  assert!(ls.values(1).unwrap().iter().all(|&v| v == 0.01));
  // errors
  assert!(ls.set_values(0, &[1.0]).is_err());
  assert!(LevelSet::new(&m, 1, false).unwrap().values(1).is_err());
  assert!(ls.of_convex(99, 0).is_err());
}
