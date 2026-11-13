//! Signed distances of the reference elements and projections onto
//! constraints (the mesher parts used to cut elements by level sets).

use getfem_geom::convex_structure::{
  parallelepiped_structure, prism_p1_structure, simplex_structure,
};
use getfem_geom::signed_distance::{
  HalfSpace, RefElement, SignedDistance, curvature_radius_estimate, multi_projection,
  try_projection,
};

/// x^2 + y^2 - r^2 (a level set of a disc)
struct Circle(f64);
impl SignedDistance for Circle {
  fn val(&self, p: &[f64]) -> f64 {
    p[0] * p[0] + p[1] * p[1] - self.0 * self.0
  }
  fn grad(&self, p: &[f64]) -> (f64, Vec<f64>) {
    (self.val(p), vec![2.0 * p[0], 2.0 * p[1]])
  }
  fn hess(&self, _: &[f64]) -> Vec<f64> {
    vec![2.0, 0.0, 0.0, 2.0]
  }
}

#[test]
fn reference_elements() {
  let tri = RefElement::of(&simplex_structure(2)).unwrap();
  assert_eq!(tri.faces().len(), 3);
  for v in [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]] {
    assert!(tri.val(&v).abs() < 1e-15, "{v:?}");
  }
  assert!(tri.val(&[0.2, 0.2]) < 0.0 && tri.val(&[0.6, 0.6]) > 0.0);
  let cube = RefElement::of(&parallelepiped_structure(3, 1)).unwrap();
  assert_eq!(cube.faces().len(), 6);
  assert!((cube.val(&[0.5, 0.5, 0.5]) + 0.5).abs() < 1e-15);
  assert!(cube.val(&[1.0, 0.3, 0.7]).abs() < 1e-15);
  // the prism's sloped face is x + y = 1 (GetFEM: 2/3)
  let prism = RefElement::of(&prism_p1_structure(3)).unwrap();
  assert_eq!(prism.faces().len(), 5);
  for v in [
    [0.5, 0.5, 0.3],
    [1.0, 0.0, 1.0],
    [0.0, 0.0, 0.0],
    [0.2, 0.2, 1.0],
  ] {
    assert!(prism.val(&v).abs() < 1e-15, "{v:?}");
  }
  assert!(prism.val(&[0.3, 0.3, 0.5]) < 0.0);
  let (d, g) = tri.grad(&[0.6, 0.6]);
  assert!(d > 0.0 && (g[0] - g[1]).abs() < 1e-15 && g[0] > 0.0);
}

#[test]
fn projections() {
  let c = Circle(0.75);
  // onto the circle
  let mut x = vec![0.3, 0.1];
  assert!(try_projection(&c, &mut x, true));
  assert!(c.val(&x).abs() < 1e-14);
  // into the disc: inside points stay
  let mut y = vec![0.1, 0.1];
  assert!(try_projection(&c, &mut y, false) && y == vec![0.1, 0.1]);
  // circle and the line y = x/2 (a half space): the intersection
  let line = HalfSpace::new(&[0.0, 0.0], &[-0.5, 1.0]);
  let mut z = vec![0.5, 0.5];
  assert!(multi_projection(&[&c, &line], &mut z));
  assert!(c.val(&z).abs() < 1e-8 && line.val(&z).abs() < 1e-8);
  assert!((z[1] - z[0] / 2.0).abs() < 1e-12);
  // two parallel half spaces: no intersection
  let a = HalfSpace::new(&[0.0, 0.0], &[0.0, 1.0]);
  let b = HalfSpace::new(&[0.0, 1.0], &[0.0, 1.0]);
  let mut w = vec![0.3, 0.4];
  assert!(!multi_projection(&[&a, &b], &mut w));
  // the corner of the reference triangle: two of its faces
  let tri = RefElement::of(&simplex_structure(2)).unwrap();
  let f = tri.faces();
  let mut v = vec![0.9, 0.2];
  assert!(multi_projection(&[&f[1], &f[2]], &mut v));
  assert!((v[0] - 1.0).abs() < 1e-12 && v[1].abs() < 1e-12, "{v:?}");
  // curvature radius of the circle
  assert!((curvature_radius_estimate(&c, &[0.3, 0.2], true) - 0.75).abs() < 1e-12);
}

/// points put exactly on the faces of the reference elements: the exact
/// orientation sees them on the face
#[test]
fn snapping_onto_faces() {
  use getfem_geom::predicates::orient;
  let tet = RefElement::of(&simplex_structure(3)).unwrap();
  let v: [&[f64]; 3] = [&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]];
  for p0 in [
    [0.3, 0.3, 0.4 + 1e-16],
    [0.1, 0.7, 0.2 - 3e-17],
    [0.123456789, 0.333, 0.543543211],
  ] {
    let mut p = p0.to_vec();
    for f in tet.faces() {
      f.snap(&mut p, 1e-12);
    }
    assert_eq!(orient(&[v[0], v[1], v[2], &p]), 0, "{p0:?} -> {p:?}");
    assert!(p.iter().zip(&p0).all(|(a, b)| (a - b).abs() < 1e-11));
  }
  // an axis face, and a point away from all faces untouched
  let mut p = vec![0.2, -4e-17, 0.3];
  let mut q = vec![0.2, 0.25, 0.3];
  for f in tet.faces() {
    f.snap(&mut p, 1e-12);
    f.snap(&mut q, 1e-12);
  }
  assert_eq!(p, vec![0.2, 0.0, 0.3]);
  assert_eq!(q, vec![0.2, 0.25, 0.3]);
}
