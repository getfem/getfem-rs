//! Delaunay triangulation of points in the reference convexes (2D, 3D):
//! random points inside and on the boundary, cocircular grids; a valid
//! triangulation (positive simplices filling the convex, faces shared by
//! at most two) with no point strictly inside a circumsphere.

use std::collections::BTreeMap;

use getfem_geom::convex_ref::{
  PRef, parallelepiped_of_reference, prism_of_reference, simplex_of_reference,
};
use getfem_geom::delaunay::delaunay;
use getfem_geom::predicates::{in_sphere, orient};
use getfem_geom::simplexify::simplexified;

fn volume(p: &[&[f64]]) -> f64 {
  let n = p.len() - 1;
  let m: Vec<Vec<f64>> = (1..=n)
    .map(|i| (0..n).map(|k| p[i][k] - p[0][k]).collect())
    .collect();
  let d = match n {
    2 => m[0][0] * m[1][1] - m[0][1] * m[1][0],
    _ => {
      m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }
  };
  d / if n == 2 { 2.0 } else { 6.0 }
}

struct Lcg(u64);
impl Lcg {
  fn next(&mut self) -> f64 {
    self.0 = self
      .0
      .wrapping_mul(6364136223846793005)
      .wrapping_add(1442695040888963407);
    (self.0 >> 11) as f64 / (1u64 << 53) as f64
  }
}

/// a random point of the convex: barycentric combination of its vertices
/// with weights k/16 (exact), some on faces (a zero weight), some on edges
fn random_point(v: &[Vec<f64>], g: &mut Lcg, zeros: usize) -> Vec<f64> {
  let nv = v.len();
  let mut w = vec![0usize; nv];
  let mut left = 16;
  // the zero weights first
  let mut free: Vec<usize> = (0..nv).collect();
  for _ in 0..zeros.min(nv - 1) {
    let k = (g.next() * free.len() as f64) as usize % free.len();
    free.remove(k);
  }
  for (j, &i) in free.iter().enumerate() {
    let wi = if j + 1 == free.len() {
      left
    } else {
      (g.next() * (left + 1) as f64) as usize % (left + 1)
    };
    w[i] = wi;
    left -= wi;
  }
  (0..v[0].len())
    .map(|k| {
      v.iter()
        .zip(&w)
        .map(|(p, &wi)| p[k] * wi as f64)
        .sum::<f64>()
        / 16.0
    })
    .collect()
}

fn check(cvr: &PRef, pts: &[Vec<f64>], label: &str) {
  let start = simplexified(cvr.structure()).unwrap();
  let t = delaunay(pts, &start).unwrap();
  let at = |s: &[usize]| -> Vec<&[f64]> { s.iter().map(|&i| pts[i].as_slice()).collect() };
  let total: f64 = t.iter().map(|s| volume(&at(s))).sum();
  let hull: f64 = start.iter().map(|s| volume(&at(s)).abs()).sum();
  assert!(
    (total - hull).abs() < 1e-13,
    "{label}: volume {total} vs {hull}"
  );
  let mut faces: BTreeMap<Vec<usize>, usize> = BTreeMap::new();
  for s in &t {
    assert_eq!(orient(&at(s)), 1, "{label}: flat or inverted {s:?}");
    for k in 0..s.len() {
      let mut f: Vec<usize> = s
        .iter()
        .enumerate()
        .filter(|(j, _)| *j != k)
        .map(|(_, &v)| v)
        .collect();
      f.sort_unstable();
      *faces.entry(f).or_default() += 1;
    }
    for (q, p) in pts.iter().enumerate() {
      assert!(
        in_sphere(&at(s), p) < 1,
        "{label}: point {q} inside the sphere of {s:?}"
      );
    }
  }
  assert!(
    faces.values().all(|&c| c <= 2),
    "{label}: a face in 3 simplices"
  );
  // every point used, except duplicates of earlier ones
  for (q, p) in pts.iter().enumerate() {
    let dup = pts[..q].contains(p);
    assert!(
      dup || t.iter().any(|s| s.contains(&q)),
      "{label}: point {q} unused"
    );
  }
}

#[test]
fn reference_convexes() {
  let mut g = Lcg(11);
  for (name, cvr) in [
    ("triangle", simplex_of_reference(2)),
    ("square", parallelepiped_of_reference(2, 1)),
    ("tetrahedron", simplex_of_reference(3)),
    ("cube", parallelepiped_of_reference(3, 1)),
    ("prism", prism_of_reference(3)),
  ] {
    let v = cvr.points().to_vec();
    for trial in 0..20 {
      let mut pts = v.clone();
      for k in 0..15 {
        // inside, on faces, on edges
        pts.push(random_point(&v, &mut g, k % 3 * (v.len() - 2) / 2));
      }
      // a duplicate vertex and a duplicate point
      pts.push(v[0].clone());
      pts.push(pts[v.len()].clone());
      check(&cvr, &pts, &format!("{name} {trial}"));
    }
    // a grid: cocircular / cospherical points everywhere
    let n = v[0].len();
    let mut pts = v.clone();
    for i in 0..=4usize.pow(n as u32) {
      let x: Vec<f64> = (0..n)
        .map(|k| ((i / 5usize.pow(k as u32)) % 5) as f64 / 4.0)
        .collect();
      // grid points of the convex (barycentric test by the volume of the
      // simplices of its start: inside the unit box, and below x+y(+z) <= 1
      // for simplices; the prism: x + y <= 1)
      let inside = match name {
        "triangle" | "tetrahedron" => x.iter().sum::<f64>() <= 1.0,
        "prism" => x[0] + x[1] <= 1.0,
        _ => true,
      };
      if inside {
        pts.push(x);
      }
    }
    check(&cvr, &pts, &format!("{name} grid"));
    // points beyond the hull, also by rounding: skipped
    let start = simplexified(cvr.structure()).unwrap();
    for (x0, used) in [(-1e-15, false), (-1e-3, false), (0.0, true)] {
      let mut out = v.clone();
      let mut p = vec![0.25; n];
      p[0] = x0;
      out.push(p);
      let t = delaunay(&out, &start).unwrap();
      assert_eq!(t.iter().any(|s| s.contains(&v.len())), used, "{name}: {x0}");
      let at = |s: &[usize]| -> Vec<&[f64]> { s.iter().map(|&i| out[i].as_slice()).collect() };
      assert!(t.iter().all(|s| orient(&at(s)) == 1));
    }
  }
}
