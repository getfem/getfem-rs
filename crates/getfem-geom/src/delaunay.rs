//! Delaunay triangulation of small point sets (the cut elements of level
//! sets; GetFEM uses qhull): Bowyer-Watson insertion with the exact
//! predicates, into a Delaunay triangulation of the convex hull, such as
//! a simplexified reference convex (its vertices are cospherical).
//! Incremental: points can be added to a built triangulation.

use std::collections::HashMap;

use crate::predicates::{in_sphere_oriented, orient};
use crate::{Error, Result};

const NONE: usize = usize::MAX;

/// The simplices (point indices, positively oriented) of a Delaunay
/// triangulation of pts, the points not used by `start` inserted in
/// order. `start` triangulates the convex hull of pts and is Delaunay.
/// A point exactly on a circumsphere does not conflict with it; a point
/// on the hull splits its faces; points equal to a vertex or beyond the
/// hull (also by rounding: put points on the hull exactly on it) are
/// skipped.
pub fn delaunay(pts: &[Vec<f64>], start: &[Vec<usize>]) -> Result<Vec<Vec<usize>>> {
  let mut t = Delaunay::new(pts.to_vec(), start)?;
  for q in 0..pts.len() {
    t.insert(q);
  }
  Ok(t.simplices())
}

/// A Delaunay triangulation under construction: its points, the simplices
/// (dead ones kept, marked), their neighbours across each face (the face
/// opposite to vertex k) and circumspheres.
pub struct Delaunay {
  pts: Vec<Vec<f64>>,
  n: usize,
  simp: Vec<[usize; 4]>,
  nb: Vec<[usize; 4]>,
  ball: Vec<Option<Ball>>,
  alive: Vec<bool>,
  used: Vec<bool>,
  /// the inserted points by their exact coordinates
  at: HashMap<Vec<u64>, usize>,
  last: usize,
}

impl Delaunay {
  /// The triangulation `start` of the convex hull of pts (Delaunay).
  pub fn new(pts: Vec<Vec<f64>>, start: &[Vec<usize>]) -> Result<Self> {
    let n = pts.first().map_or(0, |p| p.len());
    if !(1..=3).contains(&n) {
      return Err(Error::InvalidArgument("delaunay: dimension 1 to 3".into()));
    }
    let mut t = Delaunay {
      used: vec![false; pts.len()],
      pts,
      n,
      simp: vec![],
      nb: vec![],
      ball: vec![],
      alive: vec![],
      at: HashMap::new(),
      last: 0,
    };
    let mut faces: HashMap<[usize; 3], (usize, usize)> = HashMap::new();
    for s in start {
      let mut a = [NONE; 4];
      a[..=n].copy_from_slice(s);
      match orient(&t.coords(&a)) {
        0 => {
          return Err(Error::InvalidArgument(
            "delaunay: a flat start simplex".into(),
          ));
        }
        -1 => a.swap(0, 1),
        _ => {}
      }
      let c = t.push(a);
      for k in 0..=n {
        let f = t.face(c, k);
        if let Some((o, j)) = faces.remove(&f) {
          t.nb[c][k] = o;
          t.nb[o][j] = c;
        } else {
          faces.insert(f, (c, k));
        }
      }
      for &v in s {
        t.mark_used(v);
      }
    }
    Ok(t)
  }

  /// Adds a point (not inserted yet), its index.
  pub fn add_point(&mut self, p: Vec<f64>) -> usize {
    self.pts.push(p);
    self.used.push(false);
    self.pts.len() - 1
  }

  pub fn points(&self) -> &[Vec<f64>] {
    &self.pts
  }

  /// The live simplices.
  pub fn simplices(&self) -> Vec<Vec<usize>> {
    (0..self.simp.len())
      .filter(|&c| self.alive[c])
      .map(|c| self.simp[c][..=self.n].to_vec())
      .collect()
  }

  /// Inserts point q (Bowyer-Watson); false if skipped (a duplicate, in
  /// no circumsphere, beyond the hull).
  pub fn insert(&mut self, q: usize) -> bool {
    if self.used[q] || self.at.contains_key(&key(&self.pts[q])) {
      return false;
    }
    let Some(seed) = self.locate(q) else {
      return false;
    };
    let n = self.n;
    // the conflict region, connected through the faces
    let mut cavity = vec![seed];
    let mut mark: HashMap<usize, bool> = HashMap::from([(seed, true)]);
    let mut i = 0;
    while i < cavity.len() {
      let c = cavity[i];
      i += 1;
      for k in 0..=n {
        let o = self.nb[c][k];
        if o != NONE && !mark.contains_key(&o) {
          let inside = self.conflict(o, q);
          mark.insert(o, inside);
          if inside {
            cavity.push(o);
          }
        }
      }
    }
    // the boundary faces: the vertex opposite to each replaced by q,
    // positive if q is on its side; flat: q on the hull face; negative: q
    // beyond the hull, skipped
    let mut new = vec![];
    for &c in &cavity {
      for k in 0..=n {
        let o = self.nb[c][k];
        if o != NONE && mark[&o] {
          continue;
        }
        let mut s = self.simp[c];
        s[k] = q;
        match orient(&self.coords(&s)) {
          1 => new.push((s, k, o, c)),
          0 => {}
          _ => return false,
        }
      }
    }
    for &c in &cavity {
      self.alive[c] = false;
    }
    let mut faces: HashMap<[usize; 3], (usize, usize)> = HashMap::new();
    for (s, k, o, c) in new {
      let d = self.push(s);
      self.nb[d][k] = o;
      if o != NONE {
        let j = (0..=n).find(|&j| self.nb[o][j] == c).unwrap();
        self.nb[o][j] = d;
      }
      // the faces through q pair the new simplices
      for j in (0..=n).filter(|&j| j != k) {
        let f = self.face(d, j);
        if let Some((e, l)) = faces.remove(&f) {
          self.nb[d][j] = e;
          self.nb[e][l] = d;
        } else {
          faces.insert(f, (d, j));
        }
      }
    }
    self.mark_used(q);
    true
  }

  fn push(&mut self, s: [usize; 4]) -> usize {
    let b = ball(&self.coords(&s));
    self.simp.push(s);
    self.nb.push([NONE; 4]);
    self.ball.push(b);
    self.alive.push(true);
    self.last = self.simp.len() - 1;
    self.last
  }

  fn mark_used(&mut self, v: usize) {
    self.used[v] = true;
    self.at.insert(key(&self.pts[v]), v);
  }

  fn coords(&self, s: &[usize; 4]) -> Vec<&[f64]> {
    s[..=self.n]
      .iter()
      .map(|&i| self.pts[i].as_slice())
      .collect()
  }

  /// the sorted vertices of the face opposite to vertex k
  fn face(&self, c: usize, k: usize) -> [usize; 3] {
    let mut f = [NONE; 3];
    let mut j = 0;
    for (i, &v) in self.simp[c][..=self.n].iter().enumerate() {
      if i != k {
        f[j] = v;
        j += 1;
      }
    }
    f[..j].sort_unstable();
    f
  }

  /// q strictly inside the circumsphere of simplex c (the cheap test first)
  fn conflict(&self, c: usize, q: usize) -> bool {
    let x = &self.pts[q];
    match self.ball[c].as_ref().and_then(|b| b.test(x)) {
      Some(inside) => inside,
      None => in_sphere_oriented(&self.coords(&self.simp[c]), x) == 1,
    }
  }

  /// A simplex in conflict with q: walk from the last simplex towards q
  /// (the face q is beyond), else a scan.
  fn locate(&self, q: usize) -> Option<usize> {
    let n = self.n;
    let x = self.pts[q].as_slice();
    let mut c = self.last;
    let mut steps = 0;
    if self.alive.get(c) == Some(&true) {
      'walk: while steps < 4 * self.simp.len() {
        steps += 1;
        // start from a varying face: no cycles on degenerate configurations
        for i in 0..=n {
          let k = (i + steps) % (n + 1);
          let o = self.nb[c][k];
          if o == NONE {
            continue;
          }
          let mut p = self.coords(&self.simp[c]);
          p[k] = x;
          if orient(&p) < 0 {
            c = o;
            continue 'walk;
          }
        }
        if self.conflict(c, q) {
          return Some(c);
        }
        break;
      }
    }
    (0..self.simp.len()).find(|&c| self.alive[c] && self.conflict(c, q))
  }
}

fn key(p: &[f64]) -> Vec<u64> {
  // +0 and -0 alike
  p.iter().map(|v| (v + 0.0).to_bits()).collect()
}

/// A circumsphere in floating point
struct Ball {
  c: [f64; 3],
  r2: f64,
}

impl Ball {
  /// clearly inside (Some(true)), clearly outside (Some(false)), or
  /// near the sphere (None: the exact test decides)
  fn test(&self, q: &[f64]) -> Option<bool> {
    let d2: f64 = q.iter().zip(&self.c).map(|(a, b)| (a - b) * (a - b)).sum();
    if d2 > self.r2 * (1.0 + 1e-6) {
      Some(false)
    } else if d2 < self.r2 * (1.0 - 1e-6) {
      Some(true)
    } else {
      None
    }
  }
}

/// the circumsphere of a simplex (n <= 3); None for flat ones, where its
/// relative error (about eps / relative volume^2) could exceed the margin
/// of Ball::test
fn ball(p: &[&[f64]]) -> Option<Ball> {
  let n = p.len() - 1;
  if relative_volume(p).abs() < 1e-3 {
    return None;
  }
  // 2 (p_i - p_0).x = |p_i - p_0|^2, by Cramer's rule
  let mut a = [[0.0f64; 3]; 3];
  let mut b = [0.0f64; 3];
  for i in 0..n {
    for k in 0..n {
      a[i][k] = 2.0 * (p[i + 1][k] - p[0][k]);
      b[i] += (p[i + 1][k] - p[0][k]).powi(2);
    }
  }
  let d = det_rows(&a, n);
  let mut c = [0.0f64; 3];
  for k in 0..n {
    let mut m = a;
    for i in 0..n {
      m[i][k] = b[i];
    }
    c[k] = p[0][k] + det_rows(&m, n) / d;
  }
  let r2 = (0..n).map(|k| (p[0][k] - c[k]).powi(2)).sum();
  Some(Ball { c, r2 })
}

/// det[p1 - p0, ..., pn - p0] / L^n, L the largest edge (floating point,
/// n <= 3)
fn relative_volume(p: &[&[f64]]) -> f64 {
  let n = p.len() - 1;
  let mut l = 0.0f64;
  for a in p {
    for b in p {
      l = l.max(
        a.iter()
          .zip(*b)
          .map(|(x, y)| (x - y) * (x - y))
          .sum::<f64>()
          .sqrt(),
      );
    }
  }
  let mut m = [[0.0f64; 3]; 3];
  for i in 0..n {
    for k in 0..n {
      m[i][k] = p[i + 1][k] - p[0][k];
    }
  }
  det_rows(&m, n) / l.powi(n as i32)
}

/// det of the leading n x n block (floating point, n <= 3)
fn det_rows(m: &[[f64; 3]; 3], n: usize) -> f64 {
  match n {
    1 => m[0][0],
    2 => m[0][0] * m[1][1] - m[0][1] * m[1][0],
    _ => {
      m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }
  }
}
