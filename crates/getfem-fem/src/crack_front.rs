//! Crack fronts in 3D as points with unit tangents, and global hat
//! functions along them: a partition of unity near the front, to gather
//! the front enrichment along it (w(r) N_i F_j;
//! docs/design/m12-xfem-front.md). Each point carries the plane through
//! it normal to its tangent; between the planes of p_k and p_{k+1} the
//! position along the front is s = k + tau, tau = d_k / (d_k - d_{k+1}),
//! d_k(x) = (x - p_k) . t_k. Neighbouring hats share their planes: their
//! sum is 1 wherever the front's pieces do not overlap (within the reach
//! of the front).

use std::sync::Arc;

use getfem_geom::locate::Locator;

use crate::fem::FemCtx;
use crate::global_function::{EvalPoint, GlobalFunction, GlobalFunctionImpl};
use crate::level_set::LevelSet;
use crate::mesh_fem::MeshFem;
use crate::{Error, Result};
use getfem_linalg::{vect_dist2, vect_sp};

/// A crack front: points along it with their unit tangents, in order;
/// closed when the last point is followed by the first.
#[derive(Clone, Debug)]
pub struct CrackFront {
  pub points: Vec<Vec<f64>>,
  pub tangents: Vec<Vec<f64>>,
  pub closed: bool,
}

/// The fronts of a crack level set (primary phi, secondary psi: the front
/// is where both vanish). In 2D each tip is a front of one point, no
/// tangent (spacing None). In 3D seeded at about the given spacing: traced
/// from the elements where both change sign, by steps along
/// t = grad phi x grad psi, each projected back onto phi = psi = 0
/// (Newton, minimum-norm steps), until the trace closes or leaves the
/// mesh; then resampled by arc length, projected again, with unit
/// tangents oriented along the front. An open front ends on the mesh
/// boundary (the last step bisected).
pub fn crack_fronts(ls: &LevelSet, spacing: Option<f64>) -> Result<Vec<CrackFront>> {
  let dim = ls.mesh_fem().linked_mesh().dim();
  if !ls.has_secondary() {
    return Err(Error::InvalidArgument(
      "crack_fronts: a level set with a secondary".into(),
    ));
  }
  match (dim, spacing) {
    (2, None) => {
      return Ok(
        crack_tips(ls)?
          .into_iter()
          .map(|x| CrackFront {
            points: vec![x],
            tangents: vec![],
            closed: false,
          })
          .collect(),
      );
    }
    (3, Some(h)) if h > 0.0 => {}
    _ => {
      return Err(Error::InvalidArgument(format!(
        "crack_fronts: {dim}D and spacing {spacing:?} (2D: None, 3D: positive)"
      )));
    }
  }
  let spacing = spacing.unwrap();
  let md = ls.mesh_fem().linked_mesh().snapshot();
  let loc = Locator::new(&md, 1e-10)?;
  let ev = Evaluator { ls, md: &md, loc };
  let (cands, size) = candidates(ls, &md)?;
  // the trace's step: fine enough not to skip candidate elements
  let step = spacing.min(0.25 * size);
  let mut fronts = vec![];
  let mut covered = vec![false; cands.len()];
  let mut traced: Vec<Vec<f64>> = vec![];
  for c in 0..cands.len() {
    if covered[c] {
      continue;
    }
    let Some(start) = ev.project(&cands[c].1, step)? else {
      covered[c] = true;
      continue;
    };
    // a start off its element (both signs change there, the front does
    // not pass) or on a traced front
    let (lo, hi) = (&cands[c].2, &cands[c].3);
    let near = (0..3).all(|a| start[a] >= lo[a] - step && start[a] <= hi[a] + step);
    if !near
      || traced
        .iter()
        .any(|x: &Vec<f64>| vect_dist2(x, &start) < 2.0 * step)
    {
      covered[c] = true;
      continue;
    }
    let (mut line, closed) = ev.trace(&start, 1.0, step)?;
    if !closed {
      let (back, _) = ev.trace(&start, -1.0, step)?;
      line = back.into_iter().skip(1).rev().chain(line).collect();
    }
    // the candidates the trace went through
    for (k, (_, _, lo, hi)) in cands.iter().enumerate() {
      covered[k] |= line
        .iter()
        .any(|x| (0..3).all(|a| x[a] >= lo[a] - step && x[a] <= hi[a] + step));
    }
    traced.extend(line.iter().cloned());
    if line.len() < 2 {
      continue;
    }
    fronts.push(ev.resample(&line, closed, spacing, step)?);
  }
  Ok(fronts)
}

/// The elements where both level sets change sign on the nodes: (cv,
/// centre, lower and upper bounds), and the smallest of their diameters.
type Candidate = (usize, Vec<f64>, Vec<f64>, Vec<f64>);
fn candidates(ls: &LevelSet, md: &getfem_geom::mesh::MeshData) -> Result<(Vec<Candidate>, f64)> {
  let mut cands = vec![];
  let mut size = f64::MAX;
  for cv in md.convex_index() {
    let gt = md.trans_of_convex(cv);
    let nodes = gt.convex_ref().points().to_vec();
    let sign = |i: usize| -> Result<bool> {
      let l = ls.of_convex(cv, i)?;
      let v: Vec<f64> = nodes.iter().map(|x| l.val(x)).collect::<Result<_>>()?;
      Ok(v.iter().any(|&a| a <= 0.0) && v.iter().any(|&a| a >= 0.0))
    };
    if sign(0)? && sign(1)? {
      let g = md.points_of_convex(cv);
      let (lo, hi) = bounds(&g);
      let n = lo.len();
      size = size.min((0..n).map(|a| (hi[a] - lo[a]).powi(2)).sum::<f64>().sqrt());
      cands.push((cv, (0..n).map(|a| 0.5 * (lo[a] + hi[a])).collect(), lo, hi));
    }
  }
  Ok((cands, size))
}

/// The crack tips of a level set with a secondary in 2D (phi = psi = 0):
/// projected by Newton from the elements where both change sign.
fn crack_tips(ls: &LevelSet) -> Result<Vec<Vec<f64>>> {
  if !ls.has_secondary() || ls.mesh_fem().linked_mesh().dim() != 2 {
    return Err(Error::InvalidArgument(
      "crack_tips: a level set with a secondary in 2D".into(),
    ));
  }
  let md = ls.mesh_fem().linked_mesh().snapshot();
  let loc = Locator::new(&md, 1e-10)?;
  let ev = Evaluator { ls, md: &md, loc };
  let (cands, size) = candidates(ls, &md)?;
  let mut tips: Vec<Vec<f64>> = vec![];
  for (_, centre, lo, hi) in &cands {
    let Some(x) = ev.project(centre, size)? else {
      continue;
    };
    let on = (0..2).all(|a| x[a] >= lo[a] - 1e-8 * size && x[a] <= hi[a] + 1e-8 * size);
    if on && tips.iter().all(|t| vect_dist2(t, &x) > 1e-6 * size) {
      tips.push(x);
    }
  }
  Ok(tips)
}

/// The crack's local frame at points of its fronts or tips: e1 along the
/// crack beyond the front (grad psi normal to e2), e2 normal to the crack
/// (grad phi), in 3D e3 = e1 x e2; None outside the mesh.
pub fn crack_frames(ls: &LevelSet, points: &[Vec<f64>]) -> Result<Vec<Option<Vec<Vec<f64>>>>> {
  let md = ls.mesh_fem().linked_mesh().snapshot();
  let loc = Locator::new(&md, 1e-10)?;
  let ev = Evaluator { ls, md: &md, loc };
  points
    .iter()
    .map(|x| {
      Ok(ev.eval(x)?.map(|(_, g)| {
        let e2 = unit(g[0].clone());
        let p = vect_sp(&g[1], &e2);
        let e1 = unit(g[1].iter().zip(&e2).map(|(a, b)| a - p * b).collect());
        if e1.len() == 3 {
          let e3 = cross(&e1, &e2);
          vec![e1, e2, e3]
        } else {
          vec![e1, e2]
        }
      }))
    })
    .collect()
}

fn bounds(g: &getfem_linalg::tensor::Tensor<f64>) -> (Vec<f64>, Vec<f64>) {
  let n = g.nrows();
  let lo = (0..n)
    .map(|a| (0..g.ncols()).map(|j| g[(a, j)]).fold(f64::MAX, f64::min))
    .collect();
  let hi = (0..n)
    .map(|a| (0..g.ncols()).map(|j| g[(a, j)]).fold(f64::MIN, f64::max))
    .collect();
  (lo, hi)
}

fn cross(a: &[f64], b: &[f64]) -> Vec<f64> {
  vec![
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]
}

fn unit(a: Vec<f64>) -> Vec<f64> {
  let n = vect_sp(&a, &a).sqrt();
  a.into_iter().map(|v| v / n).collect()
}

/// phi, psi and their real gradients
type Values = ([f64; 2], [Vec<f64>; 2]);

struct Evaluator<'a> {
  ls: &'a LevelSet,
  md: &'a getfem_geom::mesh::MeshData,
  loc: Locator<'a>,
}

impl Evaluator<'_> {
  /// phi, psi and their real gradients at x (None outside the mesh)
  fn eval(&self, x: &[f64]) -> Result<Option<Values>> {
    let Some((cv, xref)) = self.loc.find_first(x, |_| true)? else {
      return Ok(None);
    };
    let (gt, g) = (self.md.trans_of_convex(cv), self.md.points_of_convex(cv));
    let pg = gt.at(&g, &xref)?;
    let c = FemCtx::new(gt, &g, &xref, &pg);
    let (n, p) = (g.nrows(), xref.len());
    let mut v = [0.0; 2];
    let mut gr = [vec![], vec![]];
    for i in 0..2 {
      let l = self.ls.of_convex(cv, i)?;
      v[i] = l.val(&xref)?;
      let r = l.grad(&xref)?;
      gr[i] = (0..n)
        .map(|a| (0..p).map(|j| c.b[a + n * j] * r[j]).sum())
        .collect();
    }
    Ok(Some((v, gr)))
  }
  /// x moved onto phi = psi = 0 by Newton with minimum-norm steps (None:
  /// out of the mesh, or no convergence)
  fn project(&self, x0: &[f64], step: f64) -> Result<Option<Vec<f64>>> {
    let mut x = x0.to_vec();
    for _ in 0..30 {
      let Some((v, g)) = self.eval(&x)? else {
        return Ok(None);
      };
      let (a, b, c) = (
        vect_sp(&g[0], &g[0]),
        vect_sp(&g[0], &g[1]),
        vect_sp(&g[1], &g[1]),
      );
      let det = a * c - b * b;
      if det.is_nan() || det <= 0.0 {
        return Ok(None);
      }
      // (J J^T) l = v, dx = -J^T l
      let l0 = (c * v[0] - b * v[1]) / det;
      let l1 = (a * v[1] - b * v[0]) / det;
      let dx: Vec<f64> = (0..x.len())
        .map(|k| -(l0 * g[0][k] + l1 * g[1][k]))
        .collect();
      x.iter_mut().zip(&dx).for_each(|(p, d)| *p += d);
      if vect_sp(&dx, &dx).sqrt() < 1e-12 * step {
        return Ok(self.eval(&x)?.map(|_| x));
      }
    }
    Ok(None)
  }
  /// the unit tangent grad phi x grad psi at a point of the front
  fn tangent(&self, x: &[f64]) -> Result<Option<Vec<f64>>> {
    Ok(self.eval(x)?.map(|(_, g)| unit(cross(&g[0], &g[1]))))
  }
  /// steps from start along sign * tangent: the points, and whether the
  /// trace came back to start
  fn trace(&self, start: &[f64], sign: f64, step: f64) -> Result<(Vec<Vec<f64>>, bool)> {
    let mut line = vec![start.to_vec()];
    for _ in 0..1_000_000 {
      let last = line.last().unwrap();
      let Some(t) = self.tangent(last)? else { break };
      let pred: Vec<f64> = (0..3).map(|k| last[k] + sign * step * t[k]).collect();
      let Some(x) = self.project(&pred, step)? else {
        // the end: bisect the step onto the mesh boundary
        let (mut lo, mut hi, mut end) = (0.0, step, None);
        for _ in 0..40 {
          let a = 0.5 * (lo + hi);
          let p: Vec<f64> = (0..3).map(|k| last[k] + sign * a * t[k]).collect();
          match self.project(&p, step)? {
            Some(x) => (lo, end) = (a, Some(x)),
            None => hi = a,
          }
        }
        if let Some(x) = end.filter(|x| vect_dist2(x, last) > 1e-3 * step) {
          line.push(x);
        }
        break;
      };
      if line.len() > 3 && vect_dist2(&x, start) < step {
        return Ok((line, true));
      }
      line.push(x);
    }
    Ok((line, false))
  }
  /// the polyline resampled by arc length at about spacing, projected,
  /// with tangents oriented along it
  fn resample(
    &self,
    line: &[Vec<f64>],
    closed: bool,
    spacing: f64,
    step: f64,
  ) -> Result<CrackFront> {
    let mut pts = line.to_vec();
    if closed {
      pts.push(line[0].clone());
    }
    let mut acc = vec![0.0];
    for w in pts.windows(2) {
      acc.push(acc.last().unwrap() + vect_dist2(&w[0], &w[1]));
    }
    let len = *acc.last().unwrap();
    let m = ((len / spacing).round() as usize).max(if closed { 3 } else { 1 });
    let np = if closed { m } else { m + 1 };
    let mut points = vec![];
    for k in 0..np {
      let s = len * k as f64 / m as f64;
      let j = acc.partition_point(|&a| a <= s).clamp(1, acc.len() - 1);
      let t = if acc[j] > acc[j - 1] {
        (s - acc[j - 1]) / (acc[j] - acc[j - 1])
      } else {
        0.0
      };
      let x: Vec<f64> = (0..3)
        .map(|a| pts[j - 1][a] + t * (pts[j][a] - pts[j - 1][a]))
        .collect();
      points.push(self.project(&x, step)?.unwrap_or(x));
    }
    let n = points.len();
    let mut tangents = vec![];
    for k in 0..n {
      let mut t = self.tangent(&points[k])?.unwrap_or_else(|| vec![0.0; 3]);
      // along the chain: from the previous to the next point
      let (a, b) = if closed {
        ((k + n - 1) % n, (k + 1) % n)
      } else {
        (k.saturating_sub(1), (k + 1).min(n - 1))
      };
      let d: Vec<f64> = (0..3).map(|i| points[b][i] - points[a][i]).collect();
      if vect_sp(&t, &d) < 0.0 {
        t.iter_mut().for_each(|v| *v = -*v);
      }
      tangents.push(t);
    }
    Ok(CrackFront {
      points,
      tangents,
      closed,
    })
  }
}

/// The basic DOFs of mf on the elements the front of the level set i of
/// mls passes through (its crack tip convexes in the cut): the front
/// elements, whose Lagrange DOFs carry the crack tip functions
/// (docs/design/m12-xfem-front.md).
pub fn front_element_dofs(
  mls: &crate::mesh_level_set::MeshLevelSet,
  i: usize,
  mf: &MeshFem,
) -> Result<std::collections::BTreeSet<usize>> {
  let mut dofs = std::collections::BTreeSet::new();
  for &cv in mls.crack_tip_convexes_of(i)? {
    if mf.fem_of_element(cv)?.is_some() {
      dofs.extend(mf.ind_basic_dof_of_element(cv)?);
    }
  }
  Ok(dofs)
}

/// The basic DOFs of mf whose node is within radius of the front of the
/// crack level set ls: primary^2 + secondary^2 < radius^2, the level set's
/// interpolant at the node; the distance to the front where both are
/// distances near it.
pub fn select_mesh_fem_basic_dofs_near_front(
  ls: &LevelSet,
  mf: &MeshFem,
  radius: f64,
) -> Result<std::collections::BTreeSet<usize>> {
  if !ls.has_secondary() || radius.is_nan() || radius <= 0.0 {
    return Err(Error::InvalidArgument(format!(
      "select_mesh_fem_basic_dofs_near_front: a crack level set (with a secondary) and a positive radius ({radius})"
    )));
  }
  let pts: Vec<Vec<f64>> = (0..mf.nb_basic_dof()?)
    .map(|d| mf.point_of_basic_dof(d))
    .collect::<Result<_>>()?;
  let lmf = ls.mesh_fem();
  let v0 = crate::interpolation::interpolation_at_points(&lmf, &pts, &ls.values(0)?)?;
  let v1 = crate::interpolation::interpolation_at_points(&lmf, &pts, &ls.values(1)?)?;
  Ok(
    (0..pts.len())
      .filter(|&d| v0[d] * v0[d] + v1[d] * v1[d] < radius * radius)
      .collect(),
  )
}

/// The hats along a front.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HatDegree {
  /// from the points i-1, i, i+1
  Linear,
  /// uniform cubic B-splines of the position along the front, from the
  /// points i-2 .. i+2; at open ends the B-splines beyond the ends added
  /// to the end hats
  Cubic,
}

/// The hats of a front, one per point, each zero farther than radius from
/// the front's pieces it spans (and from its end point beyond an open
/// end).
pub fn front_hats(
  front: &CrackFront,
  degree: HatDegree,
  radius: f64,
) -> Result<Vec<GlobalFunction>> {
  let n = front.points.len();
  let min = match (front.closed, degree) {
    (false, _) => 2,
    (true, HatDegree::Linear) => 3,
    (true, HatDegree::Cubic) => 5,
  };
  if n < min || front.tangents.len() != n {
    return Err(Error::InvalidArgument(format!(
      "front_hats: {n} points, {} tangents ({min} points at least)",
      front.tangents.len()
    )));
  }
  if radius.is_nan() || radius <= 0.0 {
    return Err(Error::InvalidArgument(format!(
      "front_hats: radius {radius}"
    )));
  }
  for (p, t) in front.points.iter().zip(&front.tangents) {
    let nt: f64 = t.iter().map(|v| v * v).sum::<f64>().sqrt();
    if p.len() != 3 || t.len() != 3 || (nt - 1.0).abs() > 1e-8 {
      return Err(Error::InvalidArgument(
        "front_hats: points and unit tangents in 3D expected".into(),
      ));
    }
  }
  let front = Arc::new(Front {
    f: front.clone(),
    degree,
    radius,
  });
  Ok(
    (0..n)
      .map(|i| {
        GlobalFunction::new(Hat {
          front: front.clone(),
          i,
        })
      })
      .collect(),
  )
}

struct Front {
  f: CrackFront,
  degree: HatDegree,
  radius: f64,
}

/// The position s along the front at x with its gradient and Hessian
/// (column-major), when x is on a piece or beyond an open end.
struct Position {
  s: f64,
  g: [f64; 3],
  h: [f64; 9],
}

impl Front {
  fn n(&self) -> usize {
    self.f.points.len()
  }
  /// number of pieces between the points
  fn pieces(&self) -> usize {
    if self.f.closed {
      self.n()
    } else {
      self.n() - 1
    }
  }
  fn d(&self, k: usize, x: &[f64]) -> f64 {
    let p = &self.f.points[k];
    (0..3).map(|a| (x[a] - p[a]) * self.f.tangents[k][a]).sum()
  }
  /// distance from x to the segment p_a p_b
  fn segment_distance(&self, a: usize, b: usize, x: &[f64]) -> f64 {
    let (p, q) = (&self.f.points[a], &self.f.points[b]);
    let e: Vec<f64> = (0..3).map(|i| q[i] - p[i]).collect();
    let w: Vec<f64> = (0..3).map(|i| x[i] - p[i]).collect();
    let l2 = vect_sp(&e, &e);
    let t = if l2 > 0.0 {
      (vect_sp(&w, &e) / l2).clamp(0.0, 1.0)
    } else {
      0.0
    };
    (0..3)
      .map(|i| (w[i] - t * e[i]).powi(2))
      .sum::<f64>()
      .sqrt()
  }
  /// s on piece k (between points k and k+1) if x is on it
  fn on_piece(&self, k: usize, x: &[f64]) -> Option<Position> {
    let k1 = (k + 1) % self.n();
    let (a, b) = (self.d(k, x), self.d(k1, x));
    if !(a >= 0.0 && b <= 0.0 && a - b > 0.0) || self.segment_distance(k, k1, x) > self.radius {
      return None;
    }
    let dd = a - b;
    let (u, v) = (&self.f.tangents[k], &self.f.tangents[k1]);
    // tau = a / (a - b), grad tau = (a v - b u) / D^2,
    // hess tau = (v u' - u v') / D^2 - 2 (a v - b u)(u - v)' / D^3
    let gv: Vec<f64> = (0..3).map(|i| a * v[i] - b * u[i]).collect();
    let mut g = [0.0; 3];
    let mut h = [0.0; 9];
    for i in 0..3 {
      g[i] = gv[i] / (dd * dd);
      for j in 0..3 {
        h[i + 3 * j] =
          (v[i] * u[j] - u[i] * v[j]) / (dd * dd) - 2.0 * gv[i] * (u[j] - v[j]) / (dd * dd * dd);
      }
    }
    Some(Position {
      s: k as f64 + a / dd,
      g,
      h,
    })
  }
  /// s beyond an open end (constant): 0 before the first plane, the last
  /// piece's end after the last one, within radius of the end point
  fn beyond_end(&self, x: &[f64]) -> Option<Position> {
    if self.f.closed {
      return None;
    }
    let last = self.n() - 1;
    let near = |k: usize| self.segment_distance(k, k, x) <= self.radius;
    let s = if self.d(0, x) < 0.0 && near(0) {
      0.0
    } else if self.d(last, x) > 0.0 && near(last) {
      last as f64
    } else {
      return None;
    };
    Some(Position {
      s,
      g: [0.0; 3],
      h: [0.0; 9],
    })
  }
  /// the pieces whose position can give hat i a value (distinct: closed
  /// fronts have at least 3 points, 5 with cubic hats)
  fn pieces_of(&self, i: usize) -> impl Iterator<Item = usize> + '_ {
    let w: i64 = match self.degree {
      HatDegree::Linear => 1,
      HatDegree::Cubic => 2,
    };
    let (n, m) = (self.n() as i64, self.pieces() as i64);
    (i as i64 - w..i as i64 + w).filter_map(move |k| {
      if self.f.closed {
        Some(k.rem_euclid(n) as usize)
      } else {
        (0..m).contains(&k).then_some(k as usize)
      }
    })
  }
  fn position(&self, i: usize, x: &[f64]) -> Option<Position> {
    self
      .pieces_of(i)
      .find_map(|k| self.on_piece(k, x))
      .or_else(|| self.beyond_end(x))
  }
  /// the 1D basis B(u) and its two derivatives
  fn basis(&self, u: f64) -> [f64; 3] {
    let a = u.abs();
    let sg = u.signum();
    match self.degree {
      HatDegree::Linear if a < 1.0 => [1.0 - a, -sg, 0.0],
      HatDegree::Cubic if a < 1.0 => [
        2.0 / 3.0 - a * a + 0.5 * a * a * a,
        sg * (-2.0 * a + 1.5 * a * a),
        -2.0 + 3.0 * a,
      ],
      HatDegree::Cubic if a < 2.0 => [
        (2.0 - a).powi(3) / 6.0,
        -sg * 0.5 * (2.0 - a).powi(2),
        2.0 - a,
      ],
      _ => [0.0; 3],
    }
  }
  /// the value of hat i at s with its first and second derivatives in s:
  /// B(s - i) (closed: the nearest copy), at open ends the B-splines of
  /// the centres beyond the end added
  fn in_s(&self, i: usize, s: f64) -> [f64; 3] {
    let mut u = s - i as f64;
    if self.f.closed {
      let n = self.n() as f64;
      u = (u + 0.5 * n).rem_euclid(n) - 0.5 * n;
      return self.basis(u);
    }
    let last = self.n() - 1;
    let mut r = self.basis(u);
    let extra: &[f64] = if i == 0 {
      &[1.0, 2.0]
    } else if i == last {
      &[-1.0, -2.0]
    } else {
      &[]
    };
    for e in extra {
      let b = self.basis(u + e);
      (0..3).for_each(|k| r[k] += b[k]);
    }
    r
  }
}

struct Hat {
  front: Arc<Front>,
  i: usize,
}

impl GlobalFunctionImpl for Hat {
  fn dim(&self) -> usize {
    3
  }
  fn val(&self, p: &EvalPoint) -> f64 {
    self
      .front
      .position(self.i, p.x)
      .map_or(0.0, |q| self.front.in_s(self.i, q.s)[0])
  }
  fn grad(&self, p: &EvalPoint) -> Vec<f64> {
    match self.front.position(self.i, p.x) {
      Some(q) => {
        let b = self.front.in_s(self.i, q.s);
        q.g.iter().map(|g| b[1] * g).collect()
      }
      None => vec![0.0; 3],
    }
  }
  fn hess(&self, p: &EvalPoint) -> Vec<f64> {
    match self.front.position(self.i, p.x) {
      Some(q) => {
        let b = self.front.in_s(self.i, q.s);
        let mut h = vec![0.0; 9];
        for j in 0..3 {
          for r in 0..3 {
            h[r + 3 * j] = b[2] * q.g[r] * q.g[j] + b[1] * q.h[r + 3 * j];
          }
        }
        h
      }
      None => vec![0.0; 9],
    }
  }
  fn is_in_support(&self, x: &[f64]) -> bool {
    self.front.position(self.i, x).is_some()
  }
  /// the points of its pieces, widened by the radius
  fn bounding_box(&self) -> (Vec<f64>, Vec<f64>) {
    let f = &self.front;
    let mut ks: Vec<usize> = f.pieces_of(self.i).collect();
    ks.extend(ks.clone().into_iter().map(|k| (k + 1) % f.n()));
    ks.push(self.i);
    let (mut lo, mut hi) = (vec![f64::MAX; 3], vec![f64::MIN; 3]);
    for k in ks {
      for a in 0..3 {
        lo[a] = lo[a].min(f.f.points[k][a] - f.radius);
        hi[a] = hi[a].max(f.f.points[k][a] + f.radius);
      }
    }
    (lo, hi)
  }
}
