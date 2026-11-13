//! Spatial search: kd-tree of points and r-tree of boxes (roles of
//! bgeot_kdtree / bgeot_rtree). Trees are built lazily on the first query.

use std::sync::OnceLock;

use crate::{Error, Result};

const LEAF: usize = 8;

fn check(n: usize, v: &[f64]) -> Result<()> {
  if v.len() != n {
    return Err(Error::InvalidArgument(format!(
      "point of dim {} in a tree of dim {n}",
      v.len()
    )));
  }
  Ok(())
}

// ------------------------------------------------------------------ kd-tree

enum KdNode {
  Leaf(Vec<usize>),
  Split {
    dim: usize,
    v: f64,
    lo: Box<KdNode>,
    hi: Box<KdNode>,
  },
}

/// Points with ids; box and nearest-neighbor queries.
pub struct KdTree {
  dim: usize,
  pts: Vec<f64>,
  ids: Vec<usize>,
  root: OnceLock<KdNode>,
}

impl KdTree {
  pub fn new(dim: usize) -> Self {
    KdTree {
      dim,
      pts: vec![],
      ids: vec![],
      root: OnceLock::new(),
    }
  }

  pub fn nb_points(&self) -> usize {
    self.ids.len()
  }

  fn point(&self, i: usize) -> &[f64] {
    &self.pts[i * self.dim..(i + 1) * self.dim]
  }

  /// Add a point, id = insertion index.
  pub fn add_point(&mut self, p: &[f64]) -> Result<usize> {
    let id = self.ids.len();
    self.add_point_with_id(p, id)?;
    Ok(id)
  }

  pub fn add_point_with_id(&mut self, p: &[f64], id: usize) -> Result<()> {
    check(self.dim, p)?;
    self.pts.extend_from_slice(p);
    self.ids.push(id);
    self.root = OnceLock::new();
    Ok(())
  }

  fn build(&self, mut idx: Vec<usize>, depth: usize) -> KdNode {
    if idx.len() <= LEAF {
      idx.sort_unstable();
      return KdNode::Leaf(idx);
    }
    // split the widest dimension at the median (robust to coplanar points)
    let (mut dim, mut w) = (depth % self.dim, -1.0);
    for d in 0..self.dim {
      let (lo, hi) = idx.iter().fold((f64::MAX, f64::MIN), |(l, h), &i| {
        (l.min(self.point(i)[d]), h.max(self.point(i)[d]))
      });
      if hi - lo > w {
        (dim, w) = (d, hi - lo);
      }
    }
    if w <= 0.0 {
      idx.sort_unstable();
      return KdNode::Leaf(idx);
    }
    // the median coordinate by selection; invariant: low side < v <= high side
    let c = |i: usize| self.point(i)[dim];
    let mut keys: Vec<f64> = idx.iter().map(|&i| c(i)).collect();
    let half = keys.len() / 2;
    let mut v = *keys.select_nth_unstable_by(half, f64::total_cmp).1;
    if !idx.iter().any(|&i| c(i) < v) {
      // all at least the median: split above it, if anything is
      match idx
        .iter()
        .map(|&i| c(i))
        .filter(|&x| x > v)
        .min_by(f64::total_cmp)
      {
        Some(next) => v = next,
        None => {
          idx.sort_unstable();
          return KdNode::Leaf(idx);
        }
      }
    }
    let (lo, hi): (Vec<usize>, Vec<usize>) = idx.iter().partition(|&&i| c(i) < v);
    let idx = lo;
    KdNode::Split {
      dim,
      v,
      lo: Box::new(self.build(idx, depth + 1)),
      hi: Box::new(self.build(hi, depth + 1)),
    }
  }

  fn root(&self) -> &KdNode {
    self
      .root
      .get_or_init(|| self.build((0..self.ids.len()).collect(), 0))
  }

  /// (id, point) of all points in the closed box [lo, hi].
  pub fn points_in_box(&self, lo: &[f64], hi: &[f64]) -> Result<Vec<(usize, &[f64])>> {
    check(self.dim, lo)?;
    check(self.dim, hi)?;
    let mut out = vec![];
    let mut stack = vec![self.root()];
    while let Some(n) = stack.pop() {
      match n {
        KdNode::Leaf(ix) => {
          for &i in ix {
            let p = self.point(i);
            if p
              .iter()
              .zip(lo.iter().zip(hi))
              .all(|(x, (a, b))| x >= a && x <= b)
            {
              out.push((self.ids[i], p));
            }
          }
        }
        KdNode::Split {
          dim,
          v,
          lo: l,
          hi: h,
        } => {
          if lo[*dim] < *v {
            stack.push(l);
          }
          if hi[*dim] >= *v {
            stack.push(h);
          }
        }
      }
    }
    out.sort_by_key(|&(id, _)| id);
    Ok(out)
  }

  /// Nearest point: (id, distance), None if empty.
  pub fn nearest_neighbor(&self, p: &[f64]) -> Result<Option<(usize, f64)>> {
    check(self.dim, p)?;
    let mut best: Option<(usize, f64)> = None;
    let d2 = |q: &[f64]| q.iter().zip(p).map(|(a, b)| (a - b) * (a - b)).sum::<f64>();
    let mut stack = vec![(self.root(), 0.0f64)];
    while let Some((n, bound)) = stack.pop() {
      if best.is_some_and(|(_, b)| bound > b) {
        continue;
      }
      match n {
        KdNode::Leaf(ix) => {
          for &i in ix {
            let d = d2(self.point(i));
            if best.is_none_or(|(_, b)| d < b) {
              best = Some((i, d));
            }
          }
        }
        KdNode::Split { dim, v, lo, hi } => {
          let off = p[*dim] - v;
          let (near, far) = if off < 0.0 { (lo, hi) } else { (hi, lo) };
          stack.push((far, bound.max(off * off)));
          stack.push((near, bound));
        }
      }
    }
    Ok(best.map(|(i, d)| (self.ids[i], d.sqrt())))
  }
}

// ------------------------------------------------------------------- r-tree

enum RNode {
  Leaf(Vec<usize>),
  Inner {
    lo: Vec<f64>,
    hi: Vec<f64>,
    kids: Vec<RNode>,
  },
}

/// Boxes with ids. Comparisons use eps = rel_eps * extent of all boxes.
pub struct RTree {
  dim: usize,
  rel_eps: f64,
  lo: Vec<f64>,
  hi: Vec<f64>,
  ids: Vec<usize>,
  built: OnceLock<(RNode, f64)>,
}

fn inter(l1: &[f64], h1: &[f64], l2: &[f64], h2: &[f64], e: f64) -> bool {
  (0..l1.len()).all(|i| h1[i] >= l2[i] - e && l1[i] <= h2[i] + e)
}

/// box 1 contains box 2
fn contains(l1: &[f64], h1: &[f64], l2: &[f64], h2: &[f64], e: f64) -> bool {
  (0..l1.len()).all(|i| l1[i] <= l2[i] + e && h1[i] >= h2[i] - e)
}

impl RTree {
  pub fn new(dim: usize, rel_eps: f64) -> Self {
    RTree {
      dim,
      rel_eps,
      lo: vec![],
      hi: vec![],
      ids: vec![],
      built: OnceLock::new(),
    }
  }

  fn bx(&self, i: usize) -> (&[f64], &[f64]) {
    let n = self.dim;
    (&self.lo[i * n..(i + 1) * n], &self.hi[i * n..(i + 1) * n])
  }

  pub fn add_box(&mut self, lo: &[f64], hi: &[f64], id: usize) -> Result<()> {
    check(self.dim, lo)?;
    check(self.dim, hi)?;
    if lo.iter().zip(hi).any(|(a, b)| a > b) {
      return Err(Error::InvalidArgument(format!(
        "box min {lo:?} > max {hi:?}"
      )));
    }
    self.lo.extend_from_slice(lo);
    self.hi.extend_from_slice(hi);
    self.ids.push(id);
    self.built = OnceLock::new();
    Ok(())
  }

  fn bounds(&self, ix: &[usize]) -> (Vec<f64>, Vec<f64>) {
    let mut lo = vec![f64::MAX; self.dim];
    let mut hi = vec![f64::MIN; self.dim];
    for &i in ix {
      let (l, h) = self.bx(i);
      for k in 0..self.dim {
        lo[k] = lo[k].min(l[k]);
        hi[k] = hi[k].max(h[k]);
      }
    }
    (lo, hi)
  }

  fn build(&self, mut ix: Vec<usize>) -> RNode {
    if ix.len() <= LEAF {
      ix.sort_unstable();
      return RNode::Leaf(ix);
    }
    let (lo, hi) = self.bounds(&ix);
    let d = (0..self.dim)
      .max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b])))
      .unwrap();
    // the lower half by (centre, index): a selection, not a sort
    let mut keyed: Vec<(f64, usize)> = ix
      .iter()
      .map(|&i| {
        let (l, h) = self.bx(i);
        (l[d] + h[d], i)
      })
      .collect();
    let half = keyed.len() / 2;
    keyed.select_nth_unstable_by(half, |a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    ix = keyed[..half].iter().map(|k| k.1).collect();
    let right = keyed[half..].iter().map(|k| k.1).collect();
    RNode::Inner {
      lo,
      hi,
      kids: vec![self.build(ix), self.build(right)],
    }
  }

  fn tree(&self) -> &(RNode, f64) {
    self.built.get_or_init(|| {
      let all: Vec<usize> = (0..self.ids.len()).collect();
      let (lo, hi) = self.bounds(&all);
      let ext = lo
        .iter()
        .zip(&hi)
        .map(|(a, b)| b - a)
        .fold(0.0, f64::max)
        .max(0.0);
      (self.build(all), self.rel_eps * ext)
    })
  }

  /// Ids of boxes matching `keep`, visiting nodes passing `visit`.
  fn find(
    &self,
    visit: impl Fn(&[f64], &[f64], f64) -> bool,
    keep: impl Fn(&[f64], &[f64], f64) -> bool,
  ) -> Vec<usize> {
    let (root, e) = self.tree();
    let mut out: Vec<usize> = vec![];
    let mut stack = vec![root];
    while let Some(n) = stack.pop() {
      match n {
        RNode::Leaf(ix) => out.extend(ix.iter().filter(|&&i| {
          let (l, h) = self.bx(i);
          keep(l, h, *e)
        })),
        RNode::Inner { lo, hi, kids } => {
          if visit(lo, hi, *e) {
            stack.extend(kids.iter());
          }
        }
      }
    }
    let mut ids: Vec<usize> = out.into_iter().map(|i| self.ids[i]).collect();
    ids.sort_unstable();
    ids
  }

  pub fn find_intersecting_boxes(&self, lo: &[f64], hi: &[f64]) -> Result<Vec<usize>> {
    check(self.dim, lo)?;
    check(self.dim, hi)?;
    let f = |l: &[f64], h: &[f64], e: f64| inter(lo, hi, l, h, e);
    Ok(self.find(f, f))
  }

  /// Boxes containing [lo, hi].
  pub fn find_containing_boxes(&self, lo: &[f64], hi: &[f64]) -> Result<Vec<usize>> {
    check(self.dim, lo)?;
    check(self.dim, hi)?;
    Ok(self.find(
      |l, h, e| contains(l, h, lo, hi, e),
      |l, h, e| contains(l, h, lo, hi, e),
    ))
  }

  /// Boxes contained in [lo, hi].
  pub fn find_contained_boxes(&self, lo: &[f64], hi: &[f64]) -> Result<Vec<usize>> {
    check(self.dim, lo)?;
    check(self.dim, hi)?;
    Ok(self.find(
      |l, h, e| inter(lo, hi, l, h, e),
      |l, h, e| contains(lo, hi, l, h, e),
    ))
  }

  pub fn find_boxes_at_point(&self, p: &[f64]) -> Result<Vec<usize>> {
    self.find_containing_boxes(p, p)
  }

  /// Boxes hit by the line org + t dir (optionally restricted to boxes
  /// intersecting [lo, hi]).
  pub fn find_line_intersecting_boxes(
    &self,
    org: &[f64],
    dir: &[f64],
    clip: Option<(&[f64], &[f64])>,
  ) -> Result<Vec<usize>> {
    check(self.dim, org)?;
    check(self.dim, dir)?;
    let hit = |l: &[f64], h: &[f64], e: f64| {
      let (mut t0, mut t1) = (f64::NEG_INFINITY, f64::INFINITY);
      for k in 0..org.len() {
        if dir[k] == 0.0 {
          if org[k] < l[k] - e || org[k] > h[k] + e {
            return false;
          }
        } else {
          let (a, b) = ((l[k] - e - org[k]) / dir[k], (h[k] + e - org[k]) / dir[k]);
          t0 = t0.max(a.min(b));
          t1 = t1.min(a.max(b));
        }
      }
      t0 <= t1 && clip.is_none_or(|(cl, ch)| inter(cl, ch, l, h, e))
    };
    Ok(self.find(hit, hit))
  }
}

#[cfg(test)]
mod tests;
