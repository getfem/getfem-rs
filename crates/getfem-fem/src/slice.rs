//! Mesh slices (port of getfem_mesh_slicers.cc, getfem_mesh_slice.cc):
//! each convex is refined into simplices, then slicer actions cut, select
//! or move them; a StoredMeshSlice keeps the result for export.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use getfem_geom::convex_ref::PRef;
use getfem_geom::geotrans::PGeoTrans;
use getfem_geom::mesh::MeshData;
use getfem_geom::region::RegionSet;
use getfem_linalg::tensor::Mat;

use crate::fem::real_base_values;
use crate::fem_composite::refined_simplex_mesh_for_convex;
use crate::mesh_fem::{DofNumbering, MeshFem};
use crate::mesh_level_set::MeshLevelSet;
use crate::{Error, Result};
use getfem_linalg::{vect_dist2_sqr, vect_sp};

/// GetFEM slicer_action::EPS
pub const EPS: f64 = 1e-13_f32 as f64;

/// A node: real and reference position, faces (of the convex, then of the
/// slices) it lies on.
#[derive(Clone, Debug, PartialEq)]
pub struct SliceNode {
  pub pt: Vec<f64>,
  pub pt_ref: Vec<f64>,
  pub faces: u32,
}

/// A simplex: node indices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SliceSimplex {
  pub inodes: Vec<usize>,
}

impl SliceSimplex {
  pub fn dim(&self) -> usize {
    self.inodes.len() - 1
  }
}

/// The convex being sliced (GetFEM mesh_slicer state).
pub struct SlicerState<'m> {
  pub m: &'m MeshData,
  pub cv: usize,
  pub face: Option<usize>,
  pub cvr: PRef,
  pub nodes: Vec<SliceNode>,
  pub simplexes: Vec<SliceSimplex>,
  pub simplex_index: BTreeSet<usize>,
  pub nodes_index: BTreeSet<usize>,
  /// simplexes in the slice
  pub splx_in: BTreeSet<usize>,
  /// next free face bit (slice faces after the convex faces)
  pub fcnt: usize,
}

impl SlicerState<'_> {
  pub fn add_simplex(&mut self, s: SliceSimplex, isin: bool) -> usize {
    let i = self.simplexes.len();
    self.simplexes.push(s);
    if isin {
      self.splx_in.insert(i);
    }
    self.simplex_index.insert(i);
    i
  }

  pub fn sup_simplex(&mut self, i: usize) {
    self.splx_in.remove(&i);
    self.simplex_index.remove(&i);
  }

  pub fn update_nodes_index(&mut self) {
    self.nodes_index = self
      .simplex_index
      .iter()
      .flat_map(|&j| self.simplexes[j].inodes.iter().copied())
      .collect();
  }

  pub fn dim(&self) -> usize {
    self.nodes.first().map_or(self.m.dim(), |n| n.pt.len())
  }
}

/// A slicing operation.
pub trait SlicerAction {
  fn exec(&mut self, ms: &mut SlicerState) -> Result<()>;
}

impl<T: SlicerAction + ?Sized> SlicerAction for Box<T> {
  fn exec(&mut self, ms: &mut SlicerState) -> Result<()> {
    (**self).exec(ms)
  }
}

/// Does nothing (the plain refined mesh).
pub struct SlicerNone;

impl SlicerAction for SlicerNone {
  fn exec(&mut self, _ms: &mut SlicerState) -> Result<()> {
    Ok(())
  }
}

/// A field on a MeshFem, for isovalues, deformations and exports (GetFEM
/// mesh_slice_cv_dof_data).
#[derive(Clone)]
pub struct SliceField {
  num: Arc<DofNumbering>,
  u: Vec<f64>,
  qdim: usize,
}

impl SliceField {
  /// u: the DOF values of mf (nb_dof)
  pub fn new(mf: &MeshFem, u: &[f64]) -> Result<Self> {
    let n = mf.nb_dof()?;
    if u.len() != n {
      return Err(Error::InvalidArgument(format!(
        "slice field: {} values for {n} DOFs",
        u.len()
      )));
    }
    Ok(SliceField {
      num: mf.numbering()?,
      u: mf.extend_vector(u)?,
      qdim: mf.qdim(),
    })
  }

  pub fn qdim(&self) -> usize {
    self.qdim
  }

  fn maxval(&self) -> f64 {
    self.u.iter().fold(0.0f64, |a, x| a.max(x.abs()))
  }

  /// values (qdim) at reference points of cv; None if no element there
  fn at(&self, m: &MeshData, cv: usize, xsref: &[&[f64]]) -> Result<Option<Vec<Vec<f64>>>> {
    let Some(pf) = self.num.fem_of_element(cv) else {
      return Ok(None);
    };
    let (gt, g) = (m.trans_of_convex(cv), m.points_of_convex(cv));
    let dofs = self.num.ind_basic_dof_of_element(cv);
    let t = pf.target_dim();
    let qm = self.qdim / t;
    let mut out = vec![];
    for xref in xsref {
      let (val, _, _) = real_base_values(pf, gt, &g, xref, 0)?;
      let nd = val.len() / t;
      let mut v = vec![0.0; self.qdim];
      for i in 0..nd {
        for c in 0..qm {
          let coef = self.u[dofs[i * qm + c]];
          for a in 0..t {
            v[c * t + a] += coef * val[i + nd * a];
          }
        }
      }
      out.push(v);
    }
    Ok(Some(out))
  }
}

/// What a volume slicer keeps (GetFEM slicer_volume orient).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Orient {
  /// the inside of the volume
  In,
  /// its boundary
  Bound,
  /// its outside
  Out,
  /// everything, split so that no simplex crosses the boundary
  Split,
}

enum Volume {
  HalfSpace {
    x0: Vec<f64>,
    n: Vec<f64>,
  },
  Sphere {
    x0: Vec<f64>,
    r: f64,
  },
  Cylinder {
    x0: Vec<f64>,
    d: Vec<f64>,
    r: f64,
  },
  Isovalues {
    f: SliceField,
    val: f64,
    scaling: f64,
  },
}

/// Slices by a half-space, sphere, cylinder or isovalue of a field
/// (GetFEM slicer_half_space, slicer_sphere, slicer_cylinder,
/// slicer_isovalues).
pub struct SlicerVolume {
  orient: Orient,
  kind: Volume,
  pt_in: BTreeSet<usize>,
  pt_bin: BTreeSet<usize>,
  uval: Vec<f64>,
}

impl SlicerVolume {
  fn new(orient: Orient, kind: Volume) -> Self {
    SlicerVolume {
      orient,
      kind,
      pt_in: BTreeSet::new(),
      pt_bin: BTreeSet::new(),
      uval: vec![],
    }
  }

  /// the half-space (x - x0).n <= 0 (n: from inside toward outside)
  pub fn half_space(x0: &[f64], n: &[f64], orient: Orient) -> Self {
    let l = vect_sp(n, n).sqrt();
    Self::new(
      orient,
      Volume::HalfSpace {
        x0: x0.to_vec(),
        n: n.iter().map(|v| v / l).collect(),
      },
    )
  }

  pub fn sphere(x0: &[f64], r: f64, orient: Orient) -> Self {
    Self::new(orient, Volume::Sphere { x0: x0.to_vec(), r })
  }

  /// the cylinder of axis x0 x1 and radius r (in 2D, z = 0 is added)
  pub fn cylinder(x0: &[f64], x1: &[f64], r: f64, orient: Orient) -> Self {
    let pad = |v: &[f64]| {
      let mut v = v.to_vec();
      if v.len() == 2 {
        v.push(0.0);
      }
      v
    };
    let (a, b) = (pad(x0), pad(x1));
    let mut d: Vec<f64> = b.iter().zip(&a).map(|(p, q)| p - q).collect();
    let l = vect_sp(&d, &d).sqrt();
    d.iter_mut().for_each(|v| *v /= l);
    Self::new(orient, Volume::Cylinder { x0: a, d, r })
  }

  /// the level set f = val (inside: f < val)
  pub fn isovalues(f: SliceField, val: f64, orient: Orient) -> Result<Self> {
    if f.qdim != 1 {
      return Err(Error::InvalidArgument(
        "isovalues of a scalar field only".into(),
      ));
    }
    let scaling = f.maxval();
    Ok(Self::new(orient, Volume::Isovalues { f, val, scaling }))
  }

  /// in, on the boundary
  fn test_point(&self, p: &[f64]) -> (bool, bool) {
    match &self.kind {
      Volume::HalfSpace { x0, n } => {
        let s: f64 = p.iter().zip(x0).zip(n).map(|((a, b), c)| (a - b) * c).sum();
        (s <= 0.0, s * s <= EPS)
      }
      Volume::Sphere { x0, r } => {
        let r2 = vect_dist2_sqr(p, x0);
        (
          r2 <= r * r,
          r2 >= (1.0 - EPS) * r * r && r2 <= (1.0 + EPS) * r * r,
        )
      }
      Volume::Cylinder { x0, d, r } => {
        let mut q = p.to_vec();
        if q.len() == 2 {
          q.push(0.0);
        }
        let q: Vec<f64> = q.iter().zip(x0).map(|(a, b)| a - b).collect();
        let ax = vect_sp(d, &q);
        let d2 = vect_sp(&q, &q) - ax * ax;
        (d2 < r * r, (d2 - r * r).abs() < EPS)
      }
      Volume::Isovalues { .. } => unreachable!("isovalues are prepared apart"),
    }
  }

  fn trinom(a: f64, b: f64, c: f64) -> f64 {
    let delta = b * b - 4.0 * a * c;
    if delta < 0.0 {
      return 1.0 / EPS;
    }
    let delta = delta.sqrt();
    let (s1, s2) = ((-b - delta) / (2.0 * a), (-b + delta) / (2.0 * a));
    if (s1 - 0.5).abs() < (s2 - 0.5).abs() {
      s1
    } else {
      s2
    }
  }

  fn edge_intersect(&self, ia: usize, ib: usize, nodes: &[SliceNode]) -> f64 {
    let (a, b) = (&nodes[ia].pt, &nodes[ib].pt);
    match &self.kind {
      Volume::HalfSpace { x0, n } => {
        let (mut s1, mut s2) = (0.0, 0.0);
        for i in 0..a.len() {
          s1 += (a[i] - b[i]) * n[i];
          s2 += (a[i] - x0[i]) * n[i];
        }
        if s1.abs() < EPS { 1.0 / EPS } else { s2 / s1 }
      }
      Volume::Sphere { x0, r } => {
        let ba: Vec<f64> = b.iter().zip(a).map(|(p, q)| p - q).collect();
        let aa = vect_sp(&ba, &ba);
        if aa < EPS {
          return if self.pt_bin.contains(&ia) {
            0.0
          } else {
            1.0 / EPS
          };
        }
        let ax: Vec<f64> = a.iter().zip(x0).map(|(p, q)| p - q).collect();
        Self::trinom(aa, 2.0 * vect_sp(&ax, &ba), vect_sp(&ax, &ax) - r * r)
      }
      Volume::Cylinder { x0, d, r } => {
        let pad = |v: &[f64]| {
          let mut v = v.to_vec();
          if v.len() == 2 {
            v.push(0.0);
          }
          v
        };
        let f: Vec<f64> = pad(a).iter().zip(x0).map(|(p, q)| p - q).collect();
        let dd: Vec<f64> = pad(b).iter().zip(pad(a)).map(|(p, q)| p - q).collect();
        let (fd, ddd) = (vect_sp(&f, d), vect_sp(&dd, d));
        let aa = vect_sp(&dd, &dd) - ddd * ddd;
        if aa < EPS {
          return if self.pt_bin.contains(&ia) {
            0.0
          } else {
            1.0 / EPS
          };
        }
        let bb = 2.0 * (vect_sp(&f, &dd) - fd * ddd);
        let cc = vect_sp(&f, &f) - fd * fd - r * r;
        Self::trinom(aa, bb, cc)
      }
      Volume::Isovalues { val, .. } => {
        let (ua, ub) = (self.uval[ia], self.uval[ib]);
        if (ua < *val && ub > *val) || (ua > *val && ub < *val) {
          (val - ua) / (ub - ua)
        } else {
          1.0 / EPS
        }
      }
    }
  }

  fn prepare(&mut self, ms: &SlicerState) -> Result<()> {
    self.pt_in.clear();
    self.pt_bin.clear();
    if let Volume::Isovalues { f, val, scaling } = &self.kind {
      self.uval = vec![0.0; ms.nodes.len()];
      let idx: Vec<usize> = ms.nodes_index.iter().copied().collect();
      let xsref: Vec<&[f64]> = idx.iter().map(|&i| ms.nodes[i].pt_ref.as_slice()).collect();
      let Some(v) = f.at(ms.m, ms.cv, &xsref)? else {
        return Ok(());
      };
      for (k, &i) in idx.iter().enumerate() {
        let u = v[k][0];
        self.uval[i] = u;
        let bin = (u - val).abs() < EPS * scaling;
        let mut inn = u - val < 0.0;
        if self.orient == Orient::Out || self.orient == Orient::Split {
          inn = !inn;
        }
        if inn || bin {
          self.pt_in.insert(i);
        }
        if bin {
          self.pt_bin.insert(i);
        }
      }
      return Ok(());
    }
    for &i in &ms.nodes_index {
      let (inn, bin) = self.test_point(&ms.nodes[i].pt);
      let out = matches!(self.orient, Orient::Out | Orient::Split);
      if bin || (if out { !inn } else { inn }) {
        self.pt_in.insert(i);
      }
      if bin {
        self.pt_bin.insert(i);
      }
    }
    Ok(())
  }

  /// intersect s with the slice and decompose it recursively
  fn split_simplex(
    &mut self,
    ms: &mut SlicerState,
    mut s: SliceSimplex,
    sstart: usize,
    spin: u32,
    spbin: u32,
    level: usize,
  ) -> Result<()> {
    if level >= 100 {
      return Err(Error::InvalidArgument(
        "slicer: too deep simplex splitting".into(),
      ));
    }
    let bit = |b: u32, i: usize| b & (1 << i) != 0;
    let d = s.dim();
    let mut found = None;
    'outer: for ia in 0..d {
      if bit(spbin, ia) {
        continue;
      }
      for ib in ia + 1..=d {
        if !bit(spbin, ib) && bit(spin, ia) != bit(spin, ib) {
          let alpha = self.edge_intersect(s.inodes[ia], s.inodes[ib], &ms.nodes);
          if (1e-8..=1.0 - 1e-8).contains(&alpha) {
            found = Some((ia, ib, alpha));
            break 'outer;
          }
        }
      }
    }
    if let Some((ia, ib, alpha)) = found {
      let (a, b) = (&ms.nodes[s.inodes[ia]], &ms.nodes[s.inodes[ib]]);
      let lerp = |x: &[f64], y: &[f64]| -> Vec<f64> {
        x.iter().zip(y).map(|(p, q)| p + alpha * (q - p)).collect()
      };
      let n = SliceNode {
        pt: lerp(&a.pt, &b.pt),
        pt_ref: lerp(&a.pt_ref, &b.pt_ref),
        faces: a.faces & b.faces,
      };
      let nn = ms.nodes.len();
      ms.nodes.push(n);
      self.pt_bin.insert(nn);
      self.pt_in.insert(nn);
      if let Volume::Isovalues { val, .. } = &self.kind {
        self.uval.push(*val);
      }
      let mut s1 = s.clone();
      s1.inodes[ia] = nn;
      self.split_simplex(
        ms,
        s1,
        sstart,
        spin | (1 << ia),
        spbin | (1 << ia),
        level + 1,
      )?;
      s.inodes[ib] = nn;
      self.split_simplex(
        ms,
        s,
        sstart,
        spin | (1 << ib),
        spbin | (1 << ib),
        level + 1,
      )?;
    } else {
      let all_in = (0..=d).all(|i| bit(spin, i));
      let bound = self.orient == Orient::Bound;
      let keep = (all_in && !bound) || self.orient == Orient::Split;
      ms.add_simplex(s.clone(), keep);
      if bound {
        for f in 0..=d {
          let mut face = SliceSimplex { inodes: vec![0; d] };
          let mut ok = true;
          for i in 0..d {
            let p = i + usize::from(i >= f);
            if !bit(spbin, p) {
              ok = false;
              break;
            }
            face.inodes[i] = s.inodes[p];
          }
          if ok {
            // no face twice
            face.inodes.sort();
            if !ms.simplexes[sstart..].contains(&face) {
              ms.add_simplex(face, true);
            }
          }
        }
      }
    }
    Ok(())
  }
}

impl SlicerAction for SlicerVolume {
  fn exec(&mut self, ms: &mut SlicerState) -> Result<()> {
    if ms.splx_in.is_empty() {
      return Ok(());
    }
    self.prepare(ms)?;
    let cands: Vec<usize> = ms.splx_in.iter().copied().collect();
    for cnt in cands {
      let s = ms.simplexes[cnt].clone();
      let (mut spin, mut spbin, mut nin) = (0u32, 0u32, 0);
      for (i, n) in s.inodes.iter().enumerate() {
        if self.pt_in.contains(n) {
          nin += 1;
          spin |= 1 << i;
        }
        if self.pt_bin.contains(n) {
          spbin |= 1 << i;
        }
      }
      if nin == 0 {
        if self.orient != Orient::Split {
          ms.splx_in.remove(&cnt);
        }
      } else if nin != s.dim() + 1 || self.orient == Orient::Bound {
        ms.sup_simplex(cnt);
        let start = ms.simplexes.len();
        self.split_simplex(ms, s, start, spin, spbin, 0)?;
      }
    }
    // nodes on the slice boundary get a new face bit
    if !self.pt_bin.is_empty() {
      if ms.fcnt >= 32 {
        return Err(Error::InvalidArgument(format!(
          "slicer: too many faces and slice faces in convex {}",
          ms.cv
        )));
      }
      for &i in &self.pt_bin {
        ms.nodes[i].faces |= 1 << ms.fcnt;
      }
      ms.fcnt += 1;
    }
    ms.update_nodes_index();
    Ok(())
  }
}

/// face triples of a tetrahedron, oriented (GetFEM)
const TET_FACES: [[usize; 3]; 4] = [[0, 2, 1], [1, 2, 3], [1, 3, 0], [0, 3, 2]];

/// Boundary of a slice: the simplexes on the given faces of the mesh (and
/// on the slice faces), after the action a (GetFEM slicer_boundary).
pub struct SlicerBoundary {
  a: Box<dyn SlicerAction>,
  convex_faces: BTreeMap<usize, u32>,
}

impl SlicerBoundary {
  /// faces: a set of faces (or whole convexes) of m
  pub fn new(m: &MeshData, a: impl SlicerAction + 'static, faces: &RegionSet) -> Self {
    let mut convex_faces = BTreeMap::new();
    for (cv, f) in faces.iter() {
      let e = convex_faces.entry(cv).or_insert(0u32);
      match f {
        Some(f) => *e |= 1 << f,
        None => *e = u32::MAX,
      }
    }
    // the faces that may appear after slicing are boundary too
    for cv in m.convex_index() {
      let nf = m.trans_of_convex(cv).structure().nb_faces();
      *convex_faces.entry(cv).or_insert(0) |= u32::MAX << nf;
    }
    SlicerBoundary {
      a: Box::new(a),
      convex_faces,
    }
  }

  fn on_bound(s: &SliceSimplex, fmask: u32, nodes: &[SliceNode]) -> bool {
    s.inodes.iter().fold(u32::MAX, |f, &i| f & nodes[i].faces) & fmask != 0
  }
}

impl SlicerAction for SlicerBoundary {
  fn exec(&mut self, ms: &mut SlicerState) -> Result<()> {
    self.a.exec(ms)?;
    if ms.splx_in.is_empty() {
      return Ok(());
    }
    let fmask = self.convex_faces.get(&ms.cv).copied().unwrap_or(0);
    if fmask == 0 {
      ms.splx_in.clear();
      return Ok(());
    }
    let n = ms.nodes[0].pt.len();
    let cands: Vec<usize> = ms.splx_in.iter().copied().collect();
    for cnt in cands {
      let s = ms.simplexes[cnt].clone();
      if s.dim() < n {
        if !Self::on_bound(&s, fmask, &ms.nodes) {
          ms.splx_in.remove(&cnt);
        }
      } else if s.dim() == 2 {
        ms.sup_simplex(cnt);
        for (a, b) in [(0, 1), (1, 2), (2, 0)] {
          let s2 = SliceSimplex {
            inodes: vec![s.inodes[a], s.inodes[b]],
          };
          if Self::on_bound(&s2, fmask, &ms.nodes) {
            ms.add_simplex(s2, true);
          }
        }
      } else if s.dim() == 3 {
        ms.sup_simplex(cnt);
        for o in TET_FACES {
          let s2 = SliceSimplex {
            inodes: o.iter().map(|&k| s.inodes[k]).collect(),
          };
          if Self::on_bound(&s2, fmask, &ms.nodes) {
            ms.add_simplex(s2, true);
          }
        }
      }
    }
    ms.update_nodes_index();
    Ok(())
  }
}

/// The union of the slices of a and b (GetFEM slicer_union).
pub struct SlicerUnion {
  a: Box<dyn SlicerAction>,
  b: Box<dyn SlicerAction>,
}

impl SlicerUnion {
  pub fn new(a: impl SlicerAction + 'static, b: impl SlicerAction + 'static) -> Self {
    SlicerUnion {
      a: Box::new(a),
      b: Box::new(b),
    }
  }
}

impl SlicerAction for SlicerUnion {
  fn exec(&mut self, ms: &mut SlicerState) -> Result<()> {
    let base = ms.splx_in.clone();
    let (c, fcnt0) = (ms.simplexes.len(), ms.fcnt);
    self.a.exec(ms)?;
    let in_a = ms.splx_in.clone();
    let fcnt1 = ms.fcnt;
    // b on what a left out: the base and the simplexes a created
    ms.splx_in = base
      .into_iter()
      .chain(c..ms.simplexes.len())
      .filter(|i| !in_a.contains(i) && ms.simplex_index.contains(i))
      .collect();
    self.b.exec(ms)?;
    let in_b = ms.splx_in.clone();
    ms.splx_in.extend(in_a);
    // a face of a's slice inside b's slice is not on the union's boundary
    for f in fcnt0..fcnt1 {
      for &i in &in_b {
        let s = ms.simplexes[i].inodes.clone();
        for j in 0..s.len() {
          let others = s.iter().enumerate().filter(|(k, _)| *k != j);
          if others
            .clone()
            .all(|(_, &n)| ms.nodes[n].faces & (1 << f) != 0)
          {
            for (_, &n) in others {
              ms.nodes[n].faces &= !(1 << f);
            }
          }
        }
      }
    }
    ms.update_nodes_index();
    Ok(())
  }
}

/// The intersection of the slices of a and b (GetFEM slicer_intersect).
pub struct SlicerIntersect {
  a: Box<dyn SlicerAction>,
  b: Box<dyn SlicerAction>,
}

impl SlicerIntersect {
  pub fn new(a: impl SlicerAction + 'static, b: impl SlicerAction + 'static) -> Self {
    SlicerIntersect {
      a: Box::new(a),
      b: Box::new(b),
    }
  }
}

impl SlicerAction for SlicerIntersect {
  fn exec(&mut self, ms: &mut SlicerState) -> Result<()> {
    self.a.exec(ms)?;
    self.b.exec(ms)
  }
}

/// The complement of the slice of a (GetFEM slicer_complementary).
pub struct SlicerComplementary {
  a: Box<dyn SlicerAction>,
}

impl SlicerComplementary {
  pub fn new(a: impl SlicerAction + 'static) -> Self {
    SlicerComplementary { a: Box::new(a) }
  }
}

impl SlicerAction for SlicerComplementary {
  fn exec(&mut self, ms: &mut SlicerState) -> Result<()> {
    let base = ms.splx_in.clone();
    let sz = ms.simplexes.len();
    self.a.exec(ms)?;
    let in_a = std::mem::replace(&mut ms.splx_in, base);
    let idx = &ms.simplex_index;
    ms.splx_in.retain(|i| idx.contains(i));
    let cand: Vec<usize> = ms
      .splx_in
      .iter()
      .copied()
      .chain(sz..ms.simplexes.len())
      .filter(|i| idx.contains(i))
      .collect();
    for i in cand {
      if in_a.contains(&i) {
        ms.splx_in.remove(&i);
      } else {
        ms.splx_in.insert(i);
      }
    }
    Ok(())
  }
}

/// Shrink each convex toward its center (by coef < 1), keeping only the
/// faces of tetrahedra (GetFEM slicer_explode).
pub struct SlicerExplode {
  pub coef: f64,
}

impl SlicerAction for SlicerExplode {
  fn exec(&mut self, ms: &mut SlicerState) -> Result<()> {
    if ms.nodes_index.is_empty() {
      return Ok(());
    }
    let pts: Vec<usize> = match ms.face {
      Some(f) => ms.m.ind_points_of_face_of_convex(ms.cv, f),
      None => ms.m.ind_points_of_convex(ms.cv).to_vec(),
    };
    let n = ms.m.dim();
    let mut g = vec![0.0; n];
    for &p in &pts {
      g.iter_mut().zip(ms.m.point(p)).for_each(|(a, b)| *a += b);
    }
    g.iter_mut().for_each(|v| *v /= pts.len() as f64);
    let idx: Vec<usize> = ms.nodes_index.iter().copied().collect();
    for i in idx {
      let p = &mut ms.nodes[i].pt;
      for k in 0..n {
        p[k] = g[k] + self.coef * (p[k] - g[k]);
      }
    }
    let cands: Vec<usize> = ms.splx_in.iter().copied().collect();
    for cnt in cands {
      let s = ms.simplexes[cnt].clone();
      if s.dim() == 3 {
        ms.sup_simplex(cnt);
        for o in TET_FACES {
          let s2 = SliceSimplex {
            inodes: o.iter().map(|&k| s.inodes[k]).collect(),
          };
          if s2
            .inodes
            .iter()
            .fold(u32::MAX, |f, &i| f & ms.nodes[i].faces)
            != 0
          {
            ms.add_simplex(s2, true);
          }
        }
      }
    }
    Ok(())
  }
}

/// Move the nodes by a displacement field (qdim = mesh dimension)
/// (GetFEM slicer_apply_deformation).
pub struct SlicerApplyDeformation {
  pub f: SliceField,
}

impl SlicerAction for SlicerApplyDeformation {
  fn exec(&mut self, ms: &mut SlicerState) -> Result<()> {
    if self.f.qdim != ms.m.dim() {
      return Err(Error::InvalidArgument(format!(
        "slice deformation: qdim {} for a mesh of dimension {}",
        self.f.qdim,
        ms.m.dim()
      )));
    }
    let idx: Vec<usize> = ms.nodes_index.iter().copied().collect();
    let xsref: Vec<&[f64]> = idx.iter().map(|&i| ms.nodes[i].pt_ref.as_slice()).collect();
    let Some(v) = self.f.at(ms.m, ms.cv, &xsref)? else {
      return Ok(());
    };
    for (k, &i) in idx.iter().enumerate() {
      let p = &mut ms.nodes[i].pt;
      p.resize(self.f.qdim, 0.0);
      p.iter_mut().zip(&v[k]).for_each(|(a, b)| *a += b);
    }
    Ok(())
  }
}

/// The slice of one convex.
#[derive(Clone, Debug)]
pub struct ConvexSlice {
  pub cv_num: usize,
  pub nodes: Vec<SliceNode>,
  pub simplexes: Vec<SliceSimplex>,
}

/// A slice kept for exports and interpolation (GetFEM stored_mesh_slice).
#[derive(Clone, Debug, Default)]
pub struct StoredMeshSlice {
  pub cvlst: Vec<ConvexSlice>,
  cv2pos: BTreeMap<usize, usize>,
  points_cnt: usize,
  dim: usize,
  simplex_cnt: Vec<usize>,
}

impl StoredMeshSlice {
  /// The slice of the convexes (or faces) of region by the actions, each
  /// convex refined nrefine times (GetFEM stored_mesh_slice::build).
  pub fn build(
    m: &MeshData,
    actions: &mut [&mut dyn SlicerAction],
    nrefine: usize,
    region: Option<&RegionSet>,
  ) -> Result<Self> {
    let mut sl = StoredMeshSlice::default();
    let all;
    let region = match region {
      Some(r) => r,
      None => {
        all = m.convex_set();
        &all
      }
    };
    exec(m, None, actions, nrefine, region, &mut |ms| {
      sl.set_convex(ms)
    })?;
    Ok(sl)
  }

  /// As build, on the mesh of mls with its cut convexes sliced along
  /// the level sets (GetFEM mesh_slicer(mls), stored_mesh_slice::build).
  pub fn build_on_level_set(
    mls: &MeshLevelSet,
    actions: &mut [&mut dyn SlicerAction],
    nrefine: usize,
    region: Option<&RegionSet>,
  ) -> Result<Self> {
    let m = mls.linked_mesh().snapshot();
    let mut sl = StoredMeshSlice::default();
    let region = match region {
      Some(r) => r.clone(),
      None => m.convex_set(),
    };
    exec(&m, Some(mls), actions, nrefine, &region, &mut |ms| {
      sl.set_convex(ms)
    })?;
    Ok(sl)
  }

  /// The slice made of the given points, each a 0-simplex in the convex
  /// containing it, within eps in reference space (GetFEM
  /// mesh_slicer::exec(pts), geotrans_inv's EPS: 1e-12).
  pub fn from_points(
    m: &MeshData,
    actions: &mut [&mut dyn SlicerAction],
    pts: &[Vec<f64>],
    eps: f64,
  ) -> Result<Self> {
    let mut sl = StoredMeshSlice::default();
    for cv in m.convex_index() {
      let (gt, g) = (m.trans_of_convex(cv), m.points_of_convex(cv));
      let (lo, hi) = bbox(&g);
      let tol = 1e-10_f64.max(eps * lo.iter().zip(&hi).map(|(a, b)| b - a).fold(0.0, f64::max));
      let mut nodes = vec![];
      for p in pts {
        if p
          .iter()
          .enumerate()
          .any(|(k, x)| *x < lo[k] - tol || *x > hi[k] + tol)
        {
          continue;
        }
        let inv = gt.invert(&g, p, 1e-12, false)?;
        if inv.converged && gt.convex_ref().is_in(&inv.xref)? <= eps {
          nodes.push(SliceNode {
            pt: p.clone(),
            pt_ref: inv.xref,
            faces: 0,
          });
        }
      }
      if nodes.is_empty() {
        continue;
      }
      let simplexes = (0..nodes.len())
        .map(|i| SliceSimplex { inodes: vec![i] })
        .collect();
      let mut ms = state(m, cv, None, nodes, simplexes);
      apply(&mut ms, actions)?;
      sl.set_convex(&ms);
    }
    Ok(sl)
  }

  fn set_convex(&mut self, ms: &SlicerState) {
    if ms.splx_in.is_empty() {
      return;
    }
    let pos = *self.cv2pos.entry(ms.cv).or_insert_with(|| {
      self.cvlst.push(ConvexSlice {
        cv_num: ms.cv,
        nodes: vec![],
        simplexes: vec![],
      });
      self.cvlst.len() - 1
    });
    let mut nused = vec![usize::MAX; ms.nodes.len()];
    for &sn in &ms.splx_in {
      let mut s = ms.simplexes[sn].clone();
      let sc = &mut self.cvlst[pos];
      for l in s.inodes.iter_mut() {
        if nused[*l] == usize::MAX {
          nused[*l] = sc.nodes.len();
          sc.nodes.push(ms.nodes[*l].clone());
          self.dim = self.dim.max(ms.nodes[*l].pt.len());
          self.points_cnt += 1;
        }
        *l = nused[*l];
      }
      if self.simplex_cnt.len() < self.dim + 1 {
        self.simplex_cnt.resize(self.dim + 1, 0);
      }
      self.simplex_cnt[s.dim()] += 1;
      sc.simplexes.push(s);
    }
  }

  pub fn nb_convexes(&self) -> usize {
    self.cvlst.len()
  }

  pub fn nb_points(&self) -> usize {
    self.points_cnt
  }

  pub fn dim(&self) -> usize {
    self.dim
  }

  /// number of simplexes of dimension d
  pub fn nb_simplexes(&self, d: usize) -> usize {
    self.simplex_cnt.get(d).copied().unwrap_or(0)
  }

  /// Values of a field at the nodes of the slice, qdim per node in node
  /// order (GetFEM stored_mesh_slice::interpolate); zero outside mf.
  pub fn interpolate(&self, m: &MeshData, f: &SliceField) -> Result<Vec<f64>> {
    let mut out = vec![];
    for c in &self.cvlst {
      let xsref: Vec<&[f64]> = c.nodes.iter().map(|n| n.pt_ref.as_slice()).collect();
      match f.at(m, c.cv_num, &xsref)? {
        Some(v) => v.into_iter().for_each(|x| out.extend(x)),
        None => out.extend(vec![0.0; xsref.len() * f.qdim]),
      }
    }
    Ok(out)
  }

  /// Nodes at the same place merged (for exports): the merged index of
  /// each node (in node order) and the merged points.
  pub fn merged_nodes(&self) -> (Vec<usize>, Vec<Vec<f64>>) {
    let mut m = MeshData::new(self.dim.max(1));
    let mut idx = vec![];
    let mut pts: Vec<Vec<f64>> = vec![];
    for c in &self.cvlst {
      for n in &c.nodes {
        let i = m.add_point(&n.pt).expect("slice dimension");
        if i == pts.len() {
          pts.push(n.pt.clone());
        }
        idx.push(i);
      }
    }
    (idx, pts)
  }
}

fn bbox(g: &Mat) -> (Vec<f64>, Vec<f64>) {
  let n = g.nrows();
  let mut lo = vec![f64::INFINITY; n];
  let mut hi = vec![f64::NEG_INFINITY; n];
  for j in 0..g.ncols() {
    for k in 0..n {
      lo[k] = lo[k].min(g[(k, j)]);
      hi[k] = hi[k].max(g[(k, j)]);
    }
  }
  (lo, hi)
}

fn state<'m>(
  m: &'m MeshData,
  cv: usize,
  face: Option<usize>,
  nodes: Vec<SliceNode>,
  simplexes: Vec<SliceSimplex>,
) -> SlicerState<'m> {
  let cvr = m.trans_of_convex(cv).convex_ref().clone();
  let nf = cvr.structure().nb_faces();
  SlicerState {
    m,
    cv,
    face,
    cvr,
    nodes,
    simplexes,
    simplex_index: BTreeSet::new(),
    nodes_index: BTreeSet::new(),
    splx_in: BTreeSet::new(),
    fcnt: nf,
  }
}

fn apply(ms: &mut SlicerState, actions: &mut [&mut dyn SlicerAction]) -> Result<()> {
  ms.simplex_index = (0..ms.simplexes.len()).collect();
  ms.splx_in = ms.simplex_index.clone();
  ms.nodes_index = (0..ms.nodes.len()).collect();
  for a in actions.iter_mut() {
    a.exec(ms)?;
  }
  Ok(())
}

/// negative Jacobian at the center (orientation reverted, GetFEM check_orient)
fn reverted(m: &MeshData, cv: usize, gt: &PGeoTrans) -> Result<bool> {
  if gt.dim() != m.dim() || m.dim() < 2 {
    return Ok(false);
  }
  let xref = vec![0.5; gt.dim()];
  let k = m.points_of_convex(cv).matmul(&gt.grad(&xref));
  Ok(
    getfem_linalg::tensor::Lu::new(&k)
      .map(|l| l.det())
      .unwrap_or(0.0)
      < 0.0,
  )
}

/// The mesh slicer loop (GetFEM mesh_slicer::exec_): refine each convex
/// or face of region, apply the actions, hand the result to done. With
/// mls, the cut convexes are their sub-simplices, each refined nrefine
/// times (GetFEM: Bank refinements of the sub-mesh), with nodes of their
/// own, the reference points 1% toward the simplex centre (on the side of
/// the simplex for discontinuous fields, GetFEM).
fn exec(
  m: &MeshData,
  mls: Option<&MeshLevelSet>,
  actions: &mut [&mut dyn SlicerAction],
  nrefine: usize,
  region: &RegionSet,
  done: &mut dyn FnMut(&SlicerState),
) -> Result<()> {
  for (cv, face) in region.iter() {
    let gt = m.trans_of_convex(cv);
    let cvr = gt.convex_ref().clone();
    let rev = reverted(m, cv, gt)?;
    let flag = |xref: &[f64]| -> u32 {
      (0..cvr.structure().nb_faces())
        .filter(|&f| {
          cvr
            .is_in_face(f, xref)
            .map(|v| v.abs() < 1e-10)
            .unwrap_or(false)
        })
        .fold(0, |a, f| a | (1 << f))
    };
    let g = m.points_of_convex(cv);
    let mut nodes = vec![];
    let mut simplexes = vec![];
    match mls.filter(|l| l.is_convex_cut(cv)) {
      None => {
        let rm = refined_simplex_mesh_for_convex(&cvr, nrefine)?;
        let cvms: &[Vec<usize>] = match face {
          Some(f) => &rm.faces[f],
          None => &rm.simplexes,
        };
        let mut ptsid = vec![usize::MAX; rm.points.len()];
        for s in cvms {
          let mut inodes = s.clone();
          if rev && inodes.len() > 1 {
            inodes.swap(0, 1);
          }
          for p in inodes.iter_mut() {
            if ptsid[*p] == usize::MAX {
              ptsid[*p] = nodes.len();
              let xref = &rm.points[*p];
              nodes.push(SliceNode {
                pt: gt.transform(xref, &g)?,
                pt_ref: xref.clone(),
                faces: flag(xref),
              });
            }
            *p = ptsid[*p];
          }
          simplexes.push(SliceSimplex { inodes });
        }
      }
      Some(l) => {
        for s in cut_simplexes(l.mesh_of_convex(cv)?, nrefine)? {
          let fl: Vec<u32> = s.iter().map(|x| flag(x)).collect();
          let mut pts: Vec<Vec<usize>> = match face {
            None => vec![(0..s.len()).collect()],
            Some(f) => (0..s.len())
              .map(|j| (0..s.len()).filter(|&k| k != j).collect::<Vec<_>>())
              .filter(|v| v.iter().all(|&k| fl[k] & (1 << f) != 0))
              .collect(),
          };
          let gc = s.iter().fold(vec![0.0; s[0].len()], |a, x| {
            a.iter().zip(x).map(|(a, b)| a + b).collect()
          });
          for inodes in pts.iter_mut() {
            if rev && inodes.len() > 1 {
              inodes.swap(0, 1);
            }
            for k in inodes.iter_mut() {
              let (x, faces) = (&s[*k], fl[*k]);
              let pt_ref = x
                .iter()
                .zip(&gc)
                .map(|(x, c)| x + 0.01 * (c / s.len() as f64 - x))
                .collect();
              *k = nodes.len();
              nodes.push(SliceNode {
                pt: gt.transform(x, &g)?,
                pt_ref,
                faces,
              });
            }
          }
          for inodes in pts {
            simplexes.push(SliceSimplex { inodes });
          }
        }
      }
    }
    let mut ms = state(m, cv, face, nodes, simplexes);
    apply(&mut ms, actions)?;
    done(&ms);
  }
  Ok(())
}

/// The sub-simplices of a cut convex (its sub-mesh, in the reference
/// element), each refined nrefine times through its (maybe curved)
/// transformation, as vertex lists of degree 1 simplices.
fn cut_simplexes(sub: &MeshData, nrefine: usize) -> Result<Vec<Vec<Vec<f64>>>> {
  let mut out = vec![];
  for i in sub.convex_index() {
    let (sgt, sg) = (sub.trans_of_convex(i), sub.points_of_convex(i));
    let rm = refined_simplex_mesh_for_convex(sgt.convex_ref(), nrefine)?;
    let pts = rm
      .points
      .iter()
      .map(|x| sgt.transform(x, &sg))
      .collect::<std::result::Result<Vec<_>, _>>()?;
    for s in &rm.simplexes {
      out.push(s.iter().map(|&p| pts[p].clone()).collect());
    }
  }
  Ok(out)
}

/// GetFEM's operator<< of base_node: [a, b] with 6 significant digits
fn fmt_vec(v: &[f64]) -> String {
  let s: Vec<String> = v.iter().map(|x| fmt_g(*x)).collect();
  format!("[{}]", s.join(", "))
}

/// C++ default stream format of a double (%g, 6 digits)
pub fn fmt_g(x: f64) -> String {
  if x == 0.0 {
    return "0".into();
  }
  let e = x.abs().log10().floor() as i32;
  if !(-5..6).contains(&e) {
    let s = format!("{:.5e}", x);
    let (m, ex) = s.split_once('e').unwrap();
    let m = m.trim_end_matches('0').trim_end_matches('.');
    let ex: i32 = ex.parse().unwrap();
    return format!("{m}e{}{:02}", if ex < 0 { '-' } else { '+' }, ex.abs());
  }
  let d = (5 - e).max(0) as usize;
  let s = format!("{:.*}", d, x);
  if s.contains('.') {
    s.trim_end_matches('0').trim_end_matches('.').to_string()
  } else {
    s
  }
}

impl fmt::Display for StoredMeshSlice {
  /// GetFEM's operator<< of stored_mesh_slice
  fn fmt(&self, o: &mut fmt::Formatter<'_>) -> fmt::Result {
    writeln!(
      o,
      "stored_mesh_slice, containing {} convexes",
      self.nb_convexes()
    )?;
    for (ic, c) in self.cvlst.iter().enumerate() {
      writeln!(o, "slice convex #{ic} (original = {})", c.cv_num)?;
      for (i, n) in c.nodes.iter().enumerate() {
        writeln!(
          o,
          "node {i}: {}, ref={} flist={:032b}",
          fmt_vec(&n.pt),
          fmt_vec(&n.pt_ref),
          n.faces
        )?;
      }
      for (i, s) in c.simplexes.iter().enumerate() {
        let ids: Vec<String> = s.inodes.iter().map(|x| format!("{x} ")).collect();
        writeln!(o, "simplex {i}, inodes={}", ids.concat())?;
      }
      writeln!(o)?;
    }
    Ok(())
  }
}
