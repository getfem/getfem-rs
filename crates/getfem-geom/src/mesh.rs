//! MeshData: dense points, convexes with their geometric transformation,
//! point-to-convex adjacency, regions (GetFEM's getfem_mesh,
//! bgeot_mesh_structure).
//!
//! A mesh is mutable while owned; once shared (Arc) it is frozen. Every
//! mutation gets a new version stamp.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use crate::convex_structure::PCvs;
use crate::geotrans::{PGeoTrans, PointGeom};
use crate::region::{Region, RegionId, RegionSet, check_set};
use crate::shared::Shared;
use crate::{Error, Result};
use getfem_linalg::tensor::Mat;
use getfem_linalg::vect_dist2;

static STAMP: AtomicU64 = AtomicU64::new(1);

fn next_stamp() -> u64 {
  STAMP.fetch_add(1, Ordering::Relaxed)
}

/// Relative point-merge tolerance (GetFEM: 1e4 * machine epsilon).
pub const MERGE_TOL: f64 = 1e4 * f64::EPSILON;

/// Relative tolerance on the affinity of affine convexes.
pub const AFFINE_TOL: f64 = 1e-10;

#[derive(Clone, Debug)]
struct Convex {
  gt: PGeoTrans,
  pts: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Key(f64);
impl Eq for Key {}
impl PartialOrd for Key {
  fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
    Some(self.cmp(o))
  }
}
impl Ord for Key {
  fn cmp(&self, o: &Self) -> std::cmp::Ordering {
    self.0.total_cmp(&o.0)
  }
}

#[derive(Clone, Debug)]
pub struct MeshData {
  dim: usize,
  pts: Vec<f64>,
  pt_valid: Vec<bool>,
  pt_cvs: Vec<Vec<usize>>,
  bb: Option<(Vec<f64>, Vec<f64>)>,
  dir: Vec<f64>,
  sorter: BTreeSet<(Key, usize)>,
  cvs: Vec<Option<Convex>>,
  regions: BTreeMap<usize, Arc<RegionSet>>,
  version: u64,
  /// changes with points and convexes, not with regions
  geometry_version: u64,
  /// this rank's convexes, with more than one rank (reset by touch)
  mpi_region: OnceLock<Arc<RegionSet>>,
}

/// Fixed direction for the point sorter (irrational components).
fn sort_direction(n: usize) -> Vec<f64> {
  let g = (5f64.sqrt() - 1.0) / 2.0;
  let v: Vec<f64> = (0..n).map(|i| 0.3 + ((i + 1) as f64 * g).fract()).collect();
  let l = v.iter().map(|x| x * x).sum::<f64>().sqrt();
  v.iter().map(|x| x / l).collect()
}

impl MeshData {
  pub fn new(dim: usize) -> Self {
    MeshData {
      dim,
      pts: vec![],
      pt_valid: vec![],
      pt_cvs: vec![],
      bb: None,
      dir: sort_direction(dim),
      sorter: BTreeSet::new(),
      cvs: vec![],
      regions: BTreeMap::new(),
      version: next_stamp(),
      geometry_version: next_stamp(),
      mpi_region: OnceLock::new(),
    }
  }

  /// points or convexes changed
  fn touch(&mut self) {
    self.version = next_stamp();
    self.geometry_version = self.version;
    self.mpi_region = OnceLock::new();
  }

  /// only regions changed
  fn touch_regions(&mut self) {
    self.version = next_stamp();
  }

  /// Changes with every change of the mesh.
  pub fn version(&self) -> u64 {
    self.version
  }

  /// Changes when points or convexes change (not regions): values of
  /// fields on the mesh are then invalid.
  pub fn geometry_version(&self) -> u64 {
    self.geometry_version
  }

  pub fn dim(&self) -> usize {
    self.dim
  }

  // ---------------------------------------------------------------- points

  /// Number of valid points.
  pub fn nb_points(&self) -> usize {
    self.pt_valid.iter().filter(|&&v| v).count()
  }

  /// Upper bound of point indices.
  pub fn nb_points_max(&self) -> usize {
    self.pt_valid.len()
  }

  pub fn is_point_valid(&self, i: usize) -> bool {
    self.pt_valid.get(i).copied().unwrap_or(false)
  }

  pub fn point_index(&self) -> impl Iterator<Item = usize> + '_ {
    (0..self.pt_valid.len()).filter(|&i| self.pt_valid[i])
  }

  pub fn point(&self, i: usize) -> &[f64] {
    &self.pts[i * self.dim..(i + 1) * self.dim]
  }

  fn check_dim(&self, p: &[f64]) -> Result<()> {
    if p.len() != self.dim {
      return Err(Error::InvalidArgument(format!(
        "point of dim {} in a mesh of dim {}",
        p.len(),
        self.dim
      )));
    }
    Ok(())
  }

  fn proj(&self, p: &[f64]) -> f64 {
    p.iter().zip(&self.dir).map(|(a, b)| a * b).sum()
  }

  /// Largest bounding box extent, including p.
  fn extent_with(&self, p: &[f64]) -> f64 {
    match &self.bb {
      None => 0.0,
      Some((lo, hi)) => (0..self.dim)
        .map(|i| hi[i].max(p[i]) - lo[i].min(p[i]))
        .fold(0.0, f64::max),
    }
  }

  /// Merge tolerance: relative to the bounding box.
  pub fn merge_tolerance(&self) -> f64 {
    match &self.bb {
      None => 0.0,
      Some((lo, hi)) => MERGE_TOL * lo.iter().zip(hi).map(|(a, b)| b - a).fold(0.0, f64::max),
    }
  }

  /// Existing point within the merge tolerance (or radius if larger).
  pub fn search_point(&self, p: &[f64], radius: f64) -> Option<usize> {
    if p.len() != self.dim || self.sorter.is_empty() {
      return None;
    }
    let eps = (MERGE_TOL * self.extent_with(p)).max(radius);
    let c = self.proj(p);
    self
      .sorter
      .range((Key(c - eps), 0)..=(Key(c + eps), usize::MAX))
      .map(|&(_, i)| i)
      .find(|&i| vect_dist2(self.point(i), p) <= eps)
  }

  /// Add a point, merging with an existing one within the tolerance.
  pub fn add_point(&mut self, p: &[f64]) -> Result<usize> {
    self.check_dim(p)?;
    if let Some(i) = self.search_point(p, 0.0) {
      return Ok(i);
    }
    Ok(self.add_point_unmerged(p))
  }

  fn add_point_unmerged(&mut self, p: &[f64]) -> usize {
    let i = self.pt_valid.len();
    self.place_point(i, p);
    i
  }

  /// Add a point at a given index (holes below it stay invalid), no merge.
  pub fn add_point_at(&mut self, i: usize, p: &[f64]) -> Result<()> {
    self.check_dim(p)?;
    if self.is_point_valid(i) {
      return Err(Error::InvalidArgument(format!("point {i} already exists")));
    }
    self.place_point(i, p);
    Ok(())
  }

  fn place_point(&mut self, i: usize, p: &[f64]) {
    if i >= self.pt_valid.len() {
      self.pts.resize((i + 1) * self.dim, 0.0);
      self.pt_valid.resize(i + 1, false);
      self.pt_cvs.resize(i + 1, vec![]);
    }
    self.pts[i * self.dim..(i + 1) * self.dim].copy_from_slice(p);
    self.pt_valid[i] = true;
    self.sorter.insert((Key(self.proj(p)), i));
    match &mut self.bb {
      None => self.bb = Some((p.to_vec(), p.to_vec())),
      Some((lo, hi)) => {
        for k in 0..self.dim {
          lo[k] = lo[k].min(p[k]);
          hi[k] = hi[k].max(p[k]);
        }
      }
    }
    self.touch();
  }

  /// Remove a point not used by any convex.
  pub fn sup_point(&mut self, i: usize) -> Result<()> {
    if !self.is_point_valid(i) {
      return Ok(());
    }
    if !self.pt_cvs[i].is_empty() {
      return Err(Error::InvalidArgument(format!(
        "point {i} is used by convexes"
      )));
    }
    self.sorter.remove(&(Key(self.proj(self.point(i))), i));
    self.pt_valid[i] = false;
    self.touch();
    Ok(())
  }

  /// Convexes containing point i.
  pub fn convexes_of_point(&self, i: usize) -> &[usize] {
    &self.pt_cvs[i]
  }

  fn rebuild_sorter_and_bb(&mut self) {
    self.sorter.clear();
    self.bb = None;
    let valid: Vec<usize> = self.point_index().collect();
    for i in valid {
      let p = self.point(i).to_vec();
      self.sorter.insert((Key(self.proj(&p)), i));
      match &mut self.bb {
        None => self.bb = Some((p.clone(), p)),
        Some((lo, hi)) => {
          for k in 0..self.dim {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
          }
        }
      }
    }
  }

  pub fn bounding_box(&self) -> Option<(Vec<f64>, Vec<f64>)> {
    self.bb.clone()
  }

  pub fn translate(&mut self, v: &[f64]) -> Result<()> {
    self.check_dim(v)?;
    for i in self.point_index().collect::<Vec<_>>() {
      let n = self.dim;
      self.pts[i * n..(i + 1) * n]
        .iter_mut()
        .zip(v)
        .for_each(|(x, d)| *x += d);
    }
    self.rebuild_sorter_and_bb();
    self.touch();
    Ok(())
  }

  /// Apply x -> M x; M may change the dimension (M is N' x N).
  pub fn transform(&mut self, m: &Mat) -> Result<()> {
    if m.ncols() != self.dim || m.nrows() == 0 {
      return Err(Error::InvalidArgument(format!(
        "transformation {:?} on a mesh of dim {}",
        m.shape(),
        self.dim
      )));
    }
    let n2 = m.nrows();
    let np = self.pt_valid.len();
    let mut pts = vec![0.0; np * n2];
    for i in 0..np {
      let x = Mat::from_data(&[self.dim, 1], self.point(i).to_vec())?;
      pts[i * n2..(i + 1) * n2].copy_from_slice(m.matmul(&x).data());
    }
    self.pts = pts;
    self.dim = n2;
    self.dir = sort_direction(n2);
    self.rebuild_sorter_and_bb();
    self.touch();
    Ok(())
  }

  // -------------------------------------------------------------- convexes

  /// Number of valid convexes.
  pub fn nb_convexes(&self) -> usize {
    self.cvs.iter().filter(|c| c.is_some()).count()
  }

  /// Upper bound of convex indices.
  pub fn nb_convexes_max(&self) -> usize {
    self.cvs.len()
  }

  pub fn is_convex_valid(&self, cv: usize) -> bool {
    self.cvs.get(cv).is_some_and(|c| c.is_some())
  }

  pub fn convex_index(&self) -> impl Iterator<Item = usize> + '_ {
    (0..self.cvs.len()).filter(|&i| self.cvs[i].is_some())
  }

  /// This rank's convexes: with more than one rank the METIS parts of the
  /// graph of face neighbours, one per rank, computed once per geometry
  /// (GetFEM mesh::compute_mpi_region); all convexes on one rank.
  pub fn mpi_region(&self) -> Arc<RegionSet> {
    let n = getfem_mpi::size();
    if n < 2 {
      let mut rg = RegionSet::new();
      self.convex_index().for_each(|cv| rg.add(cv));
      return Arc::new(rg);
    }
    self
      .mpi_region
      .get_or_init(|| {
        let cvs: Vec<usize> = self.convex_index().collect();
        let mut ind = vec![0; self.cvs.len()];
        cvs.iter().enumerate().for_each(|(i, &cv)| ind[cv] = i);
        let (mut xadj, mut adjncy) = (vec![0], vec![]);
        for &cv in &cvs {
          let mut nb = BTreeSet::new();
          for f in 0..self.structure_of_convex(cv).nb_faces() {
            nb.extend(self.neighbors_of_convex(cv, f));
          }
          adjncy.extend(nb.into_iter().map(|c| ind[c]));
          xadj.push(adjncy.len());
        }
        let part = getfem_mpi::partition(&xadj, &adjncy, n);
        let mut rg = RegionSet::new();
        for (&cv, p) in cvs.iter().zip(part) {
          if p == getfem_mpi::rank() {
            rg.add(cv);
          }
        }
        Arc::new(rg)
      })
      .clone()
  }

  fn cv(&self, cv: usize) -> &Convex {
    self.cvs[cv]
      .as_ref()
      .unwrap_or_else(|| panic!("invalid convex {cv}"))
  }

  pub fn trans_of_convex(&self, cv: usize) -> &PGeoTrans {
    &self.cv(cv).gt
  }

  pub fn structure_of_convex(&self, cv: usize) -> &PCvs {
    self.cv(cv).gt.structure()
  }

  pub fn ind_points_of_convex(&self, cv: usize) -> &[usize] {
    &self.cv(cv).pts
  }

  /// Global point indices of face f of convex cv.
  pub fn ind_points_of_face_of_convex(&self, cv: usize, f: usize) -> Vec<usize> {
    let c = self.cv(cv);
    c.gt
      .structure()
      .ind_points_of_face(f)
      .iter()
      .map(|&i| c.pts[i])
      .collect()
  }

  /// Node matrix G (N x nb_points) of convex cv.
  /// points_of_convex into g, reusing its storage
  pub fn points_of_convex_into(&self, cv: usize, g: &mut Mat) {
    let c = self.cv(cv);
    let n = self.dim * c.pts.len();
    // sized, then the columns copied (a flat_map costs per coordinate)
    g.refill(&[self.dim, c.pts.len()], std::iter::repeat_n(0.0, n));
    for (col, &i) in g.data_mut().chunks_exact_mut(self.dim).zip(&c.pts) {
      col.copy_from_slice(self.point(i));
    }
  }

  pub fn points_of_convex(&self, cv: usize) -> Mat {
    let c = self.cv(cv);
    // sized up front (a flat_map has no exact length: collect would regrow)
    let mut data = Vec::with_capacity(self.dim * c.pts.len());
    for &i in &c.pts {
      data.extend_from_slice(self.point(i));
    }
    Mat::from_data(&[self.dim, c.pts.len()], data).unwrap()
  }

  pub fn add_convex(&mut self, gt: &PGeoTrans, pts: &[usize]) -> Result<usize> {
    let cv = self.cvs.len();
    self.add_convex_at(cv, gt, pts)?;
    Ok(cv)
  }

  /// Add a convex at a given index (holes below it stay invalid).
  pub fn add_convex_at(&mut self, cv: usize, gt: &PGeoTrans, pts: &[usize]) -> Result<()> {
    if pts.len() != gt.nb_points() {
      return Err(Error::InvalidArgument(format!(
        "{}: {} points given, {} expected",
        gt.name(),
        pts.len(),
        gt.nb_points()
      )));
    }
    if gt.dim() > self.dim {
      return Err(Error::InvalidArgument(format!(
        "{} in a mesh of dim {}",
        gt.name(),
        self.dim
      )));
    }
    if let Some(&p) = pts.iter().find(|&&p| !self.is_point_valid(p)) {
      return Err(Error::InvalidArgument(format!("invalid point index {p}")));
    }
    if self.is_convex_valid(cv) {
      return Err(Error::InvalidArgument(format!(
        "convex {cv} already exists"
      )));
    }
    self.check_affine(gt, pts, None)?;
    for &p in pts {
      self.pt_cvs[p].push(cv);
    }
    if cv >= self.cvs.len() {
      self.cvs.resize(cv + 1, None);
    }
    self.cvs[cv] = Some(Convex {
      gt: gt.clone(),
      pts: pts.to_vec(),
    });
    self.touch();
    Ok(())
  }

  /// Add a convex from coordinates, merging points.
  pub fn add_convex_by_points(&mut self, gt: &PGeoTrans, pts: &[&[f64]]) -> Result<usize> {
    let ids = pts
      .iter()
      .map(|p| self.add_point(p))
      .collect::<Result<Vec<_>>>()?;
    self.add_convex(gt, &ids)
  }

  /// Affine transformations must keep an affine node layout.
  fn check_affine(
    &self,
    gt: &PGeoTrans,
    pts: &[usize],
    moved: Option<(usize, &[f64])>,
  ) -> Result<()> {
    if !gt.is_linear() {
      return Ok(());
    }
    let data = pts
      .iter()
      .flat_map(|&i| match moved {
        Some((m, x)) if m == i => x.to_vec(),
        _ => self.point(i).to_vec(),
      })
      .collect();
    let g = Mat::from_data(&[self.dim, pts.len()], data)?;
    let e = gt.affinity_error(&g)?;
    if e > AFFINE_TOL {
      return Err(Error::InvalidArgument(format!(
        "{} is affine but its nodes are not an affine image of the reference nodes (relative deviation {e:.2e}); use a non-linear transformation such as GT_QK",
        gt.name()
      )));
    }
    Ok(())
  }

  /// Move point i to x; fails (mesh unchanged) if an affine convex using it
  /// would become non-affine.
  pub fn set_point(&mut self, i: usize, x: &[f64]) -> Result<()> {
    self.set_points(&[(i, x.to_vec())])
  }

  /// Move several points at once (one sorter rebuild); fails with the mesh
  /// unchanged if an affine convex would become non-affine.
  pub fn set_points(&mut self, moves: &[(usize, Vec<f64>)]) -> Result<()> {
    let n = self.dim;
    for (i, x) in moves {
      self.check_dim(x)?;
      if !self.is_point_valid(*i) {
        return Err(Error::InvalidArgument(format!("invalid point index {i}")));
      }
    }
    let old = self.pts.clone();
    for (i, x) in moves {
      self.pts[i * n..(i + 1) * n].copy_from_slice(x);
    }
    let cvs: BTreeSet<usize> = moves
      .iter()
      .flat_map(|(i, _)| self.pt_cvs[*i].iter().copied())
      .collect();
    for cv in cvs {
      let c = self.cv(cv);
      if let Err(e) = self.check_affine(&c.gt, &c.pts, None) {
        self.pts = old;
        return Err(e);
      }
    }
    self.rebuild_sorter_and_bb();
    self.touch();
    Ok(())
  }

  /// Remove a convex (and its region entries), optionally its orphan points.
  pub fn sup_convex(&mut self, cv: usize, sup_points: bool) {
    let Some(c) = self.cvs.get_mut(cv).and_then(|c| c.take()) else {
      return;
    };
    for &p in &c.pts {
      self.pt_cvs[p].retain(|&x| x != cv);
    }
    if sup_points {
      for &p in &c.pts {
        let _ = self.sup_point(p);
      }
    }
    for r in self.regions.values_mut() {
      Arc::make_mut(r).sup_convex(cv);
    }
    self.touch();
  }

  fn has_points(&self, cv: usize, pts: &[usize]) -> bool {
    let c = &self.cv(cv).pts;
    pts.iter().all(|p| c.contains(p))
  }

  /// A convex of the same dimension sharing face f of cv.
  pub fn neighbor_of_convex(&self, cv: usize, f: usize) -> Option<usize> {
    self.neighbors_of_convex(cv, f).into_iter().next()
  }

  /// All convexes of the same dimension sharing face f of cv.
  pub fn neighbors_of_convex(&self, cv: usize, f: usize) -> Vec<usize> {
    let pts = self.ind_points_of_face_of_convex(cv, f);
    let d = self.structure_of_convex(cv).dim();
    self.pt_cvs[pts[0]]
      .iter()
      .copied()
      .filter(|&c| c != cv && self.structure_of_convex(c).dim() == d && self.has_points(c, &pts))
      .collect()
  }

  /// Convexes sharing with cv the points common to the faces ftab (all
  /// points of cv if ftab is empty); no dimension check. If the faces have
  /// no common point, the neighbors across all faces (GetFEM semantics).
  pub fn neighbors_of_convex_faces(&self, cv: usize, ftab: &[usize]) -> Vec<usize> {
    let ipts: Vec<usize> = match ftab.len() {
      0 => self.ind_points_of_convex(cv).to_vec(),
      1 => self.ind_points_of_face_of_convex(cv, ftab[0]),
      _ => {
        let loc = self
          .structure_of_convex(cv)
          .ind_common_points_of_faces(ftab);
        if loc.is_empty() {
          let mut out = vec![];
          for f in 0..self.structure_of_convex(cv).nb_faces() {
            for c in self.neighbors_of_convex(cv, f) {
              if !out.contains(&c) {
                out.push(c);
              }
            }
          }
          return out;
        }
        loc
          .iter()
          .map(|&i| self.ind_points_of_convex(cv)[i])
          .collect()
      }
    };
    let Some(&p0) = ipts.first() else {
      return vec![];
    };
    self.pt_cvs[p0]
      .iter()
      .copied()
      .filter(|&c| c != cv && self.has_points(c, &ipts[1..]))
      .collect()
  }

  /// (neighbor, its face) across face f of cv.
  pub fn adjacent_face(&self, cv: usize, f: usize) -> Option<(usize, usize)> {
    let n = self.neighbor_of_convex(cv, f)?;
    let pts = self.ind_points_of_face_of_convex(cv, f);
    let s = self.structure_of_convex(n);
    (0..s.nb_faces())
      .find(|&f2| {
        let fp = self.ind_points_of_face_of_convex(n, f2);
        fp.len() == pts.len() && pts.iter().all(|p| fp.contains(p))
      })
      .map(|f2| (n, f2))
  }

  /// Geometry at reference point xref of convex cv.
  pub fn geom_at(&self, cv: usize, xref: &[f64]) -> Result<PointGeom> {
    self
      .trans_of_convex(cv)
      .at(&self.points_of_convex(cv), xref)
  }

  /// Outward normal (not normalized) at reference point xref on face f: B n_ref.
  pub fn normal_of_face_of_convex(&self, cv: usize, f: usize, xref: &[f64]) -> Result<Vec<f64>> {
    let gt = self.trans_of_convex(cv);
    let g = self.geom_at(cv, xref)?;
    let nr = &gt.normals()[f];
    Ok(
      (0..self.dim)
        .map(|i| (0..gt.dim()).map(|j| g.b[(i, j)] * nr[j]).sum())
        .collect(),
    )
  }

  /// Unit outward normal at the mean of the face nodes, as GetFEM.
  pub fn mean_normal_of_face_of_convex(&self, cv: usize, f: usize) -> Result<Vec<f64>> {
    let gt = self.trans_of_convex(cv);
    let ind = gt.structure().ind_points_of_face(f);
    let mut xmref = vec![0.0; gt.dim()];
    for &i in ind {
      xmref
        .iter_mut()
        .zip(&gt.geometric_nodes()[i])
        .for_each(|(a, b)| *a += b);
    }
    xmref.iter_mut().for_each(|a| *a /= ind.len() as f64);
    let n = self.normal_of_face_of_convex(cv, f, &xmref)?;
    let l = n.iter().map(|x| x * x).sum::<f64>().sqrt();
    Ok(n.iter().map(|x| x / l).collect())
  }

  /// Element size (see GeoTrans::element_size).
  pub fn convex_size(&self, cv: usize) -> Result<f64> {
    self
      .trans_of_convex(cv)
      .element_size(&self.points_of_convex(cv))
  }

  /// Compact convex and point numbering, same swaps as GetFEM
  /// (optimize_structure(false)).
  pub fn optimize_structure(&mut self) {
    let nbc = self.nb_convexes();
    for i in 0..nbc {
      if self.cvs[i].is_none() {
        let last = (0..self.cvs.len())
          .rev()
          .find(|&j| self.cvs[j].is_some())
          .unwrap();
        self.swap_convex(i, last);
      }
    }
    self.cvs.truncate(nbc);
    if !self.pt_valid.is_empty() {
      let (mut i, mut j) = (0usize, self.pt_valid.len() - 1);
      while i < j {
        while i < j && self.pt_valid[i] {
          i += 1;
        }
        while i < j && !self.pt_valid[j] {
          j -= 1;
        }
        if i < j {
          self.swap_points(i, j);
        }
        i += 1;
        j = j.saturating_sub(1);
      }
    }
    let np = self.nb_points();
    self.pts.truncate(np * self.dim);
    self.pt_valid.truncate(np);
    self.pt_cvs.truncate(np);
    self.rebuild_sorter_and_bb();
    self.touch();
  }

  /// Merge points closer than tol (0: the merge tolerance), among the
  /// points of the convexes and faces of r only. The lowest index
  /// survives, chains collapse; the merged points leave holes. Fails, mesh
  /// unchanged, if a convex would get a point twice. Returns the number of
  /// points removed.
  pub fn merge_points(&mut self, tol: f64, r: &RegionSet) -> Result<usize> {
    let mut cand = BTreeSet::new();
    for (cv, f) in r.iter().filter(|&(cv, _)| self.is_convex_valid(cv)) {
      match f {
        None => cand.extend(self.ind_points_of_convex(cv)),
        Some(f) => cand.extend(self.ind_points_of_face_of_convex(cv, f)),
      }
    }
    let eps = if tol > 0.0 {
      tol
    } else {
      self.merge_tolerance()
    };
    // rep[j]: the surviving point of j
    let mut rep: BTreeMap<usize, usize> = BTreeMap::new();
    for &i in &cand {
      if rep.contains_key(&i) {
        continue;
      }
      let mut queue = vec![i];
      while let Some(k) = queue.pop() {
        let c = self.proj(self.point(k));
        for &(_, j) in self
          .sorter
          .range((Key(c - eps), 0)..=(Key(c + eps), usize::MAX))
        {
          if j != i
            && cand.contains(&j)
            && !rep.contains_key(&j)
            && vect_dist2(self.point(j), self.point(k)) <= eps
          {
            rep.insert(j, i);
            queue.push(j);
          }
        }
      }
    }
    let mut cvs: BTreeSet<usize> = BTreeSet::new();
    for &j in rep.keys() {
      cvs.extend(&self.pt_cvs[j]);
    }
    let mut new_pts = vec![];
    for cv in cvs {
      let pts: Vec<usize> = self
        .cv(cv)
        .pts
        .iter()
        .map(|p| *rep.get(p).unwrap_or(p))
        .collect();
      if pts.iter().collect::<BTreeSet<_>>().len() < pts.len() {
        return Err(Error::InvalidArgument(format!(
          "merge_points: convex {cv} would get a point twice (tolerance {eps:e})"
        )));
      }
      new_pts.push((cv, pts));
    }
    for (cv, pts) in new_pts {
      self.cvs[cv].as_mut().unwrap().pts = pts;
    }
    for (&j, &i) in &rep {
      let moved = std::mem::take(&mut self.pt_cvs[j]);
      self.pt_cvs[i].extend(moved);
      self.pt_cvs[i].sort_unstable();
      self.pt_valid[j] = false;
    }
    if !rep.is_empty() {
      self.rebuild_sorter_and_bb();
      self.touch();
    }
    Ok(rep.len())
  }

  fn swap_convex(&mut self, i: usize, j: usize) {
    if i == j {
      return;
    }
    let pts: BTreeSet<usize> = [i, j]
      .iter()
      .filter_map(|&c| self.cvs[c].as_ref())
      .flat_map(|c| c.pts.iter().copied())
      .collect();
    for p in pts {
      for x in self.pt_cvs[p].iter_mut() {
        if *x == i {
          *x = j;
        } else if *x == j {
          *x = i;
        }
      }
    }
    self.cvs.swap(i, j);
    for r in self.regions.values_mut() {
      Arc::make_mut(r).swap_convex(i, j);
    }
  }

  fn swap_points(&mut self, i: usize, j: usize) {
    if i == j {
      return;
    }
    let cvs: BTreeSet<usize> = self.pt_cvs[i]
      .iter()
      .chain(&self.pt_cvs[j])
      .copied()
      .collect();
    for cv in cvs {
      for p in self.cvs[cv].as_mut().unwrap().pts.iter_mut() {
        if *p == i {
          *p = j;
        } else if *p == j {
          *p = i;
        }
      }
    }
    let n = self.dim;
    for k in 0..n {
      self.pts.swap(i * n + k, j * n + k);
    }
    self.pt_valid.swap(i, j);
    self.pt_cvs.swap(i, j);
  }

  // --------------------------------------------------------------- regions

  /// The content of the stored region id.
  pub fn region_set(&self, id: usize) -> Option<&Arc<RegionSet>> {
    self.regions.get(&id)
  }

  pub fn has_region(&self, id: usize) -> bool {
    self.regions.contains_key(&id)
  }

  pub fn region_ids(&self) -> impl Iterator<Item = usize> + '_ {
    self.regions.keys().copied()
  }

  /// One more than the largest region id.
  pub fn free_region_id(&self) -> usize {
    self.regions.keys().last().map_or(0, |&i| i + 1)
  }

  /// Store region id with a set of convexes and faces of the mesh.
  pub fn set_region_set(&mut self, id: usize, r: RegionSet) -> Result<()> {
    check_set(self, &r, &format!("region {id}"))?;
    self.regions.insert(id, Arc::new(r));
    self.touch_regions();
    Ok(())
  }

  pub fn sup_region(&mut self, id: usize) {
    if self.regions.remove(&id).is_some() {
      self.touch_regions();
    }
  }

  /// The entire domain.
  pub fn all_convexes(&self) -> Region {
    Region::all_convexes()
  }

  /// The set of all convexes.
  pub fn convex_set(&self) -> RegionSet {
    let mut r = RegionSet::new();
    for cv in self.convex_index() {
      r.add(cv);
    }
    r
  }
}

#[cfg(test)]
mod tests;

// ------------------------------------------------------------------ handle

/// A mesh: a handle to one shared `MeshData` (SPEC 3). Clones are the same
/// mesh; a change made through any of them is seen by all its users.
/// Readers get owned values; `snapshot()` gives a consistent view for
/// longer work.
#[derive(Clone, Debug)]
pub struct Mesh(Shared<MeshData>);

/// forwards &self methods returning owned values
macro_rules! read {
  ($($name:ident($($a:ident: $t:ty),*) -> $r:ty;)*) => {
    $(pub fn $name(&self, $($a: $t),*) -> $r { self.0.snapshot().$name($($a),*) })*
  };
}

/// forwards &mut self methods
macro_rules! write {
  ($($name:ident($($a:ident: $t:ty),*) -> $r:ty;)*) => {
    $(pub fn $name(&self, $($a: $t),*) -> $r { self.0.modify(|m| m.$name($($a),*)) })*
  };
}

impl Mesh {
  pub fn new(dim: usize) -> Self {
    Mesh(Shared::new(MeshData::new(dim)))
  }

  /// The current version of the mesh, unaffected by later changes.
  pub fn snapshot(&self) -> Arc<MeshData> {
    self.0.snapshot()
  }

  /// Change the mesh with MeshData's methods.
  pub fn modify<R>(&self, f: impl FnOnce(&mut MeshData) -> R) -> R {
    self.0.modify(f)
  }

  /// Same mesh (not only equal)?
  pub fn same(&self, other: &Mesh) -> bool {
    self.0.same(&other.0)
  }

  read! {
    version() -> u64;
    geometry_version() -> u64;
    dim() -> usize;
    nb_points() -> usize;
    nb_points_max() -> usize;
    is_point_valid(i: usize) -> bool;
    merge_tolerance() -> f64;
    search_point(p: &[f64], radius: f64) -> Option<usize>;
    bounding_box() -> Option<(Vec<f64>, Vec<f64>)>;
    nb_convexes() -> usize;
    nb_convexes_max() -> usize;
    is_convex_valid(cv: usize) -> bool;
    ind_points_of_face_of_convex(cv: usize, f: usize) -> Vec<usize>;
    points_of_convex(cv: usize) -> Mat;
    neighbor_of_convex(cv: usize, f: usize) -> Option<usize>;
    neighbors_of_convex(cv: usize, f: usize) -> Vec<usize>;
    neighbors_of_convex_faces(cv: usize, ftab: &[usize]) -> Vec<usize>;
    adjacent_face(cv: usize, f: usize) -> Option<(usize, usize)>;
    geom_at(cv: usize, xref: &[f64]) -> Result<PointGeom>;
    normal_of_face_of_convex(cv: usize, f: usize, xref: &[f64]) -> Result<Vec<f64>>;
    mean_normal_of_face_of_convex(cv: usize, f: usize) -> Result<Vec<f64>>;
    convex_size(cv: usize) -> Result<f64>;
    has_region(id: usize) -> bool;
    free_region_id() -> usize;
    all_convexes() -> Region;
  }

  write! {
    add_point(p: &[f64]) -> Result<usize>;
    add_point_at(i: usize, p: &[f64]) -> Result<()>;
    sup_point(i: usize) -> Result<()>;
    translate(v: &[f64]) -> Result<()>;
    transform(m: &Mat) -> Result<()>;
    add_convex(gt: &PGeoTrans, pts: &[usize]) -> Result<usize>;
    add_convex_at(cv: usize, gt: &PGeoTrans, pts: &[usize]) -> Result<()>;
    add_convex_by_points(gt: &PGeoTrans, pts: &[&[f64]]) -> Result<usize>;
    set_point(i: usize, x: &[f64]) -> Result<()>;
    set_points(moves: &[(usize, Vec<f64>)]) -> Result<()>;
    sup_convex(cv: usize, sup_points: bool) -> ();
    optimize_structure() -> ();
    set_region_set(id: usize, r: RegionSet) -> Result<()>;
    sup_region(id: usize) -> ();
  }

  pub fn point_index(&self) -> Vec<usize> {
    self.0.snapshot().point_index().collect()
  }

  pub fn point(&self, i: usize) -> Vec<f64> {
    self.0.snapshot().point(i).to_vec()
  }

  pub fn convexes_of_point(&self, i: usize) -> Vec<usize> {
    self.0.snapshot().convexes_of_point(i).to_vec()
  }

  pub fn convex_index(&self) -> Vec<usize> {
    self.0.snapshot().convex_index().collect()
  }

  pub fn trans_of_convex(&self, cv: usize) -> PGeoTrans {
    self.0.snapshot().trans_of_convex(cv).clone()
  }

  pub fn structure_of_convex(&self, cv: usize) -> PCvs {
    self.0.snapshot().structure_of_convex(cv).clone()
  }

  pub fn ind_points_of_convex(&self, cv: usize) -> Vec<usize> {
    self.0.snapshot().ind_points_of_convex(cv).to_vec()
  }

  /// MeshData::merge_points, among the points of r (-1: all).
  pub fn merge_points(&self, tol: f64, r: impl Into<crate::region::RegionArg>) -> Result<usize> {
    let set = r.into().content(&self.snapshot())?;
    self.modify(|m| m.merge_points(tol, &set))
  }

  /// Region id of the mesh (its content is looked up at each use), or the
  /// entire domain for -1. An error if the mesh has no region id.
  pub fn region(&self, id: impl Into<RegionId>) -> Result<Region> {
    match id.into().0 {
      -1 => Ok(Region::all_convexes()),
      i if i < 0 => Err(Error::InvalidArgument(format!(
        "region id {i}: -1 or an id of the mesh"
      ))),
      i => {
        if !self.has_region(i as usize) {
          return Err(Error::InvalidArgument(format!(
            "the mesh has no region {i}"
          )));
        }
        Ok(Region::stored(self, i as usize))
      }
    }
  }

  /// Store region id with the current content of r (e.g. a temporary
  /// region, or another stored one).
  pub fn set_region(&self, id: usize, r: &Region) -> Result<()> {
    // resolved before the mesh is locked for writing
    let set = r.resolve(&self.snapshot())?;
    self.set_region_set(id, (*set).clone())
  }

  /// The current content of region id (a copy, for inspection).
  pub fn region_set(&self, id: usize) -> Option<RegionSet> {
    self.0.snapshot().region_set(id).map(|r| (**r).clone())
  }

  pub fn region_ids(&self) -> Vec<usize> {
    self.0.snapshot().region_ids().collect()
  }

  /// Outer faces whose unit normal makes an angle <= angle with v
  /// (GetFEM's scripting `outer faces with direction`).
  pub fn outer_faces_with_direction(&self, v: &[f64], angle: f64) -> Result<Region> {
    crate::region::select_faces_of_normal(self, crate::region::outer_faces_of_mesh(self), v, angle)
  }

  /// Outer faces with all nodes in the box [lo, hi] (GetFEM's scripting
  /// `outer faces in box`).
  pub fn outer_faces_in_box(&self, lo: &[f64], hi: &[f64]) -> Result<Region> {
    crate::region::select_faces_in_box(self, crate::region::outer_faces_of_mesh(self), lo, hi)
  }
}

/// Prismatic mesh: m extruded along a new last coordinate over [0, 1], in
/// nb_layers layers of degree `degree`. Convexes are products of m's
/// with GT_PK(1, degree), nodes layer by layer.
pub fn extrude(m: &Mesh, nb_layers: usize, degree: usize) -> Result<Mesh> {
  let nt = nb_layers * degree;
  if nt == 0 {
    return Err(Error::InvalidArgument(
      "extrude: nb_layers and degree must be positive".into(),
    ));
  }
  let src = m.snapshot();
  let dim = src.dim();
  let out = Mesh::new(dim + 1);
  let line = crate::geotrans::pk(1, degree)?;
  // the nt + 1 copies of each point
  let mut ids = vec![vec![]; src.nb_points_max()];
  for i in src.point_index() {
    let mut p = src.point(i).to_vec();
    p.push(0.0);
    ids[i] = (0..=nt)
      .map(|j| {
        p[dim] = j as f64 / nt as f64;
        out.add_point(&p)
      })
      .collect::<Result<_>>()?;
  }
  let ids = &ids;
  for cv in src.convex_index() {
    let pts = src.ind_points_of_convex(cv);
    let gt = crate::geotrans::product(src.trans_of_convex(cv), &line);
    for j in 0..nb_layers {
      let tab: Vec<usize> = (0..=degree)
        .flat_map(|d| pts.iter().map(move |&k| ids[k][j * degree + d]))
        .collect();
      out.add_convex(&gt, &tab)?;
    }
  }
  Ok(out)
}

impl From<MeshData> for Mesh {
  fn from(m: MeshData) -> Self {
    Mesh(Shared::new(m))
  }
}

/// A mesh given as a handle or as plain data, for functions that only read
/// it (`outer_faces_of_mesh(&mesh)` works with both).
pub trait AsMeshData {
  fn view(&self) -> MeshView<'_>;
}

pub enum MeshView<'a> {
  Borrowed(&'a MeshData),
  Shared(Arc<MeshData>),
}

impl std::ops::Deref for MeshView<'_> {
  type Target = MeshData;
  fn deref(&self) -> &MeshData {
    match self {
      MeshView::Borrowed(m) => m,
      MeshView::Shared(m) => m,
    }
  }
}

impl AsMeshData for MeshData {
  fn view(&self) -> MeshView<'_> {
    MeshView::Borrowed(self)
  }
}

impl AsMeshData for Mesh {
  fn view(&self) -> MeshView<'_> {
    MeshView::Shared(self.snapshot())
  }
}

impl AsMeshData for Arc<MeshData> {
  fn view(&self) -> MeshView<'_> {
    MeshView::Borrowed(self)
  }
}
