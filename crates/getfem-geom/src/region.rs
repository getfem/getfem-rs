//! Regions (port of mesh_region, simplified). A `Region` is an identifier:
//! the entire domain, a region stored in a mesh (by id), or a temporary set
//! of convexes and faces (e.g. an intersection, for one operation). Its
//! content is looked up in the mesh when it is used, so stored regions
//! follow their edits and the mesh operations. The content itself is a
//! `RegionSet`: bit 0 of a mask is the convex itself, bit f+1 its face f.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::mesh::{AsMeshData, Mesh, MeshData};
use crate::{Error, Result};

/// Id of the entire domain (GetFEM's -1).
pub const ALL: i64 = -1;
/// Id reported by temporary regions (GetFEM uses -2 for these).
pub const TEMPORARY: i64 = -2;

/// A set of convexes and convex faces.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RegionSet(BTreeMap<usize, u64>);

pub(crate) const MAX_FACES: usize = 63;

impl RegionSet {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn add(&mut self, cv: usize) {
    *self.0.entry(cv).or_default() |= 1;
  }

  pub fn add_face(&mut self, cv: usize, f: usize) {
    assert!(f < MAX_FACES, "face number {f} too large");
    *self.0.entry(cv).or_default() |= 1 << (f + 1);
  }

  pub fn sup(&mut self, cv: usize) {
    self.clear_bits(cv, 1);
  }

  pub fn sup_face(&mut self, cv: usize, f: usize) {
    self.clear_bits(cv, 1 << (f + 1));
  }

  fn clear_bits(&mut self, cv: usize, m: u64) {
    if let Some(b) = self.0.get_mut(&cv) {
      *b &= !m;
      if *b == 0 {
        self.0.remove(&cv);
      }
    }
  }

  pub(crate) fn sup_convex(&mut self, cv: usize) {
    self.0.remove(&cv);
  }

  pub(crate) fn swap_convex(&mut self, i: usize, j: usize) {
    let (a, b) = (self.0.remove(&i), self.0.remove(&j));
    if let Some(a) = a {
      self.0.insert(j, a);
    }
    if let Some(b) = b {
      self.0.insert(i, b);
    }
  }

  pub fn is_in(&self, cv: usize) -> bool {
    self.0.get(&cv).is_some_and(|b| b & 1 != 0)
  }

  pub fn is_in_face(&self, cv: usize, f: usize) -> bool {
    self.0.get(&cv).is_some_and(|b| b & (1 << (f + 1)) != 0)
  }

  pub fn is_empty(&self) -> bool {
    self.0.is_empty()
  }

  /// Number of entries (convexes and faces).
  pub fn size(&self) -> usize {
    self.0.values().map(|b| b.count_ones() as usize).sum()
  }

  /// (convex, None) for convexes, (convex, Some(face)) for faces, ordered.
  pub fn iter(&self) -> impl Iterator<Item = (usize, Option<usize>)> + '_ {
    // set bits of b, lowest first
    self.0.iter().flat_map(|(&cv, &b)| {
      std::iter::successors((b != 0).then_some(b), |&r| {
        let r = r & (r - 1);
        (r != 0).then_some(r)
      })
      .map(move |r| (cv, (r.trailing_zeros() as usize).checked_sub(1)))
    })
  }

  pub fn convexes(&self) -> impl Iterator<Item = usize> + '_ {
    self
      .0
      .iter()
      .filter(|(_, b)| *b & 1 != 0)
      .map(|(&cv, _)| cv)
  }

  pub fn is_only_convexes(&self) -> bool {
    self.0.values().all(|&b| b == 1)
  }

  pub fn is_only_faces(&self) -> bool {
    self.0.values().all(|&b| b & 1 == 0)
  }

  pub fn merge(&self, o: &RegionSet) -> RegionSet {
    let mut r = self.clone();
    for (&cv, &b) in &o.0 {
      *r.0.entry(cv).or_default() |= b;
    }
    r
  }

  pub fn subtract(&self, o: &RegionSet) -> RegionSet {
    let mut r = self.clone();
    for (&cv, &b) in &o.0 {
      r.clear_bits(cv, b);
    }
    r
  }

  /// As GetFEM: a whole convex intersected with its faces keeps the faces.
  pub fn intersect(&self, o: &RegionSet) -> RegionSet {
    let mut r = RegionSet::new();
    for (&cv, &a) in &self.0 {
      if let Some(&b) = o.0.get(&cv) {
        let m = match (a & 1 != 0, b & 1 != 0) {
          (true, false) => b,
          (false, true) => a,
          _ => a & b,
        };
        if m != 0 {
          r.0.insert(cv, m);
        }
      }
    }
    r
  }
}

/// A region id as given by the user: an id of the mesh, or -1 for the
/// entire domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegionId(pub i64);

impl From<i32> for RegionId {
  fn from(i: i32) -> Self {
    RegionId(i as i64)
  }
}
impl From<i64> for RegionId {
  fn from(i: i64) -> Self {
    RegionId(i)
  }
}
impl From<usize> for RegionId {
  fn from(i: usize) -> Self {
    RegionId(i as i64)
  }
}

/// A region argument: a region id of the mesh of the object it is given
/// to (-1: everywhere), or a region. `None` means everywhere.
#[derive(Clone, Debug)]
pub enum RegionArg {
  Id(i64),
  Region(Region),
}

impl From<i32> for RegionArg {
  fn from(i: i32) -> Self {
    RegionArg::Id(i as i64)
  }
}
impl From<i64> for RegionArg {
  fn from(i: i64) -> Self {
    RegionArg::Id(i)
  }
}
impl From<usize> for RegionArg {
  fn from(i: usize) -> Self {
    RegionArg::Id(i as i64)
  }
}
impl From<&Region> for RegionArg {
  fn from(r: &Region) -> Self {
    RegionArg::Region(r.clone())
  }
}
impl From<Region> for RegionArg {
  fn from(r: Region) -> Self {
    RegionArg::Region(r)
  }
}
impl From<Option<&Region>> for RegionArg {
  fn from(r: Option<&Region>) -> Self {
    r.map_or(RegionArg::Id(ALL), |r| RegionArg::Region(r.clone()))
  }
}

impl RegionArg {
  /// The region on mesh: an id must be a region of mesh (or -1).
  pub fn on(self, mesh: &Mesh) -> Result<Region> {
    match self {
      RegionArg::Id(i) => mesh.region(i),
      RegionArg::Region(r) => Ok(r),
    }
  }

  /// The region on the mesh data m (an id resolved by content).
  pub fn content(&self, m: &MeshData) -> Result<Arc<RegionSet>> {
    match self {
      RegionArg::Id(ALL) => Ok(Arc::new(m.convex_set())),
      RegionArg::Id(i) if *i < 0 => Err(Error::InvalidArgument(format!(
        "region id {i}: -1 or an id of the mesh"
      ))),
      RegionArg::Id(i) => m
        .region_set(*i as usize)
        .cloned()
        .ok_or_else(|| Error::InvalidArgument(format!("the mesh has no region {i}"))),
      RegionArg::Region(r) => r.resolve(m),
    }
  }

  /// Everywhere?
  pub fn is_all(&self) -> bool {
    match self {
      RegionArg::Id(i) => *i == ALL,
      RegionArg::Region(r) => r.id() == ALL,
    }
  }
}

#[derive(Clone, Debug)]
enum Kind {
  All,
  Stored {
    mesh: Mesh,
    id: usize,
  },
  /// valid for the points and convexes of this geometry version
  Temporary {
    set: Arc<RegionSet>,
    geometry: u64,
  },
}

/// A region: the entire domain, a region stored in a mesh, or a temporary
/// set for one operation.
#[derive(Clone, Debug)]
pub struct Region(Kind);

impl Region {
  /// The entire domain (GetFEM mesh_region::all_convexes()).
  pub fn all_convexes() -> Region {
    Region(Kind::All)
  }

  pub(crate) fn stored(mesh: &Mesh, id: usize) -> Region {
    Region(Kind::Stored {
      mesh: mesh.clone(),
      id,
    })
  }

  /// A temporary region: a set on the current geometry of m (points and
  /// convexes); an error to use once it changed.
  pub fn temporary(m: &impl AsMeshData, set: RegionSet) -> Result<Region> {
    let m = &*m.view();
    check_set(m, &set, "temporary region")?;
    Ok(Region(Kind::Temporary {
      set: Arc::new(set),
      geometry: m.geometry_version(),
    }))
  }

  /// The id: the mesh's for a stored region, ALL (-1) for the entire
  /// domain, TEMPORARY (-2) for a temporary region.
  pub fn id(&self) -> i64 {
    match &self.0 {
      Kind::All => ALL,
      Kind::Stored { id, .. } => *id as i64,
      Kind::Temporary { .. } => TEMPORARY,
    }
  }

  pub fn is_temporary(&self) -> bool {
    matches!(self.0, Kind::Temporary { .. })
  }

  /// Error for temporary regions: objects that outlive an operation
  /// (terms, conditions, spaces) need the entire domain or a stored region.
  pub fn lasting(&self, what: &str) -> Result<()> {
    if self.is_temporary() {
      return Err(Error::InvalidArgument(format!(
        "{what}: a temporary region cannot be kept; store it in the mesh \
         (mesh.set_region(id, &region)) and pass mesh.region(id)"
      )));
    }
    Ok(())
  }

  /// The current content, for the points and convexes of m (a state of
  /// the mesh of a stored region; its regions are read from the mesh
  /// itself, which may have newer region edits than m).
  pub fn resolve(&self, m: &MeshData) -> Result<Arc<RegionSet>> {
    match &self.0 {
      Kind::All => Ok(Arc::new(m.convex_set())),
      Kind::Stored { mesh, id } => {
        let cur = mesh.snapshot();
        let src = if cur.geometry_version() == m.geometry_version() {
          &cur
        } else {
          m
        };
        src
          .region_set(*id)
          .cloned()
          .ok_or_else(|| Error::InvalidArgument(format!("region {id} does not exist in the mesh")))
      }
      Kind::Temporary { set, geometry } => {
        if *geometry != m.geometry_version() {
          return Err(Error::InvalidArgument(
            "a temporary region was built for an older state of the mesh (points or convexes changed); build it again".into(),
          ));
        }
        Ok(set.clone())
      }
    }
  }

  /// Content and geometry version of a region that knows its mesh
  fn own(&self) -> Result<(Arc<RegionSet>, u64)> {
    match &self.0 {
      Kind::All => Err(Error::InvalidArgument(
        "the entire domain (-1) in a union or difference (as GetFEM): use the region of all its convexes, mesh.snapshot().all_convexes()".into(),
      )),
      Kind::Stored { mesh, .. } => {
        let m = mesh.snapshot();
        Ok((self.resolve(&m)?, m.geometry_version()))
      }
      Kind::Temporary { set, geometry } => Ok((set.clone(), *geometry)),
    }
  }

  fn combine(&self, o: &Region, f: impl Fn(&RegionSet, &RegionSet) -> RegionSet) -> Result<Region> {
    let ((a, ga), (b, gb)) = (self.own()?, o.own()?);
    if ga != gb {
      return Err(Error::InvalidArgument(
        "regions of different meshes or mesh states".into(),
      ));
    }
    Ok(Region(Kind::Temporary {
      set: Arc::new(f(&a, &b)),
      geometry: ga,
    }))
  }

  /// Union, as a temporary region.
  pub fn merge(&self, o: &Region) -> Result<Region> {
    self.combine(o, |a, b| a.merge(b))
  }

  /// Difference, as a temporary region.
  pub fn subtract(&self, o: &Region) -> Result<Region> {
    self.combine(o, |a, b| a.subtract(b))
  }

  /// Intersection (a whole convex with its faces keeps the faces), as a
  /// temporary region; with the entire domain the other region (GetFEM
  /// mesh_region::intersection).
  pub fn intersect(&self, o: &Region) -> Result<Region> {
    match (&self.0, &o.0) {
      (Kind::All, Kind::All) => Err(Error::InvalidArgument(
        "intersection of the entire domain with itself (as GetFEM)".into(),
      )),
      (Kind::All, _) => Ok(o.clone()),
      (_, Kind::All) => Ok(self.clone()),
      _ => self.combine(o, |a, b| a.intersect(b)),
    }
  }
}

/// Entries must exist in the mesh.
pub(crate) fn check_set(m: &MeshData, r: &RegionSet, what: &str) -> Result<()> {
  for (cv, f) in r.iter() {
    if !m.is_convex_valid(cv) {
      return Err(Error::InvalidArgument(format!(
        "{what}: invalid convex {cv}"
      )));
    }
    if let Some(f) = f
      && f >= m.structure_of_convex(cv).nb_faces()
    {
      return Err(Error::InvalidArgument(format!(
        "{what}: convex {cv} has no face {f}"
      )));
    }
  }
  Ok(())
}

fn only_convexes(r: &RegionSet) -> Result<()> {
  if !r.is_only_convexes() {
    return Err(Error::InvalidArgument(
      "expected a region of convexes".into(),
    ));
  }
  Ok(())
}

/// Faces of the convexes of r not shared with another convex of r of the
/// same dimension. GetFEM instead returns lower-dimensional convexes whole,
/// which leaves no way to get the boundary of a surface mesh in 3D.
pub fn outer_faces(m: &impl AsMeshData, r: impl Into<RegionArg>) -> Result<Region> {
  let m = &*m.view();
  let r = &*r.into().content(m)?;
  only_convexes(r)?;
  let mut out = RegionSet::new();
  for cv in r.convexes() {
    for f in 0..m.structure_of_convex(cv).nb_faces() {
      if !m.neighbors_of_convex(cv, f).iter().any(|&c2| r.is_in(c2)) {
        out.add_face(cv, f);
      }
    }
  }
  Region::temporary(m, out)
}

/// Outer faces of the whole mesh (a temporary region).
pub fn outer_faces_of_mesh(m: &impl AsMeshData) -> Region {
  let m = &*m.view();
  outer_faces(m, ALL).expect("the entire domain is a region of convexes")
}

/// Faces shared by two convexes of r, each once (port of inner_faces_of_mesh).
pub fn inner_faces(m: &impl AsMeshData, r: impl Into<RegionArg>) -> Result<Region> {
  let m = &*m.view();
  let r = &*r.into().content(m)?;
  only_convexes(r)?;
  let mut out = RegionSet::new();
  let mut visited = std::collections::BTreeSet::new();
  for cv in r.convexes() {
    let nbf = m.structure_of_convex(cv).nb_faces();
    if !(0..nbf).any(|f| {
      m.neighbors_of_convex(cv, f)
        .iter()
        .any(|n| visited.contains(n))
    }) {
      for f in 0..nbf {
        if m.neighbor_of_convex(cv, f).is_some_and(|c2| r.is_in(c2)) {
          out.add_face(cv, f);
        }
      }
      visited.insert(cv);
    }
  }
  for cv in r.convexes() {
    if visited.contains(&cv) {
      continue;
    }
    for f in 0..m.structure_of_convex(cv).nb_faces() {
      let mut ok = false;
      for n in m.neighbors_of_convex(cv, f) {
        if visited.contains(&n) {
          ok = false;
          break;
        }
        ok |= r.is_in(n);
      }
      if ok {
        out.add_face(cv, f);
      }
    }
    visited.insert(cv);
  }
  Region::temporary(m, out)
}

/// Faces of r whose unit outward normal makes an angle <= angle with v at
/// one geometric node of the face at least (GetFEM: at the mean of the
/// nodes, which can leave out a curved face facing v in part).
pub fn select_faces_of_normal(
  m: &impl AsMeshData,
  r: impl Into<RegionArg>,
  v: &[f64],
  angle: f64,
) -> Result<Region> {
  let m = &*m.view();
  if v.len() != m.dim() {
    return Err(Error::InvalidArgument(
      "direction of wrong dimension".into(),
    ));
  }
  let nv = v.iter().map(|x| x * x).sum::<f64>().sqrt();
  let threshold = nv * angle.cos();
  let r = &*r.into().content(m)?;
  let mut out = RegionSet::new();
  for (cv, f) in r.iter() {
    if let Some(f) = f {
      let gt = m.trans_of_convex(cv);
      for &i in gt.structure().ind_points_of_face(f) {
        let n = m.normal_of_face_of_convex(cv, f, &gt.geometric_nodes()[i])?;
        let nn = n.iter().map(|x| x * x).sum::<f64>().sqrt();
        if v.iter().zip(&n).map(|(a, b)| a * b).sum::<f64>() / nn - threshold >= -1e-8 {
          out.add_face(cv, f);
          break;
        }
      }
    }
  }
  Region::temporary(m, out)
}

/// Relative box tolerance: 1e-10 times the mesh bounding box extent.
fn box_tol(m: &MeshData) -> f64 {
  m.bounding_box().map_or(0.0, |(lo, hi)| {
    lo.iter().zip(&hi).map(|(a, b)| b - a).fold(0.0, f64::max)
  }) * 1e-10
}

fn in_box(p: &[f64], lo: &[f64], hi: &[f64], tol: f64) -> bool {
  p.iter()
    .zip(lo.iter().zip(hi))
    .all(|(x, (a, b))| *x >= a - tol && *x <= b + tol)
}

/// Faces of r with all nodes in the box [lo, hi] (tolerance relative to the mesh).
pub fn select_faces_in_box(
  m: &impl AsMeshData,
  r: impl Into<RegionArg>,
  lo: &[f64],
  hi: &[f64],
) -> Result<Region> {
  let m = &*m.view();
  if lo.len() != m.dim() || hi.len() != m.dim() {
    return Err(Error::InvalidArgument("box of wrong dimension".into()));
  }
  let tol = box_tol(m);
  let r = &*r.into().content(m)?;
  let mut out = RegionSet::new();
  for (cv, f) in r.iter() {
    if let Some(f) = f
      && m
        .ind_points_of_face_of_convex(cv, f)
        .iter()
        .all(|&p| in_box(m.point(p), lo, hi, tol))
    {
      out.add_face(cv, f);
    }
  }
  Region::temporary(m, out)
}

/// Convexes of r with all nodes in the box [lo, hi].
pub fn select_convexes_in_box(
  m: &impl AsMeshData,
  r: impl Into<RegionArg>,
  lo: &[f64],
  hi: &[f64],
) -> Result<Region> {
  let m = &*m.view();
  if lo.len() != m.dim() || hi.len() != m.dim() {
    return Err(Error::InvalidArgument("box of wrong dimension".into()));
  }
  let tol = box_tol(m);
  let r = &*r.into().content(m)?;
  let mut out = RegionSet::new();
  for cv in r.convexes() {
    if m
      .ind_points_of_convex(cv)
      .iter()
      .all(|&p| in_box(m.point(p), lo, hi, tol))
    {
      out.add(cv);
    }
  }
  Region::temporary(m, out)
}
