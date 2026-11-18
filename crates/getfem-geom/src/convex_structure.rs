//! Convex structures: topology of reference elements (GetFEM's
//! bgeot_convex_structure). Face and point numbering identical to GetFEM.

use std::sync::Arc;

use crate::intern::Interner;
use crate::{Error, Result};

pub type PCvs = Arc<ConvexStructure>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum CvsKey {
  Simplex { n: usize, k: usize },
  Polygon(usize),
  Product(Box<CvsKey>, Box<CvsKey>),
  Q2Incomplete(usize),
  PyramidQK(usize),
  PyramidQ2Incomplete,
  PrismIncompleteP2,
}

pub struct ConvexStructure {
  key: CvsKey,
  dim: usize,
  nb_points: usize,
  faces: Vec<Vec<usize>>,
  faces_struct: Vec<Option<PCvs>>,
  dir_points: Vec<usize>,
  basic: Option<PCvs>,
  prod: Option<(PCvs, PCvs)>,
}

impl std::fmt::Debug for ConvexStructure {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "ConvexStructure({:?})", self.key)
  }
}

impl PartialEq for ConvexStructure {
  fn eq(&self, o: &Self) -> bool {
    self.key == o.key
  }
}

impl ConvexStructure {
  pub fn key(&self) -> &CvsKey {
    &self.key
  }

  pub fn dim(&self) -> usize {
    self.dim
  }

  pub fn nb_points(&self) -> usize {
    self.nb_points
  }

  pub fn nb_faces(&self) -> usize {
    self.faces.len()
  }

  pub fn ind_points_of_face(&self, f: usize) -> &[usize] {
    &self.faces[f]
  }

  pub fn nb_points_of_face(&self, f: usize) -> usize {
    self.faces[f].len()
  }

  pub fn face_structure(&self, f: usize) -> Option<&PCvs> {
    self.faces_struct[f].as_ref()
  }

  /// Points giving the directions of the element (vertices 0, e_1, ..).
  pub fn ind_dir_points(&self) -> &[usize] {
    &self.dir_points
  }

  pub fn ind_dir_points_of_face(&self, f: usize) -> Vec<usize> {
    match &self.faces_struct[f] {
      Some(fs) => fs.dir_points.iter().map(|&i| self.faces[f][i]).collect(),
      None => vec![],
    }
  }

  /// Points common to all faces of the list.
  pub fn ind_common_points_of_faces(&self, fs: &[usize]) -> Vec<usize> {
    (0..self.nb_points)
      .filter(|&i| fs.iter().all(|&f| self.faces[f].contains(&i)))
      .collect()
  }

  pub fn is_basic(&self) -> bool {
    self.basic.is_none()
  }

  /// Linear (order 1) structure.
  pub fn basic(self: &Arc<Self>) -> PCvs {
    self.basic.clone().unwrap_or_else(|| self.clone())
  }

  pub fn product_factors(&self) -> Option<&(PCvs, PCvs)> {
    self.prod.as_ref()
  }
}

static CACHE: Interner<CvsKey, ConvexStructure> = Interner::new();

/// Lattice points {c in N^n : |c| <= k}, first coordinate fastest,
/// in GetFEM order.
pub(crate) fn simplex_lattice(n: usize, k: usize) -> Vec<Vec<usize>> {
  let mut out = vec![vec![0; n]];
  if n == 0 || k == 0 {
    return out;
  }
  let mut c = vec![0; n];
  let mut sum = 0;
  loop {
    let mut l = 0;
    c[l] += 1;
    sum += 1;
    while sum > k {
      sum -= c[l];
      c[l] = 0;
      l += 1;
      if l == n {
        return out;
      }
      c[l] += 1;
      sum += 1;
    }
    out.push(c.clone());
  }
}

pub fn simplex_structure(n: usize) -> PCvs {
  simplex_structure_k(n, 1)
}

pub fn simplex_structure_k(n: usize, k: usize) -> PCvs {
  let k = if n == 0 { 1 } else { k };
  CACHE.get_or(&CvsKey::Simplex { n, k }, || build_simplex(n, k))
}

fn build_simplex(n: usize, k: usize) -> ConvexStructure {
  let nbf = if n > 0 { n + 1 } else { 0 };
  let lat = simplex_lattice(n, k);
  let mut faces = vec![vec![]; nbf];
  let mut dir_points = vec![0; n + 1];
  if k > 0 {
    let mut pd = 0;
    for (r, c) in lat.iter().enumerate() {
      let sum: usize = c.iter().sum();
      if sum == k && n > 0 {
        faces[0].push(r);
      }
      for l in 1..=n {
        if c[l - 1] == 0 {
          faces[l].push(r);
        }
      }
      if r == 0 || (sum == k && c.contains(&k)) {
        dir_points[pd] = r;
        pd += 1;
      }
    }
  }
  let fs = (k > 0 && n > 0).then(|| simplex_structure_k(n - 1, k));
  ConvexStructure {
    key: CvsKey::Simplex { n, k },
    dim: n,
    nb_points: lat.len(),
    faces,
    faces_struct: vec![fs; nbf],
    dir_points,
    basic: (k != 1).then(|| simplex_structure(n)),
    prod: None,
  }
}

pub fn polygon_structure(nbt: usize) -> PCvs {
  if nbt <= 1 {
    return simplex_structure(0);
  }
  if nbt <= 3 {
    return simplex_structure(nbt - 1);
  }
  CACHE.get_or(&CvsKey::Polygon(nbt), || ConvexStructure {
    key: CvsKey::Polygon(nbt),
    dim: 2,
    nb_points: nbt,
    faces: (0..nbt).map(|i| vec![i, (i + 1) % nbt]).collect(),
    faces_struct: vec![Some(simplex_structure(1)); nbt],
    dir_points: vec![0, 1, nbt - 1],
    basic: None,
    prod: None,
  })
}

pub fn convex_product_structure(a: &PCvs, b: &PCvs) -> PCvs {
  let key = CvsKey::Product(Box::new(a.key.clone()), Box::new(b.key.clone()));
  CACHE.get_or(&key.clone(), || build_product(key, a, b))
}

fn build_product(key: CvsKey, a: &PCvs, b: &PCvs) -> ConvexStructure {
  let (na, nb) = (a.nb_points, b.nb_points);
  let basic =
    (!a.is_basic() || !b.is_basic()).then(|| convex_product_structure(&a.basic(), &b.basic()));
  let mut dir_points = vec![];
  if !a.dir_points.is_empty() && !b.dir_points.is_empty() {
    let (a0, b0) = (a.dir_points[0], b.dir_points[0]);
    dir_points.extend(a.dir_points.iter().map(|&i| i + b0 * na));
    dir_points.extend(b.dir_points[1..].iter().map(|&i| a0 + i * na));
  }
  let mut faces = vec![];
  let mut faces_struct = vec![];
  for f in 0..a.nb_faces() {
    let fp = &a.faces[f];
    faces_struct.push(if fp.len() == 1 {
      Some(b.clone())
    } else {
      a.faces_struct[f]
        .as_ref()
        .map(|fs| convex_product_structure(fs, b))
    });
    let mut v = vec![0; fp.len() * nb];
    for (j, &p) in fp.iter().enumerate() {
      for l in 0..nb {
        v[l * fp.len() + j] = p + l * na;
      }
    }
    faces.push(v);
  }
  for f in 0..b.nb_faces() {
    let fp = &b.faces[f];
    faces_struct.push(if fp.len() == 1 {
      Some(a.clone())
    } else {
      b.faces_struct[f]
        .as_ref()
        .map(|fs| convex_product_structure(a, fs))
    });
    let mut v = vec![0; fp.len() * na];
    for (j, &p) in fp.iter().enumerate() {
      for l in 0..na {
        v[j * na + l] = l + p * na;
      }
    }
    faces.push(v);
  }
  ConvexStructure {
    key,
    dim: a.dim + b.dim,
    nb_points: na * nb,
    faces,
    faces_struct,
    dir_points,
    basic,
    prod: Some((a.clone(), b.clone())),
  }
}

pub fn parallelepiped_structure(n: usize, k: usize) -> PCvs {
  if n <= 1 {
    return simplex_structure_k(n, k);
  }
  convex_product_structure(
    &parallelepiped_structure(n - 1, k),
    &simplex_structure_k(1, k),
  )
}

pub fn prism_p1_structure(n: usize) -> PCvs {
  convex_product_structure(&simplex_structure(n - 1), &simplex_structure(1))
}

fn fixed(
  key: CvsKey,
  dim: usize,
  nb_points: usize,
  faces: Vec<Vec<usize>>,
  fs: Vec<PCvs>,
  dir: Vec<usize>,
  basic: Option<PCvs>,
) -> ConvexStructure {
  ConvexStructure {
    key,
    dim,
    nb_points,
    faces,
    faces_struct: fs.into_iter().map(Some).collect(),
    dir_points: dir,
    basic,
    prod: None,
  }
}

pub fn q2_incomplete_structure(n: usize) -> Result<PCvs> {
  if n != 2 && n != 3 {
    return Err(Error::InvalidArgument(format!(
      "Q2 incomplete structure: dim {n}, expected 2 or 3"
    )));
  }
  let key = CvsKey::Q2Incomplete(n);
  Ok(CACHE.get_or(&key.clone(), || {
    let basic = Some(parallelepiped_structure(n, 1));
    if n == 2 {
      let f = vec![vec![2, 4, 7], vec![0, 3, 5], vec![5, 6, 7], vec![0, 1, 2]];
      fixed(
        key,
        2,
        8,
        f,
        vec![simplex_structure_k(1, 2); 4],
        vec![0, 2, 5],
        basic,
      )
    } else {
      let f = vec![
        vec![2, 4, 7, 9, 11, 14, 16, 19],
        vec![0, 3, 5, 8, 10, 12, 15, 17],
        vec![5, 6, 7, 10, 11, 17, 18, 19],
        vec![0, 1, 2, 8, 9, 12, 13, 14],
        vec![12, 13, 14, 15, 16, 17, 18, 19],
        vec![0, 1, 2, 3, 4, 5, 6, 7],
      ];
      let fs = q2_incomplete_structure(2).unwrap();
      fixed(key, 3, 20, f, vec![fs; 6], vec![0, 2, 5, 12], basic)
    }
  }))
}

pub fn pyramid_qk_structure(k: usize) -> Result<PCvs> {
  if k != 1 && k != 2 {
    return Err(Error::InvalidArgument(format!(
      "pyramid structure: degree {k}, expected 1 or 2"
    )));
  }
  let key = CvsKey::PyramidQK(k);
  Ok(CACHE.get_or(&key.clone(), || {
    let mut fs = vec![parallelepiped_structure(2, k)];
    fs.extend(std::iter::repeat_n(simplex_structure_k(2, k), 4));
    if k == 1 {
      let f = vec![
        vec![0, 1, 2, 3],
        vec![0, 1, 4],
        vec![1, 3, 4],
        vec![3, 2, 4],
        vec![2, 0, 4],
      ];
      fixed(key, 3, 5, f, fs, vec![0, 1, 2, 4], None)
    } else {
      let f = vec![
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8],
        vec![0, 1, 2, 9, 10, 13],
        vec![2, 5, 8, 10, 12, 13],
        vec![8, 7, 6, 12, 11, 13],
        vec![6, 3, 0, 11, 9, 13],
      ];
      fixed(
        key,
        3,
        14,
        f,
        fs,
        vec![0, 2, 6, 13],
        Some(pyramid_qk_structure(1).unwrap()),
      )
    }
  }))
}

pub fn pyramid_q2_incomplete_structure() -> PCvs {
  let key = CvsKey::PyramidQ2Incomplete;
  CACHE.get_or(&key.clone(), || {
    let f = vec![
      vec![0, 1, 2, 3, 4, 5, 6, 7],
      vec![0, 1, 2, 8, 9, 12],
      vec![2, 4, 7, 9, 11, 12],
      vec![7, 6, 5, 11, 10, 12],
      vec![5, 3, 0, 10, 8, 12],
    ];
    let mut fs = vec![q2_incomplete_structure(2).unwrap()];
    fs.extend(std::iter::repeat_n(simplex_structure_k(2, 2), 4));
    fixed(
      key,
      3,
      13,
      f,
      fs,
      vec![0, 2, 5, 12],
      Some(pyramid_qk_structure(1).unwrap()),
    )
  })
}

pub fn prism_incomplete_p2_structure() -> PCvs {
  let key = CvsKey::PrismIncompleteP2;
  CACHE.get_or(&key.clone(), || {
    let f = vec![
      vec![2, 4, 5, 7, 8, 11, 13, 14],
      vec![0, 3, 5, 6, 8, 9, 12, 14],
      vec![0, 1, 2, 6, 7, 9, 10, 11],
      vec![9, 10, 11, 12, 13, 14],
      vec![0, 1, 2, 3, 4, 5],
    ];
    let q = q2_incomplete_structure(2).unwrap();
    let t = simplex_structure_k(2, 2);
    let fs = vec![q.clone(), q.clone(), q, t.clone(), t];
    fixed(
      key,
      3,
      15,
      f,
      fs,
      vec![0, 2, 5, 9],
      Some(prism_p1_structure(3)),
    )
  })
}

#[cfg(test)]
mod tests;
