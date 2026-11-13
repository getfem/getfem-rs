//! Reference convexes: structure + nodes + face normals (port of
//! bgeot_convex_ref).

use std::sync::Arc;

use crate::convex_structure::{self as cs, PCvs, simplex_lattice};
use crate::intern::Interner;
use crate::{Error, Result};
use getfem_linalg::vect_sp;

pub type PRef = Arc<ConvexRef>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RefKey {
  Simplex { n: usize, k: usize },
  Product(Box<RefKey>, Box<RefKey>),
  PyramidQK(usize),
  Q2Incomplete(usize),
  PyramidQ2Incomplete,
  PrismIncompleteP2,
  Equilateral(usize),
}

enum Kind {
  Simplex,
  Product(PRef, PRef),
  Pyramid,
  /// is_in, is_in_face, project_into from the basic convex
  Delegate,
  Equilateral {
    r_inscr: f64,
  },
}

pub struct ConvexRef {
  key: RefKey,
  kind: Kind,
  cvs: PCvs,
  points: Vec<Vec<f64>>,
  normals: Vec<Vec<f64>>,
  basic: Option<PRef>,
}

impl std::fmt::Debug for ConvexRef {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "ConvexRef({:?})", self.key)
  }
}

impl PartialEq for ConvexRef {
  fn eq(&self, o: &Self) -> bool {
    self.key == o.key
  }
}

impl ConvexRef {
  pub fn key(&self) -> &RefKey {
    &self.key
  }

  pub fn structure(&self) -> &PCvs {
    &self.cvs
  }

  pub fn dim(&self) -> usize {
    self.cvs.dim()
  }

  pub fn nb_points(&self) -> usize {
    self.points.len()
  }

  pub fn points(&self) -> &[Vec<f64>] {
    &self.points
  }

  /// Outward unit normals of the faces.
  pub fn normals(&self) -> &[Vec<f64>] {
    &self.normals
  }

  pub fn is_basic(&self) -> bool {
    self.basic.is_none()
  }

  /// Associated order-1 reference convex.
  pub fn basic(self: &Arc<Self>) -> PRef {
    self.basic.clone().unwrap_or_else(|| self.clone())
  }

  fn check(&self, xref: &[f64]) -> Result<()> {
    if xref.len() != self.dim() {
      return Err(Error::InvalidArgument(format!(
        "{:?}: point of dim {}, expected {}",
        self.key,
        xref.len(),
        self.dim()
      )));
    }
    Ok(())
  }

  /// Signed "distance" to the convex, <= 0 inside (not a true distance).
  pub fn is_in(&self, xref: &[f64]) -> Result<f64> {
    self.check(xref)?;
    Ok(match &self.kind {
      Kind::Simplex => {
        let r = if xref.is_empty() {
          0.0
        } else {
          xref.iter().fold(f64::NEG_INFINITY, |r, &x| r.max(-x))
        };
        r.max(xref.iter().sum::<f64>() - 1.0)
      }
      Kind::Product(a, b) => {
        let n1 = a.dim();
        a.is_in(&xref[..n1])?.max(b.is_in(&xref[n1..])?)
      }
      Kind::Pyramid => {
        let mut r = self.is_in_face(0, xref)?;
        for f in 1..5 {
          r = r.max(self.is_in_face(f, xref)?);
        }
        r
      }
      Kind::Delegate => self.basic.as_ref().unwrap().is_in(xref)?,
      Kind::Equilateral { .. } => (0..self.normals.len())
        .map(|f| self.face_value(f, xref))
        .fold(f64::NEG_INFINITY, f64::max),
    })
  }

  fn face_value(&self, f: usize, xref: &[f64]) -> f64 {
    let x0ref = if f > 0 {
      &self.points[f - 1]
    } else {
      self.points.last().unwrap()
    };
    let d: Vec<f64> = xref.iter().zip(x0ref).map(|(a, b)| a - b).collect();
    vect_sp(&d, &self.normals[f])
  }

  /// Signed "distance" to face f (<= 0 on the inner side).
  pub fn is_in_face(&self, f: usize, xref: &[f64]) -> Result<f64> {
    self.check(xref)?;
    if f >= self.cvs.nb_faces() {
      return Err(Error::InvalidArgument(format!(
        "{:?}: no face {f}",
        self.key
      )));
    }
    Ok(match &self.kind {
      Kind::Simplex => {
        if f > 0 {
          -xref[f - 1]
        } else {
          (xref.iter().sum::<f64>() - 1.0) / (xref.len() as f64).sqrt()
        }
      }
      Kind::Product(a, b) => {
        let n1 = a.dim();
        let nf1 = a.structure().nb_faces();
        if f < nf1 {
          a.is_in_face(f, &xref[..n1])?
        } else {
          b.is_in_face(f - nf1, &xref[n1..])?
        }
      }
      Kind::Pyramid => {
        if f == 0 {
          -xref[2]
        } else {
          vect_sp(&self.normals[f], xref) - 0.5f64.sqrt()
        }
      }
      Kind::Delegate => self.basic.as_ref().unwrap().is_in_face(f, xref)?,
      Kind::Equilateral { .. } => self.face_value(f, xref),
    })
  }

  /// Project a point into the convex (not an orthogonal projection).
  pub fn project_into(&self, xref: &mut [f64]) -> Result<()> {
    self.check(xref)?;
    if let Some(b) = &self.basic {
      return b.project_into(xref);
    }
    match &self.kind {
      Kind::Simplex => {
        let s: f64 = xref.iter().sum();
        if s > 1.0 {
          xref.iter_mut().for_each(|x| *x /= s);
        }
        xref.iter_mut().for_each(|x| *x = x.clamp(0.0, 1.0));
      }
      Kind::Product(a, b) => {
        let n1 = a.dim();
        let (x1ref, x2ref) = xref.split_at_mut(n1);
        a.project_into(x1ref)?;
        b.project_into(x2ref)?;
      }
      Kind::Pyramid => {
        xref[2] = xref[2].max(0.0);
        for f in 1..5 {
          let reldist = vect_sp(&self.normals[f], xref) * 2f64.sqrt();
          if reldist > 1.0 {
            xref.iter_mut().for_each(|x| *x /= reldist);
          }
        }
      }
      Kind::Delegate => unreachable!("delegating convexes have a basic convex"),
      Kind::Equilateral { r_inscr } => {
        let n = self.dim() as f64;
        let mut gref = vec![0.0; self.dim()];
        for noderef in &self.points {
          gref.iter_mut().zip(noderef).for_each(|(a, b)| *a += b);
        }
        gref.iter_mut().for_each(|a| *a /= n + 1.0);
        for nf in &self.normals {
          let d: Vec<f64> = xref.iter().zip(&gref).map(|(a, b)| a - b).collect();
          let r = vect_sp(&d, nf);
          if r > *r_inscr {
            xref
              .iter_mut()
              .zip(&gref)
              .zip(&d)
              .for_each(|((p, gi), di)| *p = gi + r_inscr / r * di);
          }
        }
      }
    }
    Ok(())
  }
}

static CACHE: Interner<RefKey, ConvexRef> = Interner::new();

/// k/K by repeated addition, bit-identical to GetFEM's node generation.
fn lattice_coord(c: usize, k: usize) -> f64 {
  (0..c).fold(0.0, |a, _| a + 1.0 / k as f64)
}

pub fn simplex_of_reference(n: usize) -> PRef {
  simplex_of_reference_k(n, 1)
}

pub fn simplex_of_reference_k(n: usize, k: usize) -> PRef {
  CACHE.get_or(&RefKey::Simplex { n, k }, || {
    let points = if k == 0 {
      vec![vec![1.0 / (n + 1) as f64; n]]
    } else {
      simplex_lattice(n, k)
        .iter()
        .map(|c| c.iter().map(|&ci| lattice_coord(ci, k)).collect())
        .collect()
    };
    let mut normals = vec![vec![0.0; n]; n + 1];
    for (i, nm) in normals.iter_mut().enumerate().skip(1) {
      nm[i - 1] = -1.0;
    }
    if n > 0 {
      normals[0] = vec![1.0 / (n as f64).sqrt(); n];
    }
    if n == 0 {
      normals.clear();
    }
    ConvexRef {
      key: RefKey::Simplex { n, k },
      kind: Kind::Simplex,
      cvs: cs::simplex_structure_k(n, k),
      points,
      normals,
      basic: (k != 1 && n != 0).then(|| simplex_of_reference(n)),
    }
  })
}

pub fn convex_ref_product(a: &PRef, b: &PRef) -> PRef {
  let key = RefKey::Product(Box::new(a.key.clone()), Box::new(b.key.clone()));
  CACHE.get_or(&key.clone(), || {
    let cvs = cs::convex_product_structure(a.structure(), b.structure());
    let (n1, n) = (a.dim(), a.dim() + b.dim());
    let mut points = Vec::with_capacity(a.nb_points() * b.nb_points());
    for p2ref in &b.points {
      for p1ref in &a.points {
        points.push([p1ref.as_slice(), p2ref.as_slice()].concat());
      }
    }
    let mut normals = vec![];
    for nm in &a.normals {
      let mut v = vec![0.0; n];
      v[..n1].copy_from_slice(nm);
      normals.push(v);
    }
    for nm in &b.normals {
      let mut v = vec![0.0; n];
      v[n1..].copy_from_slice(nm);
      normals.push(v);
    }
    let basic =
      (!a.is_basic() || !b.is_basic()).then(|| convex_ref_product(&a.basic(), &b.basic()));
    ConvexRef {
      key,
      kind: Kind::Product(a.clone(), b.clone()),
      cvs,
      points,
      normals,
      basic,
    }
  })
}

pub fn parallelepiped_of_reference(n: usize, k: usize) -> PRef {
  if n <= 1 {
    return simplex_of_reference_k(n, k);
  }
  convex_ref_product(
    &parallelepiped_of_reference(n - 1, k),
    &simplex_of_reference_k(1, k),
  )
}

pub fn prism_of_reference(n: usize) -> PRef {
  if n <= 2 {
    return parallelepiped_of_reference(n, 1);
  }
  convex_ref_product(&simplex_of_reference(n - 1), &simplex_of_reference(1))
}

fn unit_normals(v: &[[f64; 3]]) -> Vec<Vec<f64>> {
  v.iter()
    .map(|n| {
      let l = vect_sp(n, n).sqrt();
      n.iter().map(|x| x / l).collect()
    })
    .collect()
}

fn pts(v: &[[f64; 3]], dim: usize) -> Vec<Vec<f64>> {
  v.iter().map(|ptref| ptref[..dim].to_vec()).collect()
}

pub fn pyramid_qk_of_reference(k: usize) -> Result<PRef> {
  let cvs = cs::pyramid_qk_structure(k)?;
  Ok(CACHE.get_or(&RefKey::PyramidQK(k), || {
    let normals = unit_normals(&[
      [0., 0., -1.],
      [0., -1., 1.],
      [1., 0., 1.],
      [0., 1., 1.],
      [-1., 0., 1.],
    ]);
    let points = if k == 1 {
      pts(
        &[
          [-1., -1., 0.],
          [1., -1., 0.],
          [-1., 1., 0.],
          [1., 1., 0.],
          [0., 0., 1.],
        ],
        3,
      )
    } else {
      pts(
        &[
          [-1., -1., 0.],
          [0., -1., 0.],
          [1., -1., 0.],
          [-1., 0., 0.],
          [0., 0., 0.],
          [1., 0., 0.],
          [-1., 1., 0.],
          [0., 1., 0.],
          [1., 1., 0.],
          [-0.5, -0.5, 0.5],
          [0.5, -0.5, 0.5],
          [-0.5, 0.5, 0.5],
          [0.5, 0.5, 0.5],
          [0., 0., 1.],
        ],
        3,
      )
    };
    let basic = (k != 1).then(|| pyramid_qk_of_reference(1).unwrap());
    ConvexRef {
      key: RefKey::PyramidQK(k),
      kind: Kind::Pyramid,
      cvs,
      points,
      normals,
      basic,
    }
  }))
}

pub fn q2_incomplete_of_reference(n: usize) -> Result<PRef> {
  let cvs = cs::q2_incomplete_structure(n)?;
  Ok(CACHE.get_or(&RefKey::Q2Incomplete(n), || {
    let p3 = [
      [0., 0., 0.],
      [0.5, 0., 0.],
      [1., 0., 0.],
      [0., 0.5, 0.],
      [1., 0.5, 0.],
      [0., 1., 0.],
      [0.5, 1., 0.],
      [1., 1., 0.],
      [0., 0., 0.5],
      [1., 0., 0.5],
      [0., 1., 0.5],
      [1., 1., 0.5],
      [0., 0., 1.],
      [0.5, 0., 1.],
      [1., 0., 1.],
      [0., 0.5, 1.],
      [1., 0.5, 1.],
      [0., 1., 1.],
      [0.5, 1., 1.],
      [1., 1., 1.],
    ];
    let points = pts(&p3[..if n == 2 { 8 } else { 20 }], n);
    let basic = parallelepiped_of_reference(n, 1);
    let normals = basic.normals.clone();
    ConvexRef {
      key: RefKey::Q2Incomplete(n),
      kind: Kind::Delegate,
      cvs,
      points,
      normals,
      basic: Some(basic),
    }
  }))
}

pub fn pyramid_q2_incomplete_of_reference() -> PRef {
  CACHE.get_or(&RefKey::PyramidQ2Incomplete, || {
    let points = pts(
      &[
        [-1., -1., 0.],
        [0., -1., 0.],
        [1., -1., 0.],
        [-1., 0., 0.],
        [1., 0., 0.],
        [-1., 1., 0.],
        [0., 1., 0.],
        [1., 1., 0.],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [-0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0., 0., 1.],
      ],
      3,
    );
    let basic = pyramid_qk_of_reference(1).unwrap();
    ConvexRef {
      key: RefKey::PyramidQ2Incomplete,
      kind: Kind::Delegate,
      cvs: cs::pyramid_q2_incomplete_structure(),
      points,
      normals: basic.normals.clone(),
      basic: Some(basic),
    }
  })
}

pub fn prism_incomplete_p2_of_reference() -> PRef {
  CACHE.get_or(&RefKey::PrismIncompleteP2, || {
    let points = pts(
      &[
        [0., 0., 0.],
        [0.5, 0., 0.],
        [1., 0., 0.],
        [0., 0.5, 0.],
        [0.5, 0.5, 0.],
        [0., 1., 0.],
        [0., 0., 0.5],
        [1., 0., 0.5],
        [0., 1., 0.5],
        [0., 0., 1.],
        [0.5, 0., 1.],
        [1., 0., 1.],
        [0., 0.5, 1.],
        [0.5, 0.5, 1.],
        [0., 1., 1.],
      ],
      3,
    );
    let basic = prism_of_reference(3);
    ConvexRef {
      key: RefKey::PrismIncompleteP2,
      kind: Kind::Delegate,
      cvs: cs::prism_incomplete_p2_structure(),
      points,
      normals: basic.normals.clone(),
      basic: Some(basic),
    }
  })
}

/// Regular simplex with unit edges; used for element quality.
pub fn equilateral_simplex_of_reference(n: usize) -> PRef {
  if n <= 1 {
    return simplex_of_reference(n);
  }
  CACHE.get_or(&RefKey::Equilateral(n), || {
    let prev = equilateral_simplex_of_reference(n - 1);
    let nf = n as f64;
    let mut points: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    let mut gref = vec![0.0; n];
    for i in 0..=n {
      let ptref = if i != n {
        let mut ptref = prev.points[i].clone();
        ptref.push(0.0);
        ptref
      } else {
        let mut ptref: Vec<f64> = gref.iter().map(|x| x / nf).collect();
        ptref[n - 1] = (1.0 - vect_sp(&ptref, &ptref)).sqrt();
        ptref
      };
      gref.iter_mut().zip(&ptref).for_each(|(a, b)| *a += b);
      points.push(ptref);
    }
    gref.iter_mut().for_each(|a| *a /= nf + 1.0);
    let normals = points
      .iter()
      .map(|ptref| {
        let d: Vec<f64> = gref.iter().zip(ptref).map(|(a, b)| a - b).collect();
        let l = vect_sp(&d, &d).sqrt();
        d.iter().map(|x| x / l).collect()
      })
      .collect();
    ConvexRef {
      key: RefKey::Equilateral(n),
      kind: Kind::Equilateral {
        r_inscr: 1.0 / (2.0 * nf * (nf + 1.0)).sqrt(),
      },
      cvs: cs::simplex_structure(n),
      points,
      normals,
      basic: None,
    }
  })
}

#[cfg(test)]
mod tests;
