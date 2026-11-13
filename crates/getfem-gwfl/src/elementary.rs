//! Elementary transformations (GetFEM virtual_elementary_transformation):
//! a matrix per element mapping the element DOFs of a variable to those of
//! a target space; Elementary_transformation(u, name, v) is the field of the
//! space of v with these DOFs. Built-ins of getfem_linearized_plates.cc.

use getfem_fem::fem::{PFem, fem_descriptor, real_base_values};
use getfem_fem::integration::classical_approx_im;
use getfem_geom::convex_structure::{parallelepiped_structure, simplex_structure};
use getfem_geom::geotrans::PGeoTrans;
use getfem_linalg::tensor::{Lu, Mat};

use crate::{Error, Result};

/// The finite element of a variable on one element.
pub struct ElemFem<'a> {
  pub pf: &'a PFem,
  pub qdim: usize,
  pub gt: &'a PGeoTrans,
  /// nodes of the element
  pub g: &'a Mat,
  pub cv: usize,
}

/// An elementary transformation.
pub trait ElementaryTransformation: Send + Sync + std::fmt::Debug {
  /// M (nb DOFs of dest x nb DOFs of src, vector DOFs i*qdim + c): the
  /// element DOFs of the result from those of the source
  fn give_transformation(&self, src: &ElemFem, dest: &ElemFem) -> Result<Mat>;
}

/// Vector basis values at xref: tv[k][a], k = i*q + c for scalar elements
/// with q = n copies (GetFEM vectorize_base_tensor).
fn vector_basis(
  pf: &PFem,
  gt: &PGeoTrans,
  g: &Mat,
  xref: &[f64],
  n: usize,
) -> Result<Vec<Vec<f64>>> {
  let (val, _, _) = real_base_values(pf, gt, g, xref, 0)?;
  let t = pf.target_dim();
  let nd = val.len() / t;
  let q = n / t;
  let mut tv = vec![vec![0.0; n]; nd * q];
  for i in 0..nd {
    for c in 0..q {
      for a in 0..t {
        tv[i * q + c][c * t + a] = val[i + nd * a];
      }
    }
  }
  Ok(tv)
}

/// L2 projection of the space of pf (qdim n) onto the space of pf2 on the
/// element, as a matrix of the DOFs of pf: M1^-1 B M2^-1 B^T (GetFEM, with
/// the weights of the reference element, not |J|). swap exchanges the two
/// components of pf2 (rotated RT0).
fn projection(src: &ElemFem, pf2: &PFem, n: usize, swap: bool) -> Result<Mat> {
  let pf1 = src.pf;
  let (d1, d2) = (pf1.estimated_degree(), pf2.estimated_degree());
  // GetFEM: d1 + d2, exact on parallelepipeds (rules per direction) but
  // not for the mass matrices on simplices
  let degree = if simplex(src, pf1.dim(), "projection")? {
    (d1 + d2).max(2 * d1).max(2 * d2)
  } else {
    d1 + d2
  };
  let pim = classical_approx_im(src.gt, degree)?;
  let ai = pim.approx()?;
  let mut m1: Option<Mat> = None;
  let (mut m2, mut b): (Option<Mat>, Option<Mat>) = (None, None);
  for i in 0..ai.nb_points_on_convex() {
    let (xref, w) = (ai.point(i), ai.coeff(i));
    let t1 = vector_basis(pf1, src.gt, src.g, xref, n)?;
    let mut t2 = vector_basis(pf2, src.gt, src.g, xref, n)?;
    if swap {
      t2.iter_mut().for_each(|v| v.swap(0, 1));
    }
    let (n1, n2) = (t1.len(), t2.len());
    let m1 = m1.get_or_insert_with(|| Mat::zeros(&[n1, n1]));
    let m2 = m2.get_or_insert_with(|| Mat::zeros(&[n2, n2]));
    let b = b.get_or_insert_with(|| Mat::zeros(&[n1, n2]));
    let dot = |a: &[f64], c: &[f64]| a.iter().zip(c).map(|(x, y)| x * y).sum::<f64>();
    for r in 0..n1 {
      for c in 0..n1 {
        m1[(r, c)] += w * dot(&t1[r], &t1[c]);
      }
      for c in 0..n2 {
        b[(r, c)] += w * dot(&t1[r], &t2[c]);
      }
    }
    for r in 0..n2 {
      for c in 0..n2 {
        m2[(r, c)] += w * dot(&t2[r], &t2[c]);
      }
    }
  }
  let (Some(m1), Some(m2), Some(b)) = (m1, m2, b) else {
    return Err(Error::InvalidArgument(
      "projection: no integration point".into(),
    ));
  };
  let inv = |m: &Mat| -> Result<Mat> {
    Ok(
      Lu::new(m)
        .map_err(|e| Error::InvalidArgument(e.to_string()))?
        .inverse(),
    )
  };
  let mut m = inv(&m1)?
    .matmul(&b)
    .matmul(&inv(&m2)?)
    .matmul(&b.transposed());
  m.data_mut().iter_mut().for_each(|v| {
    if v.abs() < 1e-15 {
      *v = 0.0
    }
  });
  Ok(m)
}

/// simplex (true) or parallelepiped (false) element, as GetFEM requires
fn simplex(e: &ElemFem, d: usize, what: &str) -> Result<bool> {
  let s = e.gt.structure().basic();
  if *s == *simplex_structure(d) {
    Ok(true)
  } else if *s == *parallelepiped_structure(d, 1) {
    Ok(false)
  } else {
    Err(Error::InvalidArgument(format!(
      "{what}: element {} is neither a simplex nor a parallelepiped",
      e.cv
    )))
  }
}

/// components of the field: qdim copies of a scalar element, or the
/// components of a vector element (qdim = its dimension)
fn comps(e: &ElemFem) -> usize {
  if e.pf.target_dim() > 1 {
    e.pf.target_dim()
  } else {
    e.qdim
  }
}

fn same_space(src: &ElemFem, dest: &ElemFem, what: &str) -> Result<()> {
  if src.pf.name() != dest.pf.name() || src.qdim != dest.qdim {
    return Err(Error::InvalidArgument(format!(
      "{what}: the target space has to be the space of the variable"
    )));
  }
  Ok(())
}

/// L2 projection onto the rotated RT0 element in 2D (GetFEM
/// _2D_Rotated_RT0_projection_transformation, used by MITC plates).
#[derive(Debug)]
pub struct Rotated2DRt0Projection;

impl ElementaryTransformation for Rotated2DRt0Projection {
  fn give_transformation(&self, src: &ElemFem, dest: &ElemFem) -> Result<Mat> {
    let what = "2D rotated RT0 projection";
    same_space(src, dest, what)?;
    if src.pf.dim() != 2 || comps(src) != 2 {
      return Err(Error::InvalidArgument(format!(
        "{what}: two-dimensional elements and vector fields only"
      )));
    }
    let s = simplex(src, 2, what)?;
    let pf2 = fem_descriptor(if s { "FEM_RT0(2)" } else { "FEM_RT0Q(2)" })?;
    projection(src, &pf2, 2, true)
  }
}

/// L2 projection onto piecewise constants (GetFEM _P0_projection_transformation).
#[derive(Debug)]
pub struct P0Projection;

impl ElementaryTransformation for P0Projection {
  fn give_transformation(&self, src: &ElemFem, dest: &ElemFem) -> Result<Mat> {
    let what = "P0 projection";
    same_space(src, dest, what)?;
    let d = src.pf.dim();
    let s = simplex(src, d, what)?;
    let pf2 = fem_descriptor(&format!("FEM_{}({d},0)", if s { "PK" } else { "QK" }))?;
    projection(src, &pf2, comps(src), false)
  }
}

/// L2 projection of the gradient of the variable onto the target space on
/// each element (true |J| weights): with a P0 target, the element mean of
/// the gradient, for B-bar (mean dilatation) and F-bar formulations. The
/// target space has qdim = qdim(u) * N.
#[derive(Debug)]
pub struct ProjectedGradient;

impl ElementaryTransformation for ProjectedGradient {
  fn give_transformation(&self, src: &ElemFem, dest: &ElemFem) -> Result<Mat> {
    let what = "projected gradient";
    let n = src.g.nrows();
    let (q1, q2) = (comps(src), comps(dest));
    if q2 != q1 * n {
      return Err(Error::InvalidArgument(format!(
        "{what}: the target space has to be of dimension {} (qdim of the variable times {n})",
        q1 * n
      )));
    }
    let degree = 2 * src.pf.estimated_degree().max(dest.pf.estimated_degree());
    let pim = classical_approx_im(src.gt, degree)?;
    let ai = pim.approx()?;
    let (mut m1, mut m2): (Option<Mat>, Option<Mat>) = (None, None);
    for ip in 0..ai.nb_points_on_convex() {
      let xref = ai.point(ip);
      let w = ai.coeff(ip) * src.gt.at(src.g, xref)?.j.abs();
      // gradients of the source, flat [c + q1 r]
      let (val, grad, _) = real_base_values(src.pf, src.gt, src.g, xref, 1)?;
      let t = src.pf.target_dim();
      let nd = val.len() / t;
      let qm = q1 / t;
      let mut g1 = vec![vec![0.0; q1 * n]; nd * qm];
      for i in 0..nd {
        for c in 0..qm {
          for a in 0..t {
            for r in 0..n {
              g1[i * qm + c][c * t + a + q1 * r] = grad[i + nd * (a + t * r)];
            }
          }
        }
      }
      let v2 = vector_basis(dest.pf, dest.gt, dest.g, xref, q2)?;
      let (n1, n2) = (g1.len(), v2.len());
      let m1 = m1.get_or_insert_with(|| Mat::zeros(&[n2, n1]));
      let m2 = m2.get_or_insert_with(|| Mat::zeros(&[n2, n2]));
      let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
      for j in 0..n2 {
        for i in 0..n1 {
          m1[(j, i)] += w * dot(&g1[i], &v2[j]);
        }
        for i in 0..n2 {
          m2[(j, i)] += w * dot(&v2[i], &v2[j]);
        }
      }
    }
    let (Some(m1), Some(m2)) = (m1, m2) else {
      return Err(Error::InvalidArgument(format!(
        "{what}: no integration point"
      )));
    };
    Ok(
      Lu::new(&m2)
        .map_err(|e| Error::InvalidArgument(e.to_string()))?
        .inverse()
        .matmul(&m1),
    )
  }
}
