//! HHO reconstructions and stabilizations as elementary transformations
//! (port of getfem_HHO.cc). The source is a variable on an HHO element
//! (interior DOFs first, then the faces); the reconstructed gradient and
//! value live on the space of the target variable.

use getfem_fem::fem::{PFem, classical_fem, real_base_values};
use getfem_fem::integration::classical_approx_im;
use getfem_geom::geotrans::PGeoTrans;
use getfem_linalg::tensor::{Lu, Mat};

use crate::elementary::{ElemFem, ElementaryTransformation};
use crate::{Error, Result};

/// An integration point of the element: reference point, weight times |J|,
/// and on faces B n_ref (not unit: |B n_ref| |J| is the surface measure).
struct Pt {
  xref: Vec<f64>,
  w: f64,
  un: Option<Vec<f64>>,
}

/// element points, then the points of each face
fn points(gt: &PGeoTrans, g: &Mat, degree: usize) -> Result<Vec<Pt>> {
  let pim = classical_approx_im(gt, degree)?;
  let ai = pim.approx()?;
  let n = g.nrows();
  let mut out = vec![];
  let at = |ip: usize, face: Option<usize>| -> Result<Pt> {
    let xref = ai.point(ip).to_vec();
    let pg = gt.at(g, &xref)?;
    let un = face.map(|f| {
      let nr = &gt.normals()[f];
      (0..n)
        .map(|r| (0..nr.len()).map(|j| pg.b[(r, j)] * nr[j]).sum())
        .collect()
    });
    Ok(Pt {
      w: ai.coeff(ip) * pg.j.abs(),
      xref,
      un,
    })
  };
  for ip in 0..ai.nb_points_on_convex() {
    out.push(at(ip, None)?);
  }
  for f in 0..gt.structure().nb_faces() {
    let first = ai.ind_first_point_on_face(f);
    for ip in 0..ai.nb_points_on_face(f) {
      out.push(at(first + ip, Some(f))?);
    }
  }
  Ok(out)
}

/// Vectorized basis of pf with q components at xref (GetFEM
/// vectorize_base_tensor / vectorize_grad_base_tensor): v[k][c],
/// g[k][c][r], k = i*qmult + c for scalar elements.
struct Vb {
  v: Vec<Vec<f64>>,
  g: Vec<Vec<Vec<f64>>>,
}

fn vbasis(pf: &PFem, gt: &PGeoTrans, g: &Mat, xref: &[f64], q: usize) -> Result<Vb> {
  let n = g.nrows();
  let (val, grad, _) = real_base_values(pf, gt, g, xref, 1)?;
  let t = pf.target_dim();
  let nd = val.len() / t;
  let qm = q / t;
  let mut v = vec![vec![0.0; q]; nd * qm];
  let mut gg = vec![vec![vec![0.0; n]; q]; nd * qm];
  for i in 0..nd {
    for c in 0..qm {
      for a in 0..t {
        v[i * qm + c][c * t + a] = val[i + nd * a];
        for r in 0..n {
          gg[i * qm + c][c * t + a][r] = grad[i + nd * (a + t * r)];
        }
      }
    }
  }
  Ok(Vb { v, g: gg })
}

fn inv(m: &Mat) -> Result<Mat> {
  Ok(
    Lu::new(m)
      .map_err(|e| Error::InvalidArgument(e.to_string()))?
      .inverse(),
  )
}

/// inverse of a mass matrix; for scalar elements (s copies) the block of
/// one component inverted and replicated
fn mass_inverse(m: &Mat, s: usize) -> Result<Mat> {
  let n = m.nrows();
  if s <= 1 {
    return inv(m);
  }
  let nb = n / s;
  let mut b = Mat::zeros(&[nb, nb]);
  for i in 0..nb {
    for j in 0..nb {
      b[(i, j)] = m[(i * s, j * s)];
    }
  }
  let bi = inv(&b)?;
  let mut out = Mat::zeros(&[n, n]);
  for c in 0..s {
    for i in 0..nb {
      for j in 0..nb {
        out[(i * s + c, j * s + c)] = bi[(i, j)];
      }
    }
  }
  Ok(out)
}

/// zero the entries below tol
fn clean(m: &mut Mat, tol: f64) {
  m.data_mut().iter_mut().for_each(|v| {
    if v.abs() < tol {
      *v = 0.0
    }
  });
}

fn norminf(m: &Mat) -> f64 {
  m.data().iter().fold(0.0f64, |a, v| a.max(v.abs()))
}

fn interior(pf: &PFem, what: &str) -> Result<PFem> {
  pf.hho_interior().ok_or_else(|| {
    Error::InvalidArgument(format!("{what}: the variable has to be on an HHO element"))
  })
}

/// 0.5 (t[k1 + k2 N] + t[k2 + k1 N]) for a flat N x N tensor
fn sym_at(t: &[f64], k1: usize, k2: usize, n: usize, sym: bool) -> f64 {
  if sym {
    0.5 * (t[k1 + k2 * n] + t[k2 + k1 * n])
  } else {
    t[k1 + k2 * n]
  }
}

/// Reconstructed gradient G on the target space:
/// int_T G:w = int_T Grad(v_T):w + int_dT (v_dT - v_T).(w n)
/// (sym: w replaced by Sym(w)).
#[derive(Debug)]
pub struct HhoReconstructedGradient {
  pub sym: bool,
}

impl ElementaryTransformation for HhoReconstructedGradient {
  fn give_transformation(&self, src: &ElemFem, dest: &ElemFem) -> Result<Mat> {
    let what = "HHO reconstructed gradient";
    let (pf1, pf2) = (src.pf, dest.pf);
    let pfi = interior(pf1, what)?;
    let (gt, g) = (src.gt, src.g);
    let (q, n) = (src.qdim, g.nrows());
    if self.sym && q != n {
      return Err(Error::InvalidArgument(format!(
        "{what}: vector fields of the dimension of the domain only"
      )));
    }
    if dest.qdim != q * n {
      return Err(Error::InvalidArgument(format!(
        "{what}: the target space has to be of dimension {}",
        q * n
      )));
    }
    let degree = pf1.estimated_degree().max(pf2.estimated_degree());
    let pts = points(gt, g, 2 * degree)?;
    let (mut m1, mut m2): (Option<Mat>, Option<Mat>) = (None, None);
    for p in &pts {
      let b1 = vbasis(pf1, gt, g, &p.xref, q)?;
      let b2 = vbasis(pf2, gt, g, &p.xref, q * n)?;
      let (nd1, nd2) = (b1.v.len(), b2.v.len());
      let m1 = m1.get_or_insert_with(|| Mat::zeros(&[nd2, nd1]));
      let m2 = m2.get_or_insert_with(|| Mat::zeros(&[nd2, nd2]));
      match &p.un {
        None => {
          for i in 0..nd2 {
            for j in 0..nd2 {
              m2[(i, j)] += p.w * (0..q * n).map(|k| b2.v[i][k] * b2.v[j][k]).sum::<f64>();
            }
          }
          for i in 0..nd1 {
            for j in 0..nd2 {
              let mut s = 0.0;
              for k1 in 0..q {
                for k2 in 0..n {
                  s += b1.g[i][k1][k2] * sym_at(&b2.v[j], k1, k2, q, self.sym);
                }
              }
              m1[(j, i)] += p.w * s;
            }
          }
        }
        Some(un) => {
          let bi = vbasis(&pfi, gt, g, &p.xref, q)?;
          for i in 0..nd1 {
            for j in 0..nd2 {
              for k1 in 0..q {
                let a = p.w * (b1.v[i][k1] - bi.v.get(i).map_or(0.0, |v| v[k1]));
                let mut b = 0.0;
                for (k2, u) in un.iter().enumerate() {
                  b += a * sym_at(&b2.v[j], k1, k2, q, self.sym) * u;
                }
                m1[(j, i)] += b;
              }
            }
          }
        }
      }
    }
    let (Some(m1), Some(m2)) = (m1, m2) else {
      return Err(Error::InvalidArgument(format!(
        "{what}: no integration point"
      )));
    };
    let s = if pf2.target_dim() == 1 { q * n } else { 1 };
    let mut m = mass_inverse(&m2, s)?.matmul(&m1);
    let tol = norminf(&m) * 1e-13;
    clean(&mut m, tol);
    Ok(m)
  }
}

/// Matrices of the reconstructed value D (on pf2): D = M2^-1 M1, with the
/// (penalized) mean value constraint, and the skew part for sym.
struct Recon {
  md: Mat,
}

/// the reconstruction D of v (pf1, q components) on pf2; extra(p, b1, b2)
/// receives the vectorized bases at each point (stabilization)
fn reconstruct(
  src: &ElemFem,
  pf2: &PFem,
  degree: usize,
  sym: bool,
  mut extra: impl FnMut(&Pt, &Vb, &Vb) -> Result<()>,
  what: &str,
) -> Result<Recon> {
  let pf1 = src.pf;
  let pfi = interior(pf1, what)?;
  let (gt, g) = (src.gt, src.g);
  let (q, n) = (src.qdim, g.nrows());
  if sym && q != n {
    return Err(Error::InvalidArgument(format!(
      "{what}: vector fields of the dimension of the domain only"
    )));
  }
  let pts = points(gt, g, degree)?;
  let mut area = 0.0;
  let mut mats: Option<[Mat; 6]> = None;
  for p in &pts {
    let b1 = vbasis(pf1, gt, g, &p.xref, q)?;
    let b2 = vbasis(pf2, gt, g, &p.xref, q)?;
    let (nd1, nd2) = (b1.v.len(), b2.v.len());
    let [m1, m2, m3, m4, m5, m6] = mats.get_or_insert_with(|| {
      [
        Mat::zeros(&[nd2, nd1]),
        Mat::zeros(&[nd2, nd2]),
        Mat::zeros(&[q, nd1]),
        Mat::zeros(&[q, nd2]),
        Mat::zeros(&[q * q, nd1]),
        Mat::zeros(&[q * q, nd2]),
      ]
    });
    let flat = |b: &Vb, i: usize| -> Vec<f64> {
      // [k1 + k2*q]: component k1, derivative k2
      let mut t = vec![0.0; q * n];
      for k1 in 0..q {
        for k2 in 0..n {
          t[k1 + k2 * q] = b.g[i][k1][k2];
        }
      }
      t
    };
    match &p.un {
      None => {
        area += p.w;
        let g2: Vec<Vec<f64>> = (0..nd2).map(|i| flat(&b2, i)).collect();
        for i in 0..nd2 {
          for j in 0..nd2 {
            let mut s = 0.0;
            for k1 in 0..q {
              for k2 in 0..n {
                s += g2[i][k1 + k2 * q] * sym_at(&g2[j], k1, k2, q, sym);
              }
            }
            m2[(j, i)] += p.w * s;
          }
          for k in 0..q {
            m4[(k, i)] += p.w * b2.v[i][k];
          }
          if sym {
            for k1 in 0..n {
              for k2 in 0..n {
                m6[(k1 + k2 * n, i)] += 0.5 * p.w * (g2[i][k1 + k2 * n] - g2[i][k2 + k1 * n]);
              }
            }
          }
        }
        for i in 0..nd1 {
          let g1 = flat(&b1, i);
          for j in 0..nd2 {
            let mut s = 0.0;
            for k1 in 0..q {
              for k2 in 0..n {
                s += g1[k1 + k2 * q] * sym_at(&g2[j], k1, k2, q, sym);
              }
            }
            m1[(j, i)] += p.w * s;
          }
          for k in 0..q {
            m3[(k, i)] += p.w * b1.v[i][k];
          }
        }
      }
      Some(un) => {
        let bi = vbasis(&pfi, gt, g, &p.xref, q)?;
        for i in 0..nd1 {
          for j in 0..nd2 {
            let g2 = flat(&b2, j);
            for k1 in 0..q {
              let a = p.w * (b1.v[i][k1] - bi.v.get(i).map_or(0.0, |v| v[k1]));
              let mut b = 0.0;
              for (k2, u) in un.iter().enumerate() {
                b += a * sym_at(&g2, k1, k2, q, sym) * u;
              }
              m1[(j, i)] += b;
            }
          }
          if sym {
            for k1 in 0..n {
              for k2 in 0..n {
                m5[(k1 + k2 * n, i)] += 0.5 * p.w * (b1.v[i][k1] * un[k2] - b1.v[i][k2] * un[k1]);
              }
            }
          }
        }
      }
    }
    extra(p, &b1, &b2)?;
  }
  let Some([mut m1, mut m2, m3, m4, m5, m6]) = mats else {
    return Err(Error::InvalidArgument(format!(
      "{what}: no integration point"
    )));
  };
  let nf = n as f64;
  let add = |a: &mut Mat, b: &Mat, c: f64| {
    a.data_mut()
      .iter_mut()
      .zip(b.data())
      .for_each(|(x, y)| *x += c * y);
  };
  let cp1 = area.powf(-1.0 - 2.0 / nf);
  add(&mut m2, &m4.transposed().matmul(&m4), cp1);
  add(&mut m1, &m4.transposed().matmul(&m3), cp1);
  let m2inv = if sym {
    let cp2 = area.powf(-1.0 - 1.0 / nf);
    add(&mut m2, &m6.transposed().matmul(&m6), cp2);
    add(&mut m1, &m6.transposed().matmul(&m5), cp2);
    inv(&m2)?
  } else {
    mass_inverse(&m2, if pf2.target_dim() == 1 && q > 1 { q } else { 1 })?
  };
  let mut md = m2inv.matmul(&m1);
  let tol = norminf(&md) * 1e-13;
  clean(&mut md, tol);
  Ok(Recon { md })
}

/// Reconstructed value D on the target space:
/// int_T Grad(D):Grad(w) = int_T Grad(v_T):Grad(w) + int_dT (v_dT - v_T).(Grad(w) n),
/// int_T D = int_T v_T (sym: Sym(Grad(w)), and the skew part of Grad(D)).
#[derive(Debug)]
pub struct HhoReconstructedValue {
  pub sym: bool,
}

impl ElementaryTransformation for HhoReconstructedValue {
  fn give_transformation(&self, src: &ElemFem, dest: &ElemFem) -> Result<Mat> {
    let what = "HHO reconstructed value";
    if dest.qdim != src.qdim {
      return Err(Error::InvalidArgument(format!(
        "{what}: the target space has the dimension of the variable"
      )));
    }
    let degree = src.pf.estimated_degree().max(dest.pf.estimated_degree());
    Ok(reconstruct(src, dest.pf, 2 * degree, self.sym, |_, _, _| Ok(()), what)?.md)
  }
}

/// Stabilization S(v) = P_dT(v_dT - D(v) - P_T(v_T - D(v))) on the target
/// (HHO) space, D the reconstructed value of degree k + 1.
#[derive(Debug)]
pub struct HhoStabilization {
  pub sym: bool,
}

impl ElementaryTransformation for HhoStabilization {
  fn give_transformation(&self, src: &ElemFem, dest: &ElemFem) -> Result<Mat> {
    let what = "HHO stabilization";
    let (gt, g) = (src.gt, src.g);
    let q = src.qdim;
    if dest.qdim != q {
      return Err(Error::InvalidArgument(format!(
        "{what}: the target space has the dimension of the variable"
      )));
    }
    let degree = src.pf.estimated_degree();
    let pf2 = classical_fem(gt, degree + 1, false)?;
    let pf3 = dest.pf;
    let pf3i = interior(pf3, what)?;
    let mut mats: Option<[Mat; 4]> = None;
    let r = reconstruct(
      src,
      &pf2,
      2 * degree + 2,
      self.sym,
      |p, b1, b2| {
        let b3 = vbasis(pf3, gt, g, &p.xref, q)?;
        let (nd1, nd2, nd3) = (b1.v.len(), b2.v.len(), b3.v.len());
        let b3i = vbasis(&pf3i, gt, g, &p.xref, q)?;
        let nd3i = b3i.v.len();
        let [m7, m8, m9, m10] = mats.get_or_insert_with(|| {
          [
            Mat::zeros(&[nd3, nd3]),
            Mat::zeros(&[nd3, nd2]),
            Mat::zeros(&[nd3, nd1]),
            Mat::zeros(&[nd3, nd3]),
          ]
        });
        let w = match &p.un {
          None => p.w,
          Some(un) => p.w * un.iter().map(|x| x * x).sum::<f64>().sqrt(),
        };
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        for i in 0..nd3 {
          for j in 0..nd3 {
            m7[(i, j)] += w * dot(&b3.v[i], &b3.v[j]);
          }
          for j in 0..nd2 {
            m8[(i, j)] += w * dot(&b3.v[i], &b2.v[j]);
          }
          for j in 0..nd1 {
            m9[(i, j)] += w * dot(&b3.v[i], &b1.v[j]);
          }
          if p.un.is_some() {
            for j in 0..nd3i {
              m10[(i, j)] += w * dot(&b3.v[i], &b3i.v[j]);
            }
          }
        }
        Ok(())
      },
      what,
    )?;
    let Some([m7, m8, m9, m10]) = mats else {
      return Err(Error::InvalidArgument(format!(
        "{what}: no integration point"
      )));
    };
    let m7inv = mass_inverse(&m7, if pf3.target_dim() == 1 && q > 1 { q } else { 1 })?;
    // S = (I - M7^-1 M10) M7^-1 (M9 - M8 MD)
    let nd3 = m7.nrows();
    let mut pb = Mat::identity(nd3);
    let t = m7inv.matmul(&m10);
    pb.data_mut()
      .iter_mut()
      .zip(t.data())
      .for_each(|(x, y)| *x -= y);
    let mut c = m9.clone();
    let t = m8.matmul(&r.md);
    c.data_mut()
      .iter_mut()
      .zip(t.data())
      .for_each(|(x, y)| *x -= y);
    let mut m = pb.matmul(&m7inv.matmul(&c));
    clean(&mut m, 1e-13);
    Ok(m)
  }
}
