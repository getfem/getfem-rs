//! Transformed (non tau-equivalent) scalar elements of GetFEM
//! (getfem_fem.cc): Hermite, Argyris, Morley. Each has polynomial basis
//! functions on the reference element and an element matrix M_T mapping
//! them to the real element (GetFEM mat_trans), computed once per element.

use getfem_geom::convex_ref as cr;
use getfem_geom::geotrans::PGeoTrans;
use getfem_geom::poly;
use getfem_linalg::tensor::{Lu, Mat};

use crate::fem::{DofDesc, Fem, PFem, PolyFem, cached};
use crate::{Error, Result};
use getfem_linalg::vect_norm2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TransKind {
  HermiteSegment,
  HermiteTriangle,
  HermiteTetrahedron,
  Argyris,
  Morley,
  /// normal-component DOFs scaled and oriented (RTk, BDMk)
  NormalScaling,
  /// the same for the 2n face DOFs of RT0 on parallelepipeds
  NormalScalingQ,
  Nedelec,
}

/// Basis functions from GetFEM's ';'-separated polynomial strings.
pub(crate) fn polys(n: usize, s: &str) -> Result<Vec<crate::fem::Shape>> {
  s.split(';')
    .map(str::trim)
    .filter(|t| !t.is_empty())
    .map(|t| Ok(poly::parse::<f64>(n, t)?.into()))
    .collect()
}

fn build(
  name: &str,
  n: usize,
  nodesref: Vec<Vec<f64>>,
  dofs: Vec<DofDesc>,
  basis: &str,
  degree: usize,
  kind: TransKind,
) -> Result<PFem> {
  cached(name.to_string(), |name| {
    let base = polys(n, basis)?;
    assert_eq!(base.len(), nodesref.len(), "{name}");
    Ok(
      PolyFem::new(
        name,
        cr::simplex_of_reference(n),
        dofs,
        nodesref,
        base,
        degree,
      )?
      .transformed(kind),
    )
  })
}

pub(crate) fn hermite(d: usize) -> Result<PFem> {
  let name = format!("FEM_HERMITE({d})");
  match d {
    1 => build(
      &name,
      1,
      vec![vec![0.0], vec![0.0], vec![1.0], vec![1.0]],
      vec![
        DofDesc::lagrange(1),
        DofDesc::derivative(1, 0),
        DofDesc::lagrange(1),
        DofDesc::derivative(1, 0),
      ],
      "(1 - x)^2*(2*x + 1); x*(x - 1)*(x - 1); x*x*(3 - 2*x); x*x*(x - 1)",
      3,
      TransKind::HermiteSegment,
    ),
    2 => {
      let mut nodesref = vec![];
      let mut dofs = vec![];
      for v in [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]] {
        for (i, d) in [
          DofDesc::lagrange(2),
          DofDesc::derivative(2, 0),
          DofDesc::derivative(2, 1),
        ]
        .into_iter()
        .enumerate()
        {
          let _ = i;
          nodesref.push(v.to_vec());
          dofs.push(d);
        }
      }
      nodesref.push(vec![1.0 / 3.0, 1.0 / 3.0]);
      dofs.push(DofDesc::lagrange(2));
      build(
        &name,
        2,
        nodesref,
        dofs,
        "(1 - x - y)*(1 + x + y - 2*x*x - 11*x*y - 2*y*y);
         x*(1 - x - y)*(1 - x - 2*y);
         y*(1 - x - y)*(1 - 2*x - y);
         -2*x*x*x + 7*x*x*y + 7*x*y*y + 3*x*x - 7*x*y;
         x*x*x - 2*x*x*y - 2*x*y*y - x*x + 2*x*y;
         x*y*(2*x + y - 1);
         7*x*x*y + 7*x*y*y - 2*y*y*y + 3*y*y - 7*x*y;
         x*y*(x + 2*y - 1);
         y*y*y - 2*y*y*x - 2*y*x*x - y*y + 2*x*y;
         27*x*y*(1 - x - y)",
        3,
        TransKind::HermiteTriangle,
      )
    }
    3 => {
      // GetFEM: 5 groups of 4; groups 0..3 at the vertices (value, then the
      // three derivatives), group 4 at the face centers
      let (mut nodesref, mut dofs) = (vec![], vec![]);
      for k in 0..5 {
        for i in 0..4 {
          let mut ptref = vec![if k == 4 { 1.0 / 3.0 } else { 0.0 }; 3];
          if k == 4 && i > 0 {
            ptref[i - 1] = 0.0;
          }
          if k < 4 && k > 0 {
            ptref[k - 1] = 1.0;
          }
          nodesref.push(ptref);
          dofs.push(if k == 4 || i == 0 {
            DofDesc::lagrange(3)
          } else {
            DofDesc::derivative(3, i - 1)
          });
        }
      }
      build(
        &name,
        3,
        nodesref,
        dofs,
        "1 - 3*x*x - 13*x*y - 13*x*z - 3*y*y - 13*y*z - 3*z*z + 2*x*x*x
         + 13*x*x*y + 13*x*x*z + 13*x*y*y + 33*x*y*z + 13*x*z*z + 2*y*y*y
         + 13*y*y*z + 13*y*z*z + 2*z*z*z;
         x - 2*x*x - 3*x*y - 3*x*z + x*x*x + 3*x*x*y + 3*x*x*z + 2*x*y*y
         + 4*x*y*z + 2*x*z*z;
         y - 3*x*y - 2*y*y - 3*y*z + 2*x*x*y + 3*x*y*y + 4*x*y*z
         + y*y*y + 3*y*y*z + 2*y*z*z;
         z - 3*x*z - 3*y*z - 2*z*z + 2*x*x*z + 4*x*y*z + 3*x*z*z
         + 2*y*y*z + 3*y*z*z + z*z*z;
         3*x*x - 7*x*y - 7*x*z - 2*x*x*x + 7*x*x*y + 7*x*x*z + 7*x*y*y
         + 7*x*y*z + 7*x*z*z;
         -x*x + 2*x*y + 2*x*z + x*x*x - 2*x*x*y - 2*x*x*z - 2*x*y*y
         - 2*x*y*z - 2*x*z*z;
         -x*y + 2*x*x*y + x*y*y;
         -x*z + 2*x*x*z + x*z*z;
         -7*x*y + 3*y*y - 7*y*z + 7*x*x*y + 7*x*y*y + 7*x*y*z - 2*y*y*y
         + 7*y*y*z + 7*y*z*z;
         -x*y + x*x*y + 2*x*y*y;
         2*x*y - y*y + 2*y*z - 2*x*x*y - 2*x*y*y - 2*x*y*z + y*y*y
         - 2*y*y*z - 2*y*z*z;
         -y*z + 2*y*y*z + y*z*z;
         -7*x*z - 7*y*z + 3*z*z + 7*x*x*z + 7*x*y*z + 7*x*z*z + 7*y*y*z
         + 7*y*z*z - 2*z*z*z;
         -x*z + x*x*z + 2*x*z*z;
         -y*z + y*y*z + 2*y*z*z;
         2*x*z + 2*y*z - z*z - 2*x*x*z - 2*x*y*z - 2*x*z*z - 2*y*y*z
         - 2*y*z*z + z*z*z;
         27*x*y*z;
         27*y*z - 27*x*y*z - 27*y*y*z - 27*y*z*z;
         27*x*z - 27*x*x*z - 27*x*y*z - 27*x*z*z;
         27*x*y - 27*x*x*y - 27*x*y*y - 27*x*y*z",
        3,
        TransKind::HermiteTetrahedron,
      )
    }
    _ => Err(Error::InvalidArgument(format!(
      "{name}: dimension 1, 2 or 3"
    ))),
  }
}

pub(crate) fn argyris() -> Result<PFem> {
  let (mut nodesref, mut dofs) = (vec![], vec![]);
  for k in 0..7 {
    for i in 0..3 {
      let mut ptref = vec![0.0, 0.0];
      if k == 6 {
        ptref = vec![0.5, 0.5];
        if i > 0 {
          ptref[i - 1] = 0.0;
        }
        dofs.push(DofDesc::normal_derivative(2));
      } else {
        if k / 2 > 0 {
          ptref[k / 2 - 1] = 1.0;
        }
        dofs.push(if k & 1 == 1 {
          DofDesc::second_derivative(2, (i > 0) as usize, (i == 2) as usize)
        } else if i > 0 {
          DofDesc::derivative(2, i - 1)
        } else {
          DofDesc::lagrange(2)
        });
      }
      nodesref.push(ptref);
    }
  }
  build(
    "FEM_ARGYRIS",
    2,
    nodesref,
    dofs,
    "1 - 10*x^3 - 10*y^3 + 15*x^4 - 30*x*x*y*y
     + 15*y*y*y*y - 6*x^5 + 30*x*x*x*y*y + 30*x*x*y*y*y - 6*y^5;
     x - 6*x*x*x - 11*x*y*y + 8*x*x*x*x + 10*x*x*y*y
     + 18*x*y*y*y - 3*x*x*x*x*x + x*x*x*y*y - 10*x*x*y*y*y - 8*x*y*y*y*y;
     y - 11*x*x*y - 6*y*y*y + 18*x*x*x*y + 10*x*x*y*y
     + 8*y*y*y*y - 8*x*x*x*x*y - 10*x*x*x*y*y + x*x*y*y*y - 3*y*y*y*y*y;
     0.5*x*x - 1.5*x*x*x + 1.5*x*x*x*x - 1.5*x*x*y*y
     - 0.5*x*x*x*x*x + 1.5*x*x*x*y*y + x*x*y*y*y;
     x*y - 4*x*x*y - 4*x*y*y + 5*x*x*x*y + 10*x*x*y*y
     + 5*x*y*y*y - 2*x*x*x*x*y - 6*x*x*x*y*y - 6*x*x*y*y*y - 2*x*y*y*y*y;
     0.5*y*y - 1.5*y*y*y - 1.5*x*x*y*y + 1.5*y*y*y*y + x*x*x*y*y
     + 1.5*x*x*y*y*y - 0.5*y*y*y*y*y;
     10*x^3 - 15*x^4 + 15*x*x*y*y + 6*x^5 - 15*x*x*x*y*y - 15*x*x*y*y*y;
     -4*x*x*x + 7*x*x*x*x - 3.5*x*x*y*y - 3*x*x*x*x*x + 3.5*x*x*x*y*y
     + 3.5*x*x*y*y*y;
     -5*x*x*y + 14*x*x*x*y + 18.5*x*x*y*y - 8*x*x*x*x*y
     - 18.5*x*x*x*y*y - 13.5*x*x*y*y*y;
     0.5*x*x*x - x*x*x*x + 0.25*x*x*y*y + 0.5*x*x*x*x*x
     - 0.25*x*x*x*y*y - 0.25*x*x*y*y*y;
     x*x*y - 3*x*x*x*y - 3.5*x*x*y*y + 2*x*x*x*x*y + 3.5*x*x*x*y*y
     + 2.5*x*x*y*y*y;
     1.25*x*x*y*y - 0.75*x*x*x*y*y - 1.25*x*x*y*y*y;
     10*y*y*y + 15*x*x*y*y - 15*y^4 - 15*x*x*x*y*y - 15*x*x*y*y*y + 6*y^5;
     -5*x*y*y + 18.5*x*x*y*y + 14*x*y*y*y - 13.5*x*x*x*y*y
     - 18.5*x*x*y*y*y - 8*x*y*y*y*y;
     -4*y*y*y - 3.5*x*x*y*y + 7*y*y*y*y + 3.5*x*x*x*y*y
     + 3.5*x*x*y*y*y - 3*y*y*y*y*y;
     1.25*x*x*y*y - 1.25*x*x*x*y*y - 0.75*x*x*y*y*y;
     x*y*y - 3.5*x*x*y*y - 3*x*y*y*y + 2.5*x*x*x*y*y + 3.5*x*x*y*y*y
     + 2*x*y*y*y*y;
     0.5*y*y*y + 0.25*x*x*y*y - y*y*y*y - 0.25*x*x*x*y*y
     - 0.25*x*x*y*y*y + 0.5*y*y*y*y*y;
     sqrt(2) * (-8*x*x*y*y + 8*x*x*x*y*y + 8*x*x*y*y*y);
     -16*x*y*y + 32*x*x*y*y + 32*x*y*y*y - 16*x*x*x*y*y
     - 32*x*x*y*y*y - 16*x*y*y*y*y;
     -16*x*x*y + 32*x*x*x*y + 32*x*x*y*y - 16*x*x*x*x*y
     - 32*x*x*x*y*y - 16*x*x*y*y*y",
    5,
    TransKind::Argyris,
  )
}

pub(crate) fn morley(n: usize) -> Result<PFem> {
  let name = format!("FEM_MORLEY({n})");
  match n {
    2 => build(
      &name,
      2,
      vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![0.5, 0.5],
        vec![0.0, 0.5],
        vec![0.5, 0.0],
      ],
      [
        vec![DofDesc::lagrange(2); 3],
        vec![DofDesc::normal_derivative(2); 3],
      ]
      .concat(),
      "1 - x - y + 2*x*y;  (x + y + x^2 - 2*x*y - y^2)/2;
       (x + y - x^2 - 2*x*y + y^2)/2;
       ((x+y)^2 - x - y)*sqrt(2)/2;  x*(x-1);  y*(y-1)",
      2,
      TransKind::Morley,
    ),
    3 => {
      let t = 1.0 / 3.0;
      build(
        &name,
        3,
        vec![
          vec![0.5, 0.0, 0.0],
          vec![0.0, 0.5, 0.0],
          vec![0.0, 0.0, 0.5],
          vec![0.5, 0.5, 0.0],
          vec![0.5, 0.0, 0.5],
          vec![0.0, 0.5, 0.5],
          vec![t, t, t],
          vec![0.0, t, t],
          vec![t, 0.0, t],
          vec![t, t, 0.0],
        ],
        [
          vec![DofDesc::lagrange(3); 6],
          vec![DofDesc::normal_derivative(3); 4],
        ]
        .concat(),
        "5/3 - 16/9*x - 28/9*y - 28/9*z + 8/9*x^2 + 8/3*x*y + 8/3*x*z
         - 4/9*y^2 + 20/3*y*z - 4/9*z^2;
         5/3 - 28/9*x - 16/9*y - 28/9*z - 4/9*x^2 + 8/3*x*y + 20/3*x*z
         + 8/9*y^2 + 8/3*y*z - 4/9*z^2;
         5/3 - 28/9*x - 28/9*y - 16/9*z - 4/9*x^2 + 20/3*x*y + 8/3*x*z
         - 4/9*y^2 + 8/3*y*z + 8/9*z^2;
         -4/3 + 20/9*x + 20/9*y + 32/9*z + 8/9*x^2 - 4/3*x*y - 16/3*x*z
         + 8/9*y^2 - 16/3*y*z - 16/9*z^2;
         -4/3 + 20/9*x + 32/9*y + 20/9*z + 8/9*x^2 - 16/3*x*y - 4/3*x*z
         - 16/9*y^2 - 16/3*y*z + 8/9*z^2;
         -4/3 + 32/9*x + 20/9*y + 20/9*z - 16/9*x^2 - 16/3*x*y - 16/3*x*z
         + 8/9*y^2 - 4/3*y*z + 8/9*z^2;
         sqrt(3)*(3/8 - x - y - z + 1/2*x^2 + 3/2*x*y + 3/2*x*z + 1/2*y^2
         + 3/2*y*z + 1/2*z^2);
         -1/8 - 2/3*x + 1/3*y + 1/3*z + 11/6*x^2 - 1/2*x*y - 1/2*x*z
         - 1/6*y^2 - 1/2*y*z - 1/6*z^2;
         -1/8 + 1/3*x - 2/3*y + 1/3*z - 1/6*x^2 - 1/2*x*y - 1/2*x*z
         + 11/6*y^2 - 1/2*y*z - 1/6*z^2;
         -1/8 + 1/3*x + 1/3*y - 2/3*z - 1/6*x^2 - 1/2*x*y - 1/2*x*z
         - 1/6*y^2 - 1/2*y*z + 11/6*z^2",
        2,
        TransKind::Morley,
      )
    }
    _ => Err(Error::InvalidArgument(format!("{name}: dimension 2 or 3"))),
  }
}

// ------------------------------------------------------------ M_T

/// K = G grad_ref(pgt)(xref): N x P
pub(crate) fn jac(g: &Mat, pgt: &PGeoTrans, xref: &[f64]) -> Mat {
  g.matmul(&pgt.grad(xref))
}

pub(crate) fn inverse(a: &Mat) -> Result<Mat> {
  Ok(Lu::new(a)?.inverse())
}

pub(crate) fn mat_trans(kind: TransKind, f: &PolyFem, g: &Mat, pgt: &PGeoTrans) -> Result<Mat> {
  let nd = f.nb_dof();
  let mut m = Mat::identity(nd);
  let n = g.nrows();
  let dim_err = |w: &str| Err(Error::InvalidArgument(format!("{}: {w}", f.name())));
  match kind {
    TransKind::HermiteSegment => {
      // the element may lie in 2D or 3D if its tangent is the line
      let r: Vec<f64> = (0..n).map(|i| (i as f64).exp()).collect();
      let scale = |k: &Mat| {
        if n == 1 {
          k[(0, 0)]
        } else {
          let c = k.col(0);
          let s: f64 = c.iter().zip(&r).map(|(a, b)| a * b).sum();
          vect_norm2(c) * s.signum()
        }
      };
      let k = jac(g, pgt, f.node(1));
      m[(1, 1)] = scale(&k);
      let k = if pgt.is_linear() {
        k
      } else {
        jac(g, pgt, f.node(3))
      };
      m[(3, 3)] = scale(&k);
    }
    TransKind::HermiteTriangle | TransKind::HermiteTetrahedron => {
      let d = if kind == TransKind::HermiteTriangle {
        2
      } else {
        3
      };
      if n != d {
        return dim_err("elements of the mesh dimension only");
      }
      let mut k = jac(g, pgt, f.node(0));
      for v in 0..d + 1 {
        if v > 0 && !pgt.is_linear() {
          k = jac(g, pgt, f.node(v * (d + 1)));
        }
        for r in 0..d {
          for c in 0..d {
            m[(1 + (d + 1) * v + r, 1 + (d + 1) * v + c)] = k[(r, c)];
          }
        }
      }
    }
    TransKind::Argyris => {
      if n != 2 {
        return dim_err("2D elements only");
      }
      let mut k = jac(g, pgt, f.node(0));
      for v in 0..3 {
        let o = 6 * v;
        if v > 0 && !pgt.is_linear() {
          k = jac(g, pgt, f.node(o));
        }
        for r in 0..2 {
          for c in 0..2 {
            m[(1 + o + r, 1 + o + c)] = k[(r, c)];
          }
        }
        if !pgt.is_linear() {
          let b = inverse(&k.transposed())?;
          let h = g.matmul(&pgt.hess(f.node(o)));
          let mut xx = [[[0.0; 2]; 2]; 2];
          for (j, x) in xx.iter_mut().enumerate() {
            x[0][0] = b[(0, j)] * h[(0, 0)] + b[(1, j)] * h[(1, 0)];
            x[0][1] = b[(0, j)] * h[(0, 1)] + b[(1, j)] * h[(1, 1)];
            x[1][0] = x[0][1];
            x[1][1] = b[(0, j)] * h[(0, 3)] + b[(1, j)] * h[(1, 3)];
          }
          for j in 0..2 {
            let x = |r: usize, c: usize| k[(j, 0)] * xx[0][r][c] + k[(j, 1)] * xx[1][r][c];
            m[(1 + j + o, 3 + o)] = x(0, 0);
            m[(1 + j + o, 4 + o)] = x(1, 0);
            m[(1 + j + o, 5 + o)] = x(1, 1);
          }
        }
        let (a, bb, c, d) = (k[(0, 0)], k[(0, 1)], k[(1, 0)], k[(1, 1)]);
        let blk = [
          [a * a, a * bb, bb * bb],
          [2.0 * a * c, bb * c + a * d, 2.0 * bb * d],
          [c * c, c * d, d * d],
        ];
        for (r, row) in blk.iter().enumerate() {
          for (cc, v) in row.iter().enumerate() {
            m[(3 + o + r, 3 + o + cc)] = *v;
          }
        }
      }
      normal_derivative_correction(f, g, pgt, &mut m, 18, 3, k)?;
    }
    TransKind::NormalScaling | TransKind::NormalScalingQ | TransKind::Nedelec => {
      return crate::fem_vector::mat_trans(kind, f, g, pgt);
    }
    TransKind::Morley => {
      let d = f.dim();
      if n != d {
        return dim_err("elements of the mesh dimension only");
      }
      let nfd = if d == 2 { 3 } else { 6 };
      let k = jac(g, pgt, f.node(0));
      normal_derivative_correction(f, g, pgt, &mut m, nfd, d + 1, k)?;
    }
  }
  Ok(m)
}

/// The normal-derivative DOFs first..first+nb (Argyris, Morley): real
/// normals oriented by (pi, pi^2, ..), the rows of the other DOFs corrected
/// so that their functions have zero normal derivatives there.
/// Returns the real normals.
pub(crate) fn normal_derivative_correction(
  f: &dyn Fem,
  g: &Mat,
  pgt: &PGeoTrans,
  m: &mut Mat,
  first: usize,
  nb: usize,
  k_last: Mat,
) -> Result<Vec<Vec<f64>>> {
  let (n, nd) = (g.nrows(), f.nb_dof());
  let norient: Vec<f64> = (0..n)
    .map(|i| std::f64::consts::PI.powi(i as i32 + 1))
    .collect();
  let mut kinv = if pgt.is_linear() {
    Some(inverse(&k_last)?)
  } else {
    None
  };
  let mut w = Mat::zeros(&[nb, nd]);
  let mut normals = vec![];
  for i in first..first + nb {
    let ki = match &kinv {
      Some(k) if pgt.is_linear() => k.clone(),
      _ => {
        let k = inverse(&jac(g, pgt, f.node(i)))?;
        kinv = Some(k.clone());
        k
      }
    };
    let nr = &f.ref_convex().normals()[i - first];
    let mut nn: Vec<f64> = (0..n)
      .map(|r| (0..n).map(|c| ki[(c, r)] * nr[c]).sum())
      .collect();
    let l = vect_norm2(&nn);
    nn.iter_mut().for_each(|x| *x /= l);
    if nn.iter().zip(&norient).map(|(a, b)| a * b).sum::<f64>() < 0.0 {
      nn.iter_mut().for_each(|x| *x = -*x);
    }
    normals.push(nn.clone());
    let v: Vec<f64> = (0..n)
      .map(|r| (0..n).map(|c| ki[(r, c)] * nn[c]).sum())
      .collect();
    let t = f.grad_base_value(f.node(i))?;
    for j in 0..nd {
      w[(i - first, j)] = (0..n).map(|c| t.data()[j + nd * c] * v[c]).sum();
    }
  }
  let mut a = Mat::zeros(&[nb, nb]);
  for r in 0..nb {
    for c in 0..nb {
      a[(r, c)] = w[(r, first + c)];
    }
  }
  let a = inverse(&a)?;
  for r in 0..nb {
    for c in 0..nb {
      m[(first + r, first + c)] = a[(c, r)];
    }
  }
  for j in 0..first {
    let wj: Vec<f64> = (0..nb)
      .map(|r| (0..nd).map(|c| w[(r, c)] * m[(j, c)]).sum())
      .collect();
    for r in 0..nb {
      m[(j, first + r)] = -(0..nb).map(|c| a[(r, c)] * wj[c]).sum::<f64>();
    }
  }
  Ok(normals)
}
