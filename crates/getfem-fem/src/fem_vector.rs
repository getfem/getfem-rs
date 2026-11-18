//! Vector elements of GetFEM (getfem_fem.cc): Raviart-Thomas RTk and RT0
//! on parallelepipeds (primal), Brezzi-Douglas-Marini BDMk (primal), Nedelec
//! (dual). Their normal or tangential DOFs are scaled and oriented by the
//! element matrix, against the fixed direction (pi, pi^2, ..).

use getfem_geom::convex_ref as cr;
use getfem_geom::geotrans::{PGeoTrans, geotrans_descriptor};
use getfem_geom::poly::PowerIndex;
use getfem_linalg::tensor::Mat;

use crate::fem::{DofDesc, Fem, PFem, Poly, PolyFem, Vectorial, cached, pk_basis};
use crate::fem_transformed::{TransKind, inverse, jac};
use crate::integration::classical_approx_im;
use crate::{Error, Result};
use getfem_linalg::{vect_norm2, vect_sp};

fn norient(n: usize) -> Vec<f64> {
  (0..n)
    .map(|i| std::f64::consts::PI.powi(i as i32 + 1))
    .collect()
}

/// The PK basis in the order of the reference nodes.
fn pk_polys(n: usize, k: usize) -> Vec<Poly> {
  cr::simplex_of_reference_k(n, k)
    .points()
    .iter()
    .map(|xref| pk_basis(n, k, xref))
    .collect()
}

fn monomial(n: usize, p: &[usize]) -> Poly {
  let mut m = Poly::zero(n);
  m.add_monomial(1.0, &PowerIndex::from_powers(p));
  m
}

/// Dual basis: base[i + m*nddl] = sum_j raw[j*n + m] (A^-1)_ji.
fn dual_basis(raw: &[Poly], a: &Mat, n: usize, nddl: usize) -> Result<Vec<crate::fem::Shape>> {
  let ainv = inverse(a)?;
  let mut base = vec![Poly::zero(n); nddl * n];
  for i in 0..nddl {
    for j in 0..nddl {
      let c = ainv[(j, i)];
      if c.abs() > 1e-14 {
        for m in 0..n {
          base[i + m * nddl] += &(&raw[j * n + m] * c);
        }
      }
    }
  }
  Ok(base.into_iter().map(Into::into).collect())
}

/// Faces of the reference simplex containing xref (|is_in_face| < 1e-10).
fn faces_at(n: usize, cvn: &cr::PRef, xref: &[f64]) -> Vec<usize> {
  (0..n + 1)
    .filter(|&f| {
      cvn
        .is_in_face(f, xref)
        .map(|v| v.abs() < 1e-10)
        .unwrap_or(false)
    })
    .collect()
}

/// Raviart-Thomas of degree k on simplices: RTk = (Pk)^n + x Pk.
pub(crate) fn rtk(n: usize, k: usize) -> Result<PFem> {
  if n < 2 {
    return Err(Error::InvalidArgument("FEM_RTK: dimension >= 2".into()));
  }
  cached(format!("FEM_RTK({n},{k})"), |name| {
    let pk = pk_polys(n, k);
    let nb_pk = pk.len();
    let mut nddl = k + n + 1;
    for i in 1..n {
      nddl *= k + i;
    }
    for i in 1..n {
      nddl /= i;
    }
    let mut raw = vec![Poly::zero(n); nddl * n];
    for (i, p) in pk.iter().enumerate() {
      for j in 0..n {
        raw[i * n + j * (nb_pk * n + 1)] = p.clone();
      }
    }
    // degree-k monomials times x (odometer order)
    let mut l = 0;
    let mut pi = vec![0usize; n];
    loop {
      if pi.iter().sum::<usize>() == k {
        for j in 0..n {
          raw[nb_pk * n * n + l * n + j] = &monomial(n, &pi) * &Poly::var(n, j);
        }
        l += 1;
      }
      for e in pi.iter_mut() {
        *e += 1;
        if *e == k + 1 {
          *e = 0;
        } else {
          break;
        }
      }
      if pi.iter().sum::<usize>() == 0 {
        break;
      }
    }
    assert_eq!(nb_pk * n + l, nddl);
    let cvn = cr::simplex_of_reference_k(n, k + n);
    let mut a = Mat::zeros(&[nddl, nddl]);
    let (mut nodesref, mut dofs) = (vec![], vec![]);
    let mut ip = 0;
    for xref in cvn.points() {
      let f = faces_at(n, &cvn, xref);
      match f.len() {
        0 => {
          for m in 0..n {
            for j in 0..nddl {
              a[(ip + m, j)] = raw[j * n + m].eval(xref);
            }
            nodesref.push(xref.clone());
            dofs.push(DofDesc::coord(n, m));
          }
          ip += n;
        }
        1 => {
          let nr = &cvn.normals()[f[0]];
          for j in 0..nddl {
            let v: Vec<f64> = (0..n).map(|m| raw[j * n + m].eval(xref)).collect();
            a[(ip, j)] = vect_sp(&v, nr);
          }
          nodesref.push(xref.clone());
          dofs.push(DofDesc::normal_component(n));
          ip += 1;
        }
        _ => {}
      }
    }
    assert_eq!(ip, nddl);
    let base = dual_basis(&raw, &a, n, nddl)?;
    Ok(
      PolyFem::new(name, cr::simplex_of_reference(n), dofs, nodesref, base, k)?
        .vector(n, Vectorial::Primal)
        .transformed(TransKind::NormalScaling),
    )
  })
}

/// Raviart-Thomas of lowest degree on parallelepipeds.
pub(crate) fn rt0q(n: usize) -> Result<PFem> {
  if n < 2 {
    return Err(Error::InvalidArgument("FEM_RT0Q: dimension >= 2".into()));
  }
  cached(format!("FEM_RT0Q({n})"), |name| {
    let nd = 2 * n;
    let mut base = vec![Poly::zero(n); n * nd];
    for i in 0..n {
      base[2 * i + i * nd] = Poly::var(n, i);
      base[2 * i + 1 + i * nd] = &Poly::var(n, i) - &Poly::one(n);
    }
    let (mut nodesref, mut dofs) = (vec![], vec![]);
    let mut ptref = vec![0.5; n];
    for i in 0..n {
      for v in [1.0, 0.0] {
        ptref[i] = v;
        nodesref.push(ptref.clone());
        dofs.push(DofDesc::normal_component(n));
      }
      ptref[i] = 0.5;
    }
    Ok(
      PolyFem::new(
        name,
        cr::parallelepiped_of_reference(n, 1),
        dofs,
        nodesref,
        base.into_iter().map(Into::into).collect(),
        1,
      )?
      .vector(n, Vectorial::Primal)
      .transformed(TransKind::NormalScalingQ),
    )
  })
}

/// Brezzi-Douglas-Marini of degree k (1 or 2) on simplices: (Pk)^n.
pub(crate) fn bdmk(n: usize, k: usize) -> Result<PFem> {
  if n < 2 || k == 0 {
    return Err(Error::InvalidArgument(
      "FEM_BDMK: dimension >= 2, degree >= 1".into(),
    ));
  }
  if k > 2 {
    // GetFEM completes the basis with an SVD (LAPACK) for k > 2
    return Err(Error::NotImplemented("FEM_BDMK with a degree above 2"));
  }
  if n != 2 {
    // GetFEM places the normal DOFs at lattice points inside one face only,
    // too few in 3D: it stops with an internal error
    return Err(Error::NotImplemented("FEM_BDMK in dimension 3"));
  }
  cached(format!("FEM_BDMK({n},{k})"), |name| {
    let pk = pk_polys(n, k);
    let nb_pk = pk.len();
    let nddl = nb_pk * n;
    let mut raw = vec![Poly::zero(n); nddl * n];
    for (i, p) in pk.iter().enumerate() {
      for j in 0..n {
        raw[i * n + j * (nb_pk * n + 1)] = p.clone();
      }
    }
    let cvn = cr::simplex_of_reference_k(n, k + 2);
    let mut a = Mat::zeros(&[nddl, nddl]);
    let (mut nodesref, mut dofs) = (vec![], vec![]);
    let mut ip = 0;
    for xref in cvn.points() {
      let f = faces_at(n, &cvn, xref);
      if f.len() == 1 {
        let nr = &cvn.normals()[f[0]];
        for j in 0..nddl {
          let v: Vec<f64> = (0..n).map(|m| raw[j * n + m].eval(xref)).collect();
          a[(ip, j)] = vect_sp(&v, nr);
        }
        nodesref.push(xref.clone());
        dofs.push(DofDesc::normal_component(n));
        ip += 1;
      }
    }
    if k > 1 {
      // moments of the divergence against P_{k-1}: Div(u).Test_p on the
      // reference element, in the node-major DOF order of a vector mesh_fem
      // (not the component-major order of the raw functions; still a basis)
      let q = pk_polys(n, k - 1);
      let im = classical_approx_im(&geotrans_descriptor(&format!("GT_PK({n},1)"))?, 2 * k)?;
      let ai = im.approx()?;
      for (i, qi) in q.iter().enumerate() {
        for (node, l) in pk.iter().enumerate() {
          for comp in 0..n {
            let dl = l.derived(comp);
            let v: f64 = (0..ai.nb_points_on_convex())
              .map(|p| ai.coeff(p) * dl.eval(ai.point(p)) * qi.eval(ai.point(p)))
              .sum();
            a[(ip + i, node * n + comp)] += v;
          }
        }
      }
      let baryref = vec![1.0 / (n + 1) as f64; n];
      for _ in 0..q.len() {
        nodesref.push(baryref.clone());
        dofs.push(DofDesc::ipk_center(n, ip));
        ip += 1;
      }
    }
    assert_eq!(ip, nddl);
    let base = dual_basis(&raw, &a, n, nddl)?;
    Ok(
      PolyFem::new(name, cr::simplex_of_reference(n), dofs, nodesref, base, k)?
        .vector(n, Vectorial::Primal)
        .transformed(TransKind::NormalScaling),
    )
  })
}

/// Nedelec (Whitney) edge element of the first kind, lowest degree.
pub(crate) fn nedelec(n: usize) -> Result<PFem> {
  if n < 2 {
    return Err(Error::InvalidArgument("FEM_NEDELEC: dimension >= 2".into()));
  }
  cached(format!("FEM_NEDELEC({n})"), |name| {
    let cvr = cr::simplex_of_reference(n);
    let nb = n * (n + 1) / 2;
    let mut lambda = vec![Poly::one(n)];
    let mut gl = vec![vec![-1.0; n]];
    for i in 1..=n {
      let x = Poly::var(n, i - 1);
      lambda[0] -= &x;
      lambda.push(x);
      let mut g = vec![0.0; n];
      g[i - 1] = 1.0;
      gl.push(g);
    }
    let mut base = vec![Poly::zero(n); nb * n];
    let (mut nodesref, mut dofs) = (vec![], vec![]);
    let ptsref = cvr.points();
    let mut j = 0;
    for k in 0..=n {
      for l in k + 1..=n {
        for i in 0..n {
          base[j + i * nb] = &(&lambda[k] * gl[l][i]) - &(&lambda[l] * gl[k][i]);
        }
        nodesref.push(
          ptsref[k]
            .iter()
            .zip(&ptsref[l])
            .map(|(a, b)| (a + b) / 2.0)
            .collect(),
        );
        dofs.push(DofDesc::edge_component(n));
        j += 1;
      }
    }
    Ok(
      PolyFem::new(
        name,
        cvr,
        dofs,
        nodesref,
        base.into_iter().map(Into::into).collect(),
        1,
      )?
      .vector(n, Vectorial::Dual)
      .transformed(TransKind::Nedelec),
    )
  })
}

/// Unit tangent of edge dof i (from vertex k to vertex l).
fn nedelec_tangent(n: usize, i: usize) -> Vec<f64> {
  let ptsref = cr::simplex_of_reference(n).points().to_vec();
  let mut j = 0;
  for k in 0..=n {
    for l in k + 1..=n {
      if j == i {
        let t: Vec<f64> = ptsref[l]
          .iter()
          .zip(&ptsref[k])
          .map(|(a, b)| a - b)
          .collect();
        let s = vect_norm2(&t);
        return t.iter().map(|x| x / s).collect();
      }
      j += 1;
    }
  }
  unreachable!()
}

pub(crate) fn mat_trans(kind: TransKind, f: &PolyFem, g: &Mat, pgt: &PGeoTrans) -> Result<Mat> {
  let (nd, n) = (f.nb_dof(), f.dim());
  if g.nrows() != n {
    return Err(Error::InvalidArgument(format!(
      "{}: elements of the mesh dimension only",
      f.name()
    )));
  }
  let mut m = Mat::identity(nd);
  let no = norient(n);
  match kind {
    TransKind::NormalScaling | TransKind::NormalScalingQ => {
      let mut kinv = inverse(&jac(g, pgt, f.node(0)))?;
      let dofs: Vec<usize> = if kind == TransKind::NormalScalingQ {
        (0..2 * n).collect()
      } else {
        (0..nd).collect()
      };
      for j in dofs {
        let face = if kind == TransKind::NormalScalingQ {
          Some(j)
        } else {
          f.faces_of_dof(j).first().copied()
        };
        let Some(face) = face else { continue };
        if !pgt.is_linear() {
          kinv = inverse(&jac(g, pgt, f.node(j)))?;
        }
        let nr = &f.ref_convex().normals()[face];
        let nv: Vec<f64> = (0..n)
          .map(|r| (0..n).map(|c| kinv[(c, r)] * nr[c]).sum())
          .collect();
        let l = vect_norm2(&nv);
        m[(j, j)] = if vect_sp(&nv, &no) / l < 0.0 { -l } else { l };
      }
    }
    TransKind::Nedelec => {
      for i in 0..nd {
        let pg = pgt.at(g, f.node(i))?;
        let tr = nedelec_tangent(n, i);
        let mut t: Vec<f64> = (0..n)
          .map(|r| (0..n).map(|c| pg.k[(r, c)] * tr[c]).sum())
          .collect();
        let s = vect_norm2(&t);
        t.iter_mut().for_each(|x| *x /= s);
        let mut v: Vec<f64> = (0..n)
          .map(|r| (0..n).map(|c| pg.b[(c, r)] * t[c]).sum())
          .collect();
        if vect_sp(&t, &no) < 0.0 {
          v.iter_mut().for_each(|x| *x = -*x);
        }
        let tt = f.base_value(f.node(i))?;
        for j in 0..nd {
          m[(j, i)] = (0..n).map(|k| tt.data()[j + nd * k] * v[k]).sum();
        }
      }
      m = inverse(&m)?;
    }
    _ => unreachable!(),
  }
  Ok(m)
}
