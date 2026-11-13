//! Special tau-equivalent elements of GetFEM (getfem_fem.cc): interior and
//! connectable-interior PK, cubic and face bubbles, P1 nonconforming,
//! hierarchical elements, Gauss-Lobatto elements on the segment.

use getfem_geom::convex_ref::{self as cr, PRef};

use crate::fem::{
  DofDesc, PFem, Poly, PolyFem, cached, faces_of, pk, pk_basis, poly_of, product_named, shrink,
};
use crate::gausslobatto_data::gausslobatto_coefficients;
use crate::integration::integ_descriptor;
use crate::{Error, Result};

/// GetFEM PK_int_ (SIMPLEX_IPK, QUAD_IPK, PRISM_IPK) and PK_conn_int_
/// (SIMPLEX_CIPK): the PK nodes shrunk by 0.1 toward their barycenter, on
/// the given reference convex; faces from the undeformed simplex nodes.
pub(crate) fn interior_pk(
  name: String,
  cvr: PRef,
  n: usize,
  k: usize,
  linkable: bool,
) -> Result<PFem> {
  cached(name, |name| {
    let mut nodesref = cr::simplex_of_reference_k(n, k).points().to_vec();
    let mut base: Vec<Poly> = nodesref.iter().map(|xref| pk_basis(n, k, xref)).collect();
    let dof = if linkable {
      DofDesc::lagrange(n)
    } else {
      DofDesc::lagrange_nonconforming(n)
    };
    let dofs = vec![dof; nodesref.len()];
    let faces = faces_of(&cr::simplex_of_reference(n), &dofs, &nodesref);
    shrink(&mut nodesref, &mut base, 0.1);
    PolyFem::with_faces(
      name,
      cvr,
      dofs,
      nodesref,
      faces,
      base.into_iter().map(Into::into).collect(),
      k,
    )
  })
}

/// The P1 basis of the simplex, in node order.
fn p1_basis(n: usize) -> Vec<Poly> {
  cr::simplex_of_reference_k(n, 1)
    .points()
    .iter()
    .map(|xref| pk_basis(n, 1, xref))
    .collect()
}

/// PK plus the cubic bubble at the barycenter (GetFEM PK_with_cubic_bubble_).
pub(crate) fn pk_with_cubic_bubble(n: usize, k: usize) -> Result<PFem> {
  if k > n {
    return Err(Error::InvalidArgument(format!(
      "FEM_PK_WITH_CUBIC_BUBBLE({n},{k}): k must be at most {n}"
    )));
  }
  cached(format!("FEM_PK_WITH_CUBIC_BUBBLE({n},{k})"), |name| {
    let p = poly_of(&pk(n, k, None)?)?;
    let (mut nodesref, mut dofs, mut base) = (p.nodes.clone(), p.dofs.clone(), p.base.clone());
    nodesref.push(vec![1.0 / (n + 1) as f64; n]);
    dofs.push(DofDesc::bubble1(n));
    let bubble = p1_basis(n).iter().fold(Poly::one(n), |acc, l| &acc * l);
    base.push(bubble.into());
    Ok(PolyFem::new(name, p.cvr.clone(), dofs, nodesref, base, n + 1)?.not_lagrange())
  })
}

/// P1 plus a bubble on the face opposite to vertex 0 (GetFEM P1_wabbfoaf_).
pub(crate) fn p1_bubble_face(n: usize) -> Result<PFem> {
  if n < 2 {
    return Err(Error::InvalidArgument(
      "FEM_P1_BUBBLE_FACE: dimension >= 2".into(),
    ));
  }
  cached(format!("FEM_P1_BUBBLE_FACE({n})"), |name| {
    let p = poly_of(&pk(n, 1, None)?)?;
    let (mut nodesref, mut dofs, mut base) = (p.nodes.clone(), p.dofs.clone(), p.base.clone());
    nodesref.push(vec![0.5; n]);
    dofs.push(DofDesc::bubble1(n));
    let l = p1_basis(n);
    let bubble = l[1..]
      .iter()
      .fold(&Poly::one(n) * (1u64 << n) as f64, |acc, x| &acc * x);
    base.push(bubble.into());
    Ok(PolyFem::new(name, p.cvr.clone(), dofs, nodesref, base, 2)?.not_lagrange())
  })
}

fn poly2(terms: &[(f64, usize, usize)]) -> Poly {
  // sum c x^i y^j
  terms.iter().fold(&Poly::one(2) * 0.0, |acc, &(c, i, j)| {
    let mut m = &Poly::one(2) * c;
    for _ in 0..i {
      m = &m * &Poly::var(2, 0);
    }
    for _ in 0..j {
      m = &m * &Poly::var(2, 1);
    }
    &acc + &m
  })
}

/// P1 with a Lagrange DOF at the middle of the face x + y = 1 (GetFEM
/// P1_wabbfoafla_).
pub(crate) fn p1_bubble_face_lagrange() -> Result<PFem> {
  cached("FEM_P1_BUBBLE_FACE_LAG".into(), |name| {
    let p = poly_of(&pk(2, 1, None)?)?;
    let (mut nodesref, mut dofs) = (p.nodes.clone(), p.dofs.clone());
    nodesref.push(vec![0.5, 0.5]);
    dofs.push(DofDesc::lagrange(2));
    let base = vec![
      poly2(&[(1.0, 0, 0), (-1.0, 0, 1), (-1.0, 1, 0)]),
      poly2(&[(1.0, 1, 0), (-2.0, 1, 1)]),
      poly2(&[(1.0, 0, 1), (-2.0, 1, 1)]),
      poly2(&[(4.0, 1, 1)]),
    ];
    PolyFem::new(
      name,
      p.cvr.clone(),
      dofs,
      nodesref,
      base.into_iter().map(Into::into).collect(),
      2,
    )
  })
}

/// Crouzeix-Raviart P1 on triangles (GetFEM P1_nonconforming_fem).
pub(crate) fn p1_nonconforming() -> Result<PFem> {
  cached("FEM_P1_NONCONFORMING".into(), |name| {
    let nodesref = vec![vec![0.5, 0.5], vec![0.0, 0.5], vec![0.5, 0.0]];
    let base = vec![
      poly2(&[(2.0, 1, 0), (2.0, 0, 1), (-1.0, 0, 0)]),
      poly2(&[(1.0, 0, 0), (-2.0, 1, 0)]),
      poly2(&[(1.0, 0, 0), (-2.0, 0, 1)]),
    ];
    PolyFem::new(
      name,
      cr::simplex_of_reference(2),
      vec![DofDesc::lagrange(2); 3],
      nodesref,
      base.into_iter().map(Into::into).collect(),
      1,
    )
  })
}

/// GetFEM thierach_femi: the DOFs of f1, plus those of f2 not at a node of
/// f1 (with f2's basis functions, hierarchical degree of f1).
pub(crate) fn gen_hierarchical(name: String, f1: &PFem, f2: &PFem) -> Result<PFem> {
  let (p1, p2) = (poly_of(f1)?, poly_of(f2)?);
  if p1.cvr.structure().basic() != p2.cvr.structure().basic() {
    return Err(Error::InvalidArgument(format!(
      "{name}: incompatible elements"
    )));
  }
  cached(name, |name| {
    let (mut nodesref, mut dofs, mut faces, mut base) = (
      p1.nodes.clone(),
      p1.dofs.clone(),
      p1.faces.clone(),
      p1.base.clone(),
    );
    for i in 0..p2.nodes.len() {
      let found = (0..p1.nodes.len()).any(|j| {
        let d: f64 = p2.nodes[i]
          .iter()
          .zip(&p1.nodes[j])
          .map(|(a, b)| (a - b) * (a - b))
          .sum();
        d.sqrt() < 1e-13 && p2.dofs[i].hierarchical_compatible(&p1.dofs[j])
      });
      if !found {
        let d = p2.dofs[i].with_hier_degree(p1.degree);
        faces.push(
          faces_of(
            &p1.cvr,
            std::slice::from_ref(&d),
            std::slice::from_ref(&p2.nodes[i]),
          )
          .remove(0),
        );
        nodesref.push(p2.nodes[i].clone());
        dofs.push(d);
        base.push(p2.base[i].clone());
      }
    }
    Ok(
      PolyFem::with_faces(name, p1.cvr.clone(), dofs, nodesref, faces, base, p2.degree)?
        .not_lagrange(),
    )
  })
}

/// Hierarchical PK: P1, then the new DOFs of each finer degree dividing k.
pub(crate) fn pk_hierarchical(n: usize, k: usize) -> Result<PFem> {
  if k == 0 {
    return Err(Error::InvalidArgument(
      "FEM_PK_HIERARCHICAL: degree >= 1".into(),
    ));
  }
  if k == 1 {
    return pk(n, 1, None);
  }
  let s = (2..=k).find(|s| k.is_multiple_of(*s)).unwrap();
  gen_hierarchical(
    format!("FEM_PK_HIERARCHICAL({n},{k})"),
    &pk_hierarchical(n, k / s)?,
    &pk(n, k, None)?,
  )
}

/// Hierarchical QK as a product of hierarchical PK factors.
pub(crate) fn qk_hierarchical(n: usize, k: usize) -> Result<PFem> {
  match n {
    0 => Err(Error::InvalidArgument(
      "FEM_QK_HIERARCHICAL: dimension >= 1".into(),
    )),
    1 => pk_hierarchical(1, k),
    _ => product_named(
      &qk_hierarchical(n - 1, k)?,
      &pk_hierarchical(1, k)?,
      format!("FEM_QK_HIERARCHICAL({n},{k})"),
    ),
  }
}

/// Hierarchical prism element.
pub(crate) fn prism_pk_hierarchical(n: usize, k: usize) -> Result<PFem> {
  match n {
    0 | 1 => Err(Error::InvalidArgument(
      "FEM_PRISM_PK_HIERARCHICAL: dimension >= 2".into(),
    )),
    2 => qk_hierarchical(2, k),
    _ => product_named(
      &pk_hierarchical(n - 1, k)?,
      &pk_hierarchical(1, k)?,
      format!("FEM_PRISM_PK_HIERARCHICAL({n},{k})"),
    ),
  }
}

/// Lagrange element of degree k at the Gauss-Lobatto points of the segment.
pub(crate) fn pk_gauss_lobatto(k: usize) -> Result<PFem> {
  let c = gausslobatto_coefficients(k).ok_or_else(|| {
    Error::InvalidArgument(format!("FEM_PK_GAUSSLOBATTO1D({k}): degree not tabulated"))
  })?;
  cached(format!("FEM_PK_GAUSSLOBATTO1D({k})"), |name| {
    let im = integ_descriptor(&format!("IM_GAUSSLOBATTO1D({})", 2 * k - 1))?;
    let a = im.approx()?;
    let mut nodesref: Vec<Vec<f64>> = (0..k + 1).map(|i| a.point(i).to_vec()).collect();
    nodesref.sort_by(|xref, yref| xref[0].total_cmp(&yref[0]));
    let base: Vec<Poly> = (0..k + 1)
      .map(|r| Poly::from_coeffs(1, k, c[r * (k + 1)..(r + 1) * (k + 1)].to_vec()))
      .collect::<std::result::Result<_, _>>()?;
    PolyFem::new(
      name,
      cr::simplex_of_reference(1),
      vec![DofDesc::lagrange(1); k + 1],
      nodesref,
      base.into_iter().map(Into::into).collect(),
      k,
    )
  })
}
