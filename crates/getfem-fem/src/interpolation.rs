//! Interpolation of finite element fields (port of getfem_interpolation):
//! onto a Lagrange space (same mesh or another mesh), at points, at the
//! integration points of an ImData.

use std::collections::BTreeMap;

use getfem_geom::locate::Locator;
use getfem_linalg::{Coo, Csr};

use crate::fem::real_base_values;
use crate::im_data::ImData;
use crate::mesh_fem::MeshFem;
use crate::{Error, Result};

/// Tolerance of the point location.
const EPS: f64 = 1e-10;

/// Rows of the interpolated values of source at the reference point xref of
/// convex cv: row k (component) = sum_j coef_j of source basic DOF j.
fn local_rows(source: &MeshFem, cv: usize, xref: &[f64]) -> Result<Vec<Vec<(usize, f64)>>> {
  let num = source.numbering()?;
  let m = num.linked_mesh();
  let pf = num
    .fem_of_element(cv)
    .ok_or_else(|| Error::InvalidArgument(format!("no element on convex {cv}")))?;
  let (q, t, nd) = (num.qdim(), pf.target_dim(), pf.nb_dof());
  // values of a tau-equivalent scalar element: the reference basis, no
  // geometry (flat cells of a cut mesh have no inverse)
  let val = if pf.is_equivalent() && !pf.real_element_defined() && t == 1 {
    pf.base_value(xref)?.data().to_vec()
  } else {
    real_base_values(pf, m.trans_of_convex(cv), &m.points_of_convex(cv), xref, 0)?.0
  };
  let dofs = num.ind_basic_dof_of_element(cv);
  Ok(if t == 1 {
    // scalar element, qdim copies: DOF i*q + k
    (0..q)
      .map(|k| (0..nd).map(|i| (dofs[i * q + k], val[i])).collect())
      .collect()
  } else {
    // vector element, qdim = Q target_dim: component a + t qq from DOF
    // i Q + qq
    let qm = q / t;
    (0..q)
      .map(|c| {
        let (a, qq) = (c % t, c / t);
        (0..nd)
          .map(|i| (dofs[i * qm + qq], val[i + nd * a]))
          .collect()
      })
      .collect()
  })
}

/// The target: scalar elements, Lagrange ones onto another mesh (GetFEM:
/// interpolation_same_mesh asserts target_dim() == 1 only)
fn check_target(source: &MeshFem, target: &MeshFem) -> Result<usize> {
  let same = source.linked_mesh().same(&target.linked_mesh());
  let (qs, qt) = (source.qdim(), target.qdim());
  if qs != qt && qt != 1 {
    return Err(Error::InvalidArgument(format!(
      "interpolation of a field of dimension {qs} on a space of qdim {qt}"
    )));
  }
  let num = target.numbering()?;
  for cv in num.convex_index() {
    let pf = num.fem_of_element(cv).unwrap();
    if pf.target_dim() != 1 || (!same && !pf.is_lagrange()) {
      return Err(Error::InvalidArgument(format!(
        "interpolation: the target element {} is not a scalar Lagrange element",
        pf.name()
      )));
    }
  }
  Ok(qs / qt)
}

/// Interpolation on the same mesh: source evaluated at the nodes of the
/// target elements; values from discontinuous source elements averaged.
/// emit(target row, source row) for each evaluation; returns per target
/// basic DOF the number of evaluations (to average by).
fn same_mesh(
  source: &MeshFem,
  target: &MeshFem,
  qmult: usize,
  emit: &mut dyn FnMut(usize, Vec<(usize, f64)>),
) -> Result<Vec<usize>> {
  let (ns, nt) = (source.numbering()?, target.numbering()?);
  let qs = ns.qdim();
  let mut passes = vec![0usize; nt.nb_basic_dof()];
  for cv in ns.convex_index() {
    let (Some(pf_s), Some(pf_t)) = (ns.fem_of_element(cv), nt.fem_of_element(cv)) else {
      continue;
    };
    let discontinuous = pf_s.dof_types().iter().any(|d| !d.linkable);
    let tdofs = nt.ind_basic_dof_of_element(cv);
    for i in 0..pf_t.nb_dof() {
      let d = tdofs[i * nt.qdim()];
      if !discontinuous && passes[d] > 0 {
        continue;
      }
      passes[d] += 1;
      for (k, r) in local_rows(source, cv, pf_t.node(i))?
        .into_iter()
        .enumerate()
        .take(qs)
      {
        emit(d * qmult + k, r);
      }
    }
  }
  Ok(passes)
}

/// same_mesh as rows of the interpolation matrix
fn same_mesh_rows(
  source: &MeshFem,
  target: &MeshFem,
  qmult: usize,
) -> Result<Vec<BTreeMap<usize, f64>>> {
  let mut rows = vec![BTreeMap::new(); target.numbering()?.nb_basic_dof() * qmult];
  let passes = same_mesh(source, target, qmult, &mut |row, r| {
    for (j, v) in r {
      *rows[row].entry(j).or_insert(0.0) += v;
    }
  })?;
  // all the qdim components of a target node
  let qs = source.numbering()?.qdim();
  for (d, &p) in passes.iter().enumerate() {
    if p > 1 {
      for k in 0..qs {
        rows[d * qmult + k]
          .values_mut()
          .for_each(|v| *v /= p as f64);
      }
    }
  }
  Ok(rows)
}

/// same_mesh applied to the values ue on the basic DOFs of source
fn same_mesh_values(
  source: &MeshFem,
  target: &MeshFem,
  qmult: usize,
  ue: &[f64],
) -> Result<Vec<f64>> {
  let mut out = vec![0.0; target.numbering()?.nb_basic_dof() * qmult];
  let passes = same_mesh(source, target, qmult, &mut |row, r| {
    out[row] += r.into_iter().map(|(j, v)| v * ue[j]).sum::<f64>();
  })?;
  let qs = source.numbering()?.qdim();
  for (d, &p) in passes.iter().enumerate() {
    if p > 1 {
      for k in 0..qs {
        out[d * qmult + k] /= p as f64;
      }
    }
  }
  Ok(out)
}

/// Rows at points located in the source mesh; a point outside the source mesh
/// is an error.
fn at_points(source: &MeshFem, points: &[Vec<f64>]) -> Result<Vec<BTreeMap<usize, f64>>> {
  let ns = source.numbering()?;
  let q = ns.qdim();
  let loc = Locator::new(ns.linked_mesh(), EPS)?;
  let mut rows = vec![BTreeMap::new(); points.len() * q];
  let mut missed = vec![];
  for (p, x) in points.iter().enumerate() {
    // the first convex of the source space containing x
    let found = loc.find_first(x, |cv| ns.fem_of_element(cv).is_some())?;
    let Some((cv, xref)) = found else {
      missed.push(p);
      continue;
    };
    for (k, r) in local_rows(source, cv, &xref)?.into_iter().enumerate() {
      rows[p * q + k] = r.into_iter().collect();
    }
  }
  if !missed.is_empty() {
    return Err(Error::InvalidArgument(format!(
      "interpolation: {} point(s) outside the source mesh (first: {:?})",
      missed.len(),
      points[missed[0]]
    )));
  }
  Ok(rows)
}

/// Matrix M with (values on target) = M (values on source), on the DOFs
/// of both spaces. The target must be a scalar Lagrange space of qdim 1
/// or the qdim of source; with qdim 1, each target DOF receives the qdim
/// components of source.
pub fn interpolation_matrix(source: &MeshFem, target: &MeshFem) -> Result<Csr<f64>> {
  let qmult = check_target(source, target)?;
  let rows = if source.linked_mesh().same(&target.linked_mesh()) {
    same_mesh_rows(source, target, qmult)?
  } else {
    let nt = target.numbering()?;
    let qt = nt.qdim();
    let pts = (0..nt.nb_basic_dof() / qt)
      .map(|i| nt.point_of_basic_dof(i * qt))
      .collect::<Result<Vec<_>>>()?;
    at_points(source, &pts)?
  };
  // on the basic DOFs; reduced spaces: R_target M E_source (R per
  // component of a qdim 1 target)
  let (vt, vs) = (target.dofs()?, source.dofs()?);
  let mut m = Coo::new(rows.len(), vs.numbering.nb_basic_dof());
  for (i, r) in rows.iter().enumerate() {
    for (&j, &v) in r {
      m.push(i, j, v);
    }
  }
  let mut m = m.to_csr()?;
  if let Some(rs) = vs.reduction() {
    m = m.mul(&rs.e)?;
  }
  if let Some(rt) = vt.reduction() {
    let Some(r) = &rt.r else {
      return Err(Error::InvalidArgument(
        "interpolation matrix onto a space whose reduction matrix R is not stored (computed from E by least squares): interpolate the values instead".into(),
      ));
    };
    let mut rq = Coo::new(r.nrows * qmult, r.ncols * qmult);
    for i in 0..r.nrows {
      for (j, v) in r.row(i) {
        for k in 0..qmult {
          rq.push(i * qmult + k, j * qmult + k, v);
        }
      }
    }
    m = rq.to_csr()?.mul(&m)?;
  }
  Ok(m)
}

/// Values of u (on source) interpolated on target. The values are
/// computed point by point (no matrix: its rows hold every DOF of the
/// source element at each target node).
pub fn interpolation(source: &MeshFem, target: &MeshFem, u: &[f64]) -> Result<Vec<f64>> {
  check_size(source, u)?;
  let qmult = check_target(source, target)?;
  if source.linked_mesh().same(&target.linked_mesh()) {
    let v = same_mesh_values(source, target, qmult, &source.extend_vector(u)?)?;
    return target.reduce_vector(&v);
  }
  let nt = target.numbering()?;
  let qt = nt.qdim();
  let pts = (0..nt.nb_basic_dof() / qt)
    .map(|i| nt.point_of_basic_dof(i * qt))
    .collect::<Result<Vec<_>>>()?;
  // values on the basic DOFs of target, then reduced as target
  let v = interpolation_at_points(source, &pts, u)?;
  target.reduce_vector(&v)
}

/// Values of u (on source) at points: qdim values per point.
pub fn interpolation_at_points(
  source: &MeshFem,
  points: &[Vec<f64>],
  u: &[f64],
) -> Result<Vec<f64>> {
  check_size(source, u)?;
  let ue = source.extend_vector(u)?;
  let ns = source.numbering()?;
  let loc = Locator::new(ns.linked_mesh(), EPS)?;
  let mut out = Vec::with_capacity(points.len() * ns.qdim());
  let mut missed = vec![];
  for (p, x) in points.iter().enumerate() {
    match loc.find_first(x, |cv| ns.fem_of_element(cv).is_some())? {
      Some((cv, xref)) => {
        for r in local_rows(source, cv, &xref)? {
          out.push(r.into_iter().map(|(j, v)| v * ue[j]).sum());
        }
      }
      None => missed.push(p),
    }
  }
  if !missed.is_empty() {
    return Err(Error::InvalidArgument(format!(
      "interpolation: {} point(s) outside the source mesh (first: {:?})",
      missed.len(),
      points[missed[0]]
    )));
  }
  Ok(out)
}

/// Values of u (on source) at the integration points of imd, the
/// components of the field as the tensor of imd.
pub fn interpolation_to_im_data(source: &MeshFem, imd: &ImData, u: &[f64]) -> Result<Vec<f64>> {
  check_size(source, u)?;
  let ue = source.extend_vector(u)?;
  let (pts, q) = (imd.points()?, source.qdim());
  if imd.tensor_size() != q {
    return Err(Error::InvalidArgument(format!(
      "interpolation_to_im_data: field of dimension {q}, ImData tensors of size {}",
      imd.tensor_size()
    )));
  }
  let ims = imd.linked_mesh_im().ims()?;
  if !std::sync::Arc::ptr_eq(ims.linked_mesh(), source.numbering()?.linked_mesh()) {
    return Err(Error::InvalidArgument(
      "interpolation_to_im_data: the space and the ImData must share the mesh".into(),
    ));
  }
  let mut out = vec![0.0; pts.nb_points() * q];
  for cv in ims.convex_index() {
    let Some(first) = pts.index_of_point(cv, 0) else {
      continue;
    };
    let a = ims.int_method_of_element(cv)?.approx()?;
    for i in 0..a.nb_points_on_convex() {
      for (k, r) in local_rows(source, cv, a.point(i))?.into_iter().enumerate() {
        out[(first + i) * q + k] = r.iter().map(|&(j, v)| v * ue[j]).sum();
      }
    }
  }
  Ok(out)
}

fn check_size(source: &MeshFem, u: &[f64]) -> Result<()> {
  let n = source.nb_dof()?;
  if u.len() != n {
    return Err(Error::InvalidArgument(format!(
      "interpolation: {} values for a space of {n} DOFs",
      u.len()
    )));
  }
  Ok(())
}
