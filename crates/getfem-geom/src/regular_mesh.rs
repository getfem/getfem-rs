//! Structured meshes of boxes (GetFEM's getfem_regular_meshes), with the
//! same point and convex numbering.

use crate::convex_ref::parallelepiped_of_reference;
use crate::convex_structure as cs;
use crate::geotrans::{self as gt, PGeoTrans};
use crate::mesh::{Mesh, MeshData};
use crate::simplexify::simplexified;
use crate::{Error, Result};
use getfem_linalg::tensor::Mat;

fn axpy(a: &mut [f64], s: f64, v: &[f64]) {
  a.iter_mut().zip(v).for_each(|(x, y)| *x += s * y);
}

/// Reference parallelepiped nodes mapped by the cell vectors.
fn cell_nodes(n: usize, ivect: &[Vec<f64>]) -> Vec<Vec<f64>> {
  parallelepiped_of_reference(n, 1)
    .points()
    .iter()
    .map(|noderef| {
      // in the dimension of the vectors (a cell of dimension n may lie in
      // a larger space)
      let mut a = vec![0.0; ivect.first().map_or(n, |v| v.len())];
      for k in 0..n {
        axpy(&mut a, noderef[k], &ivect[k]);
      }
      a
    })
    .collect()
}

/// Iterate over the cells (tab), GetFEM order; f(tab, total).
fn for_cells(iref: &[usize], mut f: impl FnMut(&[usize], usize) -> Result<()>) -> Result<()> {
  let n = iref.len();
  let mut tab = vec![0usize; n];
  let mut total = 0usize;
  while tab[n - 1] != iref[n - 1] {
    f(&tab, total)?;
    for l in 0..n {
      tab[l] += 1;
      total += 1;
      if l < n - 1 && tab[l] >= iref[l] {
        total -= tab[l];
        tab[l] = 0;
      } else {
        break;
      }
    }
  }
  Ok(())
}

fn origin_of_cell(org: &[f64], ivect: &[Vec<f64>], tab: &[usize]) -> Vec<f64> {
  let mut a = org.to_vec();
  for (i, &t) in tab.iter().enumerate() {
    axpy(&mut a, t as f64, &ivect[i]);
  }
  a
}

fn add_cell_points(m: &mut MeshData, a: &[f64], nodes: &[Vec<f64>]) -> Result<Vec<usize>> {
  nodes
    .iter()
    .map(|p| m.add_point(&a.iter().zip(p).map(|(x, y)| x + y).collect::<Vec<_>>()))
    .collect()
}

pub fn parallelepiped_regular_simplex_mesh(
  m: &mut MeshData,
  n: usize,
  org: &[f64],
  ivect: &[Vec<f64>],
  iref: &[usize],
) -> Result<()> {
  let pcvs = cs::parallelepiped_structure(n, 1);
  let sl = simplexified(&pcvs).ok_or(Error::NotImplemented(
    "simplex meshes in dimension >= 5 (not verified conformal)",
  ))?;
  let nodes = cell_nodes(n, ivect);
  let nbpt = nodes.len();
  let pgt = gt::pk(n, 1)?;
  for_cells(iref, |tab, total| {
    let a = origin_of_cell(org, ivect, tab);
    let t3 = add_cell_points(m, &a, &nodes)?;
    let shift = if total & 1 == 1 && n != 3 {
      nbpt / 2
    } else {
      0
    };
    for s in &sl {
      let t1: Vec<usize> = s.iter().map(|&l| t3[(l + shift) % nbpt]).collect();
      m.add_convex(&pgt, &t1)?;
    }
    Ok(())
  })
}

pub fn parallelepiped_regular_mesh(
  m: &mut MeshData,
  n: usize,
  org: &[f64],
  ivect: &[Vec<f64>],
  iref: &[usize],
) -> Result<()> {
  let nodes = cell_nodes(n, ivect);
  let pgt = gt::qk(n, 1)?;
  for_cells(iref, |tab, _| {
    let a = origin_of_cell(org, ivect, tab);
    let t3 = add_cell_points(m, &a, &nodes)?;
    m.add_convex(&pgt, &t3)?;
    Ok(())
  })
}

pub fn parallelepiped_regular_prism_mesh(
  m: &mut MeshData,
  n: usize,
  org: &[f64],
  ivect: &[Vec<f64>],
  iref: &[usize],
) -> Result<()> {
  // the simplices of dimension n-1 in dimension n, with the full vectors
  let mut aux = MeshData::new(n);
  parallelepiped_regular_simplex_mesh(&mut aux, n - 1, org, &ivect[..n - 1], &iref[..n - 1])?;
  let pgt = gt::prism_pk(n, 1)?;
  for cv in aux.convex_index() {
    let mut ptab: Vec<Vec<f64>> = aux
      .ind_points_of_convex(cv)
      .iter()
      .map(|&p| aux.point(p).to_vec())
      .collect();
    for _ in 0..iref[n - 1] {
      let top: Vec<Vec<f64>> = ptab
        .iter()
        .map(|p| p.iter().zip(&ivect[n - 1]).map(|(a, b)| a + b).collect())
        .collect();
      let all: Vec<&[f64]> = ptab.iter().chain(&top).map(|v| v.as_slice()).collect();
      m.add_convex_by_points(&pgt, &all)?;
      ptab = top;
    }
  }
  Ok(())
}

/// Unit cube mesh with nsubdiv cells per direction, elements of type pgt.
pub fn regular_unit_mesh(nsubdiv: &[usize], pgt: &PGeoTrans) -> Result<Mesh> {
  let n = nsubdiv.len();
  if n == 0 || nsubdiv.contains(&0) {
    return Err(Error::InvalidArgument(format!(
      "regular mesh: bad subdivisions {nsubdiv:?}"
    )));
  }
  if pgt.dim() != n {
    return Err(Error::InvalidArgument(format!(
      "regular mesh: {} for {n} subdivisions",
      pgt.name()
    )));
  }
  let org = vec![0.0; n];
  let ivect: Vec<Vec<f64>> = (0..n)
    .map(|i| {
      let mut v = vec![0.0; n];
      v[i] = 1.0 / nsubdiv[i] as f64;
      v
    })
    .collect();
  let mut msh = MeshData::new(n);
  let basic = pgt.basic_structure();
  if *basic == *cs::simplex_structure(n) {
    parallelepiped_regular_simplex_mesh(&mut msh, n, &org, &ivect, nsubdiv)?;
  } else if *basic == *cs::parallelepiped_structure(n, 1) {
    parallelepiped_regular_mesh(&mut msh, n, &org, &ivect, nsubdiv)?;
  } else if n >= 2 && *basic == *cs::prism_p1_structure(n) {
    parallelepiped_regular_prism_mesh(&mut msh, n, &org, &ivect, nsubdiv)?;
  } else {
    return Err(Error::NotImplemented("regular mesh for this element type"));
  }
  let mut m = MeshData::new(n);
  for cv in msh.convex_index() {
    let g = msh.points_of_convex(cv);
    let src = msh.trans_of_convex(cv);
    let pts: Vec<Vec<f64>> = if src == pgt {
      (0..g.ncols()).map(|j| g.col(j).to_vec()).collect()
    } else {
      pgt
        .geometric_nodes()
        .iter()
        .map(|xref| src.transform(xref, &g))
        .collect::<Result<_>>()?
    };
    let refs: Vec<&[f64]> = pts.iter().map(|v| v.as_slice()).collect();
    m.add_convex_by_points(pgt, &refs)?;
  }
  m.optimize_structure();
  Ok(m.into())
}

/// Box [org, org + sizes] with nsubdiv cells per direction.
pub fn regular_mesh(
  pgt: &PGeoTrans,
  nsubdiv: &[usize],
  org: &[f64],
  sizes: &[f64],
) -> Result<Mesh> {
  let n = nsubdiv.len();
  if org.len() != n || sizes.len() != n {
    return Err(Error::InvalidArgument(
      "regular mesh: org/sizes of wrong dimension".into(),
    ));
  }
  let m = regular_unit_mesh(nsubdiv, pgt)?;
  let mut d = Mat::zeros(&[n, n]);
  for i in 0..n {
    d[(i, i)] = sizes[i];
  }
  m.transform(&d)?;
  m.translate(org)?;
  Ok(m)
}

/// GetFEM's "noised" unit mesh: a fixed continuous deformation of the
/// inner points (shake, plus a radial twist in 2D). GetFEM's random part
/// has a zero factor, so the result is deterministic.
pub fn noise_unit_mesh(m: &Mesh) -> Result<()> {
  let n = m.dim();
  let mut moves = vec![];
  for ip in m.point_index() {
    let x = m.point(ip).to_vec();
    if x
      .iter()
      .any(|&v| v.abs() < 1e-10 || (v - 1.0).abs() < 1e-10)
    {
      continue;
    }
    let (mut c1, mut c2) = (1.0, 1.0);
    for &v in &x {
      c1 *= v * (1.0 - v);
      c2 *= 0.5 - (v - 0.5).abs();
    }
    let mut z: Vec<f64> = x.iter().map(|v| v + c2 / 3.0).collect();
    z[0] = x[0] + c1;
    if n == 2 {
      let (a, b) = (z[0] - 0.5, z[1] - 0.5);
      let r = (a * a + b * b).sqrt();
      let mut theta = b.atan2(a);
      if r < 0.5 - 1e-6 {
        theta += 10000.0 * r.sqrt() * (0.5 - r).powi(4) * (0.1 - r) * (0.15 - r);
      }
      z = vec![r * theta.cos() + 0.5, r * theta.sin() + 0.5];
    }
    moves.push((ip, z));
  }
  m.set_points(&moves)
}

/// Disc (ball) of the given radius centred at org ("structured_ball"): a
/// core cube scaled by 1/sqrt(2), one "peel" mesh per direction around it,
/// the points mapped radially onto the ball; nsubdiv = [core, peel]
/// subdivisions; the copies by symmetry for symmetries < dim (0: the whole
/// ball, dim: one sector).
pub fn regular_ball_mesh(
  pgt: &PGeoTrans,
  org: &[f64],
  radius: f64,
  nsubdiv: [usize; 2],
  symmetries: usize,
  noised: bool,
) -> Result<Mesh> {
  let n = pgt.dim();
  if org.len() != n {
    return Err(Error::InvalidArgument(format!(
      "regular ball mesh: org of size {} for dim {n}",
      org.len()
    )));
  }
  let core_ratio = std::f64::consts::FRAC_1_SQRT_2;
  let unit = |sub: &[usize]| -> Result<MeshData> {
    let m = regular_unit_mesh(sub, pgt)?;
    if noised {
      noise_unit_mesh(&m)?;
    }
    Ok((*m.snapshot()).clone())
  };
  let diag = |d: f64| -> Mat {
    let mut m = Mat::zeros(&[n, n]);
    for i in 0..n {
      m[(i, i)] = d;
    }
    m
  };
  let mut m = unit(&vec![nsubdiv[0]; n])?;
  m.transform(&diag(core_ratio))?;
  let peel_ratio = 1.0 - core_ratio;
  for i in 0..n {
    let mut sub = vec![nsubdiv[0]; n];
    sub[i] = nsubdiv[1];
    let mut mi = unit(&sub)?;
    let mut mm = diag(core_ratio);
    mm[(i, i)] = peel_ratio;
    mi.transform(&mm)?;
    // sheared: the other coordinates grow with the i-th
    let moves: Vec<(usize, Vec<f64>)> = mi
      .point_index()
      .map(|p| {
        let mut x = mi.point(p).to_vec();
        let xi = x[i];
        for (k, xk) in x.iter_mut().enumerate() {
          if k != i {
            *xk += (*xk / core_ratio) * xi;
          }
        }
        (p, x)
      })
      .collect();
    mi.set_points(&moves)?;
    let mut t = vec![0.0; n];
    t[i] = core_ratio;
    mi.translate(&t)?;
    add_convexes_of(&mut m, &mi)?;
  }
  // onto the ball
  let moves: Vec<(usize, Vec<f64>)> = m
    .point_index()
    .map(|p| {
      let mut x = m.point(p).to_vec();
      let (mut maxcoord, mut kmax) = (0.0, 0);
      for (k, xk) in x.iter().enumerate() {
        if xk.abs() > maxcoord {
          maxcoord = xk.abs();
          kmax = k;
        }
      }
      if maxcoord > 1e-10 {
        let mut l = 0.0;
        for (k, xk) in x.iter_mut().enumerate() {
          if k != kmax {
            let theta = std::f64::consts::FRAC_PI_4 * *xk / maxcoord;
            let c0 = maxcoord.min(1.0);
            *xk = c0 * theta.tan() * maxcoord + (1.0 - c0) * *xk;
          }
          l += *xk * *xk;
        }
        let l = l.sqrt();
        let l0 = l / maxcoord;
        let mut scale = radius;
        let c = core_ratio * (1.0 - (l0 * l0 - 1.0).sqrt()).max(0.3);
        if l > c {
          scale -= (l - c) / (l0 - c) * (radius - radius / (l / maxcoord));
        }
        for xk in x.iter_mut() {
          *xk *= scale;
        }
      }
      (p, x)
    })
    .collect();
  m.set_points(&moves)?;
  symmetric_copies(&mut m, symmetries)?;
  m.translate(org)?;
  Ok(m.into())
}

/// Spherical shell ("structured_ball_shell") centred at org, outer radius
/// and thickness: one sector per direction, each a unit mesh mapped onto
/// the shell (nsubdiv = [around, across]); the copies by symmetry as
/// regular_ball_mesh. An org of dimension dim(pgt) + 1: the sphere of
/// that radius itself (a circle in 2D, a surface in 3D: elements of
/// dimension dim(pgt), nsubdiv[0] around; thickness, nsubdiv[1] unused;
/// GetFEM: none).
pub fn regular_ball_shell_mesh(
  pgt: &PGeoTrans,
  org: &[f64],
  radius: f64,
  thickness: f64,
  nsubdiv: [usize; 2],
  symmetries: usize,
  noised: bool,
) -> Result<Mesh> {
  let p = pgt.dim();
  let surface = org.len() == p + 1;
  if org.len() != p && !surface {
    return Err(Error::InvalidArgument(format!(
      "regular ball shell mesh: org of size {} for dim {p} (or {}: the sphere)",
      org.len(),
      p + 1
    )));
  }
  let n = org.len();
  let mut sub = vec![nsubdiv[0]; p];
  if !surface {
    sub[p - 1] = nsubdiv[1];
  }
  let unit = regular_unit_mesh(&sub, pgt)?;
  if noised {
    noise_unit_mesh(&unit)?;
  }
  let unit = (*unit.snapshot()).clone();
  // onto the shell: the first n - 1 coordinates as angles, the last across
  // (on the sphere for a surface)
  let onto = |x: &[f64]| -> Vec<f64> {
    let mut y: Vec<f64> = x[..n - 1]
      .iter()
      .map(|xk| (xk * std::f64::consts::FRAC_PI_4).tan())
      .collect();
    let l = y.iter().fold(1.0, |l, v| l + v * v).sqrt();
    let r = if surface {
      radius
    } else {
      radius - thickness + thickness * x[n - 1]
    };
    y.iter_mut().for_each(|v| *v *= r / l);
    y.push(r / l);
    y
  };
  let mut m0 = MeshData::new(n);
  for cv in unit.convex_index().collect::<Vec<_>>() {
    let g = unit.points_of_convex(cv);
    let pts: Vec<Vec<f64>> = (0..g.ncols()).map(|j| onto(g.col(j))).collect();
    let refs: Vec<&[f64]> = pts.iter().map(|v| v.as_slice()).collect();
    m0.add_convex_by_points(unit.trans_of_convex(cv), &refs)?;
  }
  // one sector per direction: the coordinates permuted cyclically
  let mut perm = Mat::zeros(&[n, n]);
  for i in 0..n - 1 {
    perm[(i, i + 1)] = 1.0;
  }
  perm[(n - 1, 0)] = 1.0;
  let mut m = MeshData::new(n);
  for _ in 0..n {
    m0.transform(&perm)?;
    add_convexes_of(&mut m, &m0)?;
  }
  symmetric_copies(&mut m, symmetries)?;
  m.translate(org)?;
  Ok(m.into())
}

/// the convexes of m0 added to m by their points
fn add_convexes_of(m: &mut MeshData, m0: &MeshData) -> Result<()> {
  for cv in m0.convex_index().collect::<Vec<_>>() {
    let g = m0.points_of_convex(cv);
    let pts: Vec<Vec<f64>> = (0..g.ncols()).map(|j| g.col(j).to_vec()).collect();
    let refs: Vec<&[f64]> = pts.iter().map(|v| v.as_slice()).collect();
    m.add_convex_by_points(m0.trans_of_convex(cv), &refs)?;
  }
  Ok(())
}

/// GetFEM's copies of a ball's sectors: for symmetries < dim, rotations by
/// 90 degrees, then by 180 (the whole ball for 0)
fn symmetric_copies(m: &mut MeshData, symmetries: usize) -> Result<()> {
  let n = m.dim();
  let symmetries = symmetries.min(n);
  let mut rot = Mat::zeros(&[n, n]);
  for sym in 0..n - symmetries {
    let sym0 = (sym + 1) % n;
    if sym0 != 0 {
      rot = Mat::zeros(&[n, n]);
      rot[(sym, sym0)] = -1.0;
      rot[(sym0, sym)] = 1.0;
      for i in 0..n {
        if i != sym && i != sym0 {
          rot[(i, i)] = 1.0;
        }
      }
    } else {
      rot = rot.matmul(&rot);
    }
    let mut m0 = m.clone();
    m0.transform(&rot)?;
    add_convexes_of(m, &m0)?;
  }
  Ok(())
}
