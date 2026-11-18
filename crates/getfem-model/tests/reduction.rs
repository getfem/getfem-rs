//! A problem on a space with reduction matrices (the displacement
//! components rotated at each node) against the same problem without:
//! the solution extended to the basic DOFs is the same, with the Dirichlet
//! condition by multipliers (on a space without reduction); elimination on
//! a reduced space is not available.

use getfem_fem::mesh_fem::MeshFem;
use getfem_fem::mesh_im::MeshIm;
use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::mesh::Mesh;
use getfem_geom::region::outer_faces_of_mesh;
use getfem_geom::regular_mesh::regular_unit_mesh;
use getfem_linalg::{Coo, Csr};
use getfem_model::{Model, Newton};

/// the components rotated at each node, a different angle per node
fn rotation(nb: usize) -> (Csr<f64>, Csr<f64>) {
  let mut e = Coo::new(nb, nb);
  for i in 0..nb / 2 {
    let t = 0.2 + 0.15 * i as f64;
    let (c, s) = (t.cos(), t.sin());
    let (a, b) = (2 * i, 2 * i + 1);
    for (r, k, v) in [(a, a, c), (a, b, -s), (b, a, s), (b, b, c)] {
      e.push(r, k, v);
    }
  }
  let e = e.to_csr().unwrap();
  (e.transpose(), e)
}

/// isotropic elasticity with a body force, fixed on the boundary
/// (multipliers on mf_mult, or elimination)
fn solve(m: &Mesh, mf: &MeshFem, mf_mult: Option<&MeshFem>) -> getfem_model::Result<Vec<f64>> {
  let mim = MeshIm::new(m).with_integration_method(4).unwrap();
  let mut md = Model::new();
  md.add_fem_variable("u", mf).unwrap();
  md.add_term(
    &mim,
    None,
    "(3*Div_u*Id(2) + (Grad_u+Grad_u')):Grad_Test_u - [1;0.5+X(1)].Test_u",
  )
  .unwrap();
  match mf_mult {
    Some(mm) => md
      .add_Dirichlet_condition_with_multiplier(&mim, "u", mm, 1, None)
      .map(|_| ())?,
    None => md.add_Dirichlet_condition_with_simplification("u", 1, None)?,
  }
  assert!(Newton::default().solve(&mut md, None)?.converged());
  Ok(mf.extend_vector(md.variable("u").unwrap()).unwrap())
}

#[test]
fn rotated_space_same_solution() {
  let m = regular_unit_mesh(&[3, 3], &geotrans_descriptor("GT_PK(2,1)").unwrap()).unwrap();
  m.set_region(1, &outer_faces_of_mesh(&m)).unwrap();
  let space = || {
    MeshFem::new(&m, 2)
      .with_classical_finite_element(2)
      .unwrap()
  };
  let (mf_b, mf_r) = (space(), space());
  let (r, e) = rotation(mf_b.nb_basic_dof().unwrap());
  mf_r.set_reduction_matrices(Some(&r), &e).unwrap();
  let mf_mult = space();
  let ub = solve(&m, &mf_b, None).unwrap();
  let ur = solve(&m, &mf_r, Some(&mf_mult)).unwrap();
  let s = ub.iter().fold(0.0f64, |a, x| a.max(x.abs()));
  assert!(s > 1e-3);
  let err = ub
    .iter()
    .zip(&ur)
    .fold(0.0f64, |a, (x, y)| a.max((x - y).abs()));
  assert!(err <= 1e-10 * s, "differs by {err:e}");
  let e = solve(&m, &mf_r, None).unwrap_err().to_string();
  assert!(e.contains("not available"), "{e}");
}

/// Periodic cell with macro strain DOFs (GetFEM snippet periodicity.py):
/// E binds each DOF on the + sides to its - side master, and adds the
/// macro strain (XX, XY, YX, YY) times the position on both; the new DOFs
/// are the 4 strains and the kept basic DOFs (the cell center removed); R
/// computed from E ("auto": least squares).
/// Homogeneous material under a macro stress (explicit right-hand side on
/// the strains): the solution is affine, u = eps X.
#[test]
fn periodic_cell_with_macro_strain() {
  let m = regular_unit_mesh(&[4, 4], &geotrans_descriptor("GT_PK(2,1)").unwrap()).unwrap();
  let mf = MeshFem::new(&m, 2)
    .with_classical_finite_element(2)
    .unwrap();
  let nb = mf.nb_basic_dof().unwrap();
  // positions relative to the cell center (cell [-1/2, 1/2]^2)
  let xy: Vec<[f64; 2]> = (0..nb / 2)
    .map(|i| {
      let p = mf.point_of_basic_dof(2 * i).unwrap();
      [p[0] - 0.5, p[1] - 0.5]
    })
    .collect();
  let key = |x: f64, y: f64| ((x * 1e5).round() as i64, (y * 1e5).round() as i64);
  let at: std::collections::HashMap<_, _> = xy
    .iter()
    .enumerate()
    .map(|(i, p)| (key(p[0], p[1]), 2 * i))
    .collect();
  let (h, tol) = (0.5, 1e-5);
  let (mut kept, mut pairs) = (vec![], vec![]);
  for (i, &[x, y]) in xy.iter().enumerate() {
    let shifts: &[(f64, f64)] = if x - h > -tol && y - h > -tol {
      &[(1.0, 1.0), (1.0, 0.0), (0.0, 1.0)]
    } else if x - h > -tol {
      &[(1.0, 0.0)]
    } else if y - h > -tol {
      &[(0.0, 1.0)]
    } else {
      &[]
    };
    let master = shifts.iter().find_map(|&(a, b)| at.get(&key(x - a, y - b)));
    match master {
      Some(&md) => pairs.push((2 * i, md)),
      None if key(x, y) == (0, 0) => {} // the center: removed
      None => kept.push(2 * i),
    }
  }
  let mut newdof = vec![usize::MAX; nb];
  let mut size = 4; // 0: XX, 1: XY, 2: YX, 3: YY
  for &d in &kept {
    newdof[d] = size;
    newdof[d + 1] = size + 1;
    size += 2;
  }
  let mut e = Coo::new(nb, size);
  for &d in &kept {
    for q in 0..2 {
      e.push(d + q, newdof[d + q], 1.0);
    }
  }
  for &(s, md) in &pairs {
    for q in 0..2 {
      e.push(s + q, newdof[md + q], 1.0);
    }
  }
  // the macro strain on both DOFs of each pair (a master shared by several
  // slaves once)
  let mut bound: Vec<usize> = pairs.iter().flat_map(|&(s, md)| [s, md]).collect();
  bound.sort();
  bound.dedup();
  for d in bound {
    let [x, y] = xy[d / 2];
    e.push(d, 0, x);
    e.push(d, 2, y);
    e.push(d + 1, 1, x);
    e.push(d + 1, 3, y);
  }
  // R from E (least squares; E'E not diagonal: the strain columns)
  mf.set_reduction_matrices(None, &e.to_csr().unwrap())
    .unwrap();
  assert_eq!(mf.nb_dof().unwrap(), size);
  assert!(mf.reduction_matrix().is_err());
  // extend then reduce: back
  let q0: Vec<f64> = (0..size).map(|i| (i as f64 * 0.7).sin()).collect();
  let back = mf.reduce_vector(&mf.extend_vector(&q0).unwrap()).unwrap();
  assert!(back.iter().zip(&q0).all(|(a, b)| (a - b).abs() < 1e-12));

  let (lambda, mu) = (1.0, 1.0);
  let sigma = [[1.0, 0.3], [0.3, 0.5]];
  let mim = MeshIm::new(&m).with_integration_method(4).unwrap();
  let mut md = Model::new();
  md.add_fem_variable("u", &mf).unwrap();
  md.add_term(
    &mim,
    None,
    &format!("({lambda}*Div_u*Id(2) + {mu}*(Grad_u+Grad_u')):Grad_Test_u"),
  )
  .unwrap();
  // macro stress times the cell area on the strains; the rotation
  // (XY = -YX) has no energy: penalized by (XY - YX)^2
  let mut l = vec![0.0; size];
  l[..4].copy_from_slice(&[sigma[0][0], sigma[0][1], sigma[1][0], sigma[1][1]]);
  md.set_explicit_rhs("u", Some(l)).unwrap();
  let mut p = Coo::new(size, size);
  for (i, j, v) in [(1, 1, 1.0), (1, 2, -1.0), (2, 1, -1.0), (2, 2, 1.0)] {
    p.push(i, j, v);
  }
  md.set_explicit_matrix("u", "u", Some(p.to_csr().unwrap()))
    .unwrap();
  assert!(Newton::default().solve(&mut md, None).unwrap().converged());

  // eps = C^-1 sigma (plane, isotropic)
  let (a, b) = (lambda + 2.0 * mu, lambda);
  let det = a * a - b * b;
  let exx = (a * sigma[0][0] - b * sigma[1][1]) / det;
  let eyy = (a * sigma[1][1] - b * sigma[0][0]) / det;
  let exy = sigma[0][1] / (2.0 * mu);
  let q = md.variable("u").unwrap().to_vec();
  let want_q = [exx, exy, exy, eyy];
  assert!(
    q[..4]
      .iter()
      .zip(&want_q)
      .all(|(x, y)| (x - y).abs() < 1e-10),
    "{:?}",
    &q[..4]
  );
  let u = mf.extend_vector(&q).unwrap();
  for (i, &[x, y]) in xy.iter().enumerate() {
    let want = [exx * x + exy * y, exy * x + eyy * y];
    for c in 0..2 {
      assert!(
        (u[2 * i + c] - want[c]).abs() < 1e-10,
        "DOF {}: {} vs {}",
        2 * i + c,
        u[2 * i + c],
        want[c]
      );
    }
  }
}
