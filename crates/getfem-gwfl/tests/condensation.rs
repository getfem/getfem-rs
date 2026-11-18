//! The condensed tangent (assembly(2, true)) against the dense Schur
//! complement of the retained system: K^ = K_pp - K_pq K_qq^-1 K_qp, K' =
//! K_qq^-1 K_qp, R' = K_qq^-1 R_q, rhs = -K_pq R'.

mod common;

use getfem_fem::im_data::ImData;
use getfem_fem::mesh_fem::MeshFem;
use getfem_fem::mesh_im::MeshIm;
use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::mesh::Mesh;
use getfem_geom::regular_mesh::regular_unit_mesh;
use getfem_gwfl::workspace::Workspace;

/// dense inverse by Gauss-Jordan with partial pivoting (row-major n x n)
fn inverse(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
  let n = a.len();
  let mut m: Vec<Vec<f64>> = a
    .iter()
    .enumerate()
    .map(|(i, r)| {
      let mut r = r.clone();
      r.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
      r
    })
    .collect();
  for c in 0..n {
    let p = (c..n)
      .max_by(|&x, &y| m[x][c].abs().total_cmp(&m[y][c].abs()))
      .unwrap();
    m.swap(c, p);
    let d = m[c][c];
    m[c].iter_mut().for_each(|x| *x /= d);
    for r in 0..n {
      if r != c {
        let f = m[r][c];
        let rc = m[c].clone();
        m[r].iter_mut().zip(&rc).for_each(|(x, y)| *x -= f * y);
      }
    }
  }
  m.into_iter().map(|r| r[n..].to_vec()).collect()
}

fn mat(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
  let (n, k, m) = (a.len(), b.len(), b[0].len());
  (0..n)
    .map(|i| {
      (0..m)
        .map(|j| (0..k).map(|l| a[i][l] * b[l][j]).sum())
        .collect()
    })
    .collect()
}

fn sub(a: &[Vec<f64>], r: std::ops::Range<usize>, c: std::ops::Range<usize>) -> Vec<Vec<f64>> {
  a[r].iter().map(|row| row[c.clone()].to_vec()).collect()
}

fn max_diff(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
  a.iter()
    .flatten()
    .zip(b.iter().flatten())
    .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

/// the QK(2,2) mesh of test_condensation, nx x ny
fn qk_mesh(nx: usize, ny: usize) -> Mesh {
  regular_unit_mesh(&[nx, ny], &geotrans_descriptor("GT_QK(2,2)").unwrap()).unwrap()
}

/// u (classical degree 2 displacements), and the internal p (scalar) and
/// eps (3 components in 2D, coupled to p: one cluster of two) of
/// test_condensation, on m
fn setup(m: &Mesh, two: bool, reduced: bool) -> (Workspace, MeshIm, Vec<&'static str>) {
  let mf = MeshFem::new(m, m.dim())
    .with_classical_finite_element(2)
    .unwrap();
  if reduced {
    // as test_condensation: some basic DOFs removed (a selection)
    let nb = mf.nb_basic_dof().unwrap();
    mf.reduce_to_basic_dof(&(4..nb).collect()).unwrap();
  }
  let mim = MeshIm::new(m).with_integration_method(3).unwrap();
  let (mimd1, mimd3) = (
    ImData::new(&mim, -1, &[]).unwrap(),
    ImData::new(&mim, -1, &[3]).unwrap(),
  );
  let mut ws = Workspace::new();
  ws.add_fem_variable("u", &mf).unwrap();
  let nu = mf.nb_dof().unwrap();
  ws.set_variable(
    "u",
    &(0..nu)
      .map(|i| 1e-3 * (0.3 * i as f64).sin())
      .collect::<Vec<_>>(),
  )
  .unwrap();
  ws.add_internal_im_variable("p", &mimd1).unwrap();
  ws.add_scalar_data("G", 1.0).unwrap();
  ws.add_scalar_data("K", 1.0).unwrap();
  let n = mimd1.nb_values().unwrap();
  ws.set_variable(
    "p",
    &(0..n)
      .map(|i| 0.01 * (0.7 * i as f64).cos())
      .collect::<Vec<_>>(),
  )
  .unwrap();
  let terms = if two {
    ws.add_internal_im_variable("eps", &mimd3).unwrap();
    let n3 = mimd3.nb_values().unwrap();
    ws.set_variable(
      "eps",
      &(0..n3)
        .map(|i| 1e-3 * (0.4 * i as f64).sin())
        .collect::<Vec<_>>(),
    )
    .unwrap();
    vec![
      "(-p*Id(2)+2*G*([[2/3,0;0,-1/3],[-1/3,0;0,2/3],[0,1;1,0]].eps)):Grad_Test_u",
      "(eps-Grad_u:[[1,0;0,0],[0,0;0,1],[0,0.5;0.5,0]]).Test_eps",
      "((p+1e3*pow(p,3))+K*Div_u+eps(1)*eps(2))*Test_p",
    ]
  } else {
    vec![
      "(-p*Id(meshdim)+2*G*(Sym(Grad_u)-Div_u*Id(meshdim)/3)):Grad_Test_u",
      "((p+1e3*pow(p,3))+K*Div_u)*Test_p",
    ]
  };
  (ws, mim, terms)
}

fn check(m: &Mesh, two: bool, reduced: bool) {
  let (mut ws, mim, terms) = setup(m, two, reduced);
  let (n, np) = (ws.nb_dof(), ws.nb_primary_dof());
  assert!(ws.is_internal_variable("p") && np < n);
  assert_eq!(ws.interval_of_variable("u").unwrap(), 0..np);
  // the retained system (all unknowns)
  let mut r = vec![0.0; n];
  let mut k = vec![vec![0.0; n]; n];
  for t in &terms {
    let v = {
      ws.clear_expressions();
      ws.add_expression(t, &mim, -1).unwrap();
      ws.assembly(1, false).unwrap();
      ws.assembled_vector()
    };
    r.iter_mut().zip(v).for_each(|(a, b)| *a += b);
    let m = {
      ws.clear_expressions();
      ws.add_expression(t, &mim, -1).unwrap();
      ws.assembly(2, false).unwrap();
      ws.assembled_matrix()
    }
    .to_csr()
    .to_dense();
    k.iter_mut()
      .zip(&m)
      .for_each(|(a, b)| a.iter_mut().zip(b).for_each(|(x, y)| *x += y));
  }
  let (p, q) = (0..np, np..n);
  let kqq_inv = inverse(&sub(&k, q.clone(), q.clone()));
  let kp = mat(&kqq_inv, &sub(&k, q.clone(), p.clone()));
  let rq: Vec<Vec<f64>> = r[q.clone()].iter().map(|&x| vec![x]).collect();
  let rp = mat(&kqq_inv, &rq);
  let mut khat = sub(&k, p.clone(), p.clone());
  let corr = mat(&sub(&k, p.clone(), q.clone()), &kp);
  khat
    .iter_mut()
    .zip(&corr)
    .for_each(|(a, b)| a.iter_mut().zip(b).for_each(|(x, y)| *x -= y));
  let rhs = mat(&sub(&k, p.clone(), q.clone()), &rp);

  ws.clear_expressions();
  for t in terms {
    ws.add_expression(t, &mim, -1).unwrap();
  }
  ws.set_assembled_vector(r.clone());
  ws.assembly(2, true).unwrap();
  let kc = sub(
    &ws.assembled_matrix().to_csr().to_dense(),
    p.clone(),
    p.clone(),
  );
  let s = khat.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs()));
  assert!(
    max_diff(&kc, &khat) < 1e-10 * s,
    "K^ {:e} (scale {s:e})",
    max_diff(&kc, &khat)
  );
  let kpc = sub(
    &ws.internal_coupling_matrix().to_csr().to_dense(),
    q.clone(),
    p.clone(),
  );
  assert!(max_diff(&kpc, &kp) < 1e-10, "K' {:e}", max_diff(&kpc, &kp));
  let v = ws.assembled_vector();
  let rpc: Vec<Vec<f64>> = v[q.clone()].iter().map(|&x| vec![x]).collect();
  assert!(max_diff(&rpc, &rp) < 1e-12, "R' {:e}", max_diff(&rpc, &rp));
  let rhsc: Vec<Vec<f64>> = v[p].iter().map(|&x| vec![-x]).collect();
  assert!(
    max_diff(&rhsc, &rhs) < 1e-12,
    "rhs {:e}",
    max_diff(&rhsc, &rhs)
  );
}

#[test]
fn schur_complement_one_internal_variable() {
  check(&qk_mesh(2, 1), false, false);
}

#[test]
fn schur_complement_on_a_reduced_space() {
  check(&qk_mesh(2, 1), false, true);
  check(&qk_mesh(2, 1), true, true);
}

#[test]
fn schur_complement_cluster_of_two() {
  check(&qk_mesh(2, 1), true, false);
}

#[test]
fn schur_complement_on_mixed_meshes() {
  // triangles and quadrilaterals, of degree 1 and 2 (against the Schur
  // complement only: GetFEM's condensation failed on mixed meshes in the
  // reference build)
  for mi in [0, 1] {
    let m = common::mixed::mixed_mesh(mi, true);
    for (two, reduced) in [(false, false), (true, false), (false, true)] {
      check(&m, two, reduced);
    }
  }
}

#[test]
fn condensation_on_threads() {
  // the QK(2,2) mesh; the mixed ones (3D: p only)
  let cases = [
    (qk_mesh(4, 2), false, false),
    (qk_mesh(4, 2), false, true),
    (qk_mesh(4, 2), true, false),
    (common::mixed::mixed_mesh(0, true), true, true),
    (common::mixed::mixed_mesh(1, true), true, false),
    (common::mixed::mixed_mesh(2, true), false, true),
  ];
  for (m, two, reduced) in &cases {
    let (two, reduced) = (*two, *reduced);
    let (mut ws, mim, terms) = setup(m, two, reduced);
    let residual: Vec<f64> = (0..ws.nb_dof()).map(|i| (0.9 * i as f64).sin()).collect();
    let tag = format!(
      "{}D, {} convexes, two {two} reduced {reduced}",
      m.dim(),
      m.nb_convexes()
    );
    common::as_one_thread(&tag, || {
      ws.clear_expressions();
      for t in &terms {
        ws.add_expression(t, &mim, -1).unwrap();
      }
      ws.set_assembled_vector(residual.clone());
      common::assembled(&mut ws, 2, true)
    });
  }
}
