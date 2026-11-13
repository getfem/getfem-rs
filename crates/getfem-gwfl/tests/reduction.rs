//! Assembly on spaces with reduction matrices (GetFEM: unreduced terms
//! assembled on the basic DOFs, then E^T K E, E^T V; values E U): against
//! the assembly on the same space without reduction, with the variable's
//! basic values set to E u.

mod common;

use std::collections::BTreeSet;

use getfem_fem::mesh_fem::MeshFem;
use getfem_fem::mesh_im::MeshIm;
use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::mesh::Mesh;
use getfem_geom::region::{RegionArg, inner_faces};
use getfem_geom::regular_mesh::regular_unit_mesh;
use getfem_gwfl::workspace::Workspace;
use getfem_linalg::{Coo, Csr};

fn mesh() -> Mesh {
  regular_unit_mesh(&[2, 2], &geotrans_descriptor("GT_PK(2,1)").unwrap()).unwrap()
}

fn seq(n: usize, s: f64) -> Vec<f64> {
  (0..n).map(|i| (s * (i as f64 + 1.0)).sin()).collect()
}

/// a rotation of the two components at each node, a different angle per
/// node (basic DOF 2 i + c): R = E^T, square
fn rotation(nb: usize) -> (Csr<f64>, Csr<f64>) {
  let mut e = Coo::new(nb, nb);
  for i in 0..nb / 2 {
    let t = 0.3 + 0.1 * i as f64;
    let (c, s) = (t.cos(), t.sin());
    let (a, b) = (2 * i, 2 * i + 1);
    for (r, k, v) in [(a, a, c), (a, b, -s), (b, a, s), (b, b, c)] {
      e.push(r, k, v);
    }
  }
  let e = e.to_csr().unwrap();
  (e.transpose(), e)
}

/// workspace with u on mf_u (values u), p on a scalar P1 space
fn ws(m: &Mesh, mf_u: &MeshFem, u: &[f64]) -> Workspace {
  let mf_p = MeshFem::new(m, 1).with_classical_finite_element(1).unwrap();
  let np = mf_p.nb_dof().unwrap();
  let mut ws = Workspace::new();
  ws.add_fem_variable("u", mf_u).unwrap();
  ws.set_variable("u", u).unwrap();
  ws.add_fem_variable("p", &mf_p).unwrap();
  ws.set_variable("p", &seq(np, 1.3)).unwrap();
  ws
}

/// P (basic system -> reduced system): E on u, identity on p
fn p_matrix(e: &Csr<f64>, np: usize) -> Csr<f64> {
  let (nb, nr) = (e.nrows, e.ncols);
  let mut p = Coo::new(nb + np, nr + np);
  for i in 0..nb {
    for (j, v) in e.row(i) {
      p.push(i, j, v);
    }
  }
  for k in 0..np {
    p.push(nb + k, nr + k, 1.0);
  }
  p.to_csr().unwrap()
}

fn assert_close(a: &[f64], b: &[f64], what: &str) {
  let s = b.iter().fold(1.0f64, |m, x| m.max(x.abs()));
  let err = a
    .iter()
    .zip(b)
    .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
  assert!(
    a.len() == b.len() && err <= 1e-12 * s,
    "{what}: differs by {err:e}"
  );
}

fn dense(k: &Csr<f64>) -> Vec<f64> {
  k.to_dense().concat()
}

const MATRIX: &[&str] = &[
  "Grad_Test_u:Grad_Test2_u",
  "Div_Test_u*Test2_p + Test_p*Div_Test2_u + Grad_Test_p.Grad_Test2_p",
  "(3*Trace(Grad_Test_u)*Id(qdim(u)) + 2*(Grad_Test_u'+Grad_Test_u)):Grad_Test2_u",
  "(Grad_Test_u*Grad_u):Grad_Test2_u + p*Test_u.Test2_u",
];
const VECTOR: &[&str] = &[
  "u.Test_u + p*Div_Test_u + Grad_p.Grad_Test_p",
  "(Grad_u*u).Test_u + Div_u*Test_p",
];
const SCALAR: &[&str] = &["u.u + p*Div_u", "Norm_sqr(Grad_u)"];

/// the reduced assembly against E^T K E (P^T on the system), on region
fn check(
  mf_r: &MeshFem,
  mf_b: &MeshFem,
  e: &Csr<f64>,
  mim: &MeshIm,
  region: RegionArg,
  exprs: [&[&str]; 3],
) {
  let m = mf_b.linked_mesh();
  let ur = seq(e.ncols, 0.7);
  let ub = e.matvec(&ur).unwrap();
  let (mut wr, mut wb) = (ws(&m, mf_r, &ur), ws(&m, mf_b, &ub));
  let np = wb.interval_of_variable("p").unwrap().len();
  let p = p_matrix(e, np);
  let pt = p.transpose();
  for x in exprs[2] {
    let a = {
      wr.clear_expressions();
      wr.add_expression(x, mim, region.clone()).unwrap();
      wr.assembly(2, false).unwrap();
      wr.assembled_matrix()
    }
    .to_csr();
    let b = {
      wb.clear_expressions();
      wb.add_expression(x, mim, region.clone()).unwrap();
      wb.assembly(2, false).unwrap();
      wb.assembled_matrix()
    }
    .to_csr();
    let want = pt.mul(&b).unwrap().mul(&p).unwrap();
    assert_close(&dense(&a), &dense(&want), x);
  }
  for x in exprs[1] {
    let a = {
      wr.clear_expressions();
      wr.add_expression(x, mim, region.clone()).unwrap();
      wr.assembly(1, false).unwrap();
      wr.assembled_vector()
    };
    let b = {
      wb.clear_expressions();
      wb.add_expression(x, mim, region.clone()).unwrap();
      wb.assembly(1, false).unwrap();
      wb.assembled_vector()
    };
    assert_close(a, &pt.matvec(b).unwrap(), x);
  }
  for x in exprs[0] {
    let a = {
      wr.clear_expressions();
      wr.add_expression(x, mim, region.clone()).unwrap();
      wr.assembly(0, false).unwrap();
      wr.assembled_potential()
    };
    let b = {
      wb.clear_expressions();
      wb.add_expression(x, mim, region.clone()).unwrap();
      wb.assembly(0, false).unwrap();
      wb.assembled_potential()
    };
    assert_close(&[a], &[b], x);
  }
}

fn vector_space(m: &Mesh) -> MeshFem {
  MeshFem::new(m, 2).with_classical_finite_element(2).unwrap()
}

#[test]
fn rotated_components() {
  // a square reduction (same number of DOFs): the components rotated at
  // each node
  let m = mesh();
  let mim = MeshIm::new(&m).with_integration_method(4).unwrap();
  let (mf_r, mf_b) = (vector_space(&m), vector_space(&m));
  let (r, e) = rotation(mf_b.nb_basic_dof().unwrap());
  mf_r.set_reduction_matrices(Some(&r), &e).unwrap();
  check(
    &mf_r,
    &mf_b,
    &e,
    &mim,
    (-1).into(),
    [SCALAR, VECTOR, MATRIX],
  );
}

#[test]
fn selection_as_reduction_and_partial_space() {
  // reduce_to_basic_dof and a partial space on the same DOFs: the same
  // selection matrices
  let m = mesh();
  let mim = MeshIm::new(&m).with_integration_method(4).unwrap();
  let (mf_r, mf_b) = (vector_space(&m), vector_space(&m));
  let nb = mf_b.nb_basic_dof().unwrap();
  let kept: BTreeSet<usize> = (0..nb).filter(|d| d % 3 != 1).collect();
  mf_r.reduce_to_basic_dof(&kept).unwrap();
  let e = mf_r.extension_matrix().unwrap().unwrap();
  check(
    &mf_r,
    &mf_b,
    &e,
    &mim,
    (-1).into(),
    [SCALAR, VECTOR, MATRIX],
  );
  let part = MeshFem::partial(&vector_space(&m), &kept).unwrap();
  check(
    &part,
    &mf_b,
    &e,
    &mim,
    (-1).into(),
    [SCALAR, VECTOR, MATRIX],
  );
}

#[test]
fn reduced_variable_through_a_transformation() {
  // u seen from the neighbor element on the inner faces: its values and
  // tests on the target element through the reduction
  let m = mesh();
  let mim = MeshIm::new(&m).with_integration_method(4).unwrap();
  let (mf_r, mf_b) = (vector_space(&m), vector_space(&m));
  let (r, e) = rotation(mf_b.nb_basic_dof().unwrap());
  mf_r.set_reduction_matrices(Some(&r), &e).unwrap();
  let inner = inner_faces(&m, -1).unwrap();
  check(
    &mf_r,
    &mf_b,
    &e,
    &mim,
    inner.into(),
    [
      &["Interpolate(u,neighbor_element).u"],
      &["Interpolate(u,neighbor_element).Test_u + Interpolate(Test_u,neighbor_element).u"],
      &[
        "Interpolate(Test_u,neighbor_element).Test2_u + Test_u.Interpolate(Test2_u,neighbor_element)",
      ],
    ],
  );
}

#[test]
fn reduction_on_threads() {
  let m = mesh();
  let mim = MeshIm::new(&m).with_integration_method(4).unwrap();
  let (mf_r, mf_b) = (vector_space(&m), vector_space(&m));
  let (r, e) = rotation(mf_b.nb_basic_dof().unwrap());
  mf_r.set_reduction_matrices(Some(&r), &e).unwrap();
  let mut wr = ws(&m, &mf_r, &seq(e.ncols, 0.7));
  let inner = inner_faces(&m, -1).unwrap();
  let neighbor = [
    "Interpolate(u,neighbor_element).u",
    "Interpolate(u,neighbor_element).Test_u + Interpolate(Test_u,neighbor_element).u",
    "Interpolate(Test_u,neighbor_element).Test2_u + Test_u.Interpolate(Test2_u,neighbor_element)",
  ];
  for (order, exprs) in [SCALAR, VECTOR, MATRIX].into_iter().enumerate() {
    let all = exprs.iter().map(|x| (*x, RegionArg::from(-1)));
    for (x, rg) in all.chain([(neighbor[order], inner.clone().into())]) {
      common::as_one_thread(&format!("{x} order {order}"), || {
        wr.clear_expressions();
        wr.add_expression(x, &mim, rg.clone()).unwrap();
        common::assembled(&mut wr, order, false)
      });
    }
  }
}
