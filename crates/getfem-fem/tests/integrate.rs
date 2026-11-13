//! M3 exit: exact integration of polynomials through MeshIm, MeshFem and the
//! geometry, without GWFL. Unit squares/cubes meshed with every element type.

use getfem_fem::integration::classical_approx_im;
use getfem_fem::mesh_fem::MeshFem;
use getfem_fem::mesh_im::MeshIm;
use getfem_fem::precomp::fem_precomp;
use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::mesh::{Mesh, MeshData};
use getfem_geom::region::outer_faces_of_mesh;
use getfem_geom::regular_mesh::regular_mesh;

/// f = prod x_i^p_i
fn mono(p: &[i32], x: &[f64]) -> f64 {
  p.iter().zip(x).map(|(&k, &v)| v.powi(k)).product()
}

fn grad_mono(p: &[i32], x: &[f64]) -> Vec<f64> {
  (0..p.len())
    .map(|d| {
      if p[d] == 0 {
        0.0
      } else {
        p[d] as f64
          * (0..p.len())
            .map(|e| {
              if e == d {
                x[e].powi(p[e] - 1)
              } else {
                x[e].powi(p[e])
              }
            })
            .product::<f64>()
      }
    })
    .collect()
}

/// exact integrals over [0,1]^n
fn exact(p: &[i32]) -> f64 {
  p.iter().map(|&k| 1.0 / (k as f64 + 1.0)).product()
}

fn exact_grad2(p: &[i32]) -> f64 {
  // |grad f|^2 = sum_d p_d^2 x_d^(2p_d-2) prod_{e != d} x_e^(2p_e)
  (0..p.len())
    .filter(|&d| p[d] > 0)
    .map(|d| {
      (p[d] * p[d]) as f64
        * (0..p.len())
          .map(|e| {
            if e == d {
              1.0 / (2 * p[e] - 1) as f64
            } else {
              1.0 / (2 * p[e] + 1) as f64
            }
          })
          .product::<f64>()
    })
    .sum()
}

/// boundary of [0,1]^n: faces x_d = 0 and x_d = 1
fn exact_boundary(p: &[i32]) -> f64 {
  let n = p.len();
  (0..n)
    .map(|d| {
      let rest: f64 = (0..n)
        .filter(|&e| e != d)
        .map(|e| 1.0 / (p[e] as f64 + 1.0))
        .product();
      rest * (if p[d] == 0 { 2.0 } else { 1.0 })
    })
    .sum()
}

struct Case {
  mesh: Mesh,
  mim: MeshIm,
}

fn case(gt: &str, ns: &[usize], degree: usize) -> Case {
  let n = ns.len();
  let mesh = regular_mesh(
    &geotrans_descriptor(gt).unwrap(),
    ns,
    &vec![0.0; n],
    &vec![1.0; n],
  )
  .unwrap();
  let mim = MeshIm::new(&mesh).with_integration_method(degree).unwrap();
  Case { mesh, mim }
}

fn check(gt_name: &str, ns: &[usize], k: usize, p: &[i32]) {
  let deg = p.iter().sum::<i32>() as usize;
  let c = case(gt_name, ns, 2 * k.max(deg));
  let m = c.mesh.snapshot();
  let mf = MeshFem::new(&c.mesh, 1)
    .with_classical_finite_element(k)
    .unwrap();
  // nodal interpolation of f
  let u: Vec<f64> = (0..mf.nb_basic_dof().unwrap())
    .map(|d| mono(p, &mf.point_of_basic_dof(d).unwrap()))
    .collect();
  let (mut i_f, mut i_u, mut i_g) = (0.0, 0.0, 0.0);
  for cv in m.convex_index() {
    let (gt, g) = (m.trans_of_convex(cv), m.points_of_convex(cv));
    let pim = &c.mim.int_method_of_element(cv).unwrap();
    let a = pim.approx().unwrap();
    let pf = &mf.fem_of_element(cv).unwrap().unwrap();
    let pc = fem_precomp(pf, pim).unwrap();
    let dofs = mf.ind_basic_dof_of_element(cv).unwrap();
    for q in 0..a.nb_points_on_convex() {
      let pg = gt.at(&g, a.point(q)).unwrap();
      let w = a.coeff(q) * pg.j.abs();
      i_f += w * mono(p, &pg.x);
      let uh: f64 = dofs
        .iter()
        .enumerate()
        .map(|(i, &d)| u[d] * pc.val[q].data()[i])
        .sum();
      i_u += w * uh;
      // grad u_h = sum_i u_i B grad_ref phi_i
      let gr = &pc.grad[q];
      let nb = pf.nb_dof();
      let gref: Vec<f64> = (0..gt.dim())
        .map(|j| {
          dofs
            .iter()
            .enumerate()
            .map(|(i, &d)| u[d] * gr.data()[i + nb * j])
            .sum()
        })
        .collect();
      let gx: Vec<f64> = (0..m.dim())
        .map(|r| (0..gt.dim()).map(|j| pg.b[(r, j)] * gref[j]).sum())
        .collect();
      i_g += w * gx.iter().map(|v| v * v).sum::<f64>();
      // gradient pointwise equals the exact one when the space contains f
      if deg <= k {
        let ge = grad_mono(p, &pg.x);
        assert!(
          gx.iter().zip(&ge).all(|(a, b)| (a - b).abs() < 1e-10),
          "{gt_name} k={k} {p:?}: grad {gx:?} vs {ge:?}"
        );
      }
    }
  }
  let tag = format!("{gt_name} {ns:?} k={k} x^{p:?}");
  assert!(
    (i_f - exact(p)).abs() < 1e-13,
    "{tag}: int f = {i_f} vs {}",
    exact(p)
  );
  if deg <= k {
    assert!(
      (i_u - exact(p)).abs() < 1e-12,
      "{tag}: int u_h = {i_u} vs {}",
      exact(p)
    );
    assert!(
      (i_g - exact_grad2(p)).abs() < 1e-10,
      "{tag}: int |grad u_h|^2 = {i_g} vs {}",
      exact_grad2(p)
    );
  }
  // boundary: sum over outer faces of w |J| |B n_ref| f
  let mut i_b = 0.0;
  for (cv, f) in outer_faces_of_mesh(&m).resolve(&m).unwrap().iter() {
    let f = f.unwrap();
    let (gt, g) = (m.trans_of_convex(cv), m.points_of_convex(cv));
    let pim = c.mim.int_method_of_element(cv).unwrap();
    let a = pim.approx().unwrap();
    for q in 0..a.nb_points_on_face(f) {
      let pg = gt.at(&g, a.point_on_face(f, q)).unwrap();
      let nr = &gt.normals()[f];
      let bn: Vec<f64> = (0..m.dim())
        .map(|r| (0..gt.dim()).map(|j| pg.b[(r, j)] * nr[j]).sum())
        .collect();
      i_b += a.coeff_on_face(f, q)
        * pg.j.abs()
        * bn.iter().map(|v| v * v).sum::<f64>().sqrt()
        * mono(p, &pg.x);
    }
  }
  assert!(
    (i_b - exact_boundary(p)).abs() < 1e-12,
    "{tag}: boundary {i_b} vs {}",
    exact_boundary(p)
  );
}

#[test]
fn triangles_and_quads() {
  for k in 1..=3 {
    for p in [[0, 0], [1, 0], [0, 1], [1, 1], [2, 1], [0, 3], [2, 2]] {
      check("GT_PK(2,1)", &[3, 2], k, &p);
      check("GT_QK(2,1)", &[3, 2], k, &p);
    }
  }
}

#[test]
fn tetrahedra_hexahedra_prisms() {
  for k in 1..=2 {
    for p in [
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 1],
      [2, 0, 0],
      [1, 1, 1],
      [0, 0, 3],
    ] {
      check("GT_PK(3,1)", &[2, 2, 2], k, &p);
      check("GT_QK(3,1)", &[2, 1, 2], k, &p);
      check("GT_PRISM(3,1)", &[2, 1, 2], k, &p);
    }
  }
}

#[test]
fn pyramid_element() {
  // one pyramid on the unit square base: volume 1/3, integral of x = 1/6 (sympy)
  let gt = geotrans_descriptor("GT_PYRAMID(1)").unwrap();
  let mut m = MeshData::new(3);
  let pts: [&[f64]; 5] = [
    &[0., 0., 0.],
    &[1., 0., 0.],
    &[0., 1., 0.],
    &[1., 1., 0.],
    &[0.5, 0.5, 1.0],
  ];
  m.add_convex_by_points(&gt, &pts).unwrap();
  for d in 1..=4 {
    let pim = classical_approx_im(&gt, d).unwrap();
    let a = pim.approx().unwrap();
    let g = m.points_of_convex(0);
    let (mut vol, mut ix) = (0.0, 0.0);
    for q in 0..a.nb_points_on_convex() {
      let pg = gt.at(&g, a.point(q)).unwrap();
      vol += a.coeff(q) * pg.j.abs();
      ix += a.coeff(q) * pg.j.abs() * pg.x[0];
    }
    assert!(
      (vol - 1.0 / 3.0).abs() < 1e-14 && (ix - 1.0 / 6.0).abs() < 1e-14,
      "degree {d}: {vol} {ix}"
    );
  }
}

#[test]
fn missing_method_is_an_error() {
  let c = case("GT_PK(2,1)", &[1, 1], 1);
  let mim = MeshIm::new(&c.mesh);
  mim
    .set_integration_method(([0], &c.mim.int_method_of_element(0).unwrap()))
    .unwrap();
  assert!(mim.int_method_of_element(1).is_err());
}
