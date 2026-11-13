//! Integration on cut elements (mesh_im_level_set; 2D our continuous
//! cut-cell cubature, 3D GetFEM's sub-simplices): measures and moments of a
//! disc on triangles and quadrilaterals and of a ball on tetrahedra and
//! hexahedra against the exact values, the length of the circle, the area
//! of the sphere, and the continuity of the integrals when the level set
//! moves (3D: bounded jumps).

use getfem_fem::integration::{classical_approx_im, integ_descriptor};
use getfem_fem::level_set::LevelSet;
use getfem_fem::mesh_im_level_set::{CELL_TOLERANCE, IntegrateWhere, MeshImLevelSet};
use getfem_fem::mesh_level_set::{MeshLevelSet, jacobian};
use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::mesh::Mesh;
use getfem_geom::regular_mesh::regular_mesh;
use std::f64::consts::PI;

fn mesh(gt: &str) -> Mesh {
  regular_mesh(
    &geotrans_descriptor(gt).unwrap(),
    &[8, 8],
    &[-1.0, -1.0],
    &[2.0, 2.0],
  )
  .unwrap()
}

fn mesh3(gt: &str) -> Mesh {
  mesh3n(gt, 4)
}

/// [-1, 1]^3, n^3 cubes
fn mesh3n(gt: &str, n: usize) -> Mesh {
  regular_mesh(
    &geotrans_descriptor(gt).unwrap(),
    &[n; 3],
    &[-1.0; 3],
    &[2.0; 3],
  )
  .unwrap()
}

fn circle(m: &Mesh, r: f64) -> LevelSet {
  let ls = LevelSet::new(m, 2, false).unwrap();
  let mf = ls.mesh_fem();
  let v: Vec<f64> = (0..mf.nb_dof().unwrap())
    .map(|i| {
      let p = mf.point_of_basic_dof(i).unwrap();
      p.iter().map(|v| v * v).sum::<f64>() - r * r
    })
    .collect();
  ls.set_values(0, &v).unwrap();
  ls
}

/// integral of f with the method of each element
fn integrate(m: &Mesh, mim: &MeshImLevelSet, f: &dyn Fn(&[f64]) -> f64) -> f64 {
  let md = m.snapshot();
  let mut s = 0.0;
  for cv in md.convex_index() {
    let pim = mim.int_method_of_element(cv).unwrap();
    let a = pim.approx().unwrap();
    let (gt, g) = (md.trans_of_convex(cv), md.points_of_convex(cv));
    for i in 0..a.nb_points_on_convex() {
      let x = gt.transform(a.point(i), &g).unwrap();
      s += a.coeff(i) * jacobian(gt, &g, a.point(i)).abs() * f(&x);
    }
  }
  s
}

/// lattice n (3D: 1 until the 3D lattice)
fn mim_of(m: &Mesh, r: f64, w: IntegrateWhere, n: usize) -> MeshImLevelSet {
  let mut mls = MeshLevelSet::new(m);
  mls.add_level_set(&circle(m, r));
  let mim = MeshImLevelSet::new(&mls, w).unwrap();
  mim.set_lattice(n).unwrap();
  mim.set_integration_method(4usize).unwrap();
  // 3D: the rule on the sub-tetrahedra of the cut (hexahedra need one)
  if m.dim() == 3 {
    let gt = geotrans_descriptor("GT_PK(3,1)").unwrap();
    mim
      .set_simplex_im(&classical_approx_im(&gt, 4).unwrap(), None, CELL_TOLERANCE)
      .unwrap();
  }
  mim.adapt().unwrap();
  mim
}

#[test]
fn disc_measures_and_moments() {
  let r = 0.63;
  for gt in ["GT_PK(2,1)", "GT_QK(2,1)"] {
    let m = mesh(gt);
    let inside = mim_of(&m, r, IntegrateWhere::Inside, 2);
    // the curve: quadratic pieces (errors ~ h^4: n = 2 gives 3e-6)
    let a = integrate(&m, &inside, &|_| 1.0);
    assert!((a / (PI * r * r) - 1.0).abs() < 1e-5, "{gt} area {a}");
    let i2 = integrate(&m, &inside, &|x| x[0] * x[0] + x[1] * x[1]);
    assert!(
      (i2 / (PI * r.powi(4) / 2.0) - 1.0).abs() < 2e-5,
      "{gt} second moment {i2}"
    );
    let outside = mim_of(&m, r, IntegrateWhere::Outside, 2);
    let b = integrate(&m, &outside, &|_| 1.0);
    assert!(
      (a + b - 4.0).abs() < 1e-12,
      "{gt} inside + outside {}",
      a + b
    );
    let bd = mim_of(&m, r, IntegrateWhere::Boundary, 2);
    let l = integrate(&m, &bd, &|_| 1.0);
    assert!((l / (2.0 * PI * r) - 1.0).abs() < 2e-6, "{gt} length {l}");
  }
}

#[test]
fn continuous_in_the_level_set() {
  // a step of the integral over a sweep of the radius must shrink with the
  // step of the radius (bisection of each interval): no jump
  let m = mesh("GT_PK(2,1)");
  let f = |x: &[f64]| (2.0 * x[0]).cos() * (x[1] + 0.3).exp();
  let at = |r: f64| integrate(&m, &mim_of(&m, r, IntegrateWhere::Inside, 2), &f);
  for k in 0..4 {
    let (mut a, mut b) = (0.5 + 0.03 * k as f64, 0.53 + 0.03 * k as f64);
    let (mut ia, mut ib) = (at(a), at(b));
    for _ in 0..16 {
      let c = 0.5 * (a + b);
      let ic = at(c);
      if (ic - ia).abs() > (ib - ic).abs() {
        b = c;
        ib = ic;
      } else {
        a = c;
        ia = ic;
      }
    }
    // |dI/dr| <= 2 pi r e^1.3 < 15: a step of 0.03 / 2^16 moves I by < 1e-5
    assert!(
      (ib - ia).abs() < 1e-5,
      "jump {} at r = {a}",
      (ib - ia).abs()
    );
  }
}

#[test]
fn ball_measures_and_moments() {
  let r: f64 = 0.63;
  for gt in ["GT_PK(3,1)", "GT_QK(3,1)"] {
    let m = mesh3(gt);
    let inside = mim_of(&m, r, IntegrateWhere::Inside, 1);
    let v = integrate(&m, &inside, &|_| 1.0);
    let ve = 4.0 * PI * r.powi(3) / 3.0;
    // measured: PK 1.3e-6, QK 1.0e-5 (x^2: 2.4e-6, 1.7e-5; area 4e-7, 6e-6)
    assert!((v / ve - 1.0).abs() < 3e-5, "{gt} volume {v} vs {ve}");
    let ix = integrate(&m, &inside, &|x| x[0] * x[0]);
    let ixe = 4.0 * PI * r.powi(5) / 15.0;
    assert!((ix / ixe - 1.0).abs() < 3e-5, "{gt} x^2 {ix} vs {ixe}");
    let outside = mim_of(&m, r, IntegrateWhere::Outside, 1);
    let w = integrate(&m, &outside, &|_| 1.0);
    // the curved sub-tetrahedra overlap slightly (as GetFEM's): PK 3.9e-6
    assert!(
      (v + w - 8.0).abs() < 1e-5,
      "{gt} inside + outside {}",
      v + w
    );
    let bd = mim_of(&m, r, IntegrateWhere::Boundary, 1);
    let a = integrate(&m, &bd, &|_| 1.0);
    let ae = 4.0 * PI * r * r;
    assert!((a / ae - 1.0).abs() < 2e-5, "{gt} area {a} vs {ae}");
  }
}

/// The rule of the cut cells by their size (tau of set_simplex_im): a
/// smooth function on the ball with the degree-5 rule on every cell (tau
/// 0), then with the default tau 1e-2: 2.4 times fewer points (the cut
/// graded toward curved level sets: few small cells), the same integral
/// to 1e-5
#[test]
fn cell_rules_by_size() {
  let r: f64 = 0.63;
  let m = mesh3("GT_PK(3,1)");
  let f = |x: &[f64]| (2.0 * x[0]).cos() * (1.5 * x[1]).cos() * (x[2] + 0.3).exp();
  let tet5 = integ_descriptor("IM_TETRAHEDRON(5)").unwrap();
  let run = |tau: f64| {
    let mut mls = MeshLevelSet::new(&m);
    mls.add_level_set(&circle(&m, r));
    let mim = MeshImLevelSet::new(&mls, IntegrateWhere::Inside).unwrap();
    mim.set_integration_method(&tet5).unwrap();
    mim.set_simplex_im(&tet5, None, tau).unwrap();
    mim.adapt().unwrap();
    let md = m.snapshot();
    let np: usize = md
      .convex_index()
      .map(|cv| {
        mim
          .int_method_of_element(cv)
          .unwrap()
          .approx()
          .unwrap()
          .nb_points()
      })
      .sum();
    (integrate(&m, &mim, &f), np)
  };
  let (all, n_all) = run(0.0);
  let (sized, n_sized) = run(CELL_TOLERANCE);
  assert!(
    (sized / all - 1.0).abs() < 1e-5,
    "sized {sized} vs {all} ({n_sized} vs {n_all} points)"
  );
  assert!(2 * n_sized < n_all, "{n_sized} vs {n_all} points");
  assert!(MeshImLevelSet::new(&MeshLevelSet::new(&m), IntegrateWhere::All).is_err());
}

#[test]
fn continuous_in_the_level_set_3d() {
  let m = mesh3("GT_PK(3,1)");
  let f = |x: &[f64]| (2.0 * x[0]).cos() * (x[1] + 0.3).exp() * (1.0 + x[2]);
  let at = |r: f64| integrate(&m, &mim_of(&m, r, IntegrateWhere::Inside, 1), &f);
  let (mut a, mut b) = (0.6, 0.66);
  let (mut ia, mut ib) = (at(a), at(b));
  for _ in 0..10 {
    let c = 0.5 * (a + b);
    let ic = at(c);
    if (ic - ia).abs() > (ib - ic).abs() {
      b = c;
      ib = ic;
    } else {
      a = c;
      ia = ic;
    }
  }
  // |dI/dr| < 4 pi r^2 e^1.3 * 2 < 40: a step of 0.06 / 2^10 moves I by
  // < 3e-3; the sub-simplices jump by their integration error when the cut
  // changes (measured: the largest step 3.5e-4)
  assert!(
    (ib - ia).abs() < 3e-3,
    "jump {} at r = {a}",
    (ib - ia).abs()
  );
}

/// GetFEM's test_mesh_im_level_set (test_2d): three level sets on a disc
/// of curved triangles (degree 3); circles 2 and 3 touch circle 1 at
/// (0, 0.4).
#[test]
fn getfem_test_mesh_im_level_set() {
  let meshes = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/meshes");
  let m =
    getfem_geom::mesh_file::read_mesh_file(&format!("{meshes}/disc_2D_degree3.mesh")).unwrap();
  let ls_of = |c: [f64; 2], r: f64, sign: f64| {
    let ls = LevelSet::new(&m, 2, false).unwrap();
    let mf = ls.mesh_fem();
    let v: Vec<f64> = (0..mf.nb_dof().unwrap())
      .map(|i| {
        let p = mf.point_of_basic_dof(i).unwrap();
        sign * ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) - r * r)
      })
      .collect();
    ls.set_values(0, &v).unwrap();
    ls
  };
  let (r1, r2, r3) = (0.4, 0.1, 0.08);
  let mut mls = MeshLevelSet::new(&m);
  mls.add_level_set(&ls_of([0.0, 0.0], r1, 1.0));
  mls.add_level_set(&ls_of([0.0, 0.3], r2, 1.0));
  mls.add_level_set(&ls_of([0.0, 0.48], r3, -1.0));
  let mim = |w: IntegrateWhere, csg: &str| {
    let mim = MeshImLevelSet::new(&mls, w).unwrap();
    if !csg.is_empty() {
      mim.set_level_set_boolean_operations(csg).unwrap();
    }
    let im = getfem_fem::integration::integ_descriptor("IM_TRIANGLE(6)").unwrap();
    mim.set_integration_method(&im).unwrap();
    mim.adapt().unwrap();
    mim
  };
  // GetFEM's check: the area of the largest circle by its indicator,
  // integrated with the rule on all the parts (1e-3; measured relative:
  // lattice 1: 4e-4, 2: 4e-5, 3: 8e-6, 4: 3e-6)
  let all = mim(IntegrateWhere::All, "");
  let a = integrate(&m, &all, &|x| f64::from(u8::from(x[0].hypot(x[1]) <= r1)));
  assert!((a - PI * r1 * r1).abs() < 1e-3, "indicator: {a}");
  // the exact measures (default lattice)
  let rel = |a: f64, e: f64| (a / e - 1.0).abs();
  let a = integrate(&m, &mim(IntegrateWhere::Inside, "a"), &|_| 1.0);
  assert!(rel(a, PI * r1 * r1) < 1e-4, "disc 1: {a}");
  // the default: inside all (disc 2 is inside disc 1 and touches disc 3)
  let a = integrate(&m, &mim(IntegrateWhere::Inside, ""), &|_| 1.0);
  // circles 2 and 3 touch circle 1 (and each other) at one point; the
  // tangency costs nothing (disc 2 alone: 1.3e-2): the radii 0.1 and 0.08
  // are close to the lattice size (measured: 1.1e-2, 6.9e-4, 7.7e-3,
  // 6.9e-4; lattice 8: disc 2 4e-5)
  assert!(rel(a, PI * r2 * r2) < 2e-2, "disc 2: {a}");
  // disc 1 minus disc 2, union with the outside of disc 3 complemented
  let a = integrate(&m, &mim(IntegrateWhere::Inside, "a-b"), &|_| 1.0);
  assert!(rel(a, PI * (r1 * r1 - r2 * r2)) < 1e-3, "a-b: {a}");
  let a = integrate(&m, &mim(IntegrateWhere::Inside, "b+!c"), &|_| 1.0);
  assert!(rel(a, PI * (r2 * r2 + r3 * r3)) < 1e-2, "b+!c: {a}");
  // all the parts: inside and outside
  let a = integrate(&m, &mim(IntegrateWhere::All, ""), &|_| 1.0);
  let e = integrate(&m, &mim(IntegrateWhere::Inside, ""), &|_| 1.0)
    + integrate(&m, &mim(IntegrateWhere::Outside, ""), &|_| 1.0);
  assert!((a - e).abs() < 1e-12, "all: {a} vs {e}");
  // the circles: 1, and the boundary of a-b (circles 1 and 2)
  let l = integrate(&m, &mim(IntegrateWhere::Boundary, "a"), &|_| 1.0);
  assert!(rel(l, 2.0 * PI * r1) < 1e-4, "circle 1: {l}");
  let l = integrate(&m, &mim(IntegrateWhere::Boundary, "a-b"), &|_| 1.0);
  assert!(rel(l, 2.0 * PI * (r1 + r2)) < 1e-3, "a-b boundary: {l}");
}

/// Two crossing spheres a, b in 3D (GetFEM's set expressions): the
/// volumes of a*b (the lens), a-b, a+b and the area of the boundary of a-b
/// against the exact values, on 4^3 and 6^3 cubes (6^3: an element whose
/// first sub-mesh fails, the retries refined on the level sets too; was a
/// hang).
#[test]
fn two_spheres_3d() {
  two_spheres(4);
  two_spheres(6);
}

fn two_spheres(n: usize) {
  let m = mesh3n("GT_PK(3,1)", n);
  let sphere = |c: [f64; 3], r: f64| {
    let ls = LevelSet::new(&m, 2, false).unwrap();
    let mf = ls.mesh_fem();
    let v: Vec<f64> = (0..mf.nb_dof().unwrap())
      .map(|i| {
        let p = mf.point_of_basic_dof(i).unwrap();
        (0..3).map(|k| (p[k] - c[k]).powi(2)).sum::<f64>() - r * r
      })
      .collect();
    ls.set_values(0, &v).unwrap();
    ls
  };
  let (c1, c2, r1, r2) = ([-0.15, 0.02, 0.01], [0.25, 0.05, -0.02], 0.45f64, 0.4f64);
  let mut mls = MeshLevelSet::new(&m);
  mls.add_level_set(&sphere(c1, r1));
  mls.add_level_set(&sphere(c2, r2));
  let measure = |w: IntegrateWhere, csg: &str| {
    let mim = MeshImLevelSet::new(&mls, w).unwrap();
    mim.set_level_set_boolean_operations(csg).unwrap();
    mim.set_integration_method(4usize).unwrap();
    mim.adapt().unwrap();
    integrate(&m, &mim, &|_| 1.0)
  };
  let d = (0..3).map(|k| (c2[k] - c1[k]).powi(2)).sum::<f64>().sqrt();
  let lens = PI
    * (r1 + r2 - d).powi(2)
    * (d * d + 2.0 * d * r2 - 3.0 * r2 * r2 + 2.0 * d * r1 + 6.0 * r1 * r2 - 3.0 * r1 * r1)
    / (12.0 * d);
  let (va, vb) = (4.0 * PI * r1.powi(3) / 3.0, 4.0 * PI * r2.powi(3) / 3.0);
  let rel = |a: f64, e: f64| (a / e - 1.0).abs();
  for (csg, e) in [("a*b", lens), ("a-b", va - lens), ("a+b", va + vb - lens)] {
    let v = measure(IntegrateWhere::Inside, csg);
    // measured: 1.7e-6, 2.6e-7, 4.4e-7
    assert!(rel(v, e) < 1e-5, "n {n}, {csg}: {v} vs {e}");
  }
  // the plane of the circle where they cross, at x1 from c1: a without
  // its cap in b, plus the cap of b in a
  let x1 = (d * d + r1 * r1 - r2 * r2) / (2.0 * d);
  let ae = 4.0 * PI * r1 * r1 - 2.0 * PI * r1 * (r1 - x1) + 2.0 * PI * r2 * (r2 - (d - x1));
  let a = measure(IntegrateWhere::Boundary, "a-b");
  // measured: 3.5e-5 (faces along the crossing curve 1e-5 off the spheres)
  assert!(rel(a, ae) < 1e-4, "n {n}, a-b boundary: {a} vs {ae}");
}

/// a crack: primary f0, secondary f1 (degree k)
fn crack_ls(m: &Mesh, k: usize, f0: fn(&[f64]) -> f64, f1: fn(&[f64]) -> f64) -> LevelSet {
  let ls = LevelSet::new(m, k, true).unwrap();
  let mf = ls.mesh_fem();
  let pts: Vec<Vec<f64>> = (0..mf.nb_dof().unwrap())
    .map(|i| mf.point_of_basic_dof(i).unwrap())
    .collect();
  ls.set_values(0, &pts.iter().map(|p| f0(p)).collect::<Vec<_>>())
    .unwrap();
  ls.set_values(1, &pts.iter().map(|p| f1(p)).collect::<Vec<_>>())
    .unwrap();
  ls
}

/// Two crossing cracks whose fronts meet (both bounded by one sphere, the
/// crossing line ends on it): sub-cells with front vertices of both cracks
/// take the singular rule of one (GetFEM stops with "Two singular point in
/// one sub element"); the integrals over the cube stay exact
#[test]
fn cracks_with_meeting_fronts() {
  use getfem_fem::integration::integ_descriptor;
  let m = regular_mesh(
    &geotrans_descriptor("GT_QK(3,1)").unwrap(),
    &[3, 3, 3],
    &[-1.0; 3],
    &[2.0; 3],
  )
  .unwrap();
  let sphere = |x: &[f64]| x[0] * x[0] + (x[1] - 0.1).powi(2) + x[2] * x[2] - 0.6;
  let a = crack_ls(&m, 2, |x| x[1] - 0.1 - 0.3 * x[0], sphere);
  let b = crack_ls(&m, 2, |x| x[1] - 0.1 + 0.3 * x[0], sphere);
  let mut mls = MeshLevelSet::new(&m);
  mls.add_level_set(&a);
  mls.add_level_set(&b);
  mls.adapt().unwrap();
  // sub-cells with vertices on both fronts
  let mut mixed = 0;
  for &cv in mls.crack_tip_convexes().unwrap() {
    let sm = mls.mesh_of_convex(cv).unwrap();
    let fronts: Vec<_> = [&a, &b]
      .iter()
      .map(|l| (l.of_convex(cv, 0).unwrap(), l.of_convex(cv, 1).unwrap()))
      .collect();
    for i in sm.convex_index() {
      let ids = sm.ind_points_of_convex(i);
      let on = |k: usize| {
        [0, 2, 5, 9].iter().any(|&j| {
          let p = sm.point(ids[j]);
          fronts[k].0.val(p).unwrap().abs() < 1e-10 && fronts[k].1.val(p).unwrap().abs() < 1e-10
        })
      };
      mixed += usize::from(on(0) && on(1));
    }
  }
  assert!(mixed > 0, "no sub-cell on both fronts");
  let mim = MeshImLevelSet::new(&mls, IntegrateWhere::All).unwrap();
  let tet = integ_descriptor("IM_TETRAHEDRON(5)").unwrap();
  mim
    .set_integration_method(&integ_descriptor("IM_GAUSS_PARALLELEPIPED(3,4)").unwrap())
    .unwrap();
  mim
    .set_simplex_im(
      &tet,
      Some(&integ_descriptor("IM_PRODUCT(IM_TRIANGLE(6),IM_GAUSS1D(6))").unwrap()),
      CELL_TOLERANCE,
    )
    .unwrap();
  mim.adapt().unwrap();
  let v = integrate(&m, &mim, &|_| 1.0);
  let ix = integrate(&m, &mim, &|x| x[0] * x[0]);
  // the curved sub-tetrahedra overlap slightly (as GetFEM's)
  assert!((v - 8.0).abs() < 1e-5, "volume {v}");
  assert!((ix - 8.0 / 3.0).abs() < 1e-5, "x^2 {ix}");
}

/// The DOFs with support: Q1 integrated inside a disc, the DOFs of the
/// elements without a part inside it have none (their mass diagonal is 0)
#[test]
fn dofs_with_support_inside_a_disc() {
  let m = mesh("GT_QK(2,1)");
  let r = 0.6;
  let mut mls = MeshLevelSet::new(&m);
  mls.add_level_set(&circle(&m, r));
  mls.adapt().unwrap();
  let mim = MeshImLevelSet::new(&mls, IntegrateWhere::Inside).unwrap();
  mim.set_integration_method(4usize).unwrap();
  mim.adapt().unwrap();
  let mf = getfem_fem::mesh_fem::MeshFem::new(&m, 1)
    .with_classical_finite_element(1)
    .unwrap();
  let kept = mf.dofs_with_support(&mim, 1e-12).unwrap();
  // the elements whose square meets the open disc
  let md = m.snapshot();
  let mut want = std::collections::BTreeSet::new();
  for cv in md.convex_index() {
    let g = md.points_of_convex(cv);
    let (lo, hi): (Vec<f64>, Vec<f64>) = (0..2)
      .map(|a| {
        let v: Vec<f64> = (0..g.ncols()).map(|j| g[(a, j)]).collect();
        (
          v.iter().copied().fold(f64::MAX, f64::min),
          v.iter().copied().fold(f64::MIN, f64::max),
        )
      })
      .unzip();
    let d2: f64 = (0..2).map(|a| (0.0f64.clamp(lo[a], hi[a])).powi(2)).sum();
    if d2 < r * r {
      want.extend(mf.ind_basic_dof_of_element(cv).unwrap());
    }
  }
  assert!(!want.is_empty() && want.len() < mf.nb_dof().unwrap());
  assert_eq!(kept, want);
}
