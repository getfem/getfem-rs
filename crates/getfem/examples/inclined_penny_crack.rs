//! An inclined penny-shaped crack under remote tension: a crack of
//! radius a = 1 through the centre of a block, its normal tilted by
//! GAMMA from the z axis in the x-z plane, a uniaxial stress s along z
//! (tractions on the faces z = +-L), linearized elasticity (E = 1,
//! nu = 0.3). Mixed mode, all three stress intensity factors along the
//! front against the infinite body's (Kassir and Sih):
//!   K_I = 2 s_n sqrt(a / pi),
//!   K_II = 4 t sqrt(a / pi) cos(w) / (2 - nu),
//!   K_III = -4 (1 - nu) t sqrt(a / pi) sin(w) / (2 - nu),
//! with s_n = s cos^2(GAMMA), t = s sin(GAMMA) cos(GAMMA), w the angle on
//! the front from the shear direction and the sign of K_III in the crack
//! frame (normal e2, e1 ahead, e3 = e1 x e2). Half of the block, y >= 0
//! (symmetry on y = 0), Q1 hexahedra of size h up to 1.5 a, growing by
//! 1.3 beyond; the mesh does not follow the crack. The displacement is
//! enriched by the Heaviside function of the crack and by the crack tip
//! functions on the nodes within 2 h of the front. At the two front ends
//! on the symmetry plane K_III is not meaningful: the domain integral
//! sees half of its domain and the mode III field is not symmetric.
//!
//! Prints the factors along the front against the exact ones and writes
//! them on the front (target/showcase/inclined_penny_crack_sifs.vtu).
//! Argument: h (default 0.125).

use getfem::fem::crack_front::{crack_fronts, select_mesh_fem_basic_dofs_near_front};
use getfem::fem::crack_tip_function::{CrackTipFunction, Singular};
use getfem::fem::global_function::GlobalFunction;
use getfem::fem::integration::integ_descriptor;
use getfem::fem::level_set::LevelSet;
use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im_level_set::{CELL_TOLERANCE, IntegrateWhere, MeshImLevelSet};
use getfem::fem::mesh_level_set::MeshLevelSet;
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::region::{outer_faces_of_mesh, select_faces_of_normal};
use getfem::geom::regular_mesh::regular_mesh;
use getfem::gwfl::crack_sif::{CrackedBody, Sif, crack_sifs, sifs_on_front};
use getfem::gwfl::workspace::Workspace;
use getfem::io::vtk::{Format, VtuExport};
use getfem::model::{Model, Newton};

const A: f64 = 1.0; // crack radius
const L: f64 = 6.0; // half size of the block
const E: f64 = 1.0;
const NU: f64 = 0.3;
const S: f64 = 1.0; // remote stress along z
const GAMMA: f64 = std::f64::consts::FRAC_PI_6; // tilt of the crack
const PENALTY: f64 = 1e6;
const SUPPORT: f64 = 1e-8; // relative support below which DOFs are dropped

const SYM_Y: usize = 101;
const ENDS: [usize; 2] = [102, 103];

/// The crack normal and the shear direction on the crack (the in-plane
/// part of the z axis).
fn frame() -> ([f64; 3], [f64; 3]) {
  let (s, c) = GAMMA.sin_cos();
  ([s, 0.0, c], [-c, 0.0, s])
}

/// The node coordinates from start: steps h up to 1.5 a, then growing by
/// 1.3, up to L.
fn nodes(start: f64, h: f64) -> Vec<f64> {
  let mut x = vec![start];
  let mut step = h;
  while *x.last().unwrap() < L - 1e-9 {
    let last = *x.last().unwrap();
    if last >= 1.5 * A {
      step *= 1.3;
    }
    // the last step stretched or shrunk to end at L
    let next = if last + 1.5 * step > L {
      L
    } else {
      last + step
    };
    x.push(next);
  }
  x
}

/// The coordinates symmetric about 0 from the positive ones.
fn mirrored(p: &[f64]) -> Vec<f64> {
  let neg = p.iter().rev().filter(|&&x| x > 0.0).map(|x| -x);
  neg.chain(p.iter().copied()).collect()
}

/// The angle on the front from the shear direction and the exact factors.
pub fn exact(x: &[f64]) -> (f64, [f64; 3]) {
  let (n, t) = frame();
  let dot = |a: &[f64; 3]| a.iter().zip(x).map(|(a, b)| a * b).sum::<f64>();
  let w = x[1].atan2(dot(&t));
  let (sn, tau) = (S * n[2] * n[2], S * n[2] * t[2]);
  let r = (A / std::f64::consts::PI).sqrt();
  (
    w,
    [
      2.0 * sn * r,
      4.0 * tau * r * w.cos() / (2.0 - NU),
      -4.0 * (1.0 - NU) * tau * r * w.sin() / (2.0 - NU),
    ],
  )
}

pub struct Solution {
  pub dofs: usize,
  /// the stress intensity factors along the front
  pub sifs: Vec<Sif>,
}

/// The values of a GWFL expression in x, y, z at the nodes of the level
/// set ls (as Python's ls.set_values with a string).
fn values(ls: &LevelSet, expr: &str) -> Vec<f64> {
  let mf = ls.mesh_fem();
  let mut ws = Workspace::new();
  for (k, x) in ["x", "y", "z"]
    .iter()
    .enumerate()
    .take(mf.linked_mesh().dim())
  {
    ws.add_macro(x, &format!("X({})", k + 1)).unwrap();
  }
  ws.interpolation(expr, &mf, -1).unwrap()
}

pub fn simulate(h: f64) -> Solution {
  // the mesh: y from 0, x and z symmetric, z without a node at 0
  let xs = mirrored(&nodes(0.0, h));
  let ys = nodes(0.0, h);
  let zs = mirrored(&nodes(0.5 * h, h));
  let n = [xs.len() - 1, ys.len() - 1, zs.len() - 1];
  let mesh = regular_mesh(
    &geotrans_descriptor("GT_QK(3,1)").unwrap(),
    &n,
    &[0.0; 3],
    &[1.0; 3],
  )
  .unwrap();
  mesh.modify(|m| {
    let moves: Vec<(usize, Vec<f64>)> = m
      .point_index()
      .map(|i| {
        let p = m.point(i);
        let at = |c: &[f64], k: usize| c[(p[k] * n[k] as f64).round() as usize];
        (i, vec![at(&xs, 0), at(&ys, 1), at(&zs, 2)])
      })
      .collect();
    m.set_points(&moves).unwrap();
  });
  let border = outer_faces_of_mesh(&mesh);
  for (rg, nn) in [
    (SYM_Y, [0.0, -1.0, 0.0]),
    (ENDS[0], [0.0, 0.0, 1.0]),
    (ENDS[1], [0.0, 0.0, -1.0]),
  ] {
    mesh
      .set_region(
        rg,
        &select_faces_of_normal(&mesh, &border, &nn, 0.5).unwrap(),
      )
      .unwrap();
  }

  // the crack: primary the distance to its plane, secondary the distance
  // to the front circle in the plane
  // (the normal (sin GAMMA, 0, cos GAMMA))
  let ls = LevelSet::new(&mesh, 2, true).unwrap();
  let d = "(x*sin(pi/6)+z*cos(pi/6))";
  ls.set_values(0, &values(&ls, d)).unwrap();
  ls.set_values(1, &values(&ls, &format!("sqrt(x*x+y*y+z*z-sqr{d})-{A}")))
    .unwrap();
  let mut mls = MeshLevelSet::new(&mesh);
  mls.add_level_set(&ls);
  mls.adapt().unwrap();

  // Heaviside enrichment plus the crack tip functions near the front
  let pu_mf = MeshFem::new(&mesh, 1)
    .with_classical_finite_element(1)
    .unwrap();
  let mfls = MeshFem::from_level_set(&mls, &pu_mf).unwrap();
  let tips: Vec<GlobalFunction> = Singular::LEFM
    .iter()
    .map(|&s| GlobalFunction::on_level_set(&ls, &CrackTipFunction::crack(s)).unwrap())
    .collect();
  let global_mf = MeshFem::from_global_functions(&mesh, &tips, None).unwrap();
  let pu_dofs = select_mesh_fem_basic_dofs_near_front(&ls, &pu_mf, 2.0 * h).unwrap();
  let tip = MeshFem::from_product(&pu_mf, &global_mf, Some(&pu_dofs)).unwrap();
  let base = MeshFem::from_sum(&[&mfls, &tip]).unwrap();

  let mim = MeshImLevelSet::new(&mls, IntegrateWhere::All).unwrap();
  mim
    .set_simplex_im(
      &integ_descriptor("IM_TETRAHEDRON(5)").unwrap(),
      Some(&integ_descriptor("IM_PRODUCT(IM_TRIANGLE(3),IM_GAUSS1D(3))").unwrap()),
      CELL_TOLERANCE,
    )
    .unwrap();
  mim
    .set_integration_method(&integ_descriptor("IM_GAUSS_PARALLELEPIPED(3,4)").unwrap())
    .unwrap();
  mim.adapt().unwrap();
  // the DOFs with support (enriched DOFs without are singular)
  let kept = base.dofs_with_support(&mim, SUPPORT).unwrap();
  base.set_qdim(3);
  let mf_u = MeshFem::partial(
    &base,
    &kept
      .iter()
      .flat_map(|&i| (0..3).map(move |c| 3 * i + c))
      .collect(),
  )
  .unwrap();

  let (lambda, mu) = (
    E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU)),
    E / (2.0 * (1.0 + NU)),
  );
  let mut md = Model::new();
  md.add_fem_variable("u", &mf_u).unwrap();
  md.add_scalar_data("lambda", lambda).unwrap();
  md.add_scalar_data("mu", mu).unwrap();
  md.add_scalar_data("s", S).unwrap();
  md.add_scalar_data("pen", PENALTY).unwrap();
  md.add_linear_term(
    &mim,
    None,
    "lambda*Div_u*Div_Test_u + mu*(Grad_u + Grad_u'):Grad_Test_u",
  )
  .unwrap();
  // the uniaxial stress on the ends, symmetry on y = 0; a weak spring
  // takes the remaining rigid motions (the loads are balanced)
  for rg in ENDS {
    md.add_linear_term(
      &mim,
      Some(&mesh.region(rg).unwrap()),
      "-s*(Normal.[0,0,1])*([0,0,1].Test_u)",
    )
    .unwrap();
  }
  md.add_linear_term(
    &mim,
    Some(&mesh.region(SYM_Y).unwrap()),
    "pen*(u.Normal)*(Test_u.Normal)",
  )
  .unwrap();
  md.add_linear_term(&mim, None, "1e-6*u.Test_u").unwrap();
  Newton::default().solve(&mut md, None).unwrap();
  let u = md.variable("u").unwrap();

  // the stress intensity factors along the front (radius 0.3, points
  // 0.25 apart), the lips free
  let body = CrackedBody {
    mim: &mim,
    mf_u: &mf_u,
    u,
    ls: &ls,
    young: E,
    poisson: NU,
    lips: None,
  };
  let fronts = crack_fronts(&ls, Some(0.25)).unwrap();
  let sifs = fronts
    .iter()
    .flat_map(|f| crack_sifs(&body, 0.3, f).unwrap())
    .collect();
  Solution {
    dofs: mf_u.nb_dof().unwrap(),
    sifs,
  }
}

fn main() {
  let h: f64 = std::env::args()
    .nth(1)
    .map_or(0.125, |a| a.parse().unwrap());
  let o = simulate(h);
  println!("h {h}, {} DOFs", o.dofs);
  println!("  angle       K_I    exact      K_II    exact     K_III    exact");
  for s in &o.sifs {
    let (w, k) = exact(&s.point);
    println!(
      "{:7.2} {:9.5} {:8.5} {:9.5} {:8.5} {:9.5} {:8.5}",
      w.to_degrees(),
      s.k[0],
      k[0],
      s.k[1],
      k[1],
      s.k[2],
      k[2]
    );
  }
  let out = format!("{}/../../target/showcase", env!("CARGO_MANIFEST_DIR"));
  std::fs::create_dir_all(&out).unwrap();
  let path = format!("{out}/inclined_penny_crack_sifs.vtu");
  let (mf, kv) = sifs_on_front(&o.sifs, false).unwrap();
  let mut vtu = VtuExport::new(&path, Format::Ascii);
  vtu.exporting(&mf).unwrap();
  vtu.write_point_data(&mf, &kv, "K").unwrap();
  vtu.write().unwrap();
  println!("{path}");
}
