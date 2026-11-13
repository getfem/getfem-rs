//! A penny-shaped crack under pressure (Sneddon): a crack of radius a = 1
//! in the plane z = 0 of a block, a unit pressure on its lips, linearized
//! elasticity (E = 1, nu = 0.3). A quarter of the block, x, y in [0, 6],
//! z in [-6, 6]: symmetry on x = 0 and y = 0, the far faces clamped. Q1
//! hexahedra of size h up to 1.5 a, growing by 1.3 beyond; no node on the
//! crack plane. The displacement is enriched by the Heaviside function of
//! the crack and, near the front, by the crack tip functions:
//! - "none": no front enrichment;
//! - "zone:rho": the nodes within rho of the front (fixed zone);
//! - "elements": the nodes of the elements the front passes through;
//! - "front:r0:r1:s": the tip functions times a cutoff (1 below r0, 0
//!   beyond r1) and linear hats on front points s apart ("cfront": cubic;
//!   ":2" at the end: with the second order tip functions, GetFEM's 4 to 7);
//! - "polar:n:dr:s": the tip functions times n radial hats dr apart (0
//!   beyond n dr) and linear hats on front points s apart;
//! - "gpolar:rmax:s": the same on the radial nodes 0, h, 2h, 4h, ... up
//!   to rmax;
//! - "ppolar:nr:dr:na:s": radial hats times na angular hats around the
//!   front (a polar grid in r and theta);
//! - "lpolar:nr:dr:na:s": sqrt(r) times radial hats and the angular hats
//!   of na sectors on [-pi, pi], the end hats on the lips (no tip
//!   functions); "lpolarx": without the radial hat at the front.
//!
//! Several, summed: joined by '+' (e.g. "front:0.3:0.8:0.2:2+zone:0.125").
//!
//! Prints the opening along the ray at 45 degrees against Sneddon's
//! 8 (1 - nu^2) p sqrt(a^2 - r^2) / (pi E), the relative RMS error over
//! r < 0.95 a, and the stress intensity factors along the front against
//! K_I = 2 p sqrt(a / pi). Arguments: h (default 0.125), the enrichment (default
//! "zone:0.25").

use getfem::fem::crack_front::{
  HatDegree, crack_fronts, front_element_dofs, front_hats, select_mesh_fem_basic_dofs_near_front,
};
use getfem::fem::crack_tip_function::{CrackTipFunction, Cutoff, Singular};
use getfem::fem::global_function::GlobalFunction;
use getfem::fem::integration::integ_descriptor;
use getfem::fem::interpolation::interpolation_at_points;
use getfem::fem::level_set::LevelSet;
use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im_level_set::{CELL_TOLERANCE, IntegrateWhere, MeshImLevelSet};
use getfem::fem::mesh_level_set::MeshLevelSet;
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::region::{outer_faces_of_mesh, select_faces_of_normal};
use getfem::geom::regular_mesh::regular_mesh;
use getfem::gwfl::crack_sif::{CrackedBody, LipTractions, Sif, crack_sifs};
use getfem::gwfl::workspace::Workspace;
use getfem::model::{Model, Newton};

const A: f64 = 1.0; // crack radius
const L: f64 = 6.0; // half size of the block
const E: f64 = 1.0;
const NU: f64 = 0.3;
const P: f64 = 1.0; // pressure on the lips
const PENALTY: f64 = 1e6;
const SUPPORT: f64 = 1e-8; // relative support below which DOFs are dropped

const SYM_X: usize = 101;
const SYM_Y: usize = 102;
const FAR: [usize; 4] = [103, 104, 105, 106];

/// The front enrichment.
#[derive(Clone, Copy, Debug)]
pub enum Enrichment {
  None,
  /// the nodes within this distance of the front
  Zone(f64),
  /// the nodes of the elements the front passes through (the default of
  /// the 3D examples)
  Elements,
  /// cutoff inner and outer radii, spacing of the front points, hats,
  /// with the second order tip functions
  Front(f64, f64, f64, HatDegree, bool),
  /// the polar grid: n radial hats dr apart (0 beyond n dr), front points
  /// s apart (linear hats), with the second order tip functions
  Polar(usize, f64, f64, bool),
  /// the polar grid in r and theta: nr radial hats dr apart, na angular
  /// hats, front points s apart, with the second order tip functions
  FullPolar(usize, f64, usize, f64, bool),
  /// the lean polar grid: sqrt(r) times nr radial hats dr apart and the
  /// angular hats of na sectors on [-pi, pi] (the end hats on the lips),
  /// front points s apart; true: without the radial hat at the front
  LeanPolar(usize, f64, usize, f64, bool),
  /// the graded polar grid: radial nodes 0, h, 2h, 4h, ... up to rmax,
  /// front points s apart, with the second order tip functions
  GradedPolar(f64, f64, bool),
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

/// The opening along the ray at 45 degrees at the radii r, and Sneddon's.
pub struct Opening {
  pub r: Vec<f64>,
  pub delta: Vec<f64>,
  pub exact: Vec<f64>,
  pub dofs: usize,
  /// scalar DOFs dropped for lack of support
  pub dropped: usize,
  /// the largest coefficient of the solution (conditioning indicator)
  pub max_coef: f64,
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

pub fn simulate(h: f64, enrichments: &[Enrichment]) -> Opening {
  // the mesh: x, y from 0, z symmetric without a node at 0
  let xs = nodes(0.0, h);
  let zp = nodes(0.5 * h, h);
  let zs: Vec<f64> = zp
    .iter()
    .rev()
    .map(|z| -z)
    .chain(zp.iter().copied())
    .collect();
  let n = [xs.len() - 1, xs.len() - 1, zs.len() - 1];
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
        (i, vec![at(&xs, 0), at(&xs, 1), at(&zs, 2)])
      })
      .collect();
    m.set_points(&moves).unwrap();
  });
  let border = outer_faces_of_mesh(&mesh);
  for (rg, nn) in [
    (SYM_X, [-1.0, 0.0, 0.0]),
    (SYM_Y, [0.0, -1.0, 0.0]),
    (FAR[0], [1.0, 0.0, 0.0]),
    (FAR[1], [0.0, 1.0, 0.0]),
    (FAR[2], [0.0, 0.0, 1.0]),
    (FAR[3], [0.0, 0.0, -1.0]),
  ] {
    mesh
      .set_region(
        rg,
        &select_faces_of_normal(&mesh, &border, &nn, 0.5).unwrap(),
      )
      .unwrap();
  }

  // the crack: primary z, secondary the distance to the front circle
  let ls = LevelSet::new(&mesh, 2, true).unwrap();
  ls.set_values(0, &values(&ls, "z")).unwrap();
  ls.set_values(1, &values(&ls, &format!("sqrt(x*x+y*y)-{A}")))
    .unwrap();
  let mut mls = MeshLevelSet::new(&mesh);
  mls.add_level_set(&ls);
  mls.adapt().unwrap();

  let mf_pre = MeshFem::new(&mesh, 1)
    .with_classical_finite_element(1)
    .unwrap();
  let mfls = MeshFem::from_level_set(&mls, &mf_pre).unwrap();
  let tips = || -> Vec<GlobalFunction> {
    Singular::LEFM
      .iter()
      .map(|&s| GlobalFunction::on_level_set(&ls, &CrackTipFunction::crack(s)).unwrap())
      .collect()
  };
  let space = |enrichment: Enrichment| -> Option<MeshFem> {
    match enrichment {
      Enrichment::None => None,
      Enrichment::Zone(rho) => {
        let pu_mf = MeshFem::new(&mesh, 1)
          .with_classical_finite_element(1)
          .unwrap();
        let pu_dofs = select_mesh_fem_basic_dofs_near_front(&ls, &pu_mf, rho).unwrap();
        let global_mf = MeshFem::from_global_functions(&mesh, &tips(), None).unwrap();
        Some(MeshFem::from_product(&pu_mf, &global_mf, Some(&pu_dofs)).unwrap())
      }
      Enrichment::Elements => {
        let pu_mf = MeshFem::new(&mesh, 1)
          .with_classical_finite_element(1)
          .unwrap();
        let pu_dofs = front_element_dofs(&mls, 0, &pu_mf).unwrap();
        let global_mf = MeshFem::from_global_functions(&mesh, &tips(), None).unwrap();
        Some(MeshFem::from_product(&pu_mf, &global_mf, Some(&pu_dofs)).unwrap())
      }
      Enrichment::Polar(..)
      | Enrichment::GradedPolar(..)
      | Enrichment::FullPolar(..)
      | Enrichment::LeanPolar(..) => {
        let na = match enrichment {
          Enrichment::FullPolar(_, _, na, _, _) | Enrichment::LeanPolar(_, _, na, _, _) => na,
          _ => 1,
        };
        let lean = matches!(enrichment, Enrichment::LeanPolar(..));
        let (nodes, spacing, second) = match enrichment {
          Enrichment::Polar(nr, dr, spacing, second)
          | Enrichment::FullPolar(nr, dr, _, spacing, second) => (
            (0..=nr).map(|k| k as f64 * dr).collect::<Vec<f64>>(),
            spacing,
            second,
          ),
          Enrichment::LeanPolar(nr, dr, _, spacing, _) => (
            (0..=nr).map(|k| k as f64 * dr).collect::<Vec<f64>>(),
            spacing,
            false,
          ),
          Enrichment::GradedPolar(rmax, spacing, second) => {
            let mut nodes = vec![0.0, h];
            while *nodes.last().unwrap() < rmax {
              let next = 2.0 * nodes.last().unwrap();
              nodes.push(if next > rmax { rmax } else { next });
            }
            (nodes, spacing, second)
          }
          _ => unreachable!(),
        };
        let nr = nodes.len() - 1;
        let set: &[Singular] = if second {
          &Singular::LEFM_SECOND_ORDER
        } else {
          &Singular::LEFM
        };
        // per radial hat k and tip function: the function of the level set
        let mut radial: Vec<GlobalFunction> = vec![];
        for k in 0..nr {
          let hat = CrackTipFunction::cutoff(Cutoff::Hat {
            left: nodes[k.saturating_sub(1)],
            centre: nodes[k],
            right: nodes[k + 1],
          })
          .unwrap();
          if lean && k == 0 && matches!(enrichment, Enrichment::LeanPolar(.., true)) {
            continue;
          }
          if lean {
            // sqrt(r) times the angular hats on [-pi, pi]
            for m in 0..=na {
              let a = CrackTipFunction::angular_hat(na, m, false).unwrap();
              let g = CrackTipFunction::product(
                &CrackTipFunction::crack(Singular::SqrtR),
                &CrackTipFunction::product(&hat, &a),
              );
              radial.push(GlobalFunction::on_level_set(&ls, &g).unwrap());
            }
            continue;
          }
          for m in 0..na {
            // (na = 1: no angular grid)
            let hat = if na == 1 {
              hat.clone()
            } else {
              CrackTipFunction::product(&hat, &CrackTipFunction::angular_hat(na, m, true).unwrap())
            };
            for &f in set {
              let g = CrackTipFunction::product(&CrackTipFunction::crack(f), &hat);
              radial.push(GlobalFunction::on_level_set(&ls, &g).unwrap());
            }
          }
        }
        let mut funcs = vec![];
        for front in crack_fronts(&ls, Some(spacing)).unwrap() {
          for n in front_hats(&front, HatDegree::Linear, nodes[nr]).unwrap() {
            for g in &radial {
              funcs.push(GlobalFunction::product(&n, g).unwrap());
            }
          }
        }
        Some(MeshFem::from_global_functions(&mesh, &funcs, None).unwrap())
      }
      Enrichment::Front(r0, r1, spacing, degree, second) => {
        let cut = CrackTipFunction::cutoff(Cutoff::Polynomial {
          inner: r0,
          outer: r1,
        })
        .unwrap();
        let set: &[Singular] = if second {
          &Singular::LEFM_SECOND_ORDER
        } else {
          &Singular::LEFM
        };
        let tips: Vec<GlobalFunction> = set
          .iter()
          .map(|&s| {
            let f = CrackTipFunction::product(&CrackTipFunction::crack(s), &cut);
            GlobalFunction::on_level_set(&ls, &f).unwrap()
          })
          .collect();
        let mut funcs = vec![];
        for front in crack_fronts(&ls, Some(spacing)).unwrap() {
          for hat in front_hats(&front, degree, r1).unwrap() {
            for t in &tips {
              funcs.push(GlobalFunction::product(&hat, t).unwrap());
            }
          }
        }
        Some(MeshFem::from_global_functions(&mesh, &funcs, None).unwrap())
      }
    }
  };
  let mut parts: Vec<MeshFem> = enrichments.iter().filter_map(|&e| space(e)).collect();
  parts.push(mfls.clone());
  let base = MeshFem::from_sum(&parts.iter().collect::<Vec<_>>()).unwrap();

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
  // the DOFs with support: mass diagonal above SUPPORT times its median
  // (functions without support in the domain add singular DOFs)
  let kept: std::collections::BTreeSet<usize> = {
    let mut ws = Workspace::new();
    ws.add_fem_variable("u", &base).unwrap();
    let m = {
      ws.clear_expressions();
      ws.add_expression("Test_u*Test2_u", &mim, None).unwrap();
      ws.assembly(2, false).unwrap();
      ws.assembled_matrix()
    }
    .to_csr();
    let d: Vec<f64> = (0..base.nb_dof().unwrap()).map(|i| m.get(i, i)).collect();
    let mut sorted: Vec<f64> = d.iter().copied().filter(|v| *v > 0.0).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    (0..d.len()).filter(|&i| d[i] > SUPPORT * median).collect()
  };
  let dropped = base.nb_dof().unwrap() - kept.len();
  base.set_qdim(3);
  let mf_u = MeshFem::partial(
    &base,
    &kept
      .iter()
      .flat_map(|&i| (0..3).map(move |c| 3 * i + c))
      .collect(),
  )
  .unwrap();
  let mim_lips = MeshImLevelSet::new(&mls, IntegrateWhere::Boundary).unwrap();
  mim_lips
    .set_simplex_im(
      &integ_descriptor("IM_TETRAHEDRON(3)").unwrap(),
      // the lips graded toward the front (r^(-1/2) of the SIFs' lip term)
      Some(&integ_descriptor("IM_PRODUCT(IM_TRIANGLE(3),IM_GAUSS1D(3))").unwrap()),
      CELL_TOLERANCE,
    )
    .unwrap();
  mim_lips.set_integration_method(1usize).unwrap();
  mim_lips.adapt().unwrap();

  let mut md = Model::new();
  md.add_fem_variable("u", &mf_u).unwrap();
  md.add_scalar_data("lambda", E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU)))
    .unwrap();
  md.add_scalar_data("mu", E / (2.0 * (1.0 + NU))).unwrap();
  md.add_scalar_data("p", P).unwrap();
  md.add_scalar_data("pen", PENALTY).unwrap();
  md.add_linear_term(
    &mim,
    None,
    "lambda*Div_u*Div_Test_u + mu*(Grad_u + Grad_u'):Grad_Test_u",
  )
  .unwrap();
  md.add_linear_term(
    &mim_lips,
    None,
    "p*Normal.Xfem_minus(Test_u) - p*Normal.Xfem_plus(Test_u)",
  )
  .unwrap();
  // symmetry: the normal displacement; far faces: clamped
  for rg in [SYM_X, SYM_Y] {
    md.add_linear_term(
      &mim,
      Some(&mesh.region(rg).unwrap()),
      "pen*(u.Normal)*(Test_u.Normal)",
    )
    .unwrap();
  }
  for rg in FAR {
    md.add_linear_term(&mim, Some(&mesh.region(rg).unwrap()), "pen*u.Test_u")
      .unwrap();
  }
  Newton::default().solve(&mut md, None).unwrap();
  let u = md.variable("u").unwrap();

  // the stress intensity factors along the front (radius 0.3, points 0.25
  // apart), the pressure on the lips as in the model's term
  let tractions = LipTractions {
    mim: &mim_lips,
    plus: &format!("{}*Normal", -P),
    minus: &format!("{P}*Normal"),
  };
  let body = CrackedBody {
    mim: &mim,
    mf_u: &mf_u,
    u,
    ls: &ls,
    young: E,
    poisson: NU,
    lips: Some(tractions),
  };
  let sifs = crack_fronts(&ls, Some(0.25))
    .unwrap()
    .iter()
    .flat_map(|f| crack_sifs(&body, 0.3, f).unwrap())
    .collect();

  // the opening along the ray at 45 degrees, just above and below the crack
  let c = std::f64::consts::FRAC_1_SQRT_2;
  let r: Vec<f64> = (0..20).map(|k| 0.05 * k as f64).collect();
  let eps = 1e-7;
  let side = |dz: f64| -> Vec<Vec<f64>> { r.iter().map(|&q| vec![c * q, c * q, dz]).collect() };
  let up = interpolation_at_points(&mf_u, &side(eps), u).unwrap();
  let down = interpolation_at_points(&mf_u, &side(-eps), u).unwrap();
  let delta: Vec<f64> = (0..r.len())
    .map(|k| up[3 * k + 2] - down[3 * k + 2])
    .collect();
  let exact: Vec<f64> = r
    .iter()
    .map(|&q| {
      8.0 * (1.0 - NU * NU) * P * (A * A - q * q).max(0.0).sqrt() / (std::f64::consts::PI * E)
    })
    .collect();
  Opening {
    r,
    delta,
    exact,
    dofs: mf_u.nb_dof().unwrap(),
    dropped,
    max_coef: u.iter().fold(0.0f64, |a, v| a.max(v.abs())),
    sifs,
  }
}

/// "none", "zone:rho", "front:r0:r1:s" or "cfront:r0:r1:s" (cubic hats);
/// ":2" at the end: with the second order tip functions
pub fn parse(s: &str) -> Enrichment {
  let v: Vec<&str> = s.split(':').collect();
  let f = |i: usize| v[i].parse::<f64>().unwrap();
  match v[0] {
    "none" => Enrichment::None,
    "zone" => Enrichment::Zone(f(1)),
    "elements" => Enrichment::Elements,
    "front" | "cfront" => Enrichment::Front(
      f(1),
      f(2),
      f(3),
      if v[0] == "front" {
        HatDegree::Linear
      } else {
        HatDegree::Cubic
      },
      v.get(4) == Some(&"2"),
    ),
    "polar" => Enrichment::Polar(v[1].parse().unwrap(), f(2), f(3), v.get(4) == Some(&"2")),
    "gpolar" => Enrichment::GradedPolar(f(1), f(2), v.get(3) == Some(&"2")),
    "lpolar" | "lpolarx" => Enrichment::LeanPolar(
      v[1].parse().unwrap(),
      f(2),
      v[3].parse().unwrap(),
      f(4),
      v[0] == "lpolarx",
    ),
    "ppolar" => Enrichment::FullPolar(
      v[1].parse().unwrap(),
      f(2),
      v[3].parse().unwrap(),
      f(4),
      v.get(5) == Some(&"2"),
    ),
    _ => panic!(
      "enrichment: none, zone:rho, front:r0:r1:s[:2], cfront:r0:r1:s[:2] or polar:n:dr:s[:2]"
    ),
  }
}

fn main() {
  let h: f64 = std::env::args()
    .nth(1)
    .map_or(0.125, |a| a.parse().unwrap());
  let arg = std::env::args().nth(2).unwrap_or("zone:0.25".into());
  let e: Vec<Enrichment> = arg.split('+').map(parse).collect();
  let o = simulate(h, &e);
  println!(
    "h {h}, {e:?}, {} DOFs ({} scalar dropped), largest coefficient {:.3e}",
    o.dofs, o.dropped, o.max_coef
  );
  println!("     r    opening     Sneddon");
  for k in 0..o.r.len() {
    println!("{:6.2} {:10.6} {:10.6}", o.r[k], o.delta[k], o.exact[k]);
  }
  let (mut num, mut den) = (0.0, 0.0);
  for k in 0..o.r.len() {
    if o.r[k] < 0.95 * A {
      num += (o.delta[k] - o.exact[k]).powi(2);
      den += o.exact[k].powi(2);
    }
  }
  println!(
    "relative RMS error {:.4e}, at the centre {:.4e}",
    (num / den).sqrt(),
    o.delta[0] / o.exact[0] - 1.0
  );
  let k = 2.0 * P * (A / std::f64::consts::PI).sqrt();
  println!("  angle  K_I/exact      K_II     K_III (exact K_I {k:.5})");
  for s in &o.sifs {
    let angle = s.point[1].atan2(s.point[0]).to_degrees();
    println!(
      "{angle:7.2} {:10.5} {:9.5} {:9.5}",
      s.k[0] / k,
      s.k[1],
      s.k[2]
    );
  }
}
