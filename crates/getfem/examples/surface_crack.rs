//! Surface cracks: a block [0, W] x [-H, H] (x [0, T] in 3D), the crack
//! in the plane y = 0 from the free face x = 0, a unit traction pulling
//! the faces y = +-H apart, the face x = W clamped. Q1, linearized
//! elasticity; the displacement enriched by the Heaviside function of the
//! crack and, on the elements the front passes through, by the crack tip
//! functions; the DOFs without support dropped. Cases:
//! - "edge": an edge crack of depth A in 2D (plane strain);
//! - "through": the same through the thickness T in 3D, plane strain
//!   (u_z = 0 on z = 0 and z = T): its front ends on both faces;
//! - "semi": a semicircular surface crack of radius A centred at x = 0,
//!   z = T/2: its front a half circle ending on the face x = 0.
//!
//! Prints the opening along x in the crack plane (3D: at z = T/2) and the
//! stress intensity factors (interaction integral, radius 0.3; 3D: along
//! the front, points 0.25 apart, also written to
//! target/showcase/surface_crack_<case>_sifs.vtu). Arguments: the case
//! (default "semi"), h (default 0.125).

use getfem::fem::crack_front::{crack_fronts, front_element_dofs};
use getfem::fem::crack_tip_function::{CrackTipFunction, Singular};
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
use getfem::gwfl::crack_sif::{CrackedBody, Sif, crack_sifs, sifs_on_front};
use getfem::gwfl::workspace::Workspace;
use getfem::io::vtk::{Format, VtuExport};
use getfem::model::{Model, Newton};

const W: f64 = 2.0;
const H: f64 = 2.0;
const T: f64 = 2.0;
const A: f64 = 0.5;
const E: f64 = 1.0;
const NU: f64 = 0.3;

/// The opening along x in the crack plane, the stress intensity factors
/// (3D: along each front, closed or not).
pub struct Opening {
  pub x: Vec<f64>,
  pub delta: Vec<f64>,
  pub dofs: usize,
  pub sifs: Vec<(Vec<Sif>, bool)>,
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

/// case: "edge", "through" or "semi"; h: the element size
pub fn simulate(case: &str, h: f64) -> Opening {
  let dim = if case == "edge" { 2 } else { 3 };
  assert!(
    ["edge", "through", "semi"].contains(&case),
    "case: edge, through or semi"
  );
  let n = |len: f64| (len / h).round() as usize;
  // y nodes offset by h/2: no node on the crack plane
  let (gt, cells, org, size) = if dim == 2 {
    (
      "GT_QK(2,1)",
      vec![n(W), n(2.0 * H) + 1],
      vec![0.0, -H - 0.5 * h],
      vec![W, 2.0 * H + h],
    )
  } else {
    (
      "GT_QK(3,1)",
      vec![n(W), n(2.0 * H) + 1, n(T)],
      vec![0.0, -H - 0.5 * h, 0.0],
      vec![W, 2.0 * H + h, T],
    )
  };
  let mesh = regular_mesh(&geotrans_descriptor(gt).unwrap(), &cells, &org, &size).unwrap();
  let border = outer_faces_of_mesh(&mesh);
  let normal =
    |k: usize, s: f64| -> Vec<f64> { (0..dim).map(|a| if a == k { s } else { 0.0 }).collect() };
  for (rg, nn) in [
    (1, normal(1, 1.0)),
    (2, normal(1, -1.0)),
    (3, normal(0, 1.0)),
  ] {
    mesh
      .set_region(
        rg,
        &select_faces_of_normal(&mesh, &border, &nn, 0.5).unwrap(),
      )
      .unwrap();
  }
  if dim == 3 {
    for (rg, nn) in [(4, normal(2, 1.0)), (5, normal(2, -1.0))] {
      mesh
        .set_region(
          rg,
          &select_faces_of_normal(&mesh, &border, &nn, 0.5).unwrap(),
        )
        .unwrap();
    }
  }

  // the crack: primary y, secondary the distance to the front
  let secondary = match case {
    "semi" => format!("sqrt(x*x+sqr(z-{}))-{A}", 0.5 * T),
    _ => format!("x-{A}"),
  };
  let ls = LevelSet::new(&mesh, 2, true).unwrap();
  ls.set_values(0, &values(&ls, "y")).unwrap();
  ls.set_values(1, &values(&ls, &secondary)).unwrap();
  let mut mls = MeshLevelSet::new(&mesh);
  mls.add_level_set(&ls);
  mls.adapt().unwrap();

  let pu_mf = MeshFem::new(&mesh, 1)
    .with_classical_finite_element(1)
    .unwrap();
  let mfls = MeshFem::from_level_set(&mls, &pu_mf).unwrap();
  let funcs: Vec<GlobalFunction> = Singular::LEFM
    .iter()
    .map(|&s| GlobalFunction::on_level_set(&ls, &CrackTipFunction::crack(s)).unwrap())
    .collect();
  let global_mf = MeshFem::from_global_functions(&mesh, &funcs, None).unwrap();
  // the front elements: the elements the front passes through
  let pu_dofs = front_element_dofs(&mls, 0, &pu_mf).unwrap();
  let sing = MeshFem::from_product(&pu_mf, &global_mf, Some(&pu_dofs)).unwrap();
  let base = MeshFem::from_sum(&[&sing, &mfls]).unwrap();

  let mim = MeshImLevelSet::new(&mls, IntegrateWhere::All).unwrap();
  let (simplex, singular, regular) = if dim == 2 {
    (
      "IM_TRIANGLE(6)",
      "IM_STRUCTURED_COMPOSITE(IM_GAUSS_PARALLELEPIPED(2,6),9)",
      "IM_GAUSS_PARALLELEPIPED(2,4)",
    )
  } else {
    (
      "IM_TETRAHEDRON(5)",
      "IM_PRODUCT(IM_TRIANGLE(3),IM_GAUSS1D(3))",
      "IM_GAUSS_PARALLELEPIPED(3,4)",
    )
  };
  mim
    .set_simplex_im(
      &integ_descriptor(simplex).unwrap(),
      Some(&integ_descriptor(singular).unwrap()),
      CELL_TOLERANCE,
    )
    .unwrap();
  mim
    .set_integration_method(&integ_descriptor(regular).unwrap())
    .unwrap();
  mim.adapt().unwrap();
  // the DOFs with support (enriched DOFs without are singular)
  let kept = base.dofs_with_support(&mim, 1e-8).unwrap();
  base.set_qdim(dim);
  let mf_u = MeshFem::partial(
    &base,
    &kept
      .iter()
      .flat_map(|&i| (0..dim).map(move |c| dim * i + c))
      .collect(),
  )
  .unwrap();

  let mut md = Model::new();
  md.add_fem_variable("u", &mf_u).unwrap();
  md.add_scalar_data("lambda", E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU)))
    .unwrap();
  md.add_scalar_data("mu", E / (2.0 * (1.0 + NU))).unwrap();
  md.add_linear_term(
    &mim,
    None,
    "lambda*Div_u*Div_Test_u + mu*(Grad_u + Grad_u'):Grad_Test_u",
  )
  .unwrap();
  for rg in [1, 2] {
    md.add_linear_term(&mim, Some(&mesh.region(rg).unwrap()), "-Normal.Test_u")
      .unwrap();
  }
  md.add_linear_term(&mim, Some(&mesh.region(3).unwrap()), "1e6*u.Test_u")
    .unwrap();
  if case == "through" {
    for rg in [4, 5] {
      md.add_linear_term(
        &mim,
        Some(&mesh.region(rg).unwrap()),
        "1e6*(u.Normal)*(Test_u.Normal)",
      )
      .unwrap();
    }
  }
  Newton::default().solve(&mut md, None).unwrap();
  let u = md.variable("u").unwrap();

  // the opening along x in the crack plane (3D: z = T/2)
  let xs: Vec<f64> = (0..=20)
    .map(|k| A * k as f64 / 20.0 * 0.999 + 1e-6)
    .collect();
  let at = |dy: f64| -> Vec<Vec<f64>> {
    xs.iter()
      .map(|&x| {
        if dim == 2 {
          vec![x, dy]
        } else {
          vec![x, dy, 0.5 * T]
        }
      })
      .collect()
  };
  let up = interpolation_at_points(&mf_u, &at(1e-7), u).unwrap();
  // the stress intensity factors
  let body = CrackedBody {
    mim: &mim,
    mf_u: &mf_u,
    u,
    ls: &ls,
    young: E,
    poisson: NU,
    lips: None,
  };
  let sifs = crack_fronts(&ls, (dim == 3).then_some(0.25))
    .unwrap()
    .iter()
    .map(|f| (crack_sifs(&body, 0.3, f).unwrap(), f.closed))
    .collect();
  let down = interpolation_at_points(&mf_u, &at(-1e-7), u).unwrap();
  Opening {
    delta: (0..xs.len())
      .map(|k| up[dim * k + 1] - down[dim * k + 1])
      .collect(),
    x: xs,
    dofs: mf_u.nb_dof().unwrap(),
    sifs,
  }
}

fn main() {
  let case = std::env::args().nth(1).unwrap_or("semi".into());
  let h: f64 = std::env::args()
    .nth(2)
    .map_or(0.125, |a| a.parse().unwrap());
  let o = simulate(&case, h);
  println!("{case}, h {h}, {} DOFs", o.dofs);
  println!("       x     opening");
  for (x, d) in o.x.iter().zip(&o.delta) {
    println!("{x:8.4} {d:11.6}");
  }
  println!("  point                       K_I       K_II      K_III");
  for (k, (sifs, closed)) in o.sifs.iter().enumerate() {
    for s in sifs {
      let p: Vec<String> = s.point.iter().map(|v| format!("{v:7.4}")).collect();
      println!(
        "  ({}) {:9.5} {:9.5} {:9.5}",
        p.join(", "),
        s.k[0],
        s.k[1],
        s.k[2]
      );
    }
    if sifs.first().is_some_and(|s| s.point.len() == 3) {
      let out = format!("{}/../../target/showcase", env!("CARGO_MANIFEST_DIR"));
      std::fs::create_dir_all(&out).unwrap();
      let path = format!("{out}/surface_crack_{case}_sifs{k}.vtu");
      let (mf, kv) = sifs_on_front(sifs, *closed).unwrap();
      let mut vtu = VtuExport::new(&path, Format::Ascii);
      vtu.exporting(&mf).unwrap();
      vtu.write_point_data(&mf, &kv, "K").unwrap();
      vtu.write().unwrap();
      println!("  {path}");
    }
  }
}
