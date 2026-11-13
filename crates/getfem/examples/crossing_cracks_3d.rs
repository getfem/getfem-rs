//! Two crossing cracks in a 3D block, linearized elasticity: a 40 x 10 x
//! 20 mm block, clamped left and right, with the curved cracks
//! y = 2.5 tanh(x/7) (extruded along z) and
//! y = -2.5 tanh((x - 0.3 z)/7) + 0.15 z (oblique), inside the ellipsoid
//! (x/16)^2 + (y/6)^2 + (z/8)^2 < 1, their fronts closed curves. Q2 on
//! hexahedra, enriched by the Heaviside functions of the cracks and, on
//! the elements the fronts pass through, by the crack tip functions; the
//! DOFs without support dropped. Two load cases: a uniform
//! traction pulling the top and bottom faces apart, and a uniform
//! pressure on the lips. Writes the displacement on a grid of points in
//! the planes z = -4, 0, 4 to target/showcase/crossing_cracks_3d_<case>.txt
//! (columns x y z ux uy uz). Optional arguments: element size [mm]
//! (default 2.5), number of cracks (default 2; 1: the extruded one).

use getfem::fem::crack_front::front_element_dofs;
use getfem::fem::crack_tip_function::{CrackTipFunction, Singular};
use getfem::fem::fem::fem_descriptor;
use getfem::fem::global_function::GlobalFunction;
use getfem::fem::integration::integ_descriptor;
use getfem::fem::interpolation::interpolation_at_points;
use getfem::fem::level_set::LevelSet;
use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im_level_set::{CELL_TOLERANCE, IntegrateWhere, MeshImLevelSet};
use getfem::fem::mesh_level_set::MeshLevelSet;
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::mesh::Mesh;
use getfem::geom::region::{outer_faces_of_mesh, select_faces_of_normal};
use getfem::geom::regular_mesh::regular_mesh;
use getfem::gwfl::workspace::Workspace;
use getfem::model::{Model, Newton};
use std::io::Write;

const E: f64 = 1e3; // [N/mm^2] Young modulus
const NU: f64 = 0.3; // Poisson ratio
const SIGMA: f64 = 1.0; // [N/mm^2] traction on top and bottom
const PRESSURE: f64 = 1.0; // [N/mm^2] pressure on the lips
const L: f64 = 40.0; // [mm]
const H: f64 = 10.0; // [mm]
const W: f64 = 20.0; // [mm]
const R1: f64 = 2.5;
const TILT: f64 = 0.15;
const SHEAR: f64 = 0.3;
// semi-axes of the ellipsoid bounding the cracks
const AX: f64 = 16.0;
const AY: f64 = 6.0;
const AZ: f64 = 8.0;
// planes z = const of the output, points per mm in x and y
const PLANES: [f64; 3] = [-4.0, 0.0, 4.0];
const DENSITY: usize = 5;

const LEFT: usize = 101;
const RIGHT: usize = 102;
const BOTTOM: usize = 103;
const TOP: usize = 104;

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

/// the crack surface (primary): extruded for sign > 0, oblique otherwise;
/// the ellipsoid (secondary), scaled to a distance near the front (the
/// crack tip functions take the level sets as coordinates around the
/// front)
fn crack(m: &Mesh, sign: f64) -> LevelSet {
  let (shear, tilt) = if sign > 0.0 {
    (0.0, 0.0)
  } else {
    (SHEAR, TILT)
  };
  let ls = LevelSet::new(m, 2, true).unwrap();
  let p = values(
    &ls,
    &format!("y-({})*tanh((x-{shear}*z)/7)-{tilt}*z", sign * R1),
  );
  let s = values(
    &ls,
    &format!("(sqrt(sqr(x/{AX})+sqr(y/{AY})+sqr(z/{AZ}))-1)*{AX}"),
  );
  ls.set_values(0, &p).unwrap();
  ls.set_values(1, &s).unwrap();
  ls
}

/// The output points: grids in the planes z = const, the cell centers of
/// DENSITY cells per mm (off the mesh faces).
pub fn points() -> Vec<Vec<f64>> {
  let (nx, ny) = (L as usize * DENSITY, H as usize * DENSITY);
  let d = 1.0 / DENSITY as f64;
  let mut pts = vec![];
  for &z in &PLANES {
    for j in 0..ny {
      for i in 0..nx {
        pts.push(vec![
          -L / 2.0 + (i as f64 + 0.5) * d,
          -H / 2.0 + (j as f64 + 0.5) * d,
          z,
        ]);
      }
    }
  }
  pts
}

/// Solves both load cases with elements of size h [mm] and nb_cracks
/// cracks (1 or 2): the displacement at the points, 3 values per point,
/// for the traction and the pressure.
pub fn simulate(h: f64, nb_cracks: usize, pts: &[Vec<f64>]) -> [Vec<f64>; 2] {
  let n = |len: f64| (len / h).round() as usize;
  let mesh = regular_mesh(
    &geotrans_descriptor("GT_QK(3,1)").unwrap(),
    &[n(L), n(H), n(W)],
    &[-L / 2.0, -H / 2.0, -W / 2.0],
    &[L, H, W],
  )
  .unwrap();
  let border = outer_faces_of_mesh(&mesh);
  for (rg, n) in [
    (LEFT, [-1.0, 0.0, 0.0]),
    (RIGHT, [1.0, 0.0, 0.0]),
    (BOTTOM, [0.0, -1.0, 0.0]),
    (TOP, [0.0, 1.0, 0.0]),
  ] {
    mesh
      .set_region(
        rg,
        &select_faces_of_normal(&mesh, &border, &n, 0.5).unwrap(),
      )
      .unwrap();
  }
  let signs = &[1.0, -1.0][..nb_cracks];
  let lss: Vec<LevelSet> = signs.iter().map(|&s| crack(&mesh, s)).collect();
  let mut mls = MeshLevelSet::new(&mesh);
  for ls in &lss {
    mls.add_level_set(ls);
  }
  mls.adapt().unwrap();

  let mf_pre = MeshFem::new(&mesh, 1)
    .with_finite_element(&fem_descriptor("FEM_QK(3,2)").unwrap())
    .unwrap();
  let mfls = MeshFem::from_level_set(&mls, &mf_pre).unwrap();
  // the singular functions near the fronts, on the Q1 partition of unity
  let pu_mf = MeshFem::new(&mesh, 1)
    .with_classical_finite_element(1)
    .unwrap();
  let singular = |ls: &LevelSet| {
    let funcs: Vec<GlobalFunction> = Singular::LEFM
      .iter()
      .map(|&s| GlobalFunction::on_level_set(ls, &CrackTipFunction::crack(s)).unwrap())
      .collect();
    MeshFem::from_global_functions(&mesh, &funcs, None).unwrap()
  };
  // the crack tip functions on the Lagrange DOFs of the front elements
  let mf_sing: Vec<MeshFem> = lss
    .iter()
    .enumerate()
    .map(|(i, ls)| {
      let pu_dofs = front_element_dofs(&mls, i, &pu_mf).unwrap();
      MeshFem::from_product(&pu_mf, &singular(ls), Some(&pu_dofs)).unwrap()
    })
    .collect();
  let mut parts: Vec<&MeshFem> = mf_sing.iter().collect();
  parts.push(&mfls);
  let base = MeshFem::from_sum(&parts).unwrap();

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
  let kept = base.dofs_with_support(&mim, 1e-8).unwrap();
  base.set_qdim(3);
  let mf_u = MeshFem::partial(
    &base,
    &kept
      .iter()
      .flat_map(|&i| (0..3).map(move |c| 3 * i + c))
      .collect(),
  )
  .unwrap();
  let mim_bound: Vec<MeshImLevelSet> = ["a", "b"][..nb_cracks]
    .iter()
    .map(|c| {
      let mb = MeshImLevelSet::new(&mls, IntegrateWhere::Boundary).unwrap();
      mb.set_level_set_boolean_operations(c).unwrap();
      mb.set_simplex_im(
        &integ_descriptor("IM_TETRAHEDRON(3)").unwrap(),
        None,
        CELL_TOLERANCE,
      )
      .unwrap();
      mb.set_integration_method(1usize).unwrap();
      mb.adapt().unwrap();
      mb
    })
    .collect();

  let mut md = Model::new();
  md.add_fem_variable("u", &mf_u).unwrap();
  md.add_scalar_data("lambda", E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU)))
    .unwrap();
  md.add_scalar_data("mu", E / (2.0 * (1.0 + NU))).unwrap();
  md.add_scalar_data("sigma", 0.0).unwrap();
  md.add_scalar_data("p", 0.0).unwrap();
  md.add_linear_term(
    &mim,
    None,
    "lambda*Div_u*Div_Test_u + mu*(Grad_u + Grad_u'):Grad_Test_u",
  )
  .unwrap();
  // traction on top and bottom
  for rg in [TOP, BOTTOM] {
    md.add_linear_term(
      &mim,
      Some(&mesh.region(rg).unwrap()),
      "-sigma*Normal.Test_u",
    )
    .unwrap();
  }
  // pressure on the lips (Normal: the gradient of the level set, from
  // the minus to the plus side)
  for mb in &mim_bound {
    md.add_linear_term(
      mb,
      None,
      "p*Normal.Xfem_minus(Test_u) - p*Normal.Xfem_plus(Test_u)",
    )
    .unwrap();
  }
  // clamped left and right
  for rg in [LEFT, RIGHT] {
    md.add_Dirichlet_condition_with_multiplier(&mim, "u", 1usize, mesh.region(rg).unwrap(), None)
      .unwrap();
  }

  let mut solve = |sigma: f64, p: f64| {
    md.set_variable("sigma", &[sigma]).unwrap();
    md.set_variable("p", &[p]).unwrap();
    Newton::default().solve(&mut md, None).unwrap();
    interpolation_at_points(&mf_u, pts, md.variable("u").unwrap()).unwrap()
  };
  [solve(SIGMA, 0.0), solve(0.0, PRESSURE)]
}

fn main() {
  let h: f64 = std::env::args().nth(1).map_or(2.5, |a| a.parse().unwrap());
  let nb_cracks: usize = std::env::args().nth(2).map_or(2, |a| a.parse().unwrap());
  let out = format!("{}/../../target/showcase", env!("CARGO_MANIFEST_DIR"));
  std::fs::create_dir_all(&out).unwrap();
  let pts = points();
  let u = simulate(h, nb_cracks, &pts);
  for (case, u) in ["traction", "pressure"].iter().zip(&u) {
    let path = format!("{out}/crossing_cracks_3d_{case}.txt");
    let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
    writeln!(f, "# x y z ux uy uz").unwrap();
    for (x, v) in pts.iter().zip(u.chunks(3)) {
      writeln!(
        f,
        "{} {} {} {:e} {:e} {:e}",
        x[0], x[1], x[2], v[0], v[1], v[2]
      )
      .unwrap();
    }
    let umax = u.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    println!("{case}: max |u_i| {umax:.4e} mm, {path}");
  }
}
