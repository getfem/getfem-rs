//! XFEM: a square with a crack (GetFEM's interface/tests/python/
//! demo_crack.py). Linear elasticity on [-0.5, 0.5]^2 with a crack along
//! y = 0 for x < 0 (tip at the origin), the opening mode reference
//! solution imposed on the border (penalization). The displacement is P1
//! enriched by the Heaviside function of the crack (MeshFem::
//! from_level_set) and near the tip by the four singular functions of
//! linear fracture mechanics, either multiplied by a cutoff (variant
//! Cutoff) or by the P1 functions of the nodes within 0.1 of the tip
//! (Product). Variant TwoCracks adds a vertical crack from (-0.125,
//! -0.375) to (-0.125, 0.375) crossing the first one, enriched at both
//! tips, its jump penalized (Xfem_plus, Xfem_minus). Writes the
//! displacement on the cut mesh to target/showcase/crack.vtu. Optional
//! argument: elements per side (default 20, as the demo).

use getfem::fem::crack_tip_function::{CrackTipFunction, Cutoff, Singular};
use getfem::fem::fem::fem_descriptor;
use getfem::fem::global_function::GlobalFunction;
use getfem::fem::integration::integ_descriptor;
use getfem::fem::interpolation::interpolation;
use getfem::fem::level_set::LevelSet;
use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im_level_set::{IntegrateWhere, MeshImLevelSet};
use getfem::fem::mesh_level_set::MeshLevelSet;
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::mesh::Mesh;
use getfem::geom::region::outer_faces_of_mesh;
use getfem::geom::regular_mesh::regular_mesh;
use getfem::gwfl::workspace::Workspace;
use getfem::io::vtk::{Format, VtuExport};
use getfem::model::{Model, Newton};

const DIRICHLET: usize = 101;
const LAMBDA: f64 = 1.25e10;
const MU: f64 = 1.875e10;

#[derive(Clone, Copy, PartialEq)]
pub enum Variant {
  Cutoff,
  Product,
  TwoCracks,
}

pub struct Solution {
  pub mls: MeshLevelSet,
  pub mf_u: MeshFem,
  pub u: Vec<f64>,
  /// L2 and H1 norms of u
  pub norms: [f64; 2],
  /// L2 and H1 distances to the reference solution
  pub dist: [f64; 2],
  /// L2 norm of the jump of u across the second crack
  pub jump: Option<f64>,
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

/// a level set with its primary and secondary functions (GWFL in x, y)
fn level_set(m: &Mesh, f0: &str, f1: &str) -> LevelSet {
  let ls = LevelSet::new(m, 1, true).unwrap();
  ls.set_values(0, &values(&ls, f0)).unwrap();
  ls.set_values(1, &values(&ls, f1)).unwrap();
  ls
}

/// the space of the singular functions on the level set ls
fn singular_space(m: &Mesh, ls: &LevelSet, cutoff: Option<&CrackTipFunction>) -> MeshFem {
  let funcs: Vec<GlobalFunction> = Singular::LEFM
    .iter()
    .map(|&s| {
      let f = CrackTipFunction::crack(s);
      let f = match cutoff {
        Some(c) => CrackTipFunction::product(&f, c),
        None => f,
      };
      GlobalFunction::on_level_set(ls, &f).unwrap()
    })
    .collect();
  MeshFem::from_global_functions(m, &funcs, None).unwrap()
}

pub fn solve(variant: Variant, nx: usize) -> Solution {
  let mesh = regular_mesh(
    &geotrans_descriptor("GT_PK(2,1)").unwrap(),
    &[nx, nx],
    &[-0.5, -0.5],
    &[1.0, 1.0],
  )
  .unwrap();
  mesh
    .set_region(DIRICHLET, &outer_faces_of_mesh(&mesh))
    .unwrap();
  let ls = level_set(&mesh, "y", "x");
  let ls2 = level_set(&mesh, "x+0.125", "abs(y)-0.375");
  let mut mls = MeshLevelSet::new(&mesh);
  mls.add_level_set(&ls);
  if variant == Variant::TwoCracks {
    mls.add_level_set(&ls2);
  }
  mls.adapt().unwrap();

  let p1 = fem_descriptor("FEM_PK(2,1)").unwrap();
  let mf_pre_u = MeshFem::new(&mesh, 1).with_finite_element(&p1).unwrap();
  let mfls_u = MeshFem::from_level_set(&mls, &mf_pre_u).unwrap();
  let mf_u = if variant == Variant::Cutoff {
    let cutoff = CrackTipFunction::cutoff(Cutoff::Polynomial2 {
      inner: 0.01,
      outer: 0.4,
    })
    .unwrap();
    let global_mf = singular_space(&mesh, &ls, Some(&cutoff));
    MeshFem::from_sum(&[&global_mf, &mfls_u]).unwrap()
  } else {
    // the singular functions on the P1 partition of unity of the nodes
    // within 0.1 of the tips
    let pu_mf = MeshFem::new(&mesh, 1)
      .with_classical_finite_element(1)
      .unwrap();
    let near = |c: [f64; 2]| -> std::collections::BTreeSet<usize> {
      (0..pu_mf.nb_dof().unwrap())
        .filter(|&d| {
          let p = pu_mf.point_of_basic_dof(d).unwrap();
          // the nodes at 0.1 included (the demo's coordinates round inward)
          (p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) <= 0.1 * 0.1 + 1e-12
        })
        .collect()
    };
    let global_mf = singular_space(&mesh, &ls, None);
    let mf_xfem_sing = MeshFem::from_product(&pu_mf, &global_mf, Some(&near([0.0, 0.0]))).unwrap();
    if variant == Variant::Product {
      MeshFem::from_sum(&[&mf_xfem_sing, &mfls_u]).unwrap()
    } else {
      let global_mf2 = singular_space(&mesh, &ls2, None);
      let mut pu_dofs = near([-0.125, 0.375]);
      pu_dofs.extend(near([-0.125, -0.375]));
      let mf_xfem_sing2 = MeshFem::from_product(&pu_mf, &global_mf2, Some(&pu_dofs)).unwrap();
      MeshFem::from_sum(&[&mf_xfem_sing, &mf_xfem_sing2, &mfls_u]).unwrap()
    }
  };
  mf_u.set_qdim(2);

  let mim = MeshImLevelSet::new(&mls, IntegrateWhere::All).unwrap();
  mim
    .set_simplex_im(
      &integ_descriptor("IM_STRUCTURED_COMPOSITE(IM_TRIANGLE(6),3)").unwrap(),
      Some(&integ_descriptor("IM_STRUCTURED_COMPOSITE(IM_GAUSS_PARALLELEPIPED(2,6),9)").unwrap()),
      // the rule on every sub-triangle (tau 0)
      0.0,
    )
    .unwrap();
  mim
    .set_integration_method(&integ_descriptor("IM_STRUCTURED_COMPOSITE(IM_TRIANGLE(6),5)").unwrap())
    .unwrap();
  mim.adapt().unwrap();

  // the reference solution: the opening mode in the singular functions
  let mf_ue = singular_space(&mesh, &ls, None);
  mf_ue.set_qdim(2);
  let (a, b) = (
    2.0 + 2.0 * MU / (LAMBDA + 2.0 * MU),
    -2.0 * (LAMBDA + MU) / (LAMBDA + 2.0 * MU),
  );
  let ue: Vec<f64> = [0.0, a - b, a + b, 0.0, -b, 0.0, 0.0, b]
    .iter()
    .map(|v| v / (2.0 * std::f64::consts::PI))
    .collect();

  let mut md = Model::new();
  md.add_fem_variable("u", &mf_u).unwrap();
  md.add_scalar_data("lambda", LAMBDA).unwrap();
  md.add_scalar_data("mu", MU).unwrap();
  md.add_linear_term(
    &mim,
    None,
    "lambda*Div_u*Div_Test_u + mu*(Grad_u + Grad_u'):Grad_Test_u",
  )
  .unwrap();
  md.add_fem_data("DirichletData", &mf_ue, &ue).unwrap();
  md.add_Dirichlet_condition_with_penalization(
    &mim,
    "u",
    1e12,
    mesh.region(DIRICHLET).unwrap(),
    Some("DirichletData"),
  )
  .unwrap();
  // the jump across the second crack penalized
  let mim_bound2 = (variant == Variant::TwoCracks).then(|| {
    let mim_bound2 = MeshImLevelSet::new(&mls, IntegrateWhere::Boundary).unwrap();
    mim_bound2.set_level_set_boolean_operations("b").unwrap();
    mim_bound2
      .set_simplex_im(
        &integ_descriptor("IM_STRUCTURED_COMPOSITE(IM_TRIANGLE(6),3)").unwrap(),
        None,
        0.0,
      )
      .unwrap();
    mim_bound2.set_integration_method(1usize).unwrap();
    mim_bound2.adapt().unwrap();
    md.add_linear_term(
      &mim_bound2,
      None,
      "1e17*(Xfem_plus(u)-Xfem_minus(u)).(Xfem_plus(Test_u)-Xfem_minus(Test_u))",
    )
    .unwrap();
    mim_bound2
  });
  Newton::default().solve(&mut md, None).unwrap();
  let u = md.variable("u").unwrap().to_vec();

  let mut ws = Workspace::new();
  ws.add_fem_data("u", &mf_u, &u).unwrap();
  ws.add_fem_data("Ue", &mf_ue, &ue).unwrap();
  let mut int = |e: &str| {
    ws.clear_expressions();
    ws.add_expression(e, &mim, None).unwrap();
    ws.assembly(0, false).unwrap();
    ws.assembled_potential()
  };
  let l2 = int("Norm_sqr(u)");
  let d2 = int("Norm_sqr(u-Ue)");
  let norms = [l2.sqrt(), (l2 + int("Norm_sqr(Grad_u)")).sqrt()];
  let dist = [d2.sqrt(), (d2 + int("Norm_sqr(Grad_u-Grad_Ue)")).sqrt()];
  let jump = mim_bound2.map(|mb| {
    ({
      ws.clear_expressions();
      ws.add_expression("Norm_sqr(Xfem_plus(u)-Xfem_minus(u))", &mb, None)
        .unwrap();
      ws.assembly(0, false).unwrap();
      ws.assembled_potential()
    })
    .sqrt()
  });
  Solution {
    mls,
    mf_u,
    u,
    norms,
    dist,
    jump,
  }
}

fn main() {
  let nx: usize = std::env::args().nth(1).map_or(20, |a| a.parse().unwrap());
  let s = solve(Variant::TwoCracks, nx);
  println!(
    "{} DOFs; |u| L2 {:.6e}, H1 {:.6e}; |u - ue| L2 {:.6e}, H1 {:.6e}; jump {:.3e}",
    s.mf_u.nb_dof().unwrap(),
    s.norms[0],
    s.norms[1],
    s.dist[0],
    s.dist[1],
    s.jump.unwrap_or(0.0)
  );
  // the displacement on the cut mesh, discontinuous across the cracks
  // (alpha 0.001: the nodes just off the cracks, each side its value)
  let cut = s.mls.global_cut_mesh().unwrap();
  let mfv = MeshFem::new(&cut, 2)
    .with_finite_element(&fem_descriptor("FEM_PK_DISCONTINUOUS(2,2,0.001)").unwrap())
    .unwrap();
  let v = interpolation(&s.mf_u, &mfv, &s.u).unwrap();
  let out = format!("{}/../../target/showcase", env!("CARGO_MANIFEST_DIR"));
  std::fs::create_dir_all(&out).unwrap();
  let mut vtu = VtuExport::new(&format!("{out}/crack.vtu"), Format::Compressed);
  vtu.exporting(&mfv).unwrap();
  vtu.write_point_data(&mfv, &v, "u").unwrap();
  vtu.write().unwrap();
}
