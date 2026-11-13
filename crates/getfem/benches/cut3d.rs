//! 3D cut-cell benchmark: integration on tetrahedral meshes cut by spheres
//! (mesh_level_set + mesh_im_level_set), bench/cut3d.cc runs the same with
//! GetFEM.
//!
//!   cargo bench -p getfem --bench cut3d [-- <n> [repeats]]
//!
//! prints one CSV line per case: n,case,min_s,median_s,points,rel_error.
//! A case builds the level sets, the cut and the methods from scratch
//! (timed), then integrates 1 (not timed). Mesh: [-1,1]^3, n^3 cubes in
//! tetrahedra; rule IM_TETRAHEDRON(5) on the elements and the
//! sub-tetrahedra. Default: n = 4, 6, 8, 3 repeats.

use std::f64::consts::PI;
use std::time::Instant;

use getfem::fem::integration::integ_descriptor;
use getfem::fem::level_set::LevelSet;
use getfem::fem::mesh_im_level_set::{CELL_TOLERANCE, IntegrateWhere, MeshImLevelSet};
use getfem::fem::mesh_level_set::{MeshLevelSet, jacobian};
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::mesh::Mesh;
use getfem::geom::regular_mesh::regular_mesh;

const C0: [f64; 3] = [0.03, 0.02, 0.01];
const R0: f64 = 0.63;
const C1: [f64; 3] = [-0.15, 0.02, 0.01];
const C2: [f64; 3] = [0.25, 0.05, -0.02];
const R1: f64 = 0.45;
const R2: f64 = 0.4;

struct Case<'a> {
  name: &'a str,
  spheres: &'a [([f64; 3], f64)],
  w: IntegrateWhere,
  csg: &'a str,
  exact: f64,
}

fn sphere(m: &Mesh, c: [f64; 3], r: f64) -> LevelSet {
  let ls = LevelSet::new(m, 2, false).unwrap();
  let mf = ls.mesh_fem();
  let v: Vec<f64> = (0..mf.nb_dof().unwrap())
    .map(|i| {
      let p = mf.point_of_basic_dof(i).unwrap();
      (0..3).map(|k| (p[k] - c[k]).powi(2)).sum::<f64>() - r * r
    })
    .collect();
  ls.set_values(0, &v).unwrap();
  ls
}

/// the methods of the case (timed part)
fn build(m: &Mesh, spheres: &[([f64; 3], f64)], w: IntegrateWhere, csg: &str) -> MeshImLevelSet {
  let mut mls = MeshLevelSet::new(m);
  for &(c, r) in spheres {
    mls.add_level_set(&sphere(m, c, r));
  }
  let mim = MeshImLevelSet::new(&mls, w).unwrap();
  if !csg.is_empty() {
    mim.set_level_set_boolean_operations(csg).unwrap();
  }
  let pim = integ_descriptor("IM_TETRAHEDRON(5)").unwrap();
  mim.set_integration_method(&pim).unwrap();
  mim.set_simplex_im(&pim, None, CELL_TOLERANCE).unwrap();
  mim.adapt().unwrap();
  mim
}

/// integral of 1 and number of points
fn measure(m: &Mesh, mim: &MeshImLevelSet) -> (f64, usize) {
  let md = m.snapshot();
  let (mut s, mut np) = (0.0, 0);
  for cv in md.convex_index() {
    let a = mim.int_method_of_element(cv).unwrap();
    let a = a.approx().unwrap();
    let (gt, g) = (md.trans_of_convex(cv), md.points_of_convex(cv));
    for i in 0..a.nb_points_on_convex() {
      s += a.coeff(i) * jacobian(gt, &g, a.point(i)).abs();
    }
    np += a.nb_points_on_convex();
  }
  (s, np)
}

fn main() {
  let args: Vec<String> = std::env::args()
    .skip(1)
    .filter(|a| !a.starts_with('-'))
    .collect();
  let ns: Vec<usize> = match args.first() {
    Some(n) => vec![n.parse().unwrap()],
    None => vec![4, 6, 8],
  };
  let repeats: usize = args.get(1).map_or(3, |r| r.parse().unwrap());
  let d = (0..3).map(|k| (C2[k] - C1[k]).powi(2)).sum::<f64>().sqrt();
  let lens = PI
    * (R1 + R2 - d).powi(2)
    * (d * d + 2.0 * d * R2 - 3.0 * R2 * R2 + 2.0 * d * R1 + 6.0 * R1 * R2 - 3.0 * R1 * R1)
    / (12.0 * d);
  let (va, vb) = (4.0 * PI * R1.powi(3) / 3.0, 4.0 * PI * R2.powi(3) / 3.0);
  let one = vec![(C0, R0)];
  let two = vec![(C1, R1), (C2, R2)];
  use IntegrateWhere::{Boundary, Inside};
  let c = |name, spheres, w, csg, exact| Case {
    name,
    spheres,
    w,
    csg,
    exact,
  };
  let cases = [
    c(
      "sphere_volume",
      &one,
      Inside,
      "",
      4.0 * PI * R0.powi(3) / 3.0,
    ),
    c("sphere_area", &one, Boundary, "", 4.0 * PI * R0 * R0),
    c("a*b", &two, Inside, "a*b", lens),
    c("a-b", &two, Inside, "a-b", va - lens),
    c("a+b", &two, Inside, "a+b", va + vb - lens),
  ];
  println!("n,case,min_s,median_s,points,rel_error");
  for &n in &ns {
    let m = regular_mesh(
      &geotrans_descriptor("GT_PK(3,1)").unwrap(),
      &[n; 3],
      &[-1.0; 3],
      &[2.0; 3],
    )
    .unwrap();
    for Case {
      name,
      spheres,
      w,
      csg,
      exact,
    } in &cases
    {
      let mut times = vec![];
      let mut last = None;
      for _ in 0..repeats {
        let t = Instant::now();
        let mim = build(&m, spheres, *w, csg);
        times.push(t.elapsed().as_secs_f64());
        last = Some(mim);
      }
      times.sort_by(|a, b| a.partial_cmp(b).unwrap());
      let (v, np) = measure(&m, &last.unwrap());
      println!(
        "{n},{name},{:.6},{:.6},{np},{:.3e}",
        times[0],
        times[times.len() / 2],
        v / exact - 1.0
      );
    }
  }
}
