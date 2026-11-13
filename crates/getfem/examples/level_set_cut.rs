//! Two crossing spheres cut a tetrahedral mesh of [-1, 1]^3 (level sets of
//! degree 2): the volumes of their intersection, difference and union and
//! the area of the first sphere by mesh_im_level_set, against the exact
//! values. Writes the cut mesh (the cut tetrahedra replaced by their
//! sub-tetrahedra) with the zone of each cell (0 outside, 1 in a only, 2
//! in b only, 3 in both) to target/showcase/level_set_cut.vtu. Optional
//! argument: the cubes per direction (default 6).

use std::f64::consts::PI;

use getfem::fem::integration::integ_descriptor;
use getfem::fem::level_set::LevelSet;
use getfem::fem::mesh_im_level_set::{CELL_TOLERANCE, IntegrateWhere, MeshImLevelSet};
use getfem::fem::mesh_level_set::{MeshLevelSet, jacobian};
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::mesh::Mesh;
use getfem::geom::regular_mesh::regular_mesh;
use getfem::io::vtk::{Format, VtuExport};

const C1: [f64; 3] = [-0.15, 0.02, 0.01];
const C2: [f64; 3] = [0.25, 0.05, -0.02];
const R1: f64 = 0.45;
const R2: f64 = 0.4;

fn inside(x: &[f64], c: [f64; 3], r: f64) -> bool {
  (0..3).map(|k| (x[k] - c[k]).powi(2)).sum::<f64>() < r * r
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

/// the measure of the part selected by csg
fn measure(m: &Mesh, mls: &MeshLevelSet, w: IntegrateWhere, csg: &str) -> f64 {
  let mim = MeshImLevelSet::new(mls, w).unwrap();
  mim.set_level_set_boolean_operations(csg).unwrap();
  let pim = integ_descriptor("IM_TETRAHEDRON(5)").unwrap();
  mim.set_integration_method(&pim).unwrap();
  mim.set_simplex_im(&pim, None, CELL_TOLERANCE).unwrap();
  mim.adapt().unwrap();
  let md = m.snapshot();
  let mut s = 0.0;
  for cv in md.convex_index() {
    let a = mim.int_method_of_element(cv).unwrap();
    let a = a.approx().unwrap();
    let (gt, g) = (md.trans_of_convex(cv), md.points_of_convex(cv));
    for i in 0..a.nb_points_on_convex() {
      s += a.coeff(i) * jacobian(gt, &g, a.point(i)).abs();
    }
  }
  s
}

fn main() {
  let out = format!("{}/../../target/showcase", env!("CARGO_MANIFEST_DIR"));
  std::fs::create_dir_all(&out).unwrap();
  let n: usize = std::env::args().nth(1).map_or(6, |a| a.parse().unwrap());
  let m = regular_mesh(
    &geotrans_descriptor("GT_PK(3,1)").unwrap(),
    &[n; 3],
    &[-1.0; 3],
    &[2.0; 3],
  )
  .unwrap();
  let mut mls = MeshLevelSet::new(&m);
  mls.add_level_set(&sphere(&m, C1, R1));
  mls.add_level_set(&sphere(&m, C2, R2));
  mls.adapt().unwrap();

  // exact: the lens of the two balls, and the balls
  let d = (0..3).map(|k| (C2[k] - C1[k]).powi(2)).sum::<f64>().sqrt();
  let lens = PI
    * (R1 + R2 - d).powi(2)
    * (d * d + 2.0 * d * R2 - 3.0 * R2 * R2 + 2.0 * d * R1 + 6.0 * R1 * R2 - 3.0 * R1 * R1)
    / (12.0 * d);
  let (va, vb) = (4.0 * PI * R1.powi(3) / 3.0, 4.0 * PI * R2.powi(3) / 3.0);
  for (csg, exact) in [("a*b", lens), ("a-b", va - lens), ("a+b", va + vb - lens)] {
    let v = measure(&m, &mls, IntegrateWhere::Inside, csg);
    println!(
      "volume {csg}: {v:.8} (exact {exact:.8}, error {:.1e})",
      v / exact - 1.0
    );
  }
  let a = measure(&m, &mls, IntegrateWhere::Boundary, "a");
  let ae = 4.0 * PI * R1 * R1;
  println!(
    "area of a: {a:.8} (exact {ae:.8}, error {:.1e})",
    a / ae - 1.0
  );

  // the cut mesh with the zone of each cell (at its centroid)
  let cut = mls.global_cut_mesh().unwrap();
  let md = cut.snapshot();
  let zone: Vec<f64> = md
    .convex_index()
    .map(|cv| {
      let g = md.points_of_convex(cv);
      let c: Vec<f64> = (0..3)
        .map(|k| (0..g.ncols()).map(|j| g.col(j)[k]).sum::<f64>() / g.ncols() as f64)
        .collect();
      (inside(&c, C1, R1) as usize + 2 * inside(&c, C2, R2) as usize) as f64
    })
    .collect();
  println!(
    "{} convexes, {} in the cut mesh",
    m.snapshot().nb_convexes(),
    md.nb_convexes()
  );
  let mut vtu = VtuExport::new(&format!("{out}/level_set_cut.vtu"), Format::Compressed);
  vtu.exporting(&cut).unwrap();
  vtu.write_cell_data(&zone, "zone", 1).unwrap();
  vtu.write().unwrap();
}
