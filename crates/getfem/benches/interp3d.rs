//! 3D interpolation benchmark: evaluation of finite element fields at
//! arbitrary points of curved second order meshes; bench/interp3d.cc runs
//! the same with GetFEM.
//!
//!   cargo bench -p getfem --bench interp3d [-- <n> [repeats [case]]]
//!
//! prints one CSV line per case: n,case,min_s,median_s,points,checksum.
//! Source mesh: [0,1]^3 in n^3 cells, its inner nodes moved by
//! 0.06 sin(pi x) sin(pi y) sin(pi z) (1, -1, 1) (curved elements, planar
//! boundary), the field f = cos(2x) + y z + x y z on a second order space.
//! Cases (timed from scratch):
//! - points_hex, points_tet: f at 20000 pseudo-random points (GT_QK(3,2)
//!   with FEM_QK(3,2); GT_PK(3,2) with FEM_PK(3,2)), checksum: their sum;
//! - interpolated_fem: the hexahedral space on a tetrahedral mesh of
//!   [0.02,0.98]^3 in (n+1)^3 cubes (IM_TETRAHEDRON(5)), the integral of f
//!   assembled with it, checksum: the integral.
//!
//! Default: n = 4, 6, 8, 3 repeats.

use std::f64::consts::PI;
use std::time::Instant;

use getfem::fem::fem::fem_descriptor;
use getfem::fem::integration::integ_descriptor;
use getfem::fem::interpolated_fem::new_interpolated_fem;
use getfem::fem::interpolation::interpolation_at_points;
use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im::MeshIm;
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::mesh::Mesh;
use getfem::geom::regular_mesh::regular_mesh;
use getfem::gwfl::workspace::Workspace;

const NB_POINTS: usize = 20000;

fn f(p: &[f64]) -> f64 {
  (2.0 * p[0]).cos() + p[1] * p[2] + p[0] * p[1] * p[2]
}

/// the curved source mesh
fn source_mesh(gt: &str, n: usize) -> Mesh {
  let m = regular_mesh(
    &geotrans_descriptor(gt).unwrap(),
    &[n; 3],
    &[0.0; 3],
    &[1.0; 3],
  )
  .unwrap();
  let moves: Vec<(usize, Vec<f64>)> = m
    .point_index()
    .into_iter()
    .map(|i| {
      let p = m.point(i);
      let a = 0.06 * (PI * p[0]).sin() * (PI * p[1]).sin() * (PI * p[2]).sin();
      (i, vec![p[0] + a, p[1] - a, p[2] + a])
    })
    .collect();
  m.set_points(&moves).unwrap();
  m
}

/// the same pseudo-random points as bench/interp3d.cc
fn points() -> Vec<Vec<f64>> {
  let mut s: u64 = 12345;
  let mut next = || {
    s = s
      .wrapping_mul(6364136223846793005)
      .wrapping_add(1442695040888963407);
    (s >> 11) as f64 / (1u64 << 53) as f64
  };
  (0..NB_POINTS)
    .map(|_| (0..3).map(|_| 0.001 + 0.998 * next()).collect())
    .collect()
}

/// f at the DOFs of a scalar space
fn field(mf: &MeshFem) -> Vec<f64> {
  (0..mf.nb_dof().unwrap())
    .map(|i| f(&mf.point_of_basic_dof(i).unwrap()))
    .collect()
}

/// (seconds, points, checksum) of one run of a case
fn run(case: &str, n: usize, pts: &[Vec<f64>]) -> (f64, usize, f64) {
  let (gt, fem) = match case {
    "points_tet" => ("GT_PK(3,2)", "FEM_PK(3,2)"),
    _ => ("GT_QK(3,2)", "FEM_QK(3,2)"),
  };
  let m = source_mesh(gt, n);
  let mf = MeshFem::new(&m, 1)
    .with_finite_element(&fem_descriptor(fem).unwrap())
    .unwrap();
  let u = field(&mf);
  if case == "interpolated_fem" {
    let mt = regular_mesh(
      &geotrans_descriptor("GT_PK(3,1)").unwrap(),
      &[n + 1; 3],
      &[0.02; 3],
      &[0.96; 3],
    )
    .unwrap();
    let mim = MeshIm::new(&mt)
      .with_integration_method(&integ_descriptor("IM_TETRAHEDRON(5)").unwrap())
      .unwrap();
    let t0 = Instant::now();
    let mfi = new_interpolated_fem(&mf, &mim).unwrap();
    let mut ws = Workspace::new();
    ws.add_fem_data("u", &mfi, &u).unwrap();
    let s = {
      ws.clear_expressions();
      ws.add_expression("u", &mim, None).unwrap();
      ws.assembly(0, false).unwrap();
      ws.assembled_potential()
    };
    let l = mim.ims().unwrap();
    let np = l
      .convex_index()
      .map(|cv| {
        let a = l.int_method_of_element(cv).unwrap();
        a.approx().unwrap().nb_points_on_convex()
      })
      .sum();
    (t0.elapsed().as_secs_f64(), np, s)
  } else {
    let t0 = Instant::now();
    let v = interpolation_at_points(&mf, pts, &u).unwrap();
    (t0.elapsed().as_secs_f64(), pts.len(), v.iter().sum())
  }
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
  let pts = points();
  println!("n,case,min_s,median_s,points,checksum");
  for &n in &ns {
    for case in ["points_hex", "points_tet", "interpolated_fem"] {
      if args.get(2).is_some_and(|c| c != case) {
        continue;
      }
      let mut ts = vec![];
      let (mut np, mut sum) = (0, 0.0);
      for _ in 0..repeats {
        let (t, p, s) = run(case, n, &pts);
        ts.push(t);
        (np, sum) = (p, s);
      }
      ts.sort_by(f64::total_cmp);
      println!(
        "{n},{case},{:.4},{:.4},{np},{sum:.12e}",
        ts[0],
        ts[ts.len() / 2]
      );
    }
  }
}
