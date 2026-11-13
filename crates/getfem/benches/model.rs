//! Model assembly benchmark: the tangent and the residual of a model as
//! Newton assembles them (Model::assembly); bench/model.cc runs the same
//! with GetFEM (model::assembly(BUILD_ALL)).
//!
//!   cargo bench -p getfem --bench model [-- [repeats [case]]]
//!
//! prints one CSV line per case:
//! case,dofs,first_s,min_s,checksum,solve_s,asm_after_s.
//! Model:
//! a vector field u on [0,1]^d, the nonlinear term
//!   (mu (Grad_u + Grad_u') + lambda Trace(Grad_u) Id + 0.1 Grad_u Grad_u'):Grad_Test_u,
//! the linear terms 1e-3 u.Test_u and -f.Test_u, u = 0 on the face x_d = 0
//! by multipliers (classical integration); the named element and integration method, or the
//! classical element of degree k with integration of degree 2k. first:
//! the first assembly (the linear terms assembled and kept); min: the
//! fastest of the following ones; checksum: |residual| + nnz(tangent);
//! solve: the fastest of two MUMPS solves of the last tangent, as a Newton
//! iteration does (analysis, factorization, solve; not in bench/model.cc);
//! asm_after: the fastest assembly right after a solve (the BLAS threads of
//! MUMPS just done). Default: 4 repeats. BENCH_THREADS=t assembles with t
//! threads, BENCH_MUMPS_THREADS=m solves with m (default 1). BENCH_SCALE=s
//! (a float) multiplies the cells per side by s. BENCH_MPI=1 calls
//! getfem::mpi::init (run under mpirun, built with the feature mpi): the
//! ranks synchronized before each timing, rank 0 prints. The cases <case>_terms: the same model as five nonlinear terms
//! (as several bricks: assembled together, their trees joined). The cases
//! <case>_curved: the mesh nodes moved by curve() (curved elements; the
//! second-order transformations only).

use std::time::Instant;

use getfem::fem::fem::fem_descriptor;
use getfem::fem::integration::integ_descriptor;
use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im::MeshIm;
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::region::{outer_faces_of_mesh, select_faces_of_normal};
use getfem::geom::regular_mesh::regular_mesh;
use getfem::linalg::solver::{LinearSolver, MatrixKind};
use getfem::model::{BuildVersion, Model};

/// (name, geometric transformation, cells per side, element degree,
/// element, integration method; "": classical of degree k, 2k)
type Case = (
  &'static str,
  &'static str,
  usize,
  usize,
  &'static str,
  &'static str,
);

const CASES: [Case; 9] = [
  ("P2_tetrahedra", "GT_PK(3,1)", 10, 2, "", ""),
  ("Q2_hexahedra", "GT_QK(3,1)", 10, 2, "", ""),
  ("Q1_hexahedra", "GT_QK(3,1)", 16, 1, "", ""),
  ("P2_isoparametric_triangles", "GT_PK(2,2)", 100, 2, "", ""),
  ("P2_isoparametric_tetrahedra", "GT_PK(3,2)", 10, 2, "", ""),
  (
    "hex27_gauss27",
    "GT_QK(3,2)",
    10,
    2,
    "FEM_QK(3,2)",
    "IM_GAUSS_PARALLELEPIPED(3,5)",
  ),
  (
    "hex27_hex14",
    "GT_QK(3,2)",
    10,
    2,
    "FEM_QK(3,2)",
    "IM_HEXAHEDRON(5)",
  ),
  (
    "hex20_gauss27",
    "GT_Q2_INCOMPLETE(3)",
    10,
    2,
    "FEM_Q2_INCOMPLETE(3)",
    "IM_GAUSS_PARALLELEPIPED(3,5)",
  ),
  (
    "hex20_hex14",
    "GT_Q2_INCOMPLETE(3)",
    10,
    2,
    "FEM_Q2_INCOMPLETE(3)",
    "IM_HEXAHEDRON(5)",
  ),
];

/// the cases also run as five nonlinear terms
const SPLIT: [&str; 2] = ["P2_tetrahedra", "Q2_hexahedra"];

/// the cases also run on a curved mesh
const CURVED: [&str; 4] = [
  "P2_isoparametric_triangles",
  "P2_isoparametric_tetrahedra",
  "hex27_hex14",
  "hex20_hex14",
];

/// the unit cube's nodes moved by x + a b(x), b = prod_j sin(pi x_j): curved
/// elements (mid-side nodes off the chords), the boundary fixed
fn curve(m: &getfem::geom::mesh::Mesh) {
  let a = [0.1, 0.07, 0.05];
  let moves: Vec<_> = m
    .point_index()
    .into_iter()
    .map(|i| {
      let mut x = m.point(i).to_vec();
      let b: f64 = x
        .iter()
        .map(|xj| (std::f64::consts::PI * xj).sin())
        .product();
      x.iter_mut().zip(a).for_each(|(xj, aj)| *xj += aj * b);
      (i, x)
    })
    .collect();
  m.set_points(&moves).unwrap();
}

/// the model of the cases as five nonlinear terms
const SPLIT_TERMS: [&str; 5] = [
  "mu*(Grad_u+Grad_u'):Grad_Test_u",
  "lambda*Trace(Grad_u)*Div_Test_u",
  "0.1*(Grad_u*Grad_u'):Grad_Test_u",
  "1e-3*u.Test_u",
  "-f.Test_u",
];

/// (dofs, first, min of the others, checksum)
fn run(
  (_, gt, n, k, fem, im): Case,
  split: bool,
  curved: bool,
  repeats: usize,
) -> (usize, f64, f64, f64, f64, f64) {
  let pgt = geotrans_descriptor(gt).unwrap();
  let d = pgt.dim();
  let scale: f64 = std::env::var("BENCH_SCALE").map_or(1.0, |v| v.parse().unwrap());
  let n = (n as f64 * scale).round() as usize;
  // MPI: the ranks synchronized before a timing (a sum waits for all)
  let sync = || getfem::mpi::sum(&mut [0.0]);
  let m = regular_mesh(&pgt, &vec![n; d], &vec![0.0; d], &vec![1.0; d]).unwrap();
  if curved {
    curve(&m);
  }
  let mut down = vec![0.0; d];
  down[d - 1] = -1.0;
  let outer = outer_faces_of_mesh(&m);
  m.set_region(1, &select_faces_of_normal(&m, &outer, &down, 0.01).unwrap())
    .unwrap();
  let mf = if fem.is_empty() {
    MeshFem::new(&m, d).with_classical_finite_element(k)
  } else {
    MeshFem::new(&m, d).with_finite_element(&fem_descriptor(fem).unwrap())
  }
  .unwrap();
  let mim = if im.is_empty() {
    MeshIm::new(&m).with_integration_method(2 * k)
  } else {
    MeshIm::new(&m).with_integration_method(&integ_descriptor(im).unwrap())
  }
  .unwrap();
  let mut md = Model::new();
  md.add_fem_variable("u", &mf).unwrap();
  md.add_scalar_data("lambda", 2.0).unwrap();
  md.add_scalar_data("mu", 1.0).unwrap();
  let mut f = vec![0.0; d];
  f[d - 1] = 1.0;
  md.add_fixed_size_data("f", &[d], &f).unwrap();
  if split {
    for e in SPLIT_TERMS {
      md.add_nonlinear_term(&mim, None, e).unwrap();
    }
  } else {
    md.add_nonlinear_term(
      &mim,
      None,
      "(mu*(Grad_u+Grad_u') + lambda*Trace(Grad_u)*Id(meshdim) + 0.1*Grad_u*Grad_u'):Grad_Test_u",
    )
    .unwrap();
    md.add_linear_term(&mim, None, "1e-3*u.Test_u").unwrap();
    md.add_linear_term(&mim, None, "-f.Test_u").unwrap();
  }
  // the boundary condition with the classical method (its constraints on a
  // named non-Gauss rule may be dependent)
  let mim_d = MeshIm::new(&m).with_integration_method(2 * k).unwrap();
  md.add_Dirichlet_condition_with_multiplier(&mim_d, "u", &mf, 1, None)
    .unwrap();
  let nd = mf.nb_dof().unwrap();
  let u: Vec<f64> = (0..nd).map(|i| 1e-3 * (i % 7) as f64).collect();
  md.set_variable("u", &u).unwrap();
  let (mut first, mut best, mut sum) = (0.0, f64::MAX, 0.0);
  for r in 0..=repeats {
    sync();
    let t0 = Instant::now();
    md.assembly(BuildVersion::All).unwrap();
    let (res, kt) = (md.rhs(), md.tangent_matrix());
    let t = t0.elapsed().as_secs_f64();
    if r == 0 {
      first = t;
    } else {
      best = best.min(t);
    }
    // MPI: each rank's tangent its elements' part
    let mut nnz = [kt.nnz() as f64];
    getfem::mpi::sum(&mut nnz);
    sum = res.iter().map(|x| x * x).sum::<f64>().sqrt() + nnz[0];
  }
  let (mut solve, mut after) = (f64::MAX, f64::MAX);
  for _ in 0..2 {
    let mut b = md.rhs().to_vec();
    let mut mumps = getfem::mumps::Mumps::new();
    sync();
    let t0 = Instant::now();
    mumps
      .solve_system(md.tangent_matrix(), MatrixKind::General, &mut b)
      .unwrap();
    solve = solve.min(t0.elapsed().as_secs_f64());
    sync();
    let t0 = Instant::now();
    md.assembly(BuildVersion::All).unwrap();
    after = after.min(t0.elapsed().as_secs_f64());
  }
  (md.nb_dof(), first, best, sum, solve, after)
}

fn main() {
  let args: Vec<String> = std::env::args()
    .skip(1)
    .filter(|a| !a.starts_with('-'))
    .collect();
  let repeats: usize = args.first().map_or(4, |r| r.parse().unwrap());
  let _mpi = std::env::var_os("BENCH_MPI").map(|_| getfem::mpi::init().unwrap());
  let master = getfem::mpi::is_master();
  let threads = std::env::var("BENCH_THREADS").map_or(1, |v| v.parse().unwrap());
  getfem::set_num_threads(threads).unwrap();
  let mumps_threads = std::env::var("BENCH_MUMPS_THREADS").map_or(1, |v| v.parse().unwrap());
  getfem::mumps::set_num_threads(mumps_threads).unwrap();
  if master {
    println!("case,dofs,first_s,min_s,checksum,solve_s,asm_after_s");
  }
  // (case, split, curved)
  let plain = CASES.iter().map(|&c| (c, false, false));
  let split = CASES
    .iter()
    .filter(|c| SPLIT.contains(&c.0))
    .map(|&c| (c, true, false));
  let curved = CASES
    .iter()
    .filter(|c| CURVED.contains(&c.0))
    .map(|&c| (c, false, true));
  for (case, sp, cv) in plain.chain(split).chain(curved) {
    let name = case.0;
    let name = match (sp, cv) {
      (true, _) => format!("{name}_terms"),
      (_, true) => format!("{name}_curved"),
      _ => name.to_string(),
    };
    if args.get(1).is_some_and(|c| *c != name) {
      continue;
    }
    let (nd, first, best, sum, solve, after) = run(case, sp, cv, repeats);
    if master {
      println!("{name},{nd},{first:.4},{best:.4},{sum:.10e},{solve:.4},{after:.4}");
    }
  }
}
