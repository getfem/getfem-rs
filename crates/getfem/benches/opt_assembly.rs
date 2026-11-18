//! GetFEM's contrib/opt_assembly (its selected cases), the new assembly
//! only (bench/opt_assembly.cc runs the same with GetFEM). Random data
//! replaced by smooth fields (checksums comparable).
//!
//!   cargo bench -p getfem --bench opt_assembly [-- <config> [case] [repeats]]
//!
//! prints config,case,expr,ndof,min_s,median_s,checksum. With a case index
//! the case runs alone, once unless the repeats are given (for callgrind:
//! the assembly is the function `measured`, --toggle-collect='*measured*').
//! BENCH_THREADS=t assembles with t threads (default 1).

use std::hint::black_box;
use std::time::Instant;

use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im::MeshIm;
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::region::{outer_faces_of_mesh, select_faces_of_normal};
use getfem::geom::regular_mesh::regular_unit_mesh;
use getfem::gwfl::workspace::Workspace;
use getfem::linalg::ColMatrix;

/// (dimension, subdivisions, degree), as opt_assembly's main
const CONFIGS: [(usize, usize, usize); 5] =
  [(2, 400, 1), (3, 36, 1), (2, 200, 2), (3, 18, 2), (3, 9, 4)];

/// the cases of opt_assembly's select set, then the forms of structured
/// contractions: (order, expression, integration of degree 2k-2, else 2k)
const CASES: [(usize, &str, bool); 10] = [
  (1, "u.Test_u", false),
  (1, "Det(Grad_u)", false),
  (2, "Test_p.Test2_p", false),
  (2, "(Test_u).Test2_u", false),
  (2, "Grad_Test_p:Grad_Test2_p", true),
  (
    2,
    "(Div_Test_u*(lambda*Id(qdim(u))) + (2*mu)*Sym(Grad_Test_u)):Grad_Test2_u",
    true,
  ),
  (
    2,
    "(Div_Test_u*(lambda2*Id(meshdim)) + (2*mu2)*Sym(Grad_Test_u)):Grad_Test2_u",
    true,
  ),
  (2, "Grad_Test_u:Grad_Test2_u", true),
  (2, "Sym(Grad_Test_u):Sym(Grad_Test2_u)", true),
  (
    2,
    "lambda*Div_Test_u*Div_Test2_u + (2*mu)*Sym(Grad_Test_u):Sym(Grad_Test2_u)",
    true,
  ),
];

struct Setup {
  ws: Workspace,
  mim: MeshIm,
  mim2: MeshIm,
  ndof: usize,
}

/// u = x + 0.1 bump at the DOF nodes (Grad_u near Id: Det(Grad_u) > 0)
fn displacement(mf: &MeshFem, n: usize) -> Vec<f64> {
  (0..mf.nb_dof().unwrap())
    .map(|d| {
      let x = mf.point_of_basic_dof(d).unwrap();
      let bump: f64 = x
        .iter()
        .map(|xi| (std::f64::consts::PI * xi).sin())
        .product();
      x[d % n] + 0.1 * bump
    })
    .collect()
}

/// the setup of opt_assembly's test_new_assembly (GT_PK(n,1), FEM_PK(n,k))
fn setup((n, nx, k): (usize, usize, usize)) -> Setup {
  let m = regular_unit_mesh(
    &vec![nx; n],
    &geotrans_descriptor(&format!("GT_PK({n},1)")).unwrap(),
  )
  .unwrap();
  let mut dir = vec![0.0; n];
  dir[n - 1] = 1.0;
  let border = outer_faces_of_mesh(&m);
  let neumann = select_faces_of_normal(&m, &border, &dir, 0.1).unwrap();
  m.set_region(1, &neumann).unwrap();
  m.set_region(2, &border.subtract(&neumann).unwrap())
    .unwrap();
  let mf_u = MeshFem::new(&m, n)
    .with_classical_finite_element(k)
    .unwrap();
  let mf_p = MeshFem::new(&m, 1)
    .with_classical_finite_element(k)
    .unwrap();
  let mim = MeshIm::new(&m).with_integration_method(2 * k).unwrap();
  let mim2 = MeshIm::new(&m).with_integration_method(2 * k - 2).unwrap();
  let seq =
    |len: usize, s: f64| -> Vec<f64> { (0..len).map(|i| (s * (i as f64 + 1.0)).sin()).collect() };
  let mut ws = Workspace::new();
  ws.add_fixed_size_data("a", &[1], &[3.0]).unwrap();
  ws.add_fixed_size_data("b", &[2], &[3.0, 6.0]).unwrap();
  ws.add_scalar_data("lambda", 3.0).unwrap();
  ws.add_scalar_data("mu", 2.0).unwrap();
  ws.add_fem_variable("u", &mf_u).unwrap();
  ws.set_variable("u", &displacement(&mf_u, n)).unwrap();
  ws.add_fem_variable("p", &mf_p).unwrap();
  let np = mf_p.nb_dof().unwrap();
  ws.set_variable("p", &seq(np, 1.1)).unwrap();
  ws.add_fem_data("lambda2", &mf_p, &vec![3.0; np]).unwrap();
  ws.add_fem_data("mu2", &mf_p, &vec![2.0; np]).unwrap();
  let chi = MeshFem::partial_on_region(&mf_p, 2).unwrap();
  ws.add_fem_variable("chi", &chi).unwrap();
  ws.set_variable("chi", &seq(chi.nb_dof().unwrap(), 0.9))
    .unwrap();
  let ndof = mf_u.nb_dof().unwrap() + np + chi.nb_dof().unwrap();
  Setup {
    ws,
    mim,
    mim2,
    ndof,
  }
}

enum Assembled<'a> {
  Vector(&'a [f64]),
  Matrix(&'a ColMatrix<f64>),
}

/// one assembly (a column matrix for matrices)
#[inline(never)]
fn measured(s: &mut Setup, c: usize) -> Assembled<'_> {
  body(s, c)
}

fn body(s: &mut Setup, c: usize) -> Assembled<'_> {
  let (order, expr, mim2) = CASES[c];
  let mim = if mim2 { &s.mim2 } else { &s.mim };
  if order == 1 {
    Assembled::Vector({
      s.ws.clear_expressions();
      s.ws.add_expression(expr, mim, -1).unwrap();
      s.ws.assembly(1, false).unwrap();
      s.ws.assembled_vector()
    })
  } else {
    Assembled::Matrix({
      s.ws.clear_expressions();
      s.ws.add_expression(expr, mim, -1).unwrap();
      s.ws.assembly(2, false).unwrap();
      s.ws.assembled_matrix()
    })
  }
}

/// sum of the entries weighted by sin(0.7 (i+1)) sin(0.3 (j+1))
fn checksum(r: &Assembled) -> f64 {
  let w = |i: usize, s: f64| (s * (i as f64 + 1.0)).sin();
  match r {
    Assembled::Vector(v) => v.iter().enumerate().map(|(i, x)| x * w(i, 0.7)).sum(),
    Assembled::Matrix(k) => k
      .cols
      .iter()
      .enumerate()
      .map(|(j, col)| col.iter().map(|&(i, x)| x * w(i, 0.7)).sum::<f64>() * w(j, 0.3))
      .sum(),
  }
}

fn main() {
  // cargo bench passes --bench
  let args: Vec<String> = std::env::args()
    .skip(1)
    .filter(|a| !a.starts_with("--"))
    .collect();
  // a negative index: all (as bench/opt_assembly.cc)
  let index = |a: &String| usize::try_from(a.parse::<i64>().unwrap()).ok();
  let only_cfg: Option<usize> = args.first().and_then(index);
  let only: Option<usize> = args.get(1).and_then(index);
  let repeats: usize = args.get(2).map_or(3, |a| a.parse().unwrap());
  let threads = std::env::var("BENCH_THREADS").map_or(1, |v| v.parse().unwrap());
  getfem::set_num_threads(threads).unwrap();
  println!("config,case,expr,ndof,min_s,median_s,checksum");
  for (ic, &cf) in CONFIGS.iter().enumerate() {
    if only_cfg.is_some_and(|o| o != ic) {
      continue;
    }
    let mut s = setup(cf);
    let name = format!("{}D_P{}_{}", cf.0, cf.2, cf.1);
    for (i, &(_, expr, _)) in CASES.iter().enumerate() {
      if only.is_some_and(|o| o != i) {
        continue;
      }
      // warm-up, not measured: layouts and caches built once per space
      // (not through a function identical to measured: the compiler would merge them)
      black_box(body(&mut s, i));
      // one case: one assembly, unless the repeats are given
      let reps = if only.is_some() && args.len() < 3 {
        1
      } else {
        repeats
      };
      let mut t = vec![];
      for _ in 0..reps {
        let t0 = Instant::now();
        // the result in the workspace's storage (reused)
        black_box(measured(&mut s, i));
        t.push(t0.elapsed().as_secs_f64());
      }
      t.sort_by(f64::total_cmp);
      let sum = checksum(&body(&mut s, i));
      println!(
        "{name},{i},\"{expr}\",{},{:.6},{:.6},{:.12e}",
        s.ndof,
        t[0],
        t[t.len() / 2],
        sum
      );
    }
  }
}
