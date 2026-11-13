//! Assembly benchmark: the cases of GetFEM's tests/test_assembly.cc on larger
//! meshes (crates/getfem-tests/generators/src/bench_assembly.cc runs the same with GetFEM).
//!
//!   cargo bench -p getfem --bench assembly [-- <dim> [case] [repeats]]
//!
//! prints one CSV line per case: dim,case,kind,expr,min_s,median_s. With a
//! case index the case runs alone, once unless the repeats are given (for
//! callgrind: the assembly is the function `measured`,
//! --toggle-collect='*measured*'). BENCH_SCALE=k divides the mesh size by k
//! (by at most 2 in 3D); BENCH_GROW=g (a float, for threads) multiplies the
//! number of elements of every mesh by g^dim; BENCH_THREADS=t assembles with
//! t threads (default 1). The boundary cases (9-12) run on a flat mesh (one
//! layer, many faces), not on the cube's few faces; the cheapest volume cases
//! (0, 6, 7) on a finer cube (all cases above 5 ms). The last cases repeat
//! some of the others on a curved isoparametric mesh (GT_PK(n,2), its nodes
//! moved by curve()). After them, GetFEM-rs only (threads): the
//! neighbor_element and an expression transformation, a reduced space,
//! interpolation on a space and on an im_data, condensation (on the cube,
//! extras() added at the first of them). BENCH_MIXED=1: every mesh of mixed
//! element types (tests' mixed_unit_mesh, the same cells, the
//! transformations of degree 1 or 2).

use std::hint::black_box;
use std::time::Instant;

use getfem::fem::im_data::ImData;
use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im::MeshIm;
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::region::{inner_faces, outer_faces_of_mesh, select_faces_of_normal};
use getfem::geom::regular_mesh::regular_unit_mesh;
use getfem::gwfl::workspace::Workspace;

#[allow(dead_code)]
#[path = "../../getfem-gwfl/tests/common/mixed.rs"]
mod mixed;

#[derive(Clone, Copy)]
enum Kind {
  Scalar,
  Vector,
  Matrix,
  /// interpolation on mf_a, on imd
  InterpFem,
  InterpImd,
  /// the matrix with q condensed, the terms separated by ';'
  Condensed,
}

struct Case {
  kind: Kind,
  expr: String,
  /// 2: mim of degree 2, else degree 4
  mim2: bool,
  region: i64,
  /// on the curved mesh
  curved: bool,
  /// on the cube with extras()
  extra: bool,
}

fn cases(n: usize) -> Vec<Case> {
  let c = |kind, expr: &str, mim2, region| Case {
    kind,
    expr: expr.to_string(),
    mim2,
    region,
    curved: false,
    extra: false,
  };
  use Kind::*;
  let mut v = vec![
    c(Scalar, "1", false, -1),
    c(Scalar, "cos(pi*X(1))", false, -1),
    c(Scalar, "cos(pi*X).exp(X*0)", false, -1),
    c(Scalar, "sin(pi*X).exp(X*0)", false, -1),
    c(Scalar, "u.u", false, -1),
    c(Scalar, "Norm_sqr(u)", false, -1),
    c(Scalar, "Grad_u:Grad_u", true, -1),
    c(Scalar, "Id(meshdim)*Grad_u:Grad_u", true, -1),
    c(Vector, "u.Test_u", false, -1),
    c(Vector, "u.Test_u", false, 1),
    c(
      Vector,
      "(((Reshape(A,meshdim,meshdim))')*Normal).Test_u",
      false,
      1,
    ),
    c(Vector, "(A'*Normal).Test_u", false, 1),
    c(Vector, "p*Test_chi", false, 2),
    c(Matrix, "Test_u.Test2_u", false, -1),
    c(Matrix, "Grad_Test_p:Grad_Test2_p", true, -1),
    c(Matrix, "(Grad_p:Grad_p)/2", true, -1),
    c(Matrix, "sqr(Norm(Grad_p))/2", true, -1),
    c(Matrix, "Norm_sqr(Grad_p)/2", true, -1),
    c(
      Matrix,
      "(lambda*Trace(Grad_Test_u)*Id(qdim(u)) + mu*(Grad_Test_u'+Grad_Test_u)):Grad_Test2_u",
      true,
      -1,
    ),
    c(
      Matrix,
      "lambda*Div_Test_u*Div_Test2_u + mu*(Grad_Test_u'+Grad_Test_u):Grad_Test2_u",
      true,
      -1,
    ),
    c(
      Matrix,
      "lambda*((Grad_Test2_u@Grad_Test_u):Id(meshdim)):Id(meshdim) + mu*(Grad_Test_u'+Grad_Test_u):Grad_Test2_u",
      true,
      -1,
    ),
    c(
      Matrix,
      "lambda*Id(meshdim)@Id(meshdim)*Grad_Test_u:Grad_Test2_u + mu*(Grad_Test_u'+Grad_Test_u):Grad_Test2_u",
      true,
      -1,
    ),
    c(
      Matrix,
      "(lambda2*Trace(Grad_Test_u)*Id(meshdim) + mu2*(Grad_Test_u'+Grad_Test_u)):Grad_Test2_u",
      true,
      -1,
    ),
  ];
  if n == 2 {
    v.push(c(
      Scalar,
      "Grad_u(1,:).Grad_u(1,:) + Grad_u(2,:).Grad_u(2,:)",
      true,
      -1,
    ));
    v.push(c(
      Matrix,
      "(Grad_p(1)*Grad_p(1) + Grad_p(2)*Grad_p(2))/2",
      true,
      -1,
    ));
  }
  // Ogden (GetFEM snippet gf_ogden_3D): the two forms of the stress term,
  // residual and tangent (macros in setup)
  for kind in [Vector, Matrix] {
    v.push(c(
      kind,
      "((tauH*Id(meshdim)+tauD)*Inv(F')):Grad(Test_u)",
      false,
      -1,
    ));
    v.push(c(
      kind,
      "(tauH*Id(meshdim)+tauD):(Grad(Test_u)*Inv(F))",
      false,
      -1,
    ));
  }
  // on the curved mesh: a scalar (on a finer mesh), a vector, two matrices,
  // elasticity, the first Ogden form (residual and tangent)
  let o = v.len() - 4;
  for i in [6, 8, 13, 14, 18, o, o + 2] {
    let mut c = c(v[i].kind, &v[i].expr, v[i].mim2, v[i].region);
    c.curved = true;
    v.push(c);
  }
  // GetFEM-rs only: region 3 the inner faces; shift a transformation; chi
  // a reduced space; q internal (condensation last: the assembled vector
  // given)
  let jump = |g: &str| format!("({g}-Interpolate({g},neighbor_element)).Normal");
  for mut c in [
    c(
      Vector,
      &format!("{}*Grad_Test_p.Normal", jump("Grad_p")),
      false,
      3,
    ),
    c(
      Matrix,
      &format!("{}*{}", jump("Grad_Test_p"), jump("Grad_Test2_p")),
      false,
      3,
    ),
    c(Vector, "Interpolate(p,shift)*Test_p", false, -1),
    c(Matrix, "Interpolate(Test_p,shift)*Test2_p", false, -1),
    c(Matrix, "Grad_Test_chi.Grad_Test2_chi", false, -1),
    c(InterpFem, "Grad_u", false, -1),
    c(InterpImd, "Grad_u", false, -1),
    c(
      Condensed,
      "(-q*Id(meshdim)+2*mu*Sym(Grad_u)):Grad_Test_u;(q+1e3*pow(q,3)+lambda*Div_u)*Test_q",
      false,
      -1,
    ),
    // a compressible neo-Hookean law (a user's topology optimization of a
    // column): its tangent, and the tangent of the derivative of its
    // second variation in the direction w (third and fourth derivatives)
    c(Matrix, "(tauc*Inv(Fc')):Grad_Test_u", false, -1),
    c(Matrix, "Diff(PsiW,u,Test_u)", false, -1),
  ] {
    c.extra = true;
    v.push(c);
  }
  v
}

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

/// Ogden macros (u from displacement()), K = lambda, alpha = 2
const OGDEN: [(&str, &str); 4] = [
  ("F", "Id(meshdim)+Grad_u"),
  ("J", "Det(F)"),
  ("tauH", "lambda*log(J)"),
  ("tauD", "mu*Deviator(Expm(Deviator(Logm(F*F'))))"),
];

/// a compressible neo-Hookean law (plane strain in 2D), K = lambda, G = mu,
/// and its second variation in the direction w (PsiW = d2W[w, w], written
/// out as in the user's script)
const NEO_HOOKE: [(&str, &str); 5] = [
  ("Fc", "Id(meshdim)+Grad_u"),
  ("Dev33(A)", "A-Id(meshdim)/3*(Trace(A)+3-meshdim)"),
  (
    "tauc",
    "lambda*log(Det(Fc))*Id(meshdim)+mu*pow(Det(Fc),-2/3)*Dev33(Fc*Fc')",
  ),
  (
    "dPw",
    "(lambda*(Inv(Fc'):Grad_w)*Id(meshdim)+mu*pow(Det(Fc),-2/3)*(-2/3*(Inv(Fc'):Grad_w)*Dev33(Fc*Fc')\
     +Grad_w*Fc'+Fc*Grad_w'-Id(meshdim)/3*Trace(Grad_w*Fc'+Fc*Grad_w')))*Inv(Fc')\
     -tauc*Inv(Fc')*Grad_w'*Inv(Fc')",
  ),
  ("PsiW", "dPw:Grad_w"),
];

struct Setup {
  ws: Workspace,
  mim: MeshIm,
  mim2: MeshIm,
  mf_a: MeshFem,
  /// by extras(): the im_data of Grad_u, the residual of condensation
  imd: Option<ImData>,
  residual: Vec<f64>,
}

/// the GetFEM-rs only cases' additions to a setup (once)
fn extras(s: &mut Setup) {
  if s.imd.is_some() {
    return;
  }
  let m = s.mim.linked_mesh();
  let n = m.dim();
  m.set_region(3, &inner_faces(&m, -1).unwrap()).unwrap();
  s.ws
    .add_interpolate_transformation_from_expression("shift", &m, &m, "0.9*X")
    .unwrap();
  let q = ImData::new(&s.mim, -1, &[]).unwrap();
  s.ws.add_internal_im_variable("q", &q).unwrap();
  let nq = q.nb_values().unwrap();
  s.ws
    .set_variable(
      "q",
      &(0..nq)
        .map(|i| 0.01 * (0.7 * i as f64).cos())
        .collect::<Vec<_>>(),
    )
    .unwrap();
  s.residual = (0..s.ws.nb_dof()).map(|i| (0.9 * i as f64).sin()).collect();
  s.imd = Some(ImData::new(&s.mim, -1, &[n, n]).unwrap());
}

/// u of the setup: a homogeneous stretch and shear and a smooth bump, at
/// the DOF nodes (log strains up to 0.45, J in [0.9, 1.5]: a large strain
/// state for Ogden)
fn displacement(mf: &MeshFem, n: usize) -> Vec<f64> {
  let g: &[f64] = if n == 2 {
    &[0.3, 0.2, 0.1, -0.15]
  } else {
    &[0.3, 0.2, 0.0, 0.1, -0.15, 0.1, 0.0, 0.05, 0.2]
  };
  (0..mf.nb_dof().unwrap())
    .map(|d| {
      let x = mf.point_of_basic_dof(d).unwrap();
      let c = d % n;
      let bump: f64 = x
        .iter()
        .map(|xi| (std::f64::consts::PI * xi).sin())
        .product();
      (0..n).map(|j| g[c * n + j] * x[j]).sum::<f64>() + 0.05 * bump
    })
    .collect()
}

/// the setup of test_assembly.cc (GT_PK(n,1), FEM_PK(n,2), random data
/// replaced by fixed sequences, u by displacement()); curved: GT_PK(n,2),
/// the nodes moved by curve(); mixed: of mixed element types
fn setup(sub: &[usize], curved: bool) -> Setup {
  let n = sub.len();
  let k = if curved { 2 } else { 1 };
  let m = if std::env::var("BENCH_MIXED").is_ok_and(|v| v == "1") {
    mixed::mixed_unit_mesh(sub, k)
  } else {
    regular_unit_mesh(
      sub,
      &geotrans_descriptor(&format!("GT_PK({n},{k})")).unwrap(),
    )
    .unwrap()
  };
  if curved {
    curve(&m);
  }
  let mut dir = vec![0.0; n];
  dir[n - 1] = 1.0;
  let border = outer_faces_of_mesh(&m);
  let neumann = select_faces_of_normal(&m, &border, &dir, 0.1).unwrap();
  m.set_region(1, &neumann).unwrap();
  m.set_region(2, &border.subtract(&neumann).unwrap())
    .unwrap();
  let mf_u = MeshFem::new(&m, n)
    .with_classical_finite_element(2)
    .unwrap();
  let mf_p = MeshFem::new(&m, 1)
    .with_classical_finite_element(2)
    .unwrap();
  let mf_a = MeshFem::new(&m, 1);
  mf_a.set_qdim([n, n]);
  mf_a.set_classical_finite_element(2).unwrap();
  let mim = MeshIm::new(&m).with_integration_method(4).unwrap();
  let mim2 = MeshIm::new(&m).with_integration_method(2).unwrap();
  let seq =
    |len: usize, s: f64| -> Vec<f64> { (0..len).map(|i| (s * (i as f64 + 1.0)).sin()).collect() };
  let mut ws = Workspace::new();
  ws.add_fixed_size_data("a", &[1], &[3.0]).unwrap();
  ws.add_fixed_size_data("b", &[2], &[3.0, 6.0]).unwrap();
  ws.add_scalar_data("lambda", 3.0).unwrap();
  ws.add_scalar_data("mu", 2.0).unwrap();
  ws.add_fem_variable("u", &mf_u).unwrap();
  ws.set_variable("u", &displacement(&mf_u, n)).unwrap();
  ws.add_fem_data("A", &mf_a, &seq(mf_a.nb_dof().unwrap(), 0.3))
    .unwrap();
  ws.add_fem_variable("p", &mf_p).unwrap();
  ws.set_variable("p", &seq(mf_p.nb_dof().unwrap(), 1.1))
    .unwrap();
  let np = mf_p.nb_dof().unwrap();
  ws.add_fem_data("lambda2", &mf_p, &vec![3.0; np]).unwrap();
  ws.add_fem_data("mu2", &mf_p, &vec![2.0; np]).unwrap();
  let chi = MeshFem::partial_on_region(&mf_p, 2).unwrap();
  ws.add_fem_variable("chi", &chi).unwrap();
  ws.set_variable("chi", &seq(chi.nb_dof().unwrap(), 0.9))
    .unwrap();
  for (name, e) in OGDEN.iter().chain(&NEO_HOOKE) {
    ws.add_macro(name, e).unwrap();
  }
  ws.add_fem_data("w", &mf_u, &seq(mf_u.nb_dof().unwrap(), 0.13))
    .unwrap();
  Setup {
    ws,
    mim,
    mim2,
    mf_a,
    imd: None,
    residual: vec![],
  }
}

/// one assembly, the result as GetFEM gives it (a column matrix for matrices)
#[inline(never)]
fn measured(s: &mut Setup, c: &Case) -> f64 {
  body(s, c)
}

fn body(s: &mut Setup, c: &Case) -> f64 {
  let mim = if c.mim2 { &s.mim2 } else { &s.mim };
  let order = match c.kind {
    Kind::Scalar => 0,
    Kind::Vector => 1,
    Kind::Matrix => 2,
    Kind::InterpFem => return s.ws.interpolation(&c.expr, &s.mf_a, -1).unwrap()[0],
    Kind::InterpImd => {
      return s
        .ws
        .interpolation(&c.expr, s.imd.as_ref().unwrap(), -1)
        .unwrap()[0];
    }
    Kind::Condensed => {
      s.ws.clear_expressions();
      for t in c.expr.split(';') {
        s.ws.add_expression(t, mim, c.region).unwrap();
      }
      s.ws.set_assembled_vector(s.residual.clone());
      s.ws.assembly(2, true).unwrap();
      return s.ws.assembled_matrix().nrows as f64;
    }
  };
  s.ws.clear_expressions();
  s.ws.add_expression(&c.expr, mim, c.region).unwrap();
  s.ws.assembly(order, false).unwrap();
  match c.kind {
    Kind::Scalar => s.ws.assembled_potential(),
    Kind::Vector => s.ws.assembled_vector()[0],
    // the assembled column matrix, as GetFEM's (no conversion)
    _ => s.ws.assembled_matrix().nrows as f64,
  }
}

fn main() {
  // cargo bench passes --bench
  let args: Vec<String> = std::env::args()
    .skip(1)
    .filter(|a| !a.starts_with("--"))
    .collect();
  let dims: Vec<usize> = match args.first() {
    Some(d) => vec![d.parse().unwrap()],
    None => vec![2, 3],
  };
  let only: Option<usize> = args.get(1).map(|a| a.parse().unwrap());
  let repeats: usize = args.get(2).map_or(5, |a| a.parse().unwrap());
  let threads = std::env::var("BENCH_THREADS").map_or(1, |v| v.parse().unwrap());
  getfem::set_num_threads(threads).unwrap();
  println!("dim,case,kind,expr,min_s,median_s");
  for n in dims {
    // BENCH_SCALE (default 1) divides the mesh size, for callgrind
    let k: usize = std::env::var("BENCH_SCALE").map_or(1, |v| v.parse().unwrap());
    let g: f64 = std::env::var("BENCH_GROW").map_or(1.0, |v| v.parse().unwrap());
    let grow = |s: usize, g: f64| (s as f64 * g).round() as usize;
    let nx = grow(if n == 2 { 100 / k } else { 14 / k.min(2) }, g);
    // the boundary cases: nb x .. x 1 (2D: 15000 faces on top, 3D: 7200)
    let nb = grow(
      if n == 2 { 15000 / k } else { 60 / k.min(2) },
      g.powf(n as f64 / (n - 1) as f64),
    );
    // the cheapest volume cases: a finer cube
    let nf = grow(if n == 2 { 200 / k } else { 24 / k.min(2) }, g);
    let (mut cube, mut flat, mut fine) = (None, None, None);
    let (mut curved, mut curved_fine) = (None, None);
    for (i, c) in cases(n).iter().enumerate() {
      if only.is_some_and(|o| o != i) {
        continue;
      }
      let s = if c.curved && matches!(c.kind, Kind::Scalar) {
        curved_fine.get_or_insert_with(|| setup(&vec![nf; n], true))
      } else if c.curved {
        curved.get_or_insert_with(|| setup(&vec![nx; n], true))
      } else if (9..=12).contains(&i) {
        let mut sub = vec![nb; n];
        sub[n - 1] = 1;
        flat.get_or_insert_with(|| setup(&sub, false))
      } else if [0, 6, 7].contains(&i) {
        fine.get_or_insert_with(|| setup(&vec![nf; n], false))
      } else {
        cube.get_or_insert_with(|| setup(&vec![nx; n], false))
      };
      if c.extra {
        extras(s);
      }
      // warm-up, not measured: layouts and caches built once per space
      // (not through a function identical to measured: the compiler would merge them)
      black_box(body(s, c));
      // one case: one assembly, unless the repeats are given (perf stat
      // differences between two counts remove the setup)
      let reps = if only.is_some() && args.len() < 3 {
        1
      } else {
        repeats
      };
      let mut t = vec![];
      for _ in 0..reps {
        let t0 = Instant::now();
        black_box(measured(s, c));
        t.push(t0.elapsed().as_secs_f64());
      }
      t.sort_by(f64::total_cmp);
      let kind = [
        "scalar",
        "vector",
        "matrix",
        "interp_fem",
        "interp_imd",
        "condensed",
      ][c.kind as usize];
      println!(
        "{n},{i},{kind},\"{}\",{:.6},{:.6}",
        c.expr,
        t[0],
        t[t.len() / 2]
      );
      // the finer curved mesh serves one case: freed (the later cases on a
      // heap without it, as before it was added)
      if c.curved && matches!(c.kind, Kind::Scalar) {
        curved_fine = None;
      }
    }
  }
}
