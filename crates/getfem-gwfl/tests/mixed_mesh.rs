//! Meshes of mixed element types: 2D triangles and quadrilaterals, 3D
//! hexahedra, prisms, pyramids and tetrahedra (degree 1 and 2),
//! conforming; exactness of the assemblies on the unperturbed meshes.

mod common;

use std::collections::BTreeSet;

use getfem_fem::im_data::ImData;
use getfem_fem::mesh_fem::MeshFem;
use getfem_fem::mesh_im::MeshIm;
use getfem_geom::mesh::Mesh;
use getfem_geom::region::{RegionArg, inner_faces, outer_faces_of_mesh};
use getfem_gwfl::workspace::Workspace;

struct Setup {
  m: Mesh,
  ws: Workspace,
  mim: MeshIm,
  mfu: MeshFem,
}

/// a contraction to the center (the points stay inside the curved boundary)
fn shift(n: usize) -> &'static str {
  if n == 2 {
    "0.9*X+[0.05;0.05]"
  } else {
    "0.9*X+[0.05;0.05;0.05]"
  }
}

/// the setup of gwfl_mixed.cc on mixed mesh mi
fn setup(mi: usize, perturbed: bool) -> Setup {
  let m = common::mixed::mixed_mesh(mi, perturbed);
  let n = m.dim();
  // transformations of degree 1 + mi % 2
  let (ku, kv, im) = (2, 1 + mi % 2, if mi == 1 { 6 } else { 4 });
  let fem = |q: usize, k: usize| {
    MeshFem::new(&m, q)
      .with_classical_finite_element(k)
      .unwrap()
  };
  let (mfu, mfv, mfc, mfw) = (fem(1, ku), fem(n, kv), fem(1, 1), fem(1, 1));
  let kept: BTreeSet<usize> = (0..mfw.nb_basic_dof().unwrap())
    .filter(|d| d % 3 != 1)
    .collect();
  mfw.reduce_to_basic_dof(&kept).unwrap();
  let mim = MeshIm::new(&m).with_integration_method(im).unwrap();
  let imd = ImData::new(&mim, None, &[]).unwrap();
  let vals = |len: usize, f: &dyn Fn(f64) -> f64| (0..len).map(|i| f(i as f64)).collect::<Vec<_>>();
  let mut ws = Workspace::new();
  ws.add_fem_variable("u", &mfu).unwrap();
  ws.set_variable(
    "u",
    &vals(mfu.nb_dof().unwrap(), &|i| 0.1 * (i + 1.0).sin()),
  )
  .unwrap();
  ws.add_fem_variable("v", &mfv).unwrap();
  ws.set_variable(
    "v",
    &vals(mfv.nb_dof().unwrap(), &|i| 0.05 * (2.0 * i + 1.0).cos()),
  )
  .unwrap();
  ws.add_fem_variable("w", &mfw).unwrap();
  ws.set_variable(
    "w",
    &vals(mfw.nb_dof().unwrap(), &|i| 0.2 * (0.7 * i + 0.3).sin()),
  )
  .unwrap();
  ws.add_scalar_data("lambda", 2.0).unwrap();
  ws.add_scalar_data("mu", 3.0).unwrap();
  ws.add_fem_data(
    "c",
    &mfc,
    &vals(mfc.nb_dof().unwrap(), &|i| 1.0 + 0.5 * i.cos()),
  )
  .unwrap();
  ws.add_im_data(
    "d",
    &imd,
    &vals(imd.nb_values().unwrap(), &|i| {
      1.0 + 0.1 * (i as usize % 7) as f64
    }),
  )
  .unwrap();
  ws.add_interpolate_transformation_from_expression("shift", &m, &m, shift(n))
    .unwrap();
  Setup { m, ws, mim, mfu }
}

fn region(m: &Mesh, name: &str) -> RegionArg {
  match name {
    "boundary" => outer_faces_of_mesh(m).into(),
    "inner" => inner_faces(m, -1).unwrap().into(),
    _ => (-1).into(),
  }
}

/// the scalar of expr on region rg of s
fn integral(s: &mut Setup, expr: &str, rg: &str) -> f64 {
  let rg = region(&s.m, rg);
  s.ws.clear_expressions();
  s.ws.add_expression(expr, &s.mim, rg).unwrap();
  s.ws.assembly(0, false).unwrap();
  s.ws.assembled_potential()
}

#[test]
fn exactness() {
  for mi in 0..4 {
    let mut s = setup(mi, false);
    let n = s.m.dim();
    let check = |what: &str, got: f64, want: f64| {
      assert!(
        (got - want).abs() < 1e-12,
        "mesh {mi} {what}: {got:e} vs {want:e}"
      );
    };
    // the geometry
    check("volume", integral(&mut s, "1", "all"), 1.0);
    check("x^2", integral(&mut s, "X(1)*X(1)", "all"), 1.0 / 3.0);
    check(
      "x y",
      integral(&mut s, &format!("X(1)*X({n})"), "all"),
      0.25,
    );
    check(
      "X.Normal",
      integral(&mut s, "X.Normal", "boundary"),
      n as f64,
    );
    // the inner faces: the same points seen from the neighbor, its normal
    // opposite; continuous fields (conforming elements across the types)
    check(
      "neighbor X",
      integral(
        &mut s,
        "Norm_sqr(Interpolate(X,neighbor_element) - X)",
        "inner",
      ),
      0.0,
    );
    check(
      "neighbor Normal",
      integral(
        &mut s,
        "Interpolate(Normal,neighbor_element).Normal + 1",
        "inner",
      ),
      0.0,
    );
    for f in ["u", "v", "w"] {
      let jump = integral(
        &mut s,
        &format!("Norm_sqr({f} - Interpolate({f},neighbor_element))"),
        "inner",
      );
      check(&format!("jump of {f}"), jump, 0.0);
    }
    // a quadratic reproduced by u (values, gradients), at the points too
    let (p, gp) = match n {
      2 => (
        "1 + X(1) - 2*X(2) + X(1)*X(2) - 0.5*X(2)*X(2)",
        "[1 + X(2); -2 + X(1) - X(2)]",
      ),
      _ => (
        "1 + X(1) - 2*X(2) + X(1)*X(3) - 0.5*X(2)*X(2) + X(3)*X(3)",
        "[1 + X(3); -2 - X(2); X(1) + 2*X(3)]",
      ),
    };
    let u = s.ws.interpolation(p, &s.mfu, -1).unwrap();
    s.ws.set_variable("u", &u).unwrap();
    check(
      "u - p",
      integral(&mut s, &format!("sqr(u - ({p}))"), "all"),
      0.0,
    );
    check(
      "Grad_u - Grad p",
      integral(&mut s, &format!("Norm_sqr(Grad_u - {gp})"), "all"),
      0.0,
    );
    let imd = ImData::new(&s.mim, None, &[n]).unwrap();
    let g = s
      .ws
      .interpolation(&format!("Grad_u - {gp}"), &imd, -1)
      .unwrap();
    // pyramids: at the points next to the apex the rational basis loses
    // digits (degree 1: GetFEM 2.6e-9, GetFEM-rs 2.1e-9)
    let worst = g.iter().fold(0.0f64, |a, x| a.max(x.abs()));
    let tol = if n == 3 { 1e-8 } else { 1e-12 };
    assert!(
      worst < tol,
      "mesh {mi} Grad_u - Grad p at the points: {worst:e}"
    );
    // p at points inside the elements of each type (point location, inversion)
    let pts: Vec<Vec<f64>> = (0..23)
      .map(|k| {
        (0..n)
          .map(|d| 0.5 + 0.45 * (1.3 * k as f64 + 2.7 * d as f64).sin())
          .collect()
      })
      .collect();
    let at = s.ws.interpolation("u", (&pts, &s.m), -1).unwrap();
    let want = s.ws.interpolation(p, (&pts, &s.m), -1).unwrap();
    let worst = at
      .iter()
      .zip(&want)
      .fold(0.0f64, |a, (x, y)| a.max((x - y).abs()));
    check("u at points", worst, 0.0);
    // the local projection on discontinuous elements: p itself
    let mfd = MeshFem::new(&s.m, 1)
      .with_classical_discontinuous_finite_element(2)
      .unwrap();
    let pd = s.ws.local_projection(p, &s.mim, &mfd, -1).unwrap();
    let id = s.ws.interpolation(p, &mfd, -1).unwrap();
    let worst = pd
      .iter()
      .zip(&id)
      .fold(0.0f64, |a, (x, y)| a.max((x - y).abs()));
    check("local projection", worst, 0.0);
    // shift found to round-off, also on the perturbed meshes
    let mut s = setup(mi, true);
    check(
      "shift",
      integral(
        &mut s,
        &format!("Norm_sqr(Interpolate(X,shift) - ({}))", shift(n)),
        "all",
      ),
      0.0,
    );
  }
}
