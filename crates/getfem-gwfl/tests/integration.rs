//! Port of GetFEM's tests/integration.cc on GWFL: the mass matrix of
//! P(k/2) x P(k-k/2) elements on the reference convex and on each face,
//! computed with every method exact to degree k, equals the one computed
//! with a classical rule of degree k + 2 (or the highest available >= k).
//! Structured composite and quasi-polar rules are not in the catalogue yet.

use getfem_fem::fem::classical_fem;
use getfem_fem::integration::{PIntegration, classical_approx_im, integ_descriptor};
use getfem_fem::mesh_fem::MeshFem;
use getfem_fem::mesh_im::MeshIm;
use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::mesh::{Mesh, MeshData};
use getfem_geom::region::{Region, RegionSet};
use getfem_gwfl::workspace::Workspace;

/// mass matrices (element, then faces) of one method
fn masses(gt: &str, k: usize, pim: &PIntegration) -> Vec<Vec<f64>> {
  let pgt = geotrans_descriptor(gt).unwrap();
  let mut m = MeshData::new(pgt.dim());
  let nodes: Vec<&[f64]> = pgt.geometric_nodes().iter().map(|p| p.as_slice()).collect();
  m.add_convex_by_points(&pgt, &nodes).unwrap();
  let nf = pgt.structure().nb_faces();
  let m = Mesh::from(m);
  let fem = |deg| {
    let mf = MeshFem::new(&m, 1);
    mf.set_finite_element(([0], &classical_fem(&pgt, deg, false).unwrap()))
      .unwrap();
    mf
  };
  let mut ws = Workspace::new();
  ws.add_fem_variable("a", &fem(k / 2)).unwrap();
  ws.add_fem_variable("b", &fem(k - k / 2)).unwrap();
  let mim = MeshIm::new(&m);
  mim.set_integration_method(([0], pim)).unwrap();
  let mut dense = |rg: Option<&Region>| {
    let c = {
      ws.clear_expressions();
      ws.add_expression("Test_a*Test2_b", &mim, rg).unwrap();
      ws.assembly(2, false).unwrap();
      ws.assembled_matrix()
    }
    .to_csr();
    let n = ws.nb_dof();
    (0..n * n)
      .map(|x| c.get(x / n, x % n))
      .collect::<Vec<f64>>()
  };
  let mut out = vec![dense(None)];
  for f in 0..nf {
    let mut r = RegionSet::new();
    r.add_face(0, f);
    let r = Region::temporary(&m, r).unwrap();
    out.push(dense(Some(&r)));
  }
  out
}

#[test]
fn methods_agree() {
  let mut cases: Vec<(String, usize, String)> = vec![];
  let mut add = |im: String, k: usize, gt: &str| cases.push((im, k, gt.to_string()));
  for k in 0..15 {
    add(format!("IM_GAUSS1D({k})"), k, "GT_PK(1,1)");
    add(format!("IM_NC(1,{k})"), k, "GT_PK(1,1)");
  }
  for d in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 17, 19] {
    for k in 1..=d {
      add(format!("IM_TRIANGLE({d})"), k, "GT_PK(2,1)");
    }
  }
  for d in 2..5 {
    for i in 1..8 {
      for k in 0..i.min(5 - d) {
        add(format!("IM_NC({d},{i})"), k, &format!("GT_PK({d},1)"));
      }
    }
  }
  for d in [1, 2, 3, 5, 6, 8] {
    for k in 1..=d {
      add(format!("IM_TETRAHEDRON({d})"), k, "GT_PK(3,1)");
    }
  }
  for k in 1..=3 {
    add("IM_SIMPLEX4D(3)".into(), k, "GT_PK(4,1)");
  }
  for (d, kmax) in [(1, 10), (2, 10), (3, 6), (4, 3)] {
    let gt = format!("GT_QK({d},1)");
    for k in 0..kmax {
      add(format!("IM_GAUSS_PARALLELEPIPED({d},{k})"), k, &gt);
      add(format!("IM_NC_PARALLELEPIPED({d},{k})"), k, &gt);
      if d > 1 {
        add(
          format!(
            "IM_PRODUCT(IM_GAUSS_PARALLELEPIPED({},{k}),IM_NC(1,{k}))",
            d - 1
          ),
          k,
          &gt,
        );
      }
    }
  }
  for d in [2, 3, 5, 7, 9, 17] {
    for k in 1..=((d as f64).sqrt() as usize) {
      add(format!("IM_QUAD({d})"), k, "GT_QK(2,1)");
    }
  }
  for d in [5, 9, 11] {
    add(
      format!("IM_HEXAHEDRON({d})"),
      (d as f64).cbrt() as usize,
      "GT_QK(3,1)",
    );
  }
  for d in [5, 9] {
    add(format!("IM_CUBE4D({d})"), 1, "GT_QK(4,1)");
  }
  for d in 2..5 {
    for k in 0..7 - d {
      let gt = format!("GT_PRISM({d},{})", k.max(1));
      add(format!("IM_NC_PRISM({d},{k})"), k, &gt);
      if d == 3 {
        add("IM_PRODUCT(IM_TRIANGLE(6),IM_GAUSS1D(6))".into(), k, &gt);
      }
    }
  }
  let (mut checked, mut missing) = (0, vec![]);
  for (im, k, gt) in &cases {
    let pim = match integ_descriptor(im) {
      Ok(p) => p,
      Err(_) => {
        missing.push(im.clone());
        continue;
      }
    };
    // degree k + 2 where the catalogue has it, else the highest down to k
    let pgt = geotrans_descriptor(gt).unwrap();
    let Some(reference) = [k + 2, k + 1, *k]
      .iter()
      .find_map(|&d| classical_approx_im(&pgt, d).ok())
    else {
      missing.push(format!("reference for {gt} degree {k}"));
      continue;
    };
    let (a, r) = (masses(gt, *k, &pim), masses(gt, *k, &reference));
    for (f, (x, y)) in a.iter().zip(&r).enumerate() {
      let scale = y.iter().fold(0.0f64, |s, v| s.max(v.abs()));
      let err = x
        .iter()
        .zip(y)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, f64::max);
      let what = if f == 0 {
        "element".to_string()
      } else {
        format!("face {}", f - 1)
      };
      // high-degree Newton-Cotes: GetFEM also differs by 1.0e-7 on
      // IM_NC(1,14) (its own criterion is 1e-5)
      let tol = if im.contains("IM_NC") { 1e-6 } else { 1e-9 };
      assert!(
        err <= tol * scale,
        "{im} on {gt}, k = {k}, {what}: {err:e} (scale {scale:e})"
      );
    }
    checked += 1;
  }
  missing.sort();
  missing.dedup();
  println!("{checked} method/degree pairs checked; not available: {missing:?}");
  assert!(missing.is_empty(), "missing methods: {missing:?}");
}
