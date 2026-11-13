//! Patch tests of the element families (M9 exit): the L2 projection of a
//! polynomial of the complete degree of an element onto its space on a
//! small mesh is the polynomial itself (values and gradients), for
//! Lagrange, special, hierarchical, transformed, composite and vector
//! elements. Simplex meshes are distorted, the others affine (sheared).

use getfem::fem::fem::fem_descriptor;
use getfem::fem::integration::integ_descriptor;
use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im::MeshIm;
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::mesh::{Mesh, MeshData};
use getfem::geom::regular_mesh::regular_unit_mesh;
use getfem::linalg::tensor::Mat;
use getfem::model::{Model, Newton};

/// element, mesh, degree reproduced, vector element, integration
type Case = (&'static str, &'static str, usize, bool, &'static str);

const TRI: &str = "GT_PK(2,1)";
const QUAD: &str = "GT_QK(2,1)";
const TET: &str = "GT_PK(3,1)";
const HEX: &str = "GT_QK(3,1)";
const PRISM: &str = "GT_PRISM(3,1)";
const PYR: &str = "GT_PYRAMID(1)";
const SEG: &str = "GT_PK(1,1)";
const T10: &str = "IM_TRIANGLE(10)";
const HCTI: &str = "IM_HCT_COMPOSITE(IM_TRIANGLE(10))";
const Q10: &str = "IM_GAUSS_PARALLELEPIPED(2,10)";
const QC1I: &str = "IM_QUADC1_COMPOSITE(IM_TRIANGLE(10))";
const TET8: &str = "IM_TETRAHEDRON(8)";
const H8: &str = "IM_GAUSS_PARALLELEPIPED(3,8)";
const PR8: &str = "IM_PRODUCT(IM_TRIANGLE(8),IM_GAUSS1D(8))";
const PY8: &str = "IM_PYRAMID(IM_GAUSS_PARALLELEPIPED(3,8))";
const G10: &str = "IM_GAUSS1D(10)";

const CASES: &[Case] = &[
  // 1D
  ("FEM_PK(1,3)", SEG, 3, false, G10),
  ("FEM_HERMITE(1)", SEG, 3, false, G10),
  ("FEM_PK_HIERARCHICAL(1,4)", SEG, 4, false, G10),
  ("FEM_PK_GAUSSLOBATTO1D(4)", SEG, 4, false, G10),
  // triangles
  ("FEM_PK(2,1)", TRI, 1, false, T10),
  ("FEM_PK(2,3)", TRI, 3, false, T10),
  ("FEM_PK_DISCONTINUOUS(2,2,0.1)", TRI, 2, false, T10),
  ("FEM_SIMPLEX_IPK(2,2)", TRI, 2, false, T10),
  ("FEM_SIMPLEX_CIPK(2,2)", TRI, 2, false, T10),
  ("FEM_PK_WITH_CUBIC_BUBBLE(2,2)", TRI, 2, false, T10),
  ("FEM_P1_NONCONFORMING", TRI, 1, false, T10),
  ("FEM_P1_BUBBLE_FACE(2)", TRI, 1, false, T10),
  ("FEM_P1_BUBBLE_FACE_LAG", TRI, 1, false, T10),
  ("FEM_PK_HIERARCHICAL(2,3)", TRI, 3, false, T10),
  (
    "FEM_GEN_HIERARCHICAL(FEM_PK(2,1),FEM_PK(2,3))",
    TRI,
    3,
    false,
    T10,
  ),
  ("FEM_HERMITE(2)", TRI, 3, false, T10),
  ("FEM_ARGYRIS", TRI, 5, false, T10),
  ("FEM_MORLEY(2)", TRI, 2, false, T10),
  ("FEM_HCT_TRIANGLE", TRI, 3, false, HCTI),
  ("FEM_REDUCED_HCT_TRIANGLE", TRI, 2, false, HCTI),
  ("FEM_P1_PIECEWISE_LINEAR_BUBBLE", TRI, 1, false, HCTI),
  (
    "FEM_STRUCTURED_COMPOSITE(FEM_PK(2,2),2)",
    TRI,
    2,
    false,
    "IM_STRUCTURED_COMPOSITE(IM_TRIANGLE(6),2)",
  ),
  (
    "FEM_PK_HIERARCHICAL_COMPOSITE(2,2,2)",
    TRI,
    2,
    false,
    "IM_STRUCTURED_COMPOSITE(IM_TRIANGLE(6),2)",
  ),
  (
    "FEM_PK_FULL_HIERARCHICAL_COMPOSITE(2,2,2)",
    TRI,
    2,
    false,
    "IM_STRUCTURED_COMPOSITE(IM_TRIANGLE(6),2)",
  ),
  ("FEM_RT0(2)", TRI, 0, true, T10),
  ("FEM_RTK(2,1)", TRI, 1, true, T10),
  ("FEM_BDMK(2,2)", TRI, 2, true, T10),
  ("FEM_NEDELEC(2)", TRI, 0, true, T10),
  // quadrilaterals (parallelograms)
  ("FEM_QK(2,3)", QUAD, 3, false, Q10),
  ("FEM_QK_DISCONTINUOUS(2,2)", QUAD, 2, false, Q10),
  ("FEM_Q2_INCOMPLETE(2)", QUAD, 2, false, Q10),
  ("FEM_QK_HIERARCHICAL(2,2)", QUAD, 2, false, Q10),
  ("FEM_QUAD_IPK(2,2)", QUAD, 2, false, Q10),
  (
    "FEM_PRODUCT(FEM_PK_GAUSSLOBATTO1D(3),FEM_PK_GAUSSLOBATTO1D(3))",
    QUAD,
    3,
    false,
    Q10,
  ),
  ("FEM_QUADC1_COMPOSITE", QUAD, 3, false, QC1I),
  ("FEM_REDUCED_QUADC1_COMPOSITE", QUAD, 2, false, QC1I),
  (
    "FEM_STRUCTURED_COMPOSITE(FEM_QK(2,1),2)",
    QUAD,
    1,
    false,
    "IM_STRUCTURED_COMPOSITE(IM_GAUSS_PARALLELEPIPED(2,4),2)",
  ),
  ("FEM_RT0Q(2)", QUAD, 0, true, Q10),
  // tetrahedra
  ("FEM_PK(3,3)", TET, 3, false, TET8),
  ("FEM_PK_HIERARCHICAL(3,3)", TET, 3, false, TET8),
  ("FEM_PK_WITH_CUBIC_BUBBLE(3,1)", TET, 1, false, TET8),
  ("FEM_P1_BUBBLE_FACE(3)", TET, 1, false, TET8),
  ("FEM_HERMITE(3)", TET, 3, false, TET8),
  ("FEM_MORLEY(3)", TET, 2, false, TET8),
  (
    "FEM_STRUCTURED_COMPOSITE(FEM_PK(3,1),2)",
    TET,
    1,
    false,
    "IM_STRUCTURED_COMPOSITE(IM_TETRAHEDRON(2),2)",
  ),
  (
    "FEM_PK_HIERARCHICAL_COMPOSITE(3,1,2)",
    TET,
    1,
    false,
    "IM_STRUCTURED_COMPOSITE(IM_TETRAHEDRON(2),2)",
  ),
  ("FEM_RTK(3,0)", TET, 0, true, TET8),
  ("FEM_NEDELEC(3)", TET, 0, true, TET8),
  // hexahedra, prisms, pyramids
  ("FEM_QK(3,2)", HEX, 2, false, H8),
  ("FEM_Q2_INCOMPLETE(3)", HEX, 2, false, H8),
  ("FEM_QK_HIERARCHICAL(3,2)", HEX, 2, false, H8),
  ("FEM_RT0Q(3)", HEX, 0, true, H8),
  ("FEM_PK_PRISM(3,2)", PRISM, 2, false, PR8),
  ("FEM_PRISM_PK_HIERARCHICAL(3,2)", PRISM, 2, false, PR8),
  ("FEM_PRISM_INCOMPLETE_P2", PRISM, 2, false, PR8),
  ("FEM_PRISM_IPK(3,1)", PRISM, 1, false, PR8),
  ("FEM_PYRAMID_QK(1)", PYR, 1, false, PY8),
  ("FEM_PYRAMID_QK(2)", PYR, 2, false, PY8),
  ("FEM_PYRAMID_Q2_INCOMPLETE", PYR, 2, false, PY8),
];

fn mesh(gt: &str) -> Mesh {
  let pgt = geotrans_descriptor(gt).unwrap();
  let n = pgt.dim();
  if gt == PYR {
    let mut m = MeshData::new(3);
    let pts: [&[f64]; 5] = [
      &[0.0, 0.0, 0.0],
      &[1.0, 0.1, 0.0],
      &[0.1, 1.0, 0.0],
      &[1.1, 1.1, 0.0],
      &[0.5, 0.6, 0.9],
    ];
    m.add_convex_by_points(&pgt, &pts).unwrap();
    return Mesh::from(m);
  }
  let m = regular_unit_mesh(&vec![2; n], &pgt).unwrap();
  if pgt.basic_structure().nb_points() == n + 1 {
    // simplices: move the points (the elements stay affine)
    for i in m.point_index() {
      let p = m.point(i).to_vec();
      let q: Vec<f64> = (0..n)
        .map(|d| p[d] + 0.07 * (1.3 * (d as f64 + 1.0) * p.iter().sum::<f64>() + d as f64).sin())
        .collect();
      m.set_point(i, &q).unwrap();
    }
  } else {
    let mut t = Mat::identity(n);
    for d in 1..n {
      t[(0, d)] = 0.3;
      t[(d, d)] = 1.2;
    }
    m.transform(&t).unwrap();
  }
  m
}

/// a polynomial of degree k in dimension n, component c
fn poly(n: usize, k: usize, c: usize) -> String {
  let mut s = String::from("1");
  for j in 0..k {
    let lin: Vec<String> = (0..n)
      .map(|d| {
        format!(
          "{:.2}*X({})",
          0.4 + 0.3 * ((j + 2 * d + c) % 5) as f64,
          d + 1
        )
      })
      .collect();
    s += &format!("*({:.1} + {})", 0.5 + j as f64, lin.join(" - "));
  }
  s
}

#[test]
fn patch_tests() {
  let mut bad = vec![];
  for &(name, gt, k, vector, im) in CASES {
    let m = mesh(gt);
    let n = m.dim();
    let pf = fem_descriptor(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    let q = if vector { n } else { 1 };
    let mf = MeshFem::new(&m, q).with_finite_element(&pf).unwrap();
    let mim = MeshIm::new(&m)
      .with_integration_method(&integ_descriptor(im).unwrap())
      .unwrap();
    let p = if vector {
      format!(
        "[{}]",
        (0..n).map(|c| poly(n, k, c)).collect::<Vec<_>>().join(";")
      )
    } else {
      poly(n, k, 0)
    };
    let mut md = Model::new();
    md.add_fem_variable("u", &mf).unwrap();
    md.add_term(&mim, None, &format!("(u - {p}).Test_u"))
      .unwrap();
    let rep = Newton::default().solve(&mut md, None).unwrap();
    assert!(rep.converged(), "{name}");
    let mut s = |e: &str| {
      let ws = md.workspace_mut();
      ws.clear_expressions();
      ws.add_expression(e, &mim, None).unwrap();
      ws.assembly(0, false).unwrap();
      ws.assembled_potential()
    };
    let (e0, n0) = (
      s(&format!("Norm_sqr(u - {p})")),
      s(&format!("Norm_sqr({p})")),
    );
    let (e1, n1) = (
      s(&format!("Norm_sqr(Grad_u - Grad({p}))")),
      s(&format!("Norm_sqr(Grad({p}))")),
    );
    let r0 = (e0 / n0).sqrt();
    // constants: no gradient to compare with
    let r1 = if k > 0 { (e1 / n1).sqrt() } else { e1.sqrt() };
    println!("{name:62} {gt:14} P{k}: rel. error {r0:.1e}, gradient {r1:.1e}");
    if r0 > 1e-10 || r1 > 1e-9 {
      bad.push(name);
    }
  }
  assert!(bad.is_empty(), "not reproduced: {bad:?}");
}
