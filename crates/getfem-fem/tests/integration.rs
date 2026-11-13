//! Integration methods: exactness of every rule of known degree, pyramid
//! rules and faces, errors.

use getfem_fem::integration::{
  ApproxIntegration, PolyIntegration, classical_approx_im, classical_exact_im, integ_descriptor,
};
use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::poly::PowerIndex;

fn check_exactness(name: &str, a: &ApproxIntegration, e: &PolyIntegration, order: usize) {
  let n = a.dim();
  let mut pi = PowerIndex::new(n);
  while pi.degree() <= order {
    let mono = |xref: &[f64]| {
      pi.powers()
        .iter()
        .zip(xref)
        .map(|(&k, &v)| v.powi(k as i32))
        .product::<f64>()
    };
    let s: f64 = (0..a.nb_points_on_convex())
      .map(|i| a.coeff(i) * mono(a.point(i)))
      .sum();
    let ex = e.int_monomial(pi.powers());
    assert!(
      (s - ex).abs() < 1e-13,
      "{name}: x^{:?}: {s} vs {ex}",
      pi.powers()
    );
    for f in 0..a.ref_convex().structure().nb_faces() {
      let s: f64 = (0..a.nb_points_on_face(f))
        .map(|i| a.coeff_on_face(f, i) * mono(a.point_on_face(f, i)))
        .sum();
      let ex = e.int_monomial_on_face(pi.powers(), f);
      assert!(
        (s - ex).abs() < 1e-13,
        "{name} face {f}: x^{:?}: {s} vs {ex}",
        pi.powers()
      );
    }
    pi.next();
  }
}

#[test]
fn rules_are_exact_to_their_degree() {
  let fam = [
    ("IM_TRIANGLE", "GT_PK(2,1)"),
    ("IM_TETRAHEDRON", "GT_PK(3,1)"),
    ("IM_QUAD", "GT_QK(2,1)"),
    ("IM_HEXAHEDRON", "GT_QK(3,1)"),
    ("IM_SIMPLEX4D", "GT_PK(4,1)"),
    ("IM_CUBE4D", "GT_QK(4,1)"),
    ("IM_GAUSSLOBATTO1D", "GT_PK(1,1)"),
    ("IM_GAUSS1D", "GT_PK(1,1)"),
  ];
  let mut count = 0;
  for (f, g) in fam {
    let e = classical_exact_im(&geotrans_descriptor(g).unwrap()).unwrap();
    for k in 0..=99 {
      let name = format!("{f}({k})");
      let Ok(m) = integ_descriptor(&name) else {
        continue;
      };
      count += 1;
      // GetFEM's GAUSSLOBATTO1D(k) has exactness k; checked up to 30 (f64 powers)
      check_exactness(&name, m.approx().unwrap(), e.exact().unwrap(), k.min(30));
    }
  }
  for (n, g) in [(1, "GT_PK(1,1)"), (2, "GT_PK(2,1)"), (3, "GT_PK(3,1)")] {
    let e = classical_exact_im(&geotrans_descriptor(g).unwrap()).unwrap();
    for k in 0..=4 {
      let name = format!("IM_NC({n},{k})");
      check_exactness(
        &name,
        integ_descriptor(&name).unwrap().approx().unwrap(),
        e.exact().unwrap(),
        k,
      );
      count += 1;
    }
  }
  let prism = classical_exact_im(&geotrans_descriptor("GT_PRISM(3,1)").unwrap()).unwrap();
  check_exactness(
    "prism",
    integ_descriptor("IM_PRODUCT(IM_TRIANGLE(5),IM_GAUSS1D(5))")
      .unwrap()
      .approx()
      .unwrap(),
    prism.exact().unwrap(),
    5,
  );
  assert!(count > 150, "{count}");
}

#[test]
fn pyramid_rule_integrates_pyramid_polynomials() {
  // classical_approx_im(pyramid, d) exact for degree <= d on the pyramid;
  // reference: the cube rule of degree 30
  let gt = geotrans_descriptor("GT_PYRAMID(1)").unwrap();
  let refm = integ_descriptor("IM_PYRAMID(IM_GAUSS_PARALLELEPIPED(3,30))").unwrap();
  let r = refm.approx().unwrap();
  let integ = |a: &ApproxIntegration, pw: &[usize]| -> f64 {
    (0..a.nb_points_on_convex())
      .map(|i| {
        a.coeff(i)
          * a
            .point(i)
            .iter()
            .zip(pw)
            .map(|(&x, &k)| x.powi(k as i32))
            .product::<f64>()
      })
      .sum()
  };
  assert!((integ(r, &[0, 0, 0]) - 4.0 / 3.0).abs() < 1e-14);
  for d in 1..=6 {
    let m = classical_approx_im(&gt, d).unwrap();
    let a = m.approx().unwrap();
    let mut pi = PowerIndex::new(3);
    while pi.degree() <= d {
      let (s, e) = (integ(a, pi.powers()), integ(r, pi.powers()));
      assert!(
        (s - e).abs() < 1e-13,
        "degree {d}: x^{:?}: {s} vs {e}",
        pi.powers()
      );
      pi.next();
    }
  }
  // GetFEM's degree-1 choice is not exact for constants
  let low = integ_descriptor("IM_PYRAMID(IM_GAUSS_PARALLELEPIPED(3,1))").unwrap();
  assert!((integ(low.approx().unwrap(), &[0, 0, 0]) - 1.0).abs() < 1e-14);
}

#[test]
fn errors() {
  for s in [
    "IM_FOO(2)",
    "IM_TRIANGLE(1000)",
    "IM_PRODUCT(IM_EXACT_SIMPLEX(1),IM_GAUSS1D(2))",
    "IM_EXACT_SIMPLEX(0)",
    "GT_PK(2,1)",
  ] {
    assert!(integ_descriptor(s).is_err(), "{s}");
  }
  assert!(integ_descriptor("IM_NONE").unwrap().is_none());
  // odd Gauss names come from the table (paired point order), even ones are computed
  assert_eq!(
    integ_descriptor("IM_GAUSS1D(5)")
      .unwrap()
      .approx()
      .unwrap()
      .nb_points(),
    integ_descriptor("IM_GAUSS1D(4)")
      .unwrap()
      .approx()
      .unwrap()
      .nb_points()
  );
  assert_eq!(
    integ_descriptor("IM_GAUSS_PARALLELEPIPED(1,4)").unwrap(),
    integ_descriptor("IM_GAUSS1D(4)").unwrap()
  );
}

#[test]
fn pyramid_faces_have_their_area() {
  // base 4, each side sqrt(2) (triangle base 2, slant height sqrt(2))
  let a = integ_descriptor("IM_PYRAMID(IM_GAUSS_PARALLELEPIPED(3,4))").unwrap();
  let a = a.approx().unwrap();
  let area = |f: usize| {
    (0..a.nb_points_on_face(f))
      .map(|i| a.coeff_on_face(f, i))
      .sum::<f64>()
  };
  assert!((area(0) - 4.0).abs() < 1e-14);
  for f in 1..5 {
    assert!(
      (area(f) - 2f64.sqrt()).abs() < 1e-14,
      "face {f}: {}",
      area(f)
    );
  }
}
