//! Functions of the crack tip coordinates (M12 step 1d): the singular
//! functions and cutoffs checked by their closed forms and by central
//! differences of the derivatives; products and sums.

use getfem_fem::crack_tip_function::{CrackTipFunction, Cutoff, Singular};

const ALL: [Singular; 13] = [
  Singular::SqrtRSin,
  Singular::SqrtRCos,
  Singular::SqrtRSinSinTheta,
  Singular::SqrtRCosSinTheta,
  Singular::R32Sin,
  Singular::R32Cos,
  Singular::R32SinCos2,
  Singular::R32Cos3,
  Singular::MinusInvSqrtRSin,
  Singular::InvSqrtRCos,
  Singular::RSin2,
  Singular::InvRSin,
  Singular::InvRCos,
];

/// points around the tip, away from the crack lips (y = 0, x < 0)
fn points() -> Vec<[f64; 2]> {
  let mut p = vec![];
  for k in 0..12 {
    let t = -2.9 + 5.8 * k as f64 / 11.0;
    for r in [0.07, 0.4, 1.3] {
      let (s, c) = t.sin_cos();
      p.push([r * c, r * s]);
    }
  }
  p
}

fn check_derivatives(f: &CrackTipFunction, p: [f64; 2], tol: f64) {
  let e = 1e-6;
  let (g, h) = (f.grad(p[0], p[1]), f.hess(p[0], p[1]));
  let scale = 1.0 + g.iter().chain(&h).fold(0.0f64, |a, v| a.max(v.abs()));
  for k in 0..2 {
    let (mut pp, mut pm) = (p, p);
    pp[k] += e;
    pm[k] -= e;
    let dv = (f.val(pp[0], pp[1]) - f.val(pm[0], pm[1])) / (2.0 * e);
    assert!(
      (dv - g[k]).abs() < tol * scale,
      "grad {k} at {p:?}: {} vs {dv}",
      g[k]
    );
    let (gp, gm) = (f.grad(pp[0], pp[1]), f.grad(pm[0], pm[1]));
    for i in 0..2 {
      let dg = (gp[i] - gm[i]) / (2.0 * e);
      assert!(
        (dg - h[i + 2 * k]).abs() < tol * scale,
        "hess ({i},{k}) at {p:?}: {} vs {dg}",
        h[i + 2 * k]
      );
    }
  }
}

#[test]
fn singular_functions() {
  for s in ALL {
    let f = CrackTipFunction::crack(s);
    for p in points() {
      check_derivatives(&f, p, 1e-6);
    }
  }
  // closed forms
  for p in points() {
    let (r, t) = ((p[0] * p[0] + p[1] * p[1]).sqrt(), p[1].atan2(p[0]));
    let (s, c) = (0.5 * t).sin_cos();
    for (f, v) in [
      (Singular::SqrtRSin, r.sqrt() * s),
      (Singular::SqrtRCosSinTheta, r.sqrt() * c * t.sin()),
      (Singular::R32SinCos2, r.powf(1.5) * s * c * c),
      (Singular::MinusInvSqrtRSin, -s / r.sqrt()),
      (Singular::RSin2, r * s * s),
      (Singular::InvRCos, c / r),
    ] {
      let got = CrackTipFunction::crack(f).val(p[0], p[1]);
      assert!((got - v).abs() < 1e-13 * (1.0 + v.abs()), "{f:?} at {p:?}");
    }
  }
  // the lips: theta = pi above (y = 0), -pi below
  let f = CrackTipFunction::crack(Singular::SqrtRSin);
  assert!((f.val(-4.0, 0.0) - 2.0).abs() < 1e-14);
  assert!((f.val(-4.0, -1e-300) + 2.0).abs() < 1e-14);
  assert_eq!(f.val(0.0, 0.0), 0.0);
}

#[test]
fn cutoffs_products_sums() {
  let cuts = [
    Cutoff::None,
    Cutoff::Exponential { radius: 0.4 },
    Cutoff::Polynomial {
      inner: 0.01,
      outer: 0.4,
    },
    Cutoff::Polynomial2 {
      inner: 0.01,
      outer: 0.4,
    },
  ];
  for c in cuts {
    let f = CrackTipFunction::cutoff(c).unwrap();
    for p in points() {
      // the polynomial pieces join at inner and outer: points away from them
      let rho = (p[0] * p[0] + p[1] * p[1]).sqrt();
      if (rho - 0.4).abs() > 1e-3 {
        check_derivatives(&f, p, 1e-5);
      }
    }
    assert!((f.val(0.0, 0.005) - 1.0).abs() < 1e-14 || matches!(c, Cutoff::Exponential { .. }));
  }
  assert!(
    CrackTipFunction::cutoff(Cutoff::Polynomial {
      inner: 0.5,
      outer: 0.4
    })
    .is_err()
  );
  assert!(CrackTipFunction::cutoff(Cutoff::Exponential { radius: 0.0 }).is_err());
  let ck = CrackTipFunction::crack(Singular::SqrtRCos);
  let co = CrackTipFunction::cutoff(cuts[3]).unwrap();
  let p = CrackTipFunction::product(&ck, &co);
  let s = CrackTipFunction::sum(&[ck.clone(), co.clone()]);
  for q in points() {
    if ((q[0] * q[0] + q[1] * q[1]).sqrt() - 0.4).abs() > 1e-3 {
      check_derivatives(&p, q, 1e-5);
      check_derivatives(&s, q, 1e-5);
    }
    assert_eq!(p.val(q[0], q[1]), ck.val(q[0], q[1]) * co.val(q[0], q[1]));
  }
}

/// Angular hats around a front, periodic and on [-pi, pi]: a partition
/// of unity in theta; gradients and Hessians against central differences
/// (away from the nodes' kinks).
#[test]
fn angular_hats() {
  let n = 8;
  for periodic in [true, false] {
    let nh = if periodic { n } else { n + 1 };
    let hats: Vec<CrackTipFunction> = (0..nh)
      .map(|m| CrackTipFunction::angular_hat(n, m, periodic).unwrap())
      .collect();
    let eps = 1e-6;
    let mut checked = 0;
    for k in 0..200 {
      let t = -3.1 + 6.2 * (k as f64 + 0.37) / 200.0;
      let r = 0.3 + 0.002 * k as f64;
      let (x, y) = (r * t.cos(), r * t.sin());
      let s: f64 = hats.iter().map(|h| h.val(x, y)).sum();
      assert!((s - 1.0).abs() < 1e-13, "sum {s} at theta {t}");
      for h in &hats {
        let (g, hs) = (h.grad(x, y), h.hess(x, y));
        let (gpx, gmx, gpy, gmy) = (
          h.grad(x + eps, y),
          h.grad(x - eps, y),
          h.grad(x, y + eps),
          h.grad(x, y - eps),
        );
        // skip points next to a node (kink)
        if (gpx[0] - gmx[0]).abs() > 1e-2 || (gpy[1] - gmy[1]).abs() > 1e-2 {
          continue;
        }
        checked += 1;
        let fx = (h.val(x + eps, y) - h.val(x - eps, y)) / (2.0 * eps);
        let fy = (h.val(x, y + eps) - h.val(x, y - eps)) / (2.0 * eps);
        assert!(
          (fx - g[0]).abs() < 1e-6 && (fy - g[1]).abs() < 1e-6,
          "grad at {t}"
        );
        let hxx = (gpx[0] - gmx[0]) / (2.0 * eps);
        let hxy = (gpy[0] - gmy[0]) / (2.0 * eps);
        let hyy = (gpy[1] - gmy[1]) / (2.0 * eps);
        assert!(
          (hxx - hs[0]).abs() < 1e-4 && (hxy - hs[2]).abs() < 1e-4 && (hyy - hs[3]).abs() < 1e-4,
          "hess at {t}"
        );
      }
    }
    assert!(checked > 300, "{checked} checks");
  }
}
