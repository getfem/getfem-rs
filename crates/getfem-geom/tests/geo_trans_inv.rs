//! Port of GetFEM tests/geo_trans_inv.cc (per-element part), extended to
//! pyramids and incomplete elements and run at scales 1e-3, 1, 1e3.

use getfem_geom::geotrans::{PGeoTrans, geotrans_descriptor};
use getfem_linalg::tensor::{Mat, det};

struct Rng(u64);
impl Rng {
  fn next(&mut self) -> f64 {
    self.0 = self
      .0
      .wrapping_mul(6364136223846793005)
      .wrapping_add(1442695040888963407);
    (self.0 >> 11) as f64 / (1u64 << 53) as f64
  }
}

/// Random matrix with unit rows and |det| >= 0.7 (as GetFEM's random_base).
fn random_base(n: usize, r: &mut Rng) -> Mat {
  loop {
    let mut m = Mat::zeros(&[n, n]);
    for i in 0..n {
      let v: Vec<f64> = (0..n).map(|_| r.next() * 2.0 - 1.0).collect();
      let l = v.iter().map(|x| x * x).sum::<f64>().sqrt();
      for j in 0..n {
        m[(i, j)] = v[j] / l;
      }
    }
    if det(&m).unwrap().abs() >= 0.7 {
      return m;
    }
  }
}

fn test_inversion(gt: &PGeoTrans, scale: f64, r: &mut Rng) {
  let (n, np) = (gt.dim(), gt.nb_points());
  let m = random_base(n, r);
  let tr: Vec<f64> = (0..n).map(|_| r.next()).collect();
  let mut g = Mat::zeros(&[n, np]);
  for i in 0..np {
    let noderef = &gt.geometric_nodes()[i];
    for k in 0..n {
      let v: f64 = (0..n).map(|j| m[(k, j)] * noderef[j]).sum::<f64>() + tr[k];
      g[(k, i)] = v * scale + r.next() * 0.05 * scale;
    }
  }
  let eps = 1e-12;
  let check = |p: &[f64], pref: &[f64], expected_in: bool| {
    let inv = gt.invert(&g, p, eps, false).unwrap();
    let p2 = gt.transform(&inv.xref, &g).unwrap();
    let err = p
      .iter()
      .zip(&p2)
      .map(|(a, b)| (a - b) * (a - b))
      .sum::<f64>()
      .sqrt();
    let multiple = err < 1e-10 * scale
      && pref
        .iter()
        .zip(&inv.xref)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
        > 1e-5;
    assert!(
      err <= 1e-10 * scale,
      "{} scale {scale}: inversion of {p:?} failed, err {err}, got {:?}",
      gt.name(),
      inv.xref
    );
    assert!(inv.converged, "{}: not converged", gt.name());
    assert!(
      inv.inside == expected_in || multiple,
      "{} scale {scale}: {pref:?} inside={expected_in}, got {:?}",
      gt.name(),
      inv.xref
    );
  };
  for i in 0..np {
    check(g.col(i), &gt.geometric_nodes()[i], true);
  }
  // reference bounding box, enlarged by 25 % each side
  let (lo, hi): (Vec<f64>, Vec<f64>) = (0..n)
    .map(|k| {
      gt.geometric_nodes()
        .iter()
        .fold((f64::MAX, f64::MIN), |(l, h), noderef| {
          (l.min(noderef[k]), h.max(noderef[k]))
        })
    })
    .unzip();
  for _ in 0..100 {
    let mut pref: Vec<f64> = (0..n)
      .map(|k| lo[k] + (hi[k] - lo[k]) * (r.next() * 1.5 - 0.25))
      .collect();
    if gt.name().starts_with("GT_PYRAMID") {
      // stay off the pole z = 1 of the rational map (GetFEM does not test pyramids)
      let h = 0.1 + 0.9 * r.next() * 1.1;
      pref[2] = 1.0 - h.min(1.25);
      pref[0] = (r.next() * 2.0 - 1.0) * 1.25 * h;
      pref[1] = (r.next() * 2.0 - 1.0) * 1.25 * h;
    }
    let p = gt.transform(&pref, &g).unwrap();
    let inside = gt.convex_ref().is_in(&pref).unwrap() < 1e-10;
    check(&p, &pref, inside);
  }
}

#[test]
fn inversion_catalogue() {
  let mut names: Vec<String> = vec![];
  for n in 1..=4 {
    for k in 1..if n < 4 { 3 } else { 2 } {
      names.push(format!("GT_PK({n},{k})"));
    }
  }
  for n in 1..=3 {
    for k in 1..3 {
      names.push(format!("GT_QK({n},{k})"));
    }
  }
  for n in 2..=3 {
    for k in 1..3 {
      names.push(format!("GT_PRISM({n},{k})"));
    }
  }
  names.extend(
    [
      "GT_LINEAR_QK(5)",
      "GT_LINEAR_PRODUCT(GT_PK(2,1),GT_PK(1,1))",
      "GT_PYRAMID(1)",
      "GT_PYRAMID(2)",
      "GT_Q2_INCOMPLETE(2)",
      "GT_Q2_INCOMPLETE(3)",
      "GT_PYRAMID_Q2_INCOMPLETE",
      "GT_PRISM_INCOMPLETE_P2",
    ]
    .map(String::from),
  );
  let mut r = Rng(42);
  for name in &names {
    let gt = geotrans_descriptor(name).unwrap();
    for sc in [1e-3, 1.0, 1e3] {
      test_inversion(&gt, sc, &mut r);
    }
  }
}

#[test]
fn test0() {
  // GetFEM's "problematic test-case": the point has no preimage under this
  // bilinear map (least squares residual ~ 0.0099); GetFEM returns
  // [1.48455, 1.16208] without saying so. We report non-convergence.
  let gt = geotrans_descriptor("GT_QK(2,1)").unwrap();
  let g = Mat::from_rows(&[&[2.5, 5.0, 2.5, 3.2], &[0.6, 0.0, 1.8, 1.5]]);
  let inv = gt.invert(&g, &[3.132, 1.617], 1e-12, false).unwrap();
  assert!(!inv.converged && !inv.inside);
  // near the fold (K singular) Newton stalls where GetFEM does (residual 0.026)
  let r = inv.residual * getfem_geom::geotrans::element_extent(&g);
  assert!(
    r < 0.03 && (inv.xref[0] - 1.48455).abs() < 5e-3,
    "residual {r}, xref {:?}",
    inv.xref
  );
}

#[test]
fn codim_inversion() {
  // point off a triangle in 3D: not converged (distance > 0), foot point found
  let gt = geotrans_descriptor("GT_PK(2,1)").unwrap();
  for sc in [1e-3, 1.0, 1e3] {
    let g = Mat::from_rows(&[&[0.0, sc, 0.0], &[0.0, 0.0, sc], &[0.0, 0.0, 0.0]]);
    let on = gt
      .invert(&g, &[0.25 * sc, 0.5 * sc, 0.0], 1e-12, false)
      .unwrap();
    assert!(on.converged && on.inside);
    assert!((on.xref[0] - 0.25).abs() < 1e-14 && (on.xref[1] - 0.5).abs() < 1e-14);
    let off = gt
      .invert(&g, &[0.25 * sc, 0.5 * sc, 0.1 * sc], 1e-12, false)
      .unwrap();
    assert!(!off.converged);
    assert!((off.xref[0] - 0.25).abs() < 1e-14 && (off.xref[1] - 0.5).abs() < 1e-14);
    assert!((off.residual - 0.1).abs() < 1e-12);
  }
}
