//! Inversion on very anisotropic and strongly curved elements.

use getfem_geom::geotrans::{PGeoTrans, geotrans_descriptor};
use getfem_linalg::tensor::Mat;

struct Rng(u64);
impl Rng {
  fn f(&mut self) -> f64 {
    self.0 = self
      .0
      .wrapping_mul(6364136223846793005)
      .wrapping_add(1442695040888963407);
    (self.0 >> 11) as f64 / (1u64 << 53) as f64
  }
}

/// Node matrix from a map of the reference nodes.
fn nodes(gt: &PGeoTrans, f: impl Fn(&[f64]) -> Vec<f64>) -> Mat {
  let cols: Vec<Vec<f64>> = gt
    .geometric_nodes()
    .iter()
    .map(|noderef| f(noderef))
    .collect();
  Mat::from_data(&[cols[0].len(), cols.len()], cols.concat()).unwrap()
}

/// Random reference points strictly inside, checked to have J > 0.
/// `ar`: aspect ratio; real coordinates carry 1e-16 relative rounding, so the
/// thin-direction reference coordinate is only known to about 1e-16 * ar.
fn roundtrip(name: &str, gt: &PGeoTrans, g: &Mat, ar: f64, r: &mut Rng) {
  let p = gt.dim();
  let mut n = 0;
  while n < 200 {
    let xref: Vec<f64> = (0..p).map(|_| r.f()).collect();
    if gt.convex_ref().is_in(&xref).unwrap() > -1e-3 {
      continue;
    }
    n += 1;
    let pg = gt.at(g, &xref).unwrap();
    assert!(pg.j > 0.0, "{name}: element not valid at {xref:?}");
    let inv = gt.invert(g, &pg.x, 1e-12, false).unwrap();
    let err = xref
      .iter()
      .zip(&inv.xref)
      .map(|(a, b)| (a - b).abs())
      .fold(0.0, f64::max);
    let tol = 1e-10 + 1e-13 * ar;
    assert!(
      inv.converged && inv.inside && err < tol,
      "{name}: xref {xref:?} -> {:?} (err {err:.1e}, converged {}, residual {:.1e})",
      inv.xref,
      inv.converged,
      inv.residual
    );
  }
}

fn rot2(a: f64, v: [f64; 2]) -> Vec<f64> {
  let (s, c) = a.sin_cos();
  vec![c * v[0] - s * v[1], s * v[0] + c * v[1]]
}

#[test]
fn high_aspect_ratio() {
  let mut r = Rng(1);
  for g in ["GT_PK(2,1)", "GT_QK(2,1)", "GT_PK(2,2)", "GT_QK(2,2)"] {
    let gt = geotrans_descriptor(g).unwrap();
    for ar in [1e3, 1e6, 1e9] {
      for sc in [1e-3, 1.0, 1e3] {
        // thin, rotated, slightly distorted (non-affine for degree > 1)
        let m = nodes(&gt, |xref| {
          rot2(
            0.5,
            [
              sc * (xref[0] + 0.1 * xref[0] * xref[1]),
              sc / ar * (xref[1] + 0.05 * xref[0]),
            ],
          )
        });
        roundtrip(&format!("{g} ar={ar:e} sc={sc:e}"), &gt, &m, ar, &mut r);
      }
    }
  }
  // thin tetrahedron and hexahedron (flat plate, 1e-6 thick)
  for g in ["GT_PK(3,1)", "GT_QK(3,1)", "GT_QK(3,2)"] {
    let gt = geotrans_descriptor(g).unwrap();
    let m = nodes(&gt, |xref| {
      vec![
        xref[0] + 0.3 * xref[1],
        0.7 * xref[1] - 0.2 * xref[0] + 0.1 * xref[0] * xref[1],
        1e-6 * (xref[2] + 0.1 * xref[0]),
      ]
    });
    roundtrip(g, &gt, &m, 1e6, &mut r);
  }
}

#[test]
fn strongly_curved() {
  let mut r = Rng(2);
  // quarter annulus as one Q2 / Q3 element, from thick to very thin shells
  for g in ["GT_QK(2,2)", "GT_QK(2,3)"] {
    let gt = geotrans_descriptor(g).unwrap();
    for t in [0.5, 1e-3, 1e-6] {
      let m = nodes(&gt, |xref| {
        let (rad, th) = (1.0 + t * xref[0], std::f64::consts::FRAC_PI_2 * xref[1]);
        vec![rad * th.cos(), rad * th.sin()]
      });
      roundtrip(&format!("{g} annulus t={t:e}"), &gt, &m, 1.0 / t, &mut r);
    }
  }
  // P2 triangle with a strongly bulged edge (midside node pushed out by 0.3)
  let gt = geotrans_descriptor("GT_PK(2,2)").unwrap();
  let m = nodes(&gt, |xref| {
    let bulge = 4.0 * xref[0] * (1.0 - xref[0] - xref[1]);
    vec![xref[0], xref[1] - 0.3 * bulge]
  });
  roundtrip("P2 bulged edge", &gt, &m, 1.0, &mut r);
  // curved Q2 hexahedron: thin cylindrical shell sector
  let gt = geotrans_descriptor("GT_QK(3,2)").unwrap();
  for t in [0.2, 1e-4] {
    let m = nodes(&gt, |xref| {
      let (rad, th) = (1.0 + t * xref[0], 1.2 * xref[2]);
      vec![rad * th.cos(), 3.0 * xref[1], rad * th.sin()]
    });
    roundtrip(&format!("Q2 hex shell t={t:e}"), &gt, &m, 1.0 / t, &mut r);
  }
}

#[test]
fn far_from_origin() {
  let mut r = Rng(3);
  // nodes and points on a 2^-23 grid: shifted by 1e9 (spacing 2^-23
  // there) exactly, so both inversions see the same element
  let q = |v: f64| (v * 8388608.0).round() / 8388608.0;
  let off = 1e9;
  for g in [
    "GT_PK(2,1)",
    "GT_QK(2,2)",
    "GT_QK(2,3)",
    "GT_PK(3,2)",
    "GT_QK(3,2)",
  ] {
    let gt = geotrans_descriptor(g).unwrap();
    // curved element of size 1/60 at 0.6 (cell of a 60 x 60 mesh)
    let f = |xref: &[f64], o: f64| -> Vec<f64> {
      let w = 0.05 * xref[0] * xref[1];
      xref.iter().map(|&v| q(0.6 + (v + w) / 60.0) + o).collect()
    };
    let (m0, m1) = (
      nodes(&gt, |xref| f(xref, 0.0)),
      nodes(&gt, |xref| f(xref, off)),
    );
    let mut n = 0;
    while n < 200 {
      let xref: Vec<f64> = (0..gt.dim()).map(|_| r.f()).collect();
      if gt.convex_ref().is_in(&xref).unwrap() > -1e-3 {
        continue;
      }
      n += 1;
      let p0: Vec<f64> = gt
        .transform(&xref, &m0)
        .unwrap()
        .into_iter()
        .map(q)
        .collect();
      let p1: Vec<f64> = p0.iter().map(|v| v + off).collect();
      let (i0, i1) = (
        gt.invert(&m0, &p0, 1e-12, false).unwrap(),
        gt.invert(&m1, &p1, 1e-12, false).unwrap(),
      );
      let d = i0
        .xref
        .iter()
        .zip(&i1.xref)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
      assert!(
        i0.converged && i0.inside && i1.converged && i1.inside && d < 1e-12,
        "{g}: {:?} -> {:?} at 0.6 ({}, {}), {:?} at 1e9 ({}, {}, residual {:.1e})",
        xref,
        i0.xref,
        i0.converged,
        i0.inside,
        i1.xref,
        i1.converged,
        i1.inside,
        i1.residual
      );
    }
  }
}
