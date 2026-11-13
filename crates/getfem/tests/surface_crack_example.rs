//! The example surface_crack: the crack through a 3D block in plane strain
//! opens as the 2D edge crack (the 3D front enrichment on a straight front
//! ending on free faces), with its stress intensity factor along the whole
//! front, and the semicircular surface crack less.

#[path = "../examples/surface_crack.rs"]
#[allow(dead_code)]
mod example;

#[test]
fn through_crack_as_the_edge_crack() {
  let h = 0.25;
  let edge = example::simulate("edge", h);
  let through = example::simulate("through", h);
  let scale = edge.delta[0];
  assert!(scale > 2.0, "edge opening {scale}");
  for (k, (a, b)) in edge.delta.iter().zip(&through.delta).enumerate() {
    // measured at h = 0.25: at most 0.8 % of the opening at the surface
    assert!(
      (a - b).abs() <= 0.015 * scale,
      "x {}: 2D {a} vs 3D {b}",
      edge.x[k]
    );
  }
  // K_I along the front (ends too) as the edge crack's: measured 0.1 %
  let k = edge.sifs[0].0[0].k[0];
  assert_eq!(through.sifs.len(), 1);
  for s in &through.sifs[0].0 {
    assert!(
      (s.k[0] / k - 1.0).abs() < 5e-3,
      "{:?}: {} vs {k}",
      s.point,
      s.k[0]
    );
    assert!(
      s.k[1].abs() < 1e-2 * k && s.k[2].abs() < 1e-2 * k,
      "{:?}",
      s.k
    );
  }
  let semi = example::simulate("semi", h);
  assert!(
    semi.delta[0] > 0.3 * scale && semi.delta[0] < 0.7 * scale,
    "semi {}",
    semi.delta[0]
  );
}
