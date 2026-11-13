//! The example inclined_penny_crack: K_I, K_II and K_III along the front
//! of a penny crack tilted to a uniaxial tension, against the infinite
//! body's.

#[path = "../examples/inclined_penny_crack.rs"]
#[allow(dead_code)]
mod example;

#[test]
fn mixed_mode_along_the_front() {
  let o = example::simulate(0.25);
  assert!(o.sifs.len() > 10, "{} front points", o.sifs.len());
  // the peaks of the exact factors
  let peak = [
    example::exact(&[0.0, 1.0, 0.0]).1[0],
    example::exact(&[-1.0, 0.0, 0.0]).1[1].abs(),
    example::exact(&[0.0, 1.0, 0.0]).1[2].abs(),
  ];
  for s in &o.sifs {
    let (w, k) = example::exact(&s.point);
    // K_III not at the ends on the symmetry plane
    let modes = if s.point[1] < 1e-6 { 2 } else { 3 };
    for m in 0..modes {
      // measured at h = 0.25: at most 2.7, 3.1, 4.7 % of the peak
      assert!(
        (s.k[m] - k[m]).abs() < 0.06 * peak[m],
        "angle {:.1}, mode {}: {} vs {}",
        w.to_degrees(),
        m + 1,
        s.k[m],
        k[m]
      );
    }
  }
}
