//! gmm::random and gmm::fill_random on glibc's rand() (TYPE_3: additive
//! feedback, degree 31), so that a seed gives GetFEM's numbers.

pub const RAND_MAX: u32 = 2147483647;

#[derive(Clone, Debug)]
pub struct Rand {
  r: [u32; 34],
  k: usize,
}

impl Rand {
  /// glibc srand (0 is taken as 1)
  pub fn new(seed: u32) -> Self {
    let mut r = [0u32; 34];
    r[0] = if seed == 0 { 1 } else { seed };
    for i in 1..31 {
      // 16807 r mod (2^31 - 1), Schrage's method as glibc
      let (hi, lo) = ((r[i - 1] as i32) / 127773, (r[i - 1] as i32) % 127773);
      let mut w = 16807 * lo - 2836 * hi;
      if w < 0 {
        w += 2147483647;
      }
      r[i] = w as u32;
    }
    for i in 31..34 {
      r[i] = r[i - 31];
    }
    let mut s = Rand { r, k: 0 };
    // glibc discards the first 310 outputs
    for _ in 0..310 {
      s.next_raw();
    }
    s
  }

  fn next_raw(&mut self) -> u32 {
    // r[i] = r[i - 31] + r[i - 3], on a ring of 34
    let i = self.k;
    let v = self.r[(i + 34 - 31) % 34].wrapping_add(self.r[(i + 34 - 3) % 34]);
    self.r[i] = v;
    self.k = (i + 1) % 34;
    v >> 1
  }

  /// glibc rand()
  pub fn rand(&mut self) -> u32 {
    self.next_raw()
  }

  /// gmm::random(double()): uniform in [-1, 1)
  pub fn random(&mut self) -> f64 {
    (self.rand() as f64 * 2.0) / (RAND_MAX as f64 + 0.5) - 1.0
  }

  /// gmm::fill_random
  pub fn fill_random(&mut self, v: &mut [f64]) {
    for x in v {
      *x = self.random();
    }
  }
}

#[cfg(test)]
mod tests {
  use super::Rand;

  #[test]
  fn glibc_sequences() {
    for (seed, want) in [
      (7689, [130892403, 1323022521, 78516882, 1690008205]),
      (1, [1804289383, 846930886, 1681692777, 1714636915]),
      (0, [1804289383, 846930886, 1681692777, 1714636915]),
    ] {
      let mut r = Rand::new(seed);
      let got: Vec<u32> = (0..4).map(|_| r.rand()).collect();
      assert_eq!(got, want, "seed {seed}");
    }
  }
}
