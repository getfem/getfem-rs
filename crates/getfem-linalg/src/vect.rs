//! gmm's vector operations on slices

/// a . b (gmm::vect_sp)
pub fn vect_sp(a: &[f64], b: &[f64]) -> f64 {
  a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// |a|^2 (gmm::vect_norm2_sqr)
pub fn vect_norm2_sqr(a: &[f64]) -> f64 {
  a.iter().map(|x| x * x).sum()
}

/// |a| (gmm::vect_norm2)
pub fn vect_norm2(a: &[f64]) -> f64 {
  vect_norm2_sqr(a).sqrt()
}

/// |a - b|^2 (gmm::vect_dist2_sqr)
pub fn vect_dist2_sqr(a: &[f64], b: &[f64]) -> f64 {
  a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// |a - b| (gmm::vect_dist2)
pub fn vect_dist2(a: &[f64], b: &[f64]) -> f64 {
  vect_dist2_sqr(a, b).sqrt()
}
