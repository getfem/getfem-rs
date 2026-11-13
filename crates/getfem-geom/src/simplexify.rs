//! Simplexification tables (from bgeot_convex_ref_simplexified.cc).

use crate::convex_structure::{self as cs, PCvs};

const PAR2: &[&[usize]] = &[&[3, 0, 2], &[3, 0, 1]];

const PAR3: &[&[usize]] = &[
  &[3, 7, 0, 1],
  &[7, 0, 5, 4],
  &[7, 0, 1, 5],
  &[3, 7, 0, 2],
  &[6, 7, 0, 4],
  &[6, 7, 0, 2],
];

const PAR4: &[&[usize]] = &[
  &[9, 3, 1, 5, 0],
  &[9, 12, 10, 8, 0],
  &[9, 12, 5, 13, 15],
  &[9, 3, 10, 6, 0],
  &[9, 3, 5, 6, 0],
  &[9, 3, 10, 11, 15],
  &[9, 12, 10, 6, 0],
  &[9, 12, 5, 6, 0],
  &[9, 3, 10, 6, 15],
  &[9, 3, 5, 6, 15],
  &[9, 12, 10, 6, 15],
  &[9, 12, 5, 6, 15],
  &[12, 4, 5, 6, 0],
  &[3, 5, 6, 7, 15],
  &[12, 10, 14, 6, 15],
  &[3, 10, 6, 2, 0],
];

const PRISM3: &[&[usize]] = &[&[2, 4, 0, 1], &[2, 4, 0, 3], &[2, 4, 5, 3]];

const PRISM4: &[&[usize]] = &[
  &[6, 1, 3, 7, 0],
  &[6, 1, 3, 2, 0],
  &[6, 1, 4, 7, 0],
  &[6, 1, 4, 7, 5],
];

const PYRAMID: &[&[usize]] = &[&[0, 1, 2, 4], &[3, 2, 1, 4]];

/// Simplices (local point indices) decomposing a basic convex structure;
/// None if no table (d >= 5 not verified conformal).
pub fn simplexified(cvs: &PCvs) -> Option<Vec<Vec<usize>>> {
  if cvs.nb_points() == cvs.dim() + 1 {
    return Some(vec![(0..=cvs.dim()).collect()]);
  }
  let t = table(cvs)?;
  Some(t.iter().map(|r| r.to_vec()).collect())
}

fn table(cvs: &PCvs) -> Option<&'static [&'static [usize]]> {
  if **cvs == *cs::parallelepiped_structure(2, 1) {
    return Some(PAR2);
  }
  if **cvs == *cs::parallelepiped_structure(3, 1) {
    return Some(PAR3);
  }
  if **cvs == *cs::parallelepiped_structure(4, 1) {
    return Some(PAR4);
  }
  if **cvs == *cs::prism_p1_structure(3) {
    return Some(PRISM3);
  }
  if **cvs == *cs::prism_p1_structure(4) {
    return Some(PRISM4);
  }
  if **cvs == *cs::pyramid_qk_structure(1).unwrap() {
    return Some(PYRAMID);
  }
  None
}
