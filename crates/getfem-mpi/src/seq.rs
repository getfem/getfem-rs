//! Without the feature mpi: one rank

use getfem_linalg::Result;

pub fn init() -> Result<()> {
  if let Some(n) = super::launched_size() {
    eprintln!(
      "GetFEM-rs warning: {n} processes, but built without the feature mpi: \
       each runs the sequential code"
    );
  }
  Ok(())
}

pub fn finalize() {}

pub fn finalized() -> bool {
  false
}

#[inline(always)]
pub fn rank() -> usize {
  0
}

#[inline(always)]
pub fn size() -> usize {
  1
}

pub fn sum(_: &mut [f64]) {}

pub fn max(_: &mut [f64]) {}

pub fn all_gather(v: &[f64]) -> Vec<f64> {
  v.to_vec()
}

pub fn bcast(_: &mut [f64], _: usize) {}

pub fn partition(xadj: &[usize], _: &[usize], _: usize) -> Vec<usize> {
  vec![0; xadj.len() - 1]
}

pub fn comm_fortran() -> Option<i32> {
  None
}
