//! Direct linear solver interface: analyze once per pattern, factorize per
//! matrix, solve many right-hand sides.

use crate::Result;
use crate::col_matrix::ColMatrix;
use crate::sparse::Csr;

/// A sparse matrix as the solvers read it: its entries in any order (as
/// GetFEM's solvers take any gmm matrix; the models assemble a ColMatrix).
pub trait SparseMatrix {
  fn nrows(&self) -> usize;
  fn ncols(&self) -> usize;
  fn nnz(&self) -> usize;
  /// f(row, column, value) for each stored entry, in a fixed order
  fn for_each_entry(&self, f: &mut dyn FnMut(usize, usize, f64));
}

impl SparseMatrix for Csr<f64> {
  fn nrows(&self) -> usize {
    self.nrows
  }
  fn ncols(&self) -> usize {
    self.ncols
  }
  fn nnz(&self) -> usize {
    self.val.len()
  }
  fn for_each_entry(&self, f: &mut dyn FnMut(usize, usize, f64)) {
    for i in 0..self.nrows {
      for p in self.ptr[i]..self.ptr[i + 1] {
        f(i, self.idx[p], self.val[p]);
      }
    }
  }
}

impl SparseMatrix for ColMatrix<f64> {
  fn nrows(&self) -> usize {
    self.nrows
  }
  fn ncols(&self) -> usize {
    self.ncols
  }
  fn nnz(&self) -> usize {
    self.cols.iter().map(|c| c.len()).sum()
  }
  fn for_each_entry(&self, f: &mut dyn FnMut(usize, usize, f64)) {
    for (j, col) in self.cols.iter().enumerate() {
      for &(i, v) in col {
        f(i, j, v);
      }
    }
  }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixKind {
  General,
  Symmetric,
  SymmetricPositiveDefinite,
}

#[derive(Clone, Debug, Default)]
pub struct SolverInfo {
  pub solver: &'static str,
  pub n: usize,
  pub nnz: usize,
  /// entries in the factors, when the solver reports them
  pub nnz_factors: Option<u64>,
  /// estimated flops of the factorization
  pub flops: Option<f64>,
  pub messages: Vec<String>,
}

pub trait LinearSolver {
  /// Symbolic analysis of the pattern of a (square).
  fn analyze(&mut self, a: &dyn SparseMatrix, kind: MatrixKind) -> Result<()>;
  /// Numerical factorization; the pattern must be the analyzed one.
  fn factorize(&mut self, a: &dyn SparseMatrix) -> Result<()>;
  /// Solve in place with the current factors.
  fn solve(&mut self, b: &mut [f64]) -> Result<()>;
  fn info(&self) -> SolverInfo;
  /// Free the factors (the solver can be analyzed again).
  fn release(&mut self);

  /// analyze + factorize + solve
  fn solve_system(&mut self, a: &dyn SparseMatrix, kind: MatrixKind, b: &mut [f64]) -> Result<()> {
    self.analyze(a, kind)?;
    self.factorize(a)?;
    self.solve(b)
  }
}
