//! GetFEM's gmm layer: scalars, small dense tensors, sparse formats and the
//! LinearSolver trait (no solver of its own).
#![forbid(unsafe_code)]

pub mod cg;
pub mod col_matrix;
pub mod dense_qr;
pub mod error;
pub mod random;
pub mod range_basis;
pub mod scalar;
pub mod solver;
pub mod sparse;
pub mod tensor;
pub mod threads;
pub mod vect;

pub use cg::cg;
pub use col_matrix::{ColMatrix, RsVector, add_to_rsvector};
pub use error::{Error, Result};
pub use random::Rand;
pub use range_basis::{range_basis, range_basis_after};
pub use scalar::{Real, Scalar};
pub use solver::{LinearSolver, MatrixKind, SolverInfo, SparseMatrix};
pub use sparse::{Coo, Csr};
pub use vect::{vect_dist2, vect_dist2_sqr, vect_norm2, vect_norm2_sqr, vect_sp};
