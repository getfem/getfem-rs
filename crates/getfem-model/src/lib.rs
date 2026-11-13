//! Model (variables, data and terms around a workspace), Dirichlet
//! conditions, Newton.
#![forbid(unsafe_code)]

pub mod continuation;
pub mod error;
pub mod model;
pub mod newton;
pub use continuation::{
  ContinuationReport, ContinuationStatus, NumericalContinuation, VariableScale,
};
pub use error::{Error, Result};
pub use model::{BuildVersion, Constraint, Model, MultiplierArg, TermId, TermKind};
pub use newton::{
  CondensedStep, Linearized, Method, Newton, NewtonReport, NonlinearSystem, Status,
  VariableTolerance,
};
