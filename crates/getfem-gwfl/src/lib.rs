//! GWFL parser, IR, executor.
#![forbid(unsafe_code)]

pub mod contact;
pub mod crack_sif;
pub mod defs;
pub mod derive;
pub mod elementary;
pub mod error;
mod ga;
pub mod global_function;
pub mod hho;
pub mod laws;
pub mod lexer;
pub mod names;
pub mod operators;
pub mod parser;
pub mod semantic;
pub mod symbolic;
pub mod transform;
pub mod workspace;
pub use defs::{NativeFn, define_gwfl_function, define_gwfl_function_native};
pub use error::{Error, Result};
