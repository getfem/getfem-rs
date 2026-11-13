//! Finite elements, cubature, mesh_fem, mesh_im.
#![forbid(unsafe_code)]

pub mod crack_front;
pub mod crack_tip_function;
mod cubature_data;
mod cut2d;
pub mod error;
pub mod fem;
pub(crate) mod fem_composite;
pub mod fem_global_function;
mod fem_hho;
mod fem_level_set;
mod fem_special;
mod fem_transformed;
mod fem_vector;
mod gausslobatto_data;
pub mod global_function;
pub mod im_data;
pub mod integration;
pub mod interpolated_fem;
pub mod interpolation;
pub mod level_set;
pub mod mesh_fem;
mod mesh_fem_product;
mod mesh_fem_sum;
pub mod mesh_im;
pub mod mesh_im_level_set;
pub mod mesh_level_set;
pub mod precomp;
pub mod slice;

pub use error::{Error, Result};
