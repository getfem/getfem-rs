//! Reference convexes, geometric transformations, polynomials, mesh.
#![forbid(unsafe_code)]

pub mod convex_ref;
pub mod convex_structure;
pub mod delaunay;
pub mod error;
pub mod geotrans;
pub mod intern;
pub mod locate;
pub mod mesh;
pub mod mesh_file;
pub mod naming;
pub mod poly;
pub mod predicates;
pub mod region;
pub mod regular_mesh;
pub mod search;
pub mod shared;
pub mod signed_distance;
pub mod simplexify;

pub use error::{Error, Result};
