//! Foreign mesh formats: gmsh import, Exodus II import and export,
//! VTU/VTK export.
#![forbid(unsafe_code)]

pub mod error;
pub mod exodus;
mod exodus_types;
pub mod gmsh;
mod gmsh_types;
pub mod netcdf;
pub mod vtk;
pub mod zlib;

pub use error::{Error, Result};

/// A 3D mesh in the plane z = 0: how to make it 2D.
fn planar_hint(m: &getfem_geom::mesh::MeshData, file: &str) {
  if m.dim() == 3 && m.nb_points() > 0 && m.point_index().all(|i| m.point(i)[2] == 0.0) {
    eprintln!(
      "GetFEM-rs info: {file}: a 3D mesh in the plane z = 0; \
       m.transform([[1,0,0],[0,1,0]]) makes it 2D"
    );
  }
}
