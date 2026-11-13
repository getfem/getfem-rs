//! GetFEM-rs facade.
#![forbid(unsafe_code)]

pub use getfem_arpack as arpack;
pub use getfem_fem as fem;
pub use getfem_geom as geom;
pub use getfem_gwfl as gwfl;
pub use getfem_io as io;
pub use getfem_linalg as linalg;
pub use getfem_model as model;
pub use getfem_mpi as mpi;
pub use getfem_mumps as mumps;

pub use getfem_linalg::threads::{num_threads, set_num_threads};
