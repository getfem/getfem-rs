#[derive(Debug, thiserror::Error)]
pub enum Error {
  #[error("not implemented: {0}")]
  NotImplemented(&'static str),
  #[error("invalid argument: {0}")]
  InvalidArgument(String),
  #[error(transparent)]
  Gwfl(#[from] getfem_gwfl::Error),
  #[error(transparent)]
  Fem(#[from] getfem_fem::Error),
  #[error(transparent)]
  Geom(#[from] getfem_geom::Error),
  #[error(transparent)]
  Linalg(#[from] getfem_linalg::Error),
  #[error("{0}")]
  Solve(String),
}

pub type Result<T> = std::result::Result<T, Error>;
