#[derive(Debug, thiserror::Error)]
pub enum Error {
  #[error("invalid argument: {0}")]
  InvalidArgument(String),
  #[error("not implemented: {0}")]
  NotImplemented(&'static str),
  #[error(transparent)]
  Geom(#[from] getfem_geom::Error),
  #[error(transparent)]
  Linalg(#[from] getfem_linalg::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
