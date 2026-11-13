#[derive(Debug, thiserror::Error)]
pub enum Error {
  #[error("{0}")]
  Io(String),
  #[error("format error: {0}")]
  Format(String),
  #[error(transparent)]
  Geom(#[from] getfem_geom::Error),
  #[error(transparent)]
  Fem(#[from] getfem_fem::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
