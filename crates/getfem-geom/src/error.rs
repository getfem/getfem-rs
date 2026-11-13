#[derive(Debug, thiserror::Error)]
pub enum Error {
  #[error("invalid argument: {0}")]
  InvalidArgument(String),
  #[error("parse error: {0}")]
  Parse(String),
  #[error("singular matrix")]
  Singular,
  #[error("not implemented: {0}")]
  NotImplemented(&'static str),
  #[error(transparent)]
  Linalg(getfem_linalg::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// the dense algebra's errors as the geometry's own (Singular stops the
/// inversion of a geometric transformation)
impl From<getfem_linalg::Error> for Error {
  fn from(e: getfem_linalg::Error) -> Self {
    match e {
      getfem_linalg::Error::Singular(_) => Error::Singular,
      getfem_linalg::Error::InvalidArgument(m) => Error::InvalidArgument(m),
      e => Error::Linalg(e),
    }
  }
}
