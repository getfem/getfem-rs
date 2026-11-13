#[derive(Debug, thiserror::Error)]
pub enum Error {
  #[error("invalid argument: {0}")]
  InvalidArgument(String),
  #[error("singular matrix{0}")]
  Singular(String),
  #[error("{solver} failed: {msg}")]
  Solver {
    solver: &'static str,
    code: i64,
    msg: String,
  },
}

pub type Result<T> = std::result::Result<T, Error>;
