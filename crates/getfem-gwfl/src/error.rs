#[derive(Debug, thiserror::Error)]
pub enum Error {
  #[error("{}", caret(.expr, *.pos, .msg))]
  Syntax {
    expr: String,
    pos: usize,
    msg: String,
  },
  #[error("{}", caret(.expr, *.pos, .msg))]
  Semantic {
    expr: String,
    pos: usize,
    msg: String,
  },
  /// valid GWFL whose support is deferred to a later milestone or dropped
  #[error("{}", caret(.expr, *.pos, &format!("unsupported: {}", .msg)))]
  Unsupported {
    expr: String,
    pos: usize,
    msg: String,
  },
  #[error("invalid argument: {0}")]
  InvalidArgument(String),
  #[error(transparent)]
  Fem(#[from] getfem_fem::Error),
  #[error(transparent)]
  Geom(#[from] getfem_geom::Error),
  #[error(transparent)]
  Linalg(#[from] getfem_linalg::Error),
  #[error("internal error: {0}")]
  Internal(String),
}

/// expression, a caret under pos, then the message
fn caret(expr: &str, pos: usize, msg: &str) -> String {
  let pos = pos.min(expr.len());
  let pad: String = expr[..pos]
    .chars()
    .map(|c| if c == '\n' { '\n' } else { ' ' })
    .collect();
  format!("{expr}\n{pad}|\nerror: {msg}")
}

pub type Result<T> = std::result::Result<T, Error>;
