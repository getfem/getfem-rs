//! gf.Mesh("structured" | "structured_ball" | "structured_ball_shell",
//! param) (GetFEM: "import" first): GetFEM's parameter strings
//! ("GT='GT_QK(2,1)';ORG=[0,0];SIZES=[1,1];NSUBDIV=[4,4];NOISED=0"),
//! GetFEM's defaults (getfem_regular_meshes.cc)

use std::collections::BTreeMap;

use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::mesh::Mesh;
use getfem::geom::regular_mesh::{
  noise_unit_mesh, regular_ball_mesh, regular_ball_shell_mesh, regular_unit_mesh,
};
use getfem::linalg::tensor::Mat;
use pyo3::PyResult;
use pyo3::exceptions::PyValueError;

use crate::error::ToPy;

enum Value {
  Str(String),
  Num(f64),
  Array(Vec<f64>),
}

fn err(m: String) -> pyo3::PyErr {
  PyValueError::new_err(m)
}

/// KEY=VALUE entries separated by ';' or new lines: a quoted string, a
/// number or an array of numbers
fn parse(param: &str) -> PyResult<BTreeMap<String, Value>> {
  let mut out = BTreeMap::new();
  for e in param
    .split([';', '\n'])
    .map(str::trim)
    .filter(|e| !e.is_empty())
  {
    let (k, v) = e
      .split_once('=')
      .ok_or_else(|| err(format!("structured mesh: \"{e}\" is not KEY=VALUE")))?;
    let v = v.trim();
    let num = |s: &str| {
      s.trim()
        .parse::<f64>()
        .map_err(|_| err(format!("structured mesh: \"{s}\" is not a number")))
    };
    let value = if let Some(s) = v
      .strip_prefix('\'')
      .and_then(|s| s.strip_suffix('\''))
      .or_else(|| v.strip_prefix('"').and_then(|s| s.strip_suffix('"')))
    {
      Value::Str(s.to_string())
    } else if let Some(a) = v.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
      Value::Array(a.split(',').map(num).collect::<PyResult<_>>()?)
    } else {
      Value::Num(num(v)?)
    };
    out.insert(k.trim().to_string(), value);
  }
  Ok(out)
}

/// the parameters of one kind of structured mesh
struct Params(BTreeMap<String, Value>);

impl Params {
  fn new(param: &str, keys: &[&str]) -> PyResult<Self> {
    let p = parse(param)?;
    if let Some(k) = p.keys().find(|k| !keys.contains(&k.as_str())) {
      return Err(err(format!(
        "structured mesh: unknown parameter {k} (parameters: {})",
        keys.join(", ")
      )));
    }
    Ok(Params(p))
  }

  fn string(&self, k: &str) -> PyResult<&str> {
    match self.0.get(k) {
      Some(Value::Str(s)) => Ok(s),
      _ => Err(err(format!("structured mesh: {k} must be a quoted string"))),
    }
  }

  /// an array of n numbers (default d)
  fn array(&self, k: &str, n: usize, d: &[f64]) -> PyResult<Vec<f64>> {
    match self.0.get(k) {
      None => Ok(d.to_vec()),
      Some(Value::Array(a)) if a.len() == n => Ok(a.clone()),
      _ => Err(err(format!(
        "structured mesh: {k} must be an array of {n} numbers"
      ))),
    }
  }

  /// subdivisions: numbers rounded (GetFEM: + 0.5)
  fn counts(&self, k: &str, n: usize, d: &[usize]) -> PyResult<Vec<usize>> {
    let d: Vec<f64> = d.iter().map(|&x| x as f64).collect();
    Ok(
      self
        .array(k, n, &d)?
        .iter()
        .map(|&x| (x + 0.5) as usize)
        .collect(),
    )
  }

  fn int(&self, k: &str) -> PyResult<usize> {
    match self.0.get(k) {
      None => Ok(0),
      Some(Value::Num(x)) if *x >= 0.0 => Ok(*x as usize),
      _ => Err(err(format!("structured mesh: {k} must be a number >= 0"))),
    }
  }
}

pub fn structured(kind: &str, param: &str) -> PyResult<Mesh> {
  const BOX: &[&str] = &["GT", "ORG", "SIZES", "NSUBDIV", "NOISED"];
  const BALL: &[&str] = &["GT", "ORG", "SIZES", "NSUBDIV", "NOISED", "SYMMETRIES"];
  let keys = match kind {
    "structured" => BOX,
    _ => BALL,
  };
  let p = Params::new(param, keys)?;
  let pgt = geotrans_descriptor(p.string("GT")?).py()?;
  let n = pgt.dim();
  // structured_ball_shell: an ORG of dimension n + 1 gives the sphere
  let surface = kind == "structured_ball_shell"
    && matches!(p.0.get("ORG"), Some(Value::Array(a)) if a.len() == n + 1);
  let org = if surface {
    p.array("ORG", n + 1, &[])?
  } else {
    p.array("ORG", n, &vec![0.0; n])?
  };
  let noised = p.int("NOISED")? != 0;
  match kind {
    "structured" => {
      let sizes = p.array("SIZES", n, &vec![1.0; n])?;
      let m = regular_unit_mesh(&p.counts("NSUBDIV", n, &vec![2; n])?, &pgt).py()?;
      if noised {
        noise_unit_mesh(&m).py()?;
      }
      let mut d = Mat::zeros(&[n, n]);
      for (i, s) in sizes.iter().enumerate() {
        d[(i, i)] = *s;
      }
      m.transform(&d).py()?;
      m.translate(&org).py()?;
      Ok(m)
    }
    "structured_ball" => {
      let r = p.array("SIZES", 1, &[1.0])?[0];
      let s = p.counts("NSUBDIV", 2, &[2, 2])?;
      let sym = p.int("SYMMETRIES")?;
      regular_ball_mesh(&pgt, &org, r, [s[0], s[1]], sym, noised).py()
    }
    _ if surface => {
      let r = p.array("SIZES", 1, &[1.0])?[0];
      let s = p.counts("NSUBDIV", 1, &[3])?;
      let sym = p.int("SYMMETRIES")?;
      regular_ball_shell_mesh(&pgt, &org, r, 0.0, [s[0], 1], sym, noised).py()
    }
    _ => {
      let rt = p.array("SIZES", 2, &[1.0, 0.5])?;
      let s = p.counts("NSUBDIV", 2, &[3, 2])?;
      let sym = p.int("SYMMETRIES")?;
      regular_ball_shell_mesh(&pgt, &org, rt[0], rt[1], [s[0], s[1]], sym, noised).py()
    }
  }
}
