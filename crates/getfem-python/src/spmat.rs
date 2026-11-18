//! gf.Spmat (GetFEM's sparse matrix): md.tangent_matrix(), a read-only
//! view of the model's current tangent, no copy; or a matrix of its own
//! (gf.Spmat("empty", m, n), asm_generic of order 2)

use std::sync::RwLock;

use getfem::linalg::{ColMatrix, Csr, SparseMatrix};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

use crate::model::Model;

pub(crate) enum Inner {
  Tangent(Py<Model>),
  Own(RwLock<ColMatrix<f64>>),
}

#[pyclass(module = "getfem_rs", frozen)]
pub struct Spmat(pub(crate) Inner);

impl Spmat {
  pub(crate) fn own(k: ColMatrix<f64>) -> Self {
    Spmat(Inner::Own(RwLock::new(k)))
  }

  /// the matrix viewed, borrowed from the model or its own
  pub(crate) fn with<R>(&self, py: Python<'_>, f: impl FnOnce(&ColMatrix<f64>) -> R) -> R {
    match &self.0 {
      Inner::Tangent(md) => f(md.bind(py).borrow().0.tangent_matrix()),
      Inner::Own(k) => f(&k.read().unwrap_or_else(|e| e.into_inner())),
    }
  }
}

/// a CSR matrix by columns
pub(crate) fn from_csr(a: &Csr<f64>) -> ColMatrix<f64> {
  let mut k = ColMatrix::new(a.nrows, a.ncols);
  for i in 0..a.nrows {
    for (j, v) in a.row(i) {
      k.cols[j].push((i, v));
    }
  }
  k
}

/// K(I, J): the rows I, the columns J
pub(crate) fn sub_matrix(
  k: &ColMatrix<f64>,
  ii: &[usize],
  jj: &[usize],
) -> PyResult<ColMatrix<f64>> {
  // the new row of each old one
  let mut row = vec![usize::MAX; k.nrows];
  for (new, &i) in ii.iter().enumerate() {
    if i >= k.nrows || row[i] != usize::MAX {
      return Err(PyIndexError::new_err(format!(
        "gf.Spmat copy: row {i} out of range or repeated ({} rows)",
        k.nrows
      )));
    }
    row[i] = new;
  }
  let mut used = vec![false; k.ncols];
  let mut s = ColMatrix::new(ii.len(), jj.len());
  for (new, &j) in jj.iter().enumerate() {
    if j >= k.ncols || used[j] {
      return Err(PyIndexError::new_err(format!(
        "gf.Spmat copy: column {j} out of range or repeated ({} columns)",
        k.ncols
      )));
    }
    used[j] = true;
    let mut c: Vec<(usize, f64)> = k.cols[j]
      .iter()
      .filter(|e| row[e.0] != usize::MAX)
      .map(|e| (row[e.0], e.1))
      .collect();
    c.sort_unstable_by_key(|e| e.0);
    s.cols[new] = c;
  }
  Ok(s)
}

/// A B, exact zeros dropped
pub(crate) fn mult(a: &ColMatrix<f64>, b: &ColMatrix<f64>) -> PyResult<ColMatrix<f64>> {
  if a.ncols != b.nrows {
    return Err(PyValueError::new_err(format!(
      "gf.Spmat mult: a {} x {} matrix times a {} x {} one",
      a.nrows, a.ncols, b.nrows, b.ncols
    )));
  }
  let mut c = ColMatrix::new(a.nrows, b.ncols);
  // dense accumulator, the rows touched
  let (mut w, mut rows) = (vec![0.0; a.nrows], Vec::new());
  for (cj, bj) in c.cols.iter_mut().zip(&b.cols) {
    for &(k, bkj) in bj {
      for &(i, aik) in &a.cols[k] {
        if w[i] == 0.0 {
          rows.push(i);
        }
        w[i] += aik * bkj;
      }
    }
    rows.sort_unstable();
    rows.dedup();
    for &i in &rows {
      if w[i] != 0.0 {
        cj.push((i, w[i]));
      }
      w[i] = 0.0;
    }
    rows.clear();
  }
  Ok(c)
}

/// A + B, exact zeros dropped
pub(crate) fn add(a: &ColMatrix<f64>, b: &ColMatrix<f64>) -> PyResult<ColMatrix<f64>> {
  if (a.nrows, a.ncols) != (b.nrows, b.ncols) {
    return Err(PyValueError::new_err(format!(
      "gf.Spmat add: a {} x {} matrix plus a {} x {} one",
      a.nrows, a.ncols, b.nrows, b.ncols
    )));
  }
  let mut c = a.clone();
  for (cj, bj) in c.cols.iter_mut().zip(&b.cols) {
    for &(i, v) in bj {
      match cj.binary_search_by_key(&i, |e| e.0) {
        Ok(p) => cj[p].1 += v,
        Err(p) => cj.insert(p, (i, v)),
      }
    }
    cj.retain(|e| e.1 != 0.0);
  }
  Ok(c)
}

/// A'
pub(crate) fn transposed(a: &ColMatrix<f64>) -> ColMatrix<f64> {
  let mut t = ColMatrix::new(a.ncols, a.nrows);
  for (j, c) in a.cols.iter().enumerate() {
    for &(i, v) in c {
      t.cols[i].push((j, v));
    }
  }
  t
}

#[pymethods]
impl Spmat {
  /// gf.Spmat("empty", m[, n]): m x n (m x m), no entries;
  /// gf.Spmat("copy", K[, I[, J]]): a copy of K, or its sub-matrix K(I, J)
  /// (J = I if not given; I, J of distinct indices);
  /// gf.Spmat("mult", A, B): the product A B;
  /// gf.Spmat("add", A, B): the sum A + B;
  /// gf.Spmat("transpose", A): A transposed (not GetFEM's: its transpose
  /// is a method, in place);
  /// gf.Spmat("diag", D): the diagonal matrix of the vector D
  /// (exact zeros not stored)
  #[new]
  #[pyo3(signature = (kind, a, b = None, c = None))]
  fn new(
    py: Python<'_>,
    kind: &str,
    a: &Bound<'_, PyAny>,
    b: Option<&Bound<'_, PyAny>>,
    c: Option<&Bound<'_, PyAny>>,
  ) -> PyResult<Self> {
    match kind {
      "empty" => {
        let m: usize = a.extract()?;
        let n: Option<usize> = b.map(|b| b.extract()).transpose()?;
        Ok(Spmat::own(ColMatrix::new(m, n.unwrap_or(m))))
      }
      "copy" => {
        let src = a.cast::<Spmat>()?.get();
        let ii: Option<Vec<usize>> = b.map(|b| b.extract()).transpose()?;
        let jj: Option<Vec<usize>> = c.map(|c| c.extract()).transpose()?;
        src.with(py, |k| match ii {
          None => Ok(Spmat::own(k.clone())),
          Some(ii) => {
            let jj = jj.as_ref().unwrap_or(&ii);
            Ok(Spmat::own(sub_matrix(k, &ii, jj)?))
          }
        })
      }
      "mult" | "add" => {
        let (a, b) = (
          a.cast::<Spmat>()?.get(),
          b.ok_or_else(|| PyValueError::new_err(format!("gf.Spmat {kind}: two matrices A, B")))?,
        );
        let b = b.cast::<Spmat>()?.get();
        let f = if kind == "mult" { mult } else { add };
        a.with(py, |a| b.with(py, |b| Ok(Spmat::own(f(a, b)?))))
      }
      "transpose" => {
        let a = a.cast::<Spmat>()?.get();
        Ok(Spmat::own(a.with(py, transposed)))
      }
      "diag" => {
        let d: Vec<f64> = a.extract()?;
        let mut k = ColMatrix::new(d.len(), d.len());
        for (j, (c, &v)) in k.cols.iter_mut().zip(&d).enumerate() {
          if v != 0.0 {
            c.push((j, v));
          }
        }
        Ok(Spmat::own(k))
      }
      _ => Err(PyValueError::new_err(format!(
        "gf.Spmat: unknown kind \"{kind}\" (empty, copy, mult, add, transpose, diag)"
      ))),
    }
  }

  /// (rows, columns)
  fn size(&self, py: Python<'_>) -> (usize, usize) {
    self.with(py, |k| (k.nrows(), k.ncols()))
  }

  /// the number of stored entries
  fn nnz(&self, py: Python<'_>) -> usize {
    self.with(py, SparseMatrix::nnz)
  }

  /// a_ij = v (GetFEM's assign, one entry); not on a tangent view
  fn assign(&self, i: usize, j: usize, v: f64) -> PyResult<()> {
    let Inner::Own(k) = &self.0 else {
      return Err(PyValueError::new_err(
        "assign: a tangent matrix view is read-only",
      ));
    };
    let mut k = k.write().unwrap_or_else(|e| e.into_inner());
    if i >= k.nrows || j >= k.ncols {
      return Err(PyIndexError::new_err(format!(
        "assign: ({i}, {j}) out of a {} x {} matrix",
        k.nrows, k.ncols
      )));
    }
    let c = &mut k.cols[j];
    match c.binary_search_by_key(&i, |e| e.0) {
      Ok(p) => c[p].1 = v,
      Err(p) => c.insert(p, (i, v)),
    }
    Ok(())
  }

  /// (JC, IR): the column pointers and row indices of the CSC storage
  fn csc_ind<'py>(
    &self,
    py: Python<'py>,
  ) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    let (jc, ir) = self.with(py, |k| {
      let mut jc = Vec::with_capacity(k.ncols + 1);
      jc.push(0);
      for c in &k.cols {
        jc.push(jc.last().unwrap() + c.len() as i64);
      }
      let ir: Vec<i64> = k.cols.iter().flatten().map(|e| e.0 as i64).collect();
      (jc, ir)
    });
    (PyArray1::from_vec(py, jc), PyArray1::from_vec(py, ir))
  }

  /// the values of the CSC storage, in the order of csc_ind
  fn csc_val<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
    let v = self.with(py, |k| k.cols.iter().flatten().map(|e| e.1).collect());
    PyArray1::from_vec(py, v)
  }

  /// the diagonal; with E, a column per diagonal of rank d in E (entries
  /// (i, i+d) at row i, d < 0 below the diagonal)
  #[pyo3(signature = (e = None))]
  fn diag<'py>(&self, py: Python<'py>, e: Option<Vec<i64>>) -> PyResult<Bound<'py, PyAny>> {
    let ranks = e.clone().unwrap_or(vec![0]);
    let (n, w) = self.with(py, |k| {
      let n = k.nrows.min(k.ncols);
      let mut w = vec![0.0; n * ranks.len()];
      for (c, &d) in ranks.iter().enumerate() {
        for i in 0..n {
          let j = i as i64 + d;
          if j < 0 || j >= k.ncols as i64 {
            continue;
          }
          let col = &k.cols[j as usize];
          if let Ok(p) = col.binary_search_by_key(&i, |x| x.0) {
            w[i * ranks.len() + c] = col[p].1;
          }
        }
      }
      (n, w)
    });
    let w = PyArray1::from_vec(py, w);
    Ok(match e {
      None => w.into_any(),
      Some(r) => w.reshape([n, r.len()])?.into_any(),
    })
  }
}
