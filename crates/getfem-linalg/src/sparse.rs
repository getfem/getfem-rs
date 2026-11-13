//! Sparse matrices: COO for assembly, CSR for computation.

use crate::Scalar;

use crate::{Error, Result};

/// Triplets; duplicates are summed on conversion.
#[derive(Clone, Debug)]
pub struct Coo<T> {
  pub nrows: usize,
  pub ncols: usize,
  pub rows: Vec<usize>,
  pub cols: Vec<usize>,
  pub vals: Vec<T>,
}

impl<T: Scalar> Coo<T> {
  pub fn new(nrows: usize, ncols: usize) -> Self {
    Coo {
      nrows,
      ncols,
      rows: vec![],
      cols: vec![],
      vals: vec![],
    }
  }

  /// Out-of-range entries are reported by to_csr.
  pub fn push(&mut self, i: usize, j: usize, v: T) {
    self.rows.push(i);
    self.cols.push(j);
    self.vals.push(v);
  }

  /// Add a dense element block m (row-major rows x cols) at global indices.
  pub fn add_block(&mut self, rows: &[usize], cols: &[usize], m: &[T]) {
    for (a, &i) in rows.iter().enumerate() {
      for (b, &j) in cols.iter().enumerate() {
        self.push(i, j, m[a * cols.len() + b]);
      }
    }
  }

  /// CSR with sorted columns, duplicates summed (explicit zeros kept, so
  /// the pattern does not depend on values).
  pub fn to_csr(&self) -> Result<Csr<T>> {
    if let Some(k) =
      (0..self.vals.len()).find(|&k| self.rows[k] >= self.nrows || self.cols[k] >= self.ncols)
    {
      return Err(Error::InvalidArgument(format!(
        "entry ({}, {}) outside {}x{}",
        self.rows[k], self.cols[k], self.nrows, self.ncols
      )));
    }
    let nnz = self.vals.len();
    // stable counting sorts by column, then by row: columns ascending in
    // each row, duplicates in entry order
    let start = |keys: &mut dyn Iterator<Item = usize>, n: usize| {
      let mut c = vec![0usize; n + 1];
      for k in keys {
        c[k + 1] += 1;
      }
      for i in 0..n {
        c[i + 1] += c[i];
      }
      c
    };
    let mut pos = start(&mut self.cols.iter().copied(), self.ncols);
    let mut by_col = vec![0usize; nnz];
    for k in 0..nnz {
      let p = &mut pos[self.cols[k]];
      by_col[*p] = k;
      *p += 1;
    }
    let count = start(&mut self.rows.iter().copied(), self.nrows);
    let mut pos = count.clone();
    let mut idx = vec![0usize; nnz];
    let mut val = vec![T::zero(); nnz];
    for &k in &by_col {
      let p = &mut pos[self.rows[k]];
      idx[*p] = self.cols[k];
      val[*p] = self.vals[k];
      *p += 1;
    }
    // merge the duplicates (consecutive), in place
    let mut ptr = vec![0usize; self.nrows + 1];
    let mut n = 0;
    for i in 0..self.nrows {
      let r0 = n;
      for p in count[i]..count[i + 1] {
        if n > r0 && idx[n - 1] == idx[p] {
          let v = val[p];
          val[n - 1] += v;
        } else {
          idx[n] = idx[p];
          val[n] = val[p];
          n += 1;
        }
      }
      ptr[i + 1] = n;
    }
    idx.truncate(n);
    val.truncate(n);
    Ok(Csr {
      nrows: self.nrows,
      ncols: self.ncols,
      ptr,
      idx,
      val,
    })
  }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Csr<T> {
  pub nrows: usize,
  pub ncols: usize,
  pub ptr: Vec<usize>,
  pub idx: Vec<usize>,
  pub val: Vec<T>,
}

impl<T: Scalar> Csr<T> {
  pub fn nnz(&self) -> usize {
    self.val.len()
  }

  pub fn row(&self, i: usize) -> impl Iterator<Item = (usize, T)> + '_ {
    (self.ptr[i]..self.ptr[i + 1]).map(move |p| (self.idx[p], self.val[p]))
  }

  pub fn get(&self, i: usize, j: usize) -> T {
    let r = &self.idx[self.ptr[i]..self.ptr[i + 1]];
    r.binary_search(&j)
      .map_or(T::zero(), |p| self.val[self.ptr[i] + p])
  }

  pub fn matvec(&self, x: &[T]) -> Result<Vec<T>> {
    if x.len() != self.ncols {
      return Err(Error::InvalidArgument(format!(
        "matvec: vector of size {} for {} columns",
        x.len(),
        self.ncols
      )));
    }
    Ok(
      (0..self.nrows)
        .map(|i| self.row(i).map(|(j, v)| v * x[j]).sum())
        .collect(),
    )
  }

  /// self b (gmm::mult of sparse matrices; rows by sparse accumulation)
  pub fn mul(&self, b: &Csr<T>) -> Result<Csr<T>> {
    if self.ncols != b.nrows {
      return Err(Error::InvalidArgument(format!(
        "mul: {} x {} times {} x {}",
        self.nrows, self.ncols, b.nrows, b.ncols
      )));
    }
    let mut acc = vec![T::zero(); b.ncols];
    let mut used = vec![false; b.ncols];
    let mut cols: Vec<usize> = vec![];
    let (mut ptr, mut idx, mut val) = (vec![0], vec![], vec![]);
    for i in 0..self.nrows {
      for (k, a) in self.row(i) {
        for (j, v) in b.row(k) {
          if !used[j] {
            used[j] = true;
            cols.push(j);
          }
          acc[j] += a * v;
        }
      }
      cols.sort_unstable();
      for &j in &cols {
        idx.push(j);
        val.push(acc[j]);
        acc[j] = T::zero();
        used[j] = false;
      }
      cols.clear();
      ptr.push(idx.len());
    }
    Ok(Csr {
      nrows: self.nrows,
      ncols: b.ncols,
      ptr,
      idx,
      val,
    })
  }

  pub fn transpose(&self) -> Csr<T> {
    let mut c = Coo::new(self.ncols, self.nrows);
    for i in 0..self.nrows {
      for (j, v) in self.row(i) {
        c.push(j, i, v);
      }
    }
    c.to_csr().expect("in range")
  }

  /// Largest |a_ij - a_ji| relative to the largest |a_ij|.
  pub fn asymmetry(&self) -> <T as Scalar>::Real {
    let t = self.transpose();
    let (mut d, mut m) = (<T::Real as Scalar>::zero(), <T::Real as Scalar>::zero());
    for i in 0..self.nrows {
      for (j, v) in self.row(i) {
        let a = (v - t.get(i, j)).abs();
        d = if a > d { a } else { d };
        let b = v.abs();
        m = if b > m { b } else { m };
      }
    }
    if m == <T::Real as Scalar>::zero() {
      m
    } else {
      d / m
    }
  }

  pub fn to_dense(&self) -> Vec<Vec<T>> {
    let mut d = vec![vec![T::zero(); self.ncols]; self.nrows];
    for (i, r) in d.iter_mut().enumerate() {
      for (j, v) in self.row(i) {
        r[j] = v;
      }
    }
    d
  }
}
