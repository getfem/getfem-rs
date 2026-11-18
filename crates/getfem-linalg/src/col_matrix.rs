//! gmm's `col_matrix<rsvector>`: the sparse matrix GetFEM assembles into
//! (model_real_sparse_matrix), columns of sorted (row, value) pairs, and
//! GetFEM's `add_elem_matrix` (getfem_generic_assembly_compile_and_exec.cc).

use crate::Scalar;

use crate::Csr;

/// (index, value) pairs sorted by index
pub type RsVector<T> = Vec<(usize, T)>;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ColMatrix<T> {
  pub nrows: usize,
  pub ncols: usize,
  pub cols: Vec<RsVector<T>>,
}

impl<T: Scalar> ColMatrix<T> {
  pub fn new(nrows: usize, ncols: usize) -> Self {
    ColMatrix {
      nrows,
      ncols,
      cols: vec![vec![]; ncols],
    }
  }

  /// The new size; the entries of the kept rows and columns kept
  pub fn resize(&mut self, nrows: usize, ncols: usize) {
    self.cols.resize_with(ncols, Vec::new);
    if nrows < self.nrows {
      self.cols.iter_mut().for_each(|c| c.retain(|e| e.0 < nrows));
    }
    (self.nrows, self.ncols) = (nrows, ncols);
  }

  /// No entries, the storage of the columns kept for the next assembly
  pub fn clear(&mut self) {
    self.cols.iter_mut().for_each(Vec::clear);
  }

  pub fn nnz(&self) -> usize {
    self.cols.iter().map(|c| c.len()).sum()
  }

  pub fn get(&self, i: usize, j: usize) -> T {
    let c = &self.cols[j];
    c.binary_search_by_key(&i, |e| e.0)
      .map_or(T::zero(), |p| c[p].1)
  }

  /// a_ij += v (inserted in its column if absent)
  pub fn add_entry(&mut self, i: usize, j: usize, v: T) {
    let c = &mut self.cols[j];
    match c.binary_search_by_key(&i, |e| e.0) {
      Ok(p) => c[p].1 += v,
      Err(p) => c.insert(p, (i, v)),
    }
  }

  /// elem (column-major, rows dofs1, columns dofs2) added column by
  /// column, the rows sorted once (insertion sort) and merged into each
  /// column; entries with abs value <= threshold are dropped, and None DOFs
  /// (not in a partial space) skipped. n: the dimension, for the reserve
  /// of a new column as GetFEM. rows: work, the sorted (row, position in
  /// elem) pairs.
  pub fn add_elem_matrix(
    &mut self,
    dofs1: &[Option<usize>],
    dofs2: &[Option<usize>],
    elem: &[T],
    threshold: T::Real,
    n: usize,
    rows: &mut Vec<(usize, usize)>,
  ) {
    rows.clear();
    for (k, d) in dofs1.iter().enumerate() {
      let Some(d) = *d else { continue };
      let mut j = rows.len();
      rows.push((d, k));
      while j > 0 && d < rows[j - 1].0 {
        rows[j] = rows[j - 1];
        j -= 1;
      }
      rows[j] = (d, k);
    }
    let s1 = dofs1.len();
    if s1 == 0 {
      return;
    }
    let maxest = (n + 1) * s1;
    for (e, dof2) in elem.chunks_exact(s1).zip(dofs2) {
      let Some(dof2) = *dof2 else { continue };
      let col = &mut self.cols[dof2];
      if col.is_empty() {
        col.reserve(maxest);
        let kept = rows.iter().filter(|&&(_, k)| e[k].abs() > threshold);
        col.extend(kept.map(|&(c, k)| (c, e[k])));
        continue;
      }
      // column merge: the rows come in increasing order
      let mut ind = 0;
      for &(c, k) in rows.iter() {
        let v = e[k];
        if v.abs() <= threshold {
          continue;
        }
        ind = lower_bound(col, ind, c);
        if ind < col.len() && col[ind].0 == c {
          col[ind].1 += v;
        } else {
          col.insert(ind, (c, v));
        }
        ind += 1;
      }
    }
  }

  /// column j += entries (rows sorted, distinct): a merge
  pub fn add_to_column(&mut self, j: usize, entries: &[(usize, T)]) {
    add_to_rsvector(&mut self.cols[j], entries);
  }

  /// self += b, column by column merges
  pub fn add(&mut self, b: &ColMatrix<T>) {
    for (j, bc) in b.cols.iter().enumerate() {
      if !bc.is_empty() {
        self.add_to_column(j, bc);
      }
    }
  }

  /// y = A x
  pub fn mult(&self, x: &[T]) -> Vec<T> {
    let mut y = vec![T::zero(); self.nrows];
    for (c, &xj) in self.cols.iter().zip(x) {
      for &(i, v) in c {
        y[i] += v * xj;
      }
    }
    y
  }

  /// the same matrix by rows (columns sorted in each row)
  pub fn to_csr(&self) -> Csr<T> {
    let mut ptr = vec![0usize; self.nrows + 1];
    for c in &self.cols {
      for &(i, _) in c {
        ptr[i + 1] += 1;
      }
    }
    for i in 0..self.nrows {
      ptr[i + 1] += ptr[i];
    }
    let nnz = ptr[self.nrows];
    let mut pos = ptr.clone();
    let (mut idx, mut val) = (vec![0usize; nnz], vec![T::zero(); nnz]);
    for (j, c) in self.cols.iter().enumerate() {
      for &(i, v) in c {
        let p = &mut pos[i];
        idx[*p] = j;
        val[*p] = v;
        *p += 1;
      }
    }
    Csr {
      nrows: self.nrows,
      ncols: self.ncols,
      ptr,
      idx,
      val,
    }
  }
}

/// first position >= ind of a row >= c in col, as GetFEM's
/// add_elem_matrix (branches, not the branch-free partition_point: the
/// branches are predicted mostly, the chain of dependent loads of a
/// branch-free search costs twice the time)
#[inline]
fn lower_bound<T>(col: &[(usize, T)], mut ind: usize, c: usize) -> usize {
  let mut count = col.len() - ind;
  while count > 0 {
    let step = count / 2;
    let l = ind + step;
    if col[l].0 < c {
      ind = l + 1;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  ind
}

/// a += entries (rows sorted, distinct; gmm::add of sparse vectors): a
/// merge in place, from the back (no allocation within the capacity of a)
pub fn add_to_rsvector<T: Scalar>(a: &mut RsVector<T>, entries: &[(usize, T)]) {
  if a.is_empty() {
    a.extend_from_slice(entries);
    return;
  }
  // the rows of entries not in a
  let mut p = 0;
  let new = entries
    .iter()
    .filter(|&&(i, _)| {
      while p < a.len() && a[p].0 < i {
        p += 1;
      }
      p == a.len() || a[p].0 != i
    })
    .count();
  // from the back: a[..p] and entries[..q] still to merge, written at w;
  // done when entries are (a[..p] already in place)
  let (mut p, mut q) = (a.len(), entries.len());
  a.resize(p + new, (0, T::zero()));
  for w in (0..a.len()).rev() {
    if q == 0 {
      break;
    }
    let e = entries[q - 1];
    a[w] = if p > 0 && a[p - 1].0 > e.0 {
      p -= 1;
      a[p]
    } else if p > 0 && a[p - 1].0 == e.0 {
      p -= 1;
      q -= 1;
      (e.0, a[p].1 + e.1)
    } else {
      q -= 1;
      e
    };
  }
}
