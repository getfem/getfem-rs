//! Small dense tensors, column-major (first index fastest), as GetFEM's
//! base_tensor. A leading batch index is therefore the contiguous one (SoA),
//! which is what batched kernels use.

use std::ops::{Index, IndexMut};

use crate::{Error, Real, Result, Scalar};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tensor<T> {
  shape: Vec<usize>,
  data: Vec<T>,
}

pub type Mat = Tensor<f64>;

impl<T: Scalar> Tensor<T> {
  pub fn zeros(shape: &[usize]) -> Self {
    Tensor {
      shape: shape.to_vec(),
      data: vec![T::zero(); shape.iter().product()],
    }
  }

  pub fn from_data(shape: &[usize], data: Vec<T>) -> Result<Self> {
    if shape.iter().product::<usize>() != data.len() {
      return Err(Error::InvalidArgument(format!(
        "tensor shape {shape:?} vs {} values",
        data.len()
      )));
    }
    Ok(Tensor {
      shape: shape.to_vec(),
      data,
    })
  }

  /// Matrix from rows (row-major input), stored column-major.
  pub fn from_rows(rows: &[&[T]]) -> Self {
    let (m, n) = (rows.len(), rows.first().map_or(0, |r| r.len()));
    let mut a = Self::zeros(&[m, n]);
    for (i, r) in rows.iter().enumerate() {
      assert_eq!(r.len(), n, "ragged rows");
      for (j, &v) in r.iter().enumerate() {
        a[(i, j)] = v;
      }
    }
    a
  }

  pub fn identity(n: usize) -> Self {
    let mut a = Self::zeros(&[n, n]);
    for i in 0..n {
      a[(i, i)] = T::one();
    }
    a
  }

  pub fn shape(&self) -> &[usize] {
    &self.shape
  }

  /// Refill with a new shape, reusing the storage.
  pub fn refill(&mut self, shape: &[usize], data: impl IntoIterator<Item = T>) {
    self.shape.clear();
    self.shape.extend_from_slice(shape);
    self.data.clear();
    self.data.reserve(shape.iter().product());
    self.data.extend(data);
    assert_eq!(
      self.data.len(),
      shape.iter().product::<usize>(),
      "refill: shape {shape:?}"
    );
  }

  pub fn data(&self) -> &[T] {
    &self.data
  }

  pub fn data_mut(&mut self) -> &mut [T] {
    &mut self.data
  }

  pub fn len(&self) -> usize {
    self.data.len()
  }

  pub fn is_empty(&self) -> bool {
    self.data.is_empty()
  }

  pub fn nrows(&self) -> usize {
    self.shape[0]
  }

  pub fn ncols(&self) -> usize {
    self.shape.get(1).copied().unwrap_or(1)
  }

  pub fn col(&self, j: usize) -> &[T] {
    let m = self.nrows();
    &self.data[j * m..(j + 1) * m]
  }

  pub fn col_mut(&mut self, j: usize) -> &mut [T] {
    let m = self.nrows();
    &mut self.data[j * m..(j + 1) * m]
  }

  pub fn transposed(&self) -> Self {
    let (m, n) = (self.nrows(), self.ncols());
    let mut t = Self::zeros(&[n, m]);
    for j in 0..n {
      for i in 0..m {
        t[(j, i)] = self[(i, j)];
      }
    }
    t
  }

  /// self * b
  pub fn matmul(&self, b: &Self) -> Self {
    let (m, k, n) = (self.nrows(), self.ncols(), b.ncols());
    assert_eq!(k, b.nrows(), "matmul: {:?} x {:?}", self.shape, b.shape);
    let mut c = Self::zeros(&[m, n]);
    if m == 0 {
      return c;
    }
    // column j of c += column l of self * b(l, j), l in order
    for (cj, bj) in c
      .data
      .chunks_exact_mut(m)
      .zip(b.data.chunks_exact(k.max(1)))
    {
      for (al, &blj) in self.data.chunks_exact(m).zip(bj) {
        for (ci, &ai) in cj.iter_mut().zip(al) {
          *ci += ai * blj;
        }
      }
    }
    c
  }

  /// self^T * b
  pub fn tmatmul(&self, b: &Self) -> Self {
    let (k, m, n) = (self.nrows(), self.ncols(), b.ncols());
    assert_eq!(k, b.nrows(), "tmatmul: {:?} x {:?}", self.shape, b.shape);
    let mut c = Self::zeros(&[m, n]);
    for j in 0..n {
      for i in 0..m {
        c[(i, j)] = (0..k)
          .map(|l| self.data[l + i * k] * b.data[l + j * k])
          .sum();
      }
    }
    c
  }

  pub fn scale(&mut self, a: T) {
    self.data.iter_mut().for_each(|x| *x *= a);
  }
}

impl<T> Index<(usize, usize)> for Tensor<T> {
  type Output = T;
  fn index(&self, (i, j): (usize, usize)) -> &T {
    &self.data[i + j * self.shape[0]]
  }
}

impl<T> IndexMut<(usize, usize)> for Tensor<T> {
  fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut T {
    &mut self.data[i + j * self.shape[0]]
  }
}

/// LU factorization with partial pivoting of a square matrix.
pub struct Lu<T> {
  lu: Tensor<T>,
  piv: Vec<usize>,
  sign: bool,
}

impl<T: Scalar> Lu<T> {
  /// Fails on an exactly singular matrix (zero pivot).
  pub fn new(a: &Tensor<T>) -> Result<Self> {
    let n = a.nrows();
    if a.ncols() != n {
      return Err(Error::InvalidArgument(format!(
        "LU of non-square {:?}",
        a.shape
      )));
    }
    let mut lu = a.clone();
    let mut piv: Vec<usize> = (0..n).collect();
    let mut sign = false;
    for j in 0..n {
      let p = (j..n)
        .max_by(|&x, &y| {
          lu[(x, j)]
            .abs()
            .partial_cmp(&lu[(y, j)].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();
      if lu[(p, j)] == T::zero() {
        return Err(Error::Singular(String::new()));
      }
      if p != j {
        for c in 0..n {
          let t = lu[(p, c)];
          lu[(p, c)] = lu[(j, c)];
          lu[(j, c)] = t;
        }
        piv.swap(p, j);
        sign = !sign;
      }
      let d = lu[(j, j)];
      for i in j + 1..n {
        let f = lu[(i, j)] / d;
        lu[(i, j)] = f;
        for c in j + 1..n {
          let v = lu[(j, c)];
          lu[(i, c)] -= f * v;
        }
      }
    }
    Ok(Lu { lu, piv, sign })
  }

  pub fn det(&self) -> T {
    let n = self.lu.nrows();
    let d = (0..n).fold(T::one(), |d, i| d * self.lu[(i, i)]);
    if self.sign { -d } else { d }
  }

  pub fn solve(&self, b: &[T]) -> Vec<T> {
    let n = self.lu.nrows();
    let mut x: Vec<T> = self.piv.iter().map(|&p| b[p]).collect();
    for i in 0..n {
      for k in 0..i {
        let v = self.lu[(i, k)] * x[k];
        x[i] -= v;
      }
    }
    for i in (0..n).rev() {
      for k in i + 1..n {
        let v = self.lu[(i, k)] * x[k];
        x[i] -= v;
      }
      x[i] /= self.lu[(i, i)];
    }
    x
  }

  pub fn inverse(&self) -> Tensor<T> {
    let n = self.lu.nrows();
    let mut inv = Tensor::zeros(&[n, n]);
    let mut e = vec![T::zero(); n];
    for j in 0..n {
      e[j] = T::one();
      inv.col_mut(j).copy_from_slice(&self.solve(&e));
      e[j] = T::zero();
    }
    inv
  }
}

pub fn det<T: Scalar>(a: &Tensor<T>) -> Result<T> {
  match Lu::new(a) {
    Ok(lu) => Ok(lu.det()),
    Err(Error::Singular(_)) => Ok(T::zero()),
    Err(e) => Err(e),
  }
}

pub fn inverse<T: Scalar>(a: &Tensor<T>) -> Result<Tensor<T>> {
  Ok(Lu::new(a)?.inverse())
}

/// Eigenvalues of a small symmetric real matrix (cyclic Jacobi), ascending.
pub fn sym_eigenvalues<R: Real>(a: &Tensor<R>) -> Vec<R> {
  let n = a.nrows();
  let mut m = a.clone();
  let two = R::from_f64(2.0);
  for _ in 0..100 {
    let off: R = (0..n)
      .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
      .map(|(i, j)| m[(i, j)] * m[(i, j)])
      .sum();
    let diag: R = (0..n).map(|i| m[(i, i)] * m[(i, i)]).sum();
    if off <= R::epsilon() * R::epsilon() * diag || off == R::zero() {
      break;
    }
    for p in 0..n {
      for q in p + 1..n {
        if m[(p, q)] == R::zero() {
          continue;
        }
        // rotation angle from atan2, as per project rules
        let theta = (two * m[(p, q)]).atan2(m[(q, q)] - m[(p, p)]) / two;
        let (s, c) = sin_cos(theta);
        for k in 0..n {
          let (mkp, mkq) = (m[(k, p)], m[(k, q)]);
          m[(k, p)] = c * mkp - s * mkq;
          m[(k, q)] = s * mkp + c * mkq;
        }
        for k in 0..n {
          let (mpk, mqk) = (m[(p, k)], m[(q, k)]);
          m[(p, k)] = c * mpk - s * mqk;
          m[(q, k)] = s * mpk + c * mqk;
        }
      }
    }
  }
  let mut ev: Vec<R> = (0..n).map(|i| m[(i, i)]).collect();
  ev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
  ev
}

fn sin_cos<R: Real>(t: R) -> (R, R) {
  let (s, c) = t.to_f64().sin_cos();
  (R::from_f64(s), R::from_f64(c))
}

/// Extreme singular values (min, max) of a real matrix.
pub fn singular_value_bounds<R: Real>(a: &Tensor<R>) -> (R, R) {
  let ata = if a.nrows() >= a.ncols() {
    a.tmatmul(a)
  } else {
    a.matmul(&a.transposed())
  };
  let ev = sym_eigenvalues(&ata);
  let clamp = |x: R| if x > R::zero() { x.sqrt() } else { R::zero() };
  (clamp(ev[0]), clamp(*ev.last().unwrap()))
}

#[cfg(test)]
mod tests;
