use super::*;

fn close(a: f64, b: f64) -> bool {
  (a - b).abs() <= 1e-12 * (1.0 + b.abs())
}

#[test]
fn layout_and_products() {
  let a = Mat::from_rows(&[&[1., 2., 3.], &[4., 5., 6.]]);
  assert_eq!(a.data(), &[1., 4., 2., 5., 3., 6.]);
  assert_eq!(a[(1, 2)], 6.);
  let b = a.transposed();
  assert_eq!(b.shape(), &[3, 2]);
  let c = a.matmul(&b);
  assert_eq!(c, Mat::from_rows(&[&[14., 32.], &[32., 77.]]));
  assert_eq!(b.tmatmul(&b), c);
}

#[test]
fn lu_det_inverse() {
  let m = Mat::from_rows(&[
    &[2., -1., 0., 3.],
    &[1., 4., 1., 0.],
    &[0., 2., 5., 1.],
    &[1., 0., 1., 3.],
  ]);
  assert!(close(det(&m).unwrap(), 55.000000000000014));
  let inv = inverse(&m).unwrap();
  let id = m.matmul(&inv);
  for i in 0..4 {
    for j in 0..4 {
      assert!((id[(i, j)] - if i == j { 1. } else { 0. }).abs() < 1e-14);
    }
  }
  let s = Mat::from_rows(&[&[1., 2.], &[2., 4.]]);
  assert!(matches!(inverse(&s), Err(Error::Singular(_))));
  assert_eq!(det(&s).unwrap(), 0.0);
  // zero at the last pivot only; non-square: refused
  let s3 = Mat::from_rows(&[&[1., 0., 1.], &[0., 1., 1.], &[1., 1., 2.]]);
  assert!(matches!(Lu::new(&s3), Err(Error::Singular(_))));
  let r = Mat::from_rows(&[&[1., 2.], &[3., 4.], &[5., 6.]]);
  assert!(matches!(Lu::new(&r), Err(Error::InvalidArgument(_))));
}

#[test]
fn eigen_and_singular() {
  let a = Mat::from_rows(&[&[4., 1., 2.], &[1., 3., 0.5], &[2., 0.5, 5.]]);
  let ev = sym_eigenvalues(&a);
  for (x, y) in ev
    .iter()
    .zip([2.096521617296718, 3.0722239520050754, 6.831254430698204])
  {
    assert!(close(*x, y), "{ev:?}");
  }
  let k = Mat::from_rows(&[&[1., 2.], &[0.5, -1.], &[3., 0.25]]);
  let (lo, hi) = singular_value_bounds(&k);
  assert!(close(lo, 2.054893517715548) && close(hi, 3.3301520432032254));
  let (lo2, hi2) = singular_value_bounds(&k.transposed());
  assert!(close(lo2, lo) && close(hi2, hi));
}
