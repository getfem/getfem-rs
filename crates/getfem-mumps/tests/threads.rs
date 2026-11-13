//! MUMPS' threads reach its BLAS (process-wide: a test binary of its own).

use getfem_linalg::{Coo, LinearSolver, MatrixKind};
use getfem_mumps::{Mumps, blas_info, set_num_threads};

#[test]
fn threads_reach_the_blas() {
  let n = 50;
  let mut c = Coo::new(n, n);
  for i in 0..n {
    c.push(i, i, 4.0);
    if i > 0 {
      c.push(i, i - 1, -1.0);
      c.push(i - 1, i, -1.0);
    }
  }
  let a = c.to_csr().unwrap();
  for t in [3, 1] {
    set_num_threads(t).unwrap();
    let mut b = vec![1.0; n];
    Mumps::new()
      .solve_system(&a, MatrixKind::General, &mut b)
      .unwrap();
    let i = blas_info();
    println!("{t}: {i}");
    assert!(i.path.is_some(), "{i}");
    if i.name.is_some() && i.parallel != Some("serial") {
      assert_eq!(i.num_threads, Some(t), "{i}");
    }
  }
}
