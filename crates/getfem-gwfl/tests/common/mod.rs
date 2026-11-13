//! Shared by the tests: assembly with threads as with one thread.
#![allow(dead_code)]

use getfem_gwfl::workspace::Workspace;
use getfem_linalg::threads::set_num_threads;

/// The assembled potential, vector or dense matrix of the expressions of ws
pub fn assembled(ws: &mut Workspace, order: usize, condensation: bool) -> Vec<f64> {
  ws.assembly(order, condensation).unwrap();
  match order {
    0 => vec![ws.assembled_potential()],
    1 => ws.assembled_vector().to_vec(),
    _ => {
      let mut a = ws.assembled_matrix().to_csr().to_dense().concat();
      if condensation {
        a.extend(ws.assembled_vector());
        a.extend(ws.internal_coupling_matrix().to_csr().to_dense().concat());
      }
      a
    }
  }
}

/// run (an assembly) on 2, 3, 4, 8, 100 threads: as on one thread to
/// round-off (of the largest value, at least 1: results that vanish),
/// identical from run to run
pub fn as_one_thread(tag: &str, mut run: impl FnMut() -> Vec<f64>) {
  set_num_threads(1).unwrap();
  let one = run();
  let n = one.iter().fold(1.0f64, |a, b| a.max(b.abs()));
  for nt in [2, 3, 4, 8, 100] {
    set_num_threads(nt).unwrap();
    let a = run();
    assert_eq!(a.len(), one.len(), "{tag}: {nt} threads");
    let worst = a
      .iter()
      .zip(&one)
      .fold(0.0f64, |w, (p, q)| w.max((p - q).abs()));
    assert!(
      worst <= 1e-13 * n,
      "{tag}: {nt} threads, difference {worst:e} of {n:e}"
    );
    assert!(
      run() == a,
      "{tag}: {nt} threads, not identical from run to run"
    );
  }
  set_num_threads(1).unwrap();
}

pub mod mixed;
