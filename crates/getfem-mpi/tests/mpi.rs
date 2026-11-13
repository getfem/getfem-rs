//! init, sum, bcast on any number of ranks (ci/mpi.sh runs it under mpirun);
//! process-wide MPI: a test binary of its own, one test with MPI calls.

#[test]
fn init_sum_bcast() {
  let mpi = getfem_mpi::init().unwrap();
  let (r, n) = (getfem_mpi::rank(), getfem_mpi::size());
  assert!(r < n);
  assert_eq!(getfem_mpi::is_master(), r == 0);
  let mut v = [r as f64, 1.0];
  getfem_mpi::sum(&mut v);
  assert_eq!(v, [(n * (n - 1) / 2) as f64, n as f64]);
  let mut m = [r as f64, -(r as f64)];
  getfem_mpi::max(&mut m);
  assert_eq!(m, [(n - 1) as f64, 0.0]);
  // rank r gives r + 1 values r
  let g = getfem_mpi::all_gather(&vec![r as f64; r + 1]);
  let want: Vec<f64> = (0..n).flat_map(|q| vec![q as f64; q + 1]).collect();
  assert_eq!(g, want);
  let mut w = [r as f64; 3];
  getfem_mpi::bcast(&mut w, n - 1);
  assert_eq!(w, [(n - 1) as f64; 3]);
  drop(mpi);
  assert_eq!(getfem_mpi::size(), 1);
}

/// a 12 x 12 grid into 4 parts: balanced, every part used (METIS), or all
/// 0 without the feature mpi
#[test]
fn partition_grid() {
  let m = 12;
  let (mut xadj, mut adjncy) = (vec![0], vec![]);
  for j in 0..m {
    for i in 0..m {
      let v = i + m * j;
      if i > 0 {
        adjncy.push(v - 1);
      }
      if i + 1 < m {
        adjncy.push(v + 1);
      }
      if j > 0 {
        adjncy.push(v - m);
      }
      if j + 1 < m {
        adjncy.push(v + m);
      }
      xadj.push(adjncy.len());
    }
  }
  let part = getfem_mpi::partition(&xadj, &adjncy, 4);
  assert_eq!(part.len(), m * m);
  let mut count = [0; 4];
  part.iter().for_each(|&p| count[p] += 1);
  if cfg!(feature = "mpi") {
    assert!(count.iter().all(|&c| (32..=40).contains(&c)), "{count:?}");
  } else {
    assert_eq!(count, [m * m, 0, 0, 0]);
  }
  assert_eq!(getfem_mpi::partition(&xadj, &adjncy, 1), vec![0; m * m]);
}
