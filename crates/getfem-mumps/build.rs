// links MUMPS: the sequential one (Debian libmumps-seq-dev: libdmumps_seq
// pulls mumps_common_seq, pord_seq, mpiseq_seq, BLAS, LAPACK), or with the
// feature mpi the MPI one (libmumps-dev: libdmumps). Never both: the
// sequential one defines stub MPI_Init, MPI_Comm_rank, ...
fn main() {
  match std::env::var_os("CARGO_FEATURE_MPI") {
    Some(_) => println!("cargo:rustc-link-lib=dmumps"),
    None => println!("cargo:rustc-link-lib=dmumps_seq"),
  }
}
