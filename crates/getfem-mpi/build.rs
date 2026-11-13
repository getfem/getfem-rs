// With the feature mpi: the C shim (src/shim.c) compiled and linked by the
// flags of `$MPICC -show` (OpenMPI, MPICH; MPICC default mpicc), and METIS
// (METIS_DIR: its prefix, if not in the compiler's default paths).
use std::process::Command;

fn main() {
  println!("cargo:rerun-if-changed=src/shim.c");
  println!("cargo:rerun-if-env-changed=MPICC");
  println!("cargo:rerun-if-env-changed=METIS_DIR");
  if std::env::var_os("CARGO_FEATURE_MPI").is_none() {
    return;
  }
  let mpicc = std::env::var("MPICC").unwrap_or_else(|_| "mpicc".into());
  let out = Command::new(&mpicc)
    .arg("-show")
    .output()
    .unwrap_or_else(|e| panic!("feature mpi: `{mpicc} -show`: {e} (install an MPI, or set MPICC)"));
  assert!(out.status.success(), "feature mpi: `{mpicc} -show` failed");
  let show = String::from_utf8_lossy(&out.stdout).into_owned();
  let mut b = cc::Build::new();
  b.file("src/shim.c");
  let mut link = Vec::new();
  for t in show.split_whitespace().skip(1) {
    if let Some(d) = t.strip_prefix("-I") {
      b.include(d);
    } else if let Some(d) = t.strip_prefix("-L") {
      link.push(format!("cargo:rustc-link-search=native={d}"));
    } else if let Some(l) = t.strip_prefix("-l") {
      link.push(format!("cargo:rustc-link-lib={l}"));
    } else if t.starts_with("-Wl,") {
      link.push(format!("cargo:rustc-link-arg={t}"));
    }
  }
  if let Ok(d) = std::env::var("METIS_DIR") {
    b.include(format!("{d}/include"));
    link.push(format!("cargo:rustc-link-search=native={d}/lib"));
  }
  link.push("cargo:rustc-link-lib=metis".into());
  // the shim first, then the libraries it needs
  b.compile("getfem_mpi_shim");
  for l in link {
    println!("{l}");
  }
}
