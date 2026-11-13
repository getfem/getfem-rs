// links ARPACK-ng (Debian libarpack2-dev): its ISO C binding (dsaupd_c,
// dseupd_c), 32-bit integers
fn main() {
  println!("cargo:rustc-link-lib=arpack");
}
