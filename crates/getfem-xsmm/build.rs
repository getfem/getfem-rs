// builds libxsmm 2.1.0 header-only (libxsmm.c) on the architectures where
// it generates kernels (x86_64, aarch64), cfg xsmm.
// Its release tarball: GETFEM_LIBXSMM_TARBALL (offline builds, packages),
// else downloaded once into target/<profile>/build; checked by sha256.
use std::path::{Path, PathBuf};
use std::process::Command;

const XSMM: &str = "libxsmm-2.1.0";
const URL: &str = "https://github.com/libxsmm/libxsmm/archive/refs/tags/2.1.0.tar.gz";
const SHA256: &str = "704ed8f99b61a767798ed1ee1cadc5d185ca449f466a2bae37930f68c65961e9";

fn run(cmd: &mut Command) -> String {
  let out = cmd.output().unwrap_or_else(|e| panic!("{cmd:?}: {e}"));
  assert!(
    out.status.success(),
    "{cmd:?}: {}",
    String::from_utf8_lossy(&out.stderr)
  );
  String::from_utf8_lossy(&out.stdout).into_owned()
}

fn sha256_ok(f: &Path) -> bool {
  f.exists()
    && run(Command::new("sha256sum").arg(f))
      .split_whitespace()
      .next()
      == Some(SHA256)
}

fn tarball(out: &Path) -> PathBuf {
  println!("cargo::rerun-if-env-changed=GETFEM_LIBXSMM_TARBALL");
  if let Some(t) = std::env::var_os("GETFEM_LIBXSMM_TARBALL") {
    let t = PathBuf::from(t);
    assert!(
      sha256_ok(&t),
      "GETFEM_LIBXSMM_TARBALL {t:?}: not {XSMM}.tar.gz (sha256 {SHA256})"
    );
    return t;
  }
  let cache = out
    .ancestors()
    .nth(2)
    .unwrap()
    .join(format!("{XSMM}.tar.gz"));
  if !sha256_ok(&cache) {
    let part = cache.with_extension("part");
    let ok = Command::new("curl")
      .args(["-sSfL", "-o"])
      .arg(&part)
      .arg(URL)
      .status();
    assert!(
      matches!(ok, Ok(s) if s.success()),
      "download of {URL} failed; offline: set GETFEM_LIBXSMM_TARBALL to that file"
    );
    assert!(sha256_ok(&part), "{URL}: sha256 is not {SHA256}");
    std::fs::rename(&part, &cache).unwrap();
  }
  cache
}

fn main() {
  println!("cargo::rustc-check-cfg=cfg(xsmm)");
  let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();
  if arch == "x86_64" || arch == "aarch64" {
    println!("cargo::rerun-if-changed=libxsmm.c");
    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    let t = tarball(&out);
    let (inc, src) = (format!("{XSMM}/include"), format!("{XSMM}/src"));
    run(
      Command::new("tar")
        .arg("xzf")
        .arg(&t)
        .arg("-C")
        .arg(&out)
        .args([&inc, &src]),
    );
    cc::Build::new()
      .file("libxsmm.c")
      .include(out.join(&inc))
      .opt_level(2)
      .debug(false)
      .warnings(false)
      .compile("xsmm");
    for l in ["pthread", "dl", "m"] {
      println!("cargo:rustc-link-lib={l}");
    }
    println!("cargo::rustc-cfg=xsmm");
  }
}
