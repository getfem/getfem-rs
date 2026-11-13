//! zlib streams and base64 checked with python's zlib and base64.

use std::process::Command;

use getfem_io::zlib::{base64, compress};

fn samples() -> Vec<Vec<u8>> {
  let mut r = 12345u64;
  let mut rnd = || {
    r ^= r << 13;
    r ^= r >> 7;
    r ^= r << 17;
    r
  };
  let floats: Vec<u8> = (0..9000)
    .flat_map(|i| ((i as f64 * 0.01).sin()).to_le_bytes())
    .collect();
  vec![
    vec![],
    b"a".to_vec(),
    b"abcabcabcabcabcabcabcabc hello hello hello".to_vec(),
    vec![0u8; 100000],
    (0..5000).map(|_| rnd() as u8).collect(),
    floats,
  ]
}

#[test]
fn zlib_and_base64_decode_with_python() {
  let dir = format!("{}/../../target/zlib-check", env!("CARGO_MANIFEST_DIR"));
  std::fs::create_dir_all(&dir).unwrap();
  let py = Command::new("python3")
    .arg("-c")
    .arg("import zlib")
    .output();
  if !py.map(|o| o.status.success()).unwrap_or(false) {
    eprintln!("python3 not available: skipped");
    return;
  }
  for (k, d) in samples().iter().enumerate() {
    let (z, b) = (format!("{dir}/{k}.z"), format!("{dir}/{k}.b64"));
    std::fs::write(&z, compress(d)).unwrap();
    std::fs::write(&b, base64(d)).unwrap();
    let raw = format!("{dir}/{k}.raw");
    std::fs::write(&raw, d).unwrap();
    let o = Command::new("python3")
      .arg("-c")
      .arg(format!(
        "import zlib, base64; d=open('{raw}','rb').read(); assert zlib.decompress(open('{z}','rb').read())==d; assert base64.b64decode(open('{b}').read())==d"
      ))
      .output()
      .unwrap();
    assert!(
      o.status.success(),
      "sample {k}: {}",
      String::from_utf8_lossy(&o.stderr)
    );
  }
  // compression works on connectivity-like data (Int64 node numbers)
  let f: Vec<u8> = (0..30000i64)
    .flat_map(|i| (i / 3 + i % 3).to_le_bytes())
    .collect();
  assert!(
    compress(&f).len() < f.len() / 3,
    "{} of {}",
    compress(&f).len(),
    f.len()
  );
}
