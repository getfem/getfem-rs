//! zlib streams (RFC 1950/1951) with LZ77 and the fixed Huffman codes, and
//! base64: enough for compressed VTK XML files without dependencies.

/// bits written least significant first (deflate order)
struct Bits {
  out: Vec<u8>,
  acc: u64,
  n: u32,
}

impl Bits {
  fn put(&mut self, v: u32, len: u32) {
    self.acc |= (v as u64) << self.n;
    self.n += len;
    while self.n >= 8 {
      self.out.push(self.acc as u8);
      self.acc >>= 8;
      self.n -= 8;
    }
  }

  /// a Huffman code, most significant bit first
  fn code(&mut self, c: u32, len: u32) {
    let mut r = 0;
    for i in 0..len {
      r |= ((c >> i) & 1) << (len - 1 - i);
    }
    self.put(r, len);
  }

  fn finish(mut self) -> Vec<u8> {
    if self.n > 0 {
      self.out.push(self.acc as u8);
    }
    self.out
  }
}

/// fixed literal/length code of symbol s (0..=287)
fn lit(b: &mut Bits, s: u32) {
  match s {
    0..=143 => b.code(0x30 + s, 8),
    144..=255 => b.code(0x190 + s - 144, 9),
    256..=279 => b.code(s - 256, 7),
    _ => b.code(0xc0 + s - 280, 8),
  }
}

const LEN_BASE: [u32; 29] = [
  3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
  163, 195, 227, 258,
];
const LEN_EXTRA: [u32; 29] = [
  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u32; 30] = [
  1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049,
  3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u32; 30] = [
  0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
];

fn matched(b: &mut Bits, len: u32, dist: u32) {
  let i = LEN_BASE.iter().rposition(|&x| x <= len).unwrap();
  lit(b, 257 + i as u32);
  b.put(len - LEN_BASE[i], LEN_EXTRA[i]);
  let j = DIST_BASE.iter().rposition(|&x| x <= dist).unwrap();
  b.code(j as u32, 5);
  b.put(dist - DIST_BASE[j], DIST_EXTRA[j]);
}

fn adler32(d: &[u8]) -> u32 {
  let (mut a, mut b) = (1u32, 0u32);
  for chunk in d.chunks(5552) {
    for &x in chunk {
      a += x as u32;
      b += a;
    }
    a %= 65521;
    b %= 65521;
  }
  (b << 16) | a
}

/// A zlib stream of data: one final block with the fixed codes, LZ77 with
/// hash chains (window 32 KiB, matches 3..=258).
pub fn compress(data: &[u8]) -> Vec<u8> {
  const WIN: usize = 32768;
  const HBITS: u32 = 15;
  const CHAIN: usize = 64;
  let mut b = Bits {
    out: vec![0x78, 0x01],
    acc: 0,
    n: 0,
  };
  b.put(1, 1); // final block
  b.put(1, 2); // fixed Huffman codes
  let hash = |i: usize| -> usize {
    let v = (data[i] as u32) | (data[i + 1] as u32) << 8 | (data[i + 2] as u32) << 16;
    (v.wrapping_mul(2654435761) >> (32 - HBITS)) as usize
  };
  let mut head = vec![usize::MAX; 1 << HBITS];
  let mut prev = vec![usize::MAX; data.len()];
  let insert = |i: usize, head: &mut Vec<usize>, prev: &mut Vec<usize>| {
    if i + 2 < data.len() {
      let h = hash(i);
      prev[i] = head[h];
      head[h] = i;
    }
  };
  let mut i = 0;
  while i < data.len() {
    let mut best = (0usize, 0usize);
    if i + 2 < data.len() {
      let mut c = head[hash(i)];
      let mut k = 0;
      while c != usize::MAX && i - c <= WIN && k < CHAIN {
        let max = (data.len() - i).min(258);
        let mut l = 0;
        while l < max && data[c + l] == data[i + l] {
          l += 1;
        }
        if l > best.0 {
          best = (l, i - c);
          if l == max {
            break;
          }
        }
        c = prev[c];
        k += 1;
      }
    }
    if best.0 >= 3 {
      matched(&mut b, best.0 as u32, best.1 as u32);
      for j in i..i + best.0 {
        insert(j, &mut head, &mut prev);
      }
      i += best.0;
    } else {
      lit(&mut b, data[i] as u32);
      insert(i, &mut head, &mut prev);
      i += 1;
    }
  }
  lit(&mut b, 256);
  let mut out = b.finish();
  out.extend(adler32(data).to_be_bytes());
  out
}

/// Standard base64 with padding.
pub fn base64(d: &[u8]) -> String {
  const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  let mut s = String::with_capacity(d.len().div_ceil(3) * 4);
  for c in d.chunks(3) {
    let v =
      (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
    for k in 0..4 {
      if k <= c.len() {
        s.push(T[((v >> (18 - 6 * k)) & 63) as usize] as char);
      } else {
        s.push('=');
      }
    }
  }
  s
}
