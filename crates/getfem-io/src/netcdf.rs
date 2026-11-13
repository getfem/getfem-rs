//! netCDF files, the 64-bit offset format (CDF-2, Exodus' default) only:
//! a header (dimensions, attributes, variables), then the data, big
//! endian; the record variables (first dimension the unlimited one)
//! interleaved by record.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

use crate::{Error, Result};

const NC_DIMENSION: u32 = 10;
const NC_VARIABLE: u32 = 11;
const NC_ATTRIBUTE: u32 = 12;

/// The external types: byte, char, short, int, float, double.
pub const NC_CHAR: u32 = 2;
pub const NC_INT: u32 = 4;
pub const NC_DOUBLE: u32 = 6;

fn type_size(t: u32) -> Option<u64> {
  match t {
    1 | 2 => Some(1),
    3 => Some(2),
    4 | 5 => Some(4),
    6 => Some(8),
    _ => None,
  }
}

fn pad4(n: u64) -> u64 {
  n.div_ceil(4) * 4
}

#[derive(Clone, Debug)]
pub struct Att {
  pub name: String,
  pub ty: u32,
  /// the values, big endian
  pub bytes: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct Var {
  pub name: String,
  /// dimension ids
  pub dims: Vec<usize>,
  pub atts: Vec<Att>,
  pub ty: u32,
  pub begin: u64,
}

/// A netCDF file opened for reading: its header, the data read on demand.
pub struct NcFile {
  file: File,
  path: String,
  /// (name, length); length 0: the unlimited (record) dimension
  pub dims: Vec<(String, u64)>,
  pub atts: Vec<Att>,
  pub vars: Vec<Var>,
  pub numrecs: u64,
  recsize: u64,
}

struct Header<'a> {
  r: BufReader<&'a File>,
  path: &'a str,
}

impl Header<'_> {
  fn err(&self, m: &str) -> Error {
    Error::Format(format!("{}: {m}", self.path))
  }

  fn bytes(&mut self, n: u64) -> Result<Vec<u8>> {
    let mut b = vec![0; n as usize];
    self
      .r
      .read_exact(&mut b)
      .map_err(|_| self.err("truncated netCDF header"))?;
    Ok(b)
  }

  fn u32(&mut self) -> Result<u32> {
    Ok(u32::from_be_bytes(self.bytes(4)?.try_into().unwrap()))
  }

  fn u64(&mut self) -> Result<u64> {
    Ok(u64::from_be_bytes(self.bytes(8)?.try_into().unwrap()))
  }

  fn name(&mut self) -> Result<String> {
    let n = self.u32()? as u64;
    let b = self.bytes(pad4(n))?;
    String::from_utf8(b[..n as usize].to_vec()).map_err(|_| self.err("a name not in UTF-8"))
  }

  /// tag and count of a list; ABSENT (two zeros): 0
  fn list(&mut self, tag: u32) -> Result<usize> {
    let (t, n) = (self.u32()?, self.u32()? as usize);
    if t != tag && !(t == 0 && n == 0) {
      return Err(self.err("bad netCDF header"));
    }
    Ok(n)
  }

  fn atts(&mut self) -> Result<Vec<Att>> {
    (0..self.list(NC_ATTRIBUTE)?)
      .map(|_| {
        let name = self.name()?;
        let ty = self.u32()?;
        let n = self.u32()? as u64;
        let sz = type_size(ty).ok_or_else(|| self.err("unknown netCDF type"))?;
        let mut bytes = self.bytes(pad4(n * sz))?;
        bytes.truncate((n * sz) as usize);
        Ok(Att { name, ty, bytes })
      })
      .collect()
  }
}

impl NcFile {
  pub fn open(path: &str) -> Result<NcFile> {
    let file = File::open(path).map_err(|e| Error::Io(format!("{path}: {e}")))?;
    let mut h = Header {
      r: BufReader::new(&file),
      path,
    };
    let magic = h.bytes(4).map_err(|_| h.err("not a netCDF file"))?;
    match magic.as_slice() {
      b"CDF\x02" => {}
      [b'C', b'D', b'F', _] | [0x89, b'H', b'D', b'F'] => {
        return Err(h.err(
          "only the netCDF 64-bit offset format is read: \
           nccopy -k '64-bit offset' in.exo out.exo converts",
        ));
      }
      _ => return Err(h.err("not a netCDF file")),
    }
    let numrecs = h.u32()? as u64;
    let dims = (0..h.list(NC_DIMENSION)?)
      .map(|_| Ok((h.name()?, h.u32()? as u64)))
      .collect::<Result<Vec<_>>>()?;
    let atts = h.atts()?;
    let vars = (0..h.list(NC_VARIABLE)?)
      .map(|_| {
        let name = h.name()?;
        let dims = (0..h.u32()?)
          .map(|_| h.u32().map(|d| d as usize))
          .collect::<Result<Vec<_>>>()?;
        let atts = h.atts()?;
        let ty = h.u32()?;
        let _vsize = h.u32()?;
        let begin = h.u64()?;
        if type_size(ty).is_none() {
          return Err(h.err("bad netCDF variable"));
        }
        Ok(Var {
          name,
          dims,
          atts,
          ty,
          begin,
        })
      })
      .collect::<Result<Vec<_>>>()?;
    drop(h);
    let mut nc = NcFile {
      file,
      path: path.to_string(),
      dims,
      atts,
      vars,
      numrecs,
      recsize: 0,
    };
    for v in &nc.vars {
      if v.dims.iter().any(|&d| d >= nc.dims.len()) {
        return Err(nc.err(&format!("variable {}: bad dimension", v.name)));
      }
    }
    let rec: Vec<&Var> = nc.vars.iter().filter(|v| nc.is_record(v)).collect();
    nc.recsize = match rec.as_slice() {
      [v] => nc.rec_bytes(v),
      vs => vs.iter().map(|v| pad4(nc.rec_bytes(v))).sum(),
    };
    Ok(nc)
  }

  fn err(&self, m: &str) -> Error {
    Error::Format(format!("{}: {m}", self.path))
  }

  fn is_record(&self, v: &Var) -> bool {
    v.dims.first().is_some_and(|&d| self.dims[d].1 == 0)
  }

  /// bytes of one record of v (all of v if not a record variable)
  fn rec_bytes(&self, v: &Var) -> u64 {
    let n: u64 = v
      .dims
      .iter()
      .skip(self.is_record(v) as usize)
      .map(|&d| self.dims[d].1)
      .product();
    n * type_size(v.ty).unwrap()
  }

  /// Length of a dimension.
  pub fn dim(&self, name: &str) -> Option<usize> {
    self
      .dims
      .iter()
      .find(|d| d.0 == name)
      .map(|d| if d.1 == 0 { self.numrecs } else { d.1 } as usize)
  }

  pub fn var(&self, name: &str) -> Option<&Var> {
    self.vars.iter().find(|v| v.name == name)
  }

  fn need(&self, name: &str) -> Result<&Var> {
    self
      .var(name)
      .ok_or_else(|| self.err(&format!("no variable {name}")))
  }

  /// Shape of a variable (the record dimension: the number of records).
  pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
    let v = self.need(name)?;
    Ok(
      v.dims
        .iter()
        .map(|&d| self.dim(&self.dims[d].0).unwrap())
        .collect(),
    )
  }

  fn raw(&self, v: &Var, rec: Option<u64>) -> Result<Vec<u8>> {
    let off = v.begin + rec.map_or(0, |r| r * self.recsize);
    let mut b = vec![0; self.rec_bytes(v) as usize];
    let mut f = &self.file;
    f.seek(SeekFrom::Start(off))
      .and_then(|_| f.read_exact(&mut b))
      .map_err(|_| self.err(&format!("variable {}: truncated data", v.name)))?;
    Ok(b)
  }

  fn values(&self, name: &str, rec: Option<usize>) -> Result<(u32, Vec<u8>)> {
    let v = self.need(name)?;
    let rec = match (self.is_record(v), rec) {
      (true, Some(r)) if (r as u64) < self.numrecs => Some(r as u64),
      (false, None) => None,
      _ => return Err(self.err(&format!("variable {name}: bad record"))),
    };
    Ok((v.ty, self.raw(v, rec)?))
  }

  /// A variable (one record of a record variable) as f64.
  pub fn get_f64(&self, name: &str, rec: Option<usize>) -> Result<Vec<f64>> {
    let (ty, b) = self.values(name, rec)?;
    to_f64(ty, &b).ok_or_else(|| self.err(&format!("variable {name}: not numeric")))
  }

  /// A variable as i64 (integer types only).
  pub fn get_i64(&self, name: &str) -> Result<Vec<i64>> {
    let (ty, b) = self.values(name, None)?;
    let v: Option<Vec<i64>> = match ty {
      1 => Some(b.iter().map(|&x| x as i8 as i64).collect()),
      3 => Some(
        b.chunks(2)
          .map(|c| i16::from_be_bytes([c[0], c[1]]) as i64)
          .collect(),
      ),
      4 => Some(
        b.chunks(4)
          .map(|c| i32::from_be_bytes(c.try_into().unwrap()) as i64)
          .collect(),
      ),
      _ => None,
    };
    v.ok_or_else(|| self.err(&format!("variable {name}: not an integer")))
  }

  /// A char variable [n][len] as n strings (up to their first 0).
  pub fn get_strings(&self, name: &str) -> Result<Vec<String>> {
    let shape = self.shape(name)?;
    let (ty, b) = self.values(name, None)?;
    if ty != NC_CHAR || shape.len() != 2 {
      return Err(self.err(&format!("variable {name}: not strings")));
    }
    Ok(b.chunks(shape[1].max(1)).map(text).collect())
  }

  /// A text attribute of a variable.
  pub fn att_text(&self, var: &str, att: &str) -> Option<String> {
    let a = self.var(var)?.atts.iter().find(|a| a.name == att)?;
    (a.ty == NC_CHAR).then(|| text(&a.bytes))
  }
}

fn text(b: &[u8]) -> String {
  let n = b.iter().position(|&c| c == 0).unwrap_or(b.len());
  String::from_utf8_lossy(&b[..n]).trim_end().to_string()
}

fn to_f64(ty: u32, b: &[u8]) -> Option<Vec<f64>> {
  Some(match ty {
    1 => b.iter().map(|&x| x as i8 as f64).collect(),
    3 => b
      .chunks(2)
      .map(|c| i16::from_be_bytes([c[0], c[1]]) as f64)
      .collect(),
    4 => b
      .chunks(4)
      .map(|c| i32::from_be_bytes(c.try_into().unwrap()) as f64)
      .collect(),
    5 => b
      .chunks(4)
      .map(|c| f32::from_be_bytes(c.try_into().unwrap()) as f64)
      .collect(),
    6 => b
      .chunks(8)
      .map(|c| f64::from_be_bytes(c.try_into().unwrap()))
      .collect(),
    _ => return None,
  })
}

/// A netCDF file built in memory: dimensions, global attributes, variables
/// with their data (a record variable: one record).
#[derive(Default)]
pub struct NcWriter {
  dims: Vec<(String, u64)>,
  atts: Vec<Att>,
  vars: Vec<(Var, Vec<u8>)>,
}

impl Att {
  pub fn text(name: &str, s: &str) -> Att {
    Att {
      name: name.into(),
      ty: NC_CHAR,
      bytes: s.as_bytes().to_vec(),
    }
  }

  pub fn int(name: &str, v: i32) -> Att {
    Att {
      name: name.into(),
      ty: NC_INT,
      bytes: v.to_be_bytes().to_vec(),
    }
  }

  pub fn float(name: &str, v: f32) -> Att {
    Att {
      name: name.into(),
      ty: 5,
      bytes: v.to_be_bytes().to_vec(),
    }
  }
}

/// Big endian bytes of doubles and ints.
pub fn doubles(v: &[f64]) -> Vec<u8> {
  v.iter().flat_map(|x| x.to_be_bytes()).collect()
}

pub fn ints(v: &[i32]) -> Vec<u8> {
  v.iter().flat_map(|x| x.to_be_bytes()).collect()
}

/// Strings as a char array [n][len] (0 padded).
pub fn strings(v: &[&str], len: usize) -> Vec<u8> {
  v.iter()
    .flat_map(|s| {
      let mut b = s.as_bytes()[..s.len().min(len - 1)].to_vec();
      b.resize(len, 0);
      b
    })
    .collect()
}

fn put_name(out: &mut Vec<u8>, s: &str) {
  out.extend((s.len() as u32).to_be_bytes());
  out.extend(s.as_bytes());
  out.resize(pad4(out.len() as u64) as usize, 0);
}

fn put_atts(out: &mut Vec<u8>, atts: &[Att]) {
  let tag = if atts.is_empty() { 0 } else { NC_ATTRIBUTE };
  out.extend(tag.to_be_bytes());
  out.extend((atts.len() as u32).to_be_bytes());
  for a in atts {
    put_name(out, &a.name);
    out.extend(a.ty.to_be_bytes());
    out.extend(((a.bytes.len() as u64 / type_size(a.ty).unwrap()) as u32).to_be_bytes());
    out.extend(&a.bytes);
    out.resize(pad4(out.len() as u64) as usize, 0);
  }
}

impl NcWriter {
  /// A dimension (len 0: the unlimited one); its id.
  pub fn dim(&mut self, name: &str, len: usize) -> usize {
    self.dims.push((name.into(), len as u64));
    self.dims.len() - 1
  }

  pub fn att(&mut self, a: Att) {
    self.atts.push(a);
  }

  /// A variable with its data (big endian; a record variable: one record).
  pub fn var(&mut self, name: &str, dims: &[usize], ty: u32, atts: Vec<Att>, data: Vec<u8>) {
    let v = Var {
      name: name.into(),
      dims: dims.to_vec(),
      atts,
      ty,
      begin: 0,
    };
    self.vars.push((v, data));
  }

  fn is_record(&self, v: &Var) -> bool {
    v.dims.first().is_some_and(|&d| self.dims[d].1 == 0)
  }

  /// The file with one record (none if no record variable has data), and
  /// the offset of the records.
  pub fn bytes(&self) -> (Vec<u8>, usize) {
    let header = |begins: &[u64], numrecs: u32| {
      let mut h = b"CDF\x02".to_vec();
      h.extend(numrecs.to_be_bytes());
      let tag = if self.dims.is_empty() {
        0
      } else {
        NC_DIMENSION
      };
      h.extend(tag.to_be_bytes());
      h.extend((self.dims.len() as u32).to_be_bytes());
      for (n, l) in &self.dims {
        put_name(&mut h, n);
        h.extend((*l as u32).to_be_bytes());
      }
      put_atts(&mut h, &self.atts);
      let tag = if self.vars.is_empty() { 0 } else { NC_VARIABLE };
      h.extend(tag.to_be_bytes());
      h.extend((self.vars.len() as u32).to_be_bytes());
      for ((v, d), b) in self.vars.iter().zip(begins) {
        put_name(&mut h, &v.name);
        h.extend((v.dims.len() as u32).to_be_bytes());
        for &i in &v.dims {
          h.extend((i as u32).to_be_bytes());
        }
        put_atts(&mut h, &v.atts);
        h.extend(v.ty.to_be_bytes());
        h.extend((pad4(d.len() as u64).min(u32::MAX as u64) as u32).to_be_bytes());
        h.extend(b.to_be_bytes());
      }
      h
    };
    let size = header(&vec![0; self.vars.len()], 0).len() as u64;
    // the fixed variables, then the records
    let (mut begins, mut off) = (vec![0; self.vars.len()], size);
    for rec in [false, true] {
      for (i, (v, d)) in self.vars.iter().enumerate() {
        if self.is_record(v) == rec {
          begins[i] = off;
          off += pad4(d.len() as u64);
        }
      }
    }
    let numrecs = self
      .vars
      .iter()
      .any(|(v, d)| self.is_record(v) && !d.is_empty()) as u32;
    let mut out = header(&begins, numrecs);
    let mut rec_start = 0;
    for rec in [false, true] {
      if rec {
        rec_start = out.len();
      }
      for (v, d) in &self.vars {
        if self.is_record(v) == rec {
          out.extend(d);
          out.resize(pad4(out.len() as u64) as usize, 0);
        }
      }
    }
    (out, rec_start)
  }
}
