//! GetFEM native mesh format (.mesh): points, convexes, regions. Point and
//! convex numbers of the file are kept (holes allowed), as GetFEM.

use std::fmt::Write as _;

use crate::geotrans::geotrans_descriptor;
use crate::mesh::{AsMeshData, Mesh, MeshData};
use crate::region::{MAX_FACES, RegionSet};
use crate::{Error, Result};

fn err(m: String) -> Error {
  Error::Parse(format!("mesh file: {m}"))
}

struct Lex<'a> {
  w: Vec<&'a str>,
  i: usize,
}

impl<'a> Lex<'a> {
  fn peek(&self) -> Option<&'a str> {
    self.w.get(self.i).copied()
  }

  fn next(&mut self) -> Result<&'a str> {
    let t = self
      .peek()
      .ok_or_else(|| err("unexpected end of file".into()))?;
    self.i += 1;
    Ok(t)
  }

  fn upto(&mut self, words: &[&str]) -> Result<()> {
    while self.i + words.len() <= self.w.len() {
      if words
        .iter()
        .enumerate()
        .all(|(k, s)| self.w[self.i + k].eq_ignore_ascii_case(s))
      {
        self.i += words.len();
        return Ok(());
      }
      self.i += 1;
    }
    Err(err(format!("missing '{}'", words.join(" "))))
  }

  fn num<T: std::str::FromStr>(&mut self) -> Result<T> {
    let t = self.next()?;
    t.parse()
      .map_err(|_| err(format!("expected a number, got '{t}'")))
  }

  /// Quoted name, possibly split by blanks: 'GT_PK(2, 1)'.
  fn quoted(&mut self) -> Result<String> {
    let mut s = self.next()?.to_string();
    if !s.starts_with('\'') {
      return Err(err(format!("expected a quoted name, got '{s}'")));
    }
    while !(s.len() > 1 && s.ends_with('\'')) {
      s.push_str(self.next()?);
    }
    Ok(s.trim_matches('\'').to_string())
  }
}

fn is_number(t: &str) -> bool {
  t.starts_with(|c: char| c.is_ascii_digit() || c == '-' || c == '+' || c == '.')
}

pub fn read_mesh(text: &str) -> Result<Mesh> {
  let mut l = Lex {
    w: text.split_ascii_whitespace().collect(),
    i: 0,
  };
  l.upto(&["BEGIN", "POINTS", "LIST"])?;
  let mut pts: Vec<(usize, Vec<f64>)> = vec![];
  loop {
    let t = l.next()?;
    if t.eq_ignore_ascii_case("END") {
      break;
    }
    if !t.eq_ignore_ascii_case("POINT") {
      return Err(err(format!("expected POINT or END, got '{t}'")));
    }
    if l.peek().is_some_and(|s| s.eq_ignore_ascii_case("COUNT")) {
      l.next()?;
      l.next()?;
      continue;
    }
    let id: usize = l.num()?;
    let mut x = vec![];
    while l.peek().is_some_and(is_number) {
      x.push(l.num()?);
    }
    pts.push((id, x));
  }
  let dim = pts.first().map_or(0, |p| p.1.len());
  let mut m = MeshData::new(dim);
  for (id, x) in &pts {
    if m.is_point_valid(*id) {
      return Err(err(format!("two points with index {id}")));
    }
    if let Some(j) = m.search_point(x, 0.0) {
      return Err(err(format!(
        "points {j} and {id} have the same coordinates {x:?}"
      )));
    }
    m.add_point_at(*id, x)?;
  }
  l.upto(&["BEGIN", "MESH", "STRUCTURE", "DESCRIPTION"])?;
  loop {
    let t = l.next()?;
    if t.eq_ignore_ascii_case("END") {
      l.upto(&["DESCRIPTION"])?;
      break;
    }
    if !t.eq_ignore_ascii_case("CONVEX") {
      return Err(err(format!("expected CONVEX or END, got '{t}'")));
    }
    if l.peek().is_some_and(|s| s.eq_ignore_ascii_case("COUNT")) {
      l.next()?;
      l.next()?;
      continue;
    }
    let id: usize = l.num()?;
    let gt = geotrans_descriptor(&l.quoted()?)?;
    let ip = (0..gt.nb_points())
      .map(|_| l.num())
      .collect::<Result<Vec<usize>>>()?;
    if m.is_convex_valid(id) {
      return Err(err(format!("two convexes with index {id}")));
    }
    m.add_convex_at(id, &gt, &ip)?;
  }
  while let Some(t) = l.peek() {
    if !t.eq_ignore_ascii_case("BEGIN") {
      break;
    }
    l.next()?;
    let kind = l.next()?;
    if !(kind.eq_ignore_ascii_case("REGION") || kind.eq_ignore_ascii_case("BOUNDARY")) {
      return Err(err(format!("unknown section '{kind}'")));
    }
    let id: usize = l.num()?;
    let mut r = RegionSet::new();
    loop {
      let e = l.next()?;
      if e.eq_ignore_ascii_case("END") {
        l.next()?;
        l.next()?;
        break;
      }
      // cv/f, also with spaces around '/' (GetFEM's tokenizer)
      let (c, mut f) = match e.split_once('/') {
        Some((c, f)) => (c, Some(f)),
        None if l.peek().is_some_and(|t| t.starts_with('/')) => (e, Some(&l.next()?[1..])),
        None => (e, None),
      };
      if f == Some("") {
        f = Some(l.next()?);
      }
      let num = |x: &str| x.parse().map_err(|_| err(format!("bad entry '{e}'")));
      match f {
        Some(f) => match num(f)? {
          f if f < MAX_FACES => r.add_face(num(c)?, f),
          _ => return Err(err(format!("bad face number in '{e}'"))),
        },
        None => r.add(num(c)?),
      }
    }
    m.set_region_set(id, r)?;
  }
  Ok(m.into())
}

pub fn read_mesh_file(path: &str) -> Result<Mesh> {
  let t =
    std::fs::read_to_string(path).map_err(|e| Error::InvalidArgument(format!("{path}: {e}")))?;
  read_mesh(&t)
}

/// Same layout as GetFEM's mesh::write_to_file (coordinates in the
/// shortest form that reads back exactly).
pub fn write_mesh(m: &impl AsMeshData) -> String {
  let m = &*m.view();
  let mut s = String::from("% GETFEM MESH FILE\n% written by GetFEM-rs\n\n\nBEGIN POINTS LIST\n\n");
  writeln!(
    s,
    "  POINT COUNT {}",
    m.point_index().last().map_or(0, |i| i + 1)
  )
  .unwrap();
  for i in m.point_index() {
    write!(s, "  POINT  {i}").unwrap();
    for x in m.point(i) {
      write!(s, "  {x:?}").unwrap();
    }
    s.push('\n');
  }
  s.push_str("\nEND POINTS LIST\n\n\n\nBEGIN MESH STRUCTURE DESCRIPTION\n\n");
  writeln!(
    s,
    "  CONVEX COUNT {}",
    m.convex_index().last().map_or(0, |i| i + 1)
  )
  .unwrap();
  for cv in m.convex_index() {
    write!(s, "  CONVEX {cv}    '{}'    ", m.trans_of_convex(cv).name()).unwrap();
    for p in m.ind_points_of_convex(cv) {
      write!(s, " {p}").unwrap();
    }
    s.push('\n');
  }
  s.push_str("\nEND MESH STRUCTURE DESCRIPTION\n");
  for id in m.region_ids() {
    writeln!(s, "BEGIN REGION {id}").unwrap();
    for (cv, f) in m.region_set(id).unwrap().iter() {
      match f {
        Some(f) => write!(s, "{cv}/{f} ").unwrap(),
        None => write!(s, "{cv} ").unwrap(),
      }
    }
    writeln!(s, "\nEND REGION {id}").unwrap();
  }
  s
}

pub fn write_mesh_file(path: &str, m: &impl AsMeshData) -> Result<()> {
  let m = &*m.view();
  std::fs::write(path, write_mesh(m)).map_err(|e| Error::InvalidArgument(format!("{path}: {e}")))
}
