//! Parser for method names such as `GT_PRODUCT(GT_PK(2,1),GT_PK(1,2))`,
//! shared by geometric transformations, elements and integration methods.

use std::fmt;

use crate::{Error, Result};

#[derive(Clone, Debug, PartialEq)]
pub enum Param {
  Num(f64),
  Method(Descriptor),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Descriptor {
  pub name: String,
  pub params: Vec<Param>,
}

impl Descriptor {
  pub fn parse(s: &str) -> Result<Self> {
    let mut p = P {
      s: s.as_bytes(),
      pos: 0,
    };
    let d = p.descriptor()?;
    p.ws();
    if p.pos != p.s.len() {
      return Err(p.err("trailing characters"));
    }
    Ok(d)
  }

  /// Name without its prefix, e.g. "PK" for "GT_PK" with prefix "GT_".
  pub fn suffix(&self, prefix: &str) -> Option<&str> {
    self.name.strip_prefix(prefix)
  }

  fn arity(&self, n: usize) -> Result<()> {
    if self.params.len() != n {
      return Err(Error::InvalidArgument(format!(
        "{}: {} parameters, expected {n}",
        self.name,
        self.params.len()
      )));
    }
    Ok(())
  }

  /// All n parameters as non-negative integers.
  pub fn ints(&self, n: usize) -> Result<Vec<usize>> {
    self.arity(n)?;
    self
      .params
      .iter()
      .map(|p| match p {
        Param::Num(x) if *x >= 0.0 && x.fract() == 0.0 => Ok(*x as usize),
        _ => Err(Error::InvalidArgument(format!(
          "{}: expected integer parameters",
          self.name
        ))),
      })
      .collect()
  }

  /// All n parameters as methods.
  pub fn methods(&self, n: usize) -> Result<Vec<&Descriptor>> {
    self.arity(n)?;
    self
      .params
      .iter()
      .map(|p| match p {
        Param::Method(d) => Ok(d),
        _ => Err(Error::InvalidArgument(format!(
          "{}: expected method parameters",
          self.name
        ))),
      })
      .collect()
  }
}

impl fmt::Display for Param {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Param::Num(x) => write!(f, "{x}"),
      Param::Method(d) => write!(f, "{d}"),
    }
  }
}

impl fmt::Display for Descriptor {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}", self.name)?;
    if !self.params.is_empty() {
      write!(f, "(")?;
      for (i, p) in self.params.iter().enumerate() {
        if i > 0 {
          write!(f, ",")?;
        }
        write!(f, "{p}")?;
      }
      write!(f, ")")?;
    }
    Ok(())
  }
}

struct P<'a> {
  s: &'a [u8],
  pos: usize,
}

impl P<'_> {
  fn err(&self, m: &str) -> Error {
    Error::Parse(format!(
      "'{}' at {}: {m}",
      String::from_utf8_lossy(self.s),
      self.pos
    ))
  }

  fn ws(&mut self) {
    while self
      .s
      .get(self.pos)
      .is_some_and(|c| c.is_ascii_whitespace())
    {
      self.pos += 1;
    }
  }

  fn peek(&self) -> Option<u8> {
    self.s.get(self.pos).copied()
  }

  fn descriptor(&mut self) -> Result<Descriptor> {
    self.ws();
    let st = self.pos;
    while self
      .peek()
      .is_some_and(|c| c.is_ascii_alphanumeric() || c == b'_')
    {
      self.pos += 1;
    }
    if st == self.pos || self.s[st].is_ascii_digit() {
      return Err(self.err("expected a method name"));
    }
    let name = String::from_utf8_lossy(&self.s[st..self.pos]).into_owned();
    let mut params = vec![];
    self.ws();
    if self.peek() == Some(b'(') {
      self.pos += 1;
      self.ws();
      if self.peek() == Some(b')') {
        self.pos += 1;
      } else {
        loop {
          params.push(self.param()?);
          self.ws();
          match self.peek() {
            Some(b',') => self.pos += 1,
            Some(b')') => {
              self.pos += 1;
              break;
            }
            _ => return Err(self.err("expected ',' or ')'")),
          }
        }
      }
    }
    Ok(Descriptor { name, params })
  }

  fn param(&mut self) -> Result<Param> {
    self.ws();
    match self.peek() {
      Some(c) if c.is_ascii_digit() || c == b'-' || c == b'+' || c == b'.' => {
        let st = self.pos;
        self.pos += 1;
        while self
          .peek()
          .is_some_and(|c| c.is_ascii_digit() || matches!(c, b'.' | b'e' | b'E' | b'-' | b'+'))
        {
          self.pos += 1;
        }
        let t = String::from_utf8_lossy(&self.s[st..self.pos]);
        t.parse()
          .map(Param::Num)
          .map_err(|_| self.err("bad number"))
      }
      _ => Ok(Param::Method(self.descriptor()?)),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn parse_and_print() {
    let d = Descriptor::parse(" GT_PRODUCT( GT_PK(2,1) , GT_PK(1, 2)) ").unwrap();
    assert_eq!(d.to_string(), "GT_PRODUCT(GT_PK(2,1),GT_PK(1,2))");
    let m = d.methods(2).unwrap();
    assert_eq!(m[1].ints(2).unwrap(), vec![1, 2]);
    assert_eq!(Descriptor::parse("IM_NONE").unwrap().params.len(), 0);
    assert_eq!(
      Descriptor::parse("IM_NONE()").unwrap().to_string(),
      "IM_NONE"
    );
  }

  #[test]
  fn errors() {
    for s in [
      "",
      "GT_PK(2,",
      "GT_PK(2 1)",
      "GT_PK(2,1))",
      "2(3)",
      "GT_PK(1e)",
    ] {
      assert!(Descriptor::parse(s).is_err(), "{s}");
    }
    let d = Descriptor::parse("GT_PK(2.5,1)").unwrap();
    assert!(d.ints(2).is_err());
    assert!(d.ints(3).is_err());
    assert!(d.methods(2).is_err());
  }
}
