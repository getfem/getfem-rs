//! GWFL tokens (GetFEM's ga_get_token). Unlike GetFEM, any whitespace
//! separates tokens (GetFEM only skips spaces).

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tok {
  Invalid,
  End,
  Name,
  Scalar,
  Plus,
  Minus,
  Mult,
  Div,
  Colon,
  Quote,
  ColonEq,
  Def,
  Sym,
  Skew,
  Trace,
  Deviator,
  Interpolate,
  InterpolateFilter,
  InterpolateDerivative,
  Elementary,
  SecondaryDomain,
  XfemPlus,
  XfemMinus,
  Print,
  Dot,
  DotMult,
  DotDiv,
  TMult,
  Comma,
  DComma,
  Semicolon,
  DSemicolon,
  LPar,
  RPar,
  LBracket,
  RBracket,
  /// the unary minus is only produced by the parser
  UnaryMinus,
}

impl Tok {
  /// GetFEM's ga_operator_priorities.
  pub fn priority(self) -> u8 {
    match self {
      Tok::Plus | Tok::Minus => 1,
      Tok::Mult | Tok::Div | Tok::Colon | Tok::Dot | Tok::DotMult | Tok::DotDiv | Tok::TMult => 2,
      Tok::Quote | Tok::UnaryMinus => 3,
      Tok::Sym | Tok::Skew | Tok::Trace | Tok::Deviator | Tok::Print => 4,
      _ => 0,
    }
  }

  /// Prefix operators applying to the next term.
  pub fn is_prefix(self) -> bool {
    matches!(
      self,
      Tok::UnaryMinus | Tok::Sym | Tok::Skew | Tok::Trace | Tok::Deviator | Tok::Print
    )
  }

  pub fn symbol(self) -> &'static str {
    match self {
      Tok::Plus => "+",
      Tok::Minus | Tok::UnaryMinus => "-",
      Tok::Mult => "*",
      Tok::Div => "/",
      Tok::Colon => ":",
      Tok::Quote => "'",
      Tok::Dot => ".",
      Tok::DotMult => ".*",
      Tok::DotDiv => "./",
      Tok::TMult => "@",
      Tok::Sym => "Sym",
      Tok::Skew => "Skew",
      Tok::Trace => "Trace",
      Tok::Deviator => "Deviator",
      Tok::Print => "Print",
      _ => "?",
    }
  }
}

fn class(c: u8) -> Tok {
  match c {
    b'+' => Tok::Plus,
    b'-' => Tok::Minus,
    b'*' => Tok::Mult,
    b'/' => Tok::Div,
    b':' => Tok::Colon,
    b'\'' => Tok::Quote,
    b'.' => Tok::Dot,
    b'@' => Tok::TMult,
    b',' => Tok::Comma,
    b';' => Tok::Semicolon,
    b'(' => Tok::LPar,
    b')' => Tok::RPar,
    b'[' => Tok::LBracket,
    b']' => Tok::RBracket,
    b'=' => Tok::ColonEq,
    b'_' | b'a'..=b'z' | b'A'..=b'Z' => Tok::Name,
    b'0'..=b'9' => Tok::Scalar,
    _ => Tok::Invalid,
  }
}

/// Next token at pos: (kind, start, length); pos is advanced.
pub fn next_token(s: &[u8], pos: &mut usize) -> (Tok, usize, usize) {
  while *pos < s.len() && s[*pos].is_ascii_whitespace() {
    *pos += 1;
  }
  let start = *pos;
  if *pos >= s.len() {
    return (Tok::End, start, 0);
  }
  let mut ty = class(s[*pos]);
  *pos += 1;
  let len = |p: &usize| *p - start;
  let (mut fdot, mut fe) = (false, false);
  if ty == Tok::Dot {
    if *pos >= s.len() {
      return (ty, start, 1);
    }
    match s[*pos] {
      b'*' => {
        *pos += 1;
        return (Tok::DotMult, start, 2);
      }
      b'/' => {
        *pos += 1;
        return (Tok::DotDiv, start, 2);
      }
      c if class(c) == Tok::Scalar => {
        fdot = true;
        ty = Tok::Scalar;
      }
      _ => return (ty, start, 1),
    }
  }
  match ty {
    Tok::Scalar => {
      while *pos < s.len() {
        match class(s[*pos]) {
          Tok::Dot => {
            if fdot {
              break;
            }
            fdot = true;
            *pos += 1;
          }
          Tok::Name => {
            if fe || (s[*pos] != b'e' && s[*pos] != b'E') {
              break;
            }
            fe = true;
            fdot = true;
            *pos += 1;
            if *pos < s.len() && (s[*pos] == b'+' || s[*pos] == b'-') {
              *pos += 1;
            }
            if *pos >= s.len() || class(s[*pos]) != Tok::Scalar {
              return (Tok::Invalid, start, len(pos));
            }
          }
          Tok::Scalar => *pos += 1,
          _ => break,
        }
      }
      (Tok::Scalar, start, len(pos))
    }
    Tok::Name => {
      while *pos < s.len() && matches!(class(s[*pos]), Tok::Scalar | Tok::Name) {
        *pos += 1;
      }
      let kw = match &s[start..*pos] {
        b"Sym" => Tok::Sym,
        b"Def" => Tok::Def,
        b"Skew" => Tok::Skew,
        b"Trace" => Tok::Trace,
        b"Deviator" => Tok::Deviator,
        b"Interpolate" => Tok::Interpolate,
        b"Interpolate_derivative" => Tok::InterpolateDerivative,
        b"Interpolate_filter" => Tok::InterpolateFilter,
        b"Elementary_transformation" => Tok::Elementary,
        b"Secondary_domain" | b"Secondary_Domain" => Tok::SecondaryDomain,
        b"Xfem_plus" => Tok::XfemPlus,
        b"Xfem_minus" => Tok::XfemMinus,
        b"Print" => Tok::Print,
        _ => Tok::Name,
      };
      (kw, start, len(pos))
    }
    Tok::Comma if *pos < s.len() && class(s[*pos]) == Tok::Comma => {
      *pos += 1;
      (Tok::DComma, start, 2)
    }
    Tok::Semicolon if *pos < s.len() && class(s[*pos]) == Tok::Semicolon => {
      *pos += 1;
      (Tok::DSemicolon, start, 2)
    }
    Tok::Colon if *pos < s.len() && class(s[*pos]) == Tok::ColonEq => {
      *pos += 1;
      (Tok::ColonEq, start, 2)
    }
    // a lone '=' (GetFEM: GA_INVALID)
    Tok::ColonEq => (Tok::Invalid, start, 1),
    t => (t, start, 1),
  }
}
