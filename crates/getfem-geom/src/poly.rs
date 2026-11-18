//! Multivariate polynomials and rational fractions (GetFEM's bgeot_poly).
//!
//! Monomial order as in GetFEM: by total degree, then graded as documented
//! in `bgeot_poly.h`, so coefficient vectors are interchangeable.

use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::{Error, Result};
use getfem_linalg::{Real, Scalar};

/// Number of monomials of n variables with degree <= d: (n+d)!/(n!d!)
/// (tabulated for small n, d).
pub fn alpha(n: usize, d: usize) -> usize {
  if n < ALPHA_N && d < ALPHA_D {
    return ALPHA[n][d];
  }
  alpha_loop(n, d)
}

const fn alpha_loop(n: usize, d: usize) -> usize {
  let mut r = 1usize;
  let mut j = 1;
  while j <= d {
    r = r * (n + j) / j;
    j += 1;
  }
  r
}

/// The coefficient indices in the order Horner's scheme reads them, for
/// n variables and degree d (computed once per thread).
fn horner_order<R>(n: usize, d: usize, f: impl FnOnce(&[usize]) -> R) -> R {
  fn visit(mi: &mut PowerIndex, k: usize, de: usize, d: usize, out: &mut Vec<usize>) {
    if k == 0 {
      out.push(mi.global_index());
      return;
    }
    for m in (0..=d - de).rev() {
      mi[k - 1] = m;
      visit(mi, k - 1, de + m, d, out);
    }
    mi[k - 1] = 0;
  }
  type Order = std::rc::Rc<[usize]>;
  let build = || -> Order {
    let mut out = Vec::with_capacity(alpha(n, d));
    visit(&mut PowerIndex::new(n), n, 0, d, &mut out);
    out.into()
  };
  // small n, d in a table, the others in a map
  type Orders = [[Option<Order>; ORDER_D]; ORDER_N];
  thread_local! {
    static TABLE: std::cell::RefCell<Orders> = Default::default();
    static MAP: std::cell::RefCell<std::collections::HashMap<(usize, usize), Order>> =
      Default::default();
  }
  let order = if n < ORDER_N && d < ORDER_D {
    TABLE.with(|o| o.borrow_mut()[n][d].get_or_insert_with(build).clone())
  } else {
    MAP.with(|o| o.borrow_mut().entry((n, d)).or_insert_with(build).clone())
  };
  f(&order)
}

const ORDER_N: usize = 4;
const ORDER_D: usize = 16;

const ALPHA_N: usize = 8;
const ALPHA_D: usize = 32;
static ALPHA: [[usize; ALPHA_D]; ALPHA_N] = {
  let mut t = [[0; ALPHA_D]; ALPHA_N];
  let mut n = 0;
  while n < ALPHA_N {
    let mut d = 0;
    while d < ALPHA_D {
      t[n][d] = alpha_loop(n, d);
      d += 1;
    }
    n += 1;
  }
  t
};

/// Powers of a monomial.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PowerIndex(Vec<usize>);

impl PowerIndex {
  pub fn new(n: usize) -> Self {
    PowerIndex(vec![0; n])
  }

  pub fn from_powers(p: &[usize]) -> Self {
    PowerIndex(p.to_vec())
  }

  pub fn dim(&self) -> usize {
    self.0.len()
  }

  pub fn powers(&self) -> &[usize] {
    &self.0
  }

  pub fn degree(&self) -> usize {
    self.0.iter().sum()
  }

  /// Position of the monomial in a coefficient vector.
  pub fn global_index(&self) -> usize {
    let mut d = self.degree();
    let mut n = self.dim();
    let mut gi = 0;
    for &p in &self.0 {
      if d == 0 {
        break;
      }
      gi += alpha(n, d - 1);
      d -= p;
      n -= 1;
    }
    gi
  }

  /// Advance to the next monomial.
  pub fn next(&mut self) {
    let n = self.dim();
    if n == 0 {
      return;
    }
    let l = (0..n - 1).rev().find(|&l| self.0[l] != 0);
    let a = self.0[n - 1];
    self.0[n - 1] = 0;
    match l {
      Some(l) => {
        self.0[l + 1] = a + 1;
        self.0[l] -= 1;
      }
      None => self.0[0] = a + 1,
    }
  }

  /// Go back to the previous monomial; no-op on the constant monomial.
  pub fn prev(&mut self) {
    let n = self.dim();
    if let Some(l) = (0..n).rev().find(|&l| self.0[l] != 0) {
      let a = self.0[l];
      self.0[l] = 0;
      self.0[n - 1] = a - 1;
      if l > 0 {
        self.0[l - 1] += 1;
      }
    }
  }
}

impl std::ops::Index<usize> for PowerIndex {
  type Output = usize;
  fn index(&self, i: usize) -> &usize {
    &self.0[i]
  }
}

impl std::ops::IndexMut<usize> for PowerIndex {
  fn index_mut(&mut self, i: usize) -> &mut usize {
    &mut self.0[i]
  }
}

/// Polynomial in n variables, degree d, dense coefficients.
#[derive(Clone, Debug)]
pub struct Polynomial<T> {
  n: usize,
  d: usize,
  c: Vec<T>,
}

impl<T: Scalar> Polynomial<T> {
  pub fn zero(n: usize) -> Self {
    Self::new(n, 0)
  }

  /// Zero polynomial with room for degree d.
  pub fn new(n: usize, d: usize) -> Self {
    Polynomial {
      n,
      d,
      c: vec![T::zero(); alpha(n, d)],
    }
  }

  pub fn constant(n: usize, v: T) -> Self {
    let mut p = Self::new(n, 0);
    p.c[0] = v;
    p
  }

  pub fn one(n: usize) -> Self {
    Self::constant(n, T::one())
  }

  /// The variable x_k.
  pub fn var(n: usize, k: usize) -> Self {
    assert!(k < n, "variable index out of range");
    let mut p = Self::new(n, 1);
    p.c[k + 1] = T::one();
    p
  }

  pub fn from_coeffs(n: usize, d: usize, c: Vec<T>) -> Result<Self> {
    if c.len() != alpha(n, d) {
      return Err(Error::InvalidArgument(format!(
        "polynomial of dim {n}, degree {d} needs {} coefficients, got {}",
        alpha(n, d),
        c.len()
      )));
    }
    Ok(Polynomial { n, d, c })
  }

  pub fn dim(&self) -> usize {
    self.n
  }

  pub fn degree(&self) -> usize {
    self.d
  }

  pub fn coeffs(&self) -> &[T] {
    &self.c
  }

  /// Degree ignoring trailing zero coefficients.
  pub fn real_degree(&self) -> usize {
    let l = self
      .c
      .iter()
      .rposition(|&x| x != T::zero())
      .map_or(0, |i| i + 1);
    let mut dd = self.d;
    while dd > 0 && alpha(self.n, dd - 1) >= l {
      dd -= 1;
    }
    dd
  }

  pub fn is_zero(&self) -> bool {
    self.c.iter().all(|&x| x == T::zero())
  }

  pub fn change_degree(&mut self, dd: usize) {
    self.c.resize(alpha(self.n, dd), T::zero());
    self.d = dd;
  }

  pub fn add_monomial(&mut self, coeff: T, p: &PowerIndex) {
    assert_eq!(p.dim(), self.n, "dimensions mismatch");
    let i = p.global_index();
    if i >= self.c.len() {
      self.change_degree(p.degree());
    }
    self.c[i] += coeff;
  }

  /// Iterate over non-zero monomials as (powers, coefficient).
  fn monomials(&self) -> impl Iterator<Item = (PowerIndex, T)> + '_ {
    let mut pi = PowerIndex::new(self.n);
    self.c.iter().filter_map(move |&c| {
      let cur = pi.clone();
      pi.next();
      (c != T::zero()).then_some((cur, c))
    })
  }

  /// Product considering that the variables of q follow those of self.
  pub fn direct_product(&self, q: &Self) -> Self {
    let mut r = Self::zero(self.n + q.n);
    for (pq, cq) in q.monomials() {
      for (pa, ca) in self.monomials() {
        let mut v = pa.0.clone();
        v.extend_from_slice(&pq.0);
        r.add_monomial(ca * cq, &PowerIndex(v));
      }
    }
    r
  }

  /// Derivative with respect to x_k, in place.
  pub fn derivative(&mut self, k: usize) {
    assert!(k < self.n, "index out of range");
    let mut r = Self::new(self.n, self.d.saturating_sub(1));
    for (mut p, c) in self.monomials() {
      if p[k] > 0 {
        let e = p[k];
        p[k] -= 1;
        r.c[p.global_index()] = c * T::from_f64(e as f64);
      }
    }
    *self = r;
  }

  pub fn derived(&self, k: usize) -> Self {
    let mut r = self.clone();
    r.derivative(k);
    r
  }

  /// Value at x (x.len() >= dim), as GetFEM: direct formulas up to degree
  /// 6 in 1 to 3 variables, Horner otherwise.
  pub fn eval(&self, x: &[T]) -> T {
    match self.d {
      0 => self.c[0],
      1 => (0..self.n).fold(self.c[0], |s, i| s + x[i] * self.c[i + 1]),
      _ => self.eval_direct(x).unwrap_or_else(|| {
        horner_order(self.n, self.d, |order| match self.n {
          1..=3 => self.horner_loops(order, x),
          _ => self.horner(&mut order.iter(), self.n, 0, x),
        })
      }),
    }
  }

  /// horner for 1 to 3 variables as nested loops (the same operations)
  fn horner_loops(&self, order: &[usize], x: &[T]) -> T {
    let (d, mut it) = (self.d, order.iter());
    let mut leaf = || self.c[*it.next().expect("horner order")];
    let mut inner = |e: usize| {
      // level 1: x[0]
      let mut r = T::zero();
      for _ in 0..=d - e {
        r = leaf() + x[0] * r;
      }
      r
    };
    match self.n {
      1 => inner(0),
      2 => {
        let mut r = T::zero();
        for m in (0..=d).rev() {
          r = inner(m) + x[1] * r;
        }
        r
      }
      _ => {
        let mut r3 = T::zero();
        for m3 in (0..=d).rev() {
          let mut r2 = T::zero();
          for m2 in (0..=d - m3).rev() {
            r2 = inner(m3 + m2) + x[1] * r2;
          }
          r3 = r2 + x[2] * r3;
        }
        r3
      }
    }
  }

  /// GetFEM's Horner scheme; the coefficients in the order it reads them
  fn horner(&self, order: &mut std::slice::Iter<usize>, k: usize, de: usize, x: &[T]) -> T {
    if k == 0 {
      return self.c[*order.next().expect("horner order")];
    }
    let v = x[k - 1];
    let mut res = T::zero();
    for m in (0..=self.d - de).rev() {
      res = self.horner(order, k - 1, de + m, x) + v * res;
    }
    res
  }

  /// Substitute x_k by the univariate polynomial s.
  pub fn substitute_var(&self, s: &Self, k: usize) -> Self {
    assert!(s.n == 1 && k < self.n, "wrong arguments for substitution");
    let mut res = Self::zero(self.n);
    let mut spow = vec![Polynomial::one(1)];
    for (p, c) in self.monomials() {
      while p[k] >= spow.len() {
        let next = s * spow.last().unwrap();
        spow.push(next);
      }
      let mut p2 = p.clone();
      for (i, &sc) in spow[p[k]].c.iter().enumerate() {
        p2[k] = i;
        res.add_monomial(sc * c, &p2);
      }
    }
    res
  }

  fn check_dim(&self, q: &Self) {
    assert_eq!(self.n, q.n, "dimensions mismatch");
  }
}

impl<T: Scalar> PartialEq for Polynomial<T> {
  fn eq(&self, q: &Self) -> bool {
    let z = T::zero();
    let m = self.c.len().max(q.c.len());
    self.n == q.n && (0..m).all(|i| self.c.get(i).unwrap_or(&z) == q.c.get(i).unwrap_or(&z))
  }
}

impl<T: Scalar> AddAssign<&Polynomial<T>> for Polynomial<T> {
  fn add_assign(&mut self, q: &Self) {
    self.check_dim(q);
    if q.d > self.d {
      self.change_degree(q.d);
    }
    for (a, &b) in self.c.iter_mut().zip(&q.c) {
      *a += b;
    }
  }
}

impl<T: Scalar> SubAssign<&Polynomial<T>> for Polynomial<T> {
  fn sub_assign(&mut self, q: &Self) {
    self.check_dim(q);
    if q.d > self.d {
      self.change_degree(q.d);
    }
    for (a, &b) in self.c.iter_mut().zip(&q.c) {
      *a -= b;
    }
  }
}

impl<T: Scalar> Mul for &Polynomial<T> {
  type Output = Polynomial<T>;
  // exponents add when monomials multiply
  #[allow(clippy::suspicious_arithmetic_impl)]
  fn mul(self, q: Self) -> Polynomial<T> {
    self.check_dim(q);
    let mut r = Polynomial::zero(self.n);
    for (pq, cq) in q.monomials() {
      for (pa, ca) in self.monomials() {
        let v = pa.0.iter().zip(&pq.0).map(|(a, b)| a + b).collect();
        r.add_monomial(ca * cq, &PowerIndex(v));
      }
    }
    r
  }
}

impl<T: Scalar> MulAssign<&Polynomial<T>> for Polynomial<T> {
  fn mul_assign(&mut self, q: &Self) {
    *self = &*self * q;
  }
}

impl<T: Scalar> MulAssign<T> for Polynomial<T> {
  fn mul_assign(&mut self, e: T) {
    self.c.iter_mut().for_each(|a| *a *= e);
  }
}

impl<T: Scalar> DivAssign<T> for Polynomial<T> {
  fn div_assign(&mut self, e: T) {
    self.c.iter_mut().for_each(|a| *a /= e);
  }
}

impl<T: Scalar> Neg for &Polynomial<T> {
  type Output = Polynomial<T>;
  fn neg(self) -> Polynomial<T> {
    let mut r = self.clone();
    r.c.iter_mut().for_each(|a| *a = -*a);
    r
  }
}

impl<T: Scalar> Add for &Polynomial<T> {
  type Output = Polynomial<T>;
  fn add(self, q: Self) -> Polynomial<T> {
    let mut r = self.clone();
    r += q;
    r
  }
}

impl<T: Scalar> Sub for &Polynomial<T> {
  type Output = Polynomial<T>;
  fn sub(self, q: Self) -> Polynomial<T> {
    let mut r = self.clone();
    r -= q;
    r
  }
}

impl<T: Scalar> Mul<T> for &Polynomial<T> {
  type Output = Polynomial<T>;
  fn mul(self, e: T) -> Polynomial<T> {
    let mut r = self.clone();
    r *= e;
    r
  }
}

impl<T: Scalar> Div<T> for &Polynomial<T> {
  type Output = Polynomial<T>;
  fn div(self, e: T) -> Polynomial<T> {
    let mut r = self.clone();
    r /= e;
    r
  }
}

const VARS: &[u8] = b"xyzwvut";

impl<T: Real> fmt::Display for Polynomial<T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let mut first = true;
    for (p, c) in self.monomials() {
      if !first {
        write!(f, "{}", if c < T::zero() { " - " } else { " + " })?;
      } else if c < T::zero() {
        write!(f, "-")?;
      }
      first = false;
      let a = c.abs();
      let mut first_var = true;
      if a != T::one() || p.degree() == 0 {
        write!(f, "{a}")?;
        first_var = false;
      }
      for (j, &e) in p.powers().iter().enumerate() {
        if e == 0 {
          continue;
        }
        if !first_var {
          write!(f, "*")?;
        }
        first_var = false;
        if self.n <= VARS.len() {
          write!(f, "{}", VARS[j] as char)?;
        } else {
          write!(f, "x_{j}")?;
        }
        if e > 1 {
          write!(f, "^{e}")?;
        }
      }
    }
    if first {
      write!(f, "0")?;
    }
    Ok(())
  }
}

/// Rational fraction numerator / denominator.
#[derive(Clone, Debug)]
pub struct RationalFraction<T> {
  num: Polynomial<T>,
  den: Polynomial<T>,
}

impl<T: Scalar> PartialEq for RationalFraction<T> {
  fn eq(&self, q: &Self) -> bool {
    self.num == q.num && self.den == q.den
  }
}

impl<T: Scalar> RationalFraction<T> {
  pub fn new(num: Polynomial<T>, den: Polynomial<T>) -> Self {
    assert_eq!(num.dim(), den.dim(), "dimensions mismatch");
    RationalFraction { num, den }
  }

  pub fn from_poly(num: Polynomial<T>) -> Self {
    let den = Polynomial::one(num.dim());
    RationalFraction { num, den }
  }

  pub fn numerator(&self) -> &Polynomial<T> {
    &self.num
  }

  pub fn denominator(&self) -> &Polynomial<T> {
    &self.den
  }

  pub fn dim(&self) -> usize {
    self.num.dim()
  }

  /// Product considering that the variables of q follow those of self.
  pub fn direct_product(&self, q: &Self) -> Self {
    RationalFraction {
      num: self.num.direct_product(&q.num),
      den: self.den.direct_product(&q.den),
    }
  }

  pub fn derivative(&mut self, k: usize) {
    let dn = self.num.derived(k);
    let dd = self.den.derived(k);
    if dd.is_zero() {
      self.num = dn;
    } else {
      self.num = &(&dn * &self.den) - &(&dd * &self.num);
      self.den = &self.den * &self.den;
    }
  }

  pub fn derived(&self, k: usize) -> Self {
    let mut r = self.clone();
    r.derivative(k);
    r
  }

  /// Evaluate; at a pole (pyramid apex), the limit along the segment toward
  /// the origin (toward (1,..,1)/n from the origin), by the lowest-order
  /// Taylor coefficients of numerator and denominator on that segment.
  pub fn eval(&self, x: &[T]) -> T {
    // a polynomial (a / 1 = a): the numerator only
    if self.den.d == 0 && self.den.c[0] == T::one() {
      return self.num.eval(x);
    }
    let (a, b) = (self.num.eval(x), self.den.eval(x));
    if b != T::zero() {
      return if a == T::zero() { a } else { a / b };
    }
    let n = self.dim();
    let origin = x[..n].iter().all(|&v| v == T::zero());
    let dir: Vec<T> = if origin {
      vec![T::from_f64(1.0 / n as f64); n]
    } else {
      x[..n].iter().map(|&v| -v).collect()
    };
    let (gn, gd) = (
      self.num.on_line(&x[..n], &dir),
      self.den.on_line(&x[..n], &dir),
    );
    let scale = |c: &[T]| {
      c.iter()
        .map(|v| v.abs())
        .fold(<T::Real as Scalar>::zero(), |m, v| m.max(v))
    };
    let tol = <T::Real as Scalar>::from_f64(1e-12);
    let first = |c: &[T]| {
      let s = scale(c) * tol;
      c.iter().position(|v| v.abs() > s)
    };
    match (first(&gn), first(&gd)) {
      (None, _) => T::zero(),
      (Some(i), Some(j)) if i > j => T::zero(),
      (Some(i), Some(j)) if i == j => gn[i] / gd[j],
      _ => T::from_f64(f64::INFINITY),
    }
  }
}

impl<T: Scalar> Polynomial<T> {
  /// Coefficients in s of P(a + s b).
  pub fn on_line(&self, a: &[T], b: &[T]) -> Vec<T> {
    let mut out = vec![T::zero(); self.d + 1];
    for (p, c) in self.monomials() {
      let mut q = vec![c];
      for (i, &e) in p.powers().iter().enumerate() {
        for _ in 0..e {
          let mut r = vec![T::zero(); q.len() + 1];
          for (k, &qk) in q.iter().enumerate() {
            r[k] += qk * a[i];
            r[k + 1] += qk * b[i];
          }
          q = r;
        }
      }
      for (k, v) in q.into_iter().enumerate() {
        out[k] += v;
      }
    }
    out
  }
}

impl<T: Scalar> Add for &RationalFraction<T> {
  type Output = RationalFraction<T>;
  fn add(self, q: Self) -> RationalFraction<T> {
    RationalFraction {
      num: &(&self.num * &q.den) + &(&q.num * &self.den),
      den: &self.den * &q.den,
    }
  }
}

impl<T: Scalar> Sub for &RationalFraction<T> {
  type Output = RationalFraction<T>;
  fn sub(self, q: Self) -> RationalFraction<T> {
    RationalFraction {
      num: &(&self.num * &q.den) - &(&q.num * &self.den),
      den: &self.den * &q.den,
    }
  }
}

impl<T: Scalar> Mul for &RationalFraction<T> {
  type Output = RationalFraction<T>;
  fn mul(self, q: Self) -> RationalFraction<T> {
    RationalFraction {
      num: &self.num * &q.num,
      den: &self.den * &q.den,
    }
  }
}

impl<T: Scalar> Div for &RationalFraction<T> {
  type Output = RationalFraction<T>;
  fn div(self, q: Self) -> RationalFraction<T> {
    RationalFraction {
      num: &self.num * &q.den,
      den: &self.den * &q.num,
    }
  }
}

impl<T: Scalar> From<Polynomial<T>> for RationalFraction<T> {
  fn from(p: Polynomial<T>) -> Self {
    Self::from_poly(p)
  }
}

impl<T: Scalar> Mul<T> for &RationalFraction<T> {
  type Output = RationalFraction<T>;
  fn mul(self, e: T) -> RationalFraction<T> {
    RationalFraction {
      num: &self.num * e,
      den: self.den.clone(),
    }
  }
}

impl<T: Real> fmt::Display for RationalFraction<T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "[{}]/[{}]", self.num, self.den)
  }
}

/// Parse a polynomial in n variables named x, y, z, w, v, u, t.
pub fn parse<T: Real>(n: usize, s: &str) -> Result<Polynomial<T>> {
  let mut p = Parser {
    s: s.as_bytes(),
    pos: 0,
    n,
  };
  let r = p.expr()?;
  p.skip_ws();
  if p.peek() == Some(b';') {
    p.pos += 1;
    p.skip_ws();
  }
  if p.pos != p.s.len() {
    return Err(p.err("unexpected trailing input"));
  }
  Ok(r)
}

struct Parser<'a> {
  s: &'a [u8],
  pos: usize,
  n: usize,
}

impl Parser<'_> {
  fn err(&self, msg: &str) -> Error {
    Error::Parse(format!(
      "polynomial '{}' at {}: {msg}",
      String::from_utf8_lossy(self.s),
      self.pos
    ))
  }

  fn skip_ws(&mut self) {
    while self.peek().is_some_and(|c| c.is_ascii_whitespace()) {
      self.pos += 1;
    }
  }

  fn peek(&self) -> Option<u8> {
    self.s.get(self.pos).copied()
  }

  fn eat(&mut self, c: u8) -> bool {
    self.skip_ws();
    let ok = self.peek() == Some(c);
    if ok {
      self.pos += 1;
    }
    ok
  }

  // expr := [+|-] term {(+|-) term}
  fn expr<T: Real>(&mut self) -> Result<Polynomial<T>> {
    let neg = if self.eat(b'-') {
      true
    } else {
      self.eat(b'+');
      false
    };
    let mut r = self.term()?;
    if neg {
      r = -&r;
    }
    loop {
      if self.eat(b'+') {
        r += &self.term()?;
      } else if self.eat(b'-') {
        r -= &self.term()?;
      } else {
        return Ok(r);
      }
    }
  }

  // term := power {(*|/) power}
  fn term<T: Real>(&mut self) -> Result<Polynomial<T>> {
    let mut r = self.power()?;
    loop {
      if self.eat(b'*') {
        r *= &self.power()?;
      } else if self.eat(b'/') {
        let q = self.power()?;
        if q.real_degree() > 0 {
          return Err(self.err("division by a non-constant"));
        }
        r /= q.coeffs()[0];
      } else {
        return Ok(r);
      }
    }
  }

  // power := primary {^ primary}, left associative as in GetFEM
  fn power<T: Real>(&mut self) -> Result<Polynomial<T>> {
    let mut r = self.primary()?;
    while self.eat(b'^') {
      let e: Polynomial<T> = self.primary()?;
      let v = e.coeffs()[0].to_f64();
      if e.real_degree() > 0 || v < 0.0 || v.fract() != 0.0 {
        return Err(self.err("exponent must be a non-negative integer"));
      }
      let b = r;
      r = Polynomial::one(self.n);
      for _ in 0..v as usize {
        r *= &b;
      }
    }
    Ok(r)
  }

  fn primary<T: Real>(&mut self) -> Result<Polynomial<T>> {
    self.skip_ws();
    let start = self.pos;
    match self.peek() {
      Some(b'(') => {
        self.pos += 1;
        let r = self.expr()?;
        if !self.eat(b')') {
          return Err(self.err("expected ')'"));
        }
        Ok(r)
      }
      Some(c) if c.is_ascii_digit() || c == b'.' => {
        while self.peek().is_some_and(|c| c.is_ascii_digit() || c == b'.') {
          self.pos += 1;
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
          self.pos += 1;
          if matches!(self.peek(), Some(b'+' | b'-')) {
            self.pos += 1;
          }
          while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.pos += 1;
          }
        }
        let t = std::str::from_utf8(&self.s[start..self.pos]).unwrap();
        let v: f64 = t.parse().map_err(|_| self.err("bad number"))?;
        Ok(Polynomial::constant(self.n, T::from_f64(v)))
      }
      Some(c) if c.is_ascii_alphabetic() => {
        while self
          .peek()
          .is_some_and(|c| c.is_ascii_alphanumeric() || c == b'_')
        {
          self.pos += 1;
        }
        let id = &self.s[start..self.pos];
        if id == b"sqrt" {
          let a = self.primary::<T>()?;
          if a.real_degree() > 0 {
            return Err(self.err("sqrt of a non-constant"));
          }
          return Ok(Polynomial::constant(self.n, a.coeffs()[0].sqrt()));
        }
        match VARS.iter().position(|&v| id == [v]) {
          Some(k) if k < self.n => Ok(Polynomial::var(self.n, k)),
          _ => Err(self.err("unknown variable")),
        }
      }
      _ => Err(self.err("expected a number, variable or '('")),
    }
  }
}

mod direct;
#[cfg(test)]
mod tests;
