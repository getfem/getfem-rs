use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Real part of a scalar field.
pub trait Real: Scalar<Real = Self> + PartialOrd {
  fn sqrt(self) -> Self;
  fn atan2(self, x: Self) -> Self;
  fn epsilon() -> Self;
  fn max(self, o: Self) -> Self;
  fn min(self, o: Self) -> Self;
  fn to_f64(self) -> f64;
}

/// Field of the numeric code. v1 validates f64 only.
pub trait Scalar:
  Copy
  + Debug
  + Display
  + PartialEq
  + Default
  + Send
  + Sync
  + 'static
  + Add<Output = Self>
  + Sub<Output = Self>
  + Mul<Output = Self>
  + Div<Output = Self>
  + Neg<Output = Self>
  + AddAssign
  + SubAssign
  + MulAssign
  + DivAssign
  + Sum
{
  type Real: Real;
  fn zero() -> Self;
  fn one() -> Self;
  fn from_real(r: Self::Real) -> Self;
  fn from_f64(x: f64) -> Self;
  fn abs(self) -> Self::Real;
  fn conj(self) -> Self;
}

impl Scalar for f64 {
  type Real = f64;
  fn zero() -> Self {
    0.0
  }
  fn one() -> Self {
    1.0
  }
  fn from_real(r: f64) -> Self {
    r
  }
  fn from_f64(x: f64) -> Self {
    x
  }
  fn abs(self) -> f64 {
    f64::abs(self)
  }
  fn conj(self) -> Self {
    self
  }
}

impl Real for f64 {
  fn sqrt(self) -> Self {
    f64::sqrt(self)
  }
  fn atan2(self, x: Self) -> Self {
    f64::atan2(self, x)
  }
  fn epsilon() -> Self {
    f64::EPSILON
  }
  fn max(self, o: Self) -> Self {
    f64::max(self, o)
  }
  fn min(self, o: Self) -> Self {
    f64::min(self, o)
  }
  fn to_f64(self) -> f64 {
    self
  }
}
