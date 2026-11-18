//! Reserved GWFL names (GetFEM's function and operator tables).

/// Scalar function value: one or two arguments (None: no evaluator yet).
#[derive(Clone, Copy, Debug)]
pub enum Eval {
  One(fn(f64) -> f64),
  Two(fn(f64, f64) -> f64),
  None(usize),
}

impl Eval {
  pub fn nargs(self) -> usize {
    match self {
      Eval::One(_) => 1,
      Eval::Two(_) => 2,
      Eval::None(n) => n,
    }
  }
}

/// GetFEM's predefined scalar functions: name, value, derivative names
/// (a function name, a GWFL expression of t and u, or "").
pub struct Function {
  pub name: &'static str,
  pub eval: Eval,
  pub der1: &'static str,
  pub der2: &'static str,
}

fn heaviside(t: f64) -> f64 {
  if t >= 0.0 { 1.0 } else { 0.0 }
}
fn pos_part(t: f64) -> f64 {
  if t >= 0.0 { t } else { 0.0 }
}
fn neg_part(t: f64) -> f64 {
  if t >= 0.0 { 0.0 } else { -t }
}
fn reg_pos_part(t: f64, e: f64) -> f64 {
  if t >= e {
    t - e / 2.0
  } else if t <= 0.0 {
    0.0
  } else {
    t * t / (2.0 * e)
  }
}
fn sinc(t: f64) -> f64 {
  if t.abs() < 1e-4 {
    let t2 = t * t;
    1.0 - t2 / 6.0 + t2 * t2 / 120.0
  } else {
    t.sin() / t
  }
}
fn der_sinc(t: f64) -> f64 {
  if t.abs() < 1e-4 {
    let t2 = t * t;
    -t / 3.0 + t * t2 / 30.0 - t * t2 * t2 / 840.0
  } else {
    (t * t.cos() - t.sin()) / (t * t)
  }
}
fn der2_sinc(t: f64) -> f64 {
  if t.abs() < 1e-4 {
    let t2 = t * t;
    -1.0 / 3.0 + t2 / 10.0 - t2 * t2 / 168.0
  } else {
    ((2.0 - t * t) * t.sin() - 2.0 * t * t.cos()) / (t * t * t)
  }
}
// asin/acos/atan through atan2 (project rule)
fn asin(t: f64) -> f64 {
  t.atan2((1.0 - t * t).sqrt())
}
fn acos(t: f64) -> f64 {
  (1.0 - t * t).sqrt().atan2(t)
}
fn atan(t: f64) -> f64 {
  t.atan2(1.0)
}

macro_rules! f {
  ($n:literal, $e:expr, $d1:literal, $d2:literal) => {
    Function {
      name: $n,
      eval: $e,
      der1: $d1,
      der2: $d2,
    }
  };
}

use Eval::{None as Nev, One, Two};

pub static FUNCTIONS: &[Function] = &[
  f!("sqrt", One(f64::sqrt), "0.5/sqrt(t)", ""),
  f!("sqr", One(|t| t * t), "2*t", ""),
  f!(
    "pow",
    Two(f64::powf),
    "u*pow(t,u-1)",
    "pow(t,u)*log(abs(t))"
  ),
  f!(
    "DER_PDFUNC_SQRT",
    One(|t| 0.5 / t.sqrt()),
    "-0.25/(t*sqrt(t))",
    ""
  ),
  f!(
    "DER_PDFUNC1_POW",
    Two(|t, u| u * t.powf(u - 1.0)),
    "u*(u-1)*pow(t,u-2)",
    "pow(t,u-1)*(u*log(t)+1)"
  ),
  f!(
    "DER_PDFUNC2_POW",
    Two(|t, u| t.powf(u) * t.abs().ln()),
    "pow(t,u-1)*(u*log(t)+1)",
    "pow(t,u)*sqr(log(t))"
  ),
  f!("exp", One(f64::exp), "exp", ""),
  f!("log", One(f64::ln), "1/t", ""),
  f!("log10", One(f64::log10), "DER_PDFUNC_LOG10", ""),
  f!("sinh", One(f64::sinh), "cosh", ""),
  f!("cosh", One(f64::cosh), "sinh", ""),
  f!("tanh", One(f64::tanh), "DER_PDFUNC_TANH", ""),
  f!("asinh", One(f64::asinh), "DER_PDFUNC_ASINH", ""),
  f!("acosh", One(f64::acosh), "DER_PDFUNC_ACOSH", ""),
  f!("atanh", One(f64::atanh), "DER_PDFUNC_ATANH", ""),
  f!("DER_PDFUNC_LOG", One(|t| 1.0 / t), "-1/sqr(t)", ""),
  f!(
    "DER_PDFUNC_LOG10",
    One(|t| 1.0 / (t * 10f64.ln())),
    "-1/(sqr(t)*log(10))",
    ""
  ),
  f!(
    "DER_PDFUNC_TANH",
    One(|t| 1.0 - t.tanh() * t.tanh()),
    "2*tanh(t)*(sqr(tanh(t))-1)",
    ""
  ),
  f!(
    "DER_PDFUNC_ASINH",
    One(|t| 1.0 / (t * t + 1.0).sqrt()),
    "-t/(pow(t*t+1,1.5))",
    ""
  ),
  f!(
    "DER_PDFUNC_ACOSH",
    One(|t| 1.0 / (t * t - 1.0).sqrt()),
    "-t/(pow(t*t-1,1.5))",
    ""
  ),
  f!(
    "DER_PDFUNC_ATANH",
    One(|t| 1.0 / (1.0 - t * t)),
    "2*t/sqr(1-t*t)",
    ""
  ),
  f!("sin", One(f64::sin), "cos", ""),
  f!("cos", One(f64::cos), "DER_PDFUNC_COS", ""),
  f!("tan", One(f64::tan), "DER_PDFUNC_TAN", ""),
  f!("asin", One(asin), "DER_PDFUNC_ASIN", ""),
  f!("acos", One(acos), "DER_PDFUNC_ACOS", ""),
  f!("atan", One(atan), "DER_PDFUNC_ATAN", ""),
  f!(
    "atan2",
    Two(f64::atan2),
    "DER_PDFUNC1_ATAN2",
    "DER_PDFUNC2_ATAN2"
  ),
  f!("sinc", One(sinc), "DER_PDFUNC_SINC", ""),
  f!("DER_PDFUNC_SINC", One(der_sinc), "DER2_PDFUNC_SINC", ""),
  f!("DER2_PDFUNC_SINC", One(der2_sinc), "", ""),
  f!("DER_PDFUNC_COS", One(|t| -t.sin()), "-cos(t)", ""),
  f!(
    "DER_PDFUNC_TAN",
    One(|t| 1.0 + t.tan() * t.tan()),
    "2*tan(t)/sqr(cos(t))",
    ""
  ),
  f!(
    "DER_PDFUNC_ASIN",
    One(|t| 1.0 / (1.0 - t * t).sqrt()),
    "t/(pow(1-t*t,1.5))",
    ""
  ),
  f!(
    "DER_PDFUNC_ACOS",
    One(|t| -1.0 / (1.0 - t * t).sqrt()),
    "-t/(pow(1-t*t,1.5))",
    ""
  ),
  f!(
    "DER_PDFUNC_ATAN",
    One(|t| 1.0 / (1.0 + t * t)),
    "-2*t/sqr(1+t*t)",
    ""
  ),
  f!(
    "DER_PDFUNC1_ATAN2",
    Two(|t, u| u / (t * t + u * u)),
    "-2*t*u/sqr(sqr(u)+sqr(t))",
    "(sqr(t)-sqr(u))/sqr(sqr(u)+sqr(t))"
  ),
  f!(
    "DER_PDFUNC2_ATAN2",
    Two(|t, u| -t / (t * t + u * u)),
    "(sqr(t)-sqr(u))/sqr(sqr(u)+sqr(t))",
    "2*t*u/sqr(sqr(u)+sqr(t))"
  ),
  f!("erf", Nev(1), "DER_PDFUNC_ERF", ""),
  f!("erfc", Nev(1), "DER_PDFUNC_ERFC", ""),
  f!(
    "DER_PDFUNC_ERF",
    One(|t| (-t * t).exp() * 2.0 / std::f64::consts::PI.sqrt()),
    "exp(-t*t)*2/sqrt(pi)",
    ""
  ),
  f!(
    "DER_PDFUNC_ERFC",
    One(|t| -(-t * t).exp() * 2.0 / std::f64::consts::PI.sqrt()),
    "-exp(-t*t)*2/sqrt(pi)",
    ""
  ),
  f!("Heaviside", One(heaviside), "", ""),
  f!("sign", One(|t| if t >= 0.0 { 1.0 } else { -1.0 }), "", ""),
  f!("abs", One(f64::abs), "sign", ""),
  f!("pos_part", One(pos_part), "Heaviside", ""),
  f!(
    "sqr_pos_part",
    One(|t| if t >= 0.0 { t * t } else { 0.0 }),
    "2*pos_part(t)",
    ""
  ),
  f!(
    "half_sqr_pos_part",
    One(|t| if t >= 0.0 { 0.5 * t * t } else { 0.0 }),
    "pos_part",
    ""
  ),
  f!("neg_part", One(neg_part), "DER_PDFUNC_NEG_PART", ""),
  f!(
    "sqr_neg_part",
    One(|t| if t >= 0.0 { 0.0 } else { t * t }),
    "-2*neg_part(t)",
    ""
  ),
  f!(
    "half_sqr_neg_part",
    One(|t| if t >= 0.0 { 0.0 } else { 0.5 * t * t }),
    "-neg_part(t)",
    ""
  ),
  f!("reg_pos_part", Two(reg_pos_part), "DER_REG_POS_PART", ""),
  f!(
    "DER_REG_POS_PART",
    Two(|t, e| if t >= e {
      1.0
    } else if t <= 0.0 {
      0.0
    } else {
      t / e
    }),
    "DER2_REG_POS_PART",
    ""
  ),
  f!(
    "DER2_REG_POS_PART",
    Two(|t, e| if t >= e || t <= 0.0 { 0.0 } else { 1.0 / e }),
    "",
    ""
  ),
  f!("max", Two(f64::max), "DER_PDFUNC1_MAX", "DER_PDFUNC2_MAX"),
  f!("min", Two(f64::min), "DER_PDFUNC2_MAX", "DER_PDFUNC1_MAX"),
  f!(
    "DER_PDFUNC_NEG_PART",
    One(|t| if t >= 0.0 { 0.0 } else { -1.0 }),
    "",
    ""
  ),
  f!(
    "DER_PDFUNC1_MAX",
    Two(|t, u| if t - u >= 0.0 { 1.0 } else { 0.0 }),
    "",
    ""
  ),
  f!(
    "DER_PDFUNC2_MAX",
    Two(|t, u| if u - t >= 0.0 { 1.0 } else { 0.0 }),
    "",
    ""
  ),
];

pub fn function(name: &str) -> Option<&'static Function> {
  FUNCTIONS.iter().find(|f| f.name == name)
}

/// pi, meshdim, qdim(u), qdims(u), Id(n) (GetFEM's timestep: no time in
/// GWFL, a time step is ordinary data, SPEC 6.3)
pub const SPEC_FUNCTIONS: &[&str] = &["pi", "meshdim", "qdim", "qdims", "Id"];

pub const SPEC_OPS: &[&str] = &[
  "X",
  "element_size",
  "element_K",
  "element_B",
  "element_curvature",
  "Normal",
  "Sym",
  "Skew",
  "Def",
  "Trace",
  "Deviator",
  "Interpolate",
  "Interpolate_filter",
  "Elementary_transformation",
  "Xfem_plus",
  "Xfem_minus",
  "Print",
  "Reshape",
  "Swap_indices",
  "Index_move_last",
  "Contract",
  "Diff",
  "Grad",
];

/// GetFEM's nonlinear operators; those without an implementation here are
/// refused when analysed (not ported).
pub static OPERATORS: &[&str] = &[
  "Norm",
  "Norm_sqr",
  "Det",
  "Inv",
  "Expm",
  "Logm",
  "Matrix_i2",
  "Matrix_j1",
  "Matrix_j2",
  "Normalized",
  "Normalized_reg",
  "Ball_projection",
  "Von_Mises_projection",
  "Right_Cauchy_Green",
  "Left_Cauchy_Green",
  "Green_Lagrangian",
  "Cauchy_stress_from_PK2",
  "Transformed_unit_vector",
  "Coulomb_friction_coupled_projection",
];

/// Suffixes of the hyperelastic laws (with an optional Plane_Strain_ prefix).
const LAWS: &[&str] = &[
  "Saint_Venant_Kirchhoff",
  "Generalized_Blatz_Ko",
  "Ciarlet_Geymonat",
  "Incompressible_Mooney_Rivlin",
  "Compressible_Mooney_Rivlin",
  "Incompressible_Neo_Hookean",
  "Compressible_Neo_Hookean",
  "Compressible_Neo_Hookean_Bonet",
  "Compressible_Neo_Hookean_Ciarlet",
];

pub fn is_law(name: &str) -> bool {
  let n = name.strip_prefix("Plane_Strain_").unwrap_or(name);
  ["_sigma", "_PK2", "_potential"]
    .iter()
    .any(|s| n.strip_suffix(s).is_some_and(|l| LAWS.contains(&l)))
}

/// GetFEM ga_check_name_validity: 1 function/operator, 2 reserved prefix.
pub fn reserved(name: &str) -> Option<u8> {
  if name.starts_with("Derivative_") {
    return Some(2);
  }
  if function(name).is_some()
    || SPEC_FUNCTIONS.contains(&name)
    || SPEC_OPS.contains(&name)
    || OPERATORS.contains(&name)
    || is_law(name)
  {
    return Some(1);
  }
  if ["Grad_", "Hess_", "Div_", "Test2_", "Test_"]
    .iter()
    .any(|p| name.starts_with(p))
  {
    return Some(2);
  }
  None
}
