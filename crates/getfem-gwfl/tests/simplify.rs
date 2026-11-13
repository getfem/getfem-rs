//! Products of scalar powers of the same base merged (x*(1/x) -> 1,
//! pow(x,a)*pow(x,b) -> pow(x,a+b), pow(x,1) -> x, pow(x,0) -> 1): the
//! tree of an expression is the tree of its simplified form.

use getfem_gwfl::defs::{Defs, Support};
use getfem_gwfl::semantic::{Ctx, analyse_str};

fn tree(e: &str) -> String {
  let mut d = Defs::default();
  d.add("u", false, Support::Fem { mesh_dim: 2 }, &[1]);
  d.add("v", false, Support::Fem { mesh_dim: 2 }, &[2]);
  d.add("p", true, Support::Fixed, &[1]);
  let ctx = Ctx {
    meshdim: 2,
    ref_elt_dim: 2,
    option: 1,
    ..Ctx::default()
  };
  analyse_str(e, &d, &ctx).unwrap().0.debug_string()
}

fn same(e: &str, simplified: &str) {
  assert_eq!(tree(e), tree(simplified), "{e} is not {simplified}");
}

#[test]
fn inverse_and_powers() {
  same("u*(1/u)", "1");
  same("u/u", "1");
  same("u*u", "pow(u,2)");
  same("u*u*(1/u)", "u");
  same("pow(u,2)*pow(u,-2)*Grad_u", "Grad_u");
  same("pow(u,1)", "u");
  same("pow(u,0)*Grad_u", "Grad_u");
  // a scalar contraction as a base
  same("(Grad_v:Grad_v)/(Grad_v:Grad_v)", "1");
}

#[test]
fn non_integer_exponents() {
  same("pow(u,0.5)*pow(u,0.5)", "u");
  // 1/3 + 2/3 within an ulp of 1
  same("pow(u,1/3)*pow(u,2/3)", "u");
  same("pow(u,0.1)*pow(u,0.2)", "pow(u,0.1+0.2)");
}

#[test]
fn through_the_chain() {
  // a factor inside a contraction, a denominator, a unary minus
  same(
    "(1/Det(Grad_v))*((Det(Grad_v)*Inv(Grad_v)'):Grad_Test_v)",
    "Inv(Grad_v)':Grad_Test_v",
  );
  same(
    "(1/Det(Grad_v))*(Inv(Grad_v)'/Det(Grad_v))",
    "(1/pow(Det(Grad_v),2))*Inv(Grad_v)'",
  );
  same("u*(-Grad_v/u)", "-Grad_v");
  same("u*Test_u/u", "Test_u");
}

#[test]
fn left_alone() {
  // sums and functions stop the chain; no common base: GetFEM's tree
  assert!(tree("(1/u)*(u+p)").contains("op +"));
  assert!(tree("sin(u)/u").contains("op /"));
  assert!(tree("log(u*u)/u").contains("op /"));
}

#[test]
fn derivatives() {
  // simplified before derivation: no 1/u + u*(-1/u^2) term
  same("Diff(u*pow(u,-1)*Grad_u,u)", "Diff(Grad_u,u)");
  same("Diff(pow(u,3)/u,u)", "Diff(pow(u,2),u)");
}

#[test]
fn log_derivative_a_division() {
  // d log(x) = 1/x (GetFEM: a function DER_PDFUNC_LOG): cancels, and
  // derives any number of times
  same("u*Diff(log(u),u)", "Test_u");
  same("Diff(log(u),u)", "(1/u).*Test_u");
  assert!(tree("Diff(Diff(Diff(log(u),u,1),u,1),u)").contains("Val test u"));
}

#[test]
fn pow_sqrt_derivatives_inlined() {
  // d pow(x,c) = c*pow(x,c-1), merged with the other powers of x
  same("Diff(pow(u,3),u)", "(3*pow(u,2)).*Test_u");
  same("u*u*Diff(pow(u,-1),u)", "(-1).*Test_u");
  same("u*Diff(pow(u,-1),u)", "pow(u,-1)*((-1).*Test_u)");
  same("Diff(sqrt(u),u)", "(0.5/sqrt(u)).*Test_u");
  assert!(tree("Diff(Diff(Diff(sqrt(u),u,1),u,1),u)").contains("Val test u"));
  assert!(tree("Diff(Diff(Diff(pow(u,2.5),u,1),u,1),u)").contains("Val test u"));
  // the third derivative in the exponent needs sign's (none, as GetFEM)
  assert!(tree("Diff(Diff(pow(u,u),u,1),u)").contains("Val test u"));
}
