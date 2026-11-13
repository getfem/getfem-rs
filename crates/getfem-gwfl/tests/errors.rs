//! Error messages: position caret, term, shapes; and the debug printer.

use getfem_gwfl::defs::{Defs, Support};
use getfem_gwfl::semantic::{Ctx, analyse_str};

fn defs() -> Defs {
  let mut d = Defs::default();
  d.add("u", false, Support::Fem { mesh_dim: 2 }, &[1]);
  d.add("v", false, Support::Fem { mesh_dim: 2 }, &[2]);
  d.add("p", true, Support::Fixed, &[1]);
  d
}

fn msg(e: &str) -> String {
  let ctx = Ctx {
    meshdim: 2,
    ref_elt_dim: 2,
    option: 1,
    ..Ctx::default()
  };
  analyse_str(e, &defs(), &ctx)
    .expect_err("an error")
    .to_string()
}

#[test]
fn caret_under_the_term() {
  assert_eq!(
    msg("u + 2*v"),
    "u + 2*v\n  |\nerror: Addition or substraction of expressions of different sizes: (1) != (2)"
  );
  assert_eq!(
    msg("Grad_v.[1,2,3]"),
    "Grad_v.[1,2,3]\n      |\nerror: Dot product of expressions of different sizes.(2 != 3).  Sizes of operands: (2, 2) and (3)"
  );
  assert_eq!(
    msg("Test_u*Test_u"),
    "Test_u*Test_u\n      |\nerror: Incompatibility of test functions in product."
  );
  assert_eq!(
    msg("Test_p"),
    "Test_p\n|\nerror: Test functions of constants are not allowed."
  );
  assert_eq!(
    msg("(u+w)*2"),
    "(u+w)*2\n   |\nerror: Unknown variable, function, operator or data \"w\""
  );
  assert_eq!(
    msg("sin(u"),
    "sin(u\n     |\nerror: Parameters should be separated by ',' and parameter list ended by ')'."
  );
}

#[test]
fn unsupported_is_explicit() {
  assert!(msg("Secondary_domain(u)").contains("unsupported: Secondary_domain"));
  // no time dispatchers: an ordinary unknown name, as GetFEM without one
  assert!(
    msg("Previous_u").contains("Unknown variable, function, operator or data \"Previous_u\"")
  );
}

#[test]
fn gradient_of_expressions() {
  let ctx = Ctx {
    meshdim: 2,
    ref_elt_dim: 2,
    option: 1,
    ..Ctx::default()
  };
  let (t, _) = analyse_str("Grad(u*u)", &defs(), &ctx).unwrap();
  assert_eq!(t.n[t.root.unwrap()].sizes, [2]);
  assert!(msg("Grad(Normal)").contains("Gradient of Normal vector not implemented"));
}

#[test]
fn debug_printer() {
  let ctx = Ctx {
    meshdim: 2,
    ref_elt_dim: 2,
    option: 1,
    ..Ctx::default()
  };
  let (t, tests) = analyse_str("Grad_v:Grad_Test_v + 2*[1,2].Test_v", &defs(), &ctx).unwrap();
  let s = t.debug_string();
  assert_eq!(
    s,
    "op + (2) order 1 (v -)
  op : (2) order 1 (v -)
    Plain Grad v (2, 2)
    Plain Grad test v (2, 2, 2) order 1 (v -)
  op . (2) order 1 (v -)
    Constant (2) = [2.0, 4.0]
    Plain Val test v (2, 2) order 1 (v -)
"
  );
  assert_eq!(tests.test1.len(), 1);
}

#[test]
fn user_functions() {
  let d = defs();
  getfem_gwfl::define_gwfl_function("rho", 1, "1/(1+exp(-t))", "", "").unwrap();
  getfem_gwfl::define_gwfl_function("f2", 2, "t*u+rho(t)", "", "").unwrap();
  assert!(getfem_gwfl::define_gwfl_function("sin", 1, "t", "", "").is_err());
  assert!(getfem_gwfl::define_gwfl_function("Grad_q", 1, "t", "", "").is_err());
  assert!(getfem_gwfl::define_gwfl_function("g", 3, "t", "", "").is_err());
  assert!(getfem_gwfl::define_gwfl_function("h", 1, "t+", "", "").is_err());
  // global: a redefinition an error, also the same
  assert!(getfem_gwfl::define_gwfl_function("rho", 1, "1/(1+exp(-t))", "", "").is_err());
  assert!(getfem_gwfl::define_gwfl_function("rho", 1, "1/(1+exp(t))", "", "").is_err());
  assert!(getfem_gwfl::define_gwfl_function("DER1_rho", 1, "t", "", "").is_err());
  let ctx = Ctx {
    meshdim: 2,
    ref_elt_dim: 2,
    option: 1,
    ..Ctx::default()
  };
  let (t, tests) = analyse_str("rho(u)*Grad(Test_u).Grad(u)", &d, &ctx).unwrap();
  assert_eq!(tests.test1.len(), 1);
  assert_eq!(t.n[t.root.unwrap()].sizes, [2]);
  let (t, _) = analyse_str("f2(2, 3)", &d, &ctx).unwrap();
  let want = 6.0 + 1.0 / (1.0 + (-2f64).exp());
  assert!((t.n[t.root.unwrap()].data[0] - want).abs() < 1e-15);
  assert!(analyse_str("rho(u, u)", &d, &ctx).is_err());
  // data does not depend on the position: its gradient is a zero vector
  let (t, _) = analyse_str("Grad(p)", &d, &ctx).unwrap();
  let r = &t.n[t.root.unwrap()];
  assert!(r.ty == getfem_gwfl::semantic::Ty::Zero && r.sizes == [2]);
}

/// derivatives GetFEM does not have stop the derivation (KNOWN_ISSUES)
#[test]
fn unavailable_derivatives() {
  let ctx = Ctx {
    meshdim: 2,
    ref_elt_dim: 2,
    option: 1,
    ..Ctx::default()
  };
  let d = defs();
  let der = |e: &str| {
    let (t, _) = analyse_str(e, &d, &ctx).unwrap();
    let (t1, _) = getfem_gwfl::semantic::analyse(
      getfem_gwfl::derive::derive(&t, &d, "v", "", 1, None)
        .unwrap()
        .unwrap(),
      &d,
      &ctx,
    )
    .unwrap();
    getfem_gwfl::derive::derive(&t1, &d, "v", "", 2, None).map(|_| ())
  };
  let e = der("Norm_sqr(Expm(Grad_v))").unwrap_err().to_string();
  assert!(
    e.contains("second derivative of operator Expm w.r.t. its argument 1 not yet available"),
    "{e}"
  );
  // first derivatives are there
  assert!(der("Norm_sqr(Inv(Id(2)+Grad_v))").is_ok());
}
