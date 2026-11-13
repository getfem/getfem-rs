//! Global functions of GWFL expressions (M12 step 1a): GetFEM's
//! global_function_parser (value, gradient, Hessian of X with macros x, y,
//! z) and global_function_bounded with an expression; derivatives given
//! or derived.

use getfem_fem::global_function::GlobalFunction;
use getfem_gwfl::global_function::GlobalFunctionExpr;

fn close(a: &[f64], b: &[f64]) -> bool {
  a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-13)
}

#[test]
fn parser_values_and_derivatives() {
  let p = [0.3, -1.2];
  let (x, y) = (p[0], p[1]);
  // given gradient and Hessian
  let f = GlobalFunction::parser(2, "x*x*y", Some("[2*x*y;x*x]"), Some("[2*y,2*x;2*x,0]")).unwrap();
  assert!((f.val(&p) - x * x * y).abs() < 1e-15);
  assert!(close(&f.grad(&p), &[2.0 * x * y, x * x]));
  assert!(close(&f.hess(&p), &[2.0 * y, 2.0 * x, 2.0 * x, 0.0]));
  // derived
  let g = GlobalFunction::parser(2, "x*x*y + sin(y)", None, None).unwrap();
  assert!(close(&g.grad(&p), &[2.0 * x * y, x * x + y.cos()]));
  assert!(close(&g.hess(&p), &[2.0 * y, 2.0 * x, 2.0 * x, -y.sin()]));
  // 3D, X itself
  let q = [0.5, 2.0, -0.7];
  let h = GlobalFunction::parser(3, "X(1)*y*z", None, None).unwrap();
  assert!((h.val(&q) - q[0] * q[1] * q[2]).abs() < 1e-15);
  assert!(close(&h.grad(&q), &[q[1] * q[2], q[0] * q[2], q[0] * q[1]]));
  assert!(close(
    &h.hess(&q),
    &[0.0, q[2], q[1], q[2], 0.0, q[0], q[1], q[0], 0.0]
  ));
  // a constant: zero derivatives
  let c = GlobalFunction::parser(2, "3", None, None).unwrap();
  assert_eq!(c.val(&p), 3.0);
  assert_eq!(c.grad(&p), vec![0.0; 2]);
  assert_eq!(c.hess(&p), vec![0.0; 4]);
  // sizes
  assert!(GlobalFunction::parser(2, "[x;y]", None, None).is_err());
  assert!(GlobalFunction::parser(2, "x", Some("x"), None).is_err());
  assert!(GlobalFunction::parser(2, "x+", None, None).is_err());
}

#[test]
fn bounded_by_an_expression() {
  let f = GlobalFunction::parser(2, "x+y", None, None).unwrap();
  let b = GlobalFunction::bounded_by(&f, &[-1.0, -1.0], &[1.0, 1.0], "1-x*x-y*y").unwrap();
  assert!(b.is_in_support(&[0.5, 0.5]) && !b.is_in_support(&[0.8, 0.8]));
  assert_eq!(b.bounding_box(), (vec![-1.0, -1.0], vec![1.0, 1.0]));
  assert_eq!(b.val(&[0.8, 0.8]), f.val(&[0.8, 0.8]));
  assert!(GlobalFunction::bounded_by(&f, &[0.0], &[1.0], "1").is_err());
}
