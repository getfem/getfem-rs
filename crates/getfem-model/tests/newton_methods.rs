//! The Newton driver's methods on small systems: pure Newton where it
//! works, its update halved to a finite residual, a saturating residual
//! where pure Newton diverges (each damped method, a chosen cascade), invariance to the scaling of a variable's
//! equations, a system without root, and the variable checks.

use std::ops::Range;

use getfem_linalg::ColMatrix;
use getfem_model::{
  Linearized, Method, Newton, NonlinearSystem, Result, Status, VariableTolerance,
};

/// R and dR/du (rows)
type Fn = fn(&[f64]) -> (Vec<f64>, Vec<Vec<f64>>);

/// a system of n unknowns, one variable per unknown
struct Toy {
  u: Vec<f64>,
  f: Fn,
  /// the tangent of the last residual_and_tangent
  k: ColMatrix<f64>,
}

impl NonlinearSystem for Toy {
  fn values(&self) -> Vec<f64> {
    self.u.clone()
  }

  fn set_values(&mut self, u: &[f64]) -> Result<()> {
    self.u = u.to_vec();
    Ok(())
  }

  fn residual_and_tangent(&mut self) -> Result<Linearized> {
    let (r, j) = (self.f)(&self.u);
    let n = r.len();
    let mut k = ColMatrix::new(n, n);
    for (a, row) in j.iter().enumerate() {
      for (b, v) in row.iter().enumerate() {
        k.add_entry(a, b, *v);
      }
    }
    self.k = k;
    Ok(Linearized {
      rhs: r.iter().map(|x| -x).collect(),
      condensed: None,
    })
  }

  fn tangent_matrix(&self) -> &ColMatrix<f64> {
    &self.k
  }

  fn variables(&self) -> Vec<(String, Range<usize>)> {
    (0..self.u.len())
      .map(|k| (format!("x{k}"), k..k + 1))
      .collect()
  }
}

fn saturating(u: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>) {
  let s = 1.0 + u[0] * u[0];
  (vec![u[0] / s.sqrt()], vec![vec![s.powf(-1.5)]])
}

fn square(u: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>) {
  (vec![u[0] * u[0] - 4.0], vec![vec![2.0 * u[0]]])
}

fn log(u: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>) {
  (vec![u[0].ln()], vec![vec![1.0 / u[0]]])
}

/// x0 saturating, x1 + x1^3 - x0/10 = 0, with x1 in units 10^-C: the
/// unknown y1 = 10^C x1 and its equation (test function) divided by 10^C
fn coupled<const C: u32>(u: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>) {
  let c = 10f64.powi(C as i32);
  let (r0, j0) = saturating(&u[..1]);
  let x1 = u[1] / c;
  let r1 = (x1 + x1.powi(3) - 0.1 * u[0]) / c;
  (
    vec![r0[0], r1],
    vec![
      vec![j0[0][0], 0.0],
      vec![-0.1 / c, (1.0 + 3.0 * x1 * x1) / (c * c)],
    ],
  )
}

/// x^2 + 1 = 0: no root
fn no_root(u: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>) {
  (vec![u[0] * u[0] + 1.0], vec![vec![2.0 * u[0]]])
}

fn newton() -> Newton {
  Newton {
    atol: 1e-12,
    rtol: 0.0,
    ..Default::default()
  }
}

#[test]
fn pure_newton_where_it_works() {
  let mut sys = Toy {
    u: vec![3.0],
    f: square,
    k: ColMatrix::default(),
  };
  let rep = newton().solve(&mut sys, None).unwrap();
  assert!(
    rep.converged() && rep.method == Some(Method::Newton),
    "{rep:?}"
  );
  assert!(rep.iterations <= 5, "{}", rep.iterations);
  assert!((sys.u[0] - 2.0).abs() < 1e-12);
}

#[test]
fn pure_newton_update_halved_to_a_finite_residual() {
  // from 3 the Newton step reaches x < 0 (ln not finite): halved, pure
  let mut sys = Toy {
    u: vec![3.0],
    f: log,
    k: ColMatrix::default(),
  };
  let rep = newton().solve(&mut sys, None).unwrap();
  assert!(
    rep.converged() && rep.method == Some(Method::Newton),
    "{rep:?}"
  );
  assert!((sys.u[0] - 1.0).abs() < 1e-12);
}

#[test]
fn saturating_residual() {
  // pure Newton diverges from 2 (|x| grows); each damped method converges
  for m in [Method::NleqErr, Method::NleqRes] {
    let mut sys = Toy {
      u: vec![2.0],
      f: saturating,
      k: ColMatrix::default(),
    };
    let newton = Newton {
      methods: vec![Method::Newton, m],
      ..newton()
    };
    let rep = newton.solve(&mut sys, None).unwrap();
    assert!(rep.converged() && rep.method == Some(m), "{rep:?}");
    assert!(sys.u[0].abs() < 1e-8);
  }
  // pure Newton alone fails
  let mut sys = Toy {
    u: vec![2.0],
    f: saturating,
    k: ColMatrix::default(),
  };
  let newton = Newton {
    methods: vec![Method::Newton],
    ..newton()
  };
  let rep = newton.solve(&mut sys, None).unwrap();
  assert!(
    !rep.converged() && rep.message.contains("full Newton"),
    "{rep:?}"
  );
}

#[test]
fn units_of_a_variable_are_invisible() {
  // x1 in other units (its equation with them, and its tolerance): the
  // same path, |R|_D in energy units
  let run = |f: Fn, c: f64| {
    let newton = Newton {
      atol: f64::INFINITY,
      variable_tolerances: vec![
        VariableTolerance::new(&["x0"], 1e-11),
        VariableTolerance::new(&["x1"], 1e-11 / c),
      ],
      ..Default::default()
    };
    let mut sys = Toy {
      u: vec![2.0, c],
      f,
      k: ColMatrix::default(),
    };
    let rep = newton.solve(&mut sys, None).unwrap();
    assert!(rep.converged(), "{}", rep.message);
    (rep.method, rep.iterations, vec![sys.u[0], sys.u[1] / c])
  };
  let (p0, n0, u0) = run(coupled::<0>, 1.0);
  let (p6, n6, u6) = run(coupled::<6>, 1e6);
  assert!(
    p0 != Some(Method::Newton) && (p0, n0) == (p6, n6),
    "{p0:?} {n0} / {p6:?} {n6}"
  );
  for (x, y) in u0.iter().zip(&u6) {
    assert!((x - y).abs() < 1e-10, "{u0:?} {u6:?}");
  }
}

#[test]
fn no_root_fails_within_max_iter() {
  let newton = Newton {
    max_iter: 300,
    ..Default::default()
  };
  let mut sys = Toy {
    u: vec![2.0],
    f: no_root,
    k: ColMatrix::default(),
  };
  let rep = newton.solve(&mut sys, None).unwrap();
  assert_eq!(rep.status, Status::MaxIterations, "{}", rep.message);
  assert!(
    rep.iterations <= 300 && rep.message.contains("NLEQ-RES"),
    "{}",
    rep.message
  );
}

#[test]
fn unknown_variables_are_errors() {
  let mut sys = Toy {
    u: vec![3.0],
    f: square,
    k: ColMatrix::default(),
  };
  let newton = Newton {
    variable_tolerances: vec![VariableTolerance::new(&["phi"], 1.0)],
    ..Default::default()
  };
  let e = newton.solve(&mut sys, None).unwrap_err().to_string();
  assert!(e.contains("variable tolerance [phi]: phi"), "{e}");
}
