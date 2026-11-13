use super::*;

type P = Polynomial<f64>;

fn x(n: usize, k: usize) -> P {
  P::var(n, k)
}

fn naive_eval(p: &P, v: &[f64]) -> f64 {
  let mut pi = PowerIndex::new(p.dim());
  let mut s = 0.0;
  for &c in p.coeffs() {
    s += c
      * pi
        .powers()
        .iter()
        .zip(v)
        .map(|(&e, &x)| x.powi(e as i32))
        .product::<f64>();
    pi.next();
  }
  s
}

#[test]
fn alpha_values() {
  assert_eq!(alpha(0, 5), 1);
  assert_eq!(alpha(2, 2), 6);
  assert_eq!(alpha(3, 3), 20);
  assert_eq!(alpha(3, 6), 84);
}

#[test]
fn power_index_walk() {
  let mut p = PowerIndex::new(3);
  let mut seen = vec![];
  for i in 0..35 {
    assert_eq!(p.global_index(), i);
    seen.push(p.clone());
    p.next();
  }
  assert_eq!(seen[1].powers(), &[1, 0, 0]);
  assert_eq!(seen[3].powers(), &[0, 0, 1]);
  assert_eq!(seen[4].powers(), &[2, 0, 0]);
  assert_eq!(seen[9].powers(), &[0, 0, 2]);
  for i in (0..34).rev() {
    p.prev();
    assert_eq!(p, seen[i + 1].clone());
    assert_eq!(p.global_index(), i + 1);
    p = seen[i + 1].clone();
  }
}

#[test]
fn arithmetic() {
  let (a, b) = (x(2, 0), x(2, 1));
  let p = &(&a + &b) * &(&a - &b);
  let q = &(&a * &a) - &(&b * &b);
  assert_eq!(p, q);
  assert_eq!(p.degree(), 2);
  assert_eq!(format!("{p}"), "x^2 - y^2");
  let r = &(&p * 3.0) / 2.0;
  assert_eq!(r.eval(&[2.0, 1.0]), 4.5);
}

#[test]
fn eval_matches_naive() {
  let mut seed = 12345u64;
  let mut rnd = || {
    seed = seed
      .wrapping_mul(6364136223846793005)
      .wrapping_add(1442695040888963407);
    (seed >> 11) as f64 / (1u64 << 53) as f64
  };
  for n in 0..=4 {
    for d in 0..=6 {
      let c = (0..alpha(n, d)).map(|_| rnd()).collect();
      let p = P::from_coeffs(n, d, c).unwrap();
      let v: Vec<f64> = (0..n).map(|_| rnd()).collect();
      assert!(
        (p.eval(&v) - naive_eval(&p, &v)).abs() < 1e-13,
        "n={n} d={d}"
      );
    }
  }
}

#[test]
fn direct_product_and_derivative() {
  // (x^2 + 1) (x) y -> x^2*y + y in 2 vars
  let p = &(&x(1, 0) * &x(1, 0)) + &P::one(1);
  let dp = p.direct_product(&x(1, 0));
  assert_eq!(format!("{dp}"), "y + x^2*y");
  let mut d0 = dp.clone();
  d0.derivative(0);
  assert_eq!(format!("{d0}"), "2*x*y");
  let mut d1 = dp.clone();
  d1.derivative(1);
  assert_eq!(d1, p.direct_product(&P::one(1)));
}

#[test]
fn real_degree() {
  let mut p = x(2, 1);
  p.change_degree(4);
  assert_eq!(p.degree(), 4);
  assert_eq!(p.real_degree(), 1);
  assert_eq!(P::constant(3, 2.0).real_degree(), 0);
}

#[test]
fn substitute() {
  // x + y*x^2 with x -> x+1
  let (a, b) = (x(2, 0), x(2, 1));
  let p = &a + &(&b * &(&a * &a));
  let s = &x(1, 0) + &P::one(1);
  let a1 = &a + &P::one(2);
  assert_eq!(p.substitute_var(&s, 0), &a1 + &(&b * &(&a1 * &a1)));
}

#[test]
fn parse_roundtrip() {
  let p: P = parse(3, "3*x*y*z - (x+y)^2/2 + 1.5e-1*z^3 - sqrt(4)").unwrap();
  let v: [f64; 3] = [0.3, -1.2, 2.0];
  let expect = 3.0 * v[0] * v[1] * v[2] - (v[0] + v[1]).powi(2) / 2.0 + 0.15 * v[2].powi(3) - 2.0;
  assert!((p.eval(&v) - expect).abs() < 1e-14);
  let q: P = parse(3, &format!("{p}")).unwrap();
  assert_eq!(p, q);
  assert_eq!(parse::<f64>(2, "-x^2").unwrap(), -&(&x(2, 0) * &x(2, 0)));
}

#[test]
fn parse_errors() {
  for s in ["z", "x/y", "x^y", "x^-1", "(x", "x +", "x y", "x^1.5"] {
    assert!(parse::<f64>(2, s).is_err(), "{s}");
  }
}

#[test]
fn rational_derivative() {
  // f = x/(1+y): df/dy = -x/(1+y)^2
  let (a, b) = (x(2, 0), x(2, 1));
  let f = RationalFraction::new(a.clone(), &P::one(2) + &b);
  let g = f.derived(1);
  let v = [0.7, 0.4];
  assert!((g.eval(&v) + v[0] / (1.0 + v[1]).powi(2)).abs() < 1e-15);
  // 0/0 at origin for x*y/(x+y): finite
  let h = RationalFraction::new(&a * &b, &a + &b);
  assert!(h.eval(&[0.0, 0.0]).is_finite());
}

/// Horner with the tabulated coefficient order: the same values, bit for
/// bit, as reading each coefficient by its global index
fn reference_horner(p: &P, mi: &mut PowerIndex, k: usize, de: usize, x: &[f64]) -> f64 {
  if k == 0 {
    return p.c[mi.global_index()];
  }
  let mut res = 0.0;
  for m in (0..=p.d - de).rev() {
    mi[k - 1] = m;
    res = reference_horner(p, mi, k - 1, de + m, x) + x[k - 1] * res;
  }
  mi[k - 1] = 0;
  res
}

fn sample(n: usize, d: usize) -> P {
  let c: Vec<f64> = (0..alpha(n, d))
    .map(|i| (1.0 + i as f64).sqrt().sin())
    .collect();
  P::from_coeffs(n, d, c).unwrap()
}

fn sample_x(n: usize, s: usize) -> Vec<f64> {
  (0..n).map(|i| (0.37 * (s * n + i) as f64).cos()).collect()
}

#[test]
fn horner_order_exact() {
  for (n, d) in [(1, 5), (2, 4), (3, 3), (3, 6), (3, 8), (4, 3)] {
    let p = sample(n, d);
    for s in 0..5 {
      let x = sample_x(n, s);
      let r = reference_horner(&p, &mut PowerIndex::new(n), n, 0, &x);
      let h = horner_order(n, d, |o| p.horner(&mut o.iter(), n, 0, &x));
      assert_eq!(h.to_bits(), r.to_bits(), "n {n} d {d}");
      if n <= 3 {
        let l = horner_order(n, d, |o| p.horner_loops(o, &x));
        assert_eq!(l.to_bits(), r.to_bits(), "loops: n {n} d {d}");
      }
    }
  }
}

/// GetFEM's direct formulas (dimension 1 to 3, degree 2 to 6): Horner's
/// values up to rounding (a transcription error would change them)
#[test]
fn direct_formulas() {
  for n in 1..=3 {
    for d in 2..=6 {
      let p = sample(n, d);
      for s in 0..5 {
        let x = sample_x(n, s);
        let h = reference_horner(&p, &mut PowerIndex::new(n), n, 0, &x);
        let v = p.eval_direct(&x).expect("direct formula");
        assert!(
          (v - h).abs() <= 1e-14 * alpha(n, d) as f64,
          "n {n} d {d}: {v} {h}"
        );
      }
    }
  }
  assert!(sample(3, 7).eval_direct(&[0.1; 3]).is_none());
  assert!(sample(4, 2).eval_direct(&[0.1; 4]).is_none());
}
