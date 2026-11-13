//! Global functions (M12 step 1a): the B-spline polynomials ported from
//! GetFEM checked by their derivatives (central differences) and by the
//! partition of unity of a complete uniform basis; sums, products, boxes.
//! (Against GetFEM through the B-spline spaces, steps 1b and 1c.)

use getfem_fem::global_function::GlobalFunction;

/// points away from the knots (multiples of h/2 and h/3 avoided)
fn samples(lo: f64, hi: f64, n: usize) -> Vec<f64> {
  (0..n)
    .map(|i| lo + (hi - lo) * (i as f64 + 0.37) / n as f64)
    .collect()
}

fn check_derivatives(f: &GlobalFunction, x: &[f64]) {
  let n = x.len();
  let e = 1e-6;
  let (g, h) = (f.grad(x), f.hess(x));
  for k in 0..n {
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    xp[k] += e;
    xm[k] -= e;
    let dv = (f.val(&xp) - f.val(&xm)) / (2.0 * e);
    assert!(
      (dv - g[k]).abs() < 1e-6,
      "grad {k} at {x:?}: {} vs {dv}",
      g[k]
    );
    let (gp, gm) = (f.grad(&xp), f.grad(&xm));
    for i in 0..n {
      let dg = (gp[i] - gm[i]) / (2.0 * e);
      assert!(
        (dg - h[i + n * k]).abs() < 1e-5,
        "hess ({i},{k}) at {x:?}: {} vs {dg}",
        h[i + n * k]
      );
    }
  }
}

#[test]
fn bspline_derivatives() {
  for order in 3..=5 {
    for t in 1..=order {
      // a type-t function on [0.3, 0.3 + 0.7 t], and mirrored
      for (lo, hi) in [(0.3, 0.3 + 0.7 * t as f64), (0.3 + 0.7 * t as f64, 0.3)] {
        let f = GlobalFunction::bspline(&[lo], &[hi], order, &[t]).unwrap();
        for x in samples(lo.min(hi), lo.max(hi), 11) {
          check_derivatives(&f, &[x]);
        }
      }
      let f2 =
        GlobalFunction::bspline(&[0.0, 1.0], &[0.5 * t as f64, 0.0], order, &[t, 1]).unwrap();
      for x in samples(0.0, 0.5 * t as f64, 5) {
        for y in samples(0.0, 1.0, 5) {
          check_derivatives(&f2, &[x, y]);
        }
      }
      let f3 =
        GlobalFunction::bspline(&[0.0; 3], &[1.0, t as f64, 2.0], order, &[1, t, 2]).unwrap();
      for x in samples(0.0, 1.0, 3) {
        check_derivatives(&f3, &[x, 0.4 * t as f64, 1.3]);
      }
    }
  }
}

/// GetFEM's free-free uniform basis on [0, n h]: the boundary functions of
/// types 1 to order - 1 at both ends (mirrored at the right) and the
/// interior ones; their sum is 1 everywhere, its derivatives 0
#[test]
fn bspline_partition_of_unity() {
  let (n, h) = (8, 1.5);
  let l = n as f64 * h;
  for order in 3..=5 {
    let mut fs = vec![];
    for t in 1..order {
      fs.push(GlobalFunction::bspline(&[0.0], &[t as f64 * h], order, &[t]).unwrap());
      fs.push(GlobalFunction::bspline(&[l], &[l - t as f64 * h], order, &[t]).unwrap());
    }
    for i in 0..=n - order {
      let a = i as f64 * h;
      fs.push(GlobalFunction::bspline(&[a], &[a + order as f64 * h], order, &[order]).unwrap());
    }
    // GetFEM's count: n + order - 1
    assert_eq!(fs.len(), n + order - 1);
    let s = GlobalFunction::sum(&fs).unwrap();
    for x in samples(0.0, l, 97) {
      assert!(
        (s.val(&[x]) - 1.0).abs() < 1e-14,
        "order {order} at {x}: {}",
        s.val(&[x])
      );
      assert!(s.grad(&[x])[0].abs() < 1e-12);
      assert!(s.hess(&[x])[0].abs() < 1e-10);
    }
    // also at the knots (GetFEM's eps conventions), for the value
    for k in 0..n {
      let x = k as f64 * h;
      assert!(
        (s.val(&[x]) - 1.0).abs() < 1e-14,
        "order {order} at knot {x}"
      );
    }
  }
}

#[test]
fn supports_sums_products_boxes() {
  let f = GlobalFunction::bspline(&[0.0, 0.0], &[3.0, 2.0], 3, &[3, 2]).unwrap();
  assert_eq!(f.bounding_box(), (vec![0.0, 0.0], vec![3.0, 2.0]));
  assert!(f.is_in_support(&[1.0, 1.0]) && !f.is_in_support(&[3.5, 1.0]));
  let g = GlobalFunction::bspline(&[1.0, 1.0], &[4.0, 3.0], 3, &[3, 2]).unwrap();
  let (s, p) = (
    GlobalFunction::sum(&[f.clone(), g.clone()]).unwrap(),
    GlobalFunction::product(&f, &g).unwrap(),
  );
  assert_eq!(s.bounding_box(), (vec![0.0, 0.0], vec![4.0, 3.0]));
  assert_eq!(p.bounding_box(), (vec![1.0, 1.0], vec![3.0, 2.0]));
  assert!(s.is_in_support(&[3.5, 2.5]) && !p.is_in_support(&[3.5, 2.5]));
  for x in [[1.3, 1.4], [2.2, 1.7], [2.9, 1.1]] {
    assert!((p.val(&x) - f.val(&x) * g.val(&x)).abs() < 1e-15);
    check_derivatives(&p, &x);
    check_derivatives(&s, &x);
  }
  let b = GlobalFunction::bounded(&f, &[0.5, 0.5], &[1.0, 1.0]).unwrap();
  assert_eq!(b.bounding_box(), (vec![0.5, 0.5], vec![1.0, 1.0]));
  assert_eq!(b.val(&[2.0, 1.0]), f.val(&[2.0, 1.0]));
  // errors
  assert!(GlobalFunction::bspline(&[0.0], &[1.0], 6, &[1]).is_err());
  assert!(GlobalFunction::bspline(&[0.0], &[1.0], 3, &[4]).is_err());
  let f1 = GlobalFunction::bspline(&[0.0], &[1.0], 3, &[1]).unwrap();
  assert!(GlobalFunction::sum(&[f.clone(), f1.clone()]).is_err());
  assert!(GlobalFunction::product(&f, &f1).is_err());
}
