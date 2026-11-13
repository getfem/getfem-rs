//! M7 exit: first and second derivatives of every built-in nonlinear
//! operator against central finite differences, plus closed-form values;
//! the same for the hyperelastic laws (M8).

use getfem_gwfl::operators::{BUILTINS, NonlinearOperator, Operators};

fn pseudo(n: usize, seed: f64) -> Vec<f64> {
  (0..n)
    .map(|i| 0.3 * (seed * (i as f64 + 1.0)).sin())
    .collect()
}

/// arguments per operator: shapes and values (near the identity for matrices)
fn args_of(name: &str, n: usize) -> Vec<(Vec<usize>, Vec<f64>)> {
  let mut a = pseudo(n * n, 1.7);
  for i in 0..n {
    a[i * (n + 1)] += 1.0;
  }
  match name {
    "Norm" | "Norm_sqr" | "Normalized" => {
      vec![(vec![n], pseudo(n, 0.9).iter().map(|x| x + 0.5).collect())]
    }
    "Normalized_reg" => vec![(vec![n], pseudo(n, 0.9)), (vec![1], vec![0.2])],
    "Ball_projection" => vec![
      (
        vec![n],
        pseudo(n, 0.9).iter().map(|x| 2.0 * x + 0.7).collect(),
      ),
      (vec![1], vec![0.5]),
    ],
    // a displacement gradient and a unit normal
    "Transformed_unit_vector" => {
      let nv = pseudo(n, 0.9).iter().map(|x| x + 0.5).collect::<Vec<_>>();
      let no = nv.iter().map(|x| x * x).sum::<f64>().sqrt();
      vec![
        (vec![n, n], pseudo(n * n, 1.7)),
        (vec![n], nv.iter().map(|x| x / no).collect()),
      ]
    }
    "Right_Cauchy_Green" | "Left_Cauchy_Green" | "Green_Lagrangian" => {
      let mut r = pseudo(n * (n - 1), 1.3);
      r[0] += 1.0;
      vec![(vec![n, n - 1], r)]
    }
    _ => vec![(vec![n, n], a)],
  }
}

fn check(op: &dyn NonlinearOperator, name: &str, args: &[(Vec<usize>, Vec<f64>)]) {
  let shapes: Vec<&[usize]> = args.iter().map(|a| a.0.as_slice()).collect();
  let vals: Vec<&[f64]> = args.iter().map(|a| a.1.as_slice()).collect();
  let rs = op
    .result_shape(&shapes)
    .unwrap_or_else(|| panic!("{name}: shape"));
  let r: usize = rs.iter().product();
  // outputs prefilled with NaN: every entry must be written
  let written = |o: Vec<f64>, what: &str| {
    assert!(
      o.iter().all(|x| !x.is_nan()),
      "{name}: {what} not fully written"
    );
    o
  };
  let f = |v: &[Vec<f64>]| {
    let mut o = vec![f64::NAN; r];
    op.value(
      &v.iter().map(|x| x.as_slice()).collect::<Vec<_>>(),
      &shapes,
      &mut o,
    )
    .unwrap();
    written(o, "value")
  };
  let d = |v: &[Vec<f64>], i: usize| {
    let mut o = vec![f64::NAN; r * args[i].1.len()];
    op.derivative(
      &v.iter().map(|x| x.as_slice()).collect::<Vec<_>>(),
      &shapes,
      i,
      &mut o,
    )
    .unwrap();
    written(o, "derivative")
  };
  let dd = |i: usize, j: usize| -> Result<Vec<f64>, String> {
    let mut o = vec![f64::NAN; r * args[i].1.len() * args[j].1.len()];
    op.second_derivative(&vals, &shapes, i, j, &mut o)?;
    Ok(written(o, "second derivative"))
  };
  let base: Vec<Vec<f64>> = vals.iter().map(|x| x.to_vec()).collect();
  f(&base);
  let h = 1e-6;
  for i in 0..args.len() {
    if !op.has_derivative((i + 1, 0)) {
      let mut o = vec![0.0; r * args[i].1.len()];
      assert!(
        op.derivative(&vals, &shapes, i, &mut o).is_err(),
        "{name}: unavailable derivative"
      );
      continue;
    }
    let di = d(&base, i);
    let ni = args[i].1.len();
    for k in 0..ni {
      let (mut p, mut m) = (base.clone(), base.clone());
      p[i][k] += h;
      m[i][k] -= h;
      let (fp, fm) = (f(&p), f(&m));
      for q in 0..r {
        let fd = (fp[q] - fm[q]) / (2.0 * h);
        assert!(
          (fd - di[q + r * k]).abs() < 1e-6 * (1.0 + fd.abs()),
          "{name} d{i}[{q},{k}]: {} vs {fd}",
          di[q + r * k]
        );
      }
    }
    for j in 0..args.len() {
      if !op.has_derivative((i + 1, j + 1)) {
        assert!(dd(i, j).is_err(), "{name}: unavailable second derivative");
        continue;
      }
      let dd = dd(i, j).unwrap();
      let nj = args[j].1.len();
      for l in 0..nj {
        let (mut p, mut m) = (base.clone(), base.clone());
        p[j][l] += h;
        m[j][l] -= h;
        let (dp, dm) = (d(&p, i), d(&m, i));
        for x in 0..r * ni {
          let fd = (dp[x] - dm[x]) / (2.0 * h);
          let v = dd[x + r * ni * l];
          assert!(
            (fd - v).abs() < 1e-5 * (1.0 + fd.abs()),
            "{name} d{i}d{j}[{x},{l}]: {v} vs {fd}"
          );
        }
      }
      if i == j {
        for k in 0..ni {
          for l in 0..ni {
            for q in 0..r {
              let (a, b) = (dd[q + r * (k + ni * l)], dd[q + r * (l + ni * k)]);
              assert!(
                (a - b).abs() < 1e-10 * (1.0 + a.abs()),
                "{name}: second derivative not symmetric"
              );
            }
          }
        }
      }
    }
  }
}

#[test]
fn derivatives_match_finite_differences() {
  for op in BUILTINS {
    for n in [2, 3] {
      check(op, op.name(), &args_of(op.name(), n));
    }
  }
}

/// hyperelastic laws (M8), from the registry
#[test]
fn law_derivatives_match_finite_differences() {
  let ops = Operators::default();
  let params: &[(&str, &[f64])] = &[
    ("Saint_Venant_Kirchhoff", &[2.0, 3.0]),
    ("Generalized_Blatz_Ko", &[1.0, 1.0, 1.5, -0.5, 1.5]),
    ("Ciarlet_Geymonat", &[1.0, 1.0, 0.5]),
    ("Incompressible_Mooney_Rivlin", &[1.0, 1.0]),
    ("Compressible_Mooney_Rivlin", &[1.0, 1.0, 1.5]),
    ("Incompressible_Neo_Hookean", &[1.0]),
    ("Compressible_Neo_Hookean", &[1.0, 1.5]),
    ("Compressible_Neo_Hookean_Bonet", &[2.0, 3.0]),
    ("Compressible_Neo_Hookean_Ciarlet", &[2.0, 3.0]),
  ];
  let mut count = 0;
  for &(law, p) in params {
    for (pre, n) in [("", 3), ("Plane_Strain_", 2)] {
      for suffix in ["potential", "PK2"] {
        let name = format!("{pre}{law}_{suffix}");
        let grad_u = pseudo(n * n, 1.7);
        let args = vec![(vec![n, n], grad_u), (vec![p.len()], p.to_vec())];
        let shapes: Vec<&[usize]> = args.iter().map(|a| a.0.as_slice()).collect();
        let op = ops.get(&name).unwrap();
        assert!(op.result_shape(&shapes).is_some(), "{name}");
        check(op.as_ref(), &name, &args);
        count += 1;
      }
    }
  }
  assert_eq!(count, 36);
  assert!(ops.get("Cauchy_stress_from_PK2").is_none());
}

#[test]
fn values() {
  let v = |name: &str, s: &[usize], a: &[f64]| {
    let op = BUILTINS.iter().find(|b| b.name() == name).unwrap();
    let n: usize = op.result_shape(&[s]).unwrap().iter().product();
    let mut out = vec![f64::NAN; n];
    op.value(&[a], &[s], &mut out).unwrap();
    out
  };
  let close = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-13);
  assert!(close(&v("Det", &[2, 2], &[1.0, 3.0, 2.0, 4.0]), &[-2.0]));
  assert!(close(
    &v("Inv", &[2, 2], &[1.0, 3.0, 2.0, 4.0]),
    &[-2.0, 1.5, 1.0, -0.5]
  ));
  {
    // complex eigenvalues (a rotation and a stretch): scipy.linalg.logm's
    // values, and Expm(Logm(A)) = A; column-major
    // [c,-s,0.3; s,c,-0.2; 0.1,0.4,1.5]
    let (c, s) = (0.7f64.cos(), 0.7f64.sin());
    let m3 = [c, s, 0.1, -s, c, 0.4, 0.3, -0.2, 1.5];
    let l3 = v("Logm", &[3, 3], &m3);
    let scipy = [
      0.004432137171379603,
      0.6893056312057133,
      -0.02543923688189998,
      -0.7289212980721869,
      0.04800937072450086,
      0.33210197835307625,
      0.17889342222528593,
      -0.22985377239428462,
      0.435176440344934,
    ];
    assert!(close(&l3, &scipy), "{l3:?}");
    assert!(close(&v("Expm", &[3, 3], &l3), &m3));
  }
  // exp of a rotation generator, log back
  let t = 0.7f64;
  let r = v("Expm", &[2, 2], &[0.0, t, -t, 0.0]);
  assert!(close(&r, &[t.cos(), t.sin(), -t.sin(), t.cos()]));
  let l = v("Logm", &[2, 2], &r);
  assert!(close(&l, &[0.0, t, -t, 0.0]), "{l:?}");
  let d = v("Logm", &[2, 2], &[4.0, 0.0, 0.0, 0.5]);
  assert!(close(&d, &[4f64.ln(), 0.0, 0.0, 0.5f64.ln()]));
  assert!(close(
    &v("Matrix_i2", &[2, 2], &[1.0, 3.0, 2.0, 4.0]),
    &[-2.0]
  ));
  assert!(close(
    &v("Green_Lagrangian", &[2, 2], &[1.0, 0.0, 0.0, 1.0]),
    &[0.0; 4]
  ));
  // Transformed_unit_vector: a rotation R turns n into R n; the stretch
  // diag(2, 1) turns (1, 1)/sqrt 2 into (1, 2)/sqrt 5
  let tuv = |g: &[f64], nv: &[f64]| {
    let op = BUILTINS
      .iter()
      .find(|b| b.name() == "Transformed_unit_vector")
      .unwrap();
    let mut out = vec![f64::NAN; 2];
    op.value(&[g, nv], &[&[2, 2], &[2]], &mut out).unwrap();
    out
  };
  let (c, s) = (0.7f64.cos(), 0.7f64.sin());
  let (n0, n1) = (0.6, 0.8);
  assert!(close(
    &tuv(&[c - 1.0, s, -s, c - 1.0], &[n0, n1]),
    &[c * n0 - s * n1, s * n0 + c * n1]
  ));
  let r = 2f64.sqrt();
  let q = 5f64.sqrt();
  assert!(close(
    &tuv(&[1.0, 0.0, 0.0, 0.0], &[1.0 / r, 1.0 / r]),
    &[1.0 / q, 2.0 / q]
  ));
}

/// Expm on each Pade degree (infinity norms 0.01 .. 0.8, 1.5) and with
/// squarings (4, 40): values against mpmath (40 digits), derivatives
/// against the block identity exp([A, E; 0, A]) = [exp A, dexp(A)[E]; 0, exp A]
#[test]
fn expm_degrees() {
  let op = BUILTINS.iter().find(|b| b.name() == "Expm").unwrap();
  let expm = |a: &[f64], n: usize| {
    let mut o = vec![0.0; n * n];
    op.value(&[a], &[&[n, n]], &mut o).unwrap();
    o
  };
  // column-major, infinity norm 1
  let base = [0.3, -0.5, 0.2, 0.4, 0.1, -0.3, -0.2, 0.4, 0.5];
  let refs: [(f64, [f64; 9]); 6] = [
    (
      0.01,
      [
        1.0029924741650953,
        -0.00500596876640393,
        0.0020155268305397586,
        0.004010993624471599,
        1.0009844698482835,
        -0.003004985462404912,
        -0.001999980302168622,
        0.004017011635453974,
        1.0050044904892288,
      ],
    ),
    (
      0.2,
      [
        1.0567932482156102,
        -0.10213992367309613,
        0.04641408575462554,
        0.08434249779110285,
        1.0135612901052902,
        -0.06187795136412805,
        -0.03983768862306067,
        0.08688819990654732,
        1.1017223410695598,
      ],
    ),
    (
      0.8,
      [
        1.1788953451363688,
        -0.41967979299688535,
        0.27267745103316576,
        0.3853464534641764,
        0.9662210623290113,
        -0.2630084672002821,
        -0.14867090336649552,
        0.43334898804482297,
        1.4235713176641576,
      ],
    ),
    (
      1.5,
      [
        1.1978502898220778,
        -0.7458764109862551,
        0.7343866509185476,
        0.8030693529454226,
        0.7131026197379075,
        -0.4941251230141701,
        -0.21846609052750132,
        1.0027805376129244,
        1.8157387496845827,
      ],
    ),
    (
      4.0,
      [
        0.008644224992749007,
        0.508051691666995,
        4.4479262588936415,
        1.7948648299087495,
        -1.7764116509464425,
        -0.19223812315130026,
        1.0553391992122223,
        3.92516113627231,
        3.2138976385076474,
      ],
    ),
    (
      40.0,
      [
        2554915.4594686166,
        3661190.7396326656,
        5966672.902421112,
        588079.8532785935,
        842717.2888838659,
        1402545.8271701452,
        2825908.2100407025,
        3991659.6387476833,
        6536166.820366094,
      ],
    ),
  ];
  let maxdiff = |a: &[f64], b: &[f64]| {
    a.iter()
      .zip(b)
      .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
  };
  let maxabs = |a: &[f64]| a.iter().fold(0.0f64, |m, x| m.max(x.abs()));
  for (c, want) in refs {
    let a: Vec<f64> = base.iter().map(|x| c * x).collect();
    let e = expm(&a, 3);
    assert!(
      maxdiff(&e, &want) <= 1e-13 * maxabs(&want),
      "value, norm {c}: {e:?}"
    );
    let mut d = vec![0.0; 81];
    op.derivative(&[&a], &[&[3, 3]], 0, &mut d).unwrap();
    for kl in 0..9 {
      // [A, E_kl; 0, A] (6 x 6, column-major)
      let mut b = vec![0.0; 36];
      for j in 0..3 {
        for i in 0..3 {
          b[i + 6 * j] = a[i + 3 * j];
          b[3 + i + 6 * (3 + j)] = a[i + 3 * j];
        }
      }
      b[kl % 3 + 6 * (3 + kl / 3)] = 1.0;
      let eb = expm(&b, 6);
      let blk: Vec<f64> = (0..9).map(|ij| eb[ij % 3 + 6 * (3 + ij / 3)]).collect();
      let got = &d[9 * kl..9 * kl + 9];
      assert!(
        maxdiff(got, &blk) <= 1e-12 * maxabs(&blk),
        "derivative, norm {c}, kl {kl}: {got:?} {blk:?}"
      );
    }
  }
}
