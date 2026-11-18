//! Compiled instructions (GetFEM ga_compile): the expressions compile and
//! assemble; values are checked against GetFEM in the integration tests.

use getfem_fem::mesh_fem::MeshFem;
use getfem_fem::mesh_im::MeshIm;
use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::regular_mesh::regular_unit_mesh;

use super::Workspace;

fn setup(n: usize, nx: usize, gt: &str) -> (Workspace, MeshIm) {
  let m = regular_unit_mesh(&vec![nx; n], &geotrans_descriptor(gt).unwrap()).unwrap();
  let mf_u = MeshFem::new(&m, n)
    .with_classical_finite_element(2)
    .unwrap();
  let mf_p = MeshFem::new(&m, 1)
    .with_classical_finite_element(2)
    .unwrap();
  let seq =
    |len: usize, s: f64| -> Vec<f64> { (0..len).map(|i| (s * (i as f64 + 1.0)).sin()).collect() };
  let mut ws = Workspace::new();
  ws.add_scalar_data("lambda", 3.0).unwrap();
  ws.add_scalar_data("mu", 2.0).unwrap();
  ws.add_fem_variable("u", &mf_u).unwrap();
  ws.set_variable("u", &seq(mf_u.nb_dof().unwrap(), 0.7))
    .unwrap();
  ws.add_fem_variable("p", &mf_p).unwrap();
  ws.set_variable("p", &seq(mf_p.nb_dof().unwrap(), 1.1))
    .unwrap();
  let np = mf_p.nb_dof().unwrap();
  ws.add_fem_data("mu2", &mf_p, &seq(np, 0.3)).unwrap();
  let mim = MeshIm::new(&m).with_integration_method(4).unwrap();
  ws.add_fixed_size_variable("w", &[n]).unwrap();
  ws.set_variable("w", &seq(n, 0.9)).unwrap();
  let imd = getfem_fem::im_data::ImData::new(&mim, -1, &[]).unwrap();
  let nq = imd.nb_values().unwrap();
  ws.add_im_data("q", &imd, &seq(nq, 0.5)).unwrap();
  (ws, mim)
}

#[test]
fn scalars_compile() {
  const SCALAR: &[&str] = &[
    "1",
    "2.5",
    "cos(pi*X(1))",
    "cos(pi*X).exp(X*0)",
    "sin(pi*X).exp(X*0)",
    "-X.X/3 + X(1)*X(meshdim) - 2",
    "(Id(meshdim)*X).(X*2):Id(1)",
    "(X@X):(Id(meshdim)+X@X)/(1+X.X)",
    "u.u",
    "Norm_sqr(u)",
    "Grad_u:Grad_u",
    "Id(meshdim)*Grad_u:Grad_u",
    "p*sin(p) + Grad_p.Grad_p*X(1) + Norm(Grad_u)",
    "(Grad_u*Grad_u):Id(meshdim)",
    "Grad_u(1,:).Grad_u(1,:) + Grad_u(2,:).Grad_u(2,:) + Trace(Grad_u) + Div_u",
    "sqr(Norm(Grad_p)) + pow(1+p*p, 1.5) + Trace(Grad_u*Grad_u')",
    "Reshape(Grad_u, meshdim*meshdim)(1)",
    "Sym(Grad_u):Skew(Grad_u) + Deviator(Grad_u):Grad_u + element_size + Trace(element_K) + Norm_sqr(element_B)",
    "[u(1), p; X(1), 2]:[1,2;3,4] + (u.*u).X + (Grad_u./exp(Grad_u.*Grad_u)):Id(meshdim)",
    "Swap_indices(Grad_u,1,2):Grad_u + Index_move_last(Grad_u,1):Grad_u + Contract(Grad_u,1,2) + Contract(Grad_u,2,u,1).u",
    "Norm_sqr(Hess_p) + Hess_u(1,1,1) + (Hess_u:Id(meshdim)).u + q*p + q*q + w.u",
  ];
  for (n, gt) in [(2, "GT_PK(2,1)"), (2, "GT_QK(2,2)"), (3, "GT_PK(3,1)")] {
    let (mut ws, mim) = setup(n, 2, gt);
    let r = mim.linked_mesh().snapshot().all_convexes();
    for e in SCALAR {
      let trees = ws.trees(e, &mim, &r, 0).unwrap();
      let fixed = |v: &str| (v == "w").then(|| ws.variables["w"].value.clone());
      let cx = crate::ga::CompileCtx {
        defs: &ws.defs,
        fixed: &fixed,
        inputs: &[],
        vectorized: &|v: &str| v == "u",
        trans: &|_| None,
        trans_der: &|_, _, _| None,
        internal: &|_| false,
      };
      assert!(
        crate::ga::compile_scalar(&trees, &cx).is_some(),
        "{e} not compiled"
      );
      let a = {
        ws.clear_expressions();
        ws.add_expression(e, &mim, -1).unwrap();
        ws.assembly(0, false).unwrap();
        ws.assembled_potential()
      };
      assert!(a.is_finite(), "{gt}: {e}: {a}");
    }
  }
}

/// the same for vectors, on the domain and on boundary faces
#[test]
fn vectors_compile() {
  use getfem_geom::region::outer_faces_of_mesh;
  const VECTOR: &[&str] = &[
    "u.Test_u",
    "Grad_u:Grad_Test_u",
    "p*Test_p + Grad_p.Grad_Test_p*X(1)",
    "sin(X(1))*Test_p - Test_p/2",
    "(Grad_u*Test_u).X",
    "-(Id(meshdim)*Test_u).u",
    "Div_Test_u*p + Trace(Grad_Test_u)*Div_u + Grad_Test_p(1)",
    "(Sym(Grad_Test_u)+Skew(Grad_Test_u)):Grad_u + (Test_u.*u).X + Deviator(Grad_Test_u):Grad_u",
    "[Test_p, Test_p*2; 0, Test_p]:[1,2;3,4] + Contract(Grad_Test_u,1,2)*p + Swap_indices(Grad_Test_u,1,2):Grad_u",
    "Index_move_last(Grad_Test_u,1):Grad_u + Contract(Grad_Test_u,2,u,1).u + (Test_u./exp(u.*u)).X",
    "Hess_Test_p:Hess_p + (Hess_Test_u:Id(meshdim)).u + q*Test_p + Test_w.u + (w.u)*Test_w(1)",
  ];
  const FACES: &[&str] = &[
    "(((Reshape(Grad_u,meshdim,meshdim))')*Normal).Test_u",
    "(Grad_u*Normal).Test_u",
    "(Grad_u'*Normal).Test_u",
    "p*Test_p*(Normal.X)",
  ];
  for (n, gt) in [(2, "GT_PK(2,1)"), (2, "GT_QK(2,2)"), (3, "GT_PK(3,1)")] {
    let (mut ws, mim) = setup(n, 2, gt);
    let m = mim.linked_mesh();
    let border = outer_faces_of_mesh(&m);
    let snap = m.snapshot();
    let all = snap.all_convexes();
    for (list, region) in [(VECTOR, None), (FACES, Some(&border))] {
      for e in list {
        let r = region.cloned().unwrap_or(all.clone());
        let trees = ws.trees(e, &mim, &r, 1).unwrap();
        let fixed = |v: &str| (v == "w").then(|| ws.variables["w"].value.clone());
        let cx = crate::ga::CompileCtx {
          defs: &ws.defs,
          fixed: &fixed,
          inputs: &[],
          vectorized: &|v: &str| v == "u",
          trans: &|_| None,
          trans_der: &|_, _, _| None,
          internal: &|_| false,
        };
        assert!(
          crate::ga::compile_vector(&trees, &cx).is_some(),
          "{e} not compiled"
        );
        let a = {
          ws.clear_expressions();
          ws.add_expression(e, &mim, region).unwrap();
          ws.assembly(1, false).unwrap();
          ws.assembled_vector()
        };
        assert!(a.iter().all(|x| x.is_finite()), "{gt}: {e}");
        // fused over the points (ga/fuse.rs), GEMM threshold as set, 0 and
        // never, as point by point
        crate::ga::NO_FUSE.set(true);
        let a0 = {
          ws.clear_expressions();
          ws.add_expression(e, &mim, region).unwrap();
          ws.assembly(1, false).unwrap();
          ws.assembled_vector()
        }
        .to_vec();
        crate::ga::NO_FUSE.set(false);
        let scale = a0.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        for g in [None, Some(0), Some(usize::MAX)] {
          crate::ga::TEST_GEMM_MIN.set(g);
          let a = {
            ws.clear_expressions();
            ws.add_expression(e, &mim, region).unwrap();
            ws.assembly(1, false).unwrap();
            ws.assembled_vector()
          };
          crate::ga::TEST_GEMM_MIN.set(None);
          let err = a
            .iter()
            .zip(&a0)
            .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
          assert!(
            err <= 1e-12 * scale,
            "{gt}: {e}: fused ({g:?}) differs by {err:e} (scale {scale:e})"
          );
        }
      }
    }
  }
}

/// the same for matrices
#[test]
fn matrices_compile() {
  const MATRIX: &[&str] = &[
    "Test_u.Test2_u",
    "Grad_Test_p:Grad_Test2_p",
    "Grad_Test_u:Grad_Test2_u",
    "p*Test_p*Test2_p - Test2_p*(Grad_Test_p.X)",
    "(Grad_Test_u*Grad_u):Grad_Test2_u",
    "((Grad_Test2_u@Grad_Test_u):Grad_u):Grad_u",
    "(Test_u@Test2_u):Grad_u",
    "Test2_u.Test_u",
    "Grad_Test2_p.Grad_Test_p",
    "(Test2_u@Test_p).X",
    "(Grad_Test2_u*Test_u).X + (Grad_Test_u*Test2_u).X",
    "(Test_u*Test2_p).X + Test2_p*(Test_u.X)",
    "(Test2_p*Test_u).X + Test_p*Test2_p*3",
    "(Test2_p@Test_u).X",
    "(Test_u@Test2_p).X",
    "((Grad_Test2_u@Grad_Test_u):Grad_u):(Grad_u+Id(meshdim))",
    "(Grad_p(1)*Grad_p(1) + Grad_p(2)*Grad_p(2))/2",
    "Sym(Grad_Test_u):Sym(Grad_Test2_u) + (Test_u.*Test2_u).X + Deviator(Grad_Test_u):Grad_Test2_u",
    "Contract(Grad_Test_u,1,2)*Contract(Grad_Test2_u,1,2) + (Test2_u.*Test_u).X + Skew(Grad_Test2_u):Grad_Test_u",
    "Norm_sqr(Grad_p)/2",
    "(Grad_p:Grad_p)/2",
    "sqr(Norm(Grad_p))/2",
    "(lambda*Trace(Grad_Test_u)*Id(qdim(u)) + mu*(Grad_Test_u'+Grad_Test_u)):Grad_Test2_u",
    "lambda*Div_Test_u*Div_Test2_u + mu*(Grad_Test_u'+Grad_Test_u):Grad_Test2_u",
    "lambda*((Grad_Test2_u@Grad_Test_u):Id(meshdim)):Id(meshdim) + mu*(Grad_Test_u'+Grad_Test_u):Grad_Test2_u",
    "lambda*Id(meshdim)@Id(meshdim)*Grad_Test_u:Grad_Test2_u + mu*(Grad_Test_u'+Grad_Test_u):Grad_Test2_u",
    "(mu2*Trace(Grad_Test_u)*Id(meshdim) + mu2*(Grad_Test_u'+Grad_Test_u)):Grad_Test2_u",
    "Hess_Test_p:Hess_Test2_p + (Hess_Test_u:Id(meshdim)).Test2_u + q*Test_p*Test2_p",
    "Test_w.Test2_u + Test_w.Test2_w + (Test2_w.u)*Test_p",
    // a full tensor of the column tests contracted with the row gradient
    // (ContractionOpt02, the tangents of hyperelastic models)
    "(Grad_Test2_u*Grad_u' + Grad_u*Grad_Test2_u'):Grad_Test_u",
    "(mu*(Grad_Test2_u+Grad_Test2_u') + lambda*Trace(Grad_Test2_u)*Id(meshdim)):Grad_Test_u",
  ];
  for (n, gt) in [(2, "GT_PK(2,1)"), (2, "GT_QK(2,2)"), (3, "GT_PK(3,1)")] {
    let (mut ws, mim) = setup(n, 2, gt);
    let r = mim.linked_mesh().snapshot().all_convexes();
    let fixed = |v: &str| match v {
      "lambda" => Some(vec![3.0]),
      "mu" => Some(vec![2.0]),
      _ => None,
    };
    for e in MATRIX {
      let cx = crate::ga::CompileCtx {
        defs: &ws.defs,
        fixed: &fixed,
        inputs: &[],
        vectorized: &|v: &str| v == "u",
        trans: &|_| None,
        trans_der: &|_, _, _| None,
        internal: &|_| false,
      };
      let trees = ws.trees(e, &mim, &r, 2).unwrap();
      let ok = crate::ga::compile_matrix(&trees, &cx).is_some();
      assert!(ok, "{e} not compiled\n{}", trees[0].debug_string());
      let a = {
        ws.clear_expressions();
        ws.add_expression(e, &mim, -1).unwrap();
        ws.assembly(2, false).unwrap();
        ws.assembled_matrix()
      }
      .to_csr();
      assert!(a.val.iter().all(|x| x.is_finite()), "{gt}: {e}");
      // fused over the points (ga/fuse.rs), GEMM threshold as set, 0 and
      // never, as point by point
      crate::ga::NO_FUSE.set(true);
      let a0 = {
        ws.clear_expressions();
        ws.add_expression(e, &mim, -1).unwrap();
        ws.assembly(2, false).unwrap();
        ws.assembled_matrix()
      }
      .to_csr();
      crate::ga::NO_FUSE.set(false);
      let scale = a0.val.iter().fold(0.0f64, |m, x| m.max(x.abs()));
      let dense = |k: &getfem_linalg::Csr<f64>| -> Vec<f64> {
        let mut d = vec![0.0; k.nrows * k.ncols];
        for r in 0..k.nrows {
          for p in k.ptr[r]..k.ptr[r + 1] {
            d[r * k.ncols + k.idx[p]] = k.val[p];
          }
        }
        d
      };
      let d0 = dense(&a0);
      for g in [None, Some(0), Some(usize::MAX)] {
        crate::ga::TEST_GEMM_MIN.set(g);
        let a = {
          ws.clear_expressions();
          ws.add_expression(e, &mim, -1).unwrap();
          ws.assembly(2, false).unwrap();
          ws.assembled_matrix()
        }
        .to_csr();
        crate::ga::TEST_GEMM_MIN.set(None);
        let err = dense(&a)
          .iter()
          .zip(&d0)
          .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
        assert!(
          err <= 1e-12 * scale,
          "{gt}: {e}: fused ({g:?}) differs by {err:e} (scale {scale:e})"
        );
      }
    }
  }
}

/// interpolate transformations (expression, neighbor element), elementary
/// transformations
#[test]
fn transformations_compile() {
  use getfem_geom::region::inner_faces;
  const SCALAR: &[&str] = &[
    "Interpolate(p,T)*p + Interpolate(u,T).Interpolate(X,T) + Norm_sqr(Interpolate(Grad_u,T))",
    "Interpolate_filter(S, Interpolate(p,S), 1) + Interpolate_filter(S, 1, 0) + Norm_sqr(Interpolate(Hess_p,T))",
    "Norm_sqr(Elementary_transformation(u, P0)) + Elementary_transformation(Grad_p, P0).Grad_p",
    "Trace(Interpolate(element_K,T)) + Norm_sqr(Interpolate(element_B,T)) + Interpolate(X,T)(1) + Interpolate(element_K,T)(1,1)",
  ];
  const VECTOR: &[&str] = &[
    "Interpolate(p,T)*Test_p + Interpolate(Test_u,T).u + Interpolate(Grad_Test_p,T).Grad_p",
    "Interpolate_filter(S, Interpolate(Test_p,S)*p, 1) + Interpolate_filter(S, Test_p, 0)",
    "Interpolate(Div_Test_u,T)*Interpolate(Div_u,T) + Interpolate(Div_u,T)*Test_p",
    "Elementary_transformation(Test_u, P0).u + Elementary_transformation(Test_p, P0)*p",
  ];
  const MATRIX: &[&str] = &[
    "Interpolate(Test_p,T)*Test2_p + Interpolate(Test_u,T).Interpolate(Test2_u,T)",
    "Interpolate_filter(S, Interpolate(Grad_Test_p,S).Grad_Test2_p, 1)",
    "Elementary_transformation(Test_u, P0).Elementary_transformation(Test2_u, P0) + Elementary_transformation(Grad_Test_p, P0).Grad_Test2_p",
  ];
  const INNER_SCALAR: &[&str] =
    &["sqr(Interpolate(p,neighbor_element)-p) + Interpolate(Normal,neighbor_element).Normal"];
  const INNER_MATRIX: &[&str] = &[
    "(Interpolate(Test_p,neighbor_element)-Test_p)*(Interpolate(Test2_p,neighbor_element)-Test2_p)",
  ];

  let dense = |k: getfem_linalg::Csr<f64>| -> Vec<f64> {
    let mut d = vec![0.0; k.nrows * k.ncols];
    for i in 0..k.nrows {
      for (j, v) in k.row(i) {
        d[i * k.ncols + j] = v;
      }
    }
    d
  };
  for (n, gt) in [(2, "GT_PK(2,1)"), (2, "GT_QK(2,2)"), (3, "GT_PK(3,1)")] {
    let (mut ws, mim) = setup(n, 2, gt);
    let m = mim.linked_mesh();
    ws.add_interpolate_transformation_from_expression("T", &m, &m, "0.9*X+0.05*sin(3*X)")
      .unwrap();
    ws.add_interpolate_transformation_from_expression("S", &m, &m, "1.5*X")
      .unwrap();
    ws.add_P0_projection("P0").unwrap();
    let inner = inner_faces(&m, -1).unwrap();
    let run =
      |ws: &mut Workspace, e: &str, order: usize, region: Option<&getfem_geom::region::Region>| {
        let r: getfem_geom::region::RegionArg = match region {
          Some(r) => r.clone().into(),
          None => (-1).into(),
        };
        match order {
          0 => vec![{
            ws.clear_expressions();
            ws.add_expression(e, &mim, r).unwrap();
            ws.assembly(0, false).unwrap();
            ws.assembled_potential()
          }],
          1 => ({
            ws.clear_expressions();
            ws.add_expression(e, &mim, r).unwrap();
            ws.assembly(1, false).unwrap();
            ws.assembled_vector()
          })
          .to_vec(),
          _ => dense(
            {
              ws.clear_expressions();
              ws.add_expression(e, &mim, r).unwrap();
              ws.assembly(2, false).unwrap();
              ws.assembled_matrix()
            }
            .to_csr(),
          ),
        }
      };
    for (list, order, region) in [
      (SCALAR, 0, None),
      (VECTOR, 1, None),
      (MATRIX, 2, None),
      (INNER_SCALAR, 0, Some(&inner)),
      (INNER_MATRIX, 2, Some(&inner)),
    ] {
      for e in list {
        let a = run(&mut ws, e, order, region);
        assert!(a.iter().all(|x| x.is_finite()), "{gt}: {e}");
      }
    }
  }
}

/// vector elements on spaces of qdim Q x target_dim (GetFEM Qmult): w of
/// qdim 4 on FEM_RT0(2) is (a, b), a and b of qdim 2, DOF 2 d + q of w the
/// DOF d of copy q; components of w: r + 2 q (values and tests alike)
#[test]
fn vector_elements_with_qmult() {
  use getfem_fem::fem::fem_descriptor;
  let m = regular_unit_mesh(&[3, 2], &geotrans_descriptor("GT_PK(2,1)").unwrap()).unwrap();
  let rt0 = fem_descriptor("FEM_RT0(2)").unwrap();
  let mf = |q| MeshFem::new(&m, q).with_finite_element(&rt0).unwrap();
  let (mfw, mfa) = (mf(4), mf(2));
  let mim = MeshIm::new(&m).with_integration_method(4).unwrap();
  let nd = mfa.nb_dof().unwrap();
  assert_eq!(mfw.nb_dof().unwrap(), 2 * nd);
  let (num_w, num_a) = (mfw.numbering().unwrap(), mfa.numbering().unwrap());
  for cv in num_a.convex_index() {
    let (dw, da) = (
      num_w.ind_basic_dof_of_element(cv),
      num_a.ind_basic_dof_of_element(cv),
    );
    for (i, &d) in da.iter().enumerate() {
      assert_eq!((dw[2 * i], dw[2 * i + 1]), (2 * d, 2 * d + 1));
    }
  }
  let seq = |s: f64| -> Vec<f64> { (0..nd).map(|i| (s * (i as f64 + 1.0)).sin()).collect() };
  let (a, b) = (seq(0.7), seq(1.3));
  let w: Vec<f64> = (0..2 * nd)
    .map(|k| if k % 2 == 0 { a[k / 2] } else { b[k / 2] })
    .collect();
  let mut ws = Workspace::new();
  ws.add_fem_variable("w", &mfw).unwrap();
  ws.set_variable("w", &w).unwrap();
  let mut wa = Workspace::new();
  wa.add_fem_variable("a", &mfa).unwrap();
  wa.set_variable("a", &a).unwrap();
  let mut wb = Workspace::new();
  wb.add_fem_variable("b", &mfa).unwrap();
  wb.set_variable("b", &b).unwrap();
  let close =
    |x: f64, y: f64, e: &str| assert!((x - y).abs() <= 1e-12 * (1.0 + y.abs()), "{e}: {x} vs {y}");
  // assembled before the scalar closures borrow the workspaces
  let vw = {
    ws.clear_expressions();
    ws.add_expression("w.Test_w", &mim, -1).unwrap();
    ws.assembly(1, false).unwrap();
    ws.assembled_vector()
  }
  .to_vec();
  let vb = {
    wb.clear_expressions();
    wb.add_expression("b.Test_b", &mim, -1).unwrap();
    wb.assembly(1, false).unwrap();
    wb.assembled_vector()
  }
  .to_vec();
  let gw = {
    ws.clear_expressions();
    ws.add_expression("Grad_w:Grad_Test_w", &mim, -1).unwrap();
    ws.assembly(1, false).unwrap();
    ws.assembled_vector()
  }
  .to_vec();
  let kw = {
    ws.clear_expressions();
    ws.add_expression("Test_w.Test2_w", &mim, -1).unwrap();
    ws.assembly(2, false).unwrap();
    ws.assembled_matrix()
  }
  .to_csr();
  let ka = {
    wa.clear_expressions();
    wa.add_expression("Test_a.Test2_a", &mim, -1).unwrap();
    wa.assembly(2, false).unwrap();
    wa.assembled_matrix()
  }
  .to_csr();
  let mut sw = |e: &str| {
    ws.clear_expressions();
    ws.add_expression(e, &mim, -1).unwrap();
    ws.assembly(0, false).unwrap();
    ws.assembled_potential()
  };
  let mut sa = |e: &str| {
    wa.clear_expressions();
    wa.add_expression(e, &mim, -1).unwrap();
    wa.assembly(0, false).unwrap();
    wa.assembled_potential()
  };
  let mut sb = |e: &str| {
    wb.clear_expressions();
    wb.add_expression(e, &mim, -1).unwrap();
    wb.assembly(0, false).unwrap();
    wb.assembled_potential()
  };
  // values: components r + 2 q
  close(sw("w.w"), sa("a.a") + sb("b.b"), "w.w");
  close(sw("w(2)*X(1)"), sa("a(2)*X(1)"), "w(2)");
  close(sw("w(3)*X(2)"), sb("b(1)*X(2)"), "w(3)");
  close(sw("Grad_w(4,1)"), sb("Grad_b(2,1)"), "Grad_w(4,1)");
  close(
    sw("Norm_sqr(Grad_w)"),
    sa("Norm_sqr(Grad_a)") + sb("Norm_sqr(Grad_b)"),
    "Grad_w",
  );
  // tests: the derivatives of the values
  let dot = |x: &[f64], y: &[f64]| x.iter().zip(y).map(|(p, q)| p * q).sum::<f64>();
  close(dot(&vw, &w), sw("w.w"), "w.Test_w");
  for d in 0..nd {
    close(vw[2 * d + 1], vb[d], "w.Test_w vs b.Test_b");
  }
  close(dot(&gw, &w), sw("Norm_sqr(Grad_w)"), "Grad_w:Grad_Test_w");
  close(
    dot(&kw.matvec(&w).unwrap(), &w),
    sw("w.w"),
    "Test_w.Test2_w",
  );
  for d in 0..nd {
    for e in 0..nd {
      for (q, p) in [(0, 0), (1, 1), (0, 1), (1, 0)] {
        let want = if q == p { ka.get(d, e) } else { 0.0 };
        close(kw.get(2 * d + q, 2 * e + p), want, "Test_w.Test2_w block");
      }
    }
  }
}

/// the isotropic elasticity form written four ways (test_assembly.cc cases;
/// one compiled with contractions pushed into the tensor product,
/// ga/rewrite.rs): the same matrix
#[test]
fn elasticity_forms_agree() {
  const FORMS: &[&str] = &[
    "(lambda*Trace(Grad_Test_u)*Id(qdim(u)) + mu*(Grad_Test_u'+Grad_Test_u)):Grad_Test2_u",
    "lambda*Div_Test_u*Div_Test2_u + mu*(Grad_Test_u'+Grad_Test_u):Grad_Test2_u",
    "lambda*((Grad_Test2_u@Grad_Test_u):Id(meshdim)):Id(meshdim) + mu*(Grad_Test_u'+Grad_Test_u):Grad_Test2_u",
    "lambda*Id(meshdim)@Id(meshdim)*Grad_Test_u:Grad_Test2_u + mu*(Grad_Test_u'+Grad_Test_u):Grad_Test2_u",
  ];
  for (n, gt) in [(2, "GT_PK(2,1)"), (3, "GT_PK(3,1)")] {
    let (mut ws, mim) = setup(n, 2, gt);
    let mut dense = |e: &str| {
      let k = {
        ws.clear_expressions();
        ws.add_expression(e, &mim, -1).unwrap();
        ws.assembly(2, false).unwrap();
        ws.assembled_matrix()
      }
      .to_csr();
      let mut d = vec![0.0; k.nrows * k.ncols];
      for r in 0..k.nrows {
        for p in k.ptr[r]..k.ptr[r + 1] {
          d[r * k.ncols + k.idx[p]] = k.val[p];
        }
      }
      d
    };
    let d0 = dense(FORMS[0]);
    let scale = d0.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    for e in &FORMS[1..] {
      let err = dense(e)
        .iter()
        .zip(&d0)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
      assert!(err <= 1e-12 * scale, "{gt}: {e}: differs by {err:e}");
    }
  }
}

/// contractions of two vectorized gradients (ContractionOpt22, a scalar
/// block), also in sums with Opt11, against the same forms on dense
/// tensors (Grad_Test_u*Id: no structure); fused (GEMM threshold as set, 0,
/// never) and point by point
#[test]
fn structured_forms_agree() {
  const PAIRS: &[(&str, &str)] = &[
    (
      "Grad_Test_u:Grad_Test2_u",
      "(Grad_Test_u*Id(meshdim)):(Grad_Test2_u*Id(meshdim))",
    ),
    (
      "Test_u.Test2_u + mu2*Grad_Test_u:Grad_Test2_u",
      "(Id(meshdim)*Test_u).Test2_u + mu2*(Grad_Test_u*Id(meshdim)):Grad_Test2_u",
    ),
    (
      "Grad_Test2_u:Grad_Test_u*lambda - Test2_u.Test_u/mu",
      "(Grad_Test2_u*Id(meshdim)):Grad_Test_u*lambda - (Id(meshdim)*Test2_u).Test_u/mu",
    ),
  ];
  for (n, gt) in [(2, "GT_PK(2,1)"), (2, "GT_QK(2,2)"), (3, "GT_PK(3,1)")] {
    let (mut ws, mim) = setup(n, 2, gt);
    let mut dense = |e: &str| {
      let k = {
        ws.clear_expressions();
        ws.add_expression(e, &mim, -1).unwrap();
        ws.assembly(2, false).unwrap();
        ws.assembled_matrix()
      }
      .to_csr();
      let mut d = vec![0.0; k.nrows * k.ncols];
      for r in 0..k.nrows {
        for p in k.ptr[r]..k.ptr[r + 1] {
          d[r * k.ncols + k.idx[p]] = k.val[p];
        }
      }
      d
    };
    for (e, e0) in PAIRS {
      let d0 = dense(e0);
      let scale = d0.iter().fold(0.0f64, |m, x| m.max(x.abs()));
      for (fuse, g) in [
        (false, None),
        (true, None),
        (true, Some(0)),
        (true, Some(usize::MAX)),
      ] {
        crate::ga::NO_FUSE.set(!fuse);
        crate::ga::TEST_GEMM_MIN.set(g);
        let d = dense(e);
        crate::ga::NO_FUSE.set(false);
        crate::ga::TEST_GEMM_MIN.set(None);
        let err = d
          .iter()
          .zip(&d0)
          .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
        assert!(
          err <= 1e-12 * scale,
          "{gt}: {e} (fused {fuse}, {g:?}): differs by {err:e}"
        );
      }
    }
  }
}

/// Diff differentiates var through any transformation too (any_trans)
#[test]
fn diff_through_a_transformation() {
  let (mut ws, mim) = setup(2, 2, "GT_PK(2,1)");
  let m = mim.linked_mesh();
  ws.add_interpolate_transformation_from_expression("T", &m, &m, "0.9*X+0.05*X")
    .unwrap();
  let a = {
    ws.clear_expressions();
    ws.add_expression("Diff(Interpolate(p,T)*p, p)", &mim, -1)
      .unwrap();
    ws.assembly(1, false).unwrap();
    ws.assembled_vector()
  }
  .to_vec();
  let b = {
    ws.clear_expressions();
    ws.add_expression(
      "Interpolate(Test_p,T)*p + Interpolate(p,T)*Test_p",
      &mim,
      -1,
    )
    .unwrap();
    ws.assembly(1, false).unwrap();
    ws.assembled_vector()
  };
  let s = b.iter().fold(0.0f64, |x, v| x.max(v.abs()));
  assert!(s > 0.0);
  let err = a
    .iter()
    .zip(b)
    .fold(0.0f64, |x, (u, v)| x.max((u - v).abs()));
  assert!(err <= 1e-12 * s, "differs by {err:e}");
}

/// Interpolate_filter without a third argument: every point with a target
/// (GetFEM: nbc1 size_type(-1), status != 0)
#[test]
fn filter_without_status_passes_every_target() {
  let (mut ws, mim) = setup(2, 2, "GT_PK(2,1)");
  let m = mim.linked_mesh();
  ws.add_interpolate_transformation_from_expression("S", &m, &m, "1.5*X")
    .unwrap();
  let mut f = |e: &str| {
    ws.clear_expressions();
    ws.add_expression(e, &mim, -1).unwrap();
    ws.assembly(0, false).unwrap();
    ws.assembled_potential()
  };
  let all = f("Interpolate_filter(S, 1)");
  let (s1, s2) = (
    f("Interpolate_filter(S, 1, 1)"),
    f("Interpolate_filter(S, 1, 2)"),
  );
  let none = f("Interpolate_filter(S, 1, 0)");
  assert!(all > 0.0 && none > 0.0, "{all} {none}");
  assert!((all - (s1 + s2)).abs() < 1e-14 && (all + none - 1.0).abs() < 1e-14);
}

/// Grad(Interpolate_filter(t, e, i)) = Interpolate_filter(t, Grad(e), i)
#[test]
fn grad_of_a_filter() {
  let (mut ws, mim) = setup(2, 2, "GT_PK(2,1)");
  let m = mim.linked_mesh();
  ws.add_interpolate_transformation_from_expression("S", &m, &m, "1.5*X")
    .unwrap();
  let mut f = |e: &str| {
    ws.clear_expressions();
    ws.add_expression(e, &mim, -1).unwrap();
    ws.assembly(0, false).unwrap();
    ws.assembled_potential()
  };
  for st in ["", ", 1", ", 0"] {
    let a = f(&format!("Grad(Interpolate_filter(S, p*p*X(1){st})).[1;2]"));
    let b = f(&format!(
      "Interpolate_filter(S, (2*p*X(1)*Grad_p + p*p*[1;0]).[1;2]{st})"
    ));
    assert!(a.abs() > 1e-3 && (a - b).abs() < 1e-13, "{st}: {a} {b}");
  }
}

/// reg_pos_part: its first derivative is DER_REG_POS_PART
#[test]
fn reg_pos_part_derivatives() {
  let (mut ws, mim) = setup(2, 2, "GT_PK(2,1)");
  let mut f = |e: &str| {
    ws.clear_expressions();
    ws.add_expression(e, &mim, -1).unwrap();
    ws.assembly(0, false).unwrap();
    ws.assembled_potential()
  };
  // unit square: the integral is the value
  assert!((f("Diff(reg_pos_part(p, 0.5), p, 1)") - f("DER_REG_POS_PART(p, 0.5)")).abs() < 1e-14);
  assert!((f("DER_REG_POS_PART(0.25, 0.5)") - 0.5).abs() < 1e-14);
  assert!((f("DER2_REG_POS_PART(0.25, 0.5)") - 2.0).abs() < 1e-14);
  assert!((f("DER_REG_POS_PART(0.75, 0.5)") - 1.0).abs() < 1e-14);
}

/// Grad: element_K, element_B are constant on the element (GetFEM
/// ga_node_mark_tree_for_grad)
#[test]
fn grad_with_element_k() {
  let (mut ws, mim) = setup(2, 2, "GT_PK(2,1)");
  let mut f = |e: &str| {
    ws.clear_expressions();
    ws.add_expression(e, &mim, -1).unwrap();
    ws.assembly(0, false).unwrap();
    ws.assembled_potential()
  };
  let (a, b) = (
    f("Grad(p*element_K(1,1)).Grad_p"),
    f("element_K(1,1)*Grad_p.Grad_p"),
  );
  assert!(b != 0.0 && (a - b).abs() < 1e-12 * b.abs(), "{a} {b}");
}

/// element_K, element_B of 1D elements in 2D: vectors of size N (the
/// elements' dimension, not the mesh's; element_B N x P as K)
#[test]
fn element_k_of_a_line_in_2d() {
  use getfem_linalg::tensor::Mat;
  // [0,1] in 4 elements mapped onto (3,4) t: length 5, K = +-[3;4]/4
  // (alternating orientations)
  let m = regular_unit_mesh(&[4], &geotrans_descriptor("GT_PK(1,1)").unwrap()).unwrap();
  m.transform(&Mat::from_data(&[2, 1], vec![3.0, 4.0]).unwrap())
    .unwrap();
  let mim = MeshIm::new(&m).with_integration_method(2).unwrap();
  let mut ws = Workspace::new();
  let mut f = |e: &str| {
    ws.clear_expressions();
    ws.add_expression(e, &mim, -1).unwrap();
    ws.assembly(0, false).unwrap();
    ws.assembled_potential()
  };
  // B = K / |K|^2 = +-[3,4] 4/25
  for (e, v) in [
    ("abs(element_K(1))", 3.75),
    ("abs(element_K(2))", 5.0),
    ("abs(element_B(2))", 3.2),
    ("element_B.element_K", 5.0),
  ] {
    let a = f(e);
    assert!((a - v).abs() < 1e-13, "{e}: {a} {v}");
  }
}

/// element_B of triangles in 3D: N x P as K, B'K = I (a reshape of B
/// instead of its transpose fails with a non-orthogonal map)
#[test]
fn element_b_of_a_surface_in_3d() {
  use getfem_linalg::tensor::Mat;
  let m = regular_unit_mesh(&[2, 2], &geotrans_descriptor("GT_PK(2,1)").unwrap()).unwrap();
  // the unit square onto a parallelogram in 3D: area |a x b| = sqrt(12)
  m.transform(&Mat::from_rows(&[&[1.0, 1.0], &[0.0, 2.0], &[1.0, 3.0]]))
    .unwrap();
  let mim = MeshIm::new(&m).with_integration_method(2).unwrap();
  let mut ws = Workspace::new();
  let mut f = |e: &str| {
    ws.clear_expressions();
    ws.add_expression(e, &mim, -1).unwrap();
    ws.assembly(0, false).unwrap();
    ws.assembled_potential()
  };
  let area = 12.0_f64.sqrt();
  assert!((f("1") - area).abs() < 1e-13);
  let e = f("Norm_sqr(element_B'*element_K - Id(2))");
  assert!(e.abs() < 1e-24, "{e}");
  let t = f("Trace(element_K'*element_B)");
  assert!((t - 2.0 * area).abs() < 1e-12, "{t}");
}

/// A hypersurface x = f(s), s in [0,1]^(n-1), meshed by simplices of
/// degree k, nx subdivisions per direction, consistently oriented (GetFEM
/// test_element_curvature build_mapped_mesh)
fn mapped_mesh(
  n: usize,
  k: usize,
  nx: usize,
  f: impl Fn(&[f64]) -> Vec<f64>,
) -> getfem_geom::mesh::Mesh {
  let gt = geotrans_descriptor(&format!("GT_PK({},{k})", n - 1)).unwrap();
  let h = 1.0 / nx as f64;
  let mut simplices: Vec<Vec<[f64; 2]>> = vec![];
  for i in 0..nx {
    let x = i as f64 * h;
    if n == 2 {
      simplices.push(vec![[x, 0.0], [x + h, 0.0]]);
    } else {
      for j in 0..nx {
        let y = j as f64 * h;
        simplices.push(vec![[x, y], [x + h, y], [x, y + h]]);
        simplices.push(vec![[x + h, y], [x + h, y + h], [x, y + h]]);
      }
    }
  }
  let m = getfem_geom::mesh::Mesh::new(n);
  for sx in &simplices {
    let pts: Vec<Vec<f64>> = gt
      .geometric_nodes()
      .iter()
      .map(|xi| {
        let mut p = sx[0];
        for (a, &c) in xi.iter().enumerate() {
          for d in 0..2 {
            p[d] += c * (sx[a + 1][d] - sx[0][d]);
          }
        }
        f(&p)
      })
      .collect();
    let refs: Vec<&[f64]> = pts.iter().map(|p| &p[..]).collect();
    m.add_convex_by_points(&gt, &refs).unwrap();
  }
  m
}

/// sqrt(mean of e over the mesh)
fn rms(m: &getfem_geom::mesh::Mesh, e: &str) -> f64 {
  let mim = MeshIm::new(m).with_integration_method(10).unwrap();
  let mut ws = Workspace::new();
  let mut f = |e: &str| {
    ws.clear_expressions();
    ws.add_expression(e, &mim, -1).unwrap();
    ws.assembly(0, false).unwrap();
    ws.assembled_potential()
  };
  (f(e) / f("1")).sqrt()
}

/// element_curvature on curved meshes of degree k = 2, 3 (GetFEM's
/// test_element_curvature): sphere (tensor, mixed components, mean and
/// Gauss curvatures), cylinder, circle; errors as GetFEM's tolerances at
/// nx = 8, decreasing as h^(k-1) from nx = 4; zero on a linear plane
#[test]
fn element_curvature_of_curved_meshes() {
  let r = 2.0_f64;
  let sphere = move |p: &[f64]| {
    let (th, ph) = (0.5 + 1.5 * p[0], 1.5 * p[1]);
    vec![
      r * th.sin() * ph.cos(),
      r * th.sin() * ph.sin(),
      r * th.cos(),
    ]
  };
  let cylinder = move |p: &[f64]| {
    let ph = 2.0 * p[0];
    vec![r * ph.cos(), r * ph.sin(), 3.0 * p[1]]
  };
  let circle = move |p: &[f64]| {
    let ph = 3.0 * p[0];
    vec![r * ph.cos(), r * ph.sin()]
  };
  let c = "element_curvature";
  let pr = "(Id(3)-X@X/4)";
  let q = "([X(1);X(2);0]@[X(1);X(2);0])/4";
  let sphere_cases = [
    format!("Norm_sqr({c} - {pr}/2)*4"),
    format!("Norm_sqr(element_B'*{c}*element_K - Id(2)/2)*4"),
    format!("sqr(Trace({c})/2 - 1/2)*4"),
    format!("Norm_sqr({c}*{c} - {pr}/4)*16"),
    format!("sqr((sqr(Trace({c}))-Trace({c}*{c}))/2 - 1/4)*16"),
  ];
  for k in 2..=3 {
    let tol = if k == 2 { 2e-1 } else { 2e-2 };
    let mut cases: Vec<(getfem_geom::mesh::Mesh, getfem_geom::mesh::Mesh, String)> = vec![];
    for e in &sphere_cases {
      cases.push((
        mapped_mesh(3, k, 4, sphere),
        mapped_mesh(3, k, 8, sphere),
        e.clone(),
      ));
    }
    cases.push((
      mapped_mesh(3, k, 4, cylinder),
      mapped_mesh(3, k, 8, cylinder),
      format!("Norm_sqr({c} - (Id(3)-[0;0;1]@[0;0;1]-{q})/2)*4"),
    ));
    cases.push((
      mapped_mesh(2, k, 4, circle),
      mapped_mesh(2, k, 8, circle),
      format!("Norm_sqr({c} - (Id(2)-X@X/4)/2)*4"),
    ));
    for (m4, m8, e) in &cases {
      let (e4, e8) = (rms(m4, e), rms(m8, e));
      let rate = (e4 / e8).log2();
      assert!(e8 < tol, "k = {k}, {e}: {e8}");
      assert!(
        rate > k as f64 - 1.3,
        "k = {k}, {e}: rate {rate} ({e4}, {e8})"
      );
    }
  }
  let plane = mapped_mesh(3, 1, 4, |p| vec![p[0], p[1], p[0] + 2.0 * p[1]]);
  assert!(rms(&plane, &format!("Norm_sqr({c})")) < 1e-12);
  // a volume mesh: refused
  let (mut ws, mim) = setup(2, 1, "GT_PK(2,1)");
  ws.add_expression(&format!("Trace({c})"), &mim, -1).unwrap();
  let err = ws.assembly(0, false).unwrap_err().to_string();
  assert!(err.contains("curves in 2D and surfaces in 3D"), "{err}");
}

/// Normal inside a curve in 2D or a surface in 3D: the element's unit
/// normal, oriented as element_curvature (K_1 x K_2, (K_21, -K_11)), so
/// element_curvature = grad_s Normal (outward on the sphere and circle
/// below, element_curvature = P/R); inside other elements an error
#[test]
fn normal_inside_curves_and_surfaces() {
  let r = 2.0_f64;
  let sphere = move |p: &[f64]| {
    let (th, ph) = (0.5 + 1.5 * p[0], 1.5 * p[1]);
    vec![
      r * th.sin() * ph.cos(),
      r * th.sin() * ph.sin(),
      r * th.cos(),
    ]
  };
  let circle = move |p: &[f64]| {
    let ph = 3.0 * p[0];
    vec![r * ph.cos(), r * ph.sin()]
  };
  let plane = mapped_mesh(3, 1, 4, |p| vec![p[0], p[1], p[0] + 2.0 * p[1]]);
  let e = rms(&plane, "Norm_sqr(Normal - [-1;-2;1]/sqrt(6))");
  assert!(e < 1e-14, "{e}");
  // errors O(h^k) for Normal, O(h^(k-1)) for element_curvature
  let e = "Norm_sqr(Normal - X/2)";
  let c = "Norm_sqr(element_curvature - (Id(meshdim) - Normal@Normal)/2)";
  for (k, te, tc) in [(2, 2e-3, 2e-1), (3, 2e-4, 2e-2)] {
    let kf = k as f64;
    for (d, f) in [(3, &sphere as &dyn Fn(&[f64]) -> Vec<f64>), (2, &circle)] {
      let (m4, m8) = (mapped_mesh(d, k, 4, f), mapped_mesh(d, k, 8, f));
      for (x, tol, q) in [(e, te, kf), (c, tc, kf - 1.0)] {
        let (e4, e8) = (rms(&m4, x), rms(&m8, x));
        let rate = (e4 / e8).log2();
        assert!(e8 < tol, "k = {k}, {x}: {e8}");
        assert!(rate > q - 0.3, "k = {k}, {x}: rate {rate} ({e4}, {e8})");
      }
    }
  }
  // a 2D element in 2D, a line in 3D: refused
  let line3 = getfem_geom::mesh::Mesh::new(3);
  let gt1 = geotrans_descriptor("GT_PK(1,1)").unwrap();
  line3
    .add_convex_by_points(&gt1, &[&[0.0, 0.0, 0.0], &[1.0, 2.0, 0.0]])
    .unwrap();
  let (mut ws, mim) = setup(2, 1, "GT_PK(2,1)");
  ws.add_expression("Normal(1)", &mim, -1).unwrap();
  let err = ws.assembly(0, false).unwrap_err().to_string();
  assert!(
    err.contains("Normal inside element 0 (of dimension 2 in 2D)"),
    "{err}"
  );
  let mim = MeshIm::new(&line3).with_integration_method(2).unwrap();
  let mut ws = Workspace::new();
  ws.add_expression("Normal(1)", &mim, -1).unwrap();
  let err = ws.assembly(0, false).unwrap_err().to_string();
  assert!(err.contains("(of dimension 1 in 3D)"), "{err}");
}

/// Plane_Strain_ SVK: GetFEM's plain operators, in any dimension
#[test]
fn plane_strain_svk_in_3d() {
  let (mut ws, mim) = setup(3, 1, "GT_PK(3,1)");
  let mut v = |e: &str| {
    {
      ws.clear_expressions();
      ws.add_expression(e, &mim, -1).unwrap();
      ws.assembly(1, false).unwrap();
      ws.assembled_vector()
    }
    .to_vec()
  };
  for law in ["sigma", "PK2"] {
    let e = format!("Saint_Venant_Kirchhoff_{law}(Grad_u, [1;2]):Grad_Test_u");
    assert_eq!(v(&format!("Plane_Strain_{e}")), v(&e));
  }
  let mut s = |e: &str| {
    ws.clear_expressions();
    ws.add_expression(e, &mim, -1).unwrap();
    ws.assembly(0, false).unwrap();
    ws.assembled_potential()
  };
  let e = "Saint_Venant_Kirchhoff_potential(Grad_u, [1;2])";
  assert_eq!(s(&format!("Plane_Strain_{e}")), s(e));
}

/// 1D, vector u: Grad of any expression is [q, 1] (GetFEM ga_node_grad),
/// Grad_u is [q]
#[test]
fn grad_shape_in_1d() {
  let m = regular_unit_mesh(&[2], &geotrans_descriptor("GT_PK(1,1)").unwrap()).unwrap();
  let mim = MeshIm::new(&m).with_integration_method(2).unwrap();
  let mf = MeshFem::new(&m, 2)
    .with_classical_finite_element(1)
    .unwrap();
  let mut ws = Workspace::new();
  ws.add_fem_variable("u", &mf).unwrap();
  let n = mf.nb_dof().unwrap();
  ws.set_variable("u", &(0..n).map(|i| i as f64).collect::<Vec<_>>())
    .unwrap();
  let mut s = |e: &str| {
    ws.clear_expressions();
    ws.add_expression(e, &mim, -1).unwrap();
    ws.assembly(0, false).unwrap();
    ws.assembled_potential()
  };
  let (a, b) = (s("Grad(u)(2,1)"), s("(Grad(2*u)/2)(2,1)"));
  assert!(a.abs() > 1.0 && (a - b).abs() < 1e-13 && (a - s("Grad_u(2)")).abs() < 1e-13);
  let mut v = |e: &str| {
    {
      ws.clear_expressions();
      ws.add_expression(e, &mim, -1).unwrap();
      ws.assembly(1, false).unwrap();
      ws.assembled_vector()
    }
    .to_vec()
  };
  let (a, b) = (v("Grad(Test_u)(2,1)"), v("(Grad(2*Test_u)/2)(2,1)"));
  assert!(a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-13));
  assert!(
    a.iter()
      .zip(v("Grad_Test_u(2)"))
      .all(|(x, y)| (x - y).abs() < 1e-13)
  );
}

/// Ogden (GetFEM snippet): the two forms of the stress term agree
#[test]
fn ogden_forms_agree() {
  let (mut ws, mim) = setup(3, 1, "GT_PK(3,1)");
  for (m, e) in [
    ("F", "Id(meshdim)+Grad_u/10"),
    ("J", "Det(F)"),
    ("tauH", "3*log(J)"),
    ("tauD", "2*Deviator(Expm(Deviator(Logm(F*F'))))"),
  ] {
    ws.add_macro(m, e).unwrap();
  }
  let a = "((tauH*Id(meshdim)+tauD)*Inv(F')):Grad(Test_u)";
  let b = "(tauH*Id(meshdim)+tauD):(Grad(Test_u)*Inv(F))";
  let (va, vb) = (
    {
      ws.clear_expressions();
      ws.add_expression(a, &mim, -1).unwrap();
      ws.assembly(1, false).unwrap();
      ws.assembled_vector()
    }
    .to_vec(),
    {
      ws.clear_expressions();
      ws.add_expression(b, &mim, -1).unwrap();
      ws.assembly(1, false).unwrap();
      ws.assembled_vector()
    },
  );
  let s = va.iter().fold(0.0f64, |m, x| m.max(x.abs()));
  assert!(s > 1e-3 && va.iter().zip(vb).all(|(x, y)| (x - y).abs() < 1e-12 * s));
  let (ka, kb) = (
    {
      ws.clear_expressions();
      ws.add_expression(a, &mim, -1).unwrap();
      ws.assembly(2, false).unwrap();
      ws.assembled_matrix()
    }
    .to_csr()
    .to_dense()
    .concat(),
    {
      ws.clear_expressions();
      ws.add_expression(b, &mim, -1).unwrap();
      ws.assembly(2, false).unwrap();
      ws.assembled_matrix()
    }
    .to_csr()
    .to_dense()
    .concat(),
  );
  let s = ka.iter().fold(0.0f64, |m, x| m.max(x.abs()));
  assert!(s > 1e-3 && ka.iter().zip(&kb).all(|(x, y)| (x - y).abs() < 1e-11 * s));
}

#[test]
fn terms_of_the_same_tests_in_one_tree() {
  // GetFEM add_tree: one tree per test function, also without data to fold
  let (ws, mim) = setup(2, 2, "GT_PK(2,1)");
  let rg = getfem_geom::region::RegionArg::from(-1)
    .on(&mim.linked_mesh())
    .unwrap();
  let mut all = vec![];
  for e in ["Grad_p.Grad_Test_p", "p*Test_p", "u.Test_u"] {
    all.extend(ws.trees(e, &mim, &rg, 1).unwrap().iter().cloned());
  }
  let t = ws
    .eval_fixed_size(std::sync::Arc::new(all), &mim, &rg)
    .unwrap();
  assert_eq!(t.len(), 2);
}

/// The instructions of hyperelastic tangents (a compressible neo-Hookean
/// law, plane strain): the tangent of its stress term and of the
/// derivative of its second variation in a direction w (a user's
/// topology optimization against buckling); the instructions on tensors
/// with the tests of two variables (element matrix sized)
#[test]
fn hyperelastic_tangent_instructions() {
  let m = regular_unit_mesh(&[2, 2], &geotrans_descriptor("GT_PK(2,1)").unwrap()).unwrap();
  let mf = MeshFem::new(&m, 2)
    .with_classical_finite_element(2)
    .unwrap();
  let mim = MeshIm::new(&m).with_integration_method(4).unwrap();
  let nd = mf.nb_dof().unwrap();
  let seq = |s: f64| -> Vec<f64> {
    (0..nd)
      .map(|i| 0.1 * (s * (i as f64 + 1.0)).sin())
      .collect()
  };
  let mut ws = Workspace::new();
  ws.add_scalar_data("lambda", 3.0).unwrap();
  ws.add_scalar_data("mu", 2.0).unwrap();
  ws.add_fem_variable("u", &mf).unwrap();
  ws.set_variable("u", &seq(0.7)).unwrap();
  ws.add_fem_data("w", &mf, &seq(0.13)).unwrap();
  for (name, e) in [
    ("Fc", "Id(2)+Grad_u"),
    ("Dev33(A)", "A-Id(2)/3*(Trace(A)+1)"),
    (
      "tauc",
      "lambda*log(Det(Fc))*Id(2)+mu*pow(Det(Fc),-2/3)*Dev33(Fc*Fc')",
    ),
    (
      "dPw",
      "(lambda*(Inv(Fc'):Grad_w)*Id(2)+mu*pow(Det(Fc),-2/3)*(-2/3*(Inv(Fc'):Grad_w)*Dev33(Fc*Fc')\
       +Grad_w*Fc'+Fc*Grad_w'-Id(2)/3*Trace(Grad_w*Fc'+Fc*Grad_w')))*Inv(Fc')\
       -tauc*Inv(Fc')*Grad_w'*Inv(Fc')",
    ),
    ("PsiW", "dPw:Grad_w"),
  ] {
    ws.add_macro(name, e).unwrap();
  }
  let matrix = |ws: &mut Workspace, e: &str, off: bool| {
    crate::ga::rewrite::OFF.with(|o| o.set(off));
    ws.clear_expressions();
    ws.add_expression(e, &mim, None).unwrap();
    ws.assembly(2, false).unwrap();
    crate::ga::rewrite::OFF.with(|o| o.set(false));
    let k = ws.assembled_matrix();
    (0..k.ncols)
      .flat_map(|j| (0..k.nrows).map(move |i| (i, j)))
      .map(|(i, j)| k.get(i, j))
      .collect::<Vec<_>>()
  };
  for e in ["(tauc*Inv(Fc')):Grad_Test_u", "Diff(PsiW,u,Test_u)"] {
    // the same matrix as without the rewrite
    let k0 = matrix(&mut ws, e, true);
    crate::ga::COMPILED.with(|c| c.borrow_mut().clear());
    let k = matrix(&mut ws, e, false);
    let scale = k0.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let diff = k
      .iter()
      .zip(&k0)
      .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(diff <= 1e-12 * scale, "{e}: {diff} against {scale}");
    crate::ga::COMPILED.with(|c| {
      for s in c.borrow().iter() {
        let n: usize = s.kinds.values().sum();
        eprintln!(
          "{e}: {n} instructions, {} on two tests, fused {:?}",
          s.two_tests, s.fused_leaves
        );
        eprintln!("  {:?}", s.kinds);
        eprintln!("  on two tests: {:?}", s.two_tests_kinds);
      }
    });
    if std::env::var("GWFL_TREES").is_ok() {
      crate::ga::TREES.with(|c| c.borrow().iter().for_each(|t| eprintln!("{t}")));
    }
    crate::ga::TREES.with(|c| c.borrow_mut().clear());
  }
}

/// The contractions pushed into sums and products (ga/rewrite.rs) give
/// the matrices of the expressions as written
#[test]
fn rewrite_keeps_matrices() {
  const EXPRS: &[&str] = &[
    // a sum, a product of the tests of both variables
    "((Grad_Test_u*Grad_Test2_u) - (Grad_Test2_u*Grad_Test_u)/3):(Grad_u+Id(meshdim))",
    // products by test-free matrices on both sides
    "((Grad_u*Grad_Test_u)*(Grad_Test2_u*Grad_u')):Grad_u",
    "(Grad_Test_u*Grad_Test2_u*Grad_u):Id(meshdim)",
    // scalars with tests, a negation
    "((Div_Test_u*Div_Test2_u)*Grad_u - Trace(Grad_Test_u)*Grad_Test2_u):Grad_u",
    "-(Grad_Test_u*Grad_u'*Grad_Test2_u):(mu*Grad_u)",
    "((Test_p*Grad_Test2_u)*Grad_u'):Grad_u - (Grad_Test_p.Grad_p)*Div_Test2_u",
  ];
  for (n, gt) in [(2, "GT_PK(2,1)"), (3, "GT_PK(3,1)")] {
    let (mut ws, mim) = setup(n, 2, gt);
    for e in EXPRS {
      let mut k = vec![];
      for off in [true, false] {
        crate::ga::rewrite::OFF.with(|o| o.set(off));
        ws.clear_expressions();
        ws.add_expression(e, &mim, None).unwrap();
        ws.assembly(2, false).unwrap();
        crate::ga::rewrite::OFF.with(|o| o.set(false));
        let m = ws.assembled_matrix();
        k.push(
          (0..m.ncols)
            .flat_map(|j| (0..m.nrows).map(move |i| (i, j)))
            .map(|(i, j)| m.get(i, j))
            .collect::<Vec<_>>(),
        );
      }
      let scale = k[0].iter().fold(0.0f64, |m, x| m.max(x.abs()));
      let diff = k[0]
        .iter()
        .zip(&k[1])
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
      assert!(
        scale > 0.0 && diff <= 1e-12 * scale,
        "{n}D {e}: {diff} against {scale}"
      );
    }
  }
}
