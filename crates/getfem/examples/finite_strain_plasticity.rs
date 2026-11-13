//! Necking of a plane strain block, finite strain plasticity of
//! Eterovic-Bathe type with linear and exponential hardening/softening
//! (GetFEM's finite_strain_plasticity_ksi.py): the plastic multiplier
//! increment ksi an internal variable, condensed. A quarter of the block,
//! pulled along x by a normal displacement on the right face (a
//! multiplier), the displacement 5 lf, lf the load factor from 0 to 1. In
//! load steps (u extrapolated from the two last steps), or by Moore-Penrose
//! continuation in lf (the script's numerical_continuation settings and its
//! non-smooth option) past the force peak, to lf = 1. Writes the force per step and the final state to
//! target/showcase/finite_strain_plasticity_2D*. Optional arguments: the
//! number of steps (default 200), `continuation`.

use std::collections::BTreeSet;
use std::io::Write;

use getfem::fem::fem::fem_descriptor;
use getfem::fem::im_data::ImData;
use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im::MeshIm;
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::region::{RegionSet, outer_faces_of_mesh};
use getfem::geom::regular_mesh::regular_unit_mesh;
use getfem::io::vtk::{Format, VtuExport};
use getfem::linalg::tensor::Tensor;
use getfem::model::{
  ContinuationStatus, Method, Model, Newton, NumericalContinuation, VariableScale,
};

const L: f64 = 2.0 * 26.667;
const H: f64 = 2.0 * 6.413;
const LEFT: usize = 1;
const RIGHT: usize = 2;
const BOTTOM: usize = 3;

fn main() {
  let steps: usize = std::env::args().nth(1).map_or(200, |s| s.parse().unwrap());
  let continuation = match std::env::args().nth(2).as_deref() {
    None => false,
    Some("continuation") => true,
    Some(c) => panic!("{c}: continuation or nothing"),
  };
  let out = format!("{}/../../target/showcase", env!("CARGO_MANIFEST_DIR"));
  std::fs::create_dir_all(&out).unwrap();
  let (e, nu) = (206.9e3, 0.29);
  let (s0, sinf, hp, delta) = (0.45e3, 0.715e3, -0.012924e3, 16.93);

  // quarter of the block, graded along x, thinner at the center
  let m = regular_unit_mesh(&[20, 10], &geotrans_descriptor("GT_QK(2,2)").unwrap()).unwrap();
  m.transform(&Tensor::from_data(&[2, 2], vec![L / 2.0, 0.0, 0.0, H / 2.0]).unwrap())
    .unwrap();
  let mut rs = vec![RegionSet::new(); 4];
  for (cv, f) in outer_faces_of_mesh(&m)
    .resolve(&m.snapshot())
    .unwrap()
    .iter()
  {
    let f = f.unwrap();
    let nr = m.mean_normal_of_face_of_convex(cv, f).unwrap();
    for c in 0..2 {
      if nr[c] < -0.95 {
        rs[2 * c].add_face(cv, f);
      } else if nr[c] > 0.95 {
        rs[2 * c + 1].add_face(cv, f);
      }
    }
  }
  for (rg, k) in [(LEFT, 0), (RIGHT, 1), (BOTTOM, 2)] {
    m.set_region_set(rg, rs[k].clone()).unwrap();
  }
  let (dy, y0, dh) = (0.2, 0.04, 0.018 * H);
  let x0 = y0 / dy;
  let b2 = (1.0 - dy) / (x0 - 1.0f64).powi(2);
  let b1 = dy - 2.0 * b2 * x0;
  let b0 = 1.0 - b1 - b2;
  let moves: Vec<(usize, Vec<f64>)> = (0..m.nb_points_max())
    .filter(|&i| m.is_point_valid(i))
    .map(|i| {
      let p = m.point(i);
      let t = 2.0 * p[0].abs() / L;
      let x = if t < x0 {
        p[0] * dy
      } else {
        (b0 + b1 * t + b2 * t * t) * p[0].signum() * L / 2.0
      };
      let y = p[1] - (p[1] * dh) / (2.0 * H) * (1.0 + (2.0 * std::f64::consts::PI * x / L).cos());
      (i, vec![x, y])
    })
    .collect();
  m.set_points(&moves).unwrap();

  // displacements without the normal components on the symmetry faces
  let pf = fem_descriptor("FEM_Q2_INCOMPLETE(2)").unwrap();
  let mf_n = MeshFem::new(&m, 2).with_finite_element(&pf).unwrap();
  let mut removed = BTreeSet::new();
  for (rg, c) in [(LEFT, 0), (BOTTOM, 1)] {
    removed.extend(
      mf_n
        .basic_dof_on_region(rg)
        .unwrap()
        .into_iter()
        .filter(|d| d % 2 == c),
    );
  }
  let kept = (0..mf_n.nb_basic_dof().unwrap())
    .filter(|d| !removed.contains(d))
    .collect();
  let mfu = MeshFem::partial(&mf_n, &kept).unwrap();
  let mfdir = MeshFem::new(&m, 1).with_finite_element(&pf).unwrap();
  let mfout = MeshFem::new(&m, 1)
    .with_classical_discontinuous_finite_element(2)
    .unwrap();
  let mim = MeshIm::new(&m).with_integration_method(5).unwrap();
  let mim_vol = MeshIm::new(&m).with_integration_method(3).unwrap();
  let mimd1 = ImData::new(&mim, -1, &[]).unwrap();
  let mimd4 = ImData::new(&mim, -1, &[4]).unwrap();
  let np = mimd1.nb_points().unwrap();

  let mut md = Model::new();
  md.add_fem_variable("u", &mfu).unwrap();
  md.add_internal_im_variable("ksi", &mimd1).unwrap();
  // accumulated plastic strain and components 11, 22, 33, 12 of Cp^-1 at
  // the previous step
  md.add_im_data("gamma0", &mimd1, &vec![0.0; np]).unwrap();
  let id: Vec<f64> = (0..np).flat_map(|_| [1.0, 1.0, 1.0, 0.0]).collect();
  md.add_im_data("invCp0vec", &mimd4, &id).unwrap();
  md.add_scalar_data("lf", 0.0).unwrap();
  md.add_scalar_data("K", e / (3.0 * (1.0 - 2.0 * nu)))
    .unwrap();
  md.add_scalar_data("G", e / (2.0 * (1.0 + nu))).unwrap();
  let c = (2.0f64 / 3.0).sqrt();
  for (k, v) in [
    ("Mat3x3ToVec4", "[[[1,0,0,0],[0,0,0,0.5],[0,0,0,0]],[[0,0,0,0.5],[0,1,0,0],[0,0,0,0]],[[0,0,0,0],[0,0,0,0],[0,0,1,0]]]".to_string()),
    ("Vec4ToMat3x3", "[[[1,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,1,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,1]],[[0,1,0],[1,0,0],[0,0,0]]]".into()),
    ("F", "Id(2)+Grad(u)".into()),
    ("F3d", "Id(3)+[1,0;0,1;0,0]*Grad(u)*[1,0,0;0,1,0]".into()),
    ("J", "Det(F)".into()),
    ("invCp0", "Vec4ToMat3x3.invCp0vec".into()),
    ("devlogbetr", "Deviator(Logm(F3d*invCp0*F3d'))".into()),
    ("devlogbe", "(1-2*ksi)*devlogbetr".into()),
    ("tauD", "G*devlogbe".into()),
    ("tauD2d", "G*([1,0,0;0,1,0]*devlogbe*[1,0;0,1;0,0])".into()),
    ("tauH", "K*log(J)".into()),
    ("tau", "tauH*Id(3)+tauD".into()),
    (
      "Y(gamma)",
      format!("{}+{}*(1-exp(-{}*gamma))+{}*gamma", c * s0, c * (sinf - s0), c * delta, 2.0 / 3.0 * hp),
    ),
    ("gamma", "gamma0+ksi*Norm(devlogbetr)".into()),
    ("tauDtr", "G*devlogbetr".into()),
    (
      "invCp",
      "(Inv(F3d)*Expm(-ksi*devlogbetr)*F3d)*invCp0*(Inv(F3d)*Expm(-ksi*devlogbetr)*F3d)'".into(),
    ),
  ] {
    md.add_macro(k, &v).unwrap();
  }
  md.add_term(&mim, None, "(tauD2d*Inv(F)'):Grad(Test_u)")
    .unwrap();
  md.add_term(&mim_vol, None, "tauH*Inv(F)':Grad(Test_u)")
    .unwrap();
  // yield condition; +1e-12 ksi: the derivative in ksi does not vanish at
  // u = 0
  md.add_term(
    &mim,
    None,
    "(Norm(tauD)-min(Y(gamma),Norm(tauDtr)+1e-12*ksi))*Test_ksi",
  )
  .unwrap();
  md.add_filtered_fem_variable("dirmult", &mfdir, RIGHT)
    .unwrap();
  md.add_term(&mim, RIGHT, "(5*lf-u(1))*dirmult").unwrap();
  println!(
    "{} displacement DOFs, {} model DOFs",
    mfu.nb_dof().unwrap(),
    md.nb_dof()
  );

  // plasticity is non-smooth: NLEQ-RES after full Newton, no NLEQ-ERR
  let newton = Newton {
    methods: vec![Method::Newton, Method::NleqRes],
    atol: 1e-8,
    rtol: 1e-10,
    ..Default::default()
  };
  let mut f =
    std::fs::File::create(format!("{out}/finite_strain_plasticity_2D_forces.dat")).unwrap();
  let center = vec![vec![0.0, 0.0]];
  // the continuation in lf
  let mut cs = NumericalContinuation::new("lf", vec![VariableScale::new(&["u"], 1.0)]);
  (cs.h_init, cs.h_max, cs.h_min, cs.h_dec, cs.h_inc) =
    (1e-1, 1.0, 1e-5, 1.0 / 10f64.sqrt(), 10f64.sqrt());
  (
    cs.max_iter,
    cs.thr_iter,
    cs.max_res,
    cs.max_diff,
    cs.min_cos,
  ) = (12, 6, 1e-8, 1e-8, 0.995);
  (cs.non_smooth, cs.gamma_max) = (true, 1.0);
  let (mut tx, mut tlf, mut h) = (vec![], 1.0, 0.0);
  if continuation {
    cs.step(&mut md, &mut tx, &mut tlf, &mut h, None).unwrap();
  }
  let mut prev: Option<Vec<f64>> = None;
  let mut peak = (0.0, 0, 0.0);
  for step in 1..=steps {
    let iterations = if continuation {
      let rep = cs.step(&mut md, &mut tx, &mut tlf, &mut h, None).unwrap();
      assert_eq!(
        rep.status,
        ContinuationStatus::Converged,
        "step {step}: the continuation failed"
      );
      0
    } else {
      // x stretching 10/L x, at x = L/2
      md.set_variable("lf", &[step as f64 / steps as f64])
        .unwrap();
      // start: u extrapolated from the two last steps
      let u = md.variable("u").unwrap().to_vec();
      if let Some(p) = &prev {
        let guess: Vec<f64> = u.iter().zip(p).map(|(a, b)| 2.0 * a - b).collect();
        md.set_variable("u", &guess).unwrap();
      }
      prev = Some(u);
      let rep = newton.solve(&mut md, None).unwrap();
      assert!(rep.converged(), "step {step}: {}", rep.message);
      rep.iterations
    };
    let lf = md.variable("lf").unwrap()[0];
    let fr = {
      let ws = md.workspace_mut();
      ws.clear_expressions();
      ws.add_expression("dirmult", &mim, RIGHT).unwrap();
      ws.assembly(0, false).unwrap();
      ws.assembled_potential()
    };
    if fr > peak.0 {
      peak = (fr, step, lf);
    }
    // the script's update: the history with 0.99 ksi, the rest the start
    // of the next step
    let cc = 0.99;
    let k: Vec<f64> = md.variable("ksi").unwrap().iter().map(|x| cc * x).collect();
    md.set_variable("ksi", &k).unwrap();
    let g = md.interpolation("gamma", &mimd1, -1).unwrap();
    md.set_variable("gamma0", &g).unwrap();
    let cp = md.interpolation("Mat3x3ToVec4:invCp", &mimd4, -1).unwrap();
    md.set_variable("invCp0vec", &cp).unwrap();
    let k: Vec<f64> = k.iter().map(|x| (1.0 / cc - 1.0) * x).collect();
    md.set_variable("ksi", &k).unwrap();
    let gp = md.local_projection("gamma0", &mim, &mfout, -1).unwrap();
    let gc = getfem::fem::interpolation::interpolation_at_points(&mfout, &center, &gp).unwrap()[0];
    writeln!(
      f,
      "step={step} loadfactor={lf:e} fR=({fr:e},0) sR=({:e},0) gamma_center={gc:e} iterations={iterations} h={h:e}",
      fr / H
    )
    .unwrap();
    if step % 10 == 0 || step == steps || lf == 1.0 {
      println!("step {step}: load factor {lf:.4}, force {fr:.3}, gamma at the center {gc:.4}");
    }
    if lf == 1.0 {
      break;
    }
  }
  println!(
    "force peak {:.3} at step {} (load factor {:.4})",
    peak.0, peak.1, peak.2
  );

  let mut vtu = VtuExport::new(
    &format!("{out}/finite_strain_plasticity_2D.vtu"),
    Format::Ascii,
  );
  vtu.exporting(&mfu).unwrap();
  vtu
    .write_point_data(&mfu, md.variable("u").unwrap(), "u")
    .unwrap();
  for (e, name) in [
    ("sqrt(1.5)*Norm(tauD)/J", "Von Mises stress"),
    ("gamma0", "gamma"),
    ("tau(1,1)/J", "Cauchy stress 11"),
  ] {
    let v = md.local_projection(e, &mim, &mfout, -1).unwrap();
    vtu.write_point_data(&mfout, &v, name).unwrap();
  }
  vtu.write().unwrap();
  println!("wrote {out}/finite_strain_plasticity_2D.vtu and _forces.dat");
}
