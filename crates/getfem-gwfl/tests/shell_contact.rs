//! Raytracing with shells: the elements of a surface in 3D as slave
//! (rays along the deformed element normal) and as master (two-sided),
//! against the top face of a block, both displacements on partial spaces.
//! Exact gaps on a flat shell, derivatives by finite differences on a
//! curved deformed one.

use getfem_fem::mesh_fem::MeshFem;
use getfem_fem::mesh_im::MeshIm;
use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::mesh::Mesh;
use getfem_geom::region::{Region, RegionSet, outer_faces_of_mesh, select_faces_of_normal};
use getfem_geom::regular_mesh::regular_mesh;
use getfem_gwfl::workspace::Workspace;

const SHELL: i32 = 1;
const TOP: i32 = 2;

/// the square [0.2,0.8]^2 at height z(x, y) (GT_QK(2,2), 3x3), its normal
/// K_1 x K_2 down if flip (x, y swapped), up otherwise; region SHELL: its
/// elements
fn shell(z: impl Fn(f64, f64) -> f64, flip: bool) -> Mesh {
  let gt = geotrans_descriptor("GT_QK(2,2)").unwrap();
  let sq = regular_mesh(&gt, &[3, 3], &[0.2, 0.2], &[0.6, 0.6]).unwrap();
  let m = Mesh::new(3);
  let mut rs = RegionSet::new();
  for cv in 0..sq.nb_convexes() {
    let g = sq.points_of_convex(cv);
    let pts: Vec<Vec<f64>> = (0..g.ncols())
      .map(|j| {
        let (x, y) = if flip {
          (g[(1, j)], g[(0, j)])
        } else {
          (g[(0, j)], g[(1, j)])
        };
        vec![x, y, z(x, y)]
      })
      .collect();
    let refs: Vec<&[f64]> = pts.iter().map(|p| &p[..]).collect();
    rs.add(m.modify(|d| d.add_convex_by_points(&gt, &refs)).unwrap());
  }
  m.set_region(SHELL as usize, &Region::temporary(&m, rs).unwrap())
    .unwrap();
  m
}

/// the block [0,1]^2 x [-0.5,0] (GT_QK(3,1), 5x5x1, aligned with the
/// shell), region TOP: its top
/// faces
fn block() -> Mesh {
  let m = regular_mesh(
    &geotrans_descriptor("GT_QK(3,1)").unwrap(),
    &[5, 5, 1],
    &[0.0, 0.0, -0.5],
    &[1.0, 1.0, 0.5],
  )
  .unwrap();
  let top = select_faces_of_normal(&m, outer_faces_of_mesh(&m), &[0.0, 0.0, 1.0], 0.01).unwrap();
  m.set_region(TOP as usize, &top).unwrap();
  m
}

/// f at the basic DOFs, reduced
fn values(mf: &MeshFem, f: impl Fn(&[f64], usize) -> f64) -> Vec<f64> {
  let v: Vec<f64> = (0..mf.nb_basic_dof().unwrap())
    .map(|i| f(&mf.point_of_basic_dof(i).unwrap(), i % 3))
    .collect();
  mf.reduce_vector(&v).unwrap()
}

/// shell displacement us, block displacement ub; the shell slave (rays
/// along its normal) or master
fn setup(s: &Mesh, b: &Mesh, shell_slave: bool, deformed: bool) -> Workspace {
  // partial spaces (reduction matrices): the shell's x at x = 0.2, the
  // block's z at its bottom removed
  let partial = |mf: MeshFem, removed: &dyn Fn(&[f64], usize) -> bool| {
    let kept = (0..mf.nb_dof().unwrap())
      .filter(|&i| !removed(&mf.point_of_basic_dof(i).unwrap(), i % 3))
      .collect();
    MeshFem::partial(&mf, &kept).unwrap()
  };
  let mfs = partial(
    MeshFem::new(s, 3).with_classical_finite_element(2).unwrap(),
    &|p, c| c == 0 && p[0] < 0.2 + 1e-9,
  );
  let mfb = partial(
    MeshFem::new(b, 3).with_classical_finite_element(1).unwrap(),
    &|p, c| c == 2 && p[2] < -0.5 + 1e-9,
  );
  let mut ws = Workspace::new();
  ws.add_fem_variable("us", &mfs).unwrap();
  ws.add_fem_variable("ub", &mfb).unwrap();
  if deformed {
    let us = values(&mfs, |p, c| 0.01 * ((2 + c) as f64 * p[0] + p[1]).sin());
    let ub = values(&mfb, |p, c| 0.008 * (p[0] - (1 + c) as f64 * p[1]).cos());
    ws.set_variable("us", &us).unwrap();
    ws.set_variable("ub", &ub).unwrap();
  }
  ws.add_raytracing_transformation("ray", 0.2).unwrap();
  if shell_slave {
    ws.add_slave_contact_boundary_to_raytracing_transformation("ray", s, "us", SHELL as usize)
      .unwrap();
    ws.add_master_contact_boundary_to_raytracing_transformation("ray", b, "ub", TOP as usize)
      .unwrap();
  } else {
    ws.add_slave_contact_boundary_to_raytracing_transformation("ray", b, "ub", TOP as usize)
      .unwrap();
    ws.add_master_contact_boundary_to_raytracing_transformation("ray", s, "us", SHELL as usize)
      .unwrap();
  }
  ws
}

fn potential(ws: &mut Workspace, mim: &MeshIm, e: &str, rg: i32) -> f64 {
  ws.clear_expressions();
  ws.add_expression(e, mim, rg).unwrap();
  ws.assembly(0, false).unwrap();
  ws.assembled_potential()
}

/// a flat shell at height h: the gap h on the contact area, from either
/// side; as slave along its normal (down), as master whatever its
/// orientation
#[test]
fn shell_gaps() {
  let h = 0.05;
  let b = block();
  let area = 0.36;
  for flip in [true, false] {
    let s = shell(|_, _| h, flip);
    // shell slave: only with its normal down (flip)
    let mut ws = setup(&s, &b, true, false);
    let mim = MeshIm::new(&s).with_integration_method(4).unwrap();
    let a = potential(&mut ws, &mim, "Interpolate_filter(ray, 1, 1)", SHELL);
    let g = potential(
      &mut ws,
      &mim,
      "Interpolate_filter(ray, (Interpolate(X,ray)-X).Normal, 1)",
      SHELL,
    );
    if flip {
      assert!((a - area).abs() < 1e-12, "{a}");
      assert!((g - h * area).abs() < 1e-12, "{g}");
    } else {
      assert_eq!(a, 0.0);
    }
    // shell master, two-sided: the block's top below the shell
    let mut ws = setup(&s, &b, false, false);
    let mim = MeshIm::new(&b).with_integration_method(4).unwrap();
    let a = potential(&mut ws, &mim, "Interpolate_filter(ray, 1, 1)", TOP);
    let g = potential(
      &mut ws,
      &mim,
      "Interpolate_filter(ray, (Interpolate(X,ray)-X).Normal, 1)",
      TOP,
    );
    assert!((a - area).abs() < 1e-12, "{flip}: {a}");
    assert!((g - h * area).abs() < 1e-12, "{flip}: {g}");
  }
}

/// worst |finite difference - derivative| and the scale, all entries of
/// the gradient (potential) or tangent (residual) w.r.t. us and ub
fn fd_check(ws: &mut Workspace, mim: &MeshIm, e: &str, rg: i32, potential: bool) -> (f64, f64) {
  let h = 1e-7;
  let names = ["us", "ub"];
  let state: Vec<f64> = names
    .iter()
    .flat_map(|x| ws.variable(x).unwrap().to_vec())
    .collect();
  let set = |ws: &mut Workspace, x: &[f64]| {
    for name in names {
      let r = ws.interval_of_variable(name).unwrap();
      ws.set_variable(name, &x[r]).unwrap();
    }
  };
  let asm = |ws: &mut Workspace, order: usize| {
    ws.clear_expressions();
    ws.add_expression(e, mim, rg).unwrap();
    ws.assembly(order, false).unwrap();
  };
  let n = state.len();
  let k = (!potential).then(|| {
    asm(ws, 2);
    ws.assembled_matrix().to_csr()
  });
  asm(ws, 1);
  let g = ws.assembled_vector().to_vec();
  let (mut worst, mut scale) = (0.0f64, 0.0f64);
  for j in 0..n {
    let (mut xp, mut xm) = (state.clone(), state.clone());
    xp[j] += h;
    xm[j] -= h;
    if potential {
      set(ws, &xp);
      asm(ws, 0);
      let fp = ws.assembled_potential();
      set(ws, &xm);
      asm(ws, 0);
      let fm = ws.assembled_potential();
      scale = scale.max(g[j].abs());
      worst = worst.max(((fp - fm) / (2.0 * h) - g[j]).abs());
    } else {
      let k = k.as_ref().unwrap();
      set(ws, &xp);
      asm(ws, 1);
      let rp = ws.assembled_vector().to_vec();
      set(ws, &xm);
      asm(ws, 1);
      let rm = ws.assembled_vector().to_vec();
      for i in 0..n {
        scale = scale.max(k.get(i, j).abs());
        worst = worst.max(((rp[i] - rm[i]) / (2.0 * h) - k.get(i, j)).abs());
      }
    }
  }
  set(ws, &state);
  (worst, scale)
}

/// a curved deformed shell over the deformed block: the gap's gradient and
/// the contact term's tangent (along the deformed slave normal) by finite
/// differences, the shell as slave and as master
#[test]
fn shell_contact_derivatives_by_finite_differences() {
  let b = block();
  let s = shell(
    |x, y| 0.04 + 0.03 * (x - 0.5) * (y - 0.3) + 0.02 * x * x,
    true,
  );
  let mut bad = vec![];
  for shell_slave in [true, false] {
    let (us, ub, m, rg) = if shell_slave {
      ("us", "ub", &s, SHELL)
    } else {
      ("ub", "us", &b, TOP)
    };
    let mut ws = setup(&s, &b, shell_slave, true);
    let mim = MeshIm::new(m).with_integration_method(4).unwrap();
    let a = potential(&mut ws, &mim, "Interpolate_filter(ray, 1, 1)", rg);
    assert!(a > 0.3, "{shell_slave}: contact area {a}");
    let nx = format!("Transformed_unit_vector(Grad_{us}, Normal)");
    let gap = format!("((Interpolate(X,ray)+Interpolate({ub},ray)-X-{us}).{nx})");
    for (e, pot) in [
      (format!("Interpolate_filter(ray, sqr({gap}), 1)"), true),
      (
        format!(
          "Interpolate_filter(ray, {gap}*({nx}.Test_{us}-{nx}.Interpolate(Test_{ub},ray)), 1)"
        ),
        false,
      ),
    ] {
      let (w, sc) = fd_check(&mut ws, &mim, &e, rg, pot);
      println!("{e}: {w:.1e} (scale {sc:.1e})");
      if w > 1e-6 * sc {
        bad.push(format!("{e}: {w:e} (scale {sc:e})"));
      }
    }
  }
  assert!(bad.is_empty(), "{}", bad.join("\n"));
}
