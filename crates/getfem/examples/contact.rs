//! Contact between two elastic bodies, written in GWFL: a punch with a
//! curved bottom pressed into a block. The two meshes are coupled by a
//! raytracing transformation from the bottom of the punch (slave) to the
//! top of the block (master); the gap along the deformed normal n1 of the
//! punch (Poulios, Renard, Comput. Struct. 153, 2015) is
//!   g = (Interpolate(X,ray) + Interpolate(u2,ray) - X - u1).n1.
//! Two formulations:
//! - penalty: force r*neg_part(g);
//! - Lagrange multiplier p (contact pressure) with the augmented condition
//!   p = pos_part(p - r*g), solved by semismooth Newton.
//!
//! The punch is pushed down in steps; the deformed bodies are written as
//! compressed .vtu slices with .pvd collections (target/showcase/contact/).

use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im::MeshIm;
use getfem::fem::slice::{SliceField, SlicerApplyDeformation, StoredMeshSlice};
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::mesh::Mesh;
use getfem::geom::region::{outer_faces_of_mesh, select_faces_of_normal};
use getfem::geom::regular_mesh::regular_mesh;
use getfem::io::vtk::{Format, VtuExport, export_to_pvd};
use getfem::model::{Model, Newton};

// regions of both meshes
const CONTACT: i32 = 1;
const OUTER: i32 = 2;
// load steps and the final downward displacement of the punch top
const STEPS: usize = 5;
const PUSH: f64 = 0.05;

#[derive(Clone, Copy, PartialEq)]
enum Method {
  Penalty,
  Multiplier,
}

/// a mesh of [x0,x0+lx]x[y0,y0+ly] with regions CONTACT and OUTER on the
/// faces of normal (0, contact) and (0, -contact)
fn block(x0: f64, y0: f64, lx: f64, ly: f64, nx: usize, ny: usize, contact: f64) -> Mesh {
  let m = regular_mesh(
    &geotrans_descriptor("GT_PK(2,1)").unwrap(),
    &[nx, ny],
    &[x0, y0],
    &[lx, ly],
  )
  .unwrap();
  let b = outer_faces_of_mesh(&m);
  m.set_region(
    CONTACT as usize,
    &select_faces_of_normal(&m, &b, &[0.0, contact], 0.01).unwrap(),
  )
  .unwrap();
  m.set_region(
    OUTER as usize,
    &select_faces_of_normal(&m, &b, &[0.0, -contact], 0.01).unwrap(),
  )
  .unwrap();
  m
}

struct Bodies {
  punch: Mesh,
  base: Mesh,
  mf1: MeshFem,
  mf2: MeshFem,
}

fn bodies() -> Bodies {
  // punch [0.6,1.4]x[0,0.5], bottom bent to y = 0.01 + 0.4 (x - 1)^2
  let punch = block(0.6, 0.0, 0.8, 0.5, 16, 8, -1.0);
  for i in punch.point_index() {
    let p = punch.point(i).to_vec();
    let b = 0.01 + 0.4 * (p[0] - 1.0).powi(2);
    punch
      .set_point(i, &[p[0], b + p[1] * (0.5 - b) / 0.5])
      .unwrap();
  }
  let base = block(0.0, -0.5, 2.0, 0.5, 32, 8, 1.0);
  let mf1 = MeshFem::new(&punch, 2)
    .with_classical_finite_element(2)
    .unwrap();
  let mf2 = MeshFem::new(&base, 2)
    .with_classical_finite_element(2)
    .unwrap();
  Bodies {
    punch,
    base,
    mf1,
    mf2,
  }
}

/// per step: Newton iterations, total contact force, mean penetration
fn run(b: &Bodies, method: Method, out: Option<&str>) -> Vec<(usize, f64, f64)> {
  let mim1 = MeshIm::new(&b.punch).with_integration_method(4).unwrap();
  let mim2 = MeshIm::new(&b.base).with_integration_method(4).unwrap();
  let mut md = Model::new();
  md.add_fem_variable("u1", &b.mf1).unwrap();
  md.add_fem_variable("u2", &b.mf2).unwrap();
  md.add_scalar_data("lambda", 1.0).unwrap();
  md.add_scalar_data("mu", 1.0).unwrap();
  let top = vec![0.0; b.mf1.nb_dof().unwrap()];
  md.add_fem_data("top", &b.mf1, &top).unwrap();
  md.add_raytracing_transformation("ray", 0.1).unwrap();
  md.add_slave_contact_boundary_to_raytracing_transformation("ray", &b.punch, "u1", CONTACT)
    .unwrap();
  md.add_master_contact_boundary_to_raytracing_transformation("ray", &b.base, "u2", CONTACT)
    .unwrap();
  // the deformed normal of the slave (Poulios, Renard 2015)
  md.add_macro("n1", "Transformed_unit_vector(Grad_u1, Normal)")
    .unwrap();
  md.add_macro("g", "((Interpolate(X,ray)+Interpolate(u2,ray)-X-u1).n1)")
    .unwrap();
  md.add_macro("dg", "(n1.Test_u1 - n1.Interpolate(Test_u2,ray))")
    .unwrap();
  for (mim, u) in [(&mim1, "u1"), (&mim2, "u2")] {
    md.add_linear_term(
      mim,
      None,
      &format!("lambda*Div_{u}*Div_Test_{u} + mu*(Grad_{u}+Grad_{u}'):Grad_Test_{u}"),
    )
    .unwrap();
  }
  match method {
    Method::Penalty => {
      md.add_scalar_data("r", 1e3).unwrap();
      md.add_term(
        &mim1,
        CONTACT,
        "Interpolate_filter(ray, r*neg_part(g)*dg, 1)",
      )
      .unwrap();
    }
    Method::Multiplier => {
      let mfp = MeshFem::new(&b.punch, 1)
        .with_classical_finite_element(1)
        .unwrap();
      let mfp = MeshFem::partial_on_region(&mfp, CONTACT).unwrap();
      md.add_fem_variable("p", &mfp).unwrap();
      md.add_scalar_data("r", 10.0).unwrap();
      md.add_term(&mim1, CONTACT, "Interpolate_filter(ray, p*dg, 1)")
        .unwrap();
      // p = pos_part(p - r g) where a target is found, p = 0 elsewhere
      md.add_term(
        &mim1,
        CONTACT,
        "p/r*Test_p - Interpolate_filter(ray, pos_part(p - r*g)/r*Test_p, 1)",
      )
      .unwrap();
    }
  }
  md.add_Dirichlet_condition_with_multiplier(&mim1, "u1", &b.mf1, OUTER, Some("top"))
    .unwrap();
  md.add_Dirichlet_condition_with_multiplier(&mim2, "u2", &b.mf2, OUTER, None)
    .unwrap();
  let force = match method {
    Method::Penalty => "Interpolate_filter(ray, r*neg_part(g), 1)",
    Method::Multiplier => "p",
  };
  let mut rows = vec![];
  let (mut pvd1, mut pvd2) = (vec![], vec![]);
  for k in 1..=STEPS {
    let d = PUSH * k as f64 / STEPS as f64;
    let q = b.mf1.qdim();
    let top: Vec<f64> = (0..b.mf1.nb_dof().unwrap())
      .map(|i| if i % q == 1 { -d } else { 0.0 })
      .collect();
    md.set_variable("top", &top).unwrap();
    let rep = Newton::default().solve(&mut md, None).unwrap();
    assert!(
      rep.converged(),
      "step {k}: {:?} {}",
      rep.status,
      rep.message
    );
    let f = {
      let ws = md.workspace_mut();
      ws.clear_expressions();
      ws.add_expression(force, &mim1, CONTACT).unwrap();
      ws.assembly(0, false).unwrap();
      ws.assembled_potential()
    };
    // mean penetration where the bodies overlap
    let pen = {
      let ws = md.workspace_mut();
      ws.clear_expressions();
      ws.add_expression("Interpolate_filter(ray, neg_part(g), 1)", &mim1, CONTACT)
        .unwrap();
      ws.assembly(0, false).unwrap();
      ws.assembled_potential()
    };
    let area = {
      let ws = md.workspace_mut();
      ws.clear_expressions();
      ws.add_expression("Interpolate_filter(ray, Heaviside(-g), 1)", &mim1, CONTACT)
        .unwrap();
      ws.assembly(0, false).unwrap();
      ws.assembled_potential()
    };
    rows.push((rep.iterations, f, if area > 0.0 { pen / area } else { 0.0 }));
    if let Some(dir) = out {
      for (mesh, mf, u, name, pvd) in [
        (&b.punch, &b.mf1, "u1", "punch", &mut pvd1),
        (&b.base, &b.mf2, "u2", "base", &mut pvd2),
      ] {
        let field = SliceField::new(mf, md.variable(u).unwrap()).unwrap();
        let mut def = SlicerApplyDeformation { f: field };
        let m = mesh.snapshot();
        let sl = StoredMeshSlice::build(&m, &mut [&mut def], 2, None).unwrap();
        let file = format!("{name}_{k}.vtu");
        let mut vtu = VtuExport::new(&format!("{dir}/{file}"), Format::Compressed);
        vtu.exporting(&sl).unwrap();
        vtu
          .write_point_data(mf, md.variable(u).unwrap(), "u")
          .unwrap();
        vtu.write().unwrap();
        pvd.push((d, file));
      }
    }
  }
  if let Some(dir) = out {
    for (name, pvd) in [("punch", &pvd1), ("base", &pvd2)] {
      let steps: Vec<(f64, &str)> = pvd.iter().map(|(t, f)| (*t, f.as_str())).collect();
      export_to_pvd(&format!("{dir}/{name}.pvd"), &steps).unwrap();
    }
  }
  rows
}

fn main() {
  let out = format!(
    "{}/../../target/showcase/contact",
    env!("CARGO_MANIFEST_DIR")
  );
  std::fs::create_dir_all(&out).unwrap();
  let b = bodies();
  let pen = run(&b, Method::Penalty, None);
  let mul = run(&b, Method::Multiplier, Some(&out));
  println!(
    "step  push   | penalty: iter  force      mean penetration | multiplier: iter  force      mean penetration"
  );
  for k in 0..STEPS {
    let d = PUSH * (k + 1) as f64 / STEPS as f64;
    let (a, c) = (pen[k], mul[k]);
    println!(
      "{:4}  {d:.3}  | {:13} {:.4e} {:.3e}   | {:16} {:.4e} {:.3e}",
      k + 1,
      a.0,
      a.1,
      a.2,
      c.0,
      c.1,
      c.2
    );
  }
  println!("written: {out}/punch.pvd, {out}/base.pvd");
}
