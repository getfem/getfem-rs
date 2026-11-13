//! Spaces of global functions (M12 step 1b): the element context.

use getfem_fem::global_function::GlobalFunction;
use getfem_fem::interpolation::interpolation;
use getfem_fem::mesh_fem::MeshFem;
use getfem_fem::mesh_im::MeshIm;
use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::regular_mesh::regular_mesh;
use getfem_gwfl::workspace::Workspace;

/// The element and reference point reach the functions (EvalPoint), in
/// assembly and in interpolation: f = cv + 10 xref on 4 elements of [0, 1]
#[test]
fn element_context() {
  use getfem_fem::global_function::{EvalPoint, GlobalFunctionImpl};
  struct OfElement;
  impl GlobalFunctionImpl for OfElement {
    fn dim(&self) -> usize {
      1
    }
    fn val(&self, p: &EvalPoint) -> f64 {
      let (cv, c) = p.element.expect("an element point");
      cv as f64 + 10.0 * c.xref[0]
    }
    fn grad(&self, _: &EvalPoint) -> Vec<f64> {
      vec![0.0]
    }
    fn hess(&self, _: &EvalPoint) -> Vec<f64> {
      vec![0.0]
    }
  }
  let m = regular_mesh(
    &geotrans_descriptor("GT_PK(1,1)").unwrap(),
    &[4],
    &[0.0],
    &[1.0],
  )
  .unwrap();
  let mim = MeshIm::new(&m).with_integration_method(2).unwrap();
  let mf = MeshFem::from_global_functions(&m, &[GlobalFunction::new(OfElement)], None).unwrap();
  let mut ws = Workspace::new();
  ws.add_fem_variable("u", &mf).unwrap();
  // sum of h (cv + 5)
  let v = {
    ws.clear_expressions();
    ws.add_expression("Test_u", &mim, None).unwrap();
    ws.assembly(1, false).unwrap();
    ws.assembled_vector()
  };
  assert!((v[0] - 6.5).abs() < 1e-14, "{v:?}");
  let p0 = MeshFem::new(&m, 1)
    .with_classical_discontinuous_finite_element(0)
    .unwrap();
  let w = interpolation(&mf, &p0, &[1.0]).unwrap();
  assert_eq!(w, vec![5.0, 6.0, 7.0, 8.0]);
}

/// The values at the integration points are computed once (GetFEM
/// precomps) and again after the mesh or the context of the functions
/// changed: f = a x, a a parameter of the functions with a version
#[test]
fn precomputed_values_follow_changes() {
  use getfem_fem::global_function::{EvalPoint, GlobalFunctionImpl};
  use std::sync::atomic::{AtomicU64, Ordering};
  use std::sync::{Arc, Mutex};
  struct Linear {
    a: Arc<Mutex<f64>>,
    version: Arc<AtomicU64>,
    calls: Arc<AtomicU64>,
  }
  impl GlobalFunctionImpl for Linear {
    fn dim(&self) -> usize {
      1
    }
    fn val(&self, p: &EvalPoint) -> f64 {
      self.calls.fetch_add(1, Ordering::Relaxed);
      *self.a.lock().unwrap() * p.x[0]
    }
    fn grad(&self, _: &EvalPoint) -> Vec<f64> {
      vec![*self.a.lock().unwrap()]
    }
    fn hess(&self, _: &EvalPoint) -> Vec<f64> {
      vec![0.0]
    }
    fn version(&self) -> u64 {
      self.version.load(Ordering::Relaxed)
    }
  }
  let (a, version, calls) = (
    Arc::new(Mutex::new(1.0)),
    Arc::new(AtomicU64::new(0)),
    Arc::new(AtomicU64::new(0)),
  );
  let f = Linear {
    a: a.clone(),
    version: version.clone(),
    calls: calls.clone(),
  };
  let m = regular_mesh(
    &geotrans_descriptor("GT_PK(1,1)").unwrap(),
    &[4],
    &[0.0],
    &[1.0],
  )
  .unwrap();
  let mim = MeshIm::new(&m).with_integration_method(2).unwrap();
  let mf = MeshFem::from_global_functions(&m, &[GlobalFunction::new(f)], None).unwrap();
  let mut ws = Workspace::new();
  ws.add_fem_variable("u", &mf).unwrap();
  ws.add_expression("Test_u", &mim, None).unwrap();
  let integral = |ws: &mut Workspace| {
    ws.assembly(1, false).unwrap();
    ws.assembled_vector()[0]
  };
  // int_0^1 x = 1/2, the 8 points evaluated once
  assert!((integral(&mut ws) - 0.5).abs() < 1e-14);
  let n = calls.load(Ordering::Relaxed);
  assert_eq!(n, 8);
  assert!((integral(&mut ws) - 0.5).abs() < 1e-14);
  assert_eq!(calls.load(Ordering::Relaxed), n);
  // a = 3 with a new version
  *a.lock().unwrap() = 3.0;
  version.store(1, Ordering::Relaxed);
  assert!((integral(&mut ws) - 1.5).abs() < 1e-14);
  assert_eq!(calls.load(Ordering::Relaxed), 2 * n);
  // the mesh on [0, 2]: int_0^2 3 x = 6
  m.modify(|d| d.transform(&getfem_linalg::tensor::Mat::from_rows(&[&[2.0]])))
    .unwrap();
  ws.set_variable("u", &[0.0][..]).unwrap();
  assert!((integral(&mut ws) - 6.0).abs() < 1e-14);
}
