//! gf.MeshIm (GetFEM gf_mesh_im, gf_mesh_im_get, gf_mesh_im_set)

use getfem::fem::mesh_im::MeshIm as RsMeshIm;
use getfem::fem::mesh_im_level_set::{CELL_TOLERANCE, IntegrateWhere, MeshImLevelSet};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::sync::Mutex;

use crate::descriptors::Integ;
use crate::error::ToPy;
use crate::level_set::MeshLevelSet;
use crate::mesh::Mesh;

/// the methods, and of a "levelset" one its cut (to set the method of the
/// elements not cut again)
#[pyclass(module = "getfem_rs", frozen)]
pub struct MeshIm(pub RsMeshIm, pub Option<Mutex<MeshImLevelSet>>);

#[pymethods]
impl MeshIm {
  /// gf.MeshIm(m[, im]): im a gf.Integ or a degree;
  /// gf.MeshIm("levelset", mls, where, im[, im_tip[, im_set]], lattice=2,
  /// tau=1e-2): on the part of the elements where ("all", "inside",
  /// "outside", "boundary", followed by a set expression of the level sets
  /// a, b, ... as "(a*b)"), im on the sub-simplices of the cut elements,
  /// im_tip the base of the rules at crack tips (None: im there too), im_set (default: degree 1,
  /// as GetFEM) on the elements not cut; 2D: the lattice of each element
  /// (n triangles per direction), tau the size tolerance of the
  /// sub-simplices' rules (0: im on every sub-simplex, GetFEM). Built from
  /// the current cut of mls (a copy); set_integration_method then sets
  /// the method of the elements not cut (GetFEM set_integ)
  #[new]
  #[pyo3(signature = (first, *args, lattice = 2, tau = CELL_TOLERANCE))]
  fn new(
    first: &Bound<'_, PyAny>,
    args: &Bound<'_, PyTuple>,
    lattice: usize,
    tau: f64,
  ) -> PyResult<Self> {
    if let Ok(m) = first.cast::<Mesh>() {
      let mim = MeshIm(RsMeshIm::new(&m.get().0), None);
      match args.len() {
        0 => {}
        1 => mim.set_integration_method(&args.get_item(0)?, None)?,
        _ => return Err(PyValueError::new_err("gf.MeshIm(m[, im])")),
      }
      return Ok(mim);
    }
    if first.extract::<String>()? != "levelset" || !(3..=5).contains(&args.len()) {
      return Err(PyValueError::new_err(
        "gf.MeshIm(m[, im]) or gf.MeshIm(\"levelset\", mls, where, im[, im_tip[, im_set]])",
      ));
    }
    let mls = args.get_item(0)?;
    let mls = mls.cast::<MeshLevelSet>()?.borrow();
    let w = args.get_item(1)?.extract::<String>()?.to_lowercase();
    let (where_, csg) = [
      ("inside", IntegrateWhere::Inside),
      ("outside", IntegrateWhere::Outside),
      ("boundary", IntegrateWhere::Boundary),
      ("all", IntegrateWhere::All),
    ]
    .into_iter()
    .find_map(|(k, v)| w.strip_prefix(k).map(|c| (v, c.to_string())))
    .ok_or_else(|| {
      PyValueError::new_err("where: \"inside\", \"outside\", \"boundary\" or \"all\"")
    })?;
    let integ = |k: usize| -> PyResult<Option<_>> {
      match args.get_item(k) {
        Ok(i) if !i.is_none() => Ok(Some(i.cast::<Integ>()?.get().0.clone())),
        _ => Ok(None),
      }
    };
    let (im, im_tip, im_set) = (integ(2)?.unwrap(), integ(3)?, integ(4)?);
    let mimls = MeshImLevelSet::new(&mls.0, where_).py()?;
    if !csg.is_empty() {
      mimls.set_level_set_boolean_operations(&csg).py()?;
    }
    mimls.set_lattice(lattice).py()?;
    mimls.set_simplex_im(&im, im_tip.as_ref(), tau).py()?;
    match im_set {
      Some(i) => mimls.set_integration_method(&i),
      None => mimls.set_integration_method(1),
    }
    .py()?;
    mimls.adapt().py()?;
    Ok(MeshIm((*mimls).clone(), Some(Mutex::new(mimls))))
  }

  /// im (a gf.Integ or a degree) on the convexes cvids (all if None); on
  /// a "levelset" one, on the elements not cut (all of them)
  #[pyo3(signature = (im, cvids = None))]
  fn set_integration_method(
    &self,
    im: &Bound<'_, PyAny>,
    cvids: Option<Vec<usize>>,
  ) -> PyResult<()> {
    if let Some(mimls) = &self.1 {
      if cvids.is_some() {
        return Err(PyValueError::new_err(
          "set_integration_method: on a \"levelset\" gf.MeshIm, all the elements not cut",
        ));
      }
      let mimls = mimls.lock().unwrap();
      let r = if let Ok(pim) = im.cast::<Integ>() {
        mimls.set_integration_method(&pim.get().0)
      } else {
        mimls.set_integration_method(im.extract::<usize>()?)
      };
      r.py()?;
      return mimls.adapt().py();
    }
    let r = if let Ok(pim) = im.cast::<Integ>() {
      let pim = &pim.get().0;
      match cvids {
        None => self.0.set_integration_method(pim),
        Some(cvs) => self.0.set_integration_method((cvs, pim)),
      }
    } else if let Ok(k) = im.extract::<usize>() {
      match cvids {
        None => self.0.set_integration_method(k),
        Some(cvs) => self.0.set_integration_method((cvs, k)),
      }
    } else {
      return Err(PyValueError::new_err(
        "set_integration_method: a gf.Integ or a degree",
      ));
    };
    r.py()
  }
}
