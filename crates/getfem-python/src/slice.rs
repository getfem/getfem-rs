//! gf.Slice (GetFEM gf_slice, gf_slice_get)

use getfem::fem::slice::{
  Orient, SliceField, SlicerAction, SlicerApplyDeformation, SlicerBoundary, SlicerComplementary,
  SlicerExplode, SlicerIntersect, SlicerNone, SlicerUnion, SlicerVolume, StoredMeshSlice,
};
use getfem::geom::mesh::{Mesh as RsMesh, MeshData};
use getfem::geom::region::outer_faces_of_mesh;
use getfem::io::vtk::{Format, VtuExport};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::sync::Arc;

use crate::error::ToPy;
use crate::level_set::MeshLevelSet;
use crate::mesh::{CvFids, Mesh, region_set};
use crate::mesh_fem::MeshFem;

#[pyclass(module = "getfem_rs", frozen)]
pub struct Slice(pub(crate) StoredMeshSlice, pub(crate) RsMesh);

fn orient(o: i64) -> PyResult<Orient> {
  Ok(match o {
    -1 => Orient::In,
    0 => Orient::Bound,
    1 => Orient::Out,
    2 => Orient::Split,
    _ => {
      return Err(PyValueError::new_err(
        "orient: -1 (in), 0 (boundary), 1 (out) or 2 (split)",
      ));
    }
  })
}

/// the slicer of op, a tuple (name, args...) (GetFEM build_slicers)
fn slicer(mesh: &RsMesh, m: &MeshData, op: &Bound<'_, PyAny>) -> PyResult<Box<dyn SlicerAction>> {
  let op = op.cast::<PyTuple>()?;
  let name: String = op.get_item(0)?.extract()?;
  let a = op.get_slice(1, op.len());
  let n = a.len();
  let nargs = |lo: usize, hi: usize| {
    if n < lo || n > hi {
      return Err(PyValueError::new_err(format!(
        "slice op {name}: {n} arguments"
      )));
    }
    Ok(())
  };
  let v = |k: usize| a.get_item(k)?.extract::<Vec<f64>>();
  let f = |k: usize| a.get_item(k)?.extract::<f64>();
  let o = |k: usize| orient(a.get_item(k)?.extract()?);
  let s: Box<dyn SlicerAction> = match name.as_str() {
    "none" => {
      nargs(0, 0)?;
      Box::new(SlicerNone)
    }
    "planar" => {
      nargs(3, 3)?;
      Box::new(SlicerVolume::half_space(&v(1)?, &v(2)?, o(0)?))
    }
    "ball" => {
      nargs(3, 3)?;
      Box::new(SlicerVolume::sphere(&v(1)?, f(2)?, o(0)?))
    }
    "cylinder" => {
      nargs(4, 4)?;
      Box::new(SlicerVolume::cylinder(&v(1)?, &v(2)?, f(3)?, o(0)?))
    }
    "isovalues" => {
      nargs(4, 4)?;
      let mf = a.get_item(1)?;
      let fld = SliceField::new(&mf.cast::<MeshFem>()?.get().0, &v(2)?).py()?;
      Box::new(SlicerVolume::isovalues(fld, f(3)?, o(0)?).py()?)
    }
    "boundary" => {
      nargs(0, 1)?;
      let s = match n {
        0 => Box::new(SlicerNone),
        _ => slicer(mesh, m, &a.get_item(0)?)?,
      };
      let faces = outer_faces_of_mesh(mesh).resolve(m).py()?;
      Box::new(SlicerBoundary::new(m, s, &faces))
    }
    "explode" => {
      nargs(1, 1)?;
      Box::new(SlicerExplode { coef: f(0)? })
    }
    "union" | "intersection" => {
      nargs(1, usize::MAX)?;
      let mut s = slicer(mesh, m, &a.get_item(0)?)?;
      for k in 1..n {
        let s2 = slicer(mesh, m, &a.get_item(k)?)?;
        s = match name.as_str() {
          "union" => Box::new(SlicerUnion::new(s, s2)),
          _ => Box::new(SlicerIntersect::new(s, s2)),
        };
      }
      s
    }
    "diff" => {
      nargs(2, 2)?;
      let (s1, s2) = (
        slicer(mesh, m, &a.get_item(0)?)?,
        slicer(mesh, m, &a.get_item(1)?)?,
      );
      Box::new(SlicerIntersect::new(s1, SlicerComplementary::new(s2)))
    }
    "comp" => {
      nargs(1, 1)?;
      Box::new(SlicerComplementary::new(slicer(mesh, m, &a.get_item(0)?)?))
    }
    k => {
      return Err(PyValueError::new_err(format!(
        "slice op: none, planar, ball, cylinder, isovalues, boundary, explode, union, \
         intersection, diff or comp (not {k})"
      )));
    }
  };
  Ok(s)
}

/// a region id or convexes (and faces)
#[derive(FromPyObject)]
enum SliceRegion {
  Id(usize),
  Cvs(CvFids),
}

impl Slice {
  /// u (on mf) at the points of the slice
  pub fn interpolate(&self, mf: &getfem::fem::mesh_fem::MeshFem, u: &[f64]) -> PyResult<Vec<f64>> {
    let m = mf.linked_mesh().snapshot();
    self.0.interpolate(&m, &SliceField::new(mf, u).py()?).py()
  }
}

#[pymethods]
impl Slice {
  /// gf.Slice(op, m | mf[, U] | mls, nrefine[, region]): the slice by op
  /// of the convexes of region (an id or convexes [and faces]; all if
  /// none) of the mesh, each refined nrefine times; with U on mf, the
  /// mesh deformed by U first; on mls, the cut convexes along the level
  /// sets. op a tuple: ("none",), ("planar", orient, p, n), ("ball",
  /// orient, c, r), ("cylinder", orient, p1, p2, r), ("isovalues", orient,
  /// mf, U, s), ("boundary"[, op]), ("explode", coef), ("union", op1,
  /// ...), ("intersection", op1, ...), ("diff", op1, op2), ("comp", op);
  /// orient -1 inside, 0 the boundary, 1 outside, 2 both
  #[new]
  #[pyo3(signature = (op, source, *args))]
  fn new(
    op: &Bound<'_, PyAny>,
    source: &Bound<'_, PyAny>,
    args: &Bound<'_, PyTuple>,
  ) -> PyResult<Self> {
    let mut args: Vec<Bound<'_, PyAny>> = args.iter().collect();
    let (mesh, mls, def) = if let Ok(mf) = source.cast::<MeshFem>() {
      let mf = &mf.get().0;
      let def = match args.len() >= 2 {
        true => Some(SliceField::new(mf, &args.remove(0).extract::<Vec<f64>>()?).py()?),
        false => None,
      };
      (mf.linked_mesh().clone(), None, def)
    } else if let Ok(mls) = source.cast::<MeshLevelSet>() {
      (
        mls.borrow().0.linked_mesh().clone(),
        Some(mls.borrow()),
        None,
      )
    } else {
      (source.cast::<Mesh>()?.get().0.clone(), None, None)
    };
    let usage = || PyValueError::new_err("gf.Slice(op, m | mf[, U] | mls, nrefine[, region])");
    if args.is_empty() || args.len() > 2 {
      return Err(usage());
    }
    let nrefine: usize = args[0].extract()?;
    let m = mesh.snapshot();
    let region = match args.get(1) {
      None => None,
      Some(r) => Some(match r.extract::<SliceRegion>()? {
        SliceRegion::Id(id) => mesh.region(id).py()?.resolve(&m).py()?,
        SliceRegion::Cvs(c) => Arc::new(region_set(&c)?),
      }),
    };
    let mut s = slicer(&mesh, &m, op)?;
    let mut d = def.map(|f| SlicerApplyDeformation { f });
    let mut actions: Vec<&mut dyn SlicerAction> = vec![];
    if let Some(d) = d.as_mut() {
      actions.push(d);
    }
    actions.push(&mut s);
    let sl = match &mls {
      Some(l) => {
        StoredMeshSlice::build_on_level_set(&l.0, &mut actions, nrefine, region.as_deref())
      }
      None => StoredMeshSlice::build(&m, &mut actions, nrefine, region.as_deref()),
    }
    .py()?;
    Ok(Slice(sl, mesh))
  }

  /// the dimension of the points
  fn dim(&self) -> usize {
    self.0.dim()
  }

  fn nbpts(&self) -> usize {
    self.0.nb_points()
  }

  /// the points (dim x nbpts), the nodes of each convex in turn
  fn pts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let d = self.0.dim();
    let mut x = vec![0.0; d * self.0.nb_points()];
    let mut k = 0;
    for c in &self.0.cvlst {
      for n in &c.nodes {
        for (i, v) in n.pt.iter().enumerate() {
          x[i * self.0.nb_points() + k] = *v;
        }
        k += 1;
      }
    }
    PyArray1::from_vec(py, x).reshape([d, self.0.nb_points()])
  }

  /// the numbers of simplexes per dimension (0 to dim), or of dimension d
  #[pyo3(signature = (d = None))]
  fn nbsplxs(&self, py: Python<'_>, d: Option<usize>) -> PyResult<Py<PyAny>> {
    let v: Vec<usize> = (0..=self.0.dim()).map(|k| self.0.nb_simplexes(k)).collect();
    Ok(match d {
      Some(d) => v
        .get(d)
        .copied()
        .unwrap_or(0)
        .into_pyobject(py)?
        .into_any()
        .unbind(),
      None => v.into_pyobject(py)?.into_any().unbind(),
    })
  }

  /// the simplexes of dimension d ((d + 1) x n point numbers) and, per
  /// convex of the slice, the first of its simplexes (and their count at
  /// the end)
  #[allow(clippy::type_complexity)]
  fn splxs<'py>(
    &self,
    py: Python<'py>,
    d: usize,
  ) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray1<i64>>)> {
    let ns = self.0.nb_simplexes(d);
    let mut s = vec![0i64; (d + 1) * ns];
    let mut cv2s = vec![0i64];
    let (mut cnt, mut pcnt) = (0, 0);
    for c in &self.0.cvlst {
      for x in c.simplexes.iter().filter(|x| x.dim() == d) {
        for (k, &i) in x.inodes.iter().enumerate() {
          s[k * ns + cnt] = (i + pcnt) as i64;
        }
        cnt += 1;
      }
      pcnt += c.nodes.len();
      cv2s.push(cnt as i64);
    }
    Ok((
      PyArray1::from_vec(py, s).reshape([d + 1, ns])?,
      PyArray1::from_vec(py, cv2s),
    ))
  }

  /// the convexes of the mesh in the slice
  fn cvs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
    PyArray1::from_vec(py, self.0.cvlst.iter().map(|c| c.cv_num as i64).collect())
  }

  fn linked_mesh(&self) -> Mesh {
    Mesh(self.1.clone())
  }

  /// VTU file of the slice with the fields mf, U, name (interpolated onto
  /// it) or U, name (U at the slice's points, q x nbpts or flat), ...;
  /// binary unless ascii
  #[pyo3(signature = (filename, *fields, ascii = false))]
  fn export_to_vtu(
    &self,
    filename: &str,
    fields: &Bound<'_, PyTuple>,
    ascii: bool,
  ) -> PyResult<()> {
    let format = if ascii { Format::Ascii } else { Format::Binary };
    let mut vtu = VtuExport::new(filename, format);
    vtu.exporting(&self.0).py()?;
    let err = || PyValueError::new_err("export_to_vtu: fields as mf, U, name or U, name, ...");
    let mut k = 0;
    while k < fields.len() {
      let first = fields.get_item(k)?;
      if let Ok(mf) = first.cast::<MeshFem>() {
        if k + 2 >= fields.len() {
          return Err(err());
        }
        let u: Vec<f64> = fields.get_item(k + 1)?.extract()?;
        let name: String = fields.get_item(k + 2)?.extract()?;
        vtu.write_point_data(&mf.get().0, &u, &name).py()?;
        k += 3;
      } else {
        if k + 1 >= fields.len() {
          return Err(err());
        }
        // q x nbpts (GetFEM's layout) to q values per point
        let a: PyReadonlyArrayDyn<'_, f64> = first.extract().map_err(|_| err())?;
        let a = a.as_array();
        let u: Vec<f64> = match a.ndim() {
          2 => a.t().iter().copied().collect(),
          _ => a.iter().copied().collect(),
        };
        let name: String = fields.get_item(k + 1)?.extract()?;
        vtu.write_slice_point_data(&u, &name).py()?;
        k += 2;
      }
    }
    vtu.write().py()
  }
}
