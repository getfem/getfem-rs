//! gf.MeshFem (GetFEM gf_mesh_fem, gf_mesh_fem_get, gf_mesh_fem_set)

use std::collections::BTreeSet;

use getfem::fem::fem_global_function::BsplineBoundary;
use getfem::fem::global_function::GlobalFunction as RsGlobalFunction;
use getfem::fem::mesh_fem::MeshFem as RsMeshFem;
use getfem::geom::region::Region;
use getfem::io::exodus::{export_to_exodus, import_point_data};
use getfem::io::vtk::{Format, VtuExport};
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::descriptors::Fem;
use crate::error::ToPy;
use crate::level_set::{Gf, GlobalFunction, LevelSet, MeshLevelSet};
use crate::mesh::{CvFids, Mesh, region_set};
use crate::spmat::{Spmat, from_csr};

#[pyclass(module = "getfem_rs", frozen)]
pub struct MeshFem(pub RsMeshFem);

/// gf.interpolation(mf, U, mf2 | sl): U (on mf) interpolated onto mf2,
/// on the same or another mesh, or onto the points of the slice sl
/// (GetFEM gf_compute "interpolate on", C++ interpolation)
pub fn interpolate_on<'py>(
  py: Python<'py>,
  mf: &MeshFem,
  u: numpy::PyReadonlyArray1<'_, f64>,
  dest: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
  let v = match dest.cast::<crate::slice::Slice>() {
    Ok(sl) => sl.get().interpolate(&mf.0, u.as_slice()?)?,
    Err(_) => {
      let mf2 = dest.cast::<MeshFem>()?;
      getfem::fem::interpolation::interpolation(&mf.0, &mf2.get().0, u.as_slice()?).py()?
    }
  };
  Ok(PyArray1::from_vec(py, v))
}

#[pymethods]
impl MeshFem {
  /// gf.MeshFem(m[, q1[, q2, ...]]): q1 x q2 x ... components (1 if none);
  /// gf.MeshFem("partial", mf, dofs): mf restricted to its DOFs dofs;
  /// gf.MeshFem("bspline_uniform", m, NX[, NY[, NZ]], order[, low...[,
  /// high...]]): uniform B-splines, one interval per element, boundaries
  /// "free" (default), "periodic" or "symmetry" per direction (high as
  /// low if not given);
  /// gf.MeshFem("levelset", mls, mf): mf (qdim 1) enriched by the zones of
  /// the level sets (the Heaviside enrichment);
  /// gf.MeshFem("global function", m[, ls], [f, ...][, q]): the functions
  /// f (gf.GlobalFunction), the tip functions on the crack ls;
  /// gf.MeshFem("product", pu_mf, global_mf[, pu_dofs]): the functions of
  /// the partition of unity pu_mf at its basic DOFs pu_dofs (all if not
  /// given) times the global functions of global_mf (GetFEM: pu_dofs by
  /// set_enriched_dofs afterwards, none until then);
  /// gf.MeshFem("sum", mf1, mf2, ...): the union of their functions
  #[new]
  #[pyo3(signature = (first, *args))]
  fn new(first: &Bound<'_, PyAny>, args: &Bound<'_, PyTuple>) -> PyResult<Self> {
    if let Ok(m) = first.cast::<Mesh>() {
      let qdims: Vec<usize> = args.extract()?;
      let mf = RsMeshFem::new(&m.get().0, 1);
      if !qdims.is_empty() {
        mf.set_qdim(&qdims);
      }
      return Ok(MeshFem(mf));
    }
    match first.extract::<String>()?.as_str() {
      "partial" => {
        let (mf, dofs): (PyRef<'_, MeshFem>, Vec<usize>) = args.extract()?;
        let dofs: BTreeSet<usize> = dofs.into_iter().collect();
        RsMeshFem::partial(&mf.0, &dofs).py().map(MeshFem)
      }
      "bspline_uniform" => bspline_uniform(args),
      "levelset" => {
        let (mls, mf): (PyRef<'_, MeshLevelSet>, PyRef<'_, MeshFem>) = args.extract()?;
        RsMeshFem::from_level_set(&mls.0, &mf.0).py().map(MeshFem)
      }
      "global function" => {
        let m: PyRef<'_, Mesh> = args.get_item(0)?.extract()?;
        let (ls, rest) = match args.get_item(1)?.extract::<PyRef<'_, LevelSet>>() {
          Ok(ls) => (Some(ls), args.get_slice(2, args.len())),
          Err(_) => (None, args.get_slice(1, args.len())),
        };
        let (fs, q): (Vec<PyRef<'_, GlobalFunction>>, Option<usize>) = match rest.len() {
          1 => (rest.get_item(0)?.extract()?, None),
          _ => rest.extract()?,
        };
        let funcs = fs
          .iter()
          .map(|f| match (&f.0, &ls) {
            (Gf::Global(g), _) => Ok(g.clone()),
            (Gf::Xy(f), Some(ls)) => RsGlobalFunction::on_level_set(&ls.0, f).py(),
            (Gf::Xy(_), None) => Err(PyValueError::new_err(
              "gf.MeshFem(\"global function\"): a tip function needs a crack ls",
            )),
          })
          .collect::<PyResult<Vec<_>>>()?;
        let mf = RsMeshFem::from_global_functions(&m.0, &funcs, None).py()?;
        if let Some(q) = q {
          mf.set_qdim(q);
        }
        Ok(MeshFem(mf))
      }
      "product" => {
        let (pu_mf, global_mf, pu_dofs): (
          PyRef<'_, MeshFem>,
          PyRef<'_, MeshFem>,
          Option<Vec<usize>>,
        ) = match args.len() {
          2 => {
            let (a, b) = args.extract()?;
            (a, b, None)
          }
          _ => args.extract()?,
        };
        let pu_dofs: Option<BTreeSet<usize>> = pu_dofs.map(|d| d.into_iter().collect());
        RsMeshFem::from_product(&pu_mf.0, &global_mf.0, pu_dofs.as_ref())
          .py()
          .map(MeshFem)
      }
      "sum" => {
        let mfs: Vec<PyRef<'_, MeshFem>> = args.extract()?;
        let mfs: Vec<&RsMeshFem> = mfs.iter().map(|mf| &mf.0).collect();
        RsMeshFem::from_sum(&mfs).py().map(MeshFem)
      }
      k => Err(PyValueError::new_err(format!(
        "gf.MeshFem: a gf.Mesh, \"partial\", \"bspline_uniform\", \"levelset\", \"global function\", \"product\" or \"sum\" (not \"{k}\")"
      ))),
    }
  }

  /// the reduction R (DOFs from the basic DOFs) and extension E (basic
  /// DOFs from the DOFs) matrices, used from now on (GetFEM's
  /// reduction_matrices); R E = I checked. R "auto": computed from E by
  /// least squares, (E'E)^-1 E', stored when E'E is diagonal, otherwise
  /// not stored (each reduce solves E'E q = E'b; an info line)
  fn set_reduction_matrices(
    &self,
    py: Python<'_>,
    r: &Bound<'_, PyAny>,
    e: PyRef<'_, Spmat>,
  ) -> PyResult<()> {
    let e = e.with(py, |k| k.to_csr());
    let r = match r.extract::<String>() {
      Ok(s) if s == "auto" => None,
      Ok(s) => {
        return Err(PyValueError::new_err(format!(
          "set_reduction_matrices: R a gf.Spmat or \"auto\" (not \"{s}\")"
        )));
      }
      Err(_) => Some(r.cast::<Spmat>()?.get().with(py, |k| k.to_csr())),
    };
    self.0.set_reduction_matrices(r.as_ref(), &e).py()
  }

  /// the reduction matrix R (a gf.Spmat; None if not reduced; an error if
  /// not stored, see set_reduction_matrices "auto")
  fn reduction_matrix(&self) -> PyResult<Option<Spmat>> {
    Ok(
      self
        .0
        .reduction_matrix()
        .py()?
        .map(|r| Spmat::own(from_csr(&r))),
    )
  }

  /// the extension matrix E (a gf.Spmat; None if not reduced)
  fn extension_matrix(&self) -> PyResult<Option<Spmat>> {
    Ok(
      self
        .0
        .extension_matrix()
        .py()?
        .map(|e| Spmat::own(from_csr(&e))),
    )
  }

  /// V on the basic DOFs from V on the DOFs, E V (V itself if not
  /// reduced)
  fn extend_vector<'py>(
    &self,
    py: Python<'py>,
    v: Vec<f64>,
  ) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(PyArray1::from_vec(py, self.0.extend_vector(&v).py()?))
  }

  /// V on the DOFs from V on the basic DOFs, R V (least squares when R is
  /// not stored)
  fn reduce_vector<'py>(
    &self,
    py: Python<'py>,
    v: Vec<f64>,
  ) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(PyArray1::from_vec(py, self.0.reduce_vector(&v).py()?))
  }

  /// pf on the convexes cvids (all if None)
  #[pyo3(signature = (pf, cvids = None))]
  fn set_finite_element(&self, pf: PyRef<'_, Fem>, cvids: Option<Vec<usize>>) -> PyResult<()> {
    match cvids {
      None => self.0.set_finite_element(&pf.0),
      Some(cvs) => self.0.set_finite_element((cvs, &pf.0)),
    }
    .py()
  }

  fn set_classical_finite_element(&self, k: usize) -> PyResult<()> {
    self.0.set_classical_finite_element(k).py()
  }

  /// discontinuous Lagrange of degree k, its nodes shrunk towards the
  /// element's centre by alpha (default 0)
  #[pyo3(signature = (k, alpha = 0.0))]
  fn set_classical_discontinuous_finite_element(&self, k: usize, alpha: f64) -> PyResult<()> {
    self
      .0
      .set_classical_discontinuous_finite_element((k, alpha))
      .py()
  }

  fn nb_dof(&self) -> PyResult<usize> {
    self.0.nb_dof().py()
  }

  /// the DOFs before reduction
  fn nb_basic_dof(&self) -> PyResult<usize> {
    self.0.nb_basic_dof().py()
  }

  /// the nodes of the basic DOFs, one per column (each component's DOF its
  /// node)
  fn basic_dof_nodes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let n = self.0.nb_basic_dof().py()?;
    let d = self.0.linked_mesh().dim();
    let mut v = vec![0.0; d * n];
    for j in 0..n {
      for (i, x) in self.0.point_of_basic_dof(j).py()?.iter().enumerate() {
        v[i * n + j] = *x;
      }
    }
    PyArray1::from_vec(py, v).reshape([d, n])
  }

  /// the basic DOFs on region rg, sorted; rg a region id or convexes or
  /// faces (as m.set_region)
  fn basic_dof_on_region<'py>(
    &self,
    py: Python<'py>,
    rg: &Bound<'_, PyAny>,
  ) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let d = match rg.extract::<i64>() {
      Ok(id) => self.0.basic_dof_on_region(id),
      Err(_) => {
        let r =
          Region::temporary(&self.0.linked_mesh(), region_set(&rg.extract::<CvFids>()?)?).py()?;
        self.0.basic_dof_on_region(&r)
      }
    }
    .py()?;
    Ok(PyArray1::from_vec(
      py,
      d.into_iter().map(|i| i as i64).collect(),
    ))
  }

  /// the basic DOFs whose mass diagonal (the integral of the squared
  /// basis function with mim) exceeds fraction times its median: those
  /// with support in the domain, e.g. for a "partial" space (qdim 1)
  fn dofs_with_support<'py>(
    &self,
    py: Python<'py>,
    mim: PyRef<'_, crate::mesh_im::MeshIm>,
    fraction: f64,
  ) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let d = self.0.dofs_with_support(&mim.0, fraction).py()?;
    Ok(PyArray1::from_vec(
      py,
      d.into_iter().map(|i| i as i64).collect(),
    ))
  }

  /// the mesh of this space (shared, not a copy)
  fn linked_mesh(&self) -> Mesh {
    Mesh(self.0.linked_mesh())
  }

  fn qdim(&self) -> usize {
    self.0.qdim()
  }

  /// q components (q1 x q2 x ...)
  #[pyo3(signature = (*q))]
  fn set_qdim(&self, q: Vec<usize>) {
    self.0.set_qdim(&q);
  }

  /// VTU file of this space with the fields [mf2,] U, name, ... (U on mf2,
  /// this space if none), interpolated onto it; binary unless ascii
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
    let mut it = fields.iter();
    while let Some(a) = it.next() {
      let (mf, u) = match a.cast::<MeshFem>() {
        Ok(mf) => (mf.get().0.clone(), it.next()),
        Err(_) => (self.0.clone(), Some(a)),
      };
      let (Some(u), Some(name)) = (u, it.next()) else {
        return Err(PyValueError::new_err(
          "export_to_vtu: fields as [mf,] U, name, ...",
        ));
      };
      let u: Vec<f64> = u.extract()?;
      let name: String = name.extract()?;
      vtu.write_point_data(&mf, &u, &name).py()?;
    }
    vtu.write().py()
  }

  /// Exodus II file of the fields U, name, ... on this space at time (one
  /// space per file); append: one more time step in the file of the same
  /// mesh, space and fields
  #[pyo3(signature = (filename, *fields, time = 0.0, append = false))]
  fn export_to_exodus(
    &self,
    filename: &str,
    fields: &Bound<'_, PyTuple>,
    time: f64,
    append: bool,
  ) -> PyResult<()> {
    if fields.iter().any(|a| a.cast::<MeshFem>().is_ok()) {
      return Err(PyValueError::new_err(
        "export_to_exodus: fields U, name, ... on this space only (one file \
         per space; not export_to_vtu's mf2, U, name)",
      ));
    }
    let usage = || PyValueError::new_err("export_to_exodus: fields as U, name, ...");
    let mut vals: Vec<(Vec<f64>, String)> = vec![];
    for pair in fields.iter().collect::<Vec<_>>().chunks(2) {
      let [u, name] = pair else { return Err(usage()) };
      let name: String = name.extract().map_err(|_| usage())?;
      vals.push((u.extract()?, name));
    }
    let vals: Vec<(&[f64], &str)> = vals.iter().map(|(u, n)| (&u[..], n.as_str())).collect();
    export_to_exodus(filename, &self.0, &vals, time, append).py()
  }

  /// (U, t): the nodal field name of an Exodus file at time step `step`
  /// (from 0; negative from the end, -1 the last) on this space's DOFs, and
  /// the step's time: each file element matched to the convex of its
  /// vertices, each DOF to its node (Lagrange of the file's degree or
  /// lower, continuous or not)
  #[pyo3(signature = (filename, name, step = -1))]
  fn import_from_exodus<'py>(
    &self,
    py: Python<'py>,
    filename: &str,
    name: &str,
    step: i64,
  ) -> PyResult<(Bound<'py, PyArray1<f64>>, f64)> {
    let (v, t) = import_point_data(filename, &self.0, name, step).py()?;
    Ok((PyArray1::from_vec(py, v), t))
  }
}

fn bspline_uniform(args: &Bound<'_, PyTuple>) -> PyResult<MeshFem> {
  let usage = || {
    PyValueError::new_err(
      "gf.MeshFem(\"bspline_uniform\", m, NX[, NY[, NZ]], order[, low...[, high...]])",
    )
  };
  let m = args.get_item(0)?;
  let m = m.cast::<Mesh>().map_err(|_| usage())?;
  let dim = m.get().0.dim();
  let ints: Vec<usize> = (1..args.len())
    .map_while(|k| args.get_item(k).ok()?.extract::<usize>().ok())
    .collect();
  if ints.len() != dim + 1 {
    return Err(usage());
  }
  let bc = |s: String| match s.as_str() {
    "free" => Ok(BsplineBoundary::Free),
    "periodic" => Ok(BsplineBoundary::Periodic),
    "symmetry" => Ok(BsplineBoundary::Symmetry),
    _ => Err(usage()),
  };
  let bcs: Vec<BsplineBoundary> = (dim + 2..args.len())
    .map(|k| bc(args.get_item(k)?.extract::<String>()?))
    .collect::<PyResult<_>>()?;
  let (low, high) = match bcs.len() {
    0 => (
      vec![BsplineBoundary::Free; dim],
      vec![BsplineBoundary::Free; dim],
    ),
    n if n == dim => (bcs.clone(), bcs),
    n if n == 2 * dim => (bcs[..dim].to_vec(), bcs[dim..].to_vec()),
    _ => return Err(usage()),
  };
  RsMeshFem::uniform_bspline(&m.get().0, &ints[..dim], ints[dim], &low, &high, None)
    .py()
    .map(MeshFem)
}
