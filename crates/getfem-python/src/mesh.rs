//! gf.Mesh (GetFEM gf_mesh, gf_mesh_get, gf_mesh_set)

use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::mesh::{Mesh as RsMesh, extrude};
use getfem::geom::mesh_file::read_mesh_file;
use getfem::geom::region::{
  Region, RegionSet, inner_faces, outer_faces, outer_faces_of_mesh, select_faces_in_box,
  select_faces_of_normal,
};
use getfem::geom::regular_mesh::regular_unit_mesh;
use getfem::io::exodus::{export_mesh_to_exodus, import_from_exodus};
use getfem::io::gmsh::{GmshOptions, import_from_gmsh};
use getfem::io::vtk::{Format, VtuExport};
use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArrayLike2, PyArrayMethods, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::descriptors::GeoTrans;
use crate::error::ToPy;

#[pyclass(module = "getfem_rs", frozen)]
pub struct Mesh(pub RsMesh);

#[pymethods]
impl Mesh {
  /// gf.Mesh("empty", dim); gf.Mesh("cartesian", X[, Y[, Z, ...]]) (GT_LINEAR_QK),
  /// gf.Mesh("cartesian Q1", X, Y[, Z, ...]) (GT_QK(d,1));
  /// gf.Mesh("regular_simplices", X[, Y[, Z, ...]]) (simplices, GetFEM's
  /// split of the cells; not its "degree", "noised");
  /// gf.Mesh("structured" |
  /// "structured_ball" | "structured_ball_shell", param) (GetFEM: "import"
  /// first); gf.Mesh("import", "gmsh" | "exodus", file); gf.Mesh("load", file) (a
  /// GetFEM mesh file); gf.Mesh("prismatic", m, nl[, degree]) (m extruded
  /// over [0, 1] in nl layers); gf.Mesh("clone", m)
  #[new]
  #[pyo3(signature = (kind, *args))]
  fn new(kind: &str, args: &Bound<'_, PyTuple>) -> PyResult<Self> {
    match kind {
      "empty" => {
        let (dim,): (usize,) = args.extract()?;
        Ok(Mesh(RsMesh::new(dim)))
      }
      "cartesian" | "cartesian Q1" => {
        let xs: Vec<Vec<f64>> = args.extract()?;
        cartesian(&xs, kind == "cartesian").map(Mesh)
      }
      "regular_simplices" => {
        let xs: Vec<Vec<f64>> = args.extract()?;
        regular_simplices(&xs).map(Mesh)
      }
      "load" => {
        let (file,): (String,) = args.extract()?;
        Ok(Mesh(read_mesh_file(&file).py()?))
      }
      "import" => {
        let (format, file): (String, String) = args.extract()?;
        match format.as_str() {
          "gmsh" => Ok(Mesh(
            import_from_gmsh(&file, &GmshOptions::default()).py()?.mesh,
          )),
          "exodus" => Ok(Mesh(import_from_exodus(&file).py()?)),
          _ => Err(PyValueError::new_err(format!(
            "gf.Mesh(\"import\"): unknown format \"{format}\" (gmsh, exodus)"
          ))),
        }
      }
      "prismatic" => {
        let (m, nl, degree): (PyRef<'_, Mesh>, usize, Option<usize>) = match args.len() {
          2 => {
            let (m, nl) = args.extract()?;
            (m, nl, None)
          }
          _ => args.extract()?,
        };
        Ok(Mesh(extrude(&m.0, nl, degree.unwrap_or(1)).py()?))
      }
      "clone" => {
        let (m,): (PyRef<'_, Mesh>,) = args.extract()?;
        Ok(Mesh(RsMesh::from((*m.0.snapshot()).clone())))
      }
      "structured" | "structured_ball" | "structured_ball_shell" => {
        let (param,): (String,) = args.extract()?;
        crate::structured::structured(kind, &param).map(Mesh)
      }
      _ => Err(PyValueError::new_err(format!(
        "gf.Mesh: unknown kind \"{kind}\""
      ))),
    }
  }

  fn dim(&self) -> usize {
    self.0.dim()
  }

  fn nb_points(&self) -> usize {
    self.0.nb_points()
  }

  fn nb_convexes(&self) -> usize {
    self.0.nb_convexes()
  }

  /// the points' coordinates, one per column (deleted points: NaN)
  fn pts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let m = self.0.snapshot();
    let (d, n) = (m.dim(), m.nb_points_max());
    let mut v = vec![f64::NAN; d * n];
    for j in m.point_index() {
      for (i, x) in m.point(j).iter().enumerate() {
        v[i * n + j] = *x;
      }
    }
    PyArray1::from_vec(py, v).reshape([d, n])
  }

  /// the points of convex cv, one per column
  fn points_of_convex<'py>(
    &self,
    py: Python<'py>,
    cv: usize,
  ) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let m = self.0.snapshot();
    check_convex(&m, cv)?;
    let g = m.points_of_convex(cv);
    let (d, n) = (g.nrows(), g.ncols());
    // column-major Mat to a row-major array
    let v: Vec<f64> = (0..d)
      .flat_map(|i| (0..n).map(move |j| (i, j)))
      .map(|(i, j)| g[(i, j)])
      .collect();
    PyArray1::from_vec(py, v).reshape([d, n])
  }

  /// the geometric transformation of convex cv
  fn trans_of_convex(&self, cv: usize) -> PyResult<GeoTrans> {
    let m = self.0.snapshot();
    check_convex(&m, cv)?;
    Ok(GeoTrans(m.trans_of_convex(cv).clone()))
  }

  /// points closer than tol (0: the mesh's merge tolerance) merged, among
  /// the points of region (-1: all); the lowest index survives, the others
  /// leave holes; returns the number of points removed
  #[pyo3(signature = (tol = 0.0, region = -1))]
  fn merge_points(&self, tol: f64, region: i64) -> PyResult<usize> {
    self.0.merge_points(tol, region).py()
  }

  /// all points x moved to M x; M (N' x N, an array or nested lists or
  /// tuples) may change the dimension
  fn transform(&self, m: PyArrayLike2<'_, f64, AllowTypeChange>) -> PyResult<()> {
    let a = m.as_array();
    let (r, c) = a.dim();
    // column-major
    let data: Vec<f64> = (0..c)
      .flat_map(|j| (0..r).map(move |i| (i, j)))
      .map(|ij| a[ij])
      .collect();
    let m = getfem::linalg::tensor::Mat::from_data(&[r, c], data).py()?;
    self.0.transform(&m).py()
  }

  /// all points moved by v
  fn translate(&self, v: Vec<f64>) -> PyResult<()> {
    self.0.modify(|m| m.translate(&v)).py()
  }

  /// points and convexes renumbered without gaps
  fn optimize_structure(&self) {
    self.0.modify(|m| m.optimize_structure())
  }

  /// the faces on the boundary of the convexes CVIDs (all if none) of
  /// dimension dim (all if none; GetFEM: the mesh's), not shared by two
  /// of them: convexes (row 0), faces (row 1)
  #[pyo3(signature = (dim = None, cvids = None))]
  fn outer_faces<'py>(
    &self,
    py: Python<'py>,
    dim: Option<usize>,
    cvids: Option<Vec<usize>>,
  ) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let r = convex_set(&self.0, dim, cvids)?;
    faces(py, &self.0, &outer_faces(&self.0, r).py()?)
  }

  /// VTU file of the mesh; binary unless ascii
  #[pyo3(signature = (filename, ascii = false))]
  fn export_to_vtu(&self, filename: &str, ascii: bool) -> PyResult<()> {
    let format = if ascii { Format::Ascii } else { Format::Binary };
    let mut vtu = VtuExport::new(filename, format);
    vtu.exporting(&self.0).py()?;
    vtu.write().py()
  }

  /// Exodus II file of the mesh (no time step; its regions as sets)
  fn export_to_exodus(&self, filename: &str) -> PyResult<()> {
    export_mesh_to_exodus(filename, &self.0).py()
  }

  /// the faces shared by two convexes of CVIDs (all if none), each once:
  /// convexes (row 0), faces (row 1)
  #[pyo3(signature = (cvids = None))]
  fn inner_faces<'py>(
    &self,
    py: Python<'py>,
    cvids: Option<Vec<usize>>,
  ) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let r = convex_set(&self.0, None, cvids)?;
    faces(py, &self.0, &inner_faces(&self.0, r).py()?)
  }

  /// the outer faces with all their nodes in the box [pmin, pmax]
  fn outer_faces_in_box<'py>(
    &self,
    py: Python<'py>,
    pmin: Vec<f64>,
    pmax: Vec<f64>,
  ) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let r = select_faces_in_box(&self.0, outer_faces_of_mesh(&self.0), &pmin, &pmax).py()?;
    faces(py, &self.0, &r)
  }

  /// all points moved to P (one per column, as pts())
  fn set_pts(&self, pts: Vec<Vec<f64>>) -> PyResult<()> {
    let (d, n) = (self.0.dim(), self.0.nb_points_max());
    if pts.len() != d || pts.iter().any(|r| r.len() != n) {
      return Err(PyValueError::new_err(format!(
        "set_pts: {d} rows of {n} coordinates"
      )));
    }
    let moves: Vec<(usize, Vec<f64>)> = self
      .0
      .point_index()
      .into_iter()
      .map(|j| (j, pts.iter().map(|r| r[j]).collect()))
      .collect();
    self.0.set_points(&moves).py()
  }

  /// the outer faces whose unit outward normal makes an angle <= angle
  /// (radians) with v at one geometric node of the face at least (GetFEM:
  /// at the mean of the nodes)
  fn outer_faces_with_direction<'py>(
    &self,
    py: Python<'py>,
    v: Vec<f64>,
    angle: f64,
  ) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let r = select_faces_of_normal(&self.0, outer_faces_of_mesh(&self.0), &v, angle).py()?;
    faces(py, &self.0, &r)
  }

  /// the convexes and faces of the regions rids (an id or a list) as two
  /// rows: convexes, faces (-1 for a whole convex, as GetFEM)
  fn region<'py>(&self, py: Python<'py>, rids: Rids) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let ids = match rids {
      Rids::One(id) => vec![id],
      Rids::Many(ids) => ids,
    };
    let (mut cvs, mut fs) = (vec![], vec![]);
    for id in ids {
      let set = self.0.region(id).py()?.resolve(&self.0.snapshot()).py()?;
      for (cv, f) in set.iter() {
        cvs.push(cv as i64);
        fs.push(f.map_or(-1, |f| f as i64));
      }
    }
    let n = cvs.len();
    PyArray1::from_vec(py, [cvs, fs].concat()).reshape([2, n])
  }

  /// region id: convexes (a vector or one row) or faces (convexes, faces:
  /// two rows; face -1 the convex itself)
  fn set_region(&self, id: usize, cvfids: CvFids) -> PyResult<()> {
    self.0.set_region_set(id, region_set(&cvfids)?).py()
  }

  /// convexes of geometric transformation gt from their points' coordinates
  /// (one point per column, gt's nb_points columns per convex), points
  /// closer than the merge tolerance merged; the new convexes' ids
  fn add_convex<'py>(
    &self,
    py: Python<'py>,
    gt: PyRef<'_, GeoTrans>,
    pts: PyReadonlyArray2<'py, f64>,
  ) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let pts = pts.as_array();
    let (d, n, k) = (pts.nrows(), pts.ncols(), gt.0.nb_points());
    if d != self.0.dim() || n % k != 0 {
      return Err(PyValueError::new_err(format!(
        "add_convex: {} rows, a multiple of {k} columns",
        self.0.dim()
      )));
    }
    let cols: Vec<Vec<f64>> = pts.columns().into_iter().map(|c| c.to_vec()).collect();
    let mut ids = vec![];
    for q in cols.chunks(k) {
      let r: Vec<&[f64]> = q.iter().map(|v| v.as_slice()).collect();
      ids.push(self.0.modify(|m| m.add_convex_by_points(&gt.0, &r)).py()? as i64);
    }
    Ok(PyArray1::from_vec(py, ids))
  }

  /// region id (created if new) extended by convexes or faces, as
  /// set_region
  fn extend_region(&self, id: usize, cvfids: CvFids) -> PyResult<()> {
    let r = region_set(&cvfids)?;
    let r = match self.0.region_set(id) {
      Some(old) => old.merge(&r),
      None => r,
    };
    self.0.set_region_set(id, r).py()
  }

  /// region r1 := r1 minus r2 (a missing region empty)
  fn region_subtract(&self, r1: usize, r2: usize) -> PyResult<()> {
    self.region_op(r1, r2, RegionSet::subtract)
  }

  /// region r1 := r1 and r2 merged
  fn region_merge(&self, r1: usize, r2: usize) -> PyResult<()> {
    self.region_op(r1, r2, RegionSet::merge)
  }

  /// region r1 := r1 intersected with r2 (a whole convex with its faces
  /// keeps the faces)
  fn region_intersect(&self, r1: usize, r2: usize) -> PyResult<()> {
    self.region_op(r1, r2, RegionSet::intersect)
  }
}

impl Mesh {
  fn region_op(
    &self,
    r1: usize,
    r2: usize,
    f: fn(&RegionSet, &RegionSet) -> RegionSet,
  ) -> PyResult<()> {
    let set = |r| self.0.region_set(r).unwrap_or_default();
    self.0.set_region_set(r1, f(&set(r1), &set(r2))).py()
  }
}

/// region ids: one or a list
#[derive(FromPyObject)]
enum Rids {
  One(usize),
  Many(Vec<usize>),
}

/// a region's convexes (a vector) or rows (see region_set)
#[derive(FromPyObject)]
pub enum CvFids {
  Cvs(Vec<usize>),
  Rows(Vec<Vec<i64>>),
}

/// convexes (a vector or one row) or faces (convexes, faces: two rows;
/// face -1 the convex itself, as GetFEM)
pub fn region_set(cvfids: &CvFids) -> PyResult<RegionSet> {
  let err = || {
    PyValueError::new_err(
      "a region: convexes (a vector or one row), or two rows of equal length (convexes, faces; face -1 the convex)",
    )
  };
  let cv = |c: i64| usize::try_from(c).map_err(|_| err());
  let mut r = RegionSet::new();
  match cvfids {
    CvFids::Cvs(cvs) => cvs.iter().for_each(|&c| r.add(c)),
    CvFids::Rows(rows) => match &rows[..] {
      [cvs] => {
        for &c in cvs {
          r.add(cv(c)?);
        }
      }
      [cvs, fs] if cvs.len() == fs.len() => {
        for (&c, &f) in cvs.iter().zip(fs) {
          match f {
            -1 => r.add(cv(c)?),
            _ => r.add_face(cv(c)?, usize::try_from(f).map_err(|_| err())?),
          }
        }
      }
      _ => return Err(err()),
    },
  }
  Ok(r)
}

/// the convexes cvids (all if none) of dimension dim (all if none), a
/// temporary region
fn convex_set(m: &RsMesh, dim: Option<usize>, cvids: Option<Vec<usize>>) -> PyResult<Region> {
  let md = m.snapshot();
  let cvs = cvids.unwrap_or_else(|| md.convex_index().collect());
  let mut set = RegionSet::new();
  for cv in cvs {
    if !md.is_convex_valid(cv) {
      return Err(PyValueError::new_err(format!("no convex {cv}")));
    }
    if dim.is_none_or(|d| md.structure_of_convex(cv).dim() == d) {
      set.add(cv);
    }
  }
  Region::temporary(&md, set).py()
}

/// the faces of r as two rows: convexes, faces
fn faces<'py>(py: Python<'py>, m: &RsMesh, r: &Region) -> PyResult<Bound<'py, PyArray2<i64>>> {
  let set = r.resolve(&m.snapshot()).py()?;
  let (cvs, fs): (Vec<i64>, Vec<i64>) = set
    .iter()
    .filter_map(|(cv, f)| f.map(|f| (cv as i64, f as i64)))
    .unzip();
  let n = cvs.len();
  PyArray1::from_vec(py, [cvs, fs].concat()).reshape([2, n])
}

/// The unit simplex grid, each coordinate mapped piecewise linearly onto
/// xs
fn regular_simplices(xs: &[Vec<f64>]) -> PyResult<RsMesh> {
  if xs.is_empty() || xs.iter().any(|x| x.len() < 2) {
    return Err(PyValueError::new_err(
      "gf.Mesh(\"regular_simplices\"): at least one vector, of at least 2 values each",
    ));
  }
  let n: Vec<usize> = xs.iter().map(|x| x.len() - 1).collect();
  let gt = geotrans_descriptor(&format!("GT_PK({},1)", xs.len())).py()?;
  let m = regular_unit_mesh(&n, &gt).py()?;
  let moves: Vec<(usize, Vec<f64>)> = (0..m.nb_points_max())
    .filter(|&i| m.is_point_valid(i))
    .map(|i| {
      let p = m.point(i);
      // the unit mesh's points: k / n per direction
      let q = p
        .iter()
        .zip(xs.iter().zip(&n))
        .map(|(c, (x, &nk))| {
          let t = c * nk as f64;
          let k = ((t + 1e-6) as usize).min(nk - 1);
          let a = t - k as f64;
          (1.0 - a) * x[k] + a * x[k + 1]
        })
        .collect();
      (i, q)
    })
    .collect();
  m.set_points(&moves).py()?;
  Ok(m)
}

/// The cartesian mesh: the points in Fortran order, GT_LINEAR_QK (linear)
/// or GT_QK(d,1) ("cartesian Q1")
fn cartesian(xs: &[Vec<f64>], linear: bool) -> PyResult<RsMesh> {
  let d = xs.len();
  if d == 0 || xs.iter().any(|x| x.len() < 2) {
    return Err(PyValueError::new_err(
      "gf.Mesh(\"cartesian\"): at least one vector, of at least 2 values each",
    ));
  }
  let m = RsMesh::new(d);
  let np: usize = xs.iter().map(|x| x.len()).product();
  let mut p = vec![0.0; d];
  for i in 0..np {
    let mut k = i;
    for (pj, x) in p.iter_mut().zip(xs) {
      *pj = x[k % x.len()];
      k /= x.len();
    }
    if m.add_point(&p).py()? != i {
      return Err(PyValueError::new_err(
        "gf.Mesh(\"cartesian\"): coincident points (repeated coordinates)",
      ));
    }
  }
  let gt = match linear {
    true => geotrans_descriptor(&format!("GT_LINEAR_QK({d})")),
    false => geotrans_descriptor(&format!("GT_QK({d},1)")),
  }
  .py()?;
  let nc: usize = xs.iter().map(|x| x.len() - 1).product();
  let stride: Vec<usize> = xs
    .iter()
    .scan(1, |s, x| {
      let r = *s;
      *s *= x.len();
      Some(r)
    })
    .collect();
  for c in 0..nc {
    let mut k = c;
    let mut base = 0;
    for (x, s) in xs.iter().zip(&stride) {
      base += (k % (x.len() - 1)) * s;
      k /= x.len() - 1;
    }
    // vertex j: +1 along d where bit d of j is set
    let pts: Vec<usize> = (0..1usize << d)
      .map(|j| {
        base
          + (0..d)
            .filter(|b| j >> b & 1 == 1)
            .map(|b| stride[b])
            .sum::<usize>()
      })
      .collect();
    m.add_convex(&gt, &pts).py()?;
  }
  Ok(m)
}

fn check_convex(m: &getfem::geom::mesh::MeshData, cv: usize) -> PyResult<()> {
  if m.is_convex_valid(cv) {
    Ok(())
  } else {
    Err(PyValueError::new_err(format!("no convex {cv}")))
  }
}
