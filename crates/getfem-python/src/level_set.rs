//! gf.GlobalFunction (the crack tip functions: GetFEM's xy functions),
//! gf.LevelSet, gf.MeshLevelSet (GetFEM gf_global_function, gf_levelset,
//! gf_mesh_levelset)

use getfem::fem::crack_front::{
  CrackFront, HatDegree, crack_fronts, front_hats, select_mesh_fem_basic_dofs_near_front,
};
use getfem::fem::crack_tip_function::{CrackTipFunction, Cutoff, Singular};
use getfem::fem::global_function::GlobalFunction as RsGlobalFunction;
use getfem::fem::level_set::LevelSet as RsLevelSet;
use getfem::fem::mesh_level_set::MeshLevelSet as RsMeshLevelSet;
use getfem::fem::slice::StoredMeshSlice;
use getfem::gwfl::crack_sif::{CrackedBody, LipTractions, Sif, crack_sifs};
use getfem::gwfl::workspace::Workspace;
use getfem::linalg::vect_dist2;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::error::ToPy;
use crate::mesh::Mesh;
use crate::mesh_fem::MeshFem;
use crate::mesh_im::MeshIm;
use crate::slice::Slice;

/// a function of the polar coordinates at a crack tip (GetFEM's xy
/// function), or of the point (GetFEM's global function)
#[derive(Clone)]
pub enum Gf {
  Xy(CrackTipFunction),
  Global(RsGlobalFunction),
}

#[pyclass(module = "getfem_rs", frozen)]
pub struct GlobalFunction(pub Gf);

impl GlobalFunction {
  /// f*g or f+g, both of the same kind
  fn combine(&self, g: &GlobalFunction, product: bool) -> PyResult<Self> {
    Ok(GlobalFunction(match (&self.0, &g.0) {
      (Gf::Xy(f), Gf::Xy(g)) => Gf::Xy(match product {
        true => CrackTipFunction::product(f, g),
        false => CrackTipFunction::sum(&[f.clone(), g.clone()]),
      }),
      (Gf::Global(f), Gf::Global(g)) => Gf::Global(
        match product {
          true => RsGlobalFunction::product(f, g),
          false => RsGlobalFunction::sum(&[f.clone(), g.clone()]),
        }
        .py()?,
      ),
      _ => {
        return Err(PyValueError::new_err(
          "gf.GlobalFunction: a tip function (crack, cutoff, angular_hat) with a function of the \
           point: place it on a crack first (\"levelset\", ls, f)",
        ));
      }
    }))
  }

  /// the tip function
  pub fn xy(&self) -> PyResult<&CrackTipFunction> {
    match &self.0 {
      Gf::Xy(f) => Ok(f),
      Gf::Global(_) => Err(PyValueError::new_err(
        "gf.GlobalFunction: a tip function (crack, cutoff, angular_hat) expected",
      )),
    }
  }
}

/// GetFEM's crack_singular_xy_function index l
fn singular(l: usize) -> PyResult<Singular> {
  use Singular::*;
  Ok(match l {
    0 | 10 | 13 => SqrtRSin,
    1 | 11 => SqrtRCos,
    2 => SqrtRSinSinTheta,
    3 => SqrtRCosSinTheta,
    4 => R32Sin,
    5 => R32Cos,
    6 => R32SinCos2,
    7 => R32Cos3,
    8 => MinusInvSqrtRSin,
    9 => InvSqrtRCos,
    12 => RSin2,
    14 => InvRSin,
    15 => InvRCos,
    _ => return Err(PyValueError::new_err(format!("no crack function {l}"))),
  })
}

#[pymethods]
impl GlobalFunction {
  /// gf.GlobalFunction("crack", l): GetFEM's singular function l (0 to
  /// 15) of the crack tip asymptotics;
  /// gf.GlobalFunction("cutoff", kind, ...): a cutoff in the distance rho
  /// to the tip, kind "none", "exponential" (radius), "polynomial" (inner,
  /// outer: C1), "polynomial2" (inner, outer: C2), "hat" (left, centre,
  /// right) (GetFEM: an index and three parameters);
  /// gf.GlobalFunction("angular_hat", n, m, periodic): the hat m of n
  /// over the angle (-pi, pi), periodic: m 0 wraps around;
  /// these tip functions of the polar coordinates at a crack tip
  /// (gf.MeshFem("global function", m, ls, [f, ...]));
  /// gf.GlobalFunction("levelset", ls, f): the tip function f on the crack
  /// ls, a function of the point;
  /// gf.GlobalFunction("front hat", P, T, closed, k, radius[, degree]):
  /// the hat of point k of a front (P, T, closed) of ls.crack_fronts, zero
  /// farther than radius from it, degree "linear" (default) or "cubic";
  /// gf.GlobalFunction("product", f, g), ("add", f, g); f*g, f+g, both
  /// tip functions or both functions of the point
  #[new]
  #[pyo3(signature = (kind, *args))]
  fn new(kind: &str, args: &Bound<'_, PyTuple>) -> PyResult<Self> {
    let f = match kind {
      "crack" => Gf::Xy(CrackTipFunction::crack(singular(
        args.extract::<(usize,)>()?.0,
      )?)),
      "angular_hat" => {
        let (n, m, periodic): (usize, usize, bool) = args.extract()?;
        Gf::Xy(CrackTipFunction::angular_hat(n, m, periodic).py()?)
      }
      "levelset" => {
        let (ls, f): (PyRef<'_, LevelSet>, PyRef<'_, GlobalFunction>) = args.extract()?;
        Gf::Global(RsGlobalFunction::on_level_set(&ls.0, f.xy()?).py()?)
      }
      "front hat" => {
        let (p, t, closed, k, radius, degree): (
          PyReadonlyArray2<'_, f64>,
          PyReadonlyArray2<'_, f64>,
          bool,
          usize,
          f64,
          Option<String>,
        ) = match args.len() {
          5 => {
            let (p, t, closed, k, radius) = args.extract()?;
            (p, t, closed, k, radius, None)
          }
          _ => args.extract()?,
        };
        let degree = match degree.as_deref() {
          None | Some("linear") => HatDegree::Linear,
          Some("cubic") => HatDegree::Cubic,
          Some(d) => {
            return Err(PyValueError::new_err(format!(
              "front hat: degree linear or cubic (not {d})"
            )));
          }
        };
        let mut hats = front_hats(&front_of(p, Some(t), closed), degree, radius).py()?;
        if k >= hats.len() {
          return Err(PyValueError::new_err(format!(
            "front hat: point {k} of {}",
            hats.len()
          )));
        }
        Gf::Global(hats.swap_remove(k))
      }
      "cutoff" => {
        let name: String = args.get_item(0)?.extract()?;
        let p: Vec<f64> = args.get_slice(1, args.len()).extract()?;
        let n = |k: usize| -> PyResult<()> {
          match p.len() == k {
            true => Ok(()),
            false => Err(PyValueError::new_err(format!(
              "cutoff {name}: {k} parameters, not {}",
              p.len()
            ))),
          }
        };
        let c = match name.as_str() {
          "none" => n(0).map(|_| Cutoff::None),
          "exponential" => n(1).map(|_| Cutoff::Exponential { radius: p[0] }),
          "polynomial" => n(2).map(|_| Cutoff::Polynomial {
            inner: p[0],
            outer: p[1],
          }),
          "polynomial2" => n(2).map(|_| Cutoff::Polynomial2 {
            inner: p[0],
            outer: p[1],
          }),
          "hat" => n(3).map(|_| Cutoff::Hat {
            left: p[0],
            centre: p[1],
            right: p[2],
          }),
          k => Err(PyValueError::new_err(format!(
            "cutoff: none, exponential, polynomial, polynomial2 or hat (not {k})"
          ))),
        }?;
        Gf::Xy(CrackTipFunction::cutoff(c).py()?)
      }
      "product" | "add" => {
        let (f, g): (PyRef<'_, GlobalFunction>, PyRef<'_, GlobalFunction>) = args.extract()?;
        return f.combine(&g, kind == "product");
      }
      k => {
        return Err(PyValueError::new_err(format!(
          "gf.GlobalFunction: crack, cutoff, angular_hat, levelset, front hat, product or add \
           (not {k})"
        )));
      }
    };
    Ok(GlobalFunction(f))
  }

  fn __mul__(&self, g: PyRef<'_, GlobalFunction>) -> PyResult<Self> {
    self.combine(&g, true)
  }

  fn __add__(&self, g: PyRef<'_, GlobalFunction>) -> PyResult<Self> {
    self.combine(&g, false)
  }

  /// the value at the point x: (x, y) in the tip's frame for a tip
  /// function
  #[pyo3(signature = (*x))]
  fn val(&self, x: Vec<f64>) -> PyResult<f64> {
    match &self.0 {
      Gf::Xy(f) => {
        let [a, b] = xy(&x)?;
        Ok(f.val(a, b))
      }
      Gf::Global(f) => Ok(f.val(x.as_slice())),
    }
  }

  #[pyo3(signature = (*x))]
  fn grad(&self, x: Vec<f64>) -> PyResult<Vec<f64>> {
    match &self.0 {
      Gf::Xy(f) => {
        let [a, b] = xy(&x)?;
        Ok(f.grad(a, b).to_vec())
      }
      Gf::Global(f) => Ok(f.grad(x.as_slice())),
    }
  }

  /// the Hessian, column-major
  #[pyo3(signature = (*x))]
  fn hess(&self, x: Vec<f64>) -> PyResult<Vec<f64>> {
    match &self.0 {
      Gf::Xy(f) => {
        let [a, b] = xy(&x)?;
        Ok(f.hess(a, b).to_vec())
      }
      Gf::Global(f) => Ok(f.hess(x.as_slice())),
    }
  }
}

fn xy(x: &[f64]) -> PyResult<[f64; 2]> {
  match x {
    &[a, b] => Ok([a, b]),
    _ => Err(PyValueError::new_err(format!(
      "a tip function of (x, y), not {} coordinates",
      x.len()
    ))),
  }
}

#[pyclass(module = "getfem_rs", frozen)]
pub struct LevelSet(pub RsLevelSet);

impl LevelSet {
  /// v a GWFL expression in x, y, z or an array over the DOFs
  fn eval(&self, v: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    let mf = self.0.mesh_fem();
    match v.extract::<String>() {
      Ok(e) => {
        let mut ws = Workspace::default();
        for (k, x) in ["x", "y", "z"]
          .iter()
          .enumerate()
          .take(mf.linked_mesh().dim())
        {
          ws.add_macro(x, &format!("X({})", k + 1)).py()?;
        }
        ws.interpolation(&e, &mf, -1).py()
      }
      Err(_) => Ok(
        v.extract::<PyReadonlyArray1<'_, f64>>()?
          .as_slice()?
          .to_vec(),
      ),
    }
  }
}

#[pymethods]
impl LevelSet {
  /// gf.LevelSet(m, degree[, "ws" | f1[, f2 | "ws"]]): Lagrange of degree
  /// on m, the values f1, f2 (0 if none) a GWFL expression in x, y, z
  /// (X(1), X(2), X(3); GetFEM: its own parser) or an array over the level
  /// set's DOFs; with a secondary (f2 or "ws", "with_secondary"), a crack
  /// (the primary 0 on its plane, the secondary < 0 on it)
  #[new]
  #[pyo3(signature = (m, degree, f1 = None, f2 = None))]
  fn new(
    m: PyRef<'_, Mesh>,
    degree: usize,
    f1: Option<&Bound<'_, PyAny>>,
    f2: Option<&Bound<'_, PyAny>>,
  ) -> PyResult<Self> {
    let ws = |f: Option<&Bound<'_, PyAny>>| {
      f.and_then(|f| f.extract::<String>().ok())
        .is_some_and(|s| s == "ws" || s == "with_secondary")
    };
    let (f1, f2, secondary) = match (ws(f1), ws(f2)) {
      (true, _) => (None, None, true),
      (false, true) => (f1, None, true),
      _ => (f1, f2, f2.is_some()),
    };
    let r = LevelSet(RsLevelSet::new(&m.0, degree, secondary).py()?);
    if let Some(f1) = f1 {
      r.set_values(f1, f2)?;
    }
    Ok(r)
  }

  /// the values of the primary (nls 0) or secondary (nls 1) function
  #[pyo3(signature = (nls = 0))]
  fn values<'py>(&self, py: Python<'py>, nls: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(PyArray1::from_vec(py, self.0.values(nls).py()?))
  }

  /// the primary and secondary values, each a GWFL expression in x, y, z
  /// or an array over the DOFs
  #[pyo3(signature = (v1, v2 = None))]
  fn set_values(&self, v1: &Bound<'_, PyAny>, v2: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    self.0.set_values(0, &self.eval(v1)?).py()?;
    match v2 {
      Some(v) => self.0.set_values(1, &self.eval(v)?).py(),
      None => Ok(()),
    }
  }

  /// the Lagrange space of the values
  fn mf(&self) -> MeshFem {
    MeshFem(self.0.mesh_fem())
  }

  fn degree(&self) -> usize {
    self.0.degree()
  }

  /// the basic DOFs of mf whose node is within radius of the front:
  /// primary^2 + secondary^2 < radius^2 (the values interpolated at the
  /// node), sorted
  fn select_mesh_fem_basic_dofs_near_front<'py>(
    &self,
    py: Python<'py>,
    mf: PyRef<'_, MeshFem>,
    radius: f64,
  ) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let d = select_mesh_fem_basic_dofs_near_front(&self.0, &mf.0, radius).py()?;
    Ok(PyArray1::from_vec(
      py,
      d.into_iter().map(|i| i as i64).collect(),
    ))
  }

  /// the fronts of the crack (a level set with a secondary): a list of
  /// (P, T, closed), the points P (dim x n) along each front and their
  /// unit tangents T (dim x n) in order, closed when the last point is
  /// followed by the first; 2D: each tip a front of one point, T None
  /// (spacing None); 3D: points about spacing apart
  #[pyo3(signature = (spacing = None))]
  #[allow(clippy::type_complexity)]
  fn crack_fronts<'py>(
    &self,
    py: Python<'py>,
    spacing: Option<f64>,
  ) -> PyResult<
    Vec<(
      Bound<'py, PyArray2<f64>>,
      Option<Bound<'py, PyArray2<f64>>>,
      bool,
    )>,
  > {
    crack_fronts(&self.0, spacing)
      .py()?
      .iter()
      .map(|f| {
        let t = match f.tangents.is_empty() {
          true => None,
          false => Some(columns(py, &f.tangents)?),
        };
        Ok((columns(py, &f.points)?, t, f.closed))
      })
      .collect()
  }

  /// the stress intensity factors (K_I, K_II, K_III) of the crack by the
  /// interaction integral within radius of its fronts: u (U on mf_u,
  /// integrated with mim) in linear elasticity of Young's modulus E,
  /// Poisson's ratio nu, plane strain in 2D (plane stress: E' = E (1 + 2
  /// nu)/(1 + nu)^2, nu' = nu/(1 + nu)); the fronts of
  /// ls.crack_fronts(spacing), all or the front-th; lips (mim_lips, plus,
  /// minus): the tractions on the lips phi > 0 and phi < 0, GWFL vectors
  /// of X, Normal and numbers. Returns (K, front), K 3 x n: 2D front the
  /// tips (dim x n), 3D a gf.Slice of the fronts' points (K at its points)
  #[pyo3(signature = (mim, mf_u, u, e, nu, radius, spacing = None, front = None, lips = None))]
  #[allow(clippy::too_many_arguments)]
  fn crack_sifs<'py>(
    &self,
    py: Python<'py>,
    mim: PyRef<'_, MeshIm>,
    mf_u: PyRef<'_, MeshFem>,
    u: PyReadonlyArray1<'_, f64>,
    e: f64,
    nu: f64,
    radius: f64,
    spacing: Option<f64>,
    front: Option<usize>,
    lips: Option<(PyRef<'_, MeshIm>, String, String)>,
  ) -> PyResult<(Bound<'py, PyArray2<f64>>, Py<PyAny>)> {
    let mut fronts = crack_fronts(&self.0, spacing).py()?;
    if let Some(k) = front {
      if k >= fronts.len() {
        return Err(PyValueError::new_err(format!(
          "crack_sifs: front {k} of {}",
          fronts.len()
        )));
      }
      fronts = vec![fronts.swap_remove(k)];
    }
    let body = CrackedBody {
      mim: &mim.0,
      mf_u: &mf_u.0,
      u: u.as_slice()?,
      ls: &self.0,
      young: e,
      poisson: nu,
      lips: lips.as_ref().map(|(m, plus, minus)| LipTractions {
        mim: &m.0,
        plus,
        minus,
      }),
    };
    let mut sifs = vec![];
    for f in &fronts {
      sifs.extend(crack_sifs(&body, radius, f).py()?);
    }
    let k_of = |s: &[Sif]| -> PyResult<Bound<'py, PyArray2<f64>>> {
      let n = s.len();
      let v: Vec<f64> = (0..3).flat_map(|i| s.iter().map(move |x| x.k[i])).collect();
      PyArray1::from_vec(py, v).reshape([3, n])
    };
    let mesh = self.0.mesh_fem().linked_mesh();
    if mesh.dim() == 2 {
      let pts: Vec<Vec<f64>> = sifs.iter().map(|s| s.point.clone()).collect();
      return Ok((k_of(&sifs)?, columns(py, &pts)?.into_any().unbind()));
    }
    // a slice of the points (GetFEM's "points": a point on several
    // convexes in each; ends of a front on the boundary up to rounding),
    // K at its points
    let pts: Vec<Vec<f64>> = sifs.iter().map(|s| s.point.clone()).collect();
    let sl = StoredMeshSlice::from_points(&mesh.snapshot(), &mut [], &pts, 1e-8).py()?;
    let at: Vec<Sif> = sl
      .cvlst
      .iter()
      .flat_map(|c| &c.nodes)
      .map(|n| {
        sifs
          .iter()
          .min_by(|a, b| vect_dist2(&a.point, &n.pt).total_cmp(&vect_dist2(&b.point, &n.pt)))
          .unwrap()
          .clone()
      })
      .collect();
    let sl = Py::new(py, Slice(sl, mesh))?;
    Ok((k_of(&at)?, sl.into_any()))
  }
}

/// points as the columns of a dim x n array
fn columns<'py>(py: Python<'py>, pts: &[Vec<f64>]) -> PyResult<Bound<'py, PyArray2<f64>>> {
  let (n, d) = (pts.len(), pts.first().map_or(0, |p| p.len()));
  let v: Vec<f64> = (0..d).flat_map(|i| pts.iter().map(move |p| p[i])).collect();
  PyArray1::from_vec(py, v).reshape([d, n])
}

/// a front (P, T, closed) of ls.crack_fronts
pub fn front_of(
  p: PyReadonlyArray2<'_, f64>,
  t: Option<PyReadonlyArray2<'_, f64>>,
  closed: bool,
) -> CrackFront {
  let cols = |a: &PyReadonlyArray2<'_, f64>| -> Vec<Vec<f64>> {
    let a = a.as_array();
    a.columns().into_iter().map(|c| c.to_vec()).collect()
  };
  CrackFront {
    points: cols(&p),
    tangents: t.as_ref().map(cols).unwrap_or_default(),
    closed,
  }
}

#[pyclass(module = "getfem_rs")]
pub struct MeshLevelSet(pub RsMeshLevelSet);

#[pymethods]
impl MeshLevelSet {
  /// gf.MeshLevelSet(m): the cut of m by the level sets added
  #[new]
  fn new(m: PyRef<'_, Mesh>) -> Self {
    MeshLevelSet(RsMeshLevelSet::new(&m.0))
  }

  fn add(&mut self, ls: PyRef<'_, LevelSet>) {
    self.0.add_level_set(&ls.0);
  }

  fn sup(&mut self, ls: PyRef<'_, LevelSet>) {
    self.0.sup_level_set(&ls.0);
  }

  /// cut the convexes by the level sets
  fn adapt(&mut self, py: Python<'_>) -> PyResult<()> {
    let mls = &mut self.0;
    py.detach(|| mls.adapt()).py()
  }

  fn levelsets(&self) -> Vec<LevelSet> {
    (0..self.0.nb_level_sets())
      .map(|i| LevelSet(self.0.level_set(i).clone()))
      .collect()
  }

  fn linked_mesh(&self) -> Mesh {
    Mesh(self.0.linked_mesh().clone())
  }

  /// the mesh of the sub-elements of the cut convexes and the uncut
  /// convexes (non-conforming); region 0: the faces on a level set
  fn cut_mesh(&self) -> PyResult<Mesh> {
    self.0.global_cut_mesh().py().map(Mesh)
  }

  /// the convexes of the crack tips (fronts), of all level sets or of ls
  #[pyo3(signature = (ls = None))]
  fn crack_tip_convexes(&self, ls: Option<PyRef<'_, LevelSet>>) -> PyResult<Vec<usize>> {
    let cvs = match ls {
      None => self.0.crack_tip_convexes(),
      Some(ls) => {
        let i = (0..self.0.nb_level_sets())
          .find(|&i| self.0.level_set(i).same(&ls.0))
          .ok_or_else(|| {
            PyValueError::new_err("crack_tip_convexes: ls not in the mesh_level_set")
          })?;
        self.0.crack_tip_convexes_of(i)
      }
    };
    Ok(cvs.py()?.iter().copied().collect())
  }
}
