//! Composite elements (port of getfem_fem_composite and
//! bgeot_poly_composite): piecewise polynomial basis functions on a
//! sub-mesh of the reference convex. A point on an interface is evaluated
//! on the sub-element of the largest box rate.

use std::sync::Arc;

use getfem_geom::convex_ref::{self as cr, PRef};
use getfem_geom::geotrans::{self, PGeoTrans};
use getfem_geom::intern::Interner;
use getfem_geom::mesh::{Mesh, MeshData};
use getfem_linalg::tensor::{Mat, Tensor};

use crate::fem::{DofDesc, Fem, PFem, Shape, faces_of};
use crate::fem_transformed::{jac, normal_derivative_correction, polys};
use crate::{Error, Result};

// ------------------------------------------------------------ precomposite mesh

/// A mesh of the reference convex with linear sub-elements, prepared for
/// piecewise polynomials: the affine map of each sub-element and the
/// bounding boxes to locate points.
#[derive(Debug)]
pub(crate) struct MeshPrecomposite {
  pub(crate) mesh: MeshData,
  pub(crate) b: Vec<Mat>,
  pub(crate) det: Vec<f64>,
  org: Vec<Vec<f64>>,
  /// distinct bounding boxes (in order of creation) and their convexes
  boxes: Vec<(Vec<f64>, Vec<f64>, Vec<usize>)>,
}

impl MeshPrecomposite {
  pub(crate) fn new(mesh: MeshData) -> Result<Self> {
    let n = mesh.dim();
    let (mut b, mut det, mut org) = (vec![], vec![], vec![]);
    let mut boxes: Vec<(Vec<f64>, Vec<f64>, Vec<usize>)> = vec![];
    for cv in mesh.convex_index() {
      let pgt = mesh.trans_of_convex(cv);
      if !pgt.is_linear() || pgt.dim() != n {
        return Err(Error::InvalidArgument(
          "composite: linear sub-elements of full dimension only".into(),
        ));
      }
      let g = mesh.points_of_convex(cv);
      let pg = pgt.at(&g, &vec![0.0; n])?;
      b.push(pg.b);
      det.push(pg.j);
      org.push(pg.x);
      let pts: Vec<&[f64]> = mesh
        .ind_points_of_convex(cv)
        .iter()
        .map(|&i| mesh.point(i))
        .collect();
      let lo: Vec<f64> = (0..n)
        .map(|d| pts.iter().map(|p| p[d]).fold(f64::INFINITY, f64::min))
        .collect();
      let hi: Vec<f64> = (0..n)
        .map(|d| pts.iter().map(|p| p[d]).fold(f64::NEG_INFINITY, f64::max))
        .collect();
      let same = |a: &[f64], c: &[f64]| a.iter().zip(c).all(|(x, y)| (x - y).abs() <= 1e-13);
      match boxes
        .iter_mut()
        .find(|(l, h, _)| same(l, &lo) && same(h, &hi))
      {
        Some(bx) => bx.2.push(cv),
        None => boxes.push((lo, hi, vec![cv])),
      }
    }
    Ok(MeshPrecomposite {
      mesh,
      b,
      det,
      org,
      boxes,
    })
  }

  fn local(&self, cv: usize, xref: &[f64]) -> Vec<f64> {
    let b = &self.b[cv];
    (0..b.ncols())
      .map(|d| {
        (0..b.nrows())
          .map(|i| b[(i, d)] * (xref[i] - self.org[cv][i]))
          .sum()
      })
      .collect()
  }

  /// The sub-element of xref and the local coordinates there (GetFEM
  /// polynomial_composite::eval).
  fn locate(&self, xref: &[f64]) -> Result<(usize, Vec<f64>)> {
    let n = xref.len();
    let eps = 1e-13;
    let mut list: Vec<usize> = (0..self.boxes.len())
      .filter(|&k| {
        (0..n).all(|i| xref[i] >= self.boxes[k].0[i] - eps && xref[i] <= self.boxes[k].1[i] + eps)
      })
      .collect();
    while !list.is_empty() {
      let mut best = 0;
      if list.len() > 1 {
        let mut max_rate = -1.0;
        for (j, &k) in list.iter().enumerate() {
          let (lo, hi) = (&self.boxes[k].0, &self.boxes[k].1);
          let mut rate: f64 = 1.0;
          for i in 0..n {
            let h = hi[i] - lo[i];
            if h > 0.0 {
              rate = rate.min((hi[i] - xref[i]).min(xref[i] - lo[i]) / h);
            }
          }
          if rate > max_rate {
            best = j;
            max_rate = rate;
          }
        }
      }
      for &cv in &self.boxes[list[best]].2 {
        let p1ref = self.local(cv, xref);
        if self.mesh.trans_of_convex(cv).convex_ref().is_in(&p1ref)? < 1e-9 {
          return Ok((cv, p1ref));
        }
      }
      if list.len() == 1 {
        break;
      }
      list.remove(best);
    }
    Err(Error::InvalidArgument(format!(
      "element not found in composite polynomial: {xref:?}"
    )))
  }
}

// ------------------------------------------------------------ composite polynomial

/// A piece with its derivatives, in the coordinates of the piece.
#[derive(Clone, Debug)]
struct Piece {
  v: Shape,
  g: Vec<Shape>,
  h: Vec<Shape>,
}

impl Piece {
  fn new(v: Shape) -> Self {
    let n = v.dim();
    let g: Vec<Shape> = (0..n).map(|j| v.derived(j)).collect();
    let h = (0..n * n).map(|jk| g[jk % n].derived(jk / n)).collect();
    Piece { v, g, h }
  }
}

/// Piecewise polynomial (GetFEM polynomial_composite); pieces in local
/// coordinates of the sub-elements or in the coordinates of the whole
/// element; missing pieces are zero.
#[derive(Clone, Debug)]
pub(crate) struct CompPoly {
  sub: Arc<MeshPrecomposite>,
  local: bool,
  pieces: Vec<Option<Piece>>,
}

impl CompPoly {
  pub(crate) fn new(sub: &Arc<MeshPrecomposite>, local: bool) -> Self {
    CompPoly {
      sub: sub.clone(),
      local,
      pieces: vec![None; sub.mesh.nb_convexes_max()],
    }
  }

  pub(crate) fn set(&mut self, cv: usize, p: Shape) {
    self.pieces[cv] = Some(Piece::new(p));
  }

  /// value, gradient, Hessian at xref located on (cv, local coordinates)
  fn eval(&self, xref: &[f64], at: &(usize, Vec<f64>)) -> (f64, Vec<f64>, Vec<f64>) {
    let n = xref.len();
    let Some(p) = &self.pieces[at.0] else {
      return (0.0, vec![0.0; n], vec![0.0; n * n]);
    };
    let yref = if self.local { &at.1[..] } else { xref };
    let v = p.v.eval(yref);
    let gl: Vec<f64> = p.g.iter().map(|s| s.eval(yref)).collect();
    let hl: Vec<f64> = p.h.iter().map(|s| s.eval(yref)).collect();
    if !self.local {
      return (v, gl, hl);
    }
    // derivatives with respect to the element coordinates: B grad, B H B^T
    let b = &self.sub.b[at.0];
    let m = b.ncols();
    let g = (0..n)
      .map(|k| (0..m).map(|j| b[(k, j)] * gl[j]).sum())
      .collect();
    let mut h = vec![0.0; n * n];
    for k in 0..n {
      for q in 0..n {
        h[k + n * q] = (0..m)
          .flat_map(|j| (0..m).map(move |l| (j, l)))
          .map(|(j, l)| b[(k, j)] * b[(q, l)] * hl[j + m * l])
          .sum();
      }
    }
    (v, g, h)
  }
}

// ------------------------------------------------------------ element

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Trans {
  Hct,
  ReducedHct,
  QuadC1,
  ReducedQuadC1,
}

/// Scalar element with piecewise polynomial basis functions.
#[derive(Debug)]
pub struct CompositeFem {
  name: String,
  cvr: PRef,
  dofs: Vec<DofDesc>,
  nodes: Vec<Vec<f64>>,
  faces: Vec<Vec<usize>>,
  base: Vec<CompPoly>,
  lagrange: bool,
  degree: usize,
  trans: Option<Trans>,
  hier_raff: u16,
}

impl CompositeFem {
  fn new(
    name: String,
    cvr: PRef,
    dofs: Vec<DofDesc>,
    nodesref: Vec<Vec<f64>>,
    base: Vec<CompPoly>,
    degree: usize,
  ) -> Self {
    let faces = faces_of(&cvr, &dofs, &nodesref);
    CompositeFem {
      name,
      cvr,
      dofs,
      nodes: nodesref,
      faces,
      base,
      lagrange: false,
      degree,
      trans: None,
      hier_raff: 0,
    }
  }

  /// (values, gradients, Hessians) [nb_base, 1, ...] at xref
  fn values(&self, xref: &[f64]) -> Result<[Tensor<f64>; 3]> {
    let n = self.cvr.dim();
    if xref.len() != n {
      return Err(Error::InvalidArgument(format!(
        "{}: point of dim {}",
        self.name,
        xref.len()
      )));
    }
    let r = self.base.len();
    let mut t = [
      Tensor::zeros(&[r, 1]),
      Tensor::zeros(&[r, 1, n]),
      Tensor::zeros(&[r, 1, n * n]),
    ];
    let mut located: Vec<(*const MeshPrecomposite, (usize, Vec<f64>))> = vec![];
    for (i, b) in self.base.iter().enumerate() {
      let key = Arc::as_ptr(&b.sub);
      let at = match located.iter().find(|(k, _)| *k == key) {
        Some((_, a)) => a.clone(),
        None => {
          let a = b.sub.locate(xref)?;
          located.push((key, a.clone()));
          a
        }
      };
      let (v, g, h) = b.eval(xref, &at);
      t[0].data_mut()[i] = v;
      for (j, gj) in g.iter().enumerate() {
        t[1].data_mut()[i + r * j] = *gj;
      }
      for (j, hj) in h.iter().enumerate() {
        t[2].data_mut()[i + r * j] = *hj;
      }
    }
    Ok(t)
  }
}

impl Fem for CompositeFem {
  fn name(&self) -> &str {
    &self.name
  }
  fn ref_convex(&self) -> &PRef {
    &self.cvr
  }
  fn target_dim(&self) -> usize {
    1
  }
  fn nb_dof(&self) -> usize {
    self.dofs.len()
  }
  fn nb_base(&self) -> usize {
    self.base.len()
  }
  fn dof_types(&self) -> &[DofDesc] {
    &self.dofs
  }
  fn node(&self, i: usize) -> &[f64] {
    &self.nodes[i]
  }
  fn faces_of_dof(&self, i: usize) -> &[usize] {
    &self.faces[i]
  }
  fn is_equivalent(&self) -> bool {
    self.trans.is_none()
  }
  fn is_lagrange(&self) -> bool {
    self.lagrange
  }
  fn is_polynomial(&self) -> bool {
    false
  }
  fn estimated_degree(&self) -> usize {
    self.degree
  }
  fn base_value(&self, xref: &[f64]) -> Result<Tensor<f64>> {
    Ok(self.values(xref)?[0].clone())
  }
  fn grad_base_value(&self, xref: &[f64]) -> Result<Tensor<f64>> {
    Ok(self.values(xref)?[1].clone())
  }
  fn hess_base_value(&self, xref: &[f64]) -> Result<Tensor<f64>> {
    Ok(self.values(xref)?[2].clone())
  }
  fn mat_trans(&self, g: &Tensor<f64>, pgt: &PGeoTrans) -> Result<Option<Tensor<f64>>> {
    match self.trans {
      None => Ok(None),
      Some(t) => c1_mat_trans(t, self, g, pgt).map(Some),
    }
  }
}

// ------------------------------------------------------------ constructors

static CACHE: Interner<String, CompositeFem> = Interner::new();

fn cached(name: String, build: impl FnOnce(String) -> Result<CompositeFem>) -> Result<PFem> {
  Ok(CACHE.get_or_try(&name.clone(), || build(name))? as PFem)
}

pub(crate) fn comp_of(f: &PFem) -> Result<Arc<CompositeFem>> {
  CACHE.get_or_try::<Error>(&f.name().to_string(), || {
    Err(Error::InvalidArgument(format!(
      "{}: not a composite element",
      f.name()
    )))
  })
}

/// The precomposite mesh of the triangles given by points and vertex
/// triples.
fn mesh_precomposite(pts: &[[f64; 2]], tri: &[[usize; 3]]) -> Result<Arc<MeshPrecomposite>> {
  let mut m = MeshData::new(2);
  let ids: Vec<usize> = pts
    .iter()
    .map(|p| m.add_point(p))
    .collect::<std::result::Result<_, _>>()?;
  let gt = geotrans::pk(2, 1)?;
  for t in tri {
    m.add_convex(&gt, &t.map(|i| ids[i]))?;
  }
  Ok(Arc::new(MeshPrecomposite::new(m)?))
}

/// The triangle split at its barycenter (HCT, P1 piecewise linear bubble).
pub(crate) fn hct_mesh() -> Result<Arc<MeshPrecomposite>> {
  let t = 1.0 / 3.0;
  mesh_precomposite(
    &[[t, t], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    &[[0, 2, 3], [0, 3, 1], [0, 1, 2]],
  )
}

/// The square split into 4 triangles at its center (QUADC1).
pub(crate) fn quadc1_mesh() -> Result<Arc<MeshPrecomposite>> {
  mesh_precomposite(
    &[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
    &[[1, 3, 4], [2, 0, 4], [3, 2, 4], [0, 1, 4]],
  )
}

/// The pyramid split into 2 tetrahedra (IM_PYRAMID_COMPOSITE).
pub(crate) fn pyramid_mesh() -> Result<Arc<MeshPrecomposite>> {
  let mut m = MeshData::new(3);
  let pts = [
    [-1.0, -1.0, 0.0],
    [1.0, -1.0, 0.0],
    [-1.0, 1.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
  ];
  let ids: Vec<usize> = pts
    .iter()
    .map(|p| m.add_point(p))
    .collect::<std::result::Result<_, _>>()?;
  let gt = geotrans::pk(3, 1)?;
  for t in [[0, 1, 2, 4], [1, 3, 2, 4]] {
    m.add_convex(&gt, &t.map(|i| ids[i]))?;
  }
  Ok(Arc::new(MeshPrecomposite::new(m)?))
}

/// Basis functions from polynomial strings: function k, sub-element ic in turn.
fn read_base(sub: &Arc<MeshPrecomposite>, nb: usize, s: &str) -> Result<Vec<CompPoly>> {
  let p = polys(2, s)?;
  let nc = sub.mesh.nb_convexes();
  assert_eq!(p.len(), nb * nc);
  Ok(
    (0..nb)
      .map(|k| {
        let mut c = CompPoly::new(sub, false);
        for ic in 0..nc {
          c.set(ic, p[k * nc + ic].clone());
        }
        c
      })
      .collect(),
  )
}

/// Value and both derivatives at each vertex (vertex order of the reference).
fn c1_vertex_dofs(vertices: &[[f64; 2]]) -> (Vec<Vec<f64>>, Vec<DofDesc>) {
  let (mut nodesref, mut dofs) = (vec![], vec![]);
  for v in vertices {
    for d in [
      DofDesc::lagrange(2),
      DofDesc::derivative(2, 0),
      DofDesc::derivative(2, 1),
    ] {
      nodesref.push(v.to_vec());
      dofs.push(d);
    }
  }
  (nodesref, dofs)
}

pub(crate) fn p1_piecewise_linear_bubble() -> Result<PFem> {
  cached("FEM_P1_PIECEWISE_LINEAR_BUBBLE".into(), |name| {
    let sub = hct_mesh()?;
    let base = read_base(&sub, 4, "1-x-y;1-x-y;1-x-y;x;x;x;y;y;y;3-3*x-3*y;3*x;3*y;")?;
    let t = 1.0 / 3.0;
    let nodesref = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![t, t]];
    let dofs = [vec![DofDesc::lagrange(2); 3], vec![DofDesc::bubble1(2)]].concat();
    Ok(CompositeFem::new(
      name,
      cr::simplex_of_reference(2),
      dofs,
      nodesref,
      base,
      3,
    ))
  })
}

const HCT_BASE: &str = "-1 + 9*x + 9*y - 15*x^2 - 30*x*y - 15*y^2 + 7*x^3 + 21*x^2*y + 21*x*y^2 + 7*y^3;
  1 - 3*x^2 - 3*y^2 + 3*x^3 - 3*x^2*y + 2*y^3;
  1 - 3*x^2 - 3*y^2 + 2*x^3 - 3*x*y^2 + 3*y^3;
  -1/6 + 5/2*x - 9/2*x^2 - 4*x*y + 1/2*y^2 + 13/6*x^3 + 4*x^2*y + 3/2*x*y^2 - 1/3*y^3;
  x - 1/2*x^2 - 3*x*y - 7/6*x^3 + 2*x^2*y + 2*x*y^2;
  x - 2*x^2 - 3/2*y^2 + x^3 - 1/2*x*y^2 + 7/3*y^3;
  -1/6 + 5/2*y + 1/2*x^2 - 4*x*y - 9/2*y^2 - 1/3*x^3 + 3/2*x^2*y + 4*x*y^2 + 13/6*y^3;
  y - 3/2*x^2 - 2*y^2 + 7/3*x^3 - 1/2*x^2*y + y^3;
  y - 3*x*y - 1/2*y^2 + 2*x^2*y + 2*x*y^2 - 7/6*y^3;
  1 - 9/2*x - 9/2*y + 9*x^2 + 15*x*y + 6*y^2 - 9/2*x^3 - 21/2*x^2*y - 21/2*x*y^2 - 5/2*y^3;
  3*x^2 - 5/2*x^3 + 3/2*x^2*y;
  3*x^2 - 2*x^3 + 3/2*x*y^2 - 1/2*y^3;
  -1/6 + 3/4*x + 3/4*y - 2*x^2 - 5/2*x*y - y^2 + 17/12*x^3 + 7/4*x^2*y + 7/4*x*y^2 + 5/12*y^3;
  -x^2 + 13/12*x^3 - 1/4*x^2*y;
  -x^2 + x^3 - 1/4*x*y^2 + 1/12*y^3;
  2/3 - 13/4*x - 11/4*y + 9/2*x^2 + 19/2*x*y + 7/2*y^2 - 23/12*x^3 - 23/4*x^2*y - 25/4*x*y^2 - 17/12*y^3;
  -1/2*x^2 + 5/12*x^3 + 9/4*x^2*y;
  -x*y + 1/2*y^2 + 2*x^2*y + 7/4*x*y^2 - 13/12*y^3;
  1 - 9/2*x - 9/2*y + 6*x^2 + 15*x*y + 9*y^2 - 5/2*x^3 - 21/2*x^2*y - 21/2*x*y^2 - 9/2*y^3;
  3*y^2 - 1/2*x^3 + 3/2*x^2*y - 2*y^3;
  3*y^2 + 3/2*x*y^2 - 5/2*y^3;
  2/3 - 11/4*x - 13/4*y + 7/2*x^2 + 19/2*x*y + 9/2*y^2 - 17/12*x^3 - 25/4*x^2*y - 23/4*x*y^2 - 23/12*y^3;
  1/2*x^2 - x*y - 13/12*x^3 + 7/4*x^2*y + 2*x*y^2;
  -1/2*y^2 + 9/4*x*y^2 + 5/12*y^3;
  -1/6 + 3/4*x + 3/4*y - x^2 - 5/2*x*y - 2*y^2 + 5/12*x^3 + 7/4*x^2*y + 7/4*x*y^2 + 17/12*y^3;
  -y^2 + 1/12*x^3 - 1/4*x^2*y + y^3;
  -y^2 - 1/4*x*y^2 + 13/12*y^3;
  -sqrt(2)*2/3 + sqrt(2)*3*x + sqrt(2)*3*y - sqrt(2)*4*x^2 - sqrt(2)*10*x*y - sqrt(2)*4*y^2 + sqrt(2)*5/3*x^3 + sqrt(2)*7*x^2*y + sqrt(2)*7*x*y^2 + sqrt(2)*5/3*y^3;
  sqrt(2)*1/3*x^3 - sqrt(2)*x^2*y;
  -sqrt(2)*x*y^2 + sqrt(2)*1/3*y^3;
  2/3 - 2*x - 4*y + 2*x^2 + 8*x*y + 6*y^2 - 2/3*x^3 - 4*x^2*y - 6*x*y^2 - 8/3*y^3;
  2*x^2 - 4*x*y - 10/3*x^3 + 4*x^2*y + 4*x*y^2;
  -2*y^2 + 2*x*y^2 + 8/3*y^3;
  2/3 - 4*x - 2*y + 6*x^2 + 8*x*y + 2*y^2 - 8/3*x^3 - 6*x^2*y - 4*x*y^2 - 2/3*y^3;
  -2*x^2 + 8/3*x^3 + 2*x^2*y;
  -4*x*y + 2*y^2 + 4*x^2*y + 4*x*y^2 - 10/3*y^3;";

const TRI_VERTICES: [[f64; 2]; 3] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
const QUAD_VERTICES: [[f64; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];

pub(crate) fn hct_triangle(reduced: bool) -> Result<PFem> {
  let name = if reduced {
    "FEM_REDUCED_HCT_TRIANGLE"
  } else {
    "FEM_HCT_TRIANGLE"
  };
  cached(name.into(), |name| {
    let sub = hct_mesh()?;
    let base = read_base(&sub, 12, HCT_BASE)?;
    let (mut nodesref, mut dofs) = c1_vertex_dofs(&TRI_VERTICES);
    if !reduced {
      for ptref in [[0.5, 0.5], [0.0, 0.5], [0.5, 0.0]] {
        nodesref.push(ptref.to_vec());
        dofs.push(DofDesc::normal_derivative(2));
      }
    }
    let mut f = CompositeFem::new(name, cr::simplex_of_reference(2), dofs, nodesref, base, 5);
    f.trans = Some(if reduced {
      Trans::ReducedHct
    } else {
      Trans::Hct
    });
    Ok(f)
  })
}

const QUADC1_BASE: &str = "2 - 3*x - 3*y + 6*x*y + x^3 - 3*x^2*y;
  1 - 3*x^2 - 3*y^2 + x^3 + 3*x^2*y + 2*y^3;
  2 - 3*x - 3*y + 6*x*y - 3*x*y^2 + y^3;
  1 - 3*x^2 - 3*y^2 + 2*x^3 + 3*x*y^2 + y^3;
  1/6 + 1/2*x - y - 3/2*x^2 + 2*x*y + 5/6*x^3 - x^2*y;
  x - 1/2*x^2 - 3*x*y + 1/6*x^3 + x^2*y + 2*x*y^2;
  1/6 + 1/2*x - y - x*y + 3/2*y^2 + 1/2*x*y^2 - 2/3*y^3;
  x - 2*x^2 - 3/2*y^2 + x^3 + 3/2*x*y^2 + 2/3*y^3;
  1/6 - x + 1/2*y + 3/2*x^2 - x*y - 2/3*x^3 + 1/2*x^2*y;
  y - 3/2*x^2 - 2*y^2 + 2/3*x^3 + 3/2*x^2*y + y^3;
  1/6 - x + 1/2*y + 2*x*y - 3/2*y^2 - x*y^2 + 5/6*y^3;
  y - 3*x*y - 1/2*y^2 + 2*x^2*y + x*y^2 + 1/6*y^3;
  -1 + 3*x + 3*y - 6*x*y - 3*y^2 - x^3 + 3*x^2*y + 2*y^3;
  3*x^2 - x^3 - 3*x^2*y;
  -1 + 3*x + 3*y - 6*x*y - 3*y^2 + 3*x*y^2 + y^3;
  3*x^2 - 2*x^3 - 3*x*y^2 + y^3;
  -2/3 + 1/2*x + 2*y - x*y - 2*y^2 + 1/6*x^3 - x^2*y + 2*x*y^2;
  -x^2 + 5/6*x^3 + x^2*y;
  -2/3 + 1/2*x + 2*y - x*y - 2*y^2 + 1/2*x*y^2 + 2/3*y^3;
  -x^2 + x^3 + 3/2*x*y^2 - 2/3*y^3;
  -5/6 + x + 5/2*y + 1/2*x^2 - 3*x*y - 2*y^2 - 2/3*x^3 + 3/2*x^2*y + y^3;
  -1/2*x^2 + 2/3*x^3 + 1/2*x^2*y;
  -5/6 + x + 5/2*y - 2*x*y - 5/2*y^2 + x*y^2 + 5/6*y^3;
  -x*y + 1/2*y^2 + 2*x^2*y - x*y^2 + 1/6*y^3;
  -1 + 3*x + 3*y - 3*x^2 - 6*x*y + x^3 + 3*x^2*y;
  3*y^2 + x^3 - 3*x^2*y - 2*y^3;
  -1 + 3*x + 3*y - 3*x^2 - 6*x*y + 2*x^3 + 3*x*y^2 - y^3;
  3*y^2 - 3*x*y^2 - y^3;
  -5/6 + 5/2*x + y - 5/2*x^2 - 2*x*y + 5/6*x^3 + x^2*y;
  1/2*x^2 - x*y + 1/6*x^3 - x^2*y + 2*x*y^2;
  -5/6 + 5/2*x + y - 2*x^2 - 3*x*y + 1/2*y^2 + x^3 + 3/2*x*y^2 - 2/3*y^3;
  -1/2*y^2 + 1/2*x*y^2 + 2/3*y^3;
  -2/3 + 2*x + 1/2*y - 2*x^2 - x*y + 2/3*x^3 + 1/2*x^2*y;
  -y^2 - 2/3*x^3 + 3/2*x^2*y + y^3;
  -2/3 + 2*x + 1/2*y - 2*x^2 - x*y + 2*x^2*y - x*y^2 + 1/6*y^3;
  -y^2 + x*y^2 + 5/6*y^3;
  1 - 3*x - 3*y + 3*x^2 + 6*x*y + 3*y^2 - x^3 - 3*x^2*y - 2*y^3;
  -x^3 + 3*x^2*y;
  1 - 3*x - 3*y + 3*x^2 + 6*x*y + 3*y^2 - 2*x^3 - 3*x*y^2 - y^3;
  3*x*y^2 - y^3;
  -2/3 + 3/2*x + 2*y - x^2 - 3*x*y - 2*y^2 + 1/6*x^3 + x^2*y + 2*x*y^2;
  5/6*x^3 - x^2*y;
  -2/3 + 3/2*x + 2*y - x^2 - 3*x*y - 2*y^2 + x^3 + 3/2*x*y^2 + 2/3*y^3;
  1/2*x*y^2 - 2/3*y^3;
  -2/3 + 2*x + 3/2*y - 2*x^2 - 3*x*y - y^2 + 2/3*x^3 + 3/2*x^2*y + y^3;
  -2/3*x^3 + 1/2*x^2*y;
  -2/3 + 2*x + 3/2*y - 2*x^2 - 3*x*y - y^2 + 2*x^2*y + x*y^2 + 1/6*y^3;
  -x*y^2 + 5/6*y^3;
  4/3 - 2*x - 4*y + 4*x*y + 4*y^2 + 2/3*x^3 - 4*x*y^2;
  -2/3*x^3;
  4/3 - 2*x - 4*y + 4*x*y + 4*y^2 - 2*x*y^2 - 4/3*y^3;
  -2*x*y^2 + 4/3*y^3;
  -2/3 + 2*x - 2*x^2 + 2/3*x^3;
  2*x^2 - 4*x*y - 2/3*x^3 + 4*x*y^2;
  -2/3 + 2*x - 4*x*y + 2*y^2 + 2*x*y^2 - 4/3*y^3;
  -2*y^2 + 2*x*y^2 + 4/3*y^3;
  4/3 - 4*x - 2*y + 4*x^2 + 4*x*y - 4/3*x^3 - 2*x^2*y;
  4/3*x^3 - 2*x^2*y;
  4/3 - 4*x - 2*y + 4*x^2 + 4*x*y - 4*x^2*y + 2/3*y^3;
  -2/3*y^3;
  -2/3 + 2*y + 2*x^2 - 4*x*y - 4/3*x^3 + 2*x^2*y;
  -2*x^2 + 4/3*x^3 + 2*x^2*y;
  -2/3 + 2*y - 2*y^2 + 2/3*y^3;
  -4*x*y + 2*y^2 + 4*x^2*y - 2/3*y^3;";

pub(crate) fn quadc1(reduced: bool) -> Result<PFem> {
  let name = if reduced {
    "FEM_REDUCED_QUADC1_COMPOSITE"
  } else {
    "FEM_QUADC1_COMPOSITE"
  };
  cached(name.into(), |name| {
    let sub = quadc1_mesh()?;
    let base = read_base(&sub, 16, QUADC1_BASE)?;
    let (mut nodesref, mut dofs) = c1_vertex_dofs(&QUAD_VERTICES);
    if !reduced {
      for ptref in [[1.0, 0.5], [0.0, 0.5], [0.5, 1.0], [0.5, 0.0]] {
        nodesref.push(ptref.to_vec());
        dofs.push(DofDesc::normal_derivative(2));
      }
    }
    let mut f = CompositeFem::new(
      name,
      cr::parallelepiped_of_reference(2, 1),
      dofs,
      nodesref,
      base,
      5,
    );
    f.trans = Some(if reduced {
      Trans::ReducedQuadC1
    } else {
      Trans::QuadC1
    });
    Ok(f)
  })
}

// ------------------------------------------------------------ M_T

/// HCT and QUADC1: K at the vertices, then the normal-derivative
/// correction; the reduced variants replace the normal derivatives by the
/// mean of the vertex gradients of the edge (M = P^T M_full).
fn c1_mat_trans(t: Trans, f: &CompositeFem, g: &Mat, pgt: &PGeoTrans) -> Result<Mat> {
  if g.nrows() != 2 {
    return Err(Error::InvalidArgument(format!(
      "{}: 2D elements only",
      f.name
    )));
  }
  let (full, nv) = match t {
    Trans::Hct | Trans::ReducedHct => (hct_triangle(false)?, 3),
    Trans::QuadC1 | Trans::ReducedQuadC1 => (quadc1(false)?, 4),
  };
  let full = comp_of(&full)?;
  let nb = 4 * nv;
  let first = 3 * nv;
  let mut m = Mat::identity(nb);
  let mut k = jac(g, pgt, full.node(0));
  for v in 0..nv {
    if v > 0 && !pgt.is_linear() {
      k = jac(g, pgt, full.node(3 * v));
    }
    for r in 0..2 {
      for c in 0..2 {
        m[(1 + 3 * v + r, 1 + 3 * v + c)] = k[(r, c)];
      }
    }
  }
  let normals = normal_derivative_correction(&*full, g, pgt, &mut m, first, nv, k)?;
  if matches!(t, Trans::Hct | Trans::QuadC1) {
    return Ok(m);
  }
  // edge e (normal DOF first + e) between which vertices
  let edges: &[[usize; 2]] = if nv == 3 {
    &[[1, 2], [0, 2], [0, 1]]
  } else {
    &[[1, 3], [0, 2], [2, 3], [0, 1]]
  };
  let mut p = Mat::zeros(&[nb, first]);
  for i in 0..first {
    p[(i, i)] = 1.0;
  }
  for (e, vs) in edges.iter().enumerate() {
    for &v in vs {
      for c in 0..2 {
        p[(first + e, 1 + 3 * v + c)] = 0.5 * normals[e][c];
      }
    }
  }
  Ok(p.transposed().matmul(&m))
}

// ------------------------------------------------------------ structured composite

/// k-refinement of the reference simplex of dimension n into m, points
/// mapped by tr (GetFEM structured_mesh_for_simplex_).
fn refine_simplex(
  m: &mut MeshData,
  n: usize,
  k: usize,
  tr: &dyn Fn(&[f64]) -> Vec<f64>,
) -> Result<()> {
  let h = 1.0 / k as f64;
  let gt = geotrans::pk(n, 1)?;
  let add = |m: &mut MeshData, pts: &[Vec<f64>]| -> Result<()> {
    let ids: Vec<usize> = pts
      .iter()
      .map(|p| m.add_point(&tr(p)))
      .collect::<std::result::Result<_, _>>()?;
    m.add_convex(&gt, &ids)?;
    Ok(())
  };
  match n {
    1 => {
      for i in 0..k {
        add(m, &[vec![i as f64 * h], vec![(i + 1) as f64 * h]])?;
      }
    }
    2 => {
      for i in 0..k {
        let (a, b) = (i as f64 * h, (i + 1) as f64 * h);
        for l in 0..k - i {
          let (c, d) = (l as f64 * h, (l + 1) as f64 * h);
          let (pa, pb, pc, pd) = (vec![a, c], vec![b, c], vec![a, d], vec![b, d]);
          // all 4 points, also without the second triangle: GetFEM's point
          // numbering
          for p in [&pa, &pb, &pc, &pd] {
            m.add_point(&tr(p))?;
          }
          add(m, &[pa, pb.clone(), pc.clone()])?;
          if l + i + 1 < k {
            add(m, &[pc, pb, pd])?;
          }
        }
      }
    }
    3 => {
      for ci in 0..k {
        let x = ci as f64 * h;
        for cj in 0..k - ci {
          let y = cj as f64 * h;
          for ck in 0..k - ci - cj {
            let z = ck as f64 * h;
            let c = |dx: f64, dy: f64, dz: f64| vec![x + dx, y + dy, z + dz];
            let pts = [
              c(0.0, 0.0, 0.0),
              c(h, 0.0, 0.0),
              c(0.0, h, 0.0),
              c(h, h, 0.0),
              c(0.0, 0.0, h),
              c(h, 0.0, h),
              c(0.0, h, h),
              c(h, h, h),
            ];
            let s = ci + cj + ck;
            let mut t = [usize::MAX; 8];
            for i in [0, 1, 2, 4] {
              t[i] = m.add_point(&tr(&pts[i]))?;
            }
            if k > 1 && s + 1 < k {
              for i in [3, 5, 6] {
                t[i] = m.add_point(&tr(&pts[i]))?;
              }
            }
            if k > 2 && s + 2 < k {
              t[7] = m.add_point(&pts[7])?;
            }
            m.add_convex(&gt, &[t[0], t[1], t[2], t[4]])?;
            if k > 1 && s + 1 < k {
              m.add_convex(&gt, &[t[1], t[2], t[4], t[5]])?;
              m.add_convex(&gt, &[t[6], t[4], t[2], t[5]])?;
              m.add_convex(&gt, &[t[2], t[3], t[5], t[1]])?;
              m.add_convex(&gt, &[t[2], t[5], t[3], t[6]])?;
            }
            if k > 2 && s + 2 < k {
              m.add_convex(&gt, &[t[3], t[5], t[7], t[6]])?;
            }
          }
        }
      }
    }
    _ => {
      return Err(Error::InvalidArgument(format!(
        "structured mesh of simplices of dimension {n}"
      )));
    }
  }
  Ok(())
}

/// Regular refinement of the reference simplex or parallelepiped
/// (GetFEM structured_mesh_for_convex, without simplexification).
pub(crate) fn structured_mesh_for_convex(cvr: &PRef, k: usize) -> Result<MeshData> {
  use getfem_geom::convex_structure as cs;
  let n = cvr.dim();
  let basic = cvr.structure().basic();
  let mut m = MeshData::new(n);
  let h = 1.0 / k as f64;
  if *basic == *cs::simplex_structure(n) {
    refine_simplex(&mut m, n, k, &|p| p.to_vec())?;
  } else if *basic == *cs::parallelepiped_structure(n, 1) {
    // points in lexicographic order (first coordinate fastest)
    let mut pids = vec![];
    let mut cnt = vec![0usize; n + 1];
    while cnt[n] == 0 {
      let pt: Vec<f64> = (0..n).map(|z| h * cnt[z] as f64).collect();
      pids.push(m.add_point(&pt)?);
      let mut kk = 0;
      while kk <= n {
        cnt[kk] += 1;
        if kk < n && cnt[kk] == k + 1 {
          cnt[kk] = 0;
          kk += 1;
        } else {
          break;
        }
      }
    }
    let gt = geotrans::linear_qk(n)?;
    let strides: Vec<usize> = (0..n).map(|i| (k + 1).pow(i as u32)).collect();
    let mut cnt = vec![0usize; n + 1];
    while cnt[n] == 0 {
      let ppts: Vec<usize> = (0..1usize << n)
        .map(|v| {
          let pos: usize = (0..n).map(|z| (cnt[z] + ((v >> z) & 1)) * strides[z]).sum();
          pids[pos]
        })
        .collect();
      m.add_convex(&gt, &ppts)?;
      let mut kk = 0;
      cnt[0] += 1;
      while kk < n && cnt[kk] == k {
        cnt[kk] = 0;
        kk += 1;
        cnt[kk] += 1;
      }
    }
  } else {
    return Err(Error::InvalidArgument(
      "structured mesh: simplices and parallelepipeds only".into(),
    ));
  }
  Ok(m)
}

/// Sub-meshes by (convex, k), shared by the elements and integration methods.
static MESHES: Interner<(String, usize), MeshPrecomposite> = Interner::new();

pub(crate) fn structured_mesh_precomposite(cvr: &PRef, k: usize) -> Result<Arc<MeshPrecomposite>> {
  let key = (format!("{:?}", cvr.structure().basic()), k);
  MESHES.get_or_try(&key, || {
    MeshPrecomposite::new(structured_mesh_for_convex(cvr, k)?)
  })
}

/// pf on each sub-element of the k-refined reference convex
/// (FEM_STRUCTURED_COMPOSITE(pf, k)).
pub(crate) fn structured_composite(pf: &PFem, k: usize) -> Result<PFem> {
  if k == 0 {
    return Err(Error::InvalidArgument(
      "FEM_STRUCTURED_COMPOSITE: k >= 1".into(),
    ));
  }
  let name = format!("FEM_STRUCTURED_COMPOSITE({},{k})", pf.name());
  let pf = pf.clone();
  cached(name, |name| {
    let poly = crate::fem::poly_of(&pf)?;
    if !pf.is_equivalent() || !pf.is_polynomial() || pf.target_dim() != 1 {
      return Err(Error::InvalidArgument(format!(
        "{name}: scalar polynomial tau-equivalent elements only"
      )));
    }
    let sub = structured_mesh_precomposite(pf.ref_convex(), k)?;
    let mesh = Mesh::from(sub.mesh.clone());
    let mf = crate::mesh_fem::MeshFem::new(&mesh, 1).with_finite_element(&pf)?;
    let num = mf.numbering()?;
    let nd = num.nb_basic_dof();
    let mut base = vec![CompPoly::new(&sub, true); nd];
    let mut dofs = vec![None; nd];
    for cv in sub.mesh.convex_index() {
      for (i, &d) in num.ind_basic_dof_of_element(cv).iter().enumerate() {
        base[d].set(cv, poly.basis()[i].clone());
        dofs[d] = Some(pf.dof_types()[i].clone());
      }
    }
    let nodesref = (0..nd)
      .map(|d| num.point_of_basic_dof(d))
      .collect::<Result<Vec<_>>>()?;
    let dofs = dofs.into_iter().map(|d| d.unwrap()).collect();
    let mut f = CompositeFem::new(
      name,
      pf.ref_convex().clone(),
      dofs,
      nodesref,
      base,
      pf.estimated_degree(),
    );
    f.lagrange = pf.is_lagrange();
    Ok(f)
  })
}

/// Hierarchical composite element: f1 and the functions of f2 at new DOFs.
fn gen_hierarchical_comp(name: String, f1: &PFem, f2: &PFem) -> Result<PFem> {
  let (p1, p2) = (comp_of(f1)?, comp_of(f2)?);
  cached(name, |name| {
    let raff = p1.hier_raff + 1;
    let (mut nodesref, mut dofs, mut base) = (p1.nodes.clone(), p1.dofs.clone(), p1.base.clone());
    for i in 0..p2.nodes.len() {
      let found = (0..p1.nodes.len()).any(|j| {
        let d: f64 = p2.nodes[i]
          .iter()
          .zip(&p1.nodes[j])
          .map(|(a, b)| (a - b) * (a - b))
          .sum();
        d.sqrt() < 1e-13 && p2.dofs[i].hierarchical_compatible(&p1.dofs[j])
      });
      if !found {
        let mut d = p2.dofs[i].clone();
        d.ddl.iter_mut().for_each(|e| e.hier_raff = raff);
        nodesref.push(p2.nodes[i].clone());
        dofs.push(d);
        base.push(p2.base[i].clone());
      }
    }
    let mut f = CompositeFem::new(
      name,
      p1.cvr.clone(),
      dofs,
      nodesref,
      base,
      p1.degree.max(p2.degree),
    );
    f.hier_raff = raff;
    Ok(f)
  })
}

/// FEM_PK_HIERARCHICAL_COMPOSITE(n,k,s) (full: over PK_HIERARCHICAL).
pub(crate) fn pk_hierarchical_composite(n: usize, k: usize, s: usize, full: bool) -> Result<PFem> {
  if n == 0 || k == 0 || s == 0 || (!s.is_multiple_of(2) && s != 1) {
    return Err(Error::InvalidArgument(
      "FEM_PK_(FULL_)HIERARCHICAL_COMPOSITE(n,k,s): n, k >= 1, s = 1 or even".into(),
    ));
  }
  let pk = if full {
    crate::fem_special::pk_hierarchical(n, k)?
  } else {
    crate::fem::pk(n, k, None)?
  };
  if s == 1 {
    return structured_composite(&pk, 1);
  }
  let t = (2..=s).find(|&t| s.is_multiple_of(t)).unwrap();
  let prefix = if full {
    "FEM_PK_FULL_HIERARCHICAL_COMPOSITE"
  } else {
    "FEM_PK_HIERARCHICAL_COMPOSITE"
  };
  let f1 = pk_hierarchical_composite(n, k, s / t, full)?;
  let f2 = structured_composite(&pk, s)?;
  gen_hierarchical_comp(
    format!(
      "FEM_GEN_HIERARCHICAL({prefix}({n},{k},{}),{})",
      s / t,
      f2.name()
    ),
    &f1,
    &f2,
  )
}

// ------------------------------------------------------ refined simplex mesh

/// A convex split into simplices, each refined k times (for slices):
/// points in the reference convex, simplices, and per face of the convex
/// the sub-faces on it.
#[derive(Debug)]
pub struct RefinedSimplexMesh {
  pub points: Vec<Vec<f64>>,
  pub simplexes: Vec<Vec<usize>>,
  pub faces: Vec<Vec<Vec<usize>>>,
}

static REFINED: Interner<(String, usize), RefinedSimplexMesh> = Interner::new();

pub fn refined_simplex_mesh_for_convex(cvr: &PRef, k: usize) -> Result<Arc<RefinedSimplexMesh>> {
  let key = (format!("{:?}", cvr.structure().basic()), k);
  REFINED.get_or_try(&key, || {
    let n = cvr.dim();
    let basic = cvr.clone().basic();
    let splx = getfem_geom::simplexify::simplexified(basic.structure()).ok_or_else(|| {
      Error::InvalidArgument("refined simplex mesh: no simplexification of this convex".into())
    })?;
    let sgt = geotrans::pk(n, 1)?;
    let mut m = MeshData::new(n);
    for s in &splx {
      let pts: Vec<f64> = s.iter().flat_map(|&i| basic.points()[i].clone()).collect();
      let g = Tensor::from_data(&[n, s.len()], pts)?;
      refine_simplex(&mut m, n, k.max(1), &|p| {
        sgt.transform(p, &g).expect("simplex point")
      })?;
    }
    let points: Vec<Vec<f64>> = m.point_index().map(|i| m.point(i).to_vec()).collect();
    let simplexes: Vec<Vec<usize>> = m
      .convex_index()
      .map(|cv| m.ind_points_of_convex(cv).to_vec())
      .collect();
    let mut faces = vec![];
    for f in 0..cvr.structure().nb_faces() {
      let on: Vec<bool> = points
        .iter()
        .map(|p| {
          cvr
            .is_in_face(f, p)
            .map(|v| v.abs() < 1e-12)
            .unwrap_or(false)
        })
        .collect();
      let mut fm = vec![];
      for cv in m.convex_index() {
        for ff in 0..m.trans_of_convex(cv).structure().nb_faces() {
          let ipts = m.ind_points_of_face_of_convex(cv, ff);
          if ipts.iter().all(|&i| on[i]) {
            fm.push(ipts);
          }
        }
      }
      faces.push(fm);
    }
    Ok(RefinedSimplexMesh {
      points,
      simplexes,
      faces,
    })
  })
}
