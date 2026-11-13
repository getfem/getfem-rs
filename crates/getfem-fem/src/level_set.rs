//! Level sets (port of getfem_level_set): a primary function and an
//! optional secondary one on a Lagrange space of degree k, as a shared
//! handle. On each convex, a level set is a polynomial of the reference
//! coordinates (GetFEM mls_of_convex).

use getfem_geom::mesh::Mesh;
use getfem_geom::shared::Shared;

use crate::fem::{PFem, Poly};
use crate::mesh_fem::{MeshFem, new_stamp};
use crate::{Error, Result};

#[derive(Clone, Debug)]
struct Inner {
  mf: MeshFem,
  degree: usize,
  values: [Vec<f64>; 2],
  with_secondary: bool,
  /// changes with every change of the values
  version: u64,
}

/// A level set (GetFEM level_set): values on the nodes of a Lagrange space
/// of degree k of the mesh, primary (0) and optionally secondary (1).
#[derive(Clone, Debug)]
pub struct LevelSet(Shared<Inner>);

/// A level set on one convex, a function of the reference coordinates.
#[derive(Clone, Debug)]
pub struct ConvexLevelSet {
  pf: PFem,
  coeff: Vec<f64>,
  /// the function, its gradient and Hessian (column-major) as polynomials
  /// (GetFEM mesher_level_set), for polynomial elements
  polys: Option<(Poly, Vec<Poly>, Vec<Poly>)>,
}

impl ConvexLevelSet {
  fn new(pf: PFem, coeff: Vec<f64>) -> Self {
    let polys = pf.base_polynomials().map(|base| {
      let n = pf.dim();
      let mut f = Poly::zero(n);
      for (b, c) in base.iter().zip(&coeff) {
        f += &(b * *c);
      }
      let g: Vec<Poly> = (0..n).map(|k| f.derived(k)).collect();
      let h: Vec<Poly> = (0..n * n).map(|jk| g[jk % n].derived(jk / n)).collect();
      (f, g, h)
    });
    ConvexLevelSet { pf, coeff, polys }
  }
}

impl LevelSet {
  /// Its identity (the same for its clones), never 0: the level set an
  /// integration point is on (`ApproxIntegration::level_set_of_point`).
  pub fn id(&self) -> usize {
    self.0.id()
  }

  /// Zero level sets of degree k on mesh (GetFEM level_set(mesh, k,
  /// with_secondary)).
  pub fn new(mesh: &Mesh, degree: usize, with_secondary: bool) -> Result<Self> {
    let mf = MeshFem::new(mesh, 1).with_classical_finite_element(degree)?;
    let n = mf.nb_dof()?;
    Ok(LevelSet(Shared::new(Inner {
      mf,
      degree,
      values: [
        vec![0.0; n],
        if with_secondary { vec![0.0; n] } else { vec![] },
      ],
      with_secondary,
      version: new_stamp(),
    })))
  }

  pub fn mesh_fem(&self) -> MeshFem {
    self.0.snapshot().mf.clone()
  }
  pub fn degree(&self) -> usize {
    self.0.snapshot().degree
  }
  pub fn has_secondary(&self) -> bool {
    self.0.snapshot().with_secondary
  }
  /// changes when the values change
  pub fn version(&self) -> u64 {
    self.0.snapshot().version
  }
  pub fn same(&self, other: &LevelSet) -> bool {
    self.0.same(&other.0)
  }

  fn check(&self, i: usize) -> Result<()> {
    if i > 1 || (i == 1 && !self.has_secondary()) {
      return Err(Error::InvalidArgument(format!(
        "level set: no function {i}"
      )));
    }
    Ok(())
  }

  /// The values of the primary (0) or secondary (1) function.
  pub fn values(&self, i: usize) -> Result<Vec<f64>> {
    self.check(i)?;
    Ok(self.0.snapshot().values[i].clone())
  }

  pub fn set_values(&self, i: usize, v: &[f64]) -> Result<()> {
    self.check(i)?;
    let n = self.mesh_fem().nb_dof()?;
    if v.len() != n {
      return Err(Error::InvalidArgument(format!(
        "level set: {} values, {n} expected",
        v.len()
      )));
    }
    self.0.modify(|s| {
      s.values[i] = v.to_vec();
      s.version = new_stamp();
    });
    Ok(())
  }

  /// Sets to 0 the values below eps times the radius estimate of a convex
  /// of the node (GetFEM simplify).
  pub fn simplify(&self, eps: f64) -> Result<()> {
    let mf = self.mesh_fem();
    let num = mf.numbering()?;
    let m = num.linked_mesh();
    let mut v = self.0.snapshot().values.clone();
    for cv in num.convex_index() {
      // GetFEM convex_radius_estimate: q sqrt(N) / 2, half the element size
      let h = m
        .trans_of_convex(cv)
        .element_size(&m.points_of_convex(cv))?
        / 2.0;
      for &d in num.ind_basic_dof_of_element(cv) {
        for vi in v.iter_mut().filter(|vi| !vi.is_empty()) {
          if vi[d].abs() < h * eps {
            vi[d] = 0.0;
          }
        }
      }
    }
    self.0.modify(|s| {
      s.values = v;
      s.version = new_stamp();
    });
    Ok(())
  }

  /// The primary (0) or secondary (1) function on convex cv (GetFEM
  /// mls_of_convex, without the deprecated shift).
  pub fn of_convex(&self, cv: usize, i: usize) -> Result<ConvexLevelSet> {
    self.check(i)?;
    let s = self.0.snapshot();
    let num = s.mf.numbering()?;
    let pf = num
      .fem_of_element(cv)
      .ok_or_else(|| Error::InvalidArgument(format!("convex {cv} is not in the level set mesh")))?
      .clone();
    let coeff = num
      .ind_basic_dof_of_element(cv)
      .iter()
      .map(|&d| s.values[i][d])
      .collect();
    Ok(ConvexLevelSet::new(pf, coeff))
  }
}

impl ConvexLevelSet {
  pub fn dim(&self) -> usize {
    self.pf.dim()
  }
  pub fn val(&self, xref: &[f64]) -> Result<f64> {
    if let Some((f, _, _)) = &self.polys {
      return Ok(f.eval(xref));
    }
    let b = self.pf.base_value(xref)?;
    Ok(self.coeff.iter().zip(b.data()).map(|(c, v)| c * v).sum())
  }
  /// gradient in the reference coordinates
  pub fn grad(&self, xref: &[f64]) -> Result<Vec<f64>> {
    if let Some((_, g, _)) = &self.polys {
      return Ok(g.iter().map(|p| p.eval(xref)).collect());
    }
    let (g, nb) = (self.pf.grad_base_value(xref)?, self.coeff.len());
    Ok(
      (0..self.dim())
        .map(|k| (0..nb).map(|i| self.coeff[i] * g.data()[i + nb * k]).sum())
        .collect(),
    )
  }
  /// Hessian in the reference coordinates (column-major)
  pub fn hess(&self, xref: &[f64]) -> Result<Vec<f64>> {
    if let Some((_, _, h)) = &self.polys {
      return Ok(h.iter().map(|p| p.eval(xref)).collect());
    }
    let (h, nb, n) = (self.pf.hess_base_value(xref)?, self.coeff.len(), self.dim());
    Ok(
      (0..n * n)
        .map(|jk| (0..nb).map(|i| self.coeff[i] * h.data()[i + nb * jk]).sum())
        .collect(),
    )
  }
}

/// The level set as a constraint in the reference coordinates; NaN where
/// its element cannot be evaluated (the projections then fail).
impl getfem_geom::signed_distance::SignedDistance for ConvexLevelSet {
  fn val(&self, xref: &[f64]) -> f64 {
    ConvexLevelSet::val(self, xref).unwrap_or(f64::NAN)
  }
  fn grad(&self, xref: &[f64]) -> (f64, Vec<f64>) {
    (
      ConvexLevelSet::val(self, xref).unwrap_or(f64::NAN),
      ConvexLevelSet::grad(self, xref).unwrap_or_else(|_| vec![f64::NAN; xref.len()]),
    )
  }
  fn hess(&self, xref: &[f64]) -> Vec<f64> {
    ConvexLevelSet::hess(self, xref).unwrap_or_else(|_| vec![f64::NAN; xref.len() * xref.len()])
  }
}
