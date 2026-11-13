//! Products of spaces (GetFEM's getfem_mesh_fem_product): the functions of
//! a partition of unity pu_mf at its DOFs pu_dofs times each function of
//! global_mf (XFEM's crack tip enrichment).

use std::collections::BTreeSet;
use std::sync::Arc;

use getfem_geom::convex_ref::PRef;
use getfem_linalg::tensor::Tensor;

use crate::fem::{DofDesc, Fem, FemCtx, PFem, RealValues, real_base_values_at, reserve_xfem_index};
use crate::mesh_fem::MeshFem;
use crate::{Error, Result};

/// The element of a convex with DOFs of pu_dofs (GetFEM fem_product).
#[derive(Debug)]
struct FemProduct {
  name: String,
  pfems: [PFem; 2],
  /// the local DOFs of the first element in pu_dofs
  pu_dofs: Vec<usize>,
  dofs: Vec<DofDesc>,
}

impl FemProduct {
  fn new(pf1: PFem, pf2: PFem, cv: usize, xfem_index: usize, pu_dofs: Vec<usize>) -> Result<Self> {
    if pf1.target_dim() != 1 {
      return Err(Error::NotImplemented("fem_product of a vector element"));
    }
    if pf2.target_dim() != 1 {
      return Err(Error::InvalidArgument(
        "The second finite element should be scalar".into(),
      ));
    }
    let mut dofs = vec![];
    for &i in &pu_dofs {
      for j in 0..pf2.nb_dof() {
        dofs.push(pf1.dof_types()[i].xfem_dof(xfem_index + j));
      }
    }
    Ok(FemProduct {
      name: format!("FEM_PRODUCT({},{},{cv})", pf1.name(), pf2.name()),
      pfems: [pf1, pf2],
      pu_dofs,
      dofs,
    })
  }

  fn real_only(&self) -> Error {
    Error::InvalidArgument(format!("{}: no base values, real only element", self.name))
  }
}

impl Fem for FemProduct {
  fn name(&self) -> &str {
    &self.name
  }
  fn ref_convex(&self) -> &PRef {
    self.pfems[0].ref_convex()
  }
  fn target_dim(&self) -> usize {
    1
  }
  fn nb_dof(&self) -> usize {
    self.dofs.len()
  }
  fn dof_types(&self) -> &[DofDesc] {
    &self.dofs
  }
  fn node(&self, i: usize) -> &[f64] {
    self.pfems[0].node(self.pu_dofs[i / self.pfems[1].nb_dof()])
  }
  fn faces_of_dof(&self, i: usize) -> &[usize] {
    self.pfems[0].faces_of_dof(self.pu_dofs[i / self.pfems[1].nb_dof()])
  }
  fn is_equivalent(&self) -> bool {
    true
  }
  fn is_lagrange(&self) -> bool {
    false
  }
  fn is_polynomial(&self) -> bool {
    false
  }
  fn estimated_degree(&self) -> usize {
    5
  }
  fn base_value(&self, _: &[f64]) -> Result<Tensor<f64>> {
    Err(self.real_only())
  }
  fn grad_base_value(&self, _: &[f64]) -> Result<Tensor<f64>> {
    Err(self.real_only())
  }
  fn hess_base_value(&self, _: &[f64]) -> Result<Tensor<f64>> {
    Err(self.real_only())
  }
  fn real_element_defined(&self) -> bool {
    true
  }
  fn real_values(&self, c: &FemCtx, der: usize) -> Result<RealValues> {
    let (v0, g0, h0) = real_base_values_at(&self.pfems[0], c, der)?;
    let (v1, g1, h1) = real_base_values_at(&self.pfems[1], c, der)?;
    let (nb0, nb1, n) = (self.pfems[0].nb_dof(), self.pfems[1].nb_dof(), c.x.len());
    let nd = self.dofs.len();
    let val = self
      .pu_dofs
      .iter()
      .flat_map(|&i| v1.iter().map(|w| w * v0[i]).collect::<Vec<_>>())
      .collect();
    let mut grad = vec![];
    if der >= 1 {
      grad.reserve(nd * n);
      for k in 0..n {
        for &i in &self.pu_dofs {
          for j in 0..nb1 {
            grad.push(g0[i + nb0 * k] * v1[j] + g1[j + nb1 * k] * v0[i]);
          }
        }
      }
    }
    let mut hess = vec![];
    if der >= 2 {
      hess.reserve(nd * n * n);
      for s in 0..n {
        for r in 0..n {
          let rs = r + n * s;
          for &i in &self.pu_dofs {
            for j in 0..nb1 {
              hess.push(
                h0[i + nb0 * rs] * v1[j]
                  + h1[j + nb1 * rs] * v0[i]
                  + g0[i + nb0 * r] * g1[j + nb1 * s]
                  + g0[i + nb0 * s] * g1[j + nb1 * r],
              );
            }
          }
        }
      }
    }
    Ok((val, grad, hess))
  }
}

impl MeshFem {
  /// The product of the partition of unity pu_mf at its basic DOFs
  /// pu_dofs (all if None) and the global functions global_mf (GetFEM
  /// mesh_fem_product with set_enrichment; empty until then): on each
  /// convex with DOFs of pu_dofs, their functions times each function of
  /// global_mf; no function elsewhere. global_mf's functions joined across
  /// convexes by their local index: global functions only. Both of qdim 1,
  /// not reduced; built from their current state.
  pub fn from_product(
    pu_mf: &MeshFem,
    global_mf: &MeshFem,
    pu_dofs: Option<&BTreeSet<usize>>,
  ) -> Result<MeshFem> {
    if pu_mf.is_reduced() || global_mf.is_reduced() {
      return Err(Error::InvalidArgument(
        "Sorry, mesh_fem_product not defined for reduced mesh_fems".into(),
      ));
    }
    let xfem_index = reserve_xfem_index();
    let mesh = pu_mf.linked_mesh();
    let out = MeshFem::new(&mesh, 1);
    for cv in mesh.snapshot().convex_index() {
      let Some(pf1) = pu_mf.fem_of_element(cv)? else {
        continue;
      };
      let local: Vec<usize> = pu_mf
        .ind_basic_dof_of_element(cv)?
        .iter()
        .enumerate()
        .filter(|(_, d)| pu_dofs.is_none_or(|s| s.contains(d)))
        .map(|(i, _)| i)
        .collect();
      if local.is_empty() {
        continue;
      }
      let pf2 = global_mf.fem_of_element(cv)?.ok_or_else(|| {
        Error::InvalidArgument(format!(
          "mesh_fem_product: no element of global_mf on convex {cv}"
        ))
      })?;
      let pf: PFem = Arc::new(FemProduct::new(pf1, pf2, cv, xfem_index, local)?);
      out.set_finite_element(([cv], &pf))?;
    }
    Ok(out)
  }
}
