//! Sums of spaces (GetFEM's getfem_mesh_fem_sum): on each element the
//! functions of all the spaces. The smart global DOF linking of GetFEM is
//! not ported (crack.cc only; "does not work with discontinuous global
//! functions").

use std::sync::Arc;

use getfem_geom::convex_ref::PRef;
use getfem_linalg::tensor::Tensor;

use crate::fem::{DofDesc, Fem, FemCtx, PFem, RealValues, real_base_values_at};
use crate::mesh_fem::MeshFem;
use crate::{Error, Result};

/// The element of a convex where several spaces have functions.
#[derive(Debug)]
struct FemSum {
  name: String,
  pfems: Vec<PFem>,
  dofs: Vec<DofDesc>,
  /// per DOF: (part, its DOF)
  part_of: Vec<(usize, usize)>,
  /// per DOF: its global index (global DOFs of the parts), else MAX
  global: Vec<usize>,
}

impl FemSum {
  fn new(pfems: Vec<PFem>, cv: usize) -> Result<Self> {
    let (mut dofs, mut part_of, mut global) = (vec![], vec![], vec![]);
    for (k, pf) in pfems.iter().enumerate() {
      if pf.target_dim() != 1 {
        return Err(Error::InvalidArgument(
          "Vectorial fems not supported".into(),
        ));
      }
      dofs.extend_from_slice(pf.dof_types());
      part_of.extend((0..pf.nb_dof()).map(|i| (k, i)));
      match pf.global_dofs() {
        Some(g) => global.extend_from_slice(g),
        None => global.extend(std::iter::repeat_n(usize::MAX, pf.nb_dof())),
      }
    }
    let names: Vec<&str> = pfems.iter().map(|p| p.name()).collect();
    Ok(FemSum {
      name: format!("FEM_SUM({}, cv:{cv})", names.join(",")),
      pfems,
      dofs,
      part_of,
      global,
    })
  }

  fn real_only(&self) -> Error {
    Error::InvalidArgument(format!("{}: no base values, real only element", self.name))
  }
}

impl Fem for FemSum {
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
    let (k, j) = self.part_of[i];
    self.pfems[k].node(j)
  }
  fn faces_of_dof(&self, i: usize) -> &[usize] {
    let (k, j) = self.part_of[i];
    self.pfems[k].faces_of_dof(j)
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
  fn global_dofs(&self) -> Option<&[usize]> {
    Some(&self.global)
  }
  fn real_element_defined(&self) -> bool {
    true
  }
  fn real_values(&self, c: &FemCtx, der: usize) -> Result<RealValues> {
    let parts = self
      .pfems
      .iter()
      .map(|pf| real_base_values_at(pf, c, der))
      .collect::<Result<Vec<_>>>()?;
    // per column (component, derivative), the parts one after the other
    let concat = |get: &dyn Fn(&RealValues) -> &Vec<f64>| -> Vec<f64> {
      let nb: Vec<usize> = self.pfems.iter().map(|p| p.nb_dof()).collect();
      let cols = get(&parts[0]).len().checked_div(nb[0]).unwrap_or(0);
      let mut out = Vec::with_capacity(self.dofs.len() * cols);
      for col in 0..cols {
        for (p, &n) in parts.iter().zip(&nb) {
          out.extend_from_slice(&get(p)[n * col..n * (col + 1)]);
        }
      }
      out
    };
    Ok((concat(&|p| &p.0), concat(&|p| &p.1), concat(&|p| &p.2)))
  }
}

impl MeshFem {
  /// The sum of the spaces mfs: on each convex the functions of each space that
  /// has some there. The spaces of qdim 1, not reduced; built from their
  /// current state.
  pub fn from_sum(mfs: &[&MeshFem]) -> Result<MeshFem> {
    let Some(first) = mfs.first() else {
      return Err(Error::InvalidArgument("mesh_fem_sum: no mesh_fem".into()));
    };
    if mfs.iter().any(|mf| mf.is_reduced()) {
      return Err(Error::InvalidArgument(
        "Sorry fem_sum for reduced mesh_fem is not implemented".into(),
      ));
    }
    let mesh = first.linked_mesh();
    let out = MeshFem::new(&mesh, 1);
    for cv in mesh.snapshot().convex_index() {
      let mut pfems = vec![];
      for mf in mfs {
        if let Some(pf) = mf.fem_of_element(cv)?
          && pf.nb_dof() > 0
        {
          pfems.push(pf);
        }
      }
      match pfems.len() {
        0 => {}
        1 => out.set_finite_element(([cv], &pfems[0]))?,
        _ => {
          let pf: PFem = Arc::new(FemSum::new(pfems, cv)?);
          out.set_finite_element(([cv], &pf))?;
        }
      }
    }
    Ok(out)
  }
}
