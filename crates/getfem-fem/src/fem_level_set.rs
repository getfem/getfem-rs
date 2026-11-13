//! Enrichment by the zones of level sets (GetFEM's getfem_fem_level_set,
//! getfem_mesh_fem_level_set): a basic DOF whose support meets several
//! zones is replaced by one DOF per zone, each the base function on its
//! zone and 0 elsewhere (the Heaviside enrichment of XFEM).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use getfem_geom::convex_ref::PRef;
use getfem_linalg::tensor::Tensor;

use crate::fem::{DofDesc, Fem, FemCtx, PFem, RealValues, real_base_values_at, reserve_xfem_index};
use crate::level_set::ConvexLevelSet;
use crate::mesh_fem::MeshFem;
use crate::mesh_level_set::{MeshLevelSet, Zone, ZoneSet, merge_zoneset, merge_zonesets};
use crate::{Error, Result};

/// The element of an enriched convex (GetFEM fem_level_set).
#[derive(Debug)]
struct FemLevelSet {
  name: String,
  bfem: PFem,
  /// per DOF of bfem: its zones if enriched
  dofzones: Vec<Option<Arc<Vec<Zone>>>>,
  /// the level sets whose sign differs between the zones, on this convex,
  /// with their ids
  ls: Vec<(usize, ConvexLevelSet, usize)>,
  /// per level set: its sign common to all zones ('*' if none)
  common: Vec<u8>,
  dofs: Vec<DofDesc>,
  /// per DOF: the DOF of bfem it copies
  base_of: Vec<usize>,
  /// the zone ids (GetFEM find_zone_id) by the signs of the level sets at
  /// the point: few per element
  zone_ids: std::sync::Mutex<ZoneIds>,
}

/// per signs of the level sets: the zone ids of the enriched DOFs
type ZoneIds = Vec<(Vec<u8>, Arc<Vec<bool>>)>;

/// GetFEM are_zones_compatible_: a is compatible with the sub-zone b
fn compatible(a: &[u8], b: &[u8]) -> bool {
  a.len() == b.len() && a.iter().zip(b).all(|(x, y)| *x == b'0' || x == y)
}

impl FemLevelSet {
  fn new(
    bfem: PFem,
    dofzones: Vec<Option<Arc<Vec<Zone>>>>,
    mls: &MeshLevelSet,
    cv: usize,
    xfem_index: usize,
  ) -> Result<Self> {
    let nls = mls.nb_level_sets();
    let mut ls = vec![];
    let mut common = vec![b'*'; nls];
    for (i, c) in common.iter_mut().enumerate() {
      let mut differs = false;
      for z in dofzones.iter().flatten().flat_map(|zs| zs.iter()) {
        for sz in z {
          let d = sz.as_bytes()[i];
          if *c == b'*' {
            *c = d;
          } else if *c != d {
            differs = true;
            break;
          }
        }
      }
      if differs {
        ls.push((i, mls.level_set(i).of_convex(cv, 0)?, mls.level_set(i).id()));
      }
    }
    let (mut dofs, mut base_of) = (vec![], vec![]);
    for (k, zs) in dofzones.iter().enumerate() {
      let d = &bfem.dof_types()[k];
      match zs {
        None => {
          dofs.push(d.clone());
          base_of.push(k);
        }
        Some(zs) => {
          for j in 0..zs.len() {
            dofs.push(d.xfem_dof(j + xfem_index));
            base_of.push(k);
          }
        }
      }
    }
    Ok(FemLevelSet {
      name: format!("FEM_LEVEL_SET({})", bfem.name()),
      bfem,
      dofzones,
      ls,
      common,
      dofs,
      base_of,
      zone_ids: Default::default(),
    })
  }

  /// per enriched DOF: whether its zone contains the point (GetFEM
  /// find_zone_id); the level set the point is on takes the side of c
  fn zone_ids(&self, c: &FemCtx) -> Result<Arc<Vec<bool>>> {
    let mut z = self.common.clone();
    for (i, l, id) in &self.ls {
      let positive = match c.side_on(*id) {
        0 => l.val(c.xref)? > 0.0,
        side => side > 0,
      };
      z[*i] = if positive { b'+' } else { b'-' };
    }
    let mut cache = self.zone_ids.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(m) = cache.iter().find(|m| m.0 == z) {
      return Ok(m.1.clone());
    }
    let m: Arc<Vec<bool>> = Arc::new(
      self
        .dofzones
        .iter()
        .flatten()
        .flat_map(|zs| zs.iter())
        .map(|zone| zone.iter().any(|sz| compatible(&z, sz.as_bytes())))
        .collect(),
    );
    cache.push((z, m.clone()));
    Ok(m)
  }

  /// the base tensor t (nb_base x cols) expanded to the DOFs, the copies
  /// of an enriched DOF zero outside their zone
  fn expand(&self, t: &[f64], ids: &[bool]) -> Vec<f64> {
    let nb = self.bfem.nb_dof();
    let cols = t.len().checked_div(nb).unwrap_or(0);
    let mut out = Vec::with_capacity(self.dofs.len() * cols);
    for c in 0..cols {
      let mut cnt = 0;
      for (d, zs) in self.dofzones.iter().enumerate() {
        let v = t[d + nb * c];
        match zs {
          None => out.push(v),
          Some(zs) => {
            for _ in 0..zs.len() {
              out.push(if ids[cnt] { v } else { 0.0 });
              cnt += 1;
            }
          }
        }
      }
    }
    out
  }

  fn real_only(&self) -> Error {
    Error::InvalidArgument(format!("{}: no base values, real only element", self.name))
  }
}

impl Fem for FemLevelSet {
  fn name(&self) -> &str {
    &self.name
  }
  fn ref_convex(&self) -> &PRef {
    self.bfem.ref_convex()
  }
  fn target_dim(&self) -> usize {
    self.bfem.target_dim()
  }
  fn nb_dof(&self) -> usize {
    self.dofs.len()
  }
  fn dof_types(&self) -> &[DofDesc] {
    &self.dofs
  }
  fn node(&self, i: usize) -> &[f64] {
    self.bfem.node(self.base_of[i])
  }
  fn faces_of_dof(&self, i: usize) -> &[usize] {
    self.bfem.faces_of_dof(self.base_of[i])
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
    let (val, grad, hess) = real_base_values_at(&self.bfem, c, der)?;
    let ids = self.zone_ids(c)?;
    Ok((
      self.expand(&val, &ids),
      self.expand(&grad, &ids),
      self.expand(&hess, &ids),
    ))
  }
}

impl MeshFem {
  /// The space mf enriched by the zones of the level sets of mls (GetFEM
  /// mesh_fem_level_set): a basic DOF of mf whose support touches a cut
  /// convex and meets several zones becomes one DOF per zone. mf of qdim 1,
  /// not reduced; built from the current cut: build it again after it
  /// changes.
  pub fn from_level_set(mls: &MeshLevelSet, mf: &MeshFem) -> Result<MeshFem> {
    if mf.qdim() != 1 {
      return Err(Error::InvalidArgument(
        "base mesh_fem for mesh_fem_level_set has to be of qdim one for the moment ...".into(),
      ));
    }
    if mf.is_reduced() {
      return Err(Error::InvalidArgument(
        "Mesh fem level set not defined for reduced mesh fems".into(),
      ));
    }
    let xfem_index = reserve_xfem_index();
    let cvs = mf.convex_index()?;
    let mut dof_cvs: Vec<Vec<usize>> = vec![vec![]; mf.nb_basic_dof()?];
    for &cv in &cvs {
      for d in mf.ind_basic_dof_of_element(cv)? {
        dof_cvs[d].push(cv);
      }
    }
    // the enrichments, shared by the DOFs with equal zone sets
    let mut shared: BTreeMap<ZoneSet, Arc<Vec<Zone>>> = BTreeMap::new();
    let mut dof_enrichment: Vec<Option<Arc<Vec<Zone>>>> = vec![None; dof_cvs.len()];
    let mut enriched_elements = BTreeSet::new();
    for (d, ct) in dof_cvs.iter().enumerate() {
      if !ct.iter().any(|&cv| mls.is_convex_cut(cv)) {
        continue;
      }
      let mut zones = ZoneSet::new();
      for &cv in ct {
        if mls.is_convex_cut(cv) {
          merge_zonesets(&mut zones, mls.zoneset_of_convex(cv)?);
        } else {
          merge_zoneset(&mut zones, mls.primary_zone_of_convex(cv)?);
        }
      }
      if zones.len() != 1 {
        let e = shared
          .entry(zones)
          .or_insert_with_key(|z| Arc::new(z.iter().cloned().collect()))
          .clone();
        dof_enrichment[d] = Some(e);
        enriched_elements.extend(ct.iter().copied());
      }
    }
    let out = MeshFem::new(&mf.linked_mesh(), 1);
    for &cv in &cvs {
      let Some(pf) = mf.fem_of_element(cv)? else {
        continue;
      };
      if enriched_elements.contains(&cv) {
        let dz = mf
          .ind_basic_dof_of_element(cv)?
          .iter()
          .map(|&d| dof_enrichment[d].clone())
          .collect();
        let pf: PFem = Arc::new(FemLevelSet::new(pf, dz, mls, cv, xfem_index)?);
        out.set_finite_element(([cv], &pf))?;
      } else {
        out.set_finite_element(([cv], &pf))?;
      }
    }
    Ok(out)
  }
}
