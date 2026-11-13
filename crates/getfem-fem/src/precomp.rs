//! Basis values and gradients tabulated once per (element, point set) and
//! geometric transformation gradients per (transformation, point set).
//! Immutable and shared; keyed by names, no dependency tracking needed.

use std::sync::Arc;

use getfem_geom::geotrans::PGeoTrans;
use getfem_geom::intern::Interner;
use getfem_linalg::tensor::{Mat, Tensor};

use crate::Result;
use crate::fem::PFem;
use crate::integration::PIntegration;

/// Element tables at the points of an integration method (all points,
/// element then faces).
#[derive(Debug)]
pub struct FemPrecomp {
  pub val: Vec<Tensor<f64>>,
  pub grad: Vec<Tensor<f64>>,
}

/// Geometric transformation shape gradients at the same points.
#[derive(Debug)]
pub struct GeoPrecomp {
  pub val: Vec<Vec<f64>>,
  pub grad: Vec<Mat>,
}

static FEMP: Interner<(String, String), FemPrecomp> = Interner::new();
static GEOP: Interner<(String, String), GeoPrecomp> = Interner::new();

pub fn fem_precomp(pf: &PFem, pim: &PIntegration) -> Result<Arc<FemPrecomp>> {
  let ptsref = pim.approx()?.points();
  let build = || {
    let val = ptsref
      .iter()
      .map(|xref| pf.base_value(xref))
      .collect::<Result<_>>()?;
    let grad = ptsref
      .iter()
      .map(|xref| pf.grad_base_value(xref))
      .collect::<Result<_>>()?;
    Ok(FemPrecomp { val, grad })
  };
  if pim.is_transient() {
    let mut pfps = pim.fem_precomps.lock().expect("not poisoned");
    if let Some(p) = pfps.iter().find(|p| p.0 == pf.name()) {
      return Ok(p.1.clone());
    }
    let p = Arc::new(build()?);
    pfps.push((pf.name().to_string(), p.clone()));
    return Ok(p);
  }
  FEMP.get_or_try(&(pf.name().to_string(), pim.name().to_string()), build)
}

pub fn geo_precomp(pgt: &PGeoTrans, pim: &PIntegration) -> Result<Arc<GeoPrecomp>> {
  let ptsref = pim.approx()?.points();
  let build = || {
    Ok(GeoPrecomp {
      val: ptsref.iter().map(|xref| pgt.val(xref)).collect(),
      grad: ptsref.iter().map(|xref| pgt.grad(xref)).collect(),
    })
  };
  if pim.is_transient() {
    return build().map(Arc::new);
  }
  GEOP.get_or_try(&(pgt.name(), pim.name().to_string()), build)
}
