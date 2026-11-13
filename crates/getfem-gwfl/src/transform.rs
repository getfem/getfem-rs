//! Interpolate transformations (port of getfem_generic_assembly_interpolation):
//! the target element and reference point of each source point.

use std::collections::BTreeMap;
use std::sync::Arc;

use getfem_geom::mesh::{Mesh, MeshData};
use getfem_geom::search::RTree;

use crate::defs::Defs;
use crate::semantic::{Ctx, analyse_str};
use crate::{Error, Result};

/// An interpolate transformation, as registered in a workspace.
#[derive(Clone, Debug)]
pub enum InterpolateTransformation {
  /// the target point is an expression of the source point (GetFEM
  /// add_interpolate_transformation_from_expression)
  Expression {
    source: Mesh,
    target: Mesh,
    expr: String,
  },
  /// the neighbor element across an inner face (GetFEM neighbor_element)
  Neighbor,
  /// raytracing or projection onto contact boundaries and obstacles
  Contact(crate::contact::Contact),
}

/// A transformation ready for one assembly (one per transformation: size
/// does not matter).
#[allow(clippy::large_enum_variant)]
pub(crate) enum InitializedTransformation {
  Expression {
    source_geometry: u64,
    target: Arc<MeshData>,
    boxes: Boxes,
  },
  Neighbor,
  Contact(Box<crate::contact::InitializedContact>),
}

/// The analysed expression of a transformation from expression.
pub(crate) fn analyse_expression(
  expr: &str,
  source_dim: usize,
  defs: &Defs,
) -> Result<crate::semantic::Tree> {
  let ctx = Ctx {
    meshdim: source_dim,
    ref_elt_dim: source_dim,
    option: 1,
    ..Ctx::default()
  };
  Ok(analyse_str(expr, defs, &ctx)?.0)
}

impl InitializedTransformation {
  /// the initialized transformation of t (its expression is checked,
  /// compiled with the assembly)
  pub(crate) fn new(t: &InterpolateTransformation, defs: &Defs) -> Result<Self> {
    match t {
      InterpolateTransformation::Neighbor => Ok(InitializedTransformation::Neighbor),
      InterpolateTransformation::Contact(_) => Err(Error::Internal(
        "contact transformation initialized apart".into(),
      )),
      InterpolateTransformation::Expression {
        source,
        target,
        expr,
      } => {
        analyse_expression(expr, source.dim(), defs)?;
        let target = target.snapshot();
        Ok(InitializedTransformation::Expression {
          source_geometry: source.snapshot().geometry_version(),
          boxes: Boxes::new(&target)?,
          target,
        })
      }
    }
  }
}

/// Bounding boxes of the target convexes, enlarged as GetFEM does (by 0.2
/// times the largest extent, times 1e-4 for linear elements).
pub(crate) struct Boxes {
  tree: RTree,
  bounds: BTreeMap<usize, (Vec<f64>, Vec<f64>)>,
}

impl Boxes {
  fn new(m: &MeshData) -> Result<Self> {
    let n = m.dim();
    let mut tree = RTree::new(n, 0.0);
    let mut bounds = BTreeMap::new();
    for cv in m.convex_index() {
      let pts = m.ind_points_of_convex(cv);
      let (mut lo, mut hi) = (m.point(pts[0]).to_vec(), m.point(pts[0]).to_vec());
      for &p in &pts[1..] {
        for (k, &x) in m.point(p).iter().enumerate() {
          lo[k] = lo[k].min(x);
          hi[k] = hi[k].max(x);
        }
      }
      let mut h = (0..n).map(|k| hi[k] - lo[k]).fold(0.0, f64::max);
      if m.trans_of_convex(cv).is_linear() {
        h *= 1e-4;
      }
      lo.iter_mut().for_each(|v| *v -= h * 0.2);
      hi.iter_mut().for_each(|v| *v += h * 0.2);
      tree.add_box(&lo, &hi, cv)?;
      bounds.insert(cv, (lo, hi));
    }
    Ok(Boxes { tree, bounds })
  }

  /// The target convex and reference point of p (GetFEM
  /// interpolate_transformation_expression::transform): boxes by
  /// decreasing rating, inversion tolerance 1e-4, else the nearest convex
  /// within 5e-3 in reference coordinates.
  pub(crate) fn locate(&self, m: &MeshData, p: &[f64]) -> Result<Option<(usize, Vec<f64>)>> {
    let mut rated: Vec<(f64, usize)> = self
      .tree
      .find_boxes_at_point(p)?
      .into_iter()
      .map(|cv| {
        let (lo, hi) = &self.bounds[&cv];
        let mut r: f64 = 1.0;
        for i in 0..p.len() {
          let h = hi[i] - lo[i];
          if h > 0.0 {
            r = r.min((hi[i] - p[i]).min(p[i] - lo[i]) / h);
          }
        }
        (r, cv)
      })
      .collect();
    rated.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut best: Option<(f64, usize, Vec<f64>)> = None;
    for &(_, cv) in rated.iter().rev() {
      let gt = m.trans_of_convex(cv);
      let inv = gt.invert(&m.points_of_convex(cv), p, 1e-4, false)?;
      if !inv.converged {
        continue;
      }
      if inv.inside {
        return Ok(Some((cv, inv.xref)));
      }
      let d = gt.convex_ref().is_in(&inv.xref)?;
      if best.as_ref().is_none_or(|b| d < b.0) {
        best = Some((d, cv, inv.xref));
      }
    }
    Ok(best.filter(|b| b.0 < 5e-3).map(|b| (b.1, b.2)))
  }
}

/// Error for a transformation used from another mesh.
pub(crate) fn wrong_mesh(name: &str) -> Error {
  Error::InvalidArgument(format!(
    "transformation {name} is used on another mesh than its source mesh"
  ))
}
