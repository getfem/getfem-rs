//! Point location in a mesh: r-tree of convex bounding boxes + inversion
//! (role of GetFEM's geotrans_inv / mesh_trans_inv).

use crate::Result;
use crate::geotrans::element_extent;
use crate::mesh::MeshData;
use crate::search::RTree;
use getfem_linalg::tensor::Mat;

pub struct Locator<'a> {
  mesh: &'a MeshData,
  tree: RTree,
  eps: f64,
  /// per affine convex, at its first use: the inverse map (GetFEM sets
  /// geotrans_inv_convex up once per convex in mesh_trans_inv)
  affine: Vec<std::sync::OnceLock<Option<Affine>>>,
}

/// xref = x0ref + B^T (x - g0); x(xref) - x = g0 + K (xref - x0ref) - x
struct Affine {
  x0ref: Vec<f64>,
  g0: Vec<f64>,
  k: Mat,
  b: Mat,
  len: f64,
}

impl<'a> Locator<'a> {
  /// eps: relative tolerance for "inside" (reference space) and the
  /// inversion residual (relative to the element extent).
  pub fn new(mesh: &'a MeshData, eps: f64) -> Result<Self> {
    let mut tree = RTree::new(mesh.dim(), 1e-10);
    for cv in mesh.convex_index() {
      let g = mesh.points_of_convex(cv);
      let (mut lo, mut hi) = (vec![f64::MAX; mesh.dim()], vec![f64::MIN; mesh.dim()]);
      for j in 0..g.ncols() {
        for (k, &v) in g.col(j).iter().enumerate() {
          lo[k] = lo[k].min(v);
          hi[k] = hi[k].max(v);
        }
      }
      // curved elements may bulge past their nodes
      let pad = if mesh.trans_of_convex(cv).is_linear() {
        0.0
      } else {
        0.1 * element_extent(&g)
      };
      lo.iter_mut().for_each(|v| *v -= pad);
      hi.iter_mut().for_each(|v| *v += pad);
      tree.add_box(&lo, &hi, cv)?;
    }
    let affine = (0..mesh.nb_convexes_max())
      .map(|_| Default::default())
      .collect();
    Ok(Locator {
      mesh,
      tree,
      eps,
      affine,
    })
  }

  /// All (convex, reference point) containing x.
  pub fn find_all(&self, x: &[f64]) -> Result<Vec<(usize, Vec<f64>)>> {
    let mut out = vec![];
    for cv in self.tree.find_boxes_at_point(x)? {
      let gt = self.mesh.trans_of_convex(cv);
      let inv = gt.invert(&self.mesh.points_of_convex(cv), x, self.eps, false)?;
      if inv.converged && inv.inside {
        out.push((cv, inv.xref));
      }
    }
    Ok(out)
  }

  /// The lowest convex accepted by keep containing x (candidates inverted
  /// in increasing order, up to the first that contains x).
  pub fn find_first(
    &self,
    x: &[f64],
    keep: impl Fn(usize) -> bool,
  ) -> Result<Option<(usize, Vec<f64>)>> {
    let mut cvs = self.tree.find_boxes_at_point(x)?;
    cvs.retain(|&cv| keep(cv));
    cvs.sort_unstable();
    for cv in cvs {
      let gt = self.mesh.trans_of_convex(cv);
      if gt.is_linear()
        && let Some(a) = self.affine[cv]
          .get_or_init(|| affine(gt, &self.mesh.points_of_convex(cv)))
          .as_ref()
      {
        let d: Vec<f64> = x.iter().zip(&a.g0).map(|(p, q)| p - q).collect();
        let xref: Vec<f64> = (0..a.x0ref.len())
          .map(|j| a.x0ref[j] + (0..d.len()).map(|i| a.b[(i, j)] * d[i]).sum::<f64>())
          .collect();
        if gt.convex_ref().is_in(&xref)? > self.eps {
          continue;
        }
        // the real residual (codimension > 0: off the element)
        let r = (0..d.len())
          .map(|i| {
            let ki: f64 = (0..xref.len())
              .map(|j| a.k[(i, j)] * (xref[j] - a.x0ref[j]))
              .sum();
            (ki - d[i]).powi(2)
          })
          .sum::<f64>()
          .sqrt();
        if r <= 1e3 * self.eps * a.len {
          return Ok(Some((cv, xref)));
        }
        continue;
      }
      let inv = gt.invert(&self.mesh.points_of_convex(cv), x, self.eps, false)?;
      if inv.converged && inv.inside {
        return Ok(Some((cv, inv.xref)));
      }
    }
    Ok(None)
  }

  /// The lowest convex containing x.
  pub fn find(&self, x: &[f64]) -> Result<Option<(usize, Vec<f64>)>> {
    self.find_first(x, |_| true)
  }
}

/// the inverse map of an affine convex (None: degenerate)
fn affine(gt: &crate::geotrans::PGeoTrans, g: &Mat) -> Option<Affine> {
  let x0ref = gt.geometric_nodes()[0].clone();
  let (grel, _) = crate::geotrans::relative_to_first_node(g, &[]);
  let k = gt.k_rel(&x0ref, &grel);
  let (b, _) = crate::geotrans::pseudo_inverse(&k).ok()?;
  Some(Affine {
    g0: g.col(0).to_vec(),
    x0ref,
    k,
    b,
    len: element_extent(g),
  })
}
