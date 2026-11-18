//! Data stored at integration points (port of getfem_im_data): one tensor
//! per point of the elements of a region, and of its faces. A shared
//! handle (SPEC 3): the
//! points follow the integration methods and the mesh; values are tied to
//! the value stamp, which changes with the methods or the geometry.

use std::sync::{Arc, Mutex};

use getfem_geom::mesh::Mesh;
use getfem_geom::region::{Region, RegionArg};
use getfem_geom::shared::Shared;

use crate::Result;
use crate::mesh_fem::new_stamp;
use crate::mesh_im::MeshIm;

/// The points of an im_data for one version of the integration methods.
#[derive(Debug)]
pub struct ImdPoints {
  mim_version: u64,
  /// regions of the mesh
  mesh_version: u64,
  /// first point index of each convex (usize::MAX if not in the region)
  first: Vec<usize>,
  /// per convex, its faces in the region: (first point of the face in
  /// the method, number of points, first point index)
  faces: Vec<Vec<(usize, usize, usize)>>,
  nb_points: usize,
  stamp: u64,
}

impl ImdPoints {
  pub fn nb_points(&self) -> usize {
    self.nb_points
  }

  /// Index of point i of convex cv, i in the numbering of its method
  /// (interior points, then the points of each face); None if the point is
  /// not in the region. Only the points of the region are numbered (no
  /// second, unfiltered numbering as in GetFEM: another ImData on the
  /// entire domain gives that).
  pub fn index_of_point(&self, cv: usize, i: usize) -> Option<usize> {
    if let Some(&(a, _, k)) = self
      .faces
      .get(cv)
      .and_then(|fs| fs.iter().find(|&&(a, n, _)| (a..a + n).contains(&i)))
    {
      return Some(k + i - a);
    }
    self
      .first
      .get(cv)
      .filter(|&&f| f != usize::MAX)
      .map(|f| f + i)
  }

  /// Changes with the integration methods or the geometry: values at the
  /// points are valid while it is unchanged.
  pub fn value_stamp(&self) -> u64 {
    self.stamp
  }
}

#[derive(Debug)]
struct Inner {
  mim: MeshIm,
  region: Option<Region>,
  shape: Vec<usize>,
  cache: Mutex<Option<Arc<ImdPoints>>>,
}

impl Clone for Inner {
  fn clone(&self) -> Self {
    Inner {
      mim: self.mim.clone(),
      region: self.region.clone(),
      shape: self.shape.clone(),
      cache: Mutex::new(self.cache.lock().unwrap_or_else(|e| e.into_inner()).clone()),
    }
  }
}

/// Data at integration points, as a shared handle.
#[derive(Clone, Debug)]
pub struct ImData(Shared<Inner>);

impl ImData {
  /// Points of the convexes of `region` and of its faces (the entire
  /// domain or a stored region, followed; all convexes with a method if
  /// None).
  pub fn new(mim: &MeshIm, region: impl Into<RegionArg>, shape: &[usize]) -> Result<Self> {
    let region = region.into();
    let region = if region.is_all() {
      None
    } else {
      let r = region.on(&mim.linked_mesh())?;
      r.lasting("ImData")?;
      Some(r)
    };
    let region = region.as_ref();
    let d = ImData(Shared::new(Inner {
      mim: mim.clone(),
      region: region.cloned(),
      shape: shape.to_vec(),
      cache: Mutex::new(None),
    }));
    d.points()?;
    Ok(d)
  }

  /// Same im_data (not only equal)?
  pub fn same(&self, other: &ImData) -> bool {
    self.0.same(&other.0)
  }

  /// The points for the current integration methods.
  pub fn points(&self) -> Result<Arc<ImdPoints>> {
    let s = self.0.snapshot();
    let ims = s.mim.ims()?;
    let mut cache = s.cache.lock().unwrap_or_else(|e| e.into_inner());
    let mesh_version = s.mim.linked_mesh().version();
    if let Some(p) = cache.as_ref()
      && p.mim_version == ims.version()
      && p.mesh_version == mesh_version
    {
      return Ok(p.clone());
    }
    let region = match &s.region {
      Some(r) => Some(r.resolve(ims.linked_mesh())?),
      None => None,
    };
    // the interior points of the convexes of the region first, then the points
    // of its faces
    let nbcv = ims.linked_mesh().nb_convexes_max();
    let (mut first, mut faces) = (vec![usize::MAX; nbcv], vec![vec![]; nbcv]);
    let mut nb = 0;
    let has_convexes = region.as_ref().is_none_or(|r| !r.is_only_faces());
    if has_convexes {
      for cv in ims.convex_index() {
        if region.as_ref().is_none_or(|r| r.is_in(cv)) {
          first[cv] = nb;
          nb += ims
            .int_method_of_element(cv)?
            .approx()?
            .nb_points_on_convex();
        }
      }
    }
    if let Some(r) = region.as_ref().filter(|r| !r.is_only_convexes()) {
      for cv in ims.convex_index() {
        let ai = ims.int_method_of_element(cv)?.approx()?;
        for f in (0..ai.nb_convex_faces()).filter(|&f| r.is_in_face(cv, f)) {
          let n = ai.nb_points_on_face(f);
          faces[cv].push((ai.ind_first_point_on_face(f), n, nb));
          nb += n;
        }
      }
    }
    // new methods, geometry or points: values invalid
    let stamp = match cache.as_ref() {
      Some(o) if o.mim_version == ims.version() && o.first == first && o.faces == faces => o.stamp,
      _ => new_stamp(),
    };
    let p = Arc::new(ImdPoints {
      mim_version: ims.version(),
      mesh_version,
      first,
      faces,
      nb_points: nb,
      stamp,
    });
    *cache = Some(p.clone());
    Ok(p)
  }

  pub fn nb_points(&self) -> Result<usize> {
    Ok(self.points()?.nb_points)
  }

  pub fn linked_mesh_im(&self) -> MeshIm {
    self.0.snapshot().mim.clone()
  }
  pub fn linked_mesh(&self) -> Mesh {
    self.linked_mesh_im().linked_mesh()
  }

  pub fn tensor_size(&self) -> usize {
    self.0.snapshot().shape.iter().product()
  }

  pub fn tensor_shape(&self) -> Vec<usize> {
    self.0.snapshot().shape.clone()
  }

  /// Number of values of a data vector on these points.
  pub fn nb_values(&self) -> Result<usize> {
    Ok(self.nb_points()? * self.tensor_size())
  }

  /// Changes with the integration methods or the geometry: values at the
  /// points are valid while it is unchanged.
  pub fn value_stamp(&self) -> Result<u64> {
    Ok(self.points()?.stamp)
  }
}
