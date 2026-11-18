//! HHO elements (port of GetFEM's hho_method): an element for the interior and
//! one per face. At a point of a face only the functions of that face are
//! nonzero ("faces first"), elsewhere only the interior ones. DOFs: the
//! interior ones first, then each face in order.

use crate::error::{Error, Result};
use crate::fem::{DofDesc, Fem, PFem};
use getfem_geom::convex_ref::PRef;
use getfem_geom::geotrans::{PGeoTrans, geotrans_descriptor, pseudo_inverse};
use getfem_linalg::tensor::{Mat, Tensor};

/// An affine face of the reference convex: x = org + K p (p on the face
/// element), p = B^T (x - org).
#[derive(Debug)]
struct Face {
  pf: PFem,
  org: Vec<f64>,
  k: Mat,
  b: Mat,
}

#[derive(Debug)]
pub struct HhoFem {
  name: String,
  interior: PFem,
  faces: Vec<Face>,
  dofs: Vec<DofDesc>,
  nodes: Vec<Vec<f64>>,
  faces_of_dof: Vec<Vec<usize>>,
  /// first DOF of each face
  first: Vec<usize>,
}

fn gerr(e: getfem_geom::Error) -> Error {
  Error::InvalidArgument(e.to_string())
}

/// the linear geometric transformation of a face structure
fn face_trans(cr: &PRef, f: usize) -> Result<PGeoTrans> {
  let n = cr.dim() - 1;
  let np = cr.structure().nb_points_of_face(f);
  let name = if np == n + 1 {
    format!("GT_PK({n},1)")
  } else if np == 1 << n {
    format!("GT_QK({n},1)")
  } else {
    return Err(Error::InvalidArgument(
      "FEM_HHO: faces are simplices or parallelepipeds".into(),
    ));
  };
  geotrans_descriptor(&name).map_err(gerr)
}

impl HhoFem {
  pub(crate) fn new(name: String, interior: PFem, face_fems: Vec<PFem>) -> Result<Self> {
    let cr = interior.ref_convex().clone();
    let nbf = cr.structure().nb_faces();
    if face_fems.len() != nbf {
      return Err(Error::InvalidArgument(format!(
        "{name}: a single element for all the faces or one per face"
      )));
    }
    if !interior.is_polynomial() || face_fems.iter().any(|f| !f.is_polynomial()) {
      return Err(Error::InvalidArgument(format!(
        "{name}: polynomial elements only"
      )));
    }
    let mut out = HhoFem {
      name,
      dofs: interior.dof_types().to_vec(),
      nodes: (0..interior.nb_dof())
        .map(|i| interior.node(i).to_vec())
        .collect(),
      faces_of_dof: vec![],
      first: vec![],
      faces: vec![],
      interior,
    };
    for (f, pf) in face_fems.into_iter().enumerate() {
      let ptsref: Vec<Vec<f64>> = cr
        .structure()
        .ind_points_of_face(f)
        .iter()
        .map(|&i| cr.points()[i].clone())
        .collect();
      let gt = face_trans(&cr, f)?;
      if pf.dim() + 1 != cr.dim() || gt.nb_points() != ptsref.len() {
        return Err(Error::InvalidArgument(format!(
          "{}: wrong face element",
          out.name
        )));
      }
      let n = cr.dim();
      let gref = Tensor::from_data(&[n, ptsref.len()], ptsref.concat()).map_err(Error::from)?;
      let x0ref = vec![0.0; n - 1];
      let pg = gt.at(&gref, &x0ref).map_err(gerr)?;
      let (b, _) = pseudo_inverse(&pg.k).map_err(gerr)?;
      let face = Face {
        org: pg.x,
        k: pg.k,
        b,
        pf,
      };
      out.first.push(out.dofs.len());
      for i in 0..face.pf.nb_dof() {
        out.dofs.push(face.pf.dof_types()[i].clone());
        out.nodes.push(face.to_elem(face.pf.node(i)));
      }
      out.faces.push(face);
    }
    // by position, as GetFEM's composite elements
    out.faces_of_dof = crate::fem::faces_of(&cr, &out.dofs, &out.nodes);
    Ok(out)
  }

  /// the face whose element contains xref (within 1e-9), and xref on it
  fn face_at(&self, xref: &[f64]) -> Result<Option<(usize, Vec<f64>)>> {
    for (f, face) in self.faces.iter().enumerate() {
      let ptref = face.to_face(xref);
      let back = face.to_elem(&ptref);
      let d = back
        .iter()
        .zip(xref)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
      if d < 1e-9 && face.pf.ref_convex().is_in(&ptref).map_err(gerr)? < 1e-9 {
        return Ok(Some((f, ptref)));
      }
    }
    Ok(None)
  }

  /// values of the interior or face functions at xref: face tables mapped by
  /// map(face, table) into the element's layout [nb_dof, ..rest]
  fn eval(
    &self,
    xref: &[f64],
    rest: usize,
    interior: impl Fn(&PFem, &[f64]) -> Result<Tensor<f64>>,
    face: impl Fn(&Face, &[f64]) -> Result<Vec<f64>>,
  ) -> Result<Tensor<f64>> {
    let nd = self.dofs.len();
    let mut out = vec![0.0; nd * rest];
    match self.face_at(xref)? {
      Some((f, ptref)) => {
        let fr = &self.faces[f];
        let (n0, nf) = (self.first[f], fr.pf.nb_dof());
        let t = face(fr, &ptref)?;
        for r in 0..rest {
          for i in 0..nf {
            out[n0 + i + nd * r] = t[i + nf * r];
          }
        }
      }
      None => {
        let t = interior(&self.interior, xref)?;
        let ni = self.interior.nb_dof();
        for r in 0..rest {
          for i in 0..ni {
            out[i + nd * r] = t.data()[i + ni * r];
          }
        }
      }
    }
    Tensor::from_data(&[nd, 1, rest], out).map_err(Error::from)
  }
}

impl Face {
  fn to_elem(&self, ptref: &[f64]) -> Vec<f64> {
    (0..self.org.len())
      .map(|r| {
        self.org[r]
          + (0..ptref.len())
            .map(|j| self.k[(r, j)] * ptref[j])
            .sum::<f64>()
      })
      .collect()
  }
  fn to_face(&self, xref: &[f64]) -> Vec<f64> {
    (0..self.b.ncols())
      .map(|j| {
        (0..xref.len())
          .map(|r| self.b[(r, j)] * (xref[r] - self.org[r]))
          .sum()
      })
      .collect()
  }
}

impl Fem for HhoFem {
  fn name(&self) -> &str {
    &self.name
  }
  fn ref_convex(&self) -> &PRef {
    self.interior.ref_convex()
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
    &self.nodes[i]
  }
  fn faces_of_dof(&self, i: usize) -> &[usize] {
    &self.faces_of_dof[i]
  }
  fn is_equivalent(&self) -> bool {
    self.interior.is_equivalent() && self.faces.iter().all(|f| f.pf.is_equivalent())
  }
  fn is_lagrange(&self) -> bool {
    self.interior.is_lagrange() && self.faces.iter().all(|f| f.pf.is_lagrange())
  }
  fn is_polynomial(&self) -> bool {
    false
  }
  fn estimated_degree(&self) -> usize {
    self
      .faces
      .iter()
      .map(|f| f.pf.estimated_degree())
      .fold(self.interior.estimated_degree(), usize::max)
  }
  fn base_value(&self, xref: &[f64]) -> Result<Tensor<f64>> {
    let t = self.eval(
      xref,
      1,
      |pf, xref| pf.base_value(xref),
      |f, ptref| Ok(f.pf.base_value(ptref)?.data().to_vec()),
    )?;
    Tensor::from_data(&[self.nb_dof(), 1], t.data().to_vec()).map_err(Error::from)
  }
  fn grad_base_value(&self, xref: &[f64]) -> Result<Tensor<f64>> {
    let n = xref.len();
    // d/dxref = B d/dptref (tangential)
    self.eval(
      xref,
      n,
      |pf, xref| pf.grad_base_value(xref),
      |f, ptref| {
        let g = f.pf.grad_base_value(ptref)?;
        let nf = f.pf.nb_dof();
        let m = ptref.len();
        let mut out = vec![0.0; nf * n];
        for i in 0..nf {
          for r in 0..n {
            out[i + nf * r] = (0..m).map(|j| f.b[(r, j)] * g.data()[i + nf * j]).sum();
          }
        }
        Ok(out)
      },
    )
  }
  fn hho_interior(&self) -> Option<PFem> {
    Some(self.interior.clone())
  }
  fn hess_base_value(&self, xref: &[f64]) -> Result<Tensor<f64>> {
    let n = xref.len();
    self.eval(
      xref,
      n * n,
      |pf, xref| pf.hess_base_value(xref),
      |f, ptref| {
        let h = f.pf.hess_base_value(ptref)?;
        let nf = f.pf.nb_dof();
        let m = ptref.len();
        let mut out = vec![0.0; nf * n * n];
        for i in 0..nf {
          for r in 0..n {
            for s in 0..n {
              let mut v = 0.0;
              for j in 0..m {
                for k in 0..m {
                  v += f.b[(r, j)] * f.b[(s, k)] * h.data()[i + nf * (j + m * k)];
                }
              }
              out[i + nf * (r + n * s)] = v;
            }
          }
        }
        Ok(out)
      },
    )
  }
}
