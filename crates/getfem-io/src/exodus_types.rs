//! Exodus II element types (SEACAS Ioss): GetFEM's geometric
//! transformation, node order and side numbering.

use getfem_geom::geotrans::{PGeoTrans, geotrans_descriptor};

use crate::{Error, Result};

/// A reference shape: its Exodus names (without node count), its vertices
/// in Exodus order (coordinates in GetFEM's reference element), its sides
/// (0-based vertices).
pub(crate) struct Shape {
  pub names: &'static [&'static str],
  pub verts: &'static [&'static [f64]],
  pub sides: &'static [&'static [usize]],
}

const LINE: Shape = Shape {
  names: &["BAR", "BEAM", "TRUSS", "EDGE", "ROD"],
  verts: &[&[0.], &[1.]],
  sides: &[&[0], &[1]],
};
const TRI: Shape = Shape {
  names: &["TRI", "TRIANGLE", "TRISHELL"],
  verts: &[&[0., 0.], &[1., 0.], &[0., 1.]],
  sides: &[&[0, 1], &[1, 2], &[2, 0]],
};
const QUAD: Shape = Shape {
  names: &["QUAD", "SHELL"],
  verts: &[&[0., 0.], &[1., 0.], &[1., 1.], &[0., 1.]],
  sides: &[&[0, 1], &[1, 2], &[2, 3], &[3, 0]],
};
const TET: Shape = Shape {
  names: &["TET", "TETRA"],
  verts: &[&[0., 0., 0.], &[1., 0., 0.], &[0., 1., 0.], &[0., 0., 1.]],
  sides: &[&[0, 1, 3], &[1, 2, 3], &[0, 3, 2], &[0, 2, 1]],
};
const HEX: Shape = Shape {
  names: &["HEX", "HEXAHEDRON"],
  verts: &[
    &[0., 0., 0.],
    &[1., 0., 0.],
    &[1., 1., 0.],
    &[0., 1., 0.],
    &[0., 0., 1.],
    &[1., 0., 1.],
    &[1., 1., 1.],
    &[0., 1., 1.],
  ],
  sides: &[
    &[0, 1, 5, 4],
    &[1, 2, 6, 5],
    &[2, 3, 7, 6],
    &[0, 4, 7, 3],
    &[0, 3, 2, 1],
    &[4, 5, 6, 7],
  ],
};
const WEDGE: Shape = Shape {
  names: &["WEDGE"],
  verts: &[
    &[0., 0., 0.],
    &[1., 0., 0.],
    &[0., 1., 0.],
    &[0., 0., 1.],
    &[1., 0., 1.],
    &[0., 1., 1.],
  ],
  sides: &[
    &[0, 1, 4, 3],
    &[1, 2, 5, 4],
    &[0, 3, 5, 2],
    &[0, 2, 1],
    &[3, 4, 5],
  ],
};
const PYRAMID: Shape = Shape {
  names: &["PYRAMID", "PYRA"],
  verts: &[
    &[-1., -1., 0.],
    &[1., -1., 0.],
    &[1., 1., 0.],
    &[-1., 1., 0.],
    &[0., 0., 1.],
  ],
  sides: &[
    &[0, 1, 4],
    &[1, 2, 4],
    &[2, 3, 4],
    &[3, 0, 4],
    &[0, 3, 2, 1],
  ],
};

const QUAD_E: [&[usize]; 4] = [&[0, 1], &[1, 2], &[2, 3], &[3, 0]];
const HEX_E: [&[usize]; 12] = [
  &[0, 1],
  &[1, 2],
  &[2, 3],
  &[3, 0],
  &[0, 4],
  &[1, 5],
  &[2, 6],
  &[3, 7],
  &[4, 5],
  &[5, 6],
  &[6, 7],
  &[7, 4],
];
const WEDGE_E: [&[usize]; 9] = [
  &[0, 1],
  &[1, 2],
  &[2, 0],
  &[0, 3],
  &[1, 4],
  &[2, 5],
  &[3, 4],
  &[4, 5],
  &[5, 3],
];
const PYRAMID_E: [&[usize]; 8] = [
  &[0, 1],
  &[1, 2],
  &[2, 3],
  &[3, 0],
  &[0, 4],
  &[1, 4],
  &[2, 4],
  &[3, 4],
];

/// An element type: its name when written, shape, GetFEM's transformation,
/// the nodes after the vertices (each the centre of the vertices listed,
/// Ioss' edge and face tables).
pub(crate) struct ExoType {
  pub name: &'static str,
  pub shape: &'static Shape,
  pub gt: &'static str,
  pub extra: &'static [&'static [usize]],
}

pub(crate) const TYPES: &[ExoType] = &[
  ExoType {
    name: "BAR2",
    shape: &LINE,
    gt: "GT_PK(1,1)",
    extra: &[],
  },
  ExoType {
    name: "BAR3",
    shape: &LINE,
    gt: "GT_PK(1,2)",
    extra: &[&[0, 1]],
  },
  ExoType {
    name: "TRI3",
    shape: &TRI,
    gt: "GT_PK(2,1)",
    extra: &[],
  },
  ExoType {
    name: "TRI6",
    shape: &TRI,
    gt: "GT_PK(2,2)",
    extra: &[&[0, 1], &[1, 2], &[2, 0]],
  },
  ExoType {
    name: "QUAD4",
    shape: &QUAD,
    gt: "GT_QK(2,1)",
    extra: &[],
  },
  ExoType {
    name: "QUAD8",
    shape: &QUAD,
    gt: "GT_Q2_INCOMPLETE(2)",
    extra: &QUAD_E,
  },
  ExoType {
    name: "QUAD9",
    shape: &QUAD,
    gt: "GT_QK(2,2)",
    extra: &[&[0, 1], &[1, 2], &[2, 3], &[3, 0], &[0, 1, 2, 3]],
  },
  ExoType {
    name: "TETRA4",
    shape: &TET,
    gt: "GT_PK(3,1)",
    extra: &[],
  },
  ExoType {
    name: "TETRA10",
    shape: &TET,
    gt: "GT_PK(3,2)",
    extra: &[&[0, 1], &[1, 2], &[2, 0], &[0, 3], &[1, 3], &[2, 3]],
  },
  ExoType {
    name: "HEX8",
    shape: &HEX,
    gt: "GT_QK(3,1)",
    extra: &[],
  },
  ExoType {
    name: "HEX20",
    shape: &HEX,
    gt: "GT_Q2_INCOMPLETE(3)",
    extra: &HEX_E,
  },
  ExoType {
    name: "HEX27",
    shape: &HEX,
    gt: "GT_QK(3,2)",
    extra: &[
      &[0, 1],
      &[1, 2],
      &[2, 3],
      &[3, 0],
      &[0, 4],
      &[1, 5],
      &[2, 6],
      &[3, 7],
      &[4, 5],
      &[5, 6],
      &[6, 7],
      &[7, 4],
      &[0, 1, 2, 3, 4, 5, 6, 7],
      &[0, 3, 2, 1],
      &[4, 5, 6, 7],
      &[0, 4, 7, 3],
      &[1, 2, 6, 5],
      &[0, 1, 5, 4],
      &[2, 3, 7, 6],
    ],
  },
  ExoType {
    name: "WEDGE6",
    shape: &WEDGE,
    gt: "GT_PRISM(3,1)",
    extra: &[],
  },
  ExoType {
    name: "WEDGE15",
    shape: &WEDGE,
    gt: "GT_PRISM_INCOMPLETE_P2",
    extra: &WEDGE_E,
  },
  ExoType {
    name: "WEDGE18",
    shape: &WEDGE,
    gt: "GT_PRISM(3,2)",
    extra: &[
      &[0, 1],
      &[1, 2],
      &[2, 0],
      &[0, 3],
      &[1, 4],
      &[2, 5],
      &[3, 4],
      &[4, 5],
      &[5, 3],
      &[0, 1, 4, 3],
      &[1, 2, 5, 4],
      &[0, 3, 5, 2],
    ],
  },
  ExoType {
    name: "PYRAMID5",
    shape: &PYRAMID,
    gt: "GT_PYRAMID(1)",
    extra: &[],
  },
  ExoType {
    name: "PYRAMID13",
    shape: &PYRAMID,
    gt: "GT_PYRAMID_Q2_INCOMPLETE",
    extra: &PYRAMID_E,
  },
  ExoType {
    name: "PYRAMID14",
    shape: &PYRAMID,
    gt: "GT_PYRAMID(2)",
    extra: &[
      &[0, 1],
      &[1, 2],
      &[2, 3],
      &[3, 0],
      &[0, 4],
      &[1, 4],
      &[2, 4],
      &[3, 4],
      &[0, 1, 2, 3],
    ],
  },
];

impl ExoType {
  pub fn nb_nodes(&self) -> usize {
    self.shape.verts.len() + self.extra.len()
  }

  /// GetFEM's transformation and perm: GetFEM node i = Exodus node perm[i]
  /// (matched by reference coordinates).
  pub fn transformation(&self) -> Result<(PGeoTrans, Vec<usize>)> {
    let gt = geotrans_descriptor(self.gt).map_err(|e| Error::Format(e.to_string()))?;
    let v = self.shape.verts;
    let centre = |s: &[usize]| -> Vec<f64> {
      (0..v[0].len())
        .map(|k| s.iter().map(|&i| v[i][k]).sum::<f64>() / s.len() as f64)
        .collect()
    };
    let exo: Vec<Vec<f64>> = v
      .iter()
      .map(|x| x.to_vec())
      .chain(self.extra.iter().map(|s| centre(s)))
      .collect();
    let perm: Option<Vec<usize>> = gt
      .geometric_nodes()
      .iter()
      .map(|r| {
        exo
          .iter()
          .position(|x| x.iter().zip(r).all(|(a, b)| (a - b).abs() < 1e-12))
      })
      .collect();
    match perm {
      Some(p) if p.len() == exo.len() => Ok((gt, p)),
      _ => Err(Error::Format(format!("{}: no node match", self.name))),
    }
  }
}

impl ExoType {
  /// The Exodus side of the face whose vertices (GetFEM's local vertex
  /// numbers) are fv; sides numbered from 0.
  pub fn side_of_face(&self, fv: &[usize]) -> Option<usize> {
    let linear = TYPES
      .iter()
      .find(|t| t.shape.names[0] == self.shape.names[0] && t.extra.is_empty())?;
    let (_, perm) = linear.transformation().ok()?;
    // GetFEM vertex i = Exodus vertex perm[i]
    self
      .shape
      .sides
      .iter()
      .position(|s| s.len() == fv.len() && fv.iter().all(|&i| s.contains(&perm[i])))
  }
}

/// The type of a Lagrange element: basic structure of dimension dim with nv
/// vertices, nodes nodes.
pub(crate) fn lagrange_type(dim: usize, nv: usize, nodes: usize) -> Option<&'static ExoType> {
  TYPES
    .iter()
    .find(|t| t.shape.verts[0].len() == dim && t.shape.verts.len() == nv && t.nb_nodes() == nodes)
}

/// The type of an Exodus block: its element type name (any case, node count
/// ignored: given by nodes) and number of nodes.
pub(crate) fn exodus_type(elem_type: &str, nodes: usize) -> Option<&'static ExoType> {
  let family: String = elem_type
    .trim()
    .to_uppercase()
    .chars()
    .take_while(|c| c.is_ascii_alphabetic())
    .collect();
  TYPES
    .iter()
    .find(|t| t.shape.names.contains(&family.as_str()) && t.nb_nodes() == nodes)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn every_type_matches_its_transformation() {
    for t in TYPES {
      let (gt, p) = t.transformation().unwrap();
      let mut s = p.clone();
      s.sort_unstable();
      assert!(s.iter().copied().eq(0..t.nb_nodes()), "{}", t.name);
      assert_eq!(gt.nb_points(), t.nb_nodes(), "{}", t.name);
    }
    assert_eq!(exodus_type("hex", 27).unwrap().name, "HEX27");
    assert_eq!(exodus_type("SHELL4", 4).unwrap().name, "QUAD4");
    assert!(exodus_type("HEX", 9).is_none());
  }
}
