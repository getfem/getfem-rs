//! gmsh import: msh4 entities without a physical group, errors.

use std::collections::BTreeMap;

use getfem_io::gmsh::{GmshOptions, import_from_gmsh};

#[test]
fn errors() {
  let o = GmshOptions::default();
  assert!(
    getfem_io::gmsh::parse_gmsh("$MeshFormat 4.1 1 8 $EndMeshFormat", "x", &o)
      .unwrap_err()
      .to_string()
      .contains("binary")
  );
  assert!(getfem_io::gmsh::parse_gmsh("$MeshFormat 3 0 8 $EndMeshFormat", "x", &o).is_err());
  assert!(getfem_io::gmsh::parse_gmsh("garbage", "x", &o).is_err());
  let bad_node = "$MeshFormat 2.2 0 8 $EndMeshFormat $Nodes 1 1 0 0 0 $EndNodes $Elements 1 1 1 1 7 1 2 $EndElements";
  assert!(
    getfem_io::gmsh::parse_gmsh(bad_node, "x", &o)
      .unwrap_err()
      .to_string()
      .contains("unknown node")
  );
  assert!(import_from_gmsh("/nonexistent.msh", &o).is_err());
}

/// msh 4.1: two triangles (surface 1), a bottom line (curve 5) and a top
/// line (curve 6); with `phys`, physical groups 11 (surface) and 21 (curve
/// 5) and none on curve 6
fn square_msh4(phys: bool) -> String {
  let (s, c5) = if phys { ("1 11", "1 21") } else { ("0", "0") };
  format!(
    "$MeshFormat 4.1 0 8 $EndMeshFormat \
     $Entities 0 2 1 0 \
     5 0 0 0 1 0 0 {c5} 0 \
     6 0 1 0 1 1 0 0 0 \
     1 0 0 0 1 1 0 {s} 0 \
     $EndEntities \
     $Nodes 1 4 1 4 2 1 0 4 1 2 3 4 0 0 0 1 0 0 1 1 0 0 1 0 $EndNodes \
     $Elements 3 4 1 4 \
     2 1 2 2 1 1 2 3 2 1 3 4 \
     1 5 1 1 3 1 2 \
     1 6 1 1 4 3 4 \
     $EndElements"
  )
}

/// (number of convexes, number of faces) per region
fn region_counts(text: &str) -> BTreeMap<usize, (usize, usize)> {
  let g = getfem_io::gmsh::parse_gmsh(text, "square", &GmshOptions::default()).unwrap();
  let m = &g.mesh;
  m.region_ids()
    .into_iter()
    .map(|id| {
      let r = m.region_set(id).unwrap();
      let cvs = r.iter().filter(|(_, f)| f.is_none()).count();
      (id, (cvs, r.iter().count() - cvs))
    })
    .collect()
}

#[test]
fn msh4_entities_without_physical_group() {
  // physical groups in the file: an entity without one goes to region 0,
  // as in msh 2.2 (GetFEM)
  let r = region_counts(&square_msh4(true));
  assert_eq!(r, BTreeMap::from([(0, (0, 1)), (11, (2, 0)), (21, (0, 1))]));
  // no physical group at all: the entity tags
  let r = region_counts(&square_msh4(false));
  assert_eq!(r, BTreeMap::from([(1, (2, 0)), (5, (0, 1)), (6, (0, 1))]));
}

#[test]
fn file_nodes_kept() {
  // two triangles along the diagonal of the unit square, nodes 5, 6 at the
  // positions of 1, 3 (a crack), node 7 unused
  let text = "$MeshFormat 2.2 0 8 $EndMeshFormat \
     $Nodes 7 1 0 0 0 2 1 0 0 3 1 1 0 4 0 1 0 5 0 0 0 6 1 1 0 7 2 2 0 $EndNodes \
     $Elements 2 1 2 2 1 1 1 2 3 2 2 2 1 1 5 6 4 $EndElements";
  let g = getfem_io::gmsh::parse_gmsh(text, "crack", &GmshOptions::default()).unwrap();
  let m = &g.mesh;
  assert_eq!(m.dim(), 3, "3D, as the file");
  assert_eq!(m.nb_points(), 6, "no merging, node 7 skipped");
  // the crack stays open: no face of triangle 0 has a neighbor
  assert!((0..3).all(|f| m.neighbors_of_convex(0, f).is_empty()));
}
