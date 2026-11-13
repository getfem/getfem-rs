//! VTU export as GetFEM's vtk_export (getfem_export.h): `exporting` a
//! mesh, a space or a slice, then fields given on any space of the mesh
//! (`write_point_data`, interpolated onto what is exported) or per convex
//! (`write_cell_data`); `write` completes the file (ASCII, binary or
//! compressed). ParaView collections (.pvd) of such files by time.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use getfem_fem::fem::{PFem, classical_fem};
use getfem_fem::interpolation::interpolation;
use getfem_fem::mesh_fem::MeshFem;
use getfem_fem::slice::{SliceField, StoredMeshSlice};
use getfem_geom::geotrans::{PGeoTrans, geotrans_descriptor};
use getfem_geom::mesh::Mesh;

use crate::{Error, Result};

/// (transformations, VTK cell type, node order): vtk node j = getfem node perm[j].
/// Orders from GetFEM's getfem_export.cc.
const CELLS: &[(&[&str], u8, &[usize])] = &[
  (&["GT_PK(1,1)"], 3, &[0, 1]),
  (&["GT_PK(2,1)"], 5, &[0, 1, 2]),
  (&["GT_PK(3,1)"], 10, &[0, 1, 2, 3]),
  (&["GT_QK(2,1)", "GT_LINEAR_QK(2)"], 9, &[0, 1, 3, 2]),
  (
    &["GT_QK(3,1)", "GT_LINEAR_QK(3)"],
    12,
    &[0, 1, 3, 2, 4, 5, 7, 6],
  ),
  (
    &["GT_PRISM(3,1)", "GT_LINEAR_PRODUCT(GT_PK(2,1),GT_PK(1,1))"],
    13,
    &[0, 1, 2, 3, 4, 5],
  ),
  (&["GT_PYRAMID(1)"], 14, &[0, 1, 3, 2, 4]),
  (&["GT_PK(1,2)"], 21, &[0, 2, 1]),
  (&["GT_PK(2,2)"], 22, &[0, 2, 5, 1, 4, 3]),
  (&["GT_Q2_INCOMPLETE(2)"], 23, &[0, 2, 7, 5, 1, 4, 6, 3]),
  (&["GT_PK(3,2)"], 24, &[0, 2, 5, 9, 1, 4, 3, 6, 7, 8]),
  (
    &["GT_Q2_INCOMPLETE(3)"],
    25,
    &[
      0, 2, 7, 5, 12, 14, 19, 17, 1, 4, 6, 3, 13, 16, 18, 15, 8, 9, 11, 10,
    ],
  ),
  (
    &["GT_PRISM_INCOMPLETE_P2"],
    26,
    &[0, 2, 5, 9, 11, 14, 1, 4, 3, 10, 13, 12, 6, 7, 8],
  ),
  (
    &["GT_PYRAMID_Q2_INCOMPLETE"],
    27,
    &[0, 2, 7, 5, 12, 1, 4, 6, 3, 8, 9, 11, 10],
  ),
  (
    &["GT_PYRAMID(2)"],
    27,
    &[0, 2, 8, 6, 13, 1, 5, 7, 3, 9, 10, 12, 11],
  ),
  (&["GT_QK(2,2)"], 28, &[0, 2, 8, 6, 1, 5, 7, 3, 4]),
  // GetFEM's table misses the volume center 13
  (
    &["GT_QK(3,2)"],
    29,
    &[
      0, 2, 8, 6, 18, 20, 26, 24, 1, 5, 7, 3, 19, 23, 25, 21, 9, 11, 17, 15, 12, 14, 10, 16, 4, 22,
      13,
    ],
  ),
  (
    &["GT_PRISM(3,2)"],
    32,
    &[0, 2, 5, 12, 14, 17, 1, 4, 3, 13, 16, 15, 6, 8, 11, 7, 10, 9],
  ),
];

/// The VTK cell of a Lagrange element of nb_dof nodes on a convex of
/// dimension dim with nv vertices: the transformation of the same nodes in
/// CELLS (GetFEM's vtk mappings, one per node count).
fn lagrange_cell(dim: usize, nv: usize, nb_dof: usize) -> Option<(u8, &'static [usize])> {
  let gt = match (dim, nv, nb_dof) {
    (1, 2, 2) => "GT_PK(1,1)",
    (1, 2, 3) => "GT_PK(1,2)",
    (2, 3, 3) => "GT_PK(2,1)",
    (2, 3, 6) => "GT_PK(2,2)",
    (2, 4, 4) => "GT_QK(2,1)",
    (2, 4, 8) => "GT_Q2_INCOMPLETE(2)",
    (2, 4, 9) => "GT_QK(2,2)",
    (3, 4, 4) => "GT_PK(3,1)",
    (3, 4, 10) => "GT_PK(3,2)",
    (3, 8, 8) => "GT_QK(3,1)",
    (3, 8, 20) => "GT_Q2_INCOMPLETE(3)",
    (3, 8, 27) => "GT_QK(3,2)",
    (3, 6, 6) => "GT_PRISM(3,1)",
    (3, 6, 15) => "GT_PRISM_INCOMPLETE_P2",
    (3, 6, 18) => "GT_PRISM(3,2)",
    (3, 5, 5) => "GT_PYRAMID(1)",
    (3, 5, 13) => "GT_PYRAMID_Q2_INCOMPLETE",
    (3, 5, 14) => "GT_PYRAMID(2)",
    _ => return None,
  };
  CELLS.iter().find(|c| c.0.contains(&gt)).map(|c| (c.1, c.2))
}

/// The linear transformation of the basic structure of pgt (its classical
/// elements are the complete ones, GetFEM's classical_fem(.., complete)).
pub(crate) fn linear_of(pgt: &PGeoTrans) -> Result<PGeoTrans> {
  let b = pgt.basic_structure();
  let (n, nv) = (b.dim(), b.nb_points());
  let name = match (n, nv) {
    (_, _) if nv == n + 1 => format!("GT_PK({n},1)"),
    (_, _) if nv == 1 << n => format!("GT_QK({n},1)"),
    (3, 6) => "GT_PRISM(3,1)".to_string(),
    (3, 5) => "GT_PYRAMID(1)".to_string(),
    _ => {
      return Err(Error::Format(format!(
        "VTK export: no Lagrange element for {}",
        pgt.name()
      )));
    }
  };
  Ok(geotrans_descriptor(&name)?)
}

/// The space GetFEM exports for mf (vtk_export::exporting(mf)): per
/// element isoparametric Lagrange of degree 1, or 2 when the element is of
/// higher degree or the transformation is not linear; discontinuous where
/// the element is; the incomplete Q2, prism P2 and pyramid Q2 kept.
fn export_space(mf: &MeshFem) -> Result<MeshFem> {
  let num = mf.numbering()?;
  let mesh = mf.linked_mesh();
  let m = mesh.snapshot();
  let mut groups: BTreeMap<String, (PFem, Vec<usize>)> = BTreeMap::new();
  for cv in num.convex_index() {
    let pgt = m.trans_of_convex(cv);
    let pf = num.fem_of_element(cv).unwrap();
    let pe = if pf.name().contains("INCOMPLETE") {
      pf.clone()
    } else {
      let disc = pf.dof_types().iter().any(|d| !d.linkable);
      let p1 = classical_fem(pgt, 1, disc)?;
      let degree = if (pf.name() != p1.name() && pf.estimated_degree() > 1)
        || **pgt.structure() != *pgt.basic_structure()
      {
        2
      } else {
        1
      };
      classical_fem(&linear_of(pgt)?, degree, disc)?
    };
    groups
      .entry(pe.name().to_string())
      .or_insert_with(|| (pe.clone(), vec![]))
      .1
      .push(cv);
  }
  let pmf = MeshFem::new(&mesh, 1);
  for (pf, cvs) in groups.values() {
    pmf.set_finite_element((cvs.iter().copied(), pf))?;
  }
  Ok(pmf)
}

/// the GetFEM space of a mesh export: P1, or P2 for transformations of
/// higher complexity (vtk_export::exporting(mesh))
fn mesh_space(mesh: &Mesh) -> Result<MeshFem> {
  let m = mesh.snapshot();
  let mut groups: BTreeMap<String, (PFem, Vec<usize>)> = BTreeMap::new();
  for cv in m.convex_index() {
    let pgt = m.trans_of_convex(cv);
    let pf = classical_fem(pgt, if pgt.complexity() > 1 { 2 } else { 1 }, false)?;
    groups
      .entry(pf.name().to_string())
      .or_insert_with(|| (pf.clone(), vec![]))
      .1
      .push(cv);
  }
  let pmf = MeshFem::new(mesh, 1);
  for (pf, cvs) in groups.values() {
    pmf.set_finite_element((cvs.iter().copied(), pf))?;
  }
  Ok(pmf)
}

struct Grid {
  pts: Vec<[f64; 3]>,
  cells: Vec<(u8, Vec<usize>)>,
}

/// the nodes of the exported space and its elements
fn space_grid(pmf: &MeshFem) -> Result<Grid> {
  let num = pmf.numbering()?;
  let m = pmf.linked_mesh().snapshot();
  if m.dim() > 3 {
    return Err(Error::Format(format!(
      "VTK export of a mesh of dimension {}",
      m.dim()
    )));
  }
  let pts = (0..num.nb_basic_dof())
    .map(|d| {
      let p = num.point_of_basic_dof(d)?;
      let mut x = [0.0; 3];
      x[..p.len()].copy_from_slice(&p);
      Ok(x)
    })
    .collect::<Result<Vec<_>>>()?;
  let mut cells = vec![];
  for cv in num.convex_index() {
    let b = m.trans_of_convex(cv).basic_structure();
    let dofs = num.ind_basic_dof_of_element(cv);
    let (t, perm) = lagrange_cell(b.dim(), b.nb_points(), dofs.len()).ok_or_else(|| {
      Error::Format(format!(
        "no VTK cell for {}",
        num.fem_of_element(cv).unwrap().name()
      ))
    })?;
    cells.push((t, perm.iter().map(|&j| dofs[j]).collect()));
  }
  Ok(Grid { pts, cells })
}

/// the nodes of a slice (not merged) and its simplexes
fn slice_grid(sl: &StoredMeshSlice) -> Result<Grid> {
  if sl.dim() > 3 {
    return Err(Error::Format(format!(
      "VTK export of a slice of dimension {}",
      sl.dim()
    )));
  }
  let (mut pts, mut cells) = (vec![], vec![]);
  for c in &sl.cvlst {
    let first = pts.len();
    for n in &c.nodes {
      let mut x = [0.0; 3];
      x[..n.pt.len()].copy_from_slice(&n.pt);
      pts.push(x);
    }
    for sx in &c.simplexes {
      let t = [1u8, 3, 5, 10][sx.dim()];
      cells.push((t, sx.inodes.iter().map(|i| first + i).collect()));
    }
  }
  Ok(Grid { pts, cells })
}

/// The exported space (built from a mesh or a space) or slice.
pub enum Exported<'a> {
  Space(MeshFem),
  Slice(&'a StoredMeshSlice),
}

/// What a VtuExport exports: a mesh, a space or a slice (GetFEM's
/// overloads of vtk_export::exporting).
pub trait ExportingArg<'a> {
  fn exported(self) -> Result<Exported<'a>>;
}

impl<'a> ExportingArg<'a> for &Mesh {
  fn exported(self) -> Result<Exported<'a>> {
    Ok(Exported::Space(mesh_space(self)?))
  }
}

impl<'a> ExportingArg<'a> for &MeshFem {
  fn exported(self) -> Result<Exported<'a>> {
    Ok(Exported::Space(export_space(self)?))
  }
}

impl<'a> ExportingArg<'a> for &'a StoredMeshSlice {
  fn exported(self) -> Result<Exported<'a>> {
    Ok(Exported::Slice(self))
  }
}

/// A .vtu file (GetFEM's vtu_export): exporting(..), then the fields, then
/// write().
pub struct VtuExport<'a> {
  path: String,
  format: Format,
  exported: Option<(Exported<'a>, Grid)>,
  point_data: Vec<(String, usize, Vec<f64>)>,
  cell_data: Vec<(String, usize, Vec<f64>)>,
}

impl<'a> VtuExport<'a> {
  pub fn new(path: &str, format: Format) -> Self {
    VtuExport {
      path: path.to_string(),
      format,
      exported: None,
      point_data: vec![],
      cell_data: vec![],
    }
  }

  /// The mesh, space or slice to export (before the fields).
  pub fn exporting(&mut self, x: impl ExportingArg<'a>) -> Result<()> {
    let e = x.exported()?;
    let g = match &e {
      Exported::Space(pmf) => space_grid(pmf)?,
      Exported::Slice(sl) => slice_grid(sl)?,
    };
    self.exported = Some((e, g));
    self.point_data.clear();
    self.cell_data.clear();
    Ok(())
  }

  fn grid(&self) -> Result<&(Exported<'a>, Grid)> {
    self
      .exported
      .as_ref()
      .ok_or_else(|| Error::Format("VTU export: nothing exported yet (call exporting)".into()))
  }

  /// A field u on mf (its DOFs; mf may be reduced), interpolated onto the
  /// exported space or slice.
  pub fn write_point_data(&mut self, mf: &MeshFem, u: &[f64], name: &str) -> Result<()> {
    let (e, g) = self.grid()?;
    let v = match e {
      Exported::Space(pmf) => interpolation(mf, pmf, u)?,
      Exported::Slice(sl) => {
        let m = mf.linked_mesh().snapshot();
        sl.interpolate(&m, &SliceField::new(mf, u)?)?
      }
    };
    let n = g.pts.len();
    if n == 0 || v.len() % n != 0 {
      return Err(Error::Format(format!(
        "field {name}: {} values for {n} points",
        v.len()
      )));
    }
    let q = v.len() / n;
    self.point_data.push((name.to_string(), q, v));
    Ok(())
  }

  /// Values at the points of the exported slice (q per point, in the
  /// slice's order; GetFEM's slice export of an array on the slice).
  pub fn write_slice_point_data(&mut self, v: &[f64], name: &str) -> Result<()> {
    let (e, g) = self.grid()?;
    if !matches!(e, Exported::Slice(_)) {
      return Err(Error::Format(format!(
        "field {name}: values on a slice only"
      )));
    }
    let n = g.pts.len();
    if n == 0 || !v.len().is_multiple_of(n) {
      return Err(Error::Format(format!(
        "field {name}: {} values for {n} points",
        v.len()
      )));
    }
    self
      .point_data
      .push((name.to_string(), v.len() / n, v.to_vec()));
    Ok(())
  }

  /// Values per convex of the exported mesh or space (qdim per convex, in
  /// the order of its convexes); not for slices, as GetFEM.
  pub fn write_cell_data(&mut self, u: &[f64], name: &str, qdim: usize) -> Result<()> {
    let (e, g) = self.grid()?;
    if matches!(e, Exported::Slice(_)) {
      return Err(Error::Format("VTU export: no cell data on a slice".into()));
    }
    if qdim == 0 || u.len() != g.cells.len() * qdim {
      return Err(Error::Format(format!(
        "field {name}: {} values for {} convexes x {qdim}",
        u.len(),
        g.cells.len()
      )));
    }
    self.cell_data.push((name.to_string(), qdim, u.to_vec()));
    Ok(())
  }

  /// Completes the file.
  pub fn write(self) -> Result<()> {
    let s = self.vtu_string()?;
    write(&self.path, s)
  }

  /// The file's content.
  pub fn vtu_string(&self) -> Result<String> {
    let (_, g) = self.grid()?;
    Ok(grid_vtu(g, &self.point_data, &self.cell_data, self.format))
  }
}

fn nums(out: &mut String, v: impl Iterator<Item = String>) {
  for (i, s) in v.enumerate() {
    out.push_str(if i % 9 == 0 { "\n" } else { " " });
    out.push_str(&s);
  }
  out.push('\n');
}

/// VTK's tuples: 3 components for a vector of dimension 2, 9 for a 2 x 2
/// tensor (GetFEM write_vec, write_3x3tensor).
fn padded(q: usize, v: &[f64]) -> (usize, Vec<f64>) {
  match q {
    2 => (3, v.chunks(2).flat_map(|c| [c[0], c[1], 0.0]).collect()),
    4 => (
      9,
      v.chunks(4)
        .flat_map(|c| [c[0], c[1], 0.0, c[2], c[3], 0.0, 0.0, 0.0, 0.0])
        .collect(),
    ),
    _ => (q, v.to_vec()),
  }
}

/// How the arrays of a .vtu file are written.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Format {
  /// text
  Ascii,
  /// base64 of the raw little-endian values
  Binary,
  /// base64 of zlib-compressed values (vtkZLibDataCompressor)
  Compressed,
}

enum Arr<'a> {
  F64(&'a [f64]),
  I64(Vec<i64>),
  U8(Vec<u8>),
}

impl Arr<'_> {
  fn vtk_type(&self) -> &'static str {
    match self {
      Arr::F64(_) => "Float64",
      Arr::I64(_) => "Int64",
      Arr::U8(_) => "UInt8",
    }
  }
  fn bytes(&self) -> Vec<u8> {
    match self {
      Arr::F64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
      Arr::I64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
      Arr::U8(v) => v.clone(),
    }
  }
  fn text(&self, s: &mut String) {
    match self {
      Arr::F64(v) => nums(s, v.iter().map(|x| format!("{x:e}"))),
      Arr::I64(v) => nums(s, v.iter().map(|x| x.to_string())),
      Arr::U8(v) => nums(s, v.iter().map(|x| x.to_string())),
    }
  }
}

/// a DataArray element; attrs: Name, NumberOfComponents
fn data_array(s: &mut String, attrs: &str, a: &Arr, format: Format) {
  let f = if format == Format::Ascii {
    "ascii"
  } else {
    "binary"
  };
  write!(
    s,
    "<DataArray type=\"{}\"{attrs} format=\"{f}\">",
    a.vtk_type()
  )
  .unwrap();
  match format {
    Format::Ascii => a.text(s),
    Format::Binary => {
      let raw = a.bytes();
      let mut d = (raw.len() as u64).to_le_bytes().to_vec();
      d.extend(raw);
      writeln!(s, "\n{}", crate::zlib::base64(&d)).unwrap();
    }
    Format::Compressed => {
      // one block: [nblocks, block size, last block size, compressed size]
      let raw = a.bytes();
      let z = crate::zlib::compress(&raw);
      let n = raw.len() as u64;
      let head: Vec<u8> = [1, n, n, z.len() as u64]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
      writeln!(
        s,
        "\n{}{}",
        crate::zlib::base64(&head),
        crate::zlib::base64(&z)
      )
      .unwrap();
    }
  }
  s.push_str("</DataArray>\n");
}

fn grid_vtu(
  g: &Grid,
  point_data: &[(String, usize, Vec<f64>)],
  cell_data: &[(String, usize, Vec<f64>)],
  format: Format,
) -> String {
  let mut s = String::new();
  writeln!(s, "<?xml version=\"1.0\"?>").unwrap();
  let comp = if format == Format::Compressed {
    " compressor=\"vtkZLibDataCompressor\""
  } else {
    ""
  };
  writeln!(
    s,
    "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\"{comp}>\n<UnstructuredGrid>"
  )
  .unwrap();
  writeln!(
    s,
    "<Piece NumberOfPoints=\"{}\" NumberOfCells=\"{}\">",
    g.pts.len(),
    g.cells.len()
  )
  .unwrap();
  s.push_str("<Points>\n");
  let pts: Vec<f64> = g.pts.iter().flatten().copied().collect();
  data_array(&mut s, " NumberOfComponents=\"3\"", &Arr::F64(&pts), format);
  s.push_str("</Points>\n<Cells>\n");
  let conn: Vec<i64> = g
    .cells
    .iter()
    .flat_map(|c| c.1.iter())
    .map(|&i| i as i64)
    .collect();
  data_array(&mut s, " Name=\"connectivity\"", &Arr::I64(conn), format);
  let mut o = 0;
  let offs: Vec<i64> = g
    .cells
    .iter()
    .map(|c| {
      o += c.1.len() as i64;
      o
    })
    .collect();
  data_array(&mut s, " Name=\"offsets\"", &Arr::I64(offs), format);
  data_array(
    &mut s,
    " Name=\"types\"",
    &Arr::U8(g.cells.iter().map(|c| c.0).collect()),
    format,
  );
  s.push_str("</Cells>\n");
  for (tag, data) in [("PointData", point_data), ("CellData", cell_data)] {
    if data.is_empty() {
      continue;
    }
    writeln!(s, "<{tag}>").unwrap();
    for (name, q, v) in data {
      let (nc, v) = padded(*q, v);
      data_array(
        &mut s,
        &format!(" Name=\"{name}\" NumberOfComponents=\"{nc}\""),
        &Arr::F64(&v),
        format,
      );
    }
    writeln!(s, "</{tag}>").unwrap();
  }
  s.push_str("</Piece>\n</UnstructuredGrid>\n</VTKFile>\n");
  s
}

fn write(path: &str, s: String) -> Result<()> {
  std::fs::write(path, s).map_err(|e| Error::Io(format!("{path}: {e}")))
}

/// A ParaView collection of files by time (.pvd): (time, file) pairs, the
/// files relative to the .pvd.
pub fn export_to_pvd(path: &str, steps: &[(f64, &str)]) -> Result<()> {
  let mut s = String::from(
    "<?xml version=\"1.0\"?>\n<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n<Collection>\n",
  );
  for (t, f) in steps {
    writeln!(
      s,
      "<DataSet timestep=\"{t:e}\" group=\"\" part=\"0\" file=\"{f}\"/>"
    )
    .unwrap();
  }
  s.push_str("</Collection>\n</VTKFile>\n");
  write(path, s)
}
