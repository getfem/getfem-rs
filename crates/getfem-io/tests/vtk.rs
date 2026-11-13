//! VTK export checked by VTK itself (tests/vtk_check.py), when the python
//! vtk module is available.

#[allow(dead_code)]
#[path = "../../getfem-gwfl/tests/common/mixed.rs"]
mod mixed;

use std::process::Command;

use getfem_fem::mesh_fem::MeshFem;
use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::mesh::Mesh;
use getfem_io::gmsh::{GmshOptions, import_from_gmsh};
use getfem_io::vtk::{Format, VtuExport};

fn out_dir() -> String {
  let d = format!("{}/../../target/vtk-check", env!("CARGO_MANIFEST_DIR"));
  std::fs::create_dir_all(&d).unwrap();
  d
}

/// VTK parametric point -> GetFEM reference point for a cell type.
fn to_ref(t: u8, pc: [f64; 3], dim: usize) -> Vec<f64> {
  match t {
    14 | 27 => vec![
      (2.0 * pc[0] - 1.0) * (1.0 - pc[2]),
      (2.0 * pc[1] - 1.0) * (1.0 - pc[2]),
      pc[2],
    ],
    _ => pc[..dim].to_vec(),
  }
}

fn cell_type(s: &str, i: usize) -> u8 {
  let tag = "Name=\"types\" format=\"ascii\">";
  let a = s.find(tag).unwrap() + tag.len();
  let b = a + s[a..].find("</DataArray>").unwrap();
  s[a..b].split_whitespace().nth(i).unwrap().parse().unwrap()
}

/// Write the mesh + reference positions; returns the file pair for
/// vtk_check.py.
fn export(name: &str, mesh: &Mesh) -> Vec<String> {
  let dir = out_dir();
  let m = mesh.snapshot();
  let file = format!("{dir}/{name}.vtu");
  let mut e = VtuExport::new(&file, Format::Ascii);
  e.exporting(mesh).unwrap();
  let vtu = e.vtu_string().unwrap();
  e.write().unwrap();
  let mut lines = String::new();
  for (i, cv) in m.convex_index().enumerate() {
    let t = cell_type(&vtu, i);
    let gt = m.trans_of_convex(cv);
    let g = m.points_of_convex(cv);
    let p = gt.dim();
    lines.push_str(&t.to_string());
    for pc in [[0.2, 0.3, 0.1], [0.25, 0.25, 0.25]] {
      let pc = [
        pc[0],
        if p > 1 { pc[1] } else { 0.0 },
        if p > 2 { pc[2] } else { 0.0 },
      ];
      let mut x = gt.transform(&to_ref(t, pc, p), &g).unwrap();
      x.resize(3, 0.0);
      lines.push_str(&format!(
        " {} {} {} {:e} {:e} {:e}",
        pc[0], pc[1], pc[2], x[0], x[1], x[2]
      ));
    }
    lines.push('\n');
  }
  let refp = format!("{dir}/{name}.txt");
  std::fs::write(&refp, lines).unwrap();
  vec![file, refp]
}

fn run_vtk_check(pairs: &[String]) {
  let o = Command::new("python3")
    .arg(format!("{}/tests/vtk_check.py", env!("CARGO_MANIFEST_DIR")))
    .args(pairs)
    .output()
    .unwrap();
  assert!(
    o.status.success(),
    "{}{}",
    String::from_utf8_lossy(&o.stdout),
    String::from_utf8_lossy(&o.stderr)
  );
}

fn vtk_available() -> bool {
  let ok = Command::new("python3")
    .args(["-c", "import vtk"])
    .output()
    .is_ok_and(|o| o.status.success());
  if !ok {
    eprintln!("python vtk not available: VTK round-trip checks skipped");
  }
  ok
}

#[test]
fn every_cell_type_read_back_by_vtk() {
  let vtk_ok = vtk_available();
  let mut pairs = vec![];
  let names = [
    "GT_PK(1,1)",
    "GT_PK(2,1)",
    "GT_PK(3,1)",
    "GT_QK(2,1)",
    "GT_QK(3,1)",
    "GT_LINEAR_QK(2)",
    "GT_PRISM(3,1)",
    "GT_PYRAMID(1)",
    "GT_PK(1,2)",
    "GT_PK(2,2)",
    "GT_PK(3,2)",
    "GT_Q2_INCOMPLETE(2)",
    "GT_Q2_INCOMPLETE(3)",
    "GT_PRISM_INCOMPLETE_P2",
    "GT_PYRAMID_Q2_INCOMPLETE",
    "GT_PYRAMID(2)",
    "GT_QK(2,2)",
    "GT_QK(3,2)",
    "GT_PRISM(3,2)",
    "GT_PK(2,3)",
    "GT_QK(2,3)",
  ];
  for n in names {
    let gt = geotrans_descriptor(n).unwrap();
    let m = Mesh::new(3);
    let affine = gt.is_linear() || n.starts_with("GT_PYRAMID");
    let pts: Vec<Vec<f64>> = gt
      .geometric_nodes()
      .iter()
      .map(|x| {
        let mut y = x.clone();
        y.resize(3, 0.0);
        let (a, b, c) = (y[0], y[1], y[2]);
        let bend = if affine { 0.0 } else { 0.1 };
        vec![
          1.0 + 2.0 * a + 0.3 * b + bend * b * b,
          -0.5 + 0.2 * a + 1.5 * b + bend * a * c,
          0.3 * c + 0.1 * a + bend * a * b,
        ]
      })
      .collect();
    let refs: Vec<&[f64]> = pts.iter().map(|v| v.as_slice()).collect();
    m.add_convex_by_points(&gt, &refs).unwrap();
    // degree 3 is exported as P2 (GetFEM): positions only approximate
    let files = export(&n.replace(['(', ')', ','], "_"), &m);
    if !n.ends_with(",3)") {
      pairs.extend(files);
    }
  }
  if vtk_ok {
    run_vtk_check(&pairs);
  }
}

#[test]
fn gmsh_meshes_read_back_by_vtk() {
  let vtk_ok = vtk_available();
  let dir = format!("{}/../../data/gmsh", env!("CARGO_MANIFEST_DIR"));
  let mut files: Vec<String> = std::fs::read_dir(&dir)
    .unwrap()
    .map(|e| e.unwrap().file_name().into_string().unwrap())
    .filter(|f| f.ends_with("_msh22.msh"))
    .collect();
  files.sort();
  let mut pairs = vec![];
  for f in files {
    let g = import_from_gmsh(&format!("{dir}/{f}"), &GmshOptions::default()).unwrap();
    let e = export(&f.replace(".msh", ""), &g.mesh);
    if !f.contains("tri3") {
      pairs.extend(e);
    }
  }
  if vtk_ok {
    run_vtk_check(&pairs);
  }
}

/// Fields on spaces as GetFEM: interpolated onto the exported space (a
/// linear field exactly), vectors padded to 3, 2 x 2 tensors to 3 x 3, a
/// reduced space, discontinuous elements (nodes per element), degree 3
/// exported as degree 2; cell data; errors.
#[test]
fn fields_and_errors() {
  let gt = geotrans_descriptor("GT_PK(2,1)").unwrap();
  let m = getfem_geom::regular_mesh::regular_unit_mesh(&[2, 2], &gt).unwrap();
  let space = |q: usize, k: usize| {
    MeshFem::new(&m, q)
      .with_classical_finite_element(k)
      .unwrap()
  };
  let at = |mf: &MeshFem, f: &dyn Fn(&[f64]) -> Vec<f64>| -> Vec<f64> {
    let q = mf.qdim();
    (0..mf.nb_dof().unwrap())
      .step_by(q)
      .flat_map(|d| f(&mf.point_of_basic_dof(d).unwrap()))
      .collect()
  };
  let (p1, p2, v1, t1) = (space(1, 1), space(1, 2), space(2, 1), space(4, 1));
  let u = at(&p2, &|x| vec![x[0] + 2.0 * x[1]]);
  let v = at(&v1, &|x| vec![x[0], -x[1]]);
  let t = at(&t1, &|x| vec![1.0, x[0], x[1], 2.0]);
  let mut e = VtuExport::new("unused.vtu", Format::Ascii);
  assert!(e.write_point_data(&p1, &[0.0; 9], "u").is_err());
  e.exporting(&p1).unwrap();
  e.write_point_data(&p2, &u, "u").unwrap();
  e.write_point_data(&v1, &v, "v").unwrap();
  e.write_point_data(&t1, &t, "t").unwrap();
  let c: Vec<f64> = (0..m.nb_convexes()).map(|i| i as f64).collect();
  e.write_cell_data(&c, "cv", 1).unwrap();
  assert!(e.write_cell_data(&c[1..], "cv", 1).is_err());
  assert!(e.write_point_data(&p2, &u[1..], "u").is_err());
  let s = e.vtu_string().unwrap();
  assert!(
    s.contains("NumberOfPoints=\"9\" NumberOfCells=\"8\""),
    "{s}"
  );
  assert!(s.contains("Name=\"v\" NumberOfComponents=\"3\""));
  assert!(s.contains("Name=\"t\" NumberOfComponents=\"9\""));
  // a reduced space: its DOFs, zero on the boundary when extended
  let bnd = p1
    .basic_dof_on_region(getfem_geom::region::outer_faces_of_mesh(&m))
    .unwrap();
  let kept = (0..9).filter(|d| !bnd.contains(d)).collect();
  let pp = MeshFem::partial(&p1, &kept).unwrap();
  e.write_point_data(&pp, &[1.0], "w").unwrap();
  assert!(e.write_point_data(&pp, &[1.0, 2.0], "w").is_err());
  // discontinuous: 3 nodes per triangle; degree 3: exported as P2 (25 nodes)
  let pd = MeshFem::new(&m, 1)
    .with_classical_discontinuous_finite_element(1)
    .unwrap();
  let mut d = VtuExport::new("unused.vtu", Format::Ascii);
  d.exporting(&pd).unwrap();
  assert!(d.vtu_string().unwrap().contains("NumberOfPoints=\"24\""));
  d.exporting(&space(1, 3)).unwrap();
  assert!(d.vtu_string().unwrap().contains("NumberOfPoints=\"25\""));
  if vtk_available() {
    let f = format!("{}/fields.vtu", out_dir());
    let mut e2 = VtuExport::new(&f, Format::Ascii);
    e2.exporting(&p1).unwrap();
    e2.write_point_data(&p2, &u, "u").unwrap();
    e2.write_point_data(&v1, &v, "v").unwrap();
    e2.write_point_data(&pp, &[1.0], "w").unwrap();
    e2.write_cell_data(&c, "cv", 1).unwrap();
    e2.write().unwrap();
    let py = format!(
      r#"
import vtk
from vtk.util.numpy_support import vtk_to_numpy as n
r = vtk.vtkXMLUnstructuredGridReader(); r.SetFileName('{f}'); r.Update(); g = r.GetOutput()
x = n(g.GetPoints().GetData()); pd = g.GetPointData()
assert abs(n(pd.GetArray('u')) - (x[:, 0] + 2 * x[:, 1])).max() < 1e-14
assert abs(n(pd.GetArray('v'))[:, 1] + x[:, 1]).max() < 1e-14
w = n(pd.GetArray('w')); inner = (abs(x[:, 0] - 0.5) < 1e-12) & (abs(x[:, 1] - 0.5) < 1e-12)
assert (w[inner] == 1).all() and (w[~inner] == 0).all()
assert g.GetCellData().GetArray('cv').GetNumberOfTuples() == 8
"#
    );
    let o = Command::new("python3").args(["-c", &py]).output().unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
  }
}

/// Binary and compressed files read by VTK as the ASCII ones; a slice with
/// a field interpolated on it; a PVD collection.
#[test]
fn binary_compressed_slices_pvd() {
  use getfem_fem::slice::{Orient, SliceField, SlicerVolume, StoredMeshSlice};
  use getfem_geom::regular_mesh::regular_mesh;
  use getfem_io::vtk::export_to_pvd;
  let m = regular_mesh(
    &geotrans_descriptor("GT_PK(2,2)").unwrap(),
    &[4, 3],
    &[0.0, 0.0],
    &[1.0, 0.7],
  )
  .unwrap();
  let md = m.snapshot();
  let mf = MeshFem::new(&m, 1)
    .with_classical_finite_element(2)
    .unwrap();
  let uf: Vec<f64> = (0..mf.nb_dof().unwrap())
    .map(|i| {
      let p = mf.point_of_basic_dof(i).unwrap();
      (3.0 * p[0]).sin() + p[1]
    })
    .collect();
  let file = |f: Format| {
    let mut e = VtuExport::new("unused.vtu", f);
    e.exporting(&mf).unwrap();
    e.write_point_data(&mf, &uf, "u").unwrap();
    e
  };
  let (bin, zip) = (
    file(Format::Binary).vtu_string().unwrap(),
    file(Format::Compressed).vtu_string().unwrap(),
  );
  assert!(zip.len() < bin.len(), "{} vs {}", zip.len(), bin.len());
  // a slice: the isoline u = 0.8, with u interpolated on it
  let field = SliceField::new(&mf, &uf).unwrap();
  let mut iso = SlicerVolume::isovalues(field.clone(), 0.8, Orient::Bound).unwrap();
  let sl = StoredMeshSlice::build(&md, &mut [&mut iso], 3, None).unwrap();
  let mut es = VtuExport::new("unused.vtu", Format::Ascii);
  es.exporting(&sl).unwrap();
  es.write_point_data(&mf, &uf, "u").unwrap();
  assert!(es.write_cell_data(&[0.0], "c", 1).is_err());
  // nodes on the isoline by linear interpolation along refined edges: u = 0.8 + O(h^2)
  let v = sl.interpolate(&md, &field).unwrap();
  assert!(
    sl.nb_simplexes(1) > 0 && v.iter().all(|x| (x - 0.8).abs() < 1e-2),
    "{v:?}"
  );
  if !vtk_available() {
    return;
  }
  let dir = out_dir();
  let mut files = vec![];
  for (k, f) in [Format::Ascii, Format::Binary, Format::Compressed]
    .into_iter()
    .enumerate()
  {
    let p = format!("{dir}/fmt{k}.vtu");
    let mut e = VtuExport::new(&p, f);
    e.exporting(&mf).unwrap();
    e.write_point_data(&mf, &uf, "u").unwrap();
    e.write().unwrap();
    files.push(p);
  }
  let sp = format!("{dir}/slice.vtu");
  let mut es = VtuExport::new(&sp, Format::Compressed);
  es.exporting(&sl).unwrap();
  es.write_point_data(&mf, &uf, "u").unwrap();
  es.write().unwrap();
  let pvd = format!("{dir}/steps.pvd");
  export_to_pvd(&pvd, &[(0.0, "fmt0.vtu"), (0.5, "fmt2.vtu")]).unwrap();
  let py = format!(
    r#"
import vtk, xml.etree.ElementTree as ET
from vtk.util.numpy_support import vtk_to_numpy as n
def rd(f):
  r = vtk.vtkXMLUnstructuredGridReader(); r.SetFileName(f); r.Update(); return r.GetOutput()
g = [rd(f) for f in {files:?}]
for h in g[1:]:
  assert h.GetNumberOfCells() == g[0].GetNumberOfCells() > 0
  assert abs(n(h.GetPoints().GetData()) - n(g[0].GetPoints().GetData())).max() == 0
  assert abs(n(h.GetPointData().GetArray('u')) - n(g[0].GetPointData().GetArray('u'))).max() == 0
  assert (n(h.GetCellTypesArray()) == n(g[0].GetCellTypesArray())).all()
x = n(g[0].GetPoints().GetData())
assert abs(n(g[0].GetPointData().GetArray('u')) - (np.sin(3 * x[:, 0]) + x[:, 1])).max() < 1e-14
s = rd('{sp}')
assert s.GetNumberOfCells() == {nc} and s.GetCell(0).GetCellType() == 3
assert abs(n(s.GetPointData().GetArray('u')) - 0.8).max() < 1e-2
c = ET.parse('{pvd}').getroot().find('Collection')
assert [d.get('file') for d in c] == ['fmt0.vtu', 'fmt2.vtu']
"#,
    nc = sl.nb_simplexes(1)
  )
  .replace("import vtk, xml", "import numpy as np, vtk, xml");
  let o = Command::new("python3").args(["-c", &py]).output().unwrap();
  assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
}

#[test]
fn mixed_meshes_read_back_by_vtk() {
  // triangles and quadrilaterals (degree 1, 2), hexahedra, prisms,
  // pyramids and tetrahedra in one file
  let vtk_ok = vtk_available();
  let mut pairs = vec![];
  for mi in 0..3 {
    pairs.extend(export(
      &format!("mixed_mesh_{mi}"),
      &mixed::mixed_mesh(mi, true),
    ));
  }
  if vtk_ok {
    run_vtk_check(&pairs);
  }
}
