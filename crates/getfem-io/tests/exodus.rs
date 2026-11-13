//! Exodus II import (files of data/exodus, written by meshio and
//! netCDF4: make_refs.py) and the netCDF reader.

use getfem_io::netcdf::NcFile;
use getfem_linalg::tensor::{Lu, Mat};

fn data(f: &str) -> String {
  format!("{}/../../data/exodus/{f}", env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn netcdf_reader() {
  let nc = NcFile::open(&data("sets.exo")).unwrap();
  assert_eq!(nc.dim("num_dim"), Some(3));
  assert_eq!(nc.dim("num_nodes"), Some(6));
  assert_eq!(nc.dim("time_step"), Some(0));
  assert_eq!(
    nc.get_f64("coordx", None).unwrap(),
    [0., 1., 1., 0., 2., 2.]
  );
  assert_eq!(nc.get_i64("connect2").unwrap(), [2, 5, 6, 3]);
  assert_eq!(nc.shape("connect1").unwrap(), [1, 4]);
  assert_eq!(nc.att_text("connect2", "elem_type").unwrap(), "SHELL4");
  assert_eq!(nc.get_strings("coor_names").unwrap(), ["X", "Y", "Z"]);
  assert_eq!(nc.get_i64("eb_prop1").unwrap(), [1, 7]);
  // meshio: a record variable, packed coordinates
  let nc = NcFile::open(&data("meshio_2d.exo")).unwrap();
  assert_eq!(nc.dim("time_step"), Some(1));
  assert_eq!(nc.shape("coord").unwrap(), [2, 6]);
  let t = nc.get_f64("vals_nod_var1", Some(0)).unwrap();
  assert_eq!(t, [0., 1., 3., 2., 2., 4.]);
  assert!(nc.get_f64("vals_nod_var1", Some(1)).is_err());
  assert!(nc.get_f64("vals_nod_var1", None).is_err());
  // not netCDF, not the 64-bit offset format
  let e = NcFile::open(&data("make_refs.py"))
    .err()
    .unwrap()
    .to_string();
  assert!(e.contains("not a netCDF file"), "{e}");
}

use getfem_io::exodus::import_from_exodus;

/// region id -> (number of convexes, number of faces)
fn regions(m: &getfem_geom::mesh::Mesh) -> Vec<(usize, usize, usize)> {
  m.region_ids()
    .into_iter()
    .map(|id| {
      let r = m.region_set(id).unwrap();
      let cvs = r.iter().filter(|(_, f)| f.is_none()).count();
      (id, cvs, r.iter().count() - cvs)
    })
    .collect()
}

#[test]
fn meshio_files() {
  let m = import_from_exodus(&data("meshio_2d.exo")).unwrap();
  assert_eq!((m.dim(), m.nb_points(), m.nb_convexes()), (2, 6, 3));
  // meshio's block ids 0, 1; the node set ignored
  assert_eq!(regions(&m), [(100000, 2, 0), (100001, 1, 0)]);
  let m = import_from_exodus(&data("meshio_3d.exo")).unwrap();
  assert_eq!((m.dim(), m.nb_points(), m.nb_convexes()), (3, 16, 4));
  let names: Vec<String> = m
    .convex_index()
    .into_iter()
    .map(|cv| m.trans_of_convex(cv).name())
    .collect();
  assert_eq!(
    names,
    ["GT_QK(3,1)", "GT_PRISM(3,1)", "GT_PK(3,1)", "GT_PYRAMID(1)"]
  );
}

#[test]
fn sets() {
  let m = import_from_exodus(&data("sets.exo")).unwrap();
  assert_eq!((m.dim(), m.nb_points(), m.nb_convexes()), (3, 6, 2));
  // side set 1 (a face of convex 0) and element set 1 (convex 1) merged,
  // side set 2 (a face of convex 1), the blocks 1 and 7
  assert_eq!(
    regions(&m),
    [(1, 1, 1), (2, 0, 1), (100001, 1, 0), (100007, 1, 0)]
  );
  // the faces: x = 0 (region 1), x = 2 (region 2)
  for (r, x) in [(1, 0.), (2, 2.)] {
    for (cv, f) in m.region_set(r).unwrap().iter() {
      if let Some(f) = f {
        for p in m.ind_points_of_face_of_convex(cv, f) {
          assert_eq!(m.point(p)[0], x, "region {r}");
        }
      }
    }
  }
}

/// Least-squares residual of x = A r + b over the nodes of a convex (the
/// reference nodes r of its transformation, its points x).
fn affine_residual(m: &getfem_geom::mesh::Mesh, cv: usize) -> f64 {
  let gt = m.trans_of_convex(cv);
  let rows: Vec<Vec<f64>> = gt
    .geometric_nodes()
    .iter()
    .map(|r| r.iter().copied().chain([1.0]).collect())
    .collect();
  let x: Vec<Vec<f64>> = m
    .ind_points_of_convex(cv)
    .iter()
    .map(|&p| m.point(p).to_vec())
    .collect();
  let n = rows[0].len();
  // normal equations, per coordinate of x
  let mut nm = Mat::zeros(&[n, n]);
  for a in &rows {
    for (j, aj) in a.iter().enumerate() {
      for (k, ak) in a.iter().enumerate() {
        nm[(j, k)] += aj * ak;
      }
    }
  }
  let lu = Lu::new(&nm).unwrap();
  (0..x[0].len())
    .map(|d| {
      let b: Vec<f64> = (0..n)
        .map(|j| rows.iter().zip(&x).map(|(a, xi)| a[j] * xi[d]).sum())
        .collect();
      let c = lu.solve(&b);
      rows
        .iter()
        .zip(&x)
        .map(|(a, xi)| (a.iter().zip(&c).map(|(u, v)| u * v).sum::<f64>() - xi[d]).abs())
        .fold(0.0, f64::max)
    })
    .fold(0.0, f64::max)
}

#[test]
fn quadratic_types() {
  // one affine element of each quadratic type: the node order is right
  // iff the convex is affine in its transformation's reference nodes
  let m = import_from_exodus(&data("quadratic.exo")).unwrap();
  assert_eq!(m.nb_convexes(), 11);
  for cv in m.convex_index() {
    let name = m.trans_of_convex(cv).name();
    assert!(affine_residual(&m, cv) < 1e-12, "{name}");
  }
}

#[test]
fn netcdf_writer() {
  use getfem_io::netcdf::{Att, NC_CHAR, NC_DOUBLE, NC_INT, NcWriter, doubles, ints, strings};
  let mut w = NcWriter::default();
  let (t, n, l) = (w.dim("time_step", 0), w.dim("n", 3), w.dim("len", 5));
  w.att(Att::text("title", "writer test"));
  w.att(Att::int("file_size", 1));
  w.var("time_whole", &[t], NC_DOUBLE, vec![], doubles(&[0.5]));
  w.var("x", &[n], NC_DOUBLE, vec![], doubles(&[1., 2., 3.]));
  let a = vec![Att::text("elem_type", "TRI3")];
  w.var("c", &[n], NC_INT, a, ints(&[7, 8, 9]));
  w.var(
    "names",
    &[n, l],
    NC_CHAR,
    vec![],
    strings(&["a", "bb", "cccccc"], 5),
  );
  w.var("v", &[t, n], NC_DOUBLE, vec![], doubles(&[4., 5., 6.]));
  let path = std::env::temp_dir().join(format!("getfem_rs_nc_{}.nc", std::process::id()));
  let path = path.to_str().unwrap();
  let (bytes, rec) = w.bytes();
  std::fs::write(path, &bytes).unwrap();
  assert_eq!(bytes.len() - rec, 32, "one record: time and v");
  let nc = NcFile::open(path).unwrap();
  assert_eq!(nc.dim("time_step"), Some(1));
  assert_eq!(nc.get_f64("x", None).unwrap(), [1., 2., 3.]);
  assert_eq!(nc.get_i64("c").unwrap(), [7, 8, 9]);
  assert_eq!(nc.att_text("c", "elem_type").unwrap(), "TRI3");
  assert_eq!(nc.get_strings("names").unwrap(), ["a", "bb", "cccc"]);
  assert_eq!(nc.get_f64("v", Some(0)).unwrap(), [4., 5., 6.]);
  assert_eq!(nc.get_f64("time_whole", Some(0)).unwrap(), [0.5]);
  std::fs::remove_file(path).unwrap();
}

use getfem_fem::mesh_fem::MeshFem;
use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::region::RegionSet;
use getfem_io::exodus::{export_mesh_to_exodus, export_to_exodus, import_point_data};

fn tmp(name: &str) -> String {
  let p = std::env::temp_dir().join(format!("getfem_rs_{}_{name}", std::process::id()));
  p.to_str().unwrap().to_string()
}

/// meshio's view of an Exodus file (points, cells, fields), None without
/// meshio
fn meshio(path: &str) -> Option<Vec<String>> {
  let o = std::process::Command::new("python3")
    .arg(data("check_meshio.py"))
    .arg(path)
    .output()
    .ok()?;
  if !o.status.success() {
    eprintln!(
      "meshio check skipped: {}",
      String::from_utf8_lossy(&o.stderr)
    );
    return None;
  }
  Some(
    String::from_utf8_lossy(&o.stdout)
      .lines()
      .map(String::from)
      .collect(),
  )
}

#[test]
fn export_round_trip() {
  // 2 x 2 x 2 hexahedra; region 3: the faces at x = 0, 5: convexes 0, 1,
  // 7: convex 2 and a face of convex 3
  let gt = geotrans_descriptor("GT_QK(3,1)").unwrap();
  let m = getfem_geom::regular_mesh::regular_unit_mesh(&[2, 2, 2], &gt).unwrap();
  m.modify(|md| {
    let mut r3 = RegionSet::default();
    for cv in md.convex_index() {
      for f in 0..md.structure_of_convex(cv).nb_faces() {
        if md
          .ind_points_of_face_of_convex(cv, f)
          .iter()
          .all(|&p| md.point(p)[0] == 0.0)
        {
          r3.add_face(cv, f);
        }
      }
    }
    let mut r5 = RegionSet::default();
    r5.add(0);
    r5.add(1);
    let mut r7 = RegionSet::default();
    r7.add(2);
    r7.add_face(3, 0);
    md.set_region_set(3, r3).unwrap();
    md.set_region_set(5, r5).unwrap();
    md.set_region_set(7, r7).unwrap();
  });
  // on Q2 (HEX27 nodes): u = (x, 2 y, x + z), w = T (1, 1, 1), and on Q2
  // scalar T = x + 2 y + 3 z, in another file
  let mf1 = MeshFem::new(&m, 1)
    .with_classical_finite_element(2)
    .unwrap();
  let mf3 = MeshFem::new(&m, 3)
    .with_classical_finite_element(2)
    .unwrap();
  let f = |x: &[f64]| [x[0] + 2. * x[1] + 3. * x[2], x[0], 2. * x[1], x[0] + x[2]];
  let vals = |mf: &MeshFem, q: usize, off: usize| -> Vec<f64> {
    (0..mf.nb_basic_dof().unwrap())
      .map(|d| f(&mf.point_of_basic_dof(d).unwrap())[off + d % q])
      .collect()
  };
  let w0: Vec<f64> = (0..mf3.nb_basic_dof().unwrap())
    .map(|d| f(&mf3.point_of_basic_dof(d).unwrap())[0])
    .collect();
  let (t0, u0) = (vals(&mf1, 1, 0), vals(&mf3, 3, 1));
  let two = |v: &[f64]| v.iter().map(|x| 2. * x).collect::<Vec<_>>();
  let path = tmp("round_trip.exo");
  export_to_exodus(&path, &mf3, &[(&u0, "u"), (&w0, "w")], 0.0, false).unwrap();
  let (u1, w1) = (two(&u0), two(&w0));
  export_to_exodus(&path, &mf3, &[(&u1, "u"), (&w1, "w")], 1.0, true).unwrap();
  // the file: 2 steps, the fields at the nodes
  let nc = NcFile::open(&path).unwrap();
  assert_eq!(nc.dim("time_step"), Some(2));
  assert_eq!(nc.get_f64("time_whole", Some(1)).unwrap(), [1.0]);
  let names = nc.get_strings("name_nod_var").unwrap();
  assert_eq!(names, ["u_x", "u_y", "u_z", "w_x", "w_y", "w_z"]);
  let x: Vec<Vec<f64>> = ["coordx", "coordy", "coordz"]
    .iter()
    .map(|c| nc.get_f64(c, None).unwrap())
    .collect();
  for step in 0..2 {
    for (k, name) in names.iter().enumerate() {
      let v = nc
        .get_f64(&format!("vals_nod_var{}", k + 1), Some(step))
        .unwrap();
      for (n, vn) in v.iter().enumerate() {
        let fx = f(&[x[0][n], x[1][n], x[2][n]]);
        let want = (step + 1) as f64 * if k < 3 { fx[k + 1] } else { fx[0] };
        assert!((vn - want).abs() < 1e-12, "step {step}, {name}");
      }
    }
  }
  // append: other fields, another mesh
  let e = |r: getfem_io::Result<()>| r.unwrap_err().to_string();
  let err = e(export_to_exodus(&path, &mf3, &[(&u0, "u")], 2.0, true));
  assert!(
    err.contains("fields u_x, u_y, u_z, w_x, w_y, w_z expected (got u_x, u_y, u_z)"),
    "{err}"
  );
  let m2 = getfem_geom::regular_mesh::regular_unit_mesh(&[2, 2, 1], &gt).unwrap();
  let mf23 = MeshFem::new(&m2, 3)
    .with_classical_finite_element(2)
    .unwrap();
  let z = vec![0.; mf23.nb_basic_dof().unwrap()];
  let err = e(export_to_exodus(
    &path,
    &mf23,
    &[(&z, "u"), (&z, "w")],
    2.0,
    true,
  ));
  assert!(err.contains("another mesh or space"), "{err}");
  // a discontinuous space: no file
  let mfd = MeshFem::new(&m, 1)
    .with_classical_discontinuous_finite_element(1)
    .unwrap();
  let zd = vec![0.; mfd.nb_basic_dof().unwrap()];
  let err = e(export_to_exodus(
    &tmp("d.exo"),
    &mfd,
    &[(&zd, "T")],
    0.0,
    false,
  ));
  assert!(err.contains("discontinuous"), "{err}");
  // read back: the regions, one block (HEX27, region 100001)
  let mi = import_from_exodus(&path).unwrap();
  assert_eq!(mi.nb_convexes(), 8);
  assert_eq!(
    regions(&mi),
    [(3, 0, 4), (5, 2, 0), (7, 1, 1), (100001, 8, 0)]
  );
  for (cv, f) in mi.region_set(3).unwrap().iter() {
    for p in mi.ind_points_of_face_of_convex(cv, f.unwrap()) {
      assert_eq!(mi.point(p)[0], 0.0);
    }
  }
  if let Some(v) = meshio(&path) {
    assert_eq!(v, ["125", "hexahedron27:8", "u_x u_y u_z w_x w_y w_z"]);
  }
  // the times of the steps, negative steps from the end
  assert_eq!(import_point_data(&path, &mf3, "u", 1).unwrap().1, 1.0);
  assert_eq!(import_point_data(&path, &mf3, "u", -2).unwrap().1, 0.0);
  // the fields read back (the last step: doubled): onto the original space
  // (another mesh object, the same geometry) and onto the imported mesh
  let close = |a: &[f64], b: &[f64]| {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12)
  };
  assert!(close(
    &import_point_data(&path, &mf3, "u", -1).unwrap().0,
    &u1
  ));
  assert!(close(
    &import_point_data(&path, &mf3, "w", 0).unwrap().0,
    &w0
  ));
  let mfi = MeshFem::new(&mi, 3)
    .with_classical_finite_element(2)
    .unwrap();
  let v = import_point_data(&path, &mfi, "u", 1).unwrap().0;
  assert!(close(&v, &two(&vals(&mfi, 3, 1))), "imported mesh");
  // the scalar file: read onto continuous Q2, discontinuous Q1
  let path_t = tmp("round_trip_t.exo");
  export_to_exodus(&path_t, &mf1, &[(&t0, "T")], 0.0, false).unwrap();
  assert!(close(
    &import_point_data(&path_t, &mf1, "T", -1).unwrap().0,
    &t0
  ));
  let mfd = MeshFem::new(&mi, 1)
    .with_classical_discontinuous_finite_element(1)
    .unwrap();
  let v = import_point_data(&path_t, &mfd, "T", -1).unwrap().0;
  assert!(close(&v, &vals(&mfd, 1, 0)), "discontinuous");
  // errors: no field, no step, degree 3, another mesh
  let err = |r: getfem_io::Result<(Vec<f64>, f64)>| r.unwrap_err().to_string();
  assert!(err(import_point_data(&path_t, &mf1, "p", -1)).contains("no field p (fields: T)"));
  assert!(err(import_point_data(&path_t, &mf1, "T", 1)).contains("no time step 1 (1 steps)"));
  let mf_3 = MeshFem::new(&m, 1)
    .with_classical_finite_element(3)
    .unwrap();
  assert!(err(import_point_data(&path_t, &mf_3, "T", -1)).contains("a DOF not at a node"));
  let mf2 = MeshFem::new(&m2, 1)
    .with_classical_finite_element(2)
    .unwrap();
  assert!(err(import_point_data(&path_t, &mf2, "T", -1)).contains("no convex with its vertices"));
  std::fs::remove_file(&path).unwrap();
  std::fs::remove_file(&path_t).unwrap();
}

#[test]
fn import_meshio_field() {
  // meshio_2d.exo: T = x + 2 y at the nodes (meshio), read onto continuous
  // and discontinuous P1 on the imported mesh
  let path = data("meshio_2d.exo");
  let m = import_from_exodus(&path).unwrap();
  for disc in [false, true] {
    let mf = MeshFem::new(&m, 1);
    let mf = if disc {
      mf.with_classical_discontinuous_finite_element(1)
    } else {
      mf.with_classical_finite_element(1)
    }
    .unwrap();
    let t = import_point_data(&path, &mf, "T", -1).unwrap().0;
    for (d, td) in t.iter().enumerate() {
      let x = mf.point_of_basic_dof(d).unwrap();
      assert_eq!(*td, x[0] + 2. * x[1], "discontinuous {disc}");
    }
  }
}

#[test]
fn export_every_type() {
  // the quadratic elements exported as a mesh, with classical spaces of
  // degree 1 and 2 (the mesh's nodes: incomplete kept) and with complete
  // degree 2 spaces (complete nodes), and read back: affine, blocks kept
  let m = import_from_exodus(&data("quadratic.exo")).unwrap();
  let names = |m: &getfem_geom::mesh::Mesh| -> Vec<String> {
    m.convex_index()
      .into_iter()
      .map(|cv| m.trans_of_convex(cv).name())
      .collect()
  };
  let complete: Vec<String> = names(&m)
    .iter()
    .map(|n| match n.as_str() {
      "GT_Q2_INCOMPLETE(2)" => "GT_QK(2,2)",
      "GT_Q2_INCOMPLETE(3)" => "GT_QK(3,2)",
      "GT_PRISM_INCOMPLETE_P2" => "GT_PRISM(3,2)",
      "GT_PYRAMID_Q2_INCOMPLETE" => "GT_PYRAMID(2)",
      n => n,
    })
    .map(String::from)
    .collect();
  let path = tmp("every_type.exo");
  for k in [0, 1, 2, 3] {
    if k == 0 {
      export_mesh_to_exodus(&path, &m).unwrap();
    } else {
      let mf = MeshFem::new(&m, 1);
      for (cv, gt) in m.convex_index().into_iter().zip(&complete) {
        let pgt = if k == 3 {
          geotrans_descriptor(gt).unwrap()
        } else {
          m.trans_of_convex(cv)
        };
        let pf = getfem_fem::fem::classical_fem(&pgt, k.min(2), false).unwrap();
        mf.set_finite_element(([cv].into_iter(), &pf)).unwrap();
      }
      let u = vec![1.; mf.nb_basic_dof().unwrap()];
      export_to_exodus(&path, &mf, &[(&u, "u")], 0.0, false).unwrap();
    }
    let mi = import_from_exodus(&path).unwrap();
    let want = if k == 3 { &complete } else { &names(&m) };
    assert_eq!(&names(&mi), want, "degree {k}");
    let ids: Vec<usize> = mi.region_ids().into_iter().collect();
    assert_eq!(ids, (100001..=100011).collect::<Vec<_>>());
    for cv in mi.convex_index() {
      let name = mi.trans_of_convex(cv).name();
      assert!(affine_residual(&mi, cv) < 1e-12, "{name}");
    }
  }
  // an incomplete space on a linear mesh: incomplete nodes
  let gt = geotrans_descriptor("GT_QK(3,1)").unwrap();
  let h = getfem_geom::regular_mesh::regular_unit_mesh(&[1, 1, 1], &gt).unwrap();
  let pf = getfem_fem::fem::fem_descriptor("FEM_Q2_INCOMPLETE(3)").unwrap();
  let mf = MeshFem::new(&h, 1).with_finite_element(&pf).unwrap();
  let u = vec![1.; mf.nb_basic_dof().unwrap()];
  export_to_exodus(&path, &mf, &[(&u, "u")], 0.0, false).unwrap();
  assert_eq!(
    names(&import_from_exodus(&path).unwrap()),
    ["GT_Q2_INCOMPLETE(3)"]
  );
  std::fs::remove_file(&path).unwrap();
}

#[test]
fn crack_values_kept() {
  // two triangles along a crack (coincident, not merged nodes, gmsh); T = 1
  // on one, -1 on the other: exported, read back on each side
  let text = "$MeshFormat 2.2 0 8 $EndMeshFormat \
     $Nodes 6 1 0 0 0 2 1 0 0 3 1 1 0 4 0 1 0 5 0 0 0 6 1 1 0 $EndNodes \
     $Elements 2 1 2 2 1 1 1 2 3 2 2 2 1 1 5 6 4 $EndElements";
  let m = getfem_io::gmsh::parse_gmsh(text, "crack", &Default::default())
    .unwrap()
    .mesh;
  let mf = MeshFem::new(&m, 1)
    .with_classical_finite_element(1)
    .unwrap();
  let num = mf.numbering().unwrap();
  let mut t = vec![0.; mf.nb_basic_dof().unwrap()];
  for (cv, s) in [(0, 1.), (1, -1.)] {
    for &d in num.ind_basic_dof_of_element(cv) {
      t[d] = s;
    }
  }
  let path = tmp("crack.exo");
  export_to_exodus(&path, &mf, &[(&t, "T")], 0.0, false).unwrap();
  assert_eq!(import_point_data(&path, &mf, "T", -1).unwrap().0, t);
  let mi = import_from_exodus(&path).unwrap();
  assert_eq!(mi.nb_points(), 6);
  let mfi = MeshFem::new(&mi, 1)
    .with_classical_finite_element(1)
    .unwrap();
  let ti = import_point_data(&path, &mfi, "T", -1).unwrap().0;
  let ni = mfi.numbering().unwrap();
  for cv in mi.convex_index() {
    let x: f64 = mi
      .ind_points_of_convex(cv)
      .iter()
      .map(|&p| mi.point(p)[0])
      .sum();
    let side = if x > 1.5 { 1. } else { -1. }; // triangle 0: x sum 2
    for &d in ni.ind_basic_dof_of_element(cv) {
      assert_eq!(ti[d], side);
    }
  }
  std::fs::remove_file(&path).unwrap();
}
