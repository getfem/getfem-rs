//! Codimension > 0 geometry: lines in 2D/3D, surfaces in 3D. An isometric
//! embedding must give the planar J, element size and (rotated) face normals,
//! at scales 1e-3, 1, 1e3.

use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::mesh::{AsMeshData, MeshData};
use getfem_geom::region::outer_faces_of_mesh;
use getfem_geom::regular_mesh::regular_mesh;
use getfem_linalg::tensor::Mat;

const SCALES: [f64; 3] = [1e-3, 1.0, 1e3];

/// Columns orthonormal: embedding R^p -> R^n.
fn isometry(n: usize, p: usize) -> Mat {
  let c: Vec<Vec<f64>> = match (n, p) {
    (2, 1) => vec![vec![0.6, 0.8]],
    (3, 1) => vec![vec![1. / 3., 2. / 3., 2. / 3.]],
    (3, 2) => vec![
      vec![1. / 3., 2. / 3., 2. / 3.],
      vec![2. / 3., 1. / 3., -2. / 3.],
    ],
    _ => unreachable!(),
  };
  let q = Mat::from_data(&[n, p], c.concat()).unwrap();
  let qtq = q.tmatmul(&q);
  for i in 0..p {
    for j in 0..p {
      assert!(
        (qtq[(i, j)] - if i == j { 1. } else { 0. }).abs() < 1e-15,
        "not orthonormal"
      );
    }
  }
  q
}

fn close(a: f64, b: f64) -> bool {
  (a - b).abs() <= 1e-12 * b.abs().max(1e-300)
}

fn check(gt_name: &str, ns: &[usize], n: usize) {
  let gt = geotrans_descriptor(gt_name).unwrap();
  let p = gt.dim();
  let q = isometry(n, p);
  for sc in SCALES {
    let flat = regular_mesh(&gt, ns, &vec![0.1 * sc; p], &vec![sc; p]).unwrap();
    let mut emb: MeshData = (*flat.snapshot()).clone();
    emb.transform(&q).unwrap();
    assert_eq!(emb.dim(), n);
    let xref = vec![0.25; p];
    for cv in flat.convex_index() {
      let (g0, g1) = (
        flat.geom_at(cv, &xref).unwrap(),
        emb.geom_at(cv, &xref).unwrap(),
      );
      assert!(
        close(g1.j, g0.j.abs()),
        "{gt_name} N={n} sc={sc} cv {cv}: J {} vs {}",
        g1.j,
        g0.j
      );
      assert!(
        close(emb.convex_size(cv).unwrap(), flat.convex_size(cv).unwrap()),
        "{gt_name} size"
      );
      // B^T K = I_P
      let btk = g1.b.tmatmul(&g1.k);
      for i in 0..p {
        for j in 0..p {
          assert!((btk[(i, j)] - if i == j { 1. } else { 0. }).abs() < 1e-12);
        }
      }
      // J scales as sc^P
      assert!(close(
        g1.j / sc.powi(p as i32),
        g0.j.abs() / sc.powi(p as i32)
      ));
    }
    let (of0, of1) = (outer_faces_of_mesh(&flat), outer_faces_of_mesh(&emb));
    let (of0, of1) = (
      of0.resolve(&flat.view()).unwrap(),
      of1.resolve(&emb.view()).unwrap(),
    );
    assert_eq!(of0, of1, "{gt_name}: same boundary");
    for (cv, f) in of1.iter() {
      let f = f.unwrap();
      let n0 = flat.mean_normal_of_face_of_convex(cv, f).unwrap();
      let n1 = emb.mean_normal_of_face_of_convex(cv, f).unwrap();
      let qn0: Vec<f64> = (0..n)
        .map(|i| (0..p).map(|j| q[(i, j)] * n0[j]).sum())
        .collect();
      assert!(
        n1.iter().zip(&qn0).all(|(a, b)| (a - b).abs() < 1e-12),
        "{gt_name} normal {n1:?} vs {qn0:?}"
      );
    }
  }
}

#[test]
fn lines_in_2d_3d() {
  for g in ["GT_PK(1,1)", "GT_PK(1,2)", "GT_PK(1,3)"] {
    check(g, &[3], 2);
    check(g, &[3], 3);
  }
}

#[test]
fn surfaces_in_3d() {
  for g in ["GT_PK(2,1)", "GT_PK(2,2)", "GT_QK(2,1)", "GT_QK(2,2)"] {
    check(g, &[2, 3], 3);
  }
}

#[test]
fn segment_size_is_its_length() {
  // the reported GetFEM issue: element_size of a line in 2D
  let gt = geotrans_descriptor("GT_PK(1,1)").unwrap();
  for sc in SCALES {
    let mut m = MeshData::new(2);
    let (a, b) = (
      m.add_point(&[0.0, 0.0]).unwrap(),
      m.add_point(&[3.0 * sc, 4.0 * sc]).unwrap(),
    );
    let cv = m.add_convex(&gt, &[a, b]).unwrap();
    assert!(close(m.convex_size(cv).unwrap(), 5.0 * sc));
    assert!(close(m.geom_at(cv, &[0.5]).unwrap().j, 5.0 * sc));
  }
}
