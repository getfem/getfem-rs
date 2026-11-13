//! Affine-violation errors and point location in a mesh.

use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::locate::Locator;
use getfem_geom::mesh::MeshData;
use getfem_geom::regular_mesh::regular_mesh;

const SCALES: [f64; 3] = [1e-3, 1.0, 1e3];

#[test]
fn affine_violation_is_an_error() {
  for sc in SCALES {
    let lin = geotrans_descriptor("GT_LINEAR_QK(2)").unwrap();
    let q1 = geotrans_descriptor("GT_QK(2,1)").unwrap();
    let mut m = MeshData::new(2);
    let p: Vec<usize> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.3, 1.2]]
      .iter()
      .map(|x| m.add_point(&[x[0] * sc, x[1] * sc]).unwrap())
      .collect();
    // a parallelogram is fine, a general quadrilateral is not
    let cv = m.add_convex(&lin, &p[..4]).unwrap();
    let e = m.add_convex(&lin, &[p[0], p[1], p[2], p[4]]).unwrap_err();
    assert!(e.to_string().contains("not an affine image"), "{e}");
    assert!(m.add_convex(&q1, &[p[0], p[1], p[2], p[4]]).is_ok());
    // moving a corner of the affine cell is refused, mesh unchanged
    let v = m.version();
    assert!(m.set_point(p[3], &[1.2 * sc, 1.0 * sc]).is_err());
    assert_eq!(m.version(), v);
    assert_eq!(m.point(p[3]), &[sc, sc]);
    // a rigid motion of the whole cell keeps it affine
    m.translate(&[0.5 * sc, -2.0 * sc]).unwrap();
    assert!(m.geom_at(cv, &[0.5, 0.5]).is_ok());
    // round-off level moves are accepted
    let x = m.point(p[3]).to_vec();
    assert!(m.set_point(p[3], &[x[0] * (1.0 + 1e-14), x[1]]).is_ok());
  }
}

#[test]
fn locate_points() {
  for sc in SCALES {
    let gt = geotrans_descriptor("GT_QK(2,2)").unwrap();
    let m = regular_mesh(&gt, &[4, 3], &[0.0, 0.0], &[sc, sc]).unwrap();
    // curve the interior nodes (the boundary stays straight)
    for i in m.point_index() {
      let x = m.point(i).to_vec();
      let (u, v) = (x[0] / sc, x[1] / sc);
      if u > 1e-9 && u < 1.0 - 1e-9 && v > 1e-9 && v < 1.0 - 1e-9 {
        m.set_point(
          i,
          &[
            x[0] + 0.02 * sc * (7.0 * v).sin(),
            x[1] + 0.02 * sc * (5.0 * u).sin(),
          ],
        )
        .unwrap();
      }
    }
    let snap = m.snapshot();
    let loc = Locator::new(&snap, 1e-10).unwrap();
    let mut seed = 11u64;
    let mut rnd = || {
      seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
      (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    for _ in 0..400 {
      let x = [rnd() * sc, rnd() * sc];
      let (cv, xref) = loc
        .find(&x)
        .unwrap()
        .expect("point of the unit square not located");
      let y = m
        .trans_of_convex(cv)
        .transform(&xref, &m.points_of_convex(cv))
        .unwrap();
      assert!((y[0] - x[0]).abs() < 1e-10 * sc && (y[1] - x[1]).abs() < 1e-10 * sc);
    }
    assert!(loc.find(&[1.01 * sc, 0.5 * sc]).unwrap().is_none());
    // a boundary node shared by two cells
    assert!(loc.find_all(&[0.5 * sc, 0.0]).unwrap().len() >= 2);
  }
}
