//! Meshes of mixed element types, conforming, as crates/getfem-tests/generators/src/gwfl_mixed.cc
//! builds them (the same convexes, points and order).

use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::mesh::{Mesh, MeshData};

/// the reference nodes of gt mapped by p = o + sum_d ref_d e[d] (e swapped to
/// a positive orientation), as crates/getfem-tests/generators/src/gwfl_mixed.cc
fn add(m: &mut MeshData, gt: &str, o: &[f64], mut e: Vec<Vec<f64>>) {
  let pgt = geotrans_descriptor(gt).unwrap();
  let n = o.len();
  let det = if n == 2 {
    e[0][0] * e[1][1] - e[0][1] * e[1][0]
  } else {
    e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1])
      - e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0])
      + e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0])
  };
  if det < 0.0 {
    e.swap(0, 1);
  }
  let pts: Vec<Vec<f64>> = pgt
    .geometric_nodes()
    .iter()
    .map(|r| {
      let mut p = o.to_vec();
      for (d, rd) in r.iter().enumerate() {
        for i in 0..n {
          p[i] += rd * e[d][i];
        }
      }
      p
    })
    .collect();
  let pts: Vec<&[f64]> = pts.iter().map(|p| p.as_slice()).collect();
  m.add_convex_by_points(&pgt, &pts).unwrap();
}

/// 2D: nx x ny cells, a quadrilateral where i + j is even, else two
/// triangles; transformations of degree k
fn mixed_mesh2([nx, ny]: [usize; 2], k: usize) -> MeshData {
  let (q, t) = (&format!("GT_QK(2,{k})"), &format!("GT_PK(2,{k})"));
  let mut m = MeshData::new(2);
  for j in 0..ny {
    for i in 0..nx {
      let (x0, x1) = (i as f64 / nx as f64, (i + 1) as f64 / nx as f64);
      let (y0, y1) = (j as f64 / ny as f64, (j + 1) as f64 / ny as f64);
      if (i + j) % 2 == 0 {
        add(
          &mut m,
          q,
          &[x0, y0],
          vec![vec![x1 - x0, 0.0], vec![0.0, y1 - y0]],
        );
      } else {
        add(
          &mut m,
          t,
          &[x0, y0],
          vec![vec![x1 - x0, 0.0], vec![0.0, y1 - y0]],
        );
        add(
          &mut m,
          t,
          &[x1, y1],
          vec![vec![x0 - x1, 0.0], vec![0.0, y0 - y1]],
        );
      }
    }
  }
  m
}

/// 3D: nx x ny x nz cells; columns where i + 2 j is a multiple of 3 two
/// prisms per cell; else a hexahedron where i + j + k is even, six pyramids
/// (apex at the cell center) where odd, those on the bottom split into two
/// tetrahedra; transformations of degree kd
fn mixed_mesh3([nx, ny, nz]: [usize; 3], kd: usize) -> MeshData {
  let (prism, hex) = (&format!("GT_PRISM(3,{kd})"), &format!("GT_QK(3,{kd})"));
  let (tet, pyramid) = (&format!("GT_PK(3,{kd})"), &format!("GT_PYRAMID({kd})"));
  let mut m = MeshData::new(3);
  let sub = |a: &[f64], b: &[f64]| -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x - y).collect() };
  let addv = |a: &[f64], b: &[f64]| -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x + y).collect() };
  for k in 0..nz {
    for j in 0..ny {
      for i in 0..nx {
        let (x0, x1) = (i as f64 / nx as f64, (i + 1) as f64 / nx as f64);
        let (y0, y1) = (j as f64 / ny as f64, (j + 1) as f64 / ny as f64);
        let (z0, z1) = (k as f64 / nz as f64, (k + 1) as f64 / nz as f64);
        let (hx, hy, hz) = ((x1 - x0) / 2.0, (y1 - y0) / 2.0, (z1 - z0) / 2.0);
        let c = [x0 + hx, y0 + hy, z0 + hz];
        let box3 = |dx: f64, dy: f64| {
          vec![
            vec![dx, 0.0, 0.0],
            vec![0.0, dy, 0.0],
            vec![0.0, 0.0, z1 - z0],
          ]
        };
        if (i + 2 * j) % 3 == 0 {
          add(&mut m, prism, &[x0, y0, z0], box3(x1 - x0, y1 - y0));
          add(&mut m, prism, &[x1, y1, z0], box3(x0 - x1, y0 - y1));
        } else if (i + j + k) % 2 == 0 {
          add(&mut m, hex, &[x0, y0, z0], box3(x1 - x0, y1 - y0));
        } else {
          // faces: center, half edges a, b
          let faces = [
            ([x0, c[1], c[2]], [0.0, hy, 0.0], [0.0, 0.0, hz]),
            ([x1, c[1], c[2]], [0.0, hy, 0.0], [0.0, 0.0, hz]),
            ([c[0], y0, c[2]], [hx, 0.0, 0.0], [0.0, 0.0, hz]),
            ([c[0], y1, c[2]], [hx, 0.0, 0.0], [0.0, 0.0, hz]),
            ([c[0], c[1], z0], [hx, 0.0, 0.0], [0.0, hy, 0.0]),
            ([c[0], c[1], z1], [hx, 0.0, 0.0], [0.0, hy, 0.0]),
          ];
          for (f, (fc, a, b)) in faces.iter().enumerate() {
            let apex = sub(&c, fc);
            if f == 4 && k == 0 {
              let q0 = sub(&sub(fc, a), b);
              let q1 = sub(&addv(fc, a), b);
              let q2 = addv(&addv(fc, a), b);
              let q3 = addv(&sub(fc, a), b);
              add(
                &mut m,
                tet,
                &q0,
                vec![sub(&q1, &q0), sub(&q2, &q0), sub(&c, &q0)],
              );
              add(
                &mut m,
                tet,
                &q0,
                vec![sub(&q2, &q0), sub(&q3, &q0), sub(&c, &q0)],
              );
            } else {
              add(&mut m, pyramid, fc, vec![a.to_vec(), b.to_vec(), apex]);
            }
          }
        }
      }
    }
  }
  m
}

/// the mixed mesh of the unit square or cube, sub cells per direction,
/// transformations of degree k
pub fn mixed_unit_mesh(sub: &[usize], k: usize) -> Mesh {
  Mesh::from(match *sub {
    [nx, ny] => mixed_mesh2([nx, ny], k),
    [nx, ny, nz] => mixed_mesh3([nx, ny, nz], k),
    _ => panic!("2D or 3D"),
  })
}

/// mixed mesh mi (0, 1: 2D of degree 1, 2; 2, 3: 3D of degree 1, 2), its
/// nodes moved as gwfl_mixed.cc if perturbed
pub fn mixed_mesh(mi: usize, perturbed: bool) -> Mesh {
  let mut m = match mi {
    0 | 1 => mixed_mesh2([4, 3], 1 + mi),
    2 => mixed_mesh3([3, 3, 2], 1),
    // smaller (its reference data): still all four element types
    _ => mixed_mesh3([2, 2, 2], 2),
  };
  if perturbed {
    let moves: Vec<(usize, Vec<f64>)> = m
      .point_index()
      .collect::<Vec<_>>()
      .into_iter()
      .map(|i| {
        let p = m.point(i).to_vec();
        let mut q = p.clone();
        q[0] = p[0] + 0.05 * (2.1 * p[1] + 0.3 * p[0]).sin();
        q[1] = p[1] + 0.05 * (1.7 * p[0] + 0.2 * p[1]).cos() - 0.05;
        if p.len() == 3 {
          q[2] = p[2] + 0.04 * (p[0] + p[1]).sin();
        }
        (i, q)
      })
      .collect();
    m.set_points(&moves).unwrap();
  }
  Mesh::from(m)
}
