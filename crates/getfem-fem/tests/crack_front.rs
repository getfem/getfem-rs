//! Crack fronts: seeding (an ellipse, a straight front reaching the
//! boundary, two concentric fronts) and hats along fronts (a partition of
//! unity in a tube around a closed (circle) and an open (arc) front,
//! linear and cubic; gradients and Hessians against central differences;
//! zero outside the support).

use getfem_fem::crack_front::{
  CrackFront, HatDegree, crack_frames, crack_fronts, front_element_dofs, front_hats,
  select_mesh_fem_basic_dofs_near_front,
};
use getfem_fem::global_function::GlobalFunction;
use getfem_fem::level_set::LevelSet;
use getfem_fem::mesh_fem::MeshFem;
use getfem_fem::mesh_level_set::MeshLevelSet;
use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::mesh::Mesh;
use getfem_geom::regular_mesh::regular_mesh;

const R: f64 = 5.0;

/// n points on a circle of radius R in the plane z = 0, over the angle a1
fn arc(n: usize, a1: f64, closed: bool) -> CrackFront {
  let m = if closed { n } else { n - 1 };
  let ang: Vec<f64> = (0..n).map(|i| a1 * i as f64 / m as f64).collect();
  CrackFront {
    points: ang
      .iter()
      .map(|a| vec![R * a.cos(), R * a.sin(), 0.0])
      .collect(),
    tangents: ang.iter().map(|a| vec![-a.sin(), a.cos(), 0.0]).collect(),
    closed,
  }
}

/// pseudo-random points in the tube of radius 1.5 around the circle,
/// over the angles [a0, a1]
fn tube(a0: f64, a1: f64, n: usize) -> Vec<Vec<f64>> {
  let mut seed: u64 = 7;
  let mut rnd = || {
    seed = seed
      .wrapping_mul(6364136223846793005)
      .wrapping_add(1442695040888963407);
    (seed >> 11) as f64 / (1u64 << 53) as f64
  };
  (0..n)
    .map(|_| {
      let a = a0 + (a1 - a0) * rnd();
      let r = R + 1.5 * (2.0 * rnd() - 1.0);
      vec![r * a.cos(), r * a.sin(), 1.5 * (2.0 * rnd() - 1.0)]
    })
    .collect()
}

fn sum(hats: &[GlobalFunction], x: &[f64]) -> f64 {
  hats.iter().map(|h| h.val(x)).sum()
}

#[test]
fn partition_of_unity() {
  let tau = 2.0 * std::f64::consts::PI;
  for degree in [HatDegree::Linear, HatDegree::Cubic] {
    let closed = front_hats(&arc(24, tau, true), degree, 2.5).unwrap();
    for x in tube(0.0, tau, 5000) {
      let s = sum(&closed, &x);
      assert!((s - 1.0).abs() < 1e-13, "{degree:?} closed at {x:?}: {s}");
    }
    // open arc over 3/4 of the circle, beyond its ends too
    let open = front_hats(&arc(19, 0.75 * tau, false), degree, 2.5).unwrap();
    for x in tube(-0.2, 0.75 * tau + 0.2, 5000) {
      let s = sum(&open, &x);
      assert!((s - 1.0).abs() < 1e-13, "{degree:?} open at {x:?}: {s}");
    }
  }
}

#[test]
fn zero_outside_the_support() {
  let tau = 2.0 * std::f64::consts::PI;
  for degree in [HatDegree::Linear, HatDegree::Cubic] {
    let hats = front_hats(&arc(24, tau, true), degree, 1.0).unwrap();
    // 1.2 from the circle: beyond the radius
    for x in [[R + 1.2, 0.3, 0.0], [0.2, R, 1.2], [0.0, 0.0, 0.0]] {
      assert!(
        hats
          .iter()
          .all(|h| h.val(&x) == 0.0 && !h.is_in_support(&x))
      );
    }
  }
}

#[test]
fn derivatives_against_central_differences() {
  let tau = 2.0 * std::f64::consts::PI;
  let eps = 1e-5;
  for degree in [HatDegree::Linear, HatDegree::Cubic] {
    let hats = front_hats(&arc(24, tau, true), degree, 2.5).unwrap();
    let mut checked = 0;
    for x in tube(0.0, tau, 200) {
      for h in &hats {
        if h.val(&x) == 0.0 {
          continue;
        }
        let (g, hs) = (h.grad(&x), h.hess(&x));
        for a in 0..3 {
          let (mut xp, mut xm) = (x.clone(), x.clone());
          xp[a] += eps;
          xm[a] -= eps;
          // skip points that cross a plane (kink in s)
          let (gp, gm) = (h.grad(&xp), h.grad(&xm));
          if (0..3).any(|b| (gp[b] - gm[b]).abs() > 1e-2) {
            continue;
          }
          checked += 1;
          let fd = (h.val(&xp) - h.val(&xm)) / (2.0 * eps);
          assert!(
            (fd - g[a]).abs() < 1e-7,
            "{degree:?} grad {a} at {x:?}: {} vs {fd}",
            g[a]
          );
          for b in 0..3 {
            let fd2 = (gp[b] - gm[b]) / (2.0 * eps);
            assert!(
              (fd2 - hs[b + 3 * a]).abs() < 1e-6,
              "{degree:?} hess {b}{a} at {x:?}: {} vs {fd2}",
              hs[b + 3 * a]
            );
          }
        }
      }
    }
    assert!(checked > 100, "{degree:?}: {checked} checks");
  }
}

/// a crack in the plane z = 0.1 of [-1, 1]^3 (n^3 hexahedra), the
/// secondary f, P2 level sets
fn crack(n: usize, f: fn(&[f64]) -> f64) -> (Mesh, LevelSet) {
  let m = regular_mesh(
    &geotrans_descriptor("GT_QK(3,1)").unwrap(),
    &[n; 3],
    &[-1.0; 3],
    &[2.0; 3],
  )
  .unwrap();
  let ls = LevelSet::new(&m, 2, true).unwrap();
  let mf = ls.mesh_fem();
  let pts: Vec<Vec<f64>> = (0..mf.nb_dof().unwrap())
    .map(|i| mf.point_of_basic_dof(i).unwrap())
    .collect();
  ls.set_values(0, &pts.iter().map(|x| x[2] - 0.1).collect::<Vec<_>>())
    .unwrap();
  ls.set_values(1, &pts.iter().map(|x| f(x)).collect::<Vec<_>>())
    .unwrap();
  (m, ls)
}

/// the distances between consecutive points (closed: the last to the
/// first too)
fn gaps(f: &CrackFront) -> Vec<f64> {
  let n = f.points.len();
  let m = if f.closed { n } else { n - 1 };
  (0..m)
    .map(|k| {
      let (a, b) = (&f.points[k], &f.points[(k + 1) % n]);
      (0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>().sqrt()
    })
    .collect()
}

#[test]
fn seeding_an_ellipse() {
  let (_m, ls) = crack(8, |x| (x[0] / 0.6).powi(2) + (x[1] / 0.4).powi(2) - 1.0);
  let fronts = crack_fronts(&ls, Some(0.1)).unwrap();
  assert_eq!(fronts.len(), 1);
  let f = &fronts[0];
  assert!(f.closed);
  // the perimeter of the ellipse, by fine sampling
  let per: f64 = (0..100000)
    .map(|k| {
      let (a, b) = (
        2.0 * std::f64::consts::PI * k as f64 / 1e5,
        2.0 * std::f64::consts::PI * (k + 1) as f64 / 1e5,
      );
      ((0.6 * (b.cos() - a.cos())).powi(2) + (0.4 * (b.sin() - a.sin())).powi(2)).sqrt()
    })
    .sum();
  assert_eq!(f.points.len(), (per / 0.1).round() as usize);
  for g in gaps(f) {
    assert!((g / 0.1 - 1.0).abs() < 0.05, "gap {g}");
  }
  for (p, t) in f.points.iter().zip(&f.tangents) {
    assert!((p[2] - 0.1).abs() < 1e-10, "{p:?}");
    let e = (p[0] / 0.6).powi(2) + (p[1] / 0.4).powi(2) - 1.0;
    assert!(e.abs() < 2e-3, "{p:?}: {e}");
    // the tangent: unit, normal to z and to the ellipse's gradient
    let gr = [p[0] / 0.36, p[1] / 0.16];
    let nt = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
    assert!((nt - 1.0).abs() < 1e-12 && t[2].abs() < 1e-10);
    assert!((t[0] * gr[0] + t[1] * gr[1]).abs() < 2e-2 * (gr[0] * gr[0] + gr[1] * gr[1]).sqrt());
  }
}

#[test]
fn seeding_an_open_front() {
  let (_m, ls) = crack(8, |x| x[0] - 0.3);
  let fronts = crack_fronts(&ls, Some(0.2)).unwrap();
  assert_eq!(fronts.len(), 1);
  let f = &fronts[0];
  assert!(!f.closed);
  let (a, b) = (&f.points[0], &f.points[f.points.len() - 1]);
  // from one face of the cube to the other (y = -1 to 1, in either order)
  assert!(
    (a[1].abs() - 1.0).abs() < 1e-8 && (b[1].abs() - 1.0).abs() < 1e-8 && a[1] * b[1] < 0.0,
    "{a:?} {b:?}"
  );
  assert_eq!(f.points.len(), 11);
  for g in gaps(f) {
    assert!((g - 0.2).abs() < 1e-8, "gap {g}");
  }
  // the frame: e1 = x (beyond the front), e2 = z, e3 = e1 x e2 = -y
  for e in crack_frames(&ls, &f.points).unwrap() {
    let e = e.unwrap();
    for (v, w) in e
      .iter()
      .zip([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    {
      assert!(v.iter().zip(w).all(|(x, y)| (x - y).abs() < 1e-10), "{e:?}");
    }
  }
  for (p, t) in f.points.iter().zip(&f.tangents) {
    assert!(
      (p[0] - 0.3).abs() < 1e-10 && (p[2] - 0.1).abs() < 1e-10,
      "{p:?}"
    );
    assert!((t[1].abs() - 1.0).abs() < 1e-10);
    assert!(t[1] * (b[1] - a[1]) > 0.0, "tangents along the front");
  }
}

#[test]
fn seeding_two_fronts() {
  let (_m, ls) = crack(16, |x| {
    let r = (x[0] * x[0] + x[1] * x[1]).sqrt();
    (r - 0.3) * (r - 0.7)
  });
  let mut fronts = crack_fronts(&ls, Some(0.1)).unwrap();
  assert_eq!(fronts.len(), 2);
  fronts.sort_by_key(|f| f.points.len());
  for (f, r) in fronts.iter().zip([0.3, 0.7]) {
    assert!(f.closed);
    assert_eq!(
      f.points.len(),
      (2.0 * std::f64::consts::PI * r / 0.1).round() as usize
    );
    for p in &f.points {
      let rr = (p[0] * p[0] + p[1] * p[1]).sqrt();
      // the front of the secondary's P2 interpolant (h = 0.125)
      assert!((rr - r).abs() < 2e-3, "{rr} vs {r}");
    }
  }
}

/// The front elements' DOFs (the elements the cut finds the front in): Q1
/// nodes within one element diagonal of the elliptical front, the elements
/// holding nearly all of its points
#[test]
fn front_element_dofs_of_an_ellipse() {
  let (m, ls) = crack(8, |x| (x[0] / 0.6).powi(2) + (x[1] / 0.4).powi(2) - 1.0);
  let fronts = crack_fronts(&ls, Some(0.25 / 4.0)).unwrap();
  let mf = MeshFem::new(&m, 1)
    .with_classical_finite_element(1)
    .unwrap();
  let mut mls = MeshLevelSet::new(&m);
  mls.add_level_set(&ls);
  mls.adapt().unwrap();
  let dofs = front_element_dofs(&mls, 0, &mf).unwrap();
  assert!(!dofs.is_empty() && dofs.len() < mf.nb_dof().unwrap() / 4);
  // the distance of a node to the front, sampled
  let front: Vec<[f64; 3]> = (0..2000)
    .map(|k| {
      let t = 2.0 * std::f64::consts::PI * k as f64 / 2000.0;
      [0.6 * t.cos(), 0.4 * t.sin(), 0.1]
    })
    .collect();
  let diag = 0.25 * 3f64.sqrt();
  for &d in &dofs {
    let p = mf.point_of_basic_dof(d).unwrap();
    let dist = front
      .iter()
      .map(|q| (0..3).map(|a| (p[a] - q[a]).powi(2)).sum::<f64>().sqrt())
      .fold(f64::MAX, f64::min);
    assert!(dist <= diag + 1e-9, "node {p:?} at {dist} from the front");
  }
  // the front elements cover the front, but where it grazes elements
  // (their nodes keep the Heaviside function): the nodes strictly within
  // an element size of a front point in every direction (the nodes of its
  // element) enriched at 90 % of the points at least
  let (mut all, mut covered) = (0, 0);
  for f in &fronts {
    for p in &f.points {
      let near: Vec<usize> = (0..mf.nb_dof().unwrap())
        .filter(|&d| {
          let x = mf.point_of_basic_dof(d).unwrap();
          (0..3).all(|a| (x[a] - p[a]).abs() < 0.25 - 1e-9)
        })
        .collect();
      all += 1;
      covered += usize::from(near.iter().all(|d| dofs.contains(d)));
    }
  }
  assert!(covered as f64 >= 0.9 * all as f64, "{covered} of {all}");
}

/// 2D: the two tips of a centre crack at 30 degrees, and their frames
#[test]
fn tips_and_frames_in_2d() {
  let m = regular_mesh(
    &geotrans_descriptor("GT_PK(2,1)").unwrap(),
    &[10, 10],
    &[-1.0; 2],
    &[2.0; 2],
  )
  .unwrap();
  let ls = LevelSet::new(&m, 2, true).unwrap();
  let mf = ls.mesh_fem();
  let pts: Vec<Vec<f64>> = (0..mf.nb_dof().unwrap())
    .map(|i| mf.point_of_basic_dof(i).unwrap())
    .collect();
  let (c, s) = (3f64.sqrt() / 2.0, 0.5);
  ls.set_values(
    0,
    &pts.iter().map(|x| -s * x[0] + c * x[1]).collect::<Vec<_>>(),
  )
  .unwrap();
  ls.set_values(
    1,
    &pts
      .iter()
      .map(|x| (c * x[0] + s * x[1]).abs() - 0.5)
      .collect::<Vec<_>>(),
  )
  .unwrap();
  let fronts = crack_fronts(&ls, None).unwrap();
  assert!(
    fronts
      .iter()
      .all(|f| f.points.len() == 1 && f.tangents.is_empty() && !f.closed)
  );
  let mut tips: Vec<Vec<f64>> = fronts.into_iter().map(|f| f.points[0].clone()).collect();
  tips.sort_by(|a, b| a[0].total_cmp(&b[0]));
  assert_eq!(tips.len(), 2);
  for (t, sg) in tips.iter().zip([-1.0, 1.0]) {
    assert!(
      (t[0] - sg * 0.5 * c).abs() < 1e-10 && (t[1] - sg * 0.5 * s).abs() < 1e-10,
      "{t:?}"
    );
  }
  for (e, sg) in crack_frames(&ls, &tips)
    .unwrap()
    .into_iter()
    .zip([-1.0, 1.0])
  {
    let e = e.unwrap();
    assert_eq!(e.len(), 2);
    assert!(
      (e[0][0] - sg * c).abs() < 1e-10 && (e[0][1] - sg * s).abs() < 1e-10,
      "{e:?}"
    );
    assert!(
      (e[1][0] + s).abs() < 1e-10 && (e[1][1] - c).abs() < 1e-10,
      "{e:?}"
    );
  }
}

/// The DOFs near a straight front (z = 0.1, x = 0.3: the level sets are
/// the distances to its planes): the Q1 nodes within the radius of the
/// line, and none of the others
#[test]
fn near_front_dofs_of_a_straight_front() {
  let (m, ls) = crack(8, |x| x[0] - 0.3);
  let mf = MeshFem::new(&m, 1)
    .with_classical_finite_element(1)
    .unwrap();
  let r = 0.3;
  let dofs = select_mesh_fem_basic_dofs_near_front(&ls, &mf, r).unwrap();
  let mut n = 0;
  for d in 0..mf.nb_dof().unwrap() {
    let p = mf.point_of_basic_dof(d).unwrap();
    let dist = ((p[0] - 0.3).powi(2) + (p[2] - 0.1).powi(2)).sqrt();
    assert_eq!(dofs.contains(&d), dist < r, "{p:?} at {dist}");
    n += usize::from(dist < r);
  }
  assert!(n > 0 && n == dofs.len());
  assert!(select_mesh_fem_basic_dofs_near_front(&ls, &mf, -1.0).is_err());
}
