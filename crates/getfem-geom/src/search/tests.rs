//! Ports of GetFEM tests/test_kdtree.cc and test_rtree.cc (brute force checks).
use super::*;

struct Rng(u64);
impl Rng {
  fn f(&mut self) -> f64 {
    self.0 = self
      .0
      .wrapping_mul(6364136223846793005)
      .wrapping_add(1442695040888963407);
    (self.0 >> 11) as f64 / (1u64 << 53) as f64
  }
  fn u(&mut self, n: usize) -> usize {
    (self.f() * n as f64) as usize % n
  }
}

fn brute_box(pts: &[Vec<f64>], lo: &[f64], hi: &[f64]) -> Vec<usize> {
  (0..pts.len())
    .filter(|&i| {
      pts[i]
        .iter()
        .zip(lo.iter().zip(hi))
        .all(|(x, (a, b))| x >= a && x <= b)
    })
    .collect()
}

fn verify(pts: &[Vec<f64>], t: &KdTree, lo: &[f64], hi: &[f64]) {
  let got: Vec<usize> = t
    .points_in_box(lo, hi)
    .unwrap()
    .iter()
    .map(|p| p.0)
    .collect();
  assert_eq!(got, brute_box(pts, lo, hi), "box {lo:?} {hi:?}");
}

#[test]
fn kdtree_check_tree() {
  let mut r = Rng(3);
  let mut pts = vec![
    vec![0.4, 0.7],
    vec![1.4, 0.8],
    vec![2.4, 0.3],
    vec![3.4, 0.1],
    vec![0.5, 2.2],
    vec![-0.8, 0.4],
  ];
  for _ in 0..50 {
    pts.push(vec![r.f(), r.f()]);
  }
  let mut t = KdTree::new(2);
  for p in &pts {
    t.add_point(p).unwrap();
  }
  verify(&pts, &t, &[0., 0.], &[6., 6.]);
  for p in &pts {
    verify(
      &pts,
      &t,
      &[p[0] - 1e-5, p[1] - 1e-5],
      &[p[0] + 1e-5, p[1] + 1e-5],
    );
    verify(&pts, &t, p, p);
  }
  // degenerate clouds: repeated values, coplanar points
  for _ in 0..200 {
    let pval = [0.3, 24.0, -2.3];
    let same: Vec<bool> = (0..3).map(|_| r.u(3) == 0).collect();
    let pts: Vec<Vec<f64>> = (0..200)
      .map(|_| {
        (0..3)
          .map(|k| if same[k] { pval[1] } else { pval[r.u(3)] })
          .collect()
      })
      .collect();
    let mut t = KdTree::new(3);
    for p in &pts {
      t.add_point(p).unwrap();
    }
    verify(&pts, &t, &[-100.; 3], &[100.; 3]);
    verify(&pts, &t, &[-100.; 3], &[0.; 3]);
    verify(&pts, &t, &pts[0], &pts[0]);
    let (lo, hi): (Vec<f64>, Vec<f64>) = pts[0]
      .iter()
      .zip(&pts[1])
      .map(|(a, b)| (a.min(*b), a.max(*b)))
      .unzip();
    verify(&pts, &t, &lo, &hi);
  }
}

#[test]
fn kdtree_nearest() {
  let mut r = Rng(9);
  for sc in [1e-3, 1.0, 1e3] {
    let pts: Vec<Vec<f64>> = (0..500)
      .map(|_| (0..3).map(|_| r.f() * sc).collect())
      .collect();
    let mut t = KdTree::new(3);
    for p in &pts {
      t.add_point(p).unwrap();
    }
    for _ in 0..100 {
      let q: Vec<f64> = (0..3).map(|_| (r.f() * 1.2 - 0.1) * sc).collect();
      let d = |p: &Vec<f64>| {
        p.iter()
          .zip(&q)
          .map(|(a, b)| (a - b) * (a - b))
          .sum::<f64>()
          .sqrt()
      };
      let best = (0..pts.len())
        .min_by(|&a, &b| d(&pts[a]).total_cmp(&d(&pts[b])))
        .unwrap();
      let (id, dist) = t.nearest_neighbor(&q).unwrap().unwrap();
      assert!((dist - d(&pts[best])).abs() <= 1e-15 * sc && d(&pts[id]) == dist);
    }
  }
  assert!(
    KdTree::new(2)
      .nearest_neighbor(&[0.0, 0.0])
      .unwrap()
      .is_none()
  );
  assert!(KdTree::new(2).add_point(&[0.0]).is_err());
}

fn b_inter(l1: &[f64], h1: &[f64], l2: &[f64], h2: &[f64]) -> bool {
  (0..l1.len()).all(|i| h1[i] >= l2[i] && l1[i] <= h2[i])
}
fn b_ge(l1: &[f64], h1: &[f64], l2: &[f64], h2: &[f64]) -> bool {
  (0..l1.len()).all(|i| l1[i] <= l2[i] && h1[i] >= h2[i])
}

fn rtree_verify(bl: &[Vec<f64>], bh: &[Vec<f64>], t: &RTree, r: &mut Rng) {
  let n = bl[0].len();
  let brute = |f: &dyn Fn(&[f64], &[f64]) -> bool| {
    (0..bl.len())
      .filter(|&i| f(&bl[i], &bh[i]))
      .collect::<Vec<_>>()
  };
  for _ in 0..100 {
    let lo: Vec<f64> = (0..n).map(|_| r.f() * 1.3).collect();
    let hi: Vec<f64> = lo.iter().map(|v| v + r.f() * 0.1).collect();
    assert_eq!(
      t.find_containing_boxes(&lo, &hi).unwrap(),
      brute(&|l, h| b_ge(l, h, &lo, &hi))
    );
    assert_eq!(
      t.find_intersecting_boxes(&lo, &hi).unwrap(),
      brute(&|l, h| b_inter(&lo, &hi, l, h))
    );
    assert_eq!(
      t.find_contained_boxes(&lo, &hi).unwrap(),
      brute(&|l, h| b_ge(&lo, &hi, l, h))
    );
    assert_eq!(
      t.find_boxes_at_point(&lo).unwrap(),
      brute(&|l, h| b_ge(l, h, &lo, &lo))
    );
  }
}

#[test]
fn rtree_check_tree() {
  let mut t = RTree::new(2, 0.0);
  t.add_box(&[1.0, 0.], &[1.5, 0.], 0).unwrap();
  t.add_box(&[2.0, 0.], &[3.0, 0.], 2).unwrap();
  t.add_box(&[1.5, 0.], &[2.2, 0.], 1).unwrap();
  assert_eq!(t.find_boxes_at_point(&[2.8, 0.]).unwrap(), vec![2]);
  let mut r = Rng(5);
  for n in 1..=4 {
    for nb in [5, 50, 2000] {
      let (mut bl, mut bh) = (vec![], vec![]);
      let mut t = RTree::new(n, 0.0);
      for i in 0..nb {
        let l: Vec<f64> = (0..n).map(|_| r.f()).collect();
        let h: Vec<f64> = l.iter().map(|v| v + r.f() * 0.1).collect();
        t.add_box(&l, &h, i).unwrap();
        bl.push(l);
        bh.push(h);
      }
      rtree_verify(&bl, &bh, &t, &mut r);
      for i in 0..nb {
        assert!(t.find_boxes_at_point(&bl[i]).unwrap().contains(&i));
        assert!(t.find_boxes_at_point(&bh[i]).unwrap().contains(&i));
        assert!(t.find_contained_boxes(&bl[i], &bh[i]).unwrap().contains(&i));
        assert!(
          t.find_containing_boxes(&bl[i], &bh[i])
            .unwrap()
            .contains(&i)
        );
      }
    }
  }
}

#[test]
fn rtree_relative_eps() {
  for sc in [1e-3, 1.0, 1e3] {
    let mut t = RTree::new(1, 1e-10);
    t.add_box(&[0.0], &[0.3 * sc], 0).unwrap();
    t.add_box(&[0.3 * sc], &[1.0 * sc], 1).unwrap();
    // 3 * 0.1 != 0.3: found through the relative tolerance at every scale
    assert_eq!(
      t.find_boxes_at_point(&[3.0 * 0.1 * sc]).unwrap(),
      vec![0, 1]
    );
    assert!(t.add_box(&[1.0], &[0.0], 2).is_err());
  }
}

#[test]
fn rtree_lines() {
  // unit boxes of a 6x6 grid; line y = x + 0.5 never touches a corner
  let mut t = RTree::new(2, 0.0);
  let mut boxes = vec![];
  for i in 0..6 {
    for j in 0..6 {
      t.add_box(
        &[i as f64, j as f64],
        &[i as f64 + 1.0, j as f64 + 1.0],
        boxes.len(),
      )
      .unwrap();
      boxes.push((i as f64, j as f64));
    }
  }
  let hit = |&(i, j): &(f64, f64)| {
    (0..=70000).any(|s| {
      let x = -0.5 + s as f64 * 1e-4;
      let y = x + 0.5;
      x >= i && x <= i + 1.0 && y >= j && y <= j + 1.0
    })
  };
  let expect: Vec<usize> = (0..boxes.len()).filter(|&k| hit(&boxes[k])).collect();
  assert_eq!(
    t.find_line_intersecting_boxes(&[0.0, 0.5], &[1.0, 1.0], None)
      .unwrap(),
    expect
  );
  // clipped to the lower-left 3x3 block
  let clipped: Vec<usize> = expect
    .iter()
    .copied()
    .filter(|&k| boxes[k].0 <= 3.0 && boxes[k].1 <= 3.0)
    .collect();
  assert_eq!(
    t.find_line_intersecting_boxes(&[0.0, 0.5], &[1.0, 1.0], Some((&[0.0, 0.0], &[3.0, 3.0])))
      .unwrap(),
    clipped
  );
  // axis-parallel line x = 2.5: the column i = 2
  let col: Vec<usize> = (0..boxes.len()).filter(|&k| boxes[k].0 == 2.0).collect();
  assert_eq!(
    t.find_line_intersecting_boxes(&[2.5, -1.0], &[0.0, 1.0], None)
      .unwrap(),
    col
  );
}
