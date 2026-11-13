//! Cubature cells of a 2D element cut by level sets (`mesh_im_level_set`):
//! a lattice of triangles in reference coordinates; a lattice triangle cut
//! by one level set becomes a few 6-node P2 triangles following the rules
//! G1-G6 (design doc m12, section 3); an unresolved one (an island, the
//! curve crossing two edges twice, several level sets, a large cap) is
//! split in 4, at most `MAX_DEPTH` levels deep. At the depth limit an edge
//! crossed an even number of times counts as not crossed and islands are
//! ignored; one level set left: G3 / G4; several: the triangle clipped by
//! each into loops of quadratic pieces, every selected loop a fan of cells.

pub(crate) type P2 = [f64; 2];

pub(crate) fn sub(a: P2, b: P2) -> P2 {
  [a[0] - b[0], a[1] - b[1]]
}

pub(crate) fn add(a: P2, b: P2) -> P2 {
  [a[0] + b[0], a[1] + b[1]]
}

pub(crate) fn scal(s: f64, a: P2) -> P2 {
  [s * a[0], s * a[1]]
}

pub(crate) fn lerp(a: P2, b: P2, s: f64) -> P2 {
  [a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1])]
}

pub(crate) fn cross(a: P2, b: P2) -> f64 {
  a[0] * b[1] - a[1] * b[0]
}

pub(crate) fn norm(a: P2) -> f64 {
  a[0].hypot(a[1])
}

/// The root of g in the bracket [lo, hi] (g(lo) = glo, g(hi) = ghi of
/// opposite signs, or glo = 0): Illinois regula falsi, bisection when it
/// stalls.
pub(crate) fn illinois(
  g: &dyn Fn(f64) -> f64,
  mut lo: f64,
  mut hi: f64,
  mut glo: f64,
  mut ghi: f64,
) -> f64 {
  if glo == 0.0 {
    return lo;
  }
  let mut side = 0;
  for _ in 0..100 {
    let mut m = if ghi != glo {
      (lo * ghi - hi * glo) / (ghi - glo)
    } else {
      0.5 * (lo + hi)
    };
    if !(m > lo.min(hi) && m < lo.max(hi)) {
      m = 0.5 * (lo + hi);
    }
    let gm = g(m);
    if gm == 0.0 || (hi - lo).abs() <= 1e-15 * (1.0 + lo.abs().max(hi.abs())) {
      return m;
    }
    if (gm < 0.0) == (glo < 0.0) {
      lo = m;
      glo = gm;
      if side == -1 {
        ghi *= 0.5;
      }
      side = -1;
    } else {
      hi = m;
      ghi = gm;
      if side == 1 {
        glo *= 0.5;
      }
      side = 1;
    }
  }
  0.5 * (lo + hi)
}

/// The point of the zero set on the line through c along n nearest to c.
pub(crate) fn project(f: &dyn Fn(P2) -> f64, c: P2, n: P2, range: f64) -> Option<P2> {
  let m = 32;
  let at = |t: f64| f(add(c, scal(t, n)));
  let mut best: Option<f64> = None;
  for k in 0..m {
    for sgn in [1.0, -1.0] {
      let (t0, t1) = (
        sgn * range * k as f64 / m as f64,
        sgn * range * (k + 1) as f64 / m as f64,
      );
      let (f0, f1) = (at(t0), at(t1));
      if (f0 < 0.0) != (f1 < 0.0) || f0 == 0.0 {
        let t = crate::cut2d::illinois(&at, t0, t1, f0, f1);
        if best.is_none_or(|b: f64| t.abs() < b.abs()) {
          best = Some(t);
        }
      }
    }
    if best.is_some() {
      break;
    }
  }
  best.map(|t| add(c, scal(t, n)))
}

/// Deepest split of a lattice triangle.
pub(crate) const MAX_DEPTH: usize = 3;

/// A function in reference coordinates with its exact derivatives (the
/// level set on an element).
pub(crate) trait CutFunction {
  fn val(&self, x: P2) -> f64;
  fn grad(&self, x: P2) -> P2;
  fn hess(&self, x: P2) -> [[f64; 2]; 2];
}

fn dot(a: P2, b: P2) -> f64 {
  a[0] * b[0] + a[1] * b[1]
}

/// A quadratic curve: start, mid-node (at s = 1/2), end.
pub(crate) type Quad = [P2; 3];

pub(crate) fn quad_at(q: &Quad, s: f64) -> P2 {
  let [a, m, b] = *q;
  let (la, lm, lb) = (
    (1.0 - s) * (1.0 - 2.0 * s),
    4.0 * s * (1.0 - s),
    s * (2.0 * s - 1.0),
  );
  [
    la * a[0] + lm * m[0] + lb * b[0],
    la * a[1] + lm * m[1] + lb * b[1],
  ]
}

pub(crate) fn quad_d(q: &Quad, s: f64) -> P2 {
  let [a, m, b] = *q;
  let (da, dm, db) = (4.0 * s - 3.0, 4.0 - 8.0 * s, 4.0 * s - 1.0);
  [
    da * a[0] + dm * m[0] + db * b[0],
    da * a[1] + dm * m[1] + db * b[1],
  ]
}

/// The piece [s0, s1] of a quadratic (itself a quadratic).
fn piece(q: &Quad, s0: f64, s1: f64) -> Quad {
  [quad_at(q, s0), quad_at(q, 0.5 * (s0 + s1)), quad_at(q, s1)]
}

/// A 6-node P2 triangle: vertices a, b, c, mid-nodes of a-b, b-c, c-a.
#[derive(Clone, Copy, Debug)]
pub(crate) struct P2Cell {
  pub n: [P2; 6],
}

impl P2Cell {
  /// Straight sides where a mid-node is None.
  pub fn new(a: P2, b: P2, c: P2, mab: Option<P2>, mbc: Option<P2>, mca: Option<P2>) -> Self {
    P2Cell {
      n: [
        a,
        b,
        c,
        mab.unwrap_or(lerp(a, b, 0.5)),
        mbc.unwrap_or(lerp(b, c, 0.5)),
        mca.unwrap_or(lerp(c, a, 0.5)),
      ],
    }
  }
  pub fn straight(a: P2, b: P2, c: P2) -> Self {
    Self::new(a, b, c, None, None, None)
  }
  fn shape(xi: P2) -> ([f64; 6], [f64; 6], [f64; 6]) {
    let (l1, l2) = (xi[0], xi[1]);
    let l0 = 1.0 - l1 - l2;
    (
      [
        l0 * (2.0 * l0 - 1.0),
        l1 * (2.0 * l1 - 1.0),
        l2 * (2.0 * l2 - 1.0),
        4.0 * l0 * l1,
        4.0 * l1 * l2,
        4.0 * l2 * l0,
      ],
      [
        1.0 - 4.0 * l0,
        4.0 * l1 - 1.0,
        0.0,
        4.0 * (l0 - l1),
        4.0 * l2,
        -4.0 * l2,
      ],
      [
        1.0 - 4.0 * l0,
        0.0,
        4.0 * l2 - 1.0,
        -4.0 * l1,
        4.0 * l1,
        4.0 * (l0 - l2),
      ],
    )
  }
  /// The point of reference coordinates xi of the cell's reference triangle.
  pub fn map(&self, xi: P2) -> P2 {
    let (n, _, _) = Self::shape(xi);
    (0..6).fold([0.0, 0.0], |s, k| add(s, scal(n[k], self.n[k])))
  }
  /// Jacobian determinant of the map at xi.
  pub fn jac(&self, xi: P2) -> f64 {
    let (_, d1, d2) = Self::shape(xi);
    let g1 = (0..6).fold([0.0, 0.0], |s, k| add(s, scal(d1[k], self.n[k])));
    let g2 = (0..6).fold([0.0, 0.0], |s, k| add(s, scal(d2[k], self.n[k])));
    cross(g1, g2)
  }
}

/// One side of a level set: f = s * level set (its inside: f < 0).
#[derive(Clone, Copy)]
pub(crate) struct Side<'a> {
  pub f: &'a dyn CutFunction,
  pub s: f64,
}

impl Side<'_> {
  fn val(&self, x: P2) -> f64 {
    self.s * self.f.val(x)
  }
}

/// Sub-intervals of an edge for its roots.
const NBRACKET: usize = 32;

/// The parameters in (0, 1) where f changes sign on a-b (f = 0 counts as
/// outside); the samples are split at the extrema of f first (two close
/// roots are not missed).
pub(crate) fn roots(f: &dyn CutFunction, a: P2, b: P2) -> Vec<f64> {
  let d = sub(b, a);
  let g = |t: f64| f.val(lerp(a, b, t));
  let dg = |t: f64| dot(f.grad(lerp(a, b, t)), d);
  let s: Vec<f64> = (0..=NBRACKET).map(|k| k as f64 / NBRACKET as f64).collect();
  let gv: Vec<f64> = s.iter().map(|&t| g(t)).collect();
  let mut br = vec![(s[0], gv[0])];
  for i in 1..NBRACKET {
    let (d0, d1) = (gv[i] - gv[i - 1], gv[i + 1] - gv[i]);
    if (d0 < 0.0) != (d1 < 0.0) && d0 != 0.0 && d1 != 0.0 {
      let (x0, x1) = (dg(s[i - 1]), dg(s[i + 1]));
      let t = if (x0 < 0.0) == (x1 < 0.0) {
        s[i]
      } else {
        illinois(&dg, s[i - 1], s[i + 1], x0, x1)
      };
      br.push((t, g(t)));
    }
  }
  br.push((1.0, gv[NBRACKET]));
  let mut out = vec![];
  for w in br.windows(2) {
    let ((t0, g0), (t1, g1)) = (w[0], w[1]);
    if (g0 < 0.0) != (g1 < 0.0) {
      let t = illinois(&g, t0, t1, g0, g1);
      if t > 0.0 && t < 1.0 {
        out.push(t);
      }
    }
  }
  out
}

/// Mid-node on the zero set: the chord midpoint moved along the chord's
/// normal to the nearest root (within the chord's length).
fn mid_on(f: &dyn CutFunction, a: P2, b: P2) -> P2 {
  let m = lerp(a, b, 0.5);
  let d = sub(b, a);
  let l = norm(d);
  if l == 0.0 {
    return m;
  }
  project(&|x| f.val(x), m, [-d[1] / l, d[0] / l], l).unwrap_or(m)
}

/// Mid-node of a cap on the edge a-b: the chord midpoint moved towards the
/// triangle's third vertex w to the zero set.
fn mid_into(f: &dyn CutFunction, a: P2, b: P2, w: P2) -> P2 {
  let m = lerp(a, b, 0.5);
  let d = sub(b, a);
  let mut n = scal(1.0 / norm(d), [-d[1], d[0]]);
  if dot(n, sub(w, m)) < 0.0 {
    n = scal(-1.0, n);
  }
  let l = norm(sub(w, m));
  let g = |t: f64| f.val(add(m, scal(t, n)));
  let (mut t0, mut g0) = (0.0, g(0.0));
  for j in 1..=4 * NBRACKET {
    let t1 = l * j as f64 / (4 * NBRACKET) as f64;
    let g1 = g(t1);
    if (g0 < 0.0) != (g1 < 0.0) {
      return add(m, scal(illinois(&g, t0, t1, g0, g1), n));
    }
    (t0, g0) = (t1, g1);
  }
  m
}

/// An island: f has an extremum inside the triangle v (ccw) with the
/// sign opposite to the vertices' (no crossings assumed).
fn island(f: &dyn CutFunction, v: &[P2; 3]) -> bool {
  let c = scal(1.0 / 3.0, add(add(v[0], v[1]), v[2]));
  let fv = f.val(v[0]);
  let mut x = c;
  for _ in 0..30 {
    let (g, h) = (f.grad(x), f.hess(x));
    let det = h[0][0] * h[1][1] - h[0][1] * h[1][0];
    if det == 0.0 {
      return false;
    }
    let d = [
      (h[1][1] * g[0] - h[0][1] * g[1]) / det,
      (h[0][0] * g[1] - h[1][0] * g[0]) / det,
    ];
    x = sub(x, d);
    if norm(d) < 1e-14 {
      break;
    }
  }
  let inside = (0..3).all(|k| cross(sub(v[(k + 1) % 3], v[k]), sub(x, v[k])) > 0.0);
  inside && ((f.val(x) < 0.0) != (fv < 0.0))
}

/// u in [0, 1] minimizing n.(C(u) - p) (a quadratic in u).
fn closest_u(c: &Quad, p: P2, n: P2) -> f64 {
  let [a, m, b] = *c;
  let c1 = dot(n, sub(sub(scal(4.0, m), scal(3.0, a)), b));
  let c2 = dot(n, add(sub(scal(2.0, a), scal(4.0, m)), scal(2.0, b)));
  let mut cands = vec![0.0, 1.0];
  if c2 > 0.0 {
    cands.push((-c1 / (2.0 * c2)).clamp(0.0, 1.0));
  }
  cands
    .into_iter()
    .min_by(|x, y| dot(n, sub(quad_at(c, *x), p)).total_cmp(&dot(n, sub(quad_at(c, *y), p))))
    .unwrap()
}

/// K = mid(v1, v2) + lam nout, lam the smallest with the parabola v1-K-v2
/// outside the side's inside (f >= 0 on it): bisection on lam, the
/// parabola's largest -f by samples and golden section; then Newton on the
/// tangency with C: G(s) = C(u), tangents parallel. Returns (G, s, u).
fn touch(side: &Side, v1: P2, v2: P2, nout: P2, c: &Quad) -> (Quad, f64, f64) {
  let mid = lerp(v1, v2, 0.5);
  let par = |lam: f64| [v1, add(mid, scal(lam, nout)), v2];
  let worst = |lam: f64| -> (f64, f64) {
    let p = par(lam);
    let ss: Vec<f64> = (0..=200).map(|k| k as f64 / 200.0).collect();
    let k = (0..ss.len())
      .max_by(|i, j| (-side.val(quad_at(&p, ss[*i]))).total_cmp(&-side.val(quad_at(&p, ss[*j]))))
      .unwrap();
    let (mut a, mut b) = (ss[k.saturating_sub(1)], ss[(k + 1).min(ss.len() - 1)]);
    let gr = 0.5 * (5f64.sqrt() - 1.0);
    for _ in 0..80 {
      let (c1, c2) = (b - gr * (b - a), a + gr * (b - a));
      if -side.val(quad_at(&p, c1)) > -side.val(quad_at(&p, c2)) {
        b = c2;
      } else {
        a = c1;
      }
    }
    let s = 0.5 * (a + b);
    (-side.val(quad_at(&p, s)), s)
  };
  let (mut lo, mut hi) = (0.0, norm(sub(v2, v1)));
  for _ in 0..100 {
    let m = 0.5 * (lo + hi);
    if worst(m).0 > 0.0 {
      lo = m;
    } else {
      hi = m;
    }
  }
  let (_, s0) = worst(hi);
  let m0 = quad_at(&par(hi), s0);
  let u0 = (0..=200)
    .map(|k| k as f64 / 200.0)
    .min_by(|x, y| norm(sub(quad_at(c, *x), m0)).total_cmp(&norm(sub(quad_at(c, *y), m0))))
    .unwrap();
  // Newton: unknowns lam, s, u
  let cpp = add(sub(scal(4.0, c[0]), scal(8.0, c[1])), scal(4.0, c[2]));
  let mut x = [hi, s0, u0];
  for _ in 0..60 {
    let (lam, s, u) = (x[0], x[1], x[2]);
    let g = par(lam);
    let (gs, gd) = (quad_at(&g, s), quad_d(&g, s));
    let (cu, cd) = (quad_at(c, u), quad_d(c, u));
    let gpp = add(sub(scal(4.0, v1), scal(8.0, g[1])), scal(4.0, v2));
    let f = [gs[0] - cu[0], gs[1] - cu[1], cross(gd, cd)];
    let w = 4.0 * s * (1.0 - s);
    let j = [
      [w * nout[0], gd[0], -cd[0]],
      [w * nout[1], gd[1], -cd[1]],
      [
        cross(scal(4.0 - 8.0 * s, nout), cd),
        cross(gpp, cd),
        cross(gd, cpp),
      ],
    ];
    let Some(dx) = solve3(j, f) else { break };
    for k in 0..3 {
      x[k] -= dx[k];
    }
    if dx.iter().map(|v| v.abs()).fold(0.0, f64::max) < 1e-15 {
      break;
    }
  }
  (par(x[0]), x[1], x[2])
}

/// Solution of j x = f (Cramer).
fn solve3(j: [[f64; 3]; 3], f: [f64; 3]) -> Option<[f64; 3]> {
  let det = |m: [[f64; 3]; 3]| {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
      - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
      + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
  };
  let d = det(j);
  if d == 0.0 {
    return None;
  }
  let mut out = [0.0; 3];
  for (k, o) in out.iter_mut().enumerate() {
    let mut m = j;
    for r in 0..3 {
      m[r][k] = f[r];
    }
    *o = det(m) / d;
  }
  Some(out)
}

/// The cells of a lattice triangle v (ccw) with one or two vertices on the
/// side's inside (G3 / G4); `partner`: the neighbour across v1-v2 takes the
/// bulge (G3s with a G5, G4s with a G6), else the bulge is clipped at v1-v2.
/// Returns the cells, the curve's pieces on the zero set, and for G4s the
/// mid-node K of the neighbour's curved side.
pub(crate) fn cut_cells(
  side: &Side,
  v: [P2; 3],
  partner: bool,
) -> (Vec<P2Cell>, Vec<Quad>, Option<P2>) {
  let ins: Vec<bool> = v.iter().map(|p| side.val(*p) < 0.0).collect();
  let n = ins.iter().filter(|b| **b).count();
  let k0 = if n == 1 {
    ins.iter().position(|b| *b).unwrap()
  } else {
    ins.iter().position(|b| !*b).unwrap()
  };
  let (v0, v1, v2) = (v[k0], v[(k0 + 1) % 3], v[(k0 + 2) % 3]);
  let f = side.f;
  let at = |a: P2, b: P2| -> P2 {
    let r = roots(f, a, b);
    let t = match r.first() {
      Some(t) => *t,
      None => {
        if side.val(a).abs() < side.val(b).abs() {
          0.0
        } else {
          1.0
        }
      }
    };
    lerp(a, b, t)
  };
  let (p, q) = (at(v0, v1), at(v0, v2));
  let c: Quad = [p, mid_on(f, p, q), q];
  let e = sub(v2, v1);
  let mut nn = scal(1.0 / norm(e), [-e[1], e[0]]);
  if dot(nn, sub(v0, v1)) < 0.0 {
    nn = scal(-1.0, nn);
  }
  let across = roots(f, v1, v2);
  if across.len() == 2 && !partner {
    // clipped at v1-v2
    let (e1, e2) = (lerp(v1, v2, across[0]), lerp(v1, v2, across[1]));
    let (pe, eq) = ([p, mid_on(f, p, e1), e1], [e2, mid_on(f, e2, q), q]);
    let cells = if n == 1 {
      vec![
        P2Cell::new(v0, p, e1, None, Some(pe[1]), None),
        P2Cell::straight(v0, e1, e2),
        P2Cell::new(v0, e2, q, None, Some(eq[1]), None),
      ]
    } else {
      vec![
        P2Cell::new(v1, p, e1, None, Some(pe[1]), None),
        P2Cell::new(v2, e2, q, None, Some(eq[1]), None),
      ]
    };
    return (cells, vec![pe, eq], None);
  }
  if n == 1 {
    return (
      vec![P2Cell::new(v0, p, q, None, Some(c[1]), None)],
      vec![c],
      None,
    );
  }
  let toward = dot(nn, sub(c[1], v1)) < dot(nn, sub(lerp(p, q, 0.5), v1));
  let u = if toward { closest_u(&c, v1, nn) } else { 0.5 };
  let m = quad_at(&c, u);
  if across.len() != 2 {
    let (pm, mq) = (piece(&c, 0.0, u), piece(&c, u, 1.0));
    let cells = vec![
      P2Cell::straight(m, v1, v2),
      P2Cell::new(v1, p, m, None, Some(pm[1]), None),
      P2Cell::new(v2, m, q, None, Some(mq[1]), None),
    ];
    return (cells, vec![c], None);
  }
  // G4s: the neighbour's side v1-K-v2 touches C at M
  let (g, s, uu) = touch(side, v1, v2, scal(-1.0, nn), &c);
  let m = quad_at(&g, s);
  let cells = vec![
    P2Cell::new(
      v1,
      m,
      p,
      Some(piece(&g, 0.0, s)[1]),
      Some(piece(&c, uu, 0.0)[1]),
      None,
    ),
    P2Cell::new(
      q,
      m,
      v2,
      Some(piece(&c, 1.0, uu)[1]),
      Some(piece(&g, s, 1.0)[1]),
      None,
    ),
  ];
  (cells, vec![c], Some(g[1]))
}

/// What a lattice triangle is for the selection.
enum Class {
  All,
  None,
  Unresolved,
  /// cut by level set k, its side s selected; n vertices inside; the edge
  /// crossed twice (a cap or a bulge), if any
  Cut {
    k: usize,
    s: f64,
    n: usize,
    two: Option<usize>,
  },
}

/// The cells of the selected part of lattice triangles, and the curve's
/// pieces bounding it with their level set.
#[derive(Default)]
pub(crate) struct Cells {
  pub cells: Vec<P2Cell>,
  pub curve: Vec<(Quad, usize)>,
  /// triangles split, per level
  pub split: [usize; MAX_DEPTH + 1],
  /// triangles at the depth limit
  pub limit: usize,
  /// fans at the depth limit with a folding cell (the least folded taken)
  pub fold: usize,
}

/// A cap on the edge a-b (third vertex w) is small: the curve is a graph
/// over the edge and turns by at most 90 degrees.
fn small_cap(f: &dyn CutFunction, a: P2, b: P2, w: P2) -> bool {
  let r = roots(f, a, b);
  let (p, q) = (lerp(a, b, r[0]), lerp(a, b, r[1]));
  let d = sub(q, p);
  let mut n_in = [-d[1], d[0]];
  if dot(n_in, sub(w, p)) < 0.0 {
    n_in = scal(-1.0, n_in);
  }
  let tangent = |x: P2, into: bool| {
    let g = f.grad(x);
    let t = [-g[1], g[0]];
    if (dot(n_in, t) > 0.0) == into {
      t
    } else {
      scal(-1.0, t)
    }
  };
  let (tp, tq) = (tangent(p, true), tangent(q, false));
  dot(tp, d) > 0.0 && dot(tq, d) > 0.0 && dot(tp, tq) >= 0.0
}

/// The roots seen at the depth limit: an even number ignored, the middle
/// one of an odd number.
fn coarse(r: Vec<f64>) -> Vec<f64> {
  if r.len().is_multiple_of(2) {
    vec![]
  } else {
    vec![r[r.len() / 2]]
  }
}

/// The point of ps farthest from the zero sets of fs (the first on ties).
fn farthest(ps: impl Iterator<Item = P2>, fs: &[&dyn CutFunction]) -> P2 {
  let d = |p: P2| {
    fs.iter()
      .map(|f| f.val(p).abs())
      .fold(f64::INFINITY, f64::min)
  };
  let mut best: Option<(f64, P2)> = None;
  for p in ps {
    let dp = d(p);
    if best.is_none_or(|(b, _)| dp > b) {
      best = Some((dp, p));
    }
  }
  best.unwrap().1
}

/// The class of a lattice triangle; at the depth limit (`limit`) with the
/// roots `coarse`, no islands, the level sets not crossing it evaluated at
/// its vertex farthest from the zero sets (else at its centroid).
fn classify(
  v: &[P2; 3],
  fs: &[&dyn CutFunction],
  sel: &dyn Fn(&[bool]) -> bool,
  limit: bool,
) -> Class {
  let cen = if limit {
    farthest(v.iter().copied(), fs)
  } else {
    scal(1.0 / 3.0, add(add(v[0], v[1]), v[2]))
  };
  let mut inside: Vec<bool> = fs.iter().map(|f| f.val(cen) < 0.0).collect();
  let mut crossed = vec![];
  for (k, f) in fs.iter().enumerate() {
    let ins: Vec<bool> = v.iter().map(|p| f.val(*p) < 0.0).collect();
    let nr: Vec<usize> = (0..3)
      .map(|j| {
        let r = roots(*f, v[j], v[(j + 1) % 3]);
        if limit { coarse(r).len() } else { r.len() }
      })
      .collect();
    let none = nr.iter().all(|r| *r == 0) && ins.iter().all(|b| *b == ins[0]);
    let isl = none && !limit && island(*f, v);
    if !none || isl {
      crossed.push((k, ins, nr, isl));
    }
  }
  // only the level sets the selection depends on (flipping one changes it
  // for some insides of the other crossing ones)
  let ks: Vec<usize> = crossed.iter().map(|c| c.0).collect();
  let relevant = |k: usize| -> bool {
    let others: Vec<usize> = ks.iter().copied().filter(|j| *j != k).collect();
    (0..1usize << others.len()).any(|bits| {
      let mut ins = inside.clone();
      for (b, j) in others.iter().enumerate() {
        ins[*j] = bits >> b & 1 == 1;
      }
      ins[k] = true;
      let a = sel(&ins);
      ins[k] = false;
      a != sel(&ins)
    })
  };
  crossed.retain(|c| relevant(c.0));
  if crossed.is_empty() {
    return if sel(&inside) {
      Class::All
    } else {
      Class::None
    };
  }
  if crossed.len() > 1 {
    return Class::Unresolved;
  }
  let (k, ins, nr, isl) = crossed.pop().unwrap();
  inside[k] = true;
  let a = sel(&inside);
  inside[k] = false;
  let b = sel(&inside);
  if a == b {
    return if a { Class::All } else { Class::None };
  }
  // an island, two edges crossed twice, or an edge crossed 3 times or more
  if isl || nr.iter().filter(|r| **r == 2).count() >= 2 || nr.iter().any(|r| *r >= 3) {
    return Class::Unresolved;
  }
  let s = if a { 1.0 } else { -1.0 };
  let n = ins.iter().filter(|i| **i == a).count();
  let two = (0..3).find(|j| nr[*j] == 2);
  if let Some(j) = two
    && (n == 0 || n == 3)
    && !small_cap(fs[k], v[j], v[(j + 1) % 3], v[(j + 2) % 3])
  {
    return Class::Unresolved;
  }
  Class::Cut { k, s, n, two }
}

/// The cells of a triangle at the depth limit (crossings `coarse`): G3 /
/// G4 for one level set (M at the curve's middle), clipping for several.
fn limit_cells(
  t: &[P2; 3],
  fs: &[&dyn CutFunction],
  sel: &dyn Fn(&[bool]) -> bool,
  out: &mut Cells,
) {
  out.limit += 1;
  let (k, s, n) = match classify(t, fs, sel, true) {
    Class::All => return out.cells.push(P2Cell::straight(t[0], t[1], t[2])),
    Class::None => return,
    Class::Unresolved => return clip_fan(t, fs, sel, out),
    Class::Cut { k, s, n, .. } => (k, s, n),
  };
  let side = Side { f: fs[k], s };
  let ins: Vec<bool> = t.iter().map(|p| side.val(*p) < 0.0).collect();
  let k0 = ins.iter().position(|b| *b == (n == 1)).unwrap();
  let (v0, v1, v2) = (t[k0], t[(k0 + 1) % 3], t[(k0 + 2) % 3]);
  let root = |b: P2| {
    let r = coarse(roots(fs[k], v0, b));
    lerp(v0, b, r.first().copied().unwrap_or(0.5))
  };
  let (p, q) = (root(v1), root(v2));
  let c: Quad = [p, mid_on(fs[k], p, q), q];
  if n == 1 {
    out
      .cells
      .push(P2Cell::new(v0, p, q, None, Some(c[1]), None));
  } else {
    let m = quad_at(&c, 0.5);
    let (h1, h2) = (piece(&c, 0.0, 0.5), piece(&c, 0.5, 1.0));
    out.cells.push(P2Cell::straight(m, v1, v2));
    out
      .cells
      .push(P2Cell::new(v1, p, m, None, Some(h1[1]), None));
    out
      .cells
      .push(P2Cell::new(v2, m, q, None, Some(h2[1]), None));
  }
  out.curve.push((c, k));
}

/// A piece of a loop: a quadratic on level set k (None: straight).
#[derive(Clone, Copy)]
struct Piece {
  q: Quad,
  k: Option<usize>,
}

/// A loop of pieces (ccw) and the insides of all the level sets.
type Loop = (Vec<Piece>, Vec<bool>);

/// The point where the zero sets of fa and fb cross, by Newton from x.
fn crossing(fa: &dyn CutFunction, fb: &dyn CutFunction, mut x: P2) -> Option<P2> {
  for _ in 0..50 {
    let (ga, gb) = (fa.grad(x), fb.grad(x));
    let det = cross(ga, gb);
    if det.abs() < 1e-300 {
      return None;
    }
    let (va, vb) = (fa.val(x), fb.val(x));
    let dx = scal(
      1.0 / det,
      [va * gb[1] - vb * ga[1], vb * ga[0] - va * gb[0]],
    );
    x = sub(x, dx);
    if norm(dx) < 1e-15 * (1.0 + norm(x)) {
      break;
    }
  }
  Some(x)
}

/// The parameter where f changes sign along the piece, `coarse` (none on
/// the piece's own level set k).
fn piece_root(f: &dyn CutFunction, p: &Piece, k: usize) -> Option<f64> {
  if p.k == Some(k) {
    return None;
  }
  let r = match p.k {
    None => roots(f, p.q[0], p.q[2]),
    Some(_) => {
      let g = |t: f64| f.val(quad_at(&p.q, t));
      let gv: Vec<f64> = (0..=NBRACKET)
        .map(|i| g(i as f64 / NBRACKET as f64))
        .collect();
      (0..NBRACKET)
        .filter(|&i| (gv[i] < 0.0) != (gv[i + 1] < 0.0))
        .map(|i| {
          let (t0, t1) = (i as f64 / NBRACKET as f64, (i + 1) as f64 / NBRACKET as f64);
          illinois(&g, t0, t1, gv[i], gv[i + 1])
        })
        .collect()
    }
  };
  let r: Vec<f64> = r
    .into_iter()
    .filter(|t| *t > 1e-12 && *t < 1.0 - 1e-12)
    .collect();
  coarse(r).first().copied()
}

/// The piece split at its point x: curved halves re-fitted on their level
/// set.
fn split_piece(p: &Piece, x: P2, fs: &[&dyn CutFunction]) -> (Piece, Piece) {
  let [a, _, b] = p.q;
  let mid = |u: P2, w: P2| match p.k {
    None => lerp(u, w, 0.5),
    Some(kp) => mid_on(fs[kp], u, w),
  };
  (
    Piece {
      q: [a, mid(a, x), x],
      k: p.k,
    },
    Piece {
      q: [x, mid(x, b), b],
      k: p.k,
    },
  )
}

/// The inside of f for a loop: at its vertex farthest from the zero set.
fn far_inside(f: &dyn CutFunction, lp: &[Piece]) -> bool {
  f.val(farthest(lp.iter().map(|p| p.q[0]), &[f])) < 0.0
}

/// The loops split by the zero set of level set k where it crosses one
/// twice; a loop crossed otherwise keeps its pieces.
fn clip(loops: Vec<Loop>, k: usize, fs: &[&dyn CutFunction]) -> Vec<Loop> {
  let f = fs[k];
  let mut out = vec![];
  for (lp, mut fl) in loops {
    let cr: Vec<(usize, f64)> = lp
      .iter()
      .enumerate()
      .filter_map(|(i, p)| piece_root(f, p, k).map(|t| (i, t)))
      .collect();
    if cr.len() != 2 {
      fl[k] = far_inside(f, &lp);
      out.push((lp, fl));
      continue;
    }
    // the crossings; on another curve, exactly where the two meet
    let pt = |(i, t): (usize, f64)| {
      let p = &lp[i];
      let x = quad_at(&p.q, t);
      match p.k.and_then(|kp| crossing(fs[kp], f, x)) {
        Some(y) if norm(sub(y, x)) < 0.5 * norm(sub(p.q[2], p.q[0])) => y,
        _ => x,
      }
    };
    let ((i, _), (j, _)) = (cr[0], cr[1]);
    let (p, q) = (pt(cr[0]), pt(cr[1]));
    let (p1, p2) = split_piece(&lp[i], p, fs);
    let (q1, q2) = split_piece(&lp[j], q, fs);
    // p .. q along the loop, then the curve back; q .. p, the curve back
    let mut l1 = vec![p2];
    l1.extend_from_slice(&lp[i + 1..j]);
    l1.extend([
      q1,
      Piece {
        q: [q, mid_on(f, q, p), p],
        k: Some(k),
      },
    ]);
    let mut l2 = vec![q2];
    l2.extend_from_slice(&lp[j + 1..]);
    l2.extend_from_slice(&lp[..i]);
    l2.extend([
      p1,
      Piece {
        q: [p, mid_on(f, p, q), q],
        k: Some(k),
      },
    ]);
    for l in [l1, l2] {
      let mut g = fl.clone();
      g[k] = far_inside(f, &l);
      out.push((l, g));
    }
  }
  out
}

/// The fan of a loop from the start of its piece i.
fn fan_cells(lp: &[Piece], i: usize) -> Vec<P2Cell> {
  let n = lp.len();
  let v = lp[i].q[0];
  (1..n.saturating_sub(1))
    .map(|m| {
      let p = lp[(i + m) % n];
      let mva = (m == 1).then_some(lp[i].q[1]);
      let mbv = (m == n - 2).then_some(lp[(i + n - 1) % n].q[1]);
      P2Cell::new(v, p.q[0], p.q[2], mva, Some(p.q[1]), mbv)
    })
    .collect()
}

/// The fan of a loop: from a point where two curves meet, else from the
/// middle of the curve piece of the lowest level set, else from a vertex;
/// the first without folding, else the least folded (and true).
fn fan_loop(lp: &[Piece]) -> (Vec<P2Cell>, bool) {
  let n = lp.len();
  let mut tries: Vec<(Vec<Piece>, usize)> = vec![];
  for i in 0..n {
    if let (Some(a), Some(b)) = (lp[(i + n - 1) % n].k, lp[i].k)
      && a != b
    {
      tries.push((lp.to_vec(), i));
    }
  }
  if let Some((_, i)) = lp
    .iter()
    .enumerate()
    .filter_map(|(i, p)| p.k.map(|k| (k, i)))
    .min()
  {
    let p = lp[i];
    let mut l = lp[..i].to_vec();
    l.push(Piece {
      q: piece(&p.q, 0.0, 0.5),
      k: p.k,
    });
    l.push(Piece {
      q: piece(&p.q, 0.5, 1.0),
      k: p.k,
    });
    l.extend_from_slice(&lp[i + 1..]);
    tries.push((l, i + 1));
  }
  tries.extend((0..n).map(|i| (lp.to_vec(), i)));
  let xi = [
    [1.0 / 6.0, 1.0 / 6.0],
    [2.0 / 3.0, 1.0 / 6.0],
    [1.0 / 6.0, 2.0 / 3.0],
    [1.0 / 3.0, 1.0 / 3.0],
  ];
  let mut best: Option<(f64, Vec<P2Cell>)> = None;
  for (l, i) in tries {
    let cells = fan_cells(&l, i);
    if cells.is_empty() {
      continue;
    }
    let mj = cells
      .iter()
      .flat_map(|c| xi.iter().map(|x| c.jac(*x)))
      .fold(f64::INFINITY, f64::min);
    if mj > 0.0 {
      return (cells, false);
    }
    if best.as_ref().is_none_or(|(b, _)| mj > *b) {
      best = Some((mj, cells));
    }
  }
  (best.map(|b| b.1).unwrap_or_default(), true)
}

/// Several level sets at the depth limit: the triangle clipped by each
/// relevant one (crossing it) into loops, every selected loop a fan; the
/// curve pieces between a selected and an unselected loop bound the part.
fn clip_fan(t: &[P2; 3], fs: &[&dyn CutFunction], sel: &dyn Fn(&[bool]) -> bool, out: &mut Cells) {
  let at = farthest(t.iter().copied(), fs);
  let base: Vec<bool> = fs.iter().map(|f| f.val(at) < 0.0).collect();
  let ks: Vec<usize> = (0..fs.len())
    .filter(|&k| {
      let ins: Vec<bool> = t.iter().map(|p| fs[k].val(*p) < 0.0).collect();
      ins.iter().any(|b| *b != ins[0])
        || (0..3).any(|j| !roots(fs[k], t[j], t[(j + 1) % 3]).is_empty())
    })
    .collect();
  let relevant = |k: usize| {
    let others: Vec<usize> = ks.iter().copied().filter(|j| *j != k).collect();
    (0..1usize << others.len()).any(|bits| {
      let mut ins = base.clone();
      for (b, j) in others.iter().enumerate() {
        ins[*j] = bits >> b & 1 == 1;
      }
      ins[k] = true;
      let a = sel(&ins);
      ins[k] = false;
      a != sel(&ins)
    })
  };
  let ks: Vec<usize> = ks.iter().copied().filter(|k| relevant(*k)).collect();
  let lp: Vec<Piece> = (0..3)
    .map(|j| {
      let (a, b) = (t[j], t[(j + 1) % 3]);
      Piece {
        q: [a, lerp(a, b, 0.5), b],
        k: None,
      }
    })
    .collect();
  let mut loops = vec![(lp, base)];
  for k in ks {
    loops = clip(loops, k, fs);
  }
  for (lp, fl) in loops {
    if !sel(&fl) {
      continue;
    }
    let (cells, folded) = fan_loop(&lp);
    out.cells.extend(cells);
    out.fold += usize::from(folded);
    for p in &lp {
      if let Some(k) = p.k {
        let mut g = fl.clone();
        g[k] = !g[k];
        if !sel(&g) {
          out.curve.push((p.q, k));
        }
      }
    }
  }
}

fn subdivide(v: &[P2; 3]) -> [[P2; 3]; 4] {
  let [a, b, c] = *v;
  let (ab, bc, ca) = (lerp(a, b, 0.5), lerp(b, c, 0.5), lerp(c, a, 0.5));
  [[a, ab, ca], [ab, b, bc], [ca, bc, c], [bc, ca, ab]]
}

type EdgeKey = [[i64; 2]; 2];

fn edge_key(a: P2, b: P2) -> EdgeKey {
  let r = |p: P2| [(p[0] * 1e12).round() as i64, (p[1] * 1e12).round() as i64];
  let (x, y) = (r(a), r(b));
  if x <= y { [x, y] } else { [y, x] }
}

/// The cells of the selected part of the lattice triangles tris (ccw, one
/// level), the depth of this level given.
pub(crate) fn build(
  tris: &[[P2; 3]],
  fs: &[&dyn CutFunction],
  sel: &dyn Fn(&[bool]) -> bool,
  depth: usize,
  out: &mut Cells,
) {
  let cls: Vec<Class> = tris.iter().map(|t| classify(t, fs, sel, false)).collect();
  let mut edges: std::collections::HashMap<EdgeKey, Vec<usize>> = Default::default();
  for (i, t) in tris.iter().enumerate() {
    for j in 0..3 {
      edges
        .entry(edge_key(t[j], t[(j + 1) % 3]))
        .or_default()
        .push(i);
    }
  }
  // the edge opposite the odd vertex of a G3 / G4 (v1-v2)
  let opposite = |t: &[P2; 3], side: &Side, n: usize| -> (P2, P2) {
    let ins: Vec<bool> = t.iter().map(|p| side.val(*p) < 0.0).collect();
    let k0 = if n == 1 {
      ins.iter().position(|b| *b).unwrap()
    } else {
      ins.iter().position(|b| !*b).unwrap()
    };
    (t[(k0 + 1) % 3], t[(k0 + 2) % 3])
  };
  // partners: a G3 / G4 bulging across v1-v2 and the cap (n = 0 / 3, the
  // same level set and side) across that edge
  let mut covered: std::collections::HashMap<EdgeKey, Option<P2>> = Default::default();
  let mut partner = vec![false; tris.len()];
  for (i, t) in tris.iter().enumerate() {
    let Class::Cut { k, s, n, .. } = cls[i] else {
      continue;
    };
    if n != 1 && n != 2 {
      continue;
    }
    let side = Side { f: fs[k], s };
    let (v1, v2) = opposite(t, &side, n);
    if roots(fs[k], v1, v2).len() != 2 {
      continue;
    }
    let key = edge_key(v1, v2);
    partner[i] = edges[&key].iter().any(|&j| {
      j != i && matches!(cls[j], Class::Cut { k: k2, s: s2, n: n2, .. } if k2 == k && s2 == s && n2 == if n == 1 { 0 } else { 3 })
    });
  }
  for (i, t) in tris.iter().enumerate() {
    match cls[i] {
      Class::All => out.cells.push(P2Cell::straight(t[0], t[1], t[2])),
      Class::None => {}
      Class::Unresolved => {
        if depth >= MAX_DEPTH {
          limit_cells(t, fs, sel, out);
        } else {
          out.split[depth] += 1;
          build(&subdivide(t), fs, sel, depth + 1, out);
        }
      }
      Class::Cut { k, s, n, .. } if n == 1 || n == 2 => {
        let side = Side { f: fs[k], s };
        let (cells, curve, kk) = cut_cells(&side, *t, partner[i]);
        out.cells.extend(cells);
        out.curve.extend(curve.into_iter().map(|c| (c, k)));
        if partner[i] {
          let (v1, v2) = opposite(t, &side, n);
          covered.insert(edge_key(v1, v2), kk);
        }
      }
      Class::Cut {
        n: 3, two: None, ..
      } => out.cells.push(P2Cell::straight(t[0], t[1], t[2])),
      Class::Cut { .. } => {}
    }
  }
  // caps (n = 0: inside cap; n = 3: outside cap): their partner's cells
  // cover them (G5 nothing; G6 the curved side v1-K-v2), else a cap
  // triangle / a fan from the opposite vertex
  for (i, t) in tris.iter().enumerate() {
    let Class::Cut {
      k, n, two: Some(j), ..
    } = cls[i]
    else {
      continue;
    };
    if n != 0 && n != 3 {
      continue;
    }
    let (a, b, w) = (t[j], t[(j + 1) % 3], t[(j + 2) % 3]);
    match covered.get(&edge_key(a, b)) {
      Some(None) if n == 0 => continue,
      Some(Some(kk)) if n == 3 => {
        out.cells.push(P2Cell::new(a, b, w, Some(*kk), None, None));
        continue;
      }
      _ => {}
    }
    let f = fs[k];
    let r = roots(f, a, b);
    let (p, q) = (lerp(a, b, r[0]), lerp(a, b, r[1]));
    let c: Quad = [p, mid_into(f, p, q, w), q];
    let m = quad_at(&c, 0.5);
    let (h1, h2) = (piece(&c, 0.0, 0.5), piece(&c, 0.5, 1.0));
    if n == 0 {
      out
        .cells
        .push(P2Cell::new(p, m, q, Some(h1[1]), Some(h2[1]), None));
    } else {
      out.cells.push(P2Cell::straight(w, a, p));
      out
        .cells
        .push(P2Cell::new(w, p, m, None, Some(h1[1]), None));
      out
        .cells
        .push(P2Cell::new(w, m, q, None, Some(h2[1]), None));
      out.cells.push(P2Cell::straight(w, q, b));
    }
    out.curve.push((c, k));
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn p2_cell_area() {
    // a straight cell: |J| = twice its area; a curved side a-b bulging by
    // d at its mid-node adds 2/3 * |ab| * d
    let c = P2Cell::straight([0.0, 0.0], [1.0, 0.0], [0.0, 1.0]);
    assert!((c.jac([0.2, 0.3]) - 1.0).abs() < 1e-15);
    let d = 0.1;
    let c = P2Cell::new(
      [0.0, 0.0],
      [1.0, 0.0],
      [0.0, 1.0],
      Some([0.5, -d]),
      None,
      None,
    );
    let (x, w) = (
      [
        [1.0 / 6.0, 1.0 / 6.0],
        [2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0],
      ],
      [1.0 / 6.0; 3],
    );
    let a: f64 = x.iter().zip(w).map(|(p, w)| w * c.jac(*p)).sum();
    assert!((a - (0.5 + 2.0 / 3.0 * d)).abs() < 1e-14, "{a}");
    assert_eq!(
      quad_at(&[[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]], 0.5),
      [0.5, 1.0]
    );
  }

  /// sg (|x - c|^2 - R^2) with its exact derivatives
  pub(super) struct Circle {
    pub c: P2,
    pub r: f64,
    pub sg: f64,
  }

  impl CutFunction for Circle {
    fn val(&self, x: P2) -> f64 {
      self.sg * (norm(sub(x, self.c)).powi(2) - self.r * self.r)
    }
    fn grad(&self, x: P2) -> P2 {
      scal(2.0 * self.sg, sub(x, self.c))
    }
    fn hess(&self, _: P2) -> [[f64; 2]; 2] {
      [[2.0 * self.sg, 0.0], [0.0, 2.0 * self.sg]]
    }
  }

  pub(super) fn area(cells: &[P2Cell]) -> f64 {
    let x = [
      [1.0 / 6.0, 1.0 / 6.0],
      [2.0 / 3.0, 1.0 / 6.0],
      [1.0 / 6.0, 2.0 / 3.0],
    ];
    cells
      .iter()
      .map(|c| x.iter().map(|p| c.jac(*p) / 6.0).sum::<f64>().abs())
      .sum()
  }

  /// y - (y0 + a sin(k x)) with its exact derivatives
  struct Wave {
    y0: f64,
    a: f64,
    k: f64,
  }

  impl CutFunction for Wave {
    fn val(&self, x: P2) -> f64 {
      x[1] - self.y0 - self.a * (self.k * x[0]).sin()
    }
    fn grad(&self, x: P2) -> P2 {
      [-self.a * self.k * (self.k * x[0]).cos(), 1.0]
    }
    fn hess(&self, x: P2) -> [[f64; 2]; 2] {
      [
        [self.a * self.k * self.k * (self.k * x[0]).sin(), 0.0],
        [0.0, 0.0],
      ]
    }
  }

  #[test]
  fn edge_crossed_three_times_or_more() {
    // the bottom edge crossed 9 times: unresolved, split (it was cut by
    // its first root only); the area by fine integration over x
    let s3 = 3f64.sqrt();
    let v = [[0.0, 0.0], [1.0, 0.0], [0.5, s3 / 2.0]];
    let w = Wave {
      y0: 0.01,
      a: 0.02,
      k: 30.0,
    };
    assert_eq!(roots(&w, v[0], v[1]).len(), 9);
    let mut out = Cells::default();
    build(&[v], &[&w], &|i: &[bool]| i[0], 0, &mut out);
    assert_eq!(out.split[0], 1);
    let n = 200_000;
    let exact: f64 = (0..n)
      .map(|i| {
        let x = (i as f64 + 0.5) / n as f64;
        let top = (s3 * x).min(s3 * (1.0 - x));
        (w.y0 + w.a * (w.k * x).sin()).clamp(0.0, top) / n as f64
      })
      .sum();
    // before: 80 % lost; now the bumps' quadratic pieces: 2.3 %
    let a = area(&out.cells);
    assert!((a / exact - 1.0).abs() < 3e-2, "area {a} vs {exact}");
  }

  #[test]
  fn single_curve_cases_as_the_prototype() {
    // areas from the Python prototype (scratchpad rules2d, rules_p2d.py)
    let s3 = 3f64.sqrt() / 2.0;
    let v = [[0.0, 0.0], [1.0, 0.0], [0.5, s3]];
    let cases: [(&str, P2, f64, f64, bool, usize, f64); 7] = [
      ("G3", [0.0, 0.0], 0.6, 1.0, true, 1, 0.18803847577293364),
      ("G3s", [0.3, 0.25], 0.55, 1.0, true, 1, 0.4216521947821749),
      (
        "G3s clipped",
        [0.3, 0.25],
        0.55,
        1.0,
        false,
        3,
        0.401837188573283,
      ),
      (
        "G4 away",
        [0.5, s3],
        0.6,
        -1.0,
        true,
        3,
        0.24497422611928565,
      ),
      (
        "G4 towards",
        [0.5, 1.3],
        0.95,
        -1.0,
        true,
        3,
        0.2876408222906713,
      ),
      ("G4s", [0.5, 0.6], 0.65, -1.0, true, 2, 0.04551379174970069),
      (
        "G4s clipped",
        [0.5, 0.6],
        0.65,
        -1.0,
        false,
        2,
        0.026249073857973926,
      ),
    ];
    for (name, c, r, sg, partner, ncells, a) in cases {
      let f = Circle { c, r, sg };
      let (cells, _, k) = cut_cells(&Side { f: &f, s: 1.0 }, v, partner);
      assert_eq!(cells.len(), ncells, "{name}");
      assert!(
        (area(&cells) - a).abs() < 1e-12,
        "{name}: {} vs {a}",
        area(&cells)
      );
      if name == "G4s" {
        let k = k.unwrap();
        assert!(norm(sub(k, [0.5, -0.05])) < 1e-12, "{name}: K {k:?}");
      }
    }
  }

  /// the prototype's lattice of equilateral triangles over [-1, 1]^2
  fn lattice(h: f64) -> Vec<[P2; 3]> {
    let s3 = 3f64.sqrt() / 2.0;
    let (x0, x1, y0, y1) = (-1.0, 1.0, -1.0, 1.0);
    let nj = ((y1 - y0) / (h * s3)).ceil() as i64 + 1;
    let ni = ((x1 - x0) / h).ceil() as i64 + nj + 2;
    let p = |a: i64, b: i64| [x0 + (a as f64 + 0.5 * b as f64) * h, y0 + b as f64 * h * s3];
    let mut out = vec![];
    for j in 0..nj {
      for i in -nj - 1..ni {
        let up = [p(i, j), p(i + 1, j), p(i, j + 1)];
        let dn = [p(i + 1, j), p(i + 1, j + 1), p(i, j + 1)];
        for t in [up, dn] {
          let cx = (t[0][0] + t[1][0] + t[2][0]) / 3.0;
          if x0 - h <= cx && cx <= x1 + h {
            out.push(t);
          }
        }
      }
    }
    out
  }

  /// n.x - c
  struct Line {
    n: P2,
    c: f64,
  }

  impl CutFunction for Line {
    fn val(&self, x: P2) -> f64 {
      dot(self.n, x) - self.c
    }
    fn grad(&self, _: P2) -> P2 {
      self.n
    }
    fn hess(&self, _: P2) -> [[f64; 2]; 2] {
      [[0.0; 2]; 2]
    }
  }

  #[test]
  fn lattice_cases_as_the_prototype() {
    // cells, areas, splits per level, depth-limit triangles and folding
    // fans from the Python prototype (rules_p2h.py, L = 3)
    let two = [([-0.15, 0.02], 0.45), ([0.25, 0.05], 0.4)];
    let tang = [([0.0, 0.0], 0.4), ([0.0, 0.3], 0.1), ([0.0, 0.48], 0.08)];
    let lines = [
      ([1.0, 0.3], 0.07),
      ([-0.4, 1.0], 0.005),
      ([0.7, -1.0], 0.01),
    ];
    let circles = |c: &[(P2, f64)], sg: f64| -> Vec<Box<dyn CutFunction>> {
      c.iter()
        .map(|&(c, r)| Box::new(Circle { c, r, sg }) as Box<dyn CutFunction>)
        .collect()
    };
    let line_fs = || -> Vec<Box<dyn CutFunction>> {
      lines
        .iter()
        .map(|&(n, c)| Box::new(Line { n, c }) as Box<dyn CutFunction>)
        .collect()
    };
    let island = [([0.021, 0.16770054], 0.08)];
    type Case = (
      &'static str,
      f64,
      Vec<Box<dyn CutFunction>>,
      &'static str,
      usize,
      f64,
      [usize; 3],
      usize,
      usize,
    );
    let cases: Vec<Case> = vec![
      (
        "island disc",
        0.5,
        circles(&island, 1.0),
        "a",
        6,
        0.019965139003941236,
        [1, 2, 0],
        0,
        0,
      ),
      (
        "island hole",
        0.5,
        circles(&island, -1.0),
        "a",
        98,
        8.423782547894339,
        [1, 2, 0],
        0,
        0,
      ),
      (
        "disc on an edge",
        0.5,
        circles(&[([0.13, 0.09], 0.14)], 1.0),
        "a",
        6,
        0.06142319064452617,
        [2, 0, 0],
        0,
        0,
      ),
      (
        "two circles a*b",
        0.25,
        circles(&two, 1.0),
        "a*b",
        82,
        0.23860943606693472,
        [7, 6, 6],
        6,
        0,
      ),
      (
        "two circles a-b",
        0.25,
        circles(&two, 1.0),
        "a-b",
        81,
        0.3975386297793996,
        [7, 6, 6],
        6,
        0,
      ),
      (
        "large circle",
        0.25,
        circles(&[([0.03, 0.02], 0.6)], 1.0),
        "a",
        93,
        1.1309571500267548,
        [0, 0, 0],
        0,
        0,
      ),
      (
        "tangent circles a-b",
        0.25,
        circles(&tang, 1.0),
        "a-b",
        107,
        0.47123597134337253,
        [3, 3, 5],
        9,
        1,
      ),
      (
        "tangent circles b-c",
        0.25,
        circles(&tang, 1.0),
        "b-c",
        28,
        0.03135298608992858,
        [1, 1, 3],
        7,
        0,
      ),
      (
        "three lines a*b*c",
        0.25,
        line_fs(),
        "a*b*c",
        117,
        0.27905969704639955,
        [5, 7, 7],
        6,
        0,
      ),
      (
        "three lines a+b-c",
        0.25,
        line_fs(),
        "a+b-c",
        186,
        2.357899584351925,
        [7, 5, 5],
        7,
        0,
      ),
    ];
    for (name, h, fs, e, ncells, a, split, limit, fold) in cases {
      let fr: Vec<&dyn CutFunction> = fs.iter().map(|f| f.as_ref()).collect();
      let sel = |i: &[bool]| match e {
        "a*b" => i[0] && i[1],
        "a-b" => i[0] && !i[1],
        "b-c" => i[1] && !i[2],
        "a*b*c" => i[0] && i[1] && i[2],
        "a+b-c" => (i[0] || i[1]) && !i[2],
        _ => i[0],
      };
      let mut out = Cells::default();
      build(&lattice(h), &fr, &sel, 0, &mut out);
      assert_eq!(out.cells.len(), ncells, "{name}: cells");
      assert_eq!(out.split[..3], split, "{name}: splits");
      assert_eq!((out.limit, out.fold), (limit, fold), "{name}: limit, fold");
      assert!(
        (area(&out.cells) - a).abs() < 1e-11,
        "{name}: {} vs {a}",
        area(&out.cells)
      );
    }
  }
}
