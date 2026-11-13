//! Stress intensity factors of a crack in linear elasticity, by the
//! interaction integral in domain form (Moës, Dolbow, Belytschko 1999):
//! ```text
//! I = int su:(Grad(v) e1 @ Grad(q)) + sv:(Grad(u) e1 @ Grad(q))
//!       - (su:Grad(v)) Grad(q).e1 + q Div(sv).(Grad(u) e1)
//!     + int_lips q t.(Grad(v) e1),
//! ```
//! v the auxiliary field of one mode (the crack tip asymptotics in the
//! crack's frame, of the level sets' r and theta: on a curved front not in
//! equilibrium, hence the Div(sv) term), t the tractions on the lips, q 1
//! at the tip and 0 farther than a radius (Lagrange P1/Q1, the elements
//! crossing the radius carry its gradient). In 3D q is the hat of a front
//! point times that cutoff, the frame frozen at the point; I divided by
//! the integral of q along the front. K_I = E* I_I / 2, K_II = E* I_II /
//! 2, K_III = mu I_III, E* = 4 mu (lambda + mu) / (lambda + 2 mu), lambda
//! and mu from E and nu (plane strain; plane stress in 2D by E', nu', see
//! CrackedBody). The frame: e1 along the
//! crack beyond the front, e2 = grad phi, e3 = e1 x e2; K_II > 0 when the
//! lip phi > 0 slides along e1 relative to the other.

use getfem_fem::crack_front::{CrackFront, HatDegree, crack_frames, front_hats};
use getfem_fem::crack_tip_function::{CrackTipFunction, Singular};
use getfem_fem::global_function::GlobalFunction;
use getfem_fem::interpolation::interpolation_at_points;
use getfem_fem::level_set::LevelSet;
use getfem_fem::mesh_fem::MeshFem;
use getfem_fem::mesh_im::MeshIm;
use getfem_geom::geotrans::geotrans_descriptor;
use getfem_geom::mesh::Mesh;
use getfem_geom::region::{Region, RegionSet};

use crate::workspace::Workspace;
use crate::{Error, Result};
use getfem_linalg::vect_dist2;

/// Tractions on the lips of the crack: GWFL vector expressions (of X,
/// Normal and numbers) of the force per area on the lip on the side
/// phi > 0 (plus) and phi < 0 (minus), as in the model's term
/// plus.Xfem_plus(Test_u) + minus.Xfem_minus(Test_u); mim on the lips
/// (IntegrateWhere::Boundary, with a sing rule: graded at the front). A
/// pressure p: plus "-p*Normal", minus "p*Normal".
#[derive(Clone, Copy)]
pub struct LipTractions<'a> {
  pub mim: &'a MeshIm,
  pub plus: &'a str,
  pub minus: &'a str,
}

/// The cracked body: the displacement u on mf_u, integrated with mim, the
/// crack ls (primary phi, secondary psi), Young's modulus and Poisson's
/// ratio (plane strain in 2D; plane stress: E' = E (1 + 2 nu)/(1 + nu)^2,
/// nu' = nu/(1 + nu), the 2D Lamé coefficients of plane stress), the
/// tractions on the lips if any.
pub struct CrackedBody<'a> {
  pub mim: &'a MeshIm,
  pub mf_u: &'a MeshFem,
  pub u: &'a [f64],
  pub ls: &'a LevelSet,
  pub young: f64,
  pub poisson: f64,
  pub lips: Option<LipTractions<'a>>,
}

/// The stress intensity factors K_I, K_II, K_III at a crack tip or a
/// point of a crack front (K_III 0 in 2D).
#[derive(Clone, Debug)]
pub struct Sif {
  pub point: Vec<f64>,
  pub k: [f64; 3],
}

/// The stress intensity factors at the points of a front of the crack
/// (see crack_fronts), radius the cutoff's. In 2D, at the tip (a front of
/// one point): q 1 within radius, a few elements (the elements at the tip
/// carry no gradient of q), short of other tips, kinks of the secondary
/// and the outer boundary. In 3D: q the front's linear hats within radius
/// of it (the front may end on the outer boundary), the spacing of the
/// points a few elements.
pub fn crack_sifs(body: &CrackedBody, radius: f64, front: &CrackFront) -> Result<Vec<Sif>> {
  let mf_q = lagrange(body.mf_u)?;
  let nodes = nodes(&mf_q)?;
  if front.tangents.is_empty() {
    let weights: Vec<(Vec<f64>, f64)> = front
      .points
      .iter()
      .map(|t| {
        (
          nodes
            .iter()
            .map(|x| if vect_dist2(x, t) < radius { 1.0 } else { 0.0 })
            .collect(),
          1.0,
        )
      })
      .collect();
    return integrals(body, &mf_q, &front.points, &weights);
  }
  let n = front.points.len();
  let segments: Vec<(usize, usize)> = (0..if front.closed { n } else { n - 1 })
    .map(|k| (k, (k + 1) % n))
    .collect();
  let near: Vec<bool> = nodes
    .iter()
    .map(|x| {
      segments
        .iter()
        .any(|&(a, b)| segment_distance(&front.points[a], &front.points[b], x) < radius)
    })
    .collect();
  let hats = front_hats(front, HatDegree::Linear, radius)?;
  // samples along the front: Gauss points of its segments
  let gauss = [0.5 - 0.5 / 3f64.sqrt(), 0.5 + 0.5 / 3f64.sqrt()];
  let mut samples = vec![];
  let mut lengths = vec![];
  for &(a, b) in &segments {
    let (p, q) = (&front.points[a], &front.points[b]);
    for t in gauss {
      samples.push(
        p.iter()
          .zip(q)
          .map(|(x, y)| x + t * (y - x))
          .collect::<Vec<f64>>(),
      );
      lengths.push(0.5 * vect_dist2(p, q));
    }
  }
  let mut weights = vec![];
  for hat in &hats {
    let q: Vec<f64> = nodes
      .iter()
      .zip(&near)
      .map(|(x, &on)| if on { hat.val(x) } else { 0.0 })
      .collect();
    let along = interpolation_at_points(&mf_q, &samples, &q)?;
    let len = along.iter().zip(&lengths).map(|(v, l)| v * l).sum();
    weights.push((q, len));
  }
  integrals(body, &mf_q, &front.points, &weights)
}

/// P1/Q1 on the mesh of mf_u
fn lagrange(mf_u: &MeshFem) -> Result<MeshFem> {
  Ok(MeshFem::new(&mf_u.linked_mesh(), 1).with_classical_finite_element(1)?)
}

fn nodes(mf: &MeshFem) -> Result<Vec<Vec<f64>>> {
  (0..mf.nb_dof()?)
    .map(|i| Ok(mf.point_of_basic_dof(i)?))
    .collect()
}

fn segment_distance(p: &[f64], q: &[f64], x: &[f64]) -> f64 {
  let e: Vec<f64> = p.iter().zip(q).map(|(a, b)| b - a).collect();
  let w: Vec<f64> = p.iter().zip(x).map(|(a, b)| b - a).collect();
  let l2: f64 = e.iter().map(|v| v * v).sum();
  let t = if l2 > 0.0 {
    (w.iter().zip(&e).map(|(a, b)| a * b).sum::<f64>() / l2).clamp(0.0, 1.0)
  } else {
    0.0
  };
  w.iter()
    .zip(&e)
    .map(|(a, b)| (a - t * b).powi(2))
    .sum::<f64>()
    .sqrt()
}

/// The interaction integrals at the points, with their weights q (on
/// mf_q) and the integrals of q along the front.
fn integrals(
  body: &CrackedBody,
  mf_q: &MeshFem,
  points: &[Vec<f64>],
  weights: &[(Vec<f64>, f64)],
) -> Result<Vec<Sif>> {
  let CrackedBody {
    mim,
    mf_u,
    u,
    ls,
    young,
    poisson: nu,
    ref lips,
  } = *body;
  let lambda = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
  let mu = young / (2.0 * (1.0 + nu));
  let mesh = mf_u.linked_mesh();
  let dim = mesh.dim();
  let funcs: Vec<GlobalFunction> = Singular::LEFM
    .iter()
    .map(|&s| GlobalFunction::on_level_set(ls, &CrackTipFunction::crack(s)))
    .collect::<getfem_fem::Result<_>>()?;
  let mf_v = MeshFem::from_global_functions(&mesh, &funcs, None)?;
  mf_v.set_qdim(dim);
  let kappa = (lambda + 3.0 * mu) / (lambda + mu);
  let estar = 4.0 * mu * (lambda + mu) / (lambda + 2.0 * mu);
  let c = 1.0 / (2.0 * mu * (2.0 * std::f64::consts::PI).sqrt());
  let frames = crack_frames(ls, points)?;
  let md = mesh.snapshot();
  let mut sifs = vec![];
  for ((x, frame), (q, len)) in points.iter().zip(frames).zip(weights) {
    let Some(e) = frame else {
      return Err(Error::InvalidArgument(format!(
        "crack sifs: point {x:?} outside the mesh"
      )));
    };
    // the elements where q is not zero
    let mut set = RegionSet::new();
    for cv in md.convex_index() {
      if mf_q
        .ind_basic_dof_of_element(cv)?
        .iter()
        .any(|&i| q[i] != 0.0)
      {
        set.add(cv);
      }
    }
    let region = Region::temporary(&mesh, set)?;
    // the modes on the 4 crack tip functions F_l, c_l in R^dim: I: u1 =
    // c ((kappa-1) F1 + F2), u2 = c ((kappa+1) F0 - F3); II: u1 = c
    // ((kappa+1) F0 + F3), u2 = c (F2 - (kappa-1) F1); III: u3 = 4 c F0
    let comb = |a: &[(f64, usize)]| -> Vec<f64> {
      (0..dim)
        .map(|i| a.iter().map(|&(s, k)| s * e[k][i]).sum::<f64>() * c)
        .collect()
    };
    let mut modes = vec![
      [
        comb(&[(kappa + 1.0, 1)]),
        comb(&[(kappa - 1.0, 0)]),
        comb(&[(1.0, 0)]),
        comb(&[(-1.0, 1)]),
      ],
      [
        comb(&[(kappa + 1.0, 0)]),
        comb(&[(1.0 - kappa, 1)]),
        comb(&[(1.0, 1)]),
        comb(&[(1.0, 0)]),
      ],
    ];
    if dim == 3 {
      let z = vec![0.0; 3];
      modes.push([comb(&[(4.0, 2)]), z.clone(), z.clone(), z]);
    }
    let mut k = [0.0; 3];
    for (m, mode) in modes.iter().enumerate() {
      let v: Vec<f64> = mode.iter().flatten().copied().collect();
      let mut ws = Workspace::new();
      ws.add_fem_data("u", mf_u, u)?;
      ws.add_fem_data("v", &mf_v, &v)?;
      ws.add_fem_data("q", mf_q, q)?;
      ws.add_fixed_size_data("e1", &[dim], &e[0])?;
      ws.add_scalar_data("lambda", lambda)?;
      ws.add_scalar_data("mu", mu)?;
      ws.add_macro(
        "su",
        &format!("lambda*Div(u)*Id({dim}) + mu*(Grad(u)+Grad(u)')"),
      )?;
      ws.add_macro(
        "sv",
        &format!("lambda*Div(v)*Id({dim}) + mu*(Grad(v)+Grad(v)')"),
      )?;
      ws.add_expression(
        "su:((Grad(v)*e1)@Grad(q)) + sv:((Grad(u)*e1)@Grad(q)) - (su:Sym(Grad(v)))*(Grad(q).e1) \
         + q*((lambda+mu)*Contract(Hess(v),1,2) + mu*Contract(Hess(v),2,3)).(Grad(u)*e1)",
        mim,
        &region,
      )?;
      if let Some(l) = lips {
        ws.add_expression(
          &format!(
            "q*(({}).(Xfem_plus(Grad_v)*e1) + ({}).(Xfem_minus(Grad_v)*e1))",
            l.plus, l.minus
          ),
          l.mim,
          &region,
        )?;
      }
      ws.assembly(0, false)?;
      let i = ws.assembled_potential();
      let i = i / len;
      k[m] = if m < 2 { 0.5 * estar * i } else { mu * i };
    }
    sifs.push(Sif {
      point: x.clone(),
      k,
    });
  }
  Ok(sifs)
}

/// The points of the factors along a front as a 1D mesh (segments in
/// order; closed: the last to the first too), P1 on it and the factors
/// as a field (qdim 3: K_I, K_II, K_III), e.g. for VtuExport.
pub fn sifs_on_front(sifs: &[Sif], closed: bool) -> Result<(MeshFem, Vec<f64>)> {
  let n = sifs.len();
  if n < 2 {
    return Err(Error::InvalidArgument(format!("sifs_on_front: {n} points")));
  }
  let dim = sifs[0].point.len();
  let mesh = Mesh::new(dim);
  let gt = geotrans_descriptor("GT_PK(1,1)")?;
  for k in 0..if closed { n } else { n - 1 } {
    let (a, b) = (&sifs[k].point, &sifs[(k + 1) % n].point);
    mesh.add_convex_by_points(&gt, &[a, b])?;
  }
  let mf = MeshFem::new(&mesh, 3).with_classical_finite_element(1)?;
  let mut k = vec![0.0; mf.nb_dof()?];
  for (i, v) in k.iter_mut().enumerate() {
    let x = mf.point_of_basic_dof(i)?;
    let s = sifs
      .iter()
      .min_by(|a, b| vect_dist2(&a.point, &x).total_cmp(&vect_dist2(&b.point, &x)))
      .unwrap();
    *v = s.k[i % 3];
  }
  Ok((mf, k))
}
