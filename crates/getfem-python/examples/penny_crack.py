#!/usr/bin/env python3
"""A penny-shaped crack under pressure (Sneddon), with GetFEM-rs, as the
Rust example penny_crack (its enrichments "zone" and "front" only): a
crack of radius a = 1 in the plane z = 0 of a block, a unit pressure on
its lips, linearized elasticity (E = 1, nu = 0.3). A quarter of the
block, x, y in [0, 6], z in [-6, 6]: symmetry on x = 0 and y = 0, the far
faces clamped. Q1 hexahedra of size h up to 1.5 a, growing by 1.3 beyond;
no node on the crack plane. The displacement is enriched by the
Heaviside function of the crack and, near the front, by the crack tip
functions:
- "none": no front enrichment;
- "zone:rho": the nodes within rho of the front (fixed zone);
- "front:r0:r1:s": the tip functions times a cutoff (1 below r0, 0
  beyond r1) and linear hats on front points s apart ("cfront": cubic;
  ":2" at the end: with the second order tip functions, GetFEM's 4 to 7).
Several, summed: joined by '+' (e.g. "front:0.3:0.8:0.2:2+zone:0.125").

Prints the opening along the ray at 45 degrees against Sneddon's
8 (1 - nu^2) p sqrt(a^2 - r^2)/(pi E), the relative RMS error over
r < 0.95 a, and the stress intensity factors along the front against
K_I = 2 p sqrt(a/pi). Arguments: h (default 0.125), the enrichment
(default "zone:0.25")."""
import sys

import numpy as np

import getfem_rs as gf

A = 1.               # crack radius
L = 6.               # half size of the block
E, NU = 1., 0.3
P = 1.               # pressure on the lips
PENALTY = 1e6
SUPPORT = 1e-8       # relative support below which DOFs are dropped
SYM_X, SYM_Y, FAR = 101, 102, [103, 104, 105, 106]


def nodes(start, h):
  """the node coordinates from start: steps h up to 1.5 a, then growing
  by 1.3, up to L (the last step stretched or shrunk to end at L)"""
  x, step = [start], h
  while x[-1] < L - 1e-9:
    if x[-1] >= 1.5*A:
      step *= 1.3
    x.append(L if x[-1] + 1.5*step > L else x[-1] + step)
  return np.array(x)


def space(enrichment, m, ls):
  """the front enrichment "none", "zone:rho" or "front:r0:r1:s[:2]"
  ("cfront": cubic hats), None for "none" """
  v = enrichment.split(":")
  if v[0] == "none":
    return None
  pu_mf = gf.MeshFem(m)
  pu_mf.set_classical_finite_element(1)
  if v[0] == "zone":
    global_mf = gf.MeshFem("global function", m, ls, [gf.GlobalFunction("crack", i) for i in range(4)])
    return gf.MeshFem("product", pu_mf, global_mf, ls.select_mesh_fem_basic_dofs_near_front(pu_mf, float(v[1])))
  assert v[0] in ["front", "cfront"], "enrichment: none, zone:rho, front:r0:r1:s[:2] or cfront"
  r0, r1, spacing = map(float, v[1:4])
  cut = gf.GlobalFunction("cutoff", "polynomial", r0, r1)
  # the tip functions times the cutoff, on the crack
  tips = [gf.GlobalFunction("levelset", ls, gf.GlobalFunction("crack", i)*cut)
          for i in range(8 if v[4:] == ["2"] else 4)]
  funcs = []
  for Pf, Tf, closed in ls.crack_fronts(spacing):
    for k in range(Pf.shape[1]):
      hat = gf.GlobalFunction("front hat", Pf, Tf, closed, k, r1,
                              "linear" if v[0] == "front" else "cubic")
      funcs += [hat*t for t in tips]
  return gf.MeshFem("global function", m, funcs)


def simulate(h, enrichments):
  # x, y from 0, z symmetric without a node at 0
  xs, zp = nodes(0., h), nodes(0.5*h, h)
  m = gf.Mesh("cartesian Q1", xs, xs, np.concatenate([-zp[::-1], zp]))
  for rg, nn in [(SYM_X, [-1., 0., 0.]), (SYM_Y, [0., -1., 0.]), (FAR[0], [1., 0., 0.]),
                 (FAR[1], [0., 1., 0.]), (FAR[2], [0., 0., 1.]), (FAR[3], [0., 0., -1.])]:
    m.set_region(rg, m.outer_faces_with_direction(nn, 0.5))

  # the crack: primary z, secondary the distance to the front circle
  ls = gf.LevelSet(m, 2, "z", "sqrt(x*x+y*y)-%r" % A)
  mls = gf.MeshLevelSet(m)
  mls.add(ls)
  mls.adapt()

  mf_pre = gf.MeshFem(m)
  mf_pre.set_classical_finite_element(1)
  parts = [s for s in (space(e, m, ls) for e in enrichments) if s is not None]
  base = gf.MeshFem("sum", *parts, gf.MeshFem("levelset", mls, mf_pre))
  mim = gf.MeshIm("levelset", mls, "all", gf.Integ("IM_TETRAHEDRON(5)"),
                  gf.Integ("IM_PRODUCT(IM_TRIANGLE(3),IM_GAUSS1D(3))"),
                  gf.Integ("IM_GAUSS_PARALLELEPIPED(3,4)"))
  # the DOFs with support (functions without support in the domain add
  # singular DOFs)
  kept = base.dofs_with_support(mim, SUPPORT)
  dropped = base.nb_dof() - len(kept)
  base.set_qdim(3)
  mf_u = gf.MeshFem("partial", base, [3*i + c for i in kept for c in range(3)])
  # the lips graded toward the front (r^(-1/2) of the SIFs' lip term)
  mim_lips = gf.MeshIm("levelset", mls, "boundary", gf.Integ("IM_TETRAHEDRON(3)"),
                       gf.Integ("IM_PRODUCT(IM_TRIANGLE(3),IM_GAUSS1D(3))"))

  md = gf.Model()
  md.add_fem_variable("u", mf_u)
  md.add_initialized_data("lambda", E*NU/((1 + NU)*(1 - 2*NU)))
  md.add_initialized_data("mu", E/(2*(1 + NU)))
  md.add_initialized_data("p", P)
  md.add_initialized_data("pen", PENALTY)
  md.add_linear_term(mim, "lambda*Div_u*Div_Test_u + mu*(Grad_u + Grad_u'):Grad_Test_u")
  md.add_linear_term(mim_lips, "p*Normal.Xfem_minus(Test_u) - p*Normal.Xfem_plus(Test_u)")
  # symmetry: the normal displacement; far faces: clamped
  for rg in [SYM_X, SYM_Y]:
    md.add_linear_term(mim, "pen*(u.Normal)*(Test_u.Normal)", rg)
  for rg in FAR:
    md.add_linear_term(mim, "pen*u.Test_u", rg)
  nit, conv, info = md.solve()
  assert conv, info["message"]
  u = md.variable("u")
  # the stress intensity factors along the front (radius 0.3, points 0.25
  # apart), the pressure on the lips as in the model's term
  K, front = ls.crack_sifs(mim, mf_u, u, E, NU, 0.3, 0.25,
                           lips=(mim_lips, "%r*Normal" % -P, "%r*Normal" % P))
  # the opening along the ray at 45 degrees, just above and below the
  # crack
  r = 0.05*np.arange(20)
  above = [[ri/np.sqrt(2), ri/np.sqrt(2), 1e-7] for ri in r]
  below = [[ri/np.sqrt(2), ri/np.sqrt(2), -1e-7] for ri in r]
  delta = (md.interpolation("u(3)", np.transpose(above), m)
           - md.interpolation("u(3)", np.transpose(below), m))
  exact = 8*(1 - NU*NU)*P*np.sqrt(np.maximum(A*A - r*r, 0))/(np.pi*E)
  return dict(r=r, delta=delta, exact=exact, dofs=mf_u.nb_dof(), dropped=dropped,
              max_coef=np.abs(u).max(), K=K, front=front)


if __name__ == "__main__":
  h = float(sys.argv[1]) if len(sys.argv) > 1 else 0.125
  e = sys.argv[2] if len(sys.argv) > 2 else "zone:0.25"
  o = simulate(h, e.split("+"))
  print("h %g, %s, %i DOFs (%i scalar dropped), largest coefficient %.3e"
        % (h, e, o["dofs"], o["dropped"], o["max_coef"]))
  print("     r    opening     Sneddon")
  for r, d, x in zip(o["r"], o["delta"], o["exact"]):
    print("%6.2f %10.6f %10.6f" % (r, d, x))
  inner = o["r"] < 0.95*A
  rms = np.sqrt(((o["delta"] - o["exact"])[inner]**2).sum()/(o["exact"][inner]**2).sum())
  print("relative RMS error %.4e, at the centre %.4e" % (rms, o["delta"][0]/o["exact"][0] - 1))
  k = 2*P*np.sqrt(A/np.pi)
  print("  angle  K_I/exact      K_II     K_III (exact K_I %.5f)" % k)
  # the front's points once each (a point on several convexes is in the
  # slice once per convex), by angle
  X = o["front"].pts()
  rows = {np.degrees(np.arctan2(x[1], x[0])).round(6): K for x, K in zip(X.T, o["K"].T)}
  for angle in sorted(rows):
    print("%7.2f %10.5f %9.5f %9.5f" % (angle, rows[angle][0]/k, rows[angle][1], rows[angle][2]))
