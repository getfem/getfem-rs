#!/usr/bin/env python3
"""An inclined penny-shaped crack under remote tension, with GetFEM-rs,
as the Rust example inclined_penny_crack: a crack of radius a = 1 through
the centre of a block, its normal tilted by GAMMA from the z axis in the
x-z plane, a uniaxial stress s along z (tractions on the faces z = +-L),
linearized elasticity (E = 1, nu = 0.3). Mixed mode, all three stress
intensity factors along the front against the infinite body's (Kassir
and Sih):
  K_I = 2 s_n sqrt(a/pi),
  K_II = 4 t sqrt(a/pi) cos(w)/(2 - nu),
  K_III = -4 (1 - nu) t sqrt(a/pi) sin(w)/(2 - nu),
with s_n = s cos^2(GAMMA), t = s sin(GAMMA) cos(GAMMA), w the angle on
the front from the shear direction and the sign of K_III in the crack
frame (normal e2, e1 ahead, e3 = e1 x e2). Half of the block, y >= 0
(symmetry on y = 0), Q1 hexahedra of size h up to 1.5 a, growing by 1.3
beyond; the mesh does not follow the crack. The displacement is enriched
by the Heaviside function of the crack and by the crack tip functions on
the nodes within 2 h of the front. At the two front ends on the symmetry
plane K_III is not meaningful: the domain integral sees half of its
domain and the mode III field is not symmetric.

Prints the factors along the front against the exact ones and writes
them at the front's points to inclined_penny_crack_sifs.vtu. Argument: h
(default 0.125)."""
import sys

import numpy as np

import getfem_rs as gf

A = 1.               # crack radius
L = 6.               # half size of the block
E, NU = 1., 0.3
S = 1.               # remote stress along z
GAMMA = np.pi/6      # tilt of the crack
PENALTY = 1e6
SUPPORT = 1e-8       # relative support below which DOFs are dropped
SYM_Y, ENDS = 101, [102, 103]
# the crack normal and the shear direction on the crack (the in-plane
# part of the z axis)
N = np.array([np.sin(GAMMA), 0., np.cos(GAMMA)])
T = np.array([-np.cos(GAMMA), 0., np.sin(GAMMA)])


def nodes(start, h):
  """the node coordinates from start: steps h up to 1.5 a, then growing
  by 1.3, up to L (the last step stretched or shrunk to end at L)"""
  x, step = [start], h
  while x[-1] < L - 1e-9:
    if x[-1] >= 1.5*A:
      step *= 1.3
    x.append(L if x[-1] + 1.5*step > L else x[-1] + step)
  return np.array(x)


def mirrored(p):
  return np.concatenate([-p[p > 0][::-1], p])


def exact(x):
  """the angle on the front from the shear direction and the exact
  factors"""
  w = np.arctan2(x[1], T @ x)
  sn, tau = S*N[2]*N[2], S*N[2]*T[2]
  r = np.sqrt(A/np.pi)
  return w, [2*sn*r, 4*tau*r*np.cos(w)/(2 - NU), -4*(1 - NU)*tau*r*np.sin(w)/(2 - NU)]


def simulate(h):
  # y from 0, x and z symmetric, z without a node at 0
  m = gf.Mesh("cartesian Q1", mirrored(nodes(0., h)), nodes(0., h), mirrored(nodes(0.5*h, h)))
  for rg, n in [(SYM_Y, [0., -1., 0.]), (ENDS[0], [0., 0., 1.]), (ENDS[1], [0., 0., -1.])]:
    m.set_region(rg, m.outer_faces_with_direction(n, 0.5))

  # the crack: primary the distance to its plane, secondary the distance
  # to the front circle in the plane
  # (the normal (sin GAMMA, 0, cos GAMMA))
  d = "(x*sin(pi/6)+z*cos(pi/6))"
  ls = gf.LevelSet(m, 2, d, "sqrt(x*x+y*y+z*z-sqr%s)-%r" % (d, A))
  mls = gf.MeshLevelSet(m)
  mls.add(ls)
  mls.adapt()

  # Heaviside enrichment plus the crack tip functions near the front
  pu_mf = gf.MeshFem(m)
  pu_mf.set_classical_finite_element(1)
  global_mf = gf.MeshFem("global function", m, ls, [gf.GlobalFunction("crack", i) for i in range(4)])
  tip = gf.MeshFem("product", pu_mf, global_mf, ls.select_mesh_fem_basic_dofs_near_front(pu_mf, 2*h))
  base = gf.MeshFem("sum", gf.MeshFem("levelset", mls, pu_mf), tip)
  mim = gf.MeshIm("levelset", mls, "all", gf.Integ("IM_TETRAHEDRON(5)"),
                  gf.Integ("IM_PRODUCT(IM_TRIANGLE(3),IM_GAUSS1D(3))"),
                  gf.Integ("IM_GAUSS_PARALLELEPIPED(3,4)"))
  # the DOFs with support (enriched DOFs without are singular)
  kept = base.dofs_with_support(mim, SUPPORT)
  base.set_qdim(3)
  mf_u = gf.MeshFem("partial", base, [3*i + c for i in kept for c in range(3)])

  md = gf.Model()
  md.add_fem_variable("u", mf_u)
  md.add_initialized_data("lambda", E*NU/((1 + NU)*(1 - 2*NU)))
  md.add_initialized_data("mu", E/(2*(1 + NU)))
  md.add_initialized_data("s", S)
  md.add_initialized_data("pen", PENALTY)
  md.add_linear_term(mim, "lambda*Div_u*Div_Test_u + mu*(Grad_u + Grad_u'):Grad_Test_u")
  # the uniaxial stress on the ends, symmetry on y = 0; a weak spring
  # takes the remaining rigid motions (the loads are balanced)
  for rg in ENDS:
    md.add_linear_term(mim, "-s*(Normal.[0,0,1])*([0,0,1].Test_u)", rg)
  md.add_linear_term(mim, "pen*(u.Normal)*(Test_u.Normal)", SYM_Y)
  md.add_linear_term(mim, "1e-6*u.Test_u")
  nit, conv, info = md.solve()
  assert conv, info["message"]

  # the stress intensity factors along the front (radius 0.3, points
  # 0.25 apart), the lips free
  K, front = ls.crack_sifs(mim, mf_u, md.variable("u"), E, NU, 0.3, 0.25)
  return dict(dofs=mf_u.nb_dof(), K=K, front=front)


if __name__ == "__main__":
  h = float(sys.argv[1]) if len(sys.argv) > 1 else 0.125
  s = simulate(h)
  print("h %g, %i DOFs" % (h, s["dofs"]))
  print("  angle       K_I    exact      K_II    exact     K_III    exact")
  # the front's points once each (a point on several convexes is in the
  # slice once per convex), by angle
  X = s["front"].pts()
  rows = {}
  for x, K in zip(X.T, s["K"].T):
    w, k = exact(x)
    rows[np.degrees(w).round(6)] = (K, k)
  for angle in sorted(rows):
    K, k = rows[angle]
    print("%7.2f %9.5f %8.5f %9.5f %8.5f %9.5f %8.5f" % (angle, K[0], k[0], K[1], k[1], K[2], k[2]))
  s["front"].export_to_vtu("inclined_penny_crack_sifs.vtu", s["K"], "K")
  print("wrote inclined_penny_crack_sifs.vtu")
