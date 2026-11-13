#!/usr/bin/env python3
"""Surface cracks, with GetFEM-rs, as the Rust example surface_crack: a
block [0, W] x [-H, H] (x [0, T] in 3D), the crack in the plane y = 0 from
the free face x = 0, a unit traction pulling the faces y = +-H apart, the
face x = W clamped. Q1, linearized elasticity; the displacement enriched
by the Heaviside function of the crack and, on the elements the front
passes through, by the crack tip functions; the DOFs without support
dropped. Cases:
- "edge": an edge crack of depth A in 2D (plane strain);
- "through": the same through the thickness T in 3D, plane strain
  (u_z = 0 on z = 0 and z = T): its front ends on both faces;
- "semi": a semicircular surface crack of radius A centred at x = 0,
  z = T/2: its front a half circle ending on the face x = 0.

Prints the opening along x in the crack plane (3D: at z = T/2) and the
stress intensity factors (interaction integral, radius 0.3; 3D: along the
front, points 0.25 apart, also written to surface_crack_<case>_sifs.vtu).
Arguments: the case (default "semi"), h (default 0.125)."""
import sys

import numpy as np

import getfem_rs as gf

W, H, T, A = 2., 2., 2., 0.5
E, NU = 1., 0.3


def simulate(case, h):
  assert case in ["edge", "through", "semi"], "case: edge, through or semi"
  dim = 2 if case == "edge" else 3
  x = np.linspace(0, W, round(W/h) + 1)
  # y nodes offset by h/2: no node on the crack plane
  y = np.linspace(-H - 0.5*h, H + 0.5*h, round(2*H/h) + 2)
  z = np.linspace(0, T, round(T/h) + 1)
  m = gf.Mesh("cartesian Q1", x, y) if dim == 2 else gf.Mesh("cartesian Q1", x, y, z)
  for rg, nn in [(1, [0, 1, 0]),
                 (2, [0,-1, 0]),
                 (3, [1, 0, 0]),
                 (4, [0, 0, 1]),
                 (5, [0, 0,-1])][:2*dim - 1]:
    m.set_region(rg, m.outer_faces_with_direction(nn[:dim], 0.5))

  # the crack: primary y, secondary the distance to the front
  secondary = "sqrt(x*x+sqr(z-%r))-%r" % (0.5*T, A) if case == "semi" else "x-%r" % A
  ls = gf.LevelSet(m, 2, "y", secondary)
  mls = gf.MeshLevelSet(m)
  mls.add(ls)
  mls.adapt()

  pu_mf = gf.MeshFem(m)
  pu_mf.set_classical_finite_element(1)
  global_mf = gf.MeshFem("global function", m, ls, [gf.GlobalFunction("crack", i) for i in range(4)])
  # the front elements: the elements the front passes through
  sing = gf.MeshFem("product", pu_mf, global_mf, pu_mf.basic_dof_on_region(mls.crack_tip_convexes(ls)))
  base = gf.MeshFem("sum", sing, gf.MeshFem("levelset", mls, pu_mf))
  if dim == 2:
    mim = gf.MeshIm("levelset", mls, "all", gf.Integ("IM_TRIANGLE(6)"),
                    gf.Integ("IM_STRUCTURED_COMPOSITE(IM_GAUSS_PARALLELEPIPED(2,6),9)"),
                    gf.Integ("IM_GAUSS_PARALLELEPIPED(2,4)"))
  else:
    mim = gf.MeshIm("levelset", mls, "all", gf.Integ("IM_TETRAHEDRON(5)"),
                    gf.Integ("IM_PRODUCT(IM_TRIANGLE(3),IM_GAUSS1D(3))"),
                    gf.Integ("IM_GAUSS_PARALLELEPIPED(3,4)"))
  # the DOFs with support (enriched DOFs without are singular)
  kept = base.dofs_with_support(mim, 1e-8)
  base.set_qdim(dim)
  mf_u = gf.MeshFem("partial", base, [dim*i + c for i in kept for c in range(dim)])

  md = gf.Model()
  md.add_fem_variable("u", mf_u)
  md.add_initialized_data("lambda", E*NU/((1 + NU)*(1 - 2*NU)))
  md.add_initialized_data("mu", E/(2*(1 + NU)))
  md.add_linear_term(mim, "lambda*Div_u*Div_Test_u + mu*(Grad_u + Grad_u'):Grad_Test_u")
  for rg in [1, 2]:
    md.add_linear_term(mim, "-Normal.Test_u", rg)
  md.add_linear_term(mim, "1e6*u.Test_u", 3)
  if case == "through":
    for rg in [4, 5]:
      md.add_linear_term(mim, "1e6*(u.Normal)*(Test_u.Normal)", rg)
  nit, conv, info = md.solve()
  assert conv, info["message"]

  # the opening along x in the crack plane (3D: z = T/2)
  x = A*np.arange(21)/20*0.999 + 1e-6
  above = [[xi, 1e-7, 0.5*T][:dim] for xi in x]
  below = [[xi, -1e-7, 0.5*T][:dim] for xi in x]
  delta = (md.interpolation("u(2)", np.transpose(above), m)
           - md.interpolation("u(2)", np.transpose(below), m))
  # the stress intensity factors
  K, front = ls.crack_sifs(mim, mf_u, md.variable("u"), E, NU, 0.3, 0.25 if dim == 3 else None)
  return dict(x=x, delta=delta, dofs=mf_u.nb_dof(), K=K, front=front)


if __name__ == "__main__":
  case = sys.argv[1] if len(sys.argv) > 1 else "semi"
  h = float(sys.argv[2]) if len(sys.argv) > 2 else 0.125
  s = simulate(case, h)
  print("%s, h %g, %i DOFs" % (case, h, s["dofs"]))
  print("       x     opening")
  for x, d in zip(s["x"], s["delta"]):
    print("%8.4f %11.6f" % (x, d))
  print("  point                       K_I       K_II      K_III")
  # 2D: the tip; 3D: the points of the front's slice (a point on several
  # convexes once per convex)
  X = s["front"] if case == "edge" else s["front"].pts()
  rows = {tuple(x.round(4)): k for x, k in zip(X.T, s["K"].T)}
  for x in sorted(rows):
    print("  (%s) %9.5f %9.5f %9.5f" % (", ".join("%7.4f" % v for v in x), *rows[x]))
  if case != "edge":
    path = "surface_crack_%s_sifs.vtu" % case
    s["front"].export_to_vtu(path, s["K"], "K")
    print("  wrote %s" % path)
