#!/usr/bin/env python3
"""Two crossing cracks in a 3D block, linearized elasticity, with
GetFEM-rs, as the Rust example crossing_cracks_3d: a 40 x 10 x 20 mm
block, clamped left and right, with the curved cracks y = 2.5 tanh(x/7)
(extruded along z) and y = -2.5 tanh((x - 0.3 z)/7) + 0.15 z (oblique),
inside the ellipsoid (x/16)^2 + (y/6)^2 + (z/8)^2 < 1, their fronts closed
curves. Q2 on hexahedra, enriched by the Heaviside functions of the cracks
and, on the elements the fronts pass through, by the crack tip functions;
the DOFs without support dropped. Two load cases: a uniform traction
pulling the top and bottom faces apart, and a uniform pressure on the
lips. Writes the displacement on a grid of points in the planes z = -4,
0, 4 to crossing_cracks_3d_<case>.txt (columns x y z ux uy uz). Optional
arguments: element size [mm] (default 2.5), number of cracks (default 2;
1: the extruded one)."""
import sys

import numpy as np

import getfem_rs as gf

E = 1e3             # [N/mm^2] Young modulus
NU = 0.3            # Poisson ratio
SIGMA = 1.          # [N/mm^2] traction on top and bottom
PRESSURE = 1.       # [N/mm^2] pressure on the lips
L, H, W = 40., 10., 20.  # [mm]
R1, TILT, SHEAR = 2.5, 0.15, 0.3
AX, AY, AZ = 16., 6., 8.  # semi-axes of the ellipsoid bounding the cracks
# planes z = const of the output, points per mm in x and y
PLANES = [-4., 0., 4.]
DENSITY = 5
LEFT, RIGHT, BOTTOM, TOP = 101, 102, 103, 104


def crack(m, sign):
  """the crack surface (primary): extruded for sign > 0, oblique
  otherwise; the ellipsoid (secondary), scaled to a distance near the
  front (the crack tip functions take the level sets as coordinates
  around the front)"""
  shear, tilt = (0., 0.) if sign > 0 else (SHEAR, TILT)
  return gf.LevelSet(m, 2, "y-(%r)*tanh((x-%r*z)/7)-%r*z" % (sign*R1, shear, tilt),
                     "(sqrt(sqr(x/%r)+sqr(y/%r)+sqr(z/%r))-1)*%r" % (AX, AY, AZ, AX))


def points():
  """the output points (3 x n): grids in the planes z = const, the cell
  centers of DENSITY cells per mm (off the mesh faces)"""
  pts = [[-L/2 + (i + 0.5)/DENSITY, -H/2 + (j + 0.5)/DENSITY, z]
         for z in PLANES for j in range(int(H)*DENSITY) for i in range(int(L)*DENSITY)]
  return np.transpose(pts)


def simulate(h, nb_cracks, pts):
  """both load cases with elements of size h [mm] and nb_cracks cracks
  (1 or 2): the displacement at the points pts, 3 values per point, for
  the traction and the pressure"""
  m = gf.Mesh("cartesian Q1", np.linspace(-L/2, L/2, round(L/h) + 1),
              np.linspace(-H/2, H/2, round(H/h) + 1), np.linspace(-W/2, W/2, round(W/h) + 1))
  for rg, nn in [(LEFT, [-1., 0., 0.]), (RIGHT, [1., 0., 0.]), (BOTTOM, [0., -1., 0.]),
                 (TOP, [0., 1., 0.])]:
    m.set_region(rg, m.outer_faces_with_direction(nn, 0.5))
  lss = [crack(m, s) for s in [1., -1.][:nb_cracks]]
  mls = gf.MeshLevelSet(m)
  for ls in lss:
    mls.add(ls)
  mls.adapt()

  mf_pre = gf.MeshFem(m)
  mf_pre.set_finite_element(gf.Fem("FEM_QK(3,2)"))
  mfls = gf.MeshFem("levelset", mls, mf_pre)
  # the crack tip functions on the Q1 partition of unity of the Lagrange
  # DOFs of the front elements
  pu_mf = gf.MeshFem(m)
  pu_mf.set_classical_finite_element(1)
  ck = [gf.GlobalFunction("crack", i) for i in range(4)]
  mf_sing = [gf.MeshFem("product", pu_mf, gf.MeshFem("global function", m, ls, ck),
                        pu_mf.basic_dof_on_region(mls.crack_tip_convexes(ls)))
             for ls in lss]
  base = gf.MeshFem("sum", *mf_sing, mfls)
  mim = gf.MeshIm("levelset", mls, "all", gf.Integ("IM_TETRAHEDRON(5)"),
                  gf.Integ("IM_PRODUCT(IM_TRIANGLE(3),IM_GAUSS1D(3))"),
                  gf.Integ("IM_GAUSS_PARALLELEPIPED(3,4)"))
  # the DOFs with support (enriched DOFs without are singular)
  kept = base.dofs_with_support(mim, 1e-8)
  base.set_qdim(3)
  mf_u = gf.MeshFem("partial", base, [3*i + c for i in kept for c in range(3)])
  mim_bound = [gf.MeshIm("levelset", mls, "boundary(%s)" % c, gf.Integ("IM_TETRAHEDRON(3)"))
               for c in "ab"[:nb_cracks]]

  md = gf.Model()
  md.add_fem_variable("u", mf_u)
  md.add_initialized_data("lambda", E*NU/((1 + NU)*(1 - 2*NU)))
  md.add_initialized_data("mu", E/(2*(1 + NU)))
  md.add_initialized_data("sigma", 0.)
  md.add_initialized_data("p", 0.)
  md.add_linear_term(mim, "lambda*Div_u*Div_Test_u + mu*(Grad_u + Grad_u'):Grad_Test_u")
  # traction on top and bottom
  for rg in [TOP, BOTTOM]:
    md.add_linear_term(mim, "-sigma*Normal.Test_u", rg)
  # pressure on the lips (Normal: the gradient of the level set, from the
  # minus to the plus side)
  for mb in mim_bound:
    md.add_linear_term(mb, "p*Normal.Xfem_minus(Test_u) - p*Normal.Xfem_plus(Test_u)")
  # clamped left and right
  for rg in [LEFT, RIGHT]:
    md.add_Dirichlet_condition_with_multiplier(mim, "u", 1, rg)

  def solve(sigma, p):
    md.set_variable("sigma", [sigma])
    md.set_variable("p", [p])
    nit, conv, info = md.solve()
    assert conv, info["message"]
    return md.interpolation("u", pts, m)
  return [solve(SIGMA, 0.), solve(0., PRESSURE)]


if __name__ == "__main__":
  h = float(sys.argv[1]) if len(sys.argv) > 1 else 2.5
  nb_cracks = int(sys.argv[2]) if len(sys.argv) > 2 else 2
  pts = points()
  for case, u in zip(["traction", "pressure"], simulate(h, nb_cracks, pts)):
    path = "crossing_cracks_3d_%s.txt" % case
    np.savetxt(path, np.hstack([pts.T, u.reshape(-1, 3)]), fmt="%.10g %.10g %.10g %e %e %e",
               header="x y z ux uy uz")
    print("%s: max |u_i| %.4e mm, %s" % (case, np.abs(u).max(), path))
