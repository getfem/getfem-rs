#!/usr/bin/env python3
"""XFEM: a square with a crack (GetFEM's demo_crack.py), with GetFEM-rs,
as the Rust example crack. Linear elasticity on [-0.5, 0.5]^2 with a crack
along y = 0 for x < 0 (tip at the origin), the opening mode reference
solution imposed on the border (penalization). The displacement is P1
enriched by the Heaviside function of the crack ("levelset") and near the
tip by the four singular functions of linear fracture mechanics, either
multiplied by a cutoff (variant "cutoff") or by the P1 functions of the
nodes within 0.1 of the tip ("product"). Variant "two_cracks" adds a
vertical crack from (-0.125, -0.375) to (-0.125, 0.375) crossing the
first one, enriched at both tips, its jump penalized (Xfem_plus,
Xfem_minus). Prints the norms of u and its distances to the reference
solution, writes the displacement on the cut mesh to crack.vtu. Optional
arguments: the variant (default two_cracks), elements per side (default
20, as the demo)."""
import sys

import numpy as np

import getfem_rs as gf

DIRICHLET = 101
LAMBDA, MU = 1.25e10, 1.875e10


def solve(variant, nx):
  x = -0.5 + np.arange(nx + 1)/nx
  m = gf.Mesh("regular_simplices", x, x)
  m.set_region(DIRICHLET, m.outer_faces())
  ck = [gf.GlobalFunction("crack", i) for i in range(4)]
  ls = gf.LevelSet(m, 1, "y", "x")
  ls2 = gf.LevelSet(m, 1, "x+0.125", "abs(y)-0.375")
  mls = gf.MeshLevelSet(m)
  mls.add(ls)
  if variant == "two_cracks":
    mls.add(ls2)
  mls.adapt()

  mf_pre_u = gf.MeshFem(m)
  mf_pre_u.set_finite_element(gf.Fem("FEM_PK(2,1)"))
  mfls_u = gf.MeshFem("levelset", mls, mf_pre_u)
  if variant == "cutoff":
    coff = gf.GlobalFunction("cutoff", "polynomial2", 0.01, 0.4)
    global_mf = gf.MeshFem("global function", m, ls, [f*coff for f in ck])
    mf_u = gf.MeshFem("sum", global_mf, mfls_u)
  else:
    # the singular functions on the P1 partition of unity of the nodes
    # within 0.1 of the tips
    pu_mf = gf.MeshFem(m)
    pu_mf.set_classical_finite_element(1)
    pts = pu_mf.basic_dof_nodes()
    def near(c):
      # the nodes at 0.1 included (the demo's coordinates round inward)
      d = (pts[0] - c[0])**2 + (pts[1] - c[1])**2
      return np.nonzero(d <= 0.1**2 + 1e-12)[0]
    global_mf = gf.MeshFem("global function", m, ls, ck)
    mf_xfem_sing = gf.MeshFem("product", pu_mf, global_mf, near([0., 0.]))
    if variant == "product":
      mf_u = gf.MeshFem("sum", mf_xfem_sing, mfls_u)
    else:
      global_mf2 = gf.MeshFem("global function", m, ls2, ck)
      pu_dofs = np.union1d(near([-0.125, 0.375]), near([-0.125, -0.375]))
      mf_xfem_sing2 = gf.MeshFem("product", pu_mf, global_mf2, pu_dofs)
      mf_u = gf.MeshFem("sum", mf_xfem_sing, mf_xfem_sing2, mfls_u)
  mf_u.set_qdim(2)
  # GetFEM's rule on every sub-triangle (tau 0)
  mim = gf.MeshIm("levelset", mls, "all",
                  gf.Integ("IM_STRUCTURED_COMPOSITE(IM_TRIANGLE(6),3)"),
                  gf.Integ("IM_STRUCTURED_COMPOSITE(IM_GAUSS_PARALLELEPIPED(2,6),9)"),
                  gf.Integ("IM_STRUCTURED_COMPOSITE(IM_TRIANGLE(6),5)"), tau=0.)

  # the reference solution: the opening mode in the singular functions
  mf_ue = gf.MeshFem("global function", m, ls, ck, 2)
  a, b = 2 + 2*MU/(LAMBDA + 2*MU), -2*(LAMBDA + MU)/(LAMBDA + 2*MU)
  ue = np.array([0., a - b, a + b, 0., -b, 0., 0., b])/(2*np.pi)

  md = gf.Model()
  md.add_fem_variable("u", mf_u)
  md.add_initialized_data("lambda", LAMBDA)
  md.add_initialized_data("mu", MU)
  md.add_linear_term(mim, "lambda*Div_u*Div_Test_u + mu*(Grad_u + Grad_u'):Grad_Test_u")
  md.add_initialized_fem_data("DirichletData", mf_ue, ue)
  md.add_Dirichlet_condition_with_penalization(mim, "u", 1e12, DIRICHLET, "DirichletData")
  if variant == "two_cracks":
    # the jump across the second crack penalized
    mim_bound2 = gf.MeshIm("levelset", mls, "boundary(b)",
                           gf.Integ("IM_STRUCTURED_COMPOSITE(IM_TRIANGLE(6),3)"), tau=0.)
    md.add_linear_term(mim_bound2, "1e17*(Xfem_plus(u)-Xfem_minus(u)).(Xfem_plus(Test_u)-Xfem_minus(Test_u))")
  nit, conv, info = md.solve()
  assert conv, info["message"]

  def integral(e, mim=mim):
    return gf.asm_generic(mim, 0, e, -1, md)
  l2, d2 = integral("Norm_sqr(u)"), integral("Norm_sqr(u-DirichletData)")
  r = dict(mls=mls, mf_u=mf_u, u=md.variable("u"),
           norms=[np.sqrt(l2), np.sqrt(l2 + integral("Norm_sqr(Grad_u)"))],
           dist=[np.sqrt(d2), np.sqrt(d2 + integral("Norm_sqr(Grad_u-Grad_DirichletData)"))])
  if variant == "two_cracks":
    r["jump"] = np.sqrt(integral("Norm_sqr(Xfem_plus(u)-Xfem_minus(u))", mim_bound2))
  return r


if __name__ == "__main__":
  variant = sys.argv[1] if len(sys.argv) > 1 else "two_cracks"
  nx = int(sys.argv[2]) if len(sys.argv) > 2 else 20
  s = solve(variant, nx)
  print("%i DOFs; |u| L2 %.6e, H1 %.6e; |u - ue| L2 %.6e, H1 %.6e; jump %.3e"
        % ((s["mf_u"].nb_dof(),) + tuple(s["norms"]) + tuple(s["dist"]) + (s.get("jump", 0.),)))
  # the displacement on the cut mesh, discontinuous across the cracks
  # (alpha 0.001: the nodes just off the cracks, each side its value)
  mfv = gf.MeshFem(s["mls"].cut_mesh(), 2)
  mfv.set_classical_discontinuous_finite_element(2, 0.001)
  v = gf.interpolation(s["mf_u"], s["u"], mfv)
  mfv.export_to_vtu("crack.vtu", v, "u")
  print("wrote crack.vtu")
