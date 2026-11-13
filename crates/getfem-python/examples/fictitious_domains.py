#!/usr/bin/env python3
"""Fictitious domain (GetFEM's demo_fictitious_domains.py), with
GetFEM-rs: linear elasticity (lambda = mu = 1, volume load (0, 10)) on
[-0.5, 0.5]^2 meshed by 40 x 40 squares, Q3, outside 25 holes (level set
a: the minimum of their (x - xc)^2 + (y - yc)^2 - R^2, degree 2) or on a
segment of length 0.38 through the origin at 60 degrees (level set b and
its secondary), u = 0 on the holes' boundaries and on the segment
(multipliers Q1 on the cut elements: the independent ones, GetFEM drops
the dependent ones silently). Prints the norms of u and the volume and
boundary length integrated, writes u and the von Mises stress to
fictitious_domains.vtu."""
import numpy as np

import getfem_rs as gf


def solve(n=40):
  x = np.linspace(-0.5, 0.5, n + 1)
  m = gf.Mesh("cartesian", x, x)
  ls = gf.LevelSet(m, 2)
  ls2 = gf.LevelSet(m, 2, "with_secondary")
  px, py = ls.mf().basic_dof_nodes()
  uls = 1000*np.ones(px.shape)
  for ix in range(5):
    for iy in range(5):
      xc = ix/4*0.8 - 0.4 + (0.05 if iy % 2 == 1 else -0.05)
      yc = iy/4*0.8 - 0.4
      r = 0.03 + 0.005*iy
      uls = np.minimum(uls, (px - xc)**2 + (py - yc)**2 - r**2)
  ls.set_values(uls)
  t = np.pi/3
  ls2.set_values("-x*sin(%r)+y*cos(%r)" % (t, t), "abs(y)+abs(x)-0.19")
  mls = gf.MeshLevelSet(m)
  mls.add(ls)
  mls.add(ls2)
  mls.adapt()
  mim_bound = gf.MeshIm("levelset", mls, "boundary(a+b)", gf.Integ("IM_TRIANGLE(6)"))
  mim = gf.MeshIm("levelset", mls, "all(a+b)", gf.Integ("IM_TRIANGLE(6)"))
  mim.set_integration_method(4)

  mfu = gf.MeshFem(m, 2)
  mfu.set_finite_element(gf.Fem("FEM_QK(2,3)"))
  mf_mult = gf.MeshFem(m, 2)
  mf_mult.set_finite_element(gf.Fem("FEM_QK(2,1)"))
  md = gf.Model()
  md.add_fem_variable("u", mfu)
  md.add_initialized_data("lambda", 1.)
  md.add_initialized_data("mu", 1.)
  md.add_initialized_data("VolumicData", [0., 10.])
  md.add_linear_term(mim, "lambda*Div_u*Div_Test_u + mu*(Grad_u + Grad_u'):Grad_Test_u")
  md.add_linear_term(mim, "-VolumicData.Test_u")
  dofs = md.select_multiplier_dofs_for_variable("u", mim_bound, mf_mult, -1)
  md.add_Dirichlet_condition_with_multiplier(mim_bound, "u", gf.MeshFem("partial", mf_mult, dofs), -1)
  nit, conv, info = md.solve()
  assert conv, info["message"]

  def integral(e, mim=mim):
    return gf.asm_generic(mim, 0, e, -1, md)
  return dict(m=m, md=md, mfu=mfu, mim=mim, nb_mult=len(dofs),
              area=integral("1"), length=integral("1", mim_bound),
              norms=[np.sqrt(integral("Norm_sqr(u)")), np.sqrt(integral("Norm_sqr(Grad_u)"))],
              trace=np.sqrt(integral("Norm_sqr(u)", mim_bound)))


if __name__ == "__main__":
  s = solve()
  print("%i multipliers; area %.6f, boundary length %.6f; |u| L2 %.6e, |grad u| L2 %.6e,"
        " |u| L2 on the boundary %.3e" % ((s["nb_mult"], s["area"], s["length"]) + tuple(s["norms"])
                                          + (s["trace"],)))
  mfdu = gf.MeshFem(s["m"], 1)
  mfdu.set_finite_element(gf.Fem("FEM_QK_DISCONTINUOUS(2,2)"))
  vm = s["md"].local_projection(s["mim"], "sqrt(3/2)*Norm(Deviator(lambda*Div_u*Id(2)"
                                "+mu*(Grad_u+Grad_u')))", mfdu)
  s["mfu"].export_to_vtu("fictitious_domains.vtu", s["md"].variable("u"), "u",
                         mfdu, vm, "von Mises")
  print("wrote fictitious_domains.vtu")
