#!/usr/bin/env python3
"""The GWFL paper's example 1 (crates/getfem-tests/generators/paper/getfem_gwfl_example1.py), with
GetFEM-rs: an inflated hyperelastic membrane. A disc of diameter 100 and
thickness 2 (structured_ball, Q2, embedded in 3D), the displacement u and
the thickness vector n as unknowns (F = I2 + Grad u + n x e3), a
compressible neo-Hookean energy, a follower pressure gamma * pmax, the rim
clamped; Moore-Penrose continuation in gamma past the limit load. Writes
gamma, the step length and max |u| per step to membrane.dat and the final
u to membrane.vtu. Optional arguments: the core and peel subdivisions
(default 10, 3, as the paper), steps (default 300)."""
import sys

import numpy as np

import getfem_rs as gf

DIR_BOUNDARY = 6

def arg(k, default):
  return int(sys.argv[k]) if len(sys.argv) > k else default
nsub, steps = [arg(1, 10), arg(2, 3)], arg(3, 300)
d, h_thick, e, nu, pmax = 100., 2., 5e3, 0.4, 1.5e2
m = gf.Mesh("structured_ball", "GT='GT_QK(2,2)';ORG=[0,0];SIZES=[%r];NSUBDIV=[%i,%i];"
            "SYMMETRIES=0" % (d/2, nsub[0], nsub[1]))
m.transform(np.array([[1., 0.], [0., 1.], [0., 0.]]))
m.set_region(DIR_BOUNDARY, m.outer_faces())
mfu = gf.MeshFem(m, 3)
mfu.set_classical_finite_element(2)
mfn = gf.MeshFem(m, 3)
mfn.set_classical_finite_element(2)
mim = gf.MeshIm(m, 5)
md = gf.Model()
md.add_fem_variable("u", mfu)
md.add_fem_variable("n", mfn)
md.set_variable("n", md.interpolation("[0,0,1]", mfn))
for k, v in [("gamma", 0.), ("pmax", pmax), ("kappa", e/(3*(1 - 2*nu))),
             ("mu", e/(2*(1 + nu))), ("H", h_thick)]:
  md.add_initialized_data(k, v)
md.add_macro("F", "[1,0,0;0,1,0;0,0,0]+Grad(u)+n@[0,0,1]")
md.add_nonlinear_term(mim, "H*0.5*kappa*sqr(log(Det(F)))+H*0.5*mu*(pow(Det(F),-2/3)*Norm_sqr(F)-3)")
md.add_nonlinear_term(mim, "gamma*pmax*Det(F)*((Inv(F)'*[0,0,-1]).Test_u)")
md.add_Dirichlet_condition_with_multiplier(mim, "u", mfu, DIR_BOUNDARY)
nu_ = mfu.nb_dof()
print("%i displacement DOFs, %i model DOFs" % (nu_, md.nb_dof()))
options = dict(h_init=1e-1, h_max=1., h_min=1e-4, h_dec=1/np.sqrt(10),
               h_inc=np.sqrt(10), max_iter=10, thr_iter=6, max_res=1e-5, max_diff=1e-8,
               min_cos=0.999)
limit = (0., 0, 0.)
T, h = None, 0.
with open("membrane.dat", "w") as f:
  for step in range(steps + 1):
    if md.variable("gamma")[0] >= 1:
      break
    if step == 0:
      nit, conv, info = md.solve(atol=1e-7)
      assert conv, info["message"]
    T, h, info = md.numerical_continuation_step("gamma", [("u", 1.)], T, h, **options)
    assert info["status"] != "failed", "continuation failed at step %i" % step
    umax = np.abs(md.variable("u")).max()
    g = md.variable("gamma")[0]
    if g > limit[0]:
      limit = (g, step, umax)
    f.write("step=%i gamma=%.17g h=%.17g umax=%.17g\n" % (step, g, h, umax))
    if step % 10 == 0:
      print("step %i: gamma %.6f, max |u| %.3f, h %.3e" % (step, g, umax, h))
print("limit load gamma %.6f at step %i (max |u| %.3f)" % limit)
mfu.export_to_vtu("membrane.vtu", md.variable("u"), "u")
print("wrote membrane.vtu and membrane.dat")
