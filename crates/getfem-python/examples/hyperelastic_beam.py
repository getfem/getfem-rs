#!/usr/bin/env python3
"""Large deflection of a compressible neo-Hookean cantilever (Q2 hexahedra),
with GetFEM-rs: clamped at x = 0, loaded by 10 times its own weight in 10
steps. Only the stored energy is written: GWFL derives the residual and
the tangent. Newton's three phases (full Newton where it converges).
Writes hyperelastic.vtu."""
import numpy as np

import getfem_rs as gf

ENERGY = ("K/2*sqr(log(Det(Id(3)+Grad_u)))"
          " + G/2*(pow(Det(Id(3)+Grad_u),-2/3)*Norm_sqr(Id(3)+Grad_u) - 3)"
          " - load*[0,0,-1].u")

L, E, nu = 10., 1e4, 0.3
m = gf.Mesh("structured", "GT='GT_QK(3,1)';ORG=[0,0,0];SIZES=[%g,1,1];NSUBDIV=[20,2,2]" % L)
mf = gf.MeshFem(m, 3)
mf.set_classical_finite_element(2)
mim = gf.MeshIm(m, 4)
md = gf.Model()
md.add_fem_variable("u", mf)
md.add_initialized_data("K", E/(3*(1-2*nu)))
md.add_initialized_data("G", E/(2*(1+nu)))
md.add_initialized_data("load", 0.)
md.add_nonlinear_term(mim, ENERGY)
# the clamped end x = 0, region 1
m.set_region(1, m.outer_faces_in_box([0., 0., 0.], [0., 1., 1.]))
md.add_Dirichlet_condition_with_simplification("u", 1)
# ten times the gravity of the linear cantilever example
steps = 10
for step in range(1, steps + 1):
  load = 10.*step/steps
  md.set_variable("load", [load])
  nit, conv, info = md.solve()
  if not conv:
    raise RuntimeError("no convergence at load %g: %s" % (load, info["message"]))
  print("load %5.2f: %i iterations (%s)" % (load, nit, info["method"]))
u = md.variable("u")
# tip: mean displacement of the nodes at x = L
tip = np.abs(mf.basic_dof_nodes()[0] - L) < 1e-12
print("%i DOFs, tip displacement: vertical %.3f, horizontal %.3f (linear elasticity: 14.97, 0)"
      % (md.nb_dof(), u[2::3][tip[2::3]].mean(), u[0::3][tip[0::3]].mean()))
mf.export_to_vtu("hyperelastic.vtu", u, "u", ascii=True)
