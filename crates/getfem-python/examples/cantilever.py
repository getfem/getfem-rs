#!/usr/bin/env python3
"""3D cantilever under its own weight, with GetFEM-rs: linear elasticity
with Q2 hexahedra, L = 10, square section 1 x 1, E = 1e4, nu = 0.3,
clamped at x = 0. Tip deflection vs Timoshenko beam theory. Writes
cantilever.vtu."""
import numpy as np

import getfem_rs as gf

L, E, nu = 10., 1e4, 0.3
lam, mu = E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))
m = gf.Mesh("structured", "GT='GT_QK(3,1)';ORG=[0,0,0];SIZES=[%g,1,1];NSUBDIV=[20,2,2]" % L)
mf = gf.MeshFem(m, 3)
mf.set_classical_finite_element(2)
mim = gf.MeshIm(m, 4)
# u = 0 on the clamp: the unknown on the other DOFs
m.set_region(1, m.outer_faces_in_box([0., 0., 0.], [0., 1., 1.]))
clamped = set(mf.basic_dof_on_region(1))
mfu = gf.MeshFem("partial", mf, [d for d in range(mf.nb_dof()) if d not in clamped])
md = gf.Model()
md.add_fem_variable("u", mfu)
md.add_initialized_data("lambda", lam)
md.add_initialized_data("mu", mu)
md.add_linear_term(mim, "lambda*Div_u*Div_Test_u + mu*(Grad_u + Grad_u'):Grad_Test_u"
                        " - [0,0,-1].Test_u")
md.solve()
u = md.interpolation("u", mf)
# tip: mean vertical displacement of the nodes at x = L
tip = u[2::3][np.abs(mf.basic_dof_nodes()[0, 2::3] - L) < 1e-12].mean()
i2, area = 1/12, 1.
bending = L**4/(8*E*i2)
shear = L*L/(2*(5/6)*mu*area)
print("%i DOFs, tip deflection %.4f, beam theory %.4f (bending %.4f + shear %.4f)"
      % (mf.nb_dof(), -tip, bending + shear, bending, shear))
mf.export_to_vtu("cantilever.vtu", mfu, md.variable("u"), "u", ascii=True)
