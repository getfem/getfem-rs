#!/usr/bin/env python3
"""Thermo-electro-elastic plate with three holes (GetFEM's
thermo_elasticity_electrical_coupling), with GetFEM-rs: electric potential
with a temperature dependent resistivity, Joule heating, thermal
expansion. Mesh from GetFEM's mesher (data/meshes). Writes thermo.vtu."""
import os

import numpy as np

import getfem_rs as gf

# regions of the mesh file, and the union of top and bottom
RIGHT, LEFT, TOP, BOTTOM, TOP_BOTTOM = 1, 2, 3, 4, 5

m = gf.Mesh("load", os.path.join(os.path.dirname(os.path.abspath(__file__)),
            "../../../data/meshes/thermo_elasticity_electrical_coupling.mesh"))
def space(q, k):
  mf = gf.MeshFem(m, q)
  mf.set_classical_finite_element(k)
  return mf
mfu, mft = space(2, 2), space(1, 2)
mim = gf.MeshIm(m, 4)

md = gf.Model()
md.add_fem_variable("u", mfu)
md.add_fem_variable("T", mft)
md.add_fem_variable("V", mft)
for n, v in [("t", 1.), ("E", 21e6), ("nu", 0.3), ("alpha_th", 16.6e-6), ("T0", 20.),
             ("F", 100e2), ("rho_0", 1.754e-8), ("alpha", 0.0039), ("V_left", 0.1),
             ("kappa", 4.), ("D", 10.), ("T_air", 20.)]:
  md.add_initialized_data(n, v)
md.add_macro("sigma",
             "E/(1+nu)*(nu/(1-nu)*(Div_u-2*alpha_th*T)*Id(2) + Sym(Grad_u)-alpha_th*T*Id(2))")
md.add_macro("rho", "rho_0*(1+alpha*(T-T0))")
# elasticity with thermal expansion, clamped left, traction right
md.add_linear_term(mim, "t*sigma:Grad_Test_u")
md.add_linear_term(mim, "-t*F*Test_u(1)", RIGHT)
md.add_Dirichlet_condition_with_multiplier(mim, "u", space(2, 1), LEFT)
# current with temperature dependent resistivity, 0.1 V across
md.add_nonlinear_term(mim, "t/rho*Grad_V.Grad_Test_V")
md.add_Dirichlet_condition_with_multiplier(mim, "V", space(1, 1), RIGHT)
md.add_Dirichlet_condition_with_multiplier(mim, "V", space(1, 1), LEFT, "V_left")
# heat: conduction, convection on the faces and edges, Joule heating
md.add_nonlinear_term(mim, "t*kappa*Grad_T.Grad_Test_T + 2*D*(T-T_air)*Test_T"
                           " - t/rho*Norm_sqr(Grad_V)*Test_T")
# convection on top and bottom
m.set_region(TOP_BOTTOM, m.region([TOP, BOTTOM]))
md.add_linear_term(mim, "t*D*(T-T_air)*Test_T", TOP_BOTTOM)

# one Newton over all fields (the coupling is one-way, T -> u)
nit, conv, info = md.solve(rtol=1e-10)
if not conv:
  raise RuntimeError(info["message"])
l2 = np.sqrt(gf.asm_generic(mim, 0, "sqr(T)", -1, md))
print("%i DOFs, L2 norm of the temperature %.6f" % (md.nb_dof(), l2))
mfu.export_to_vtu("thermo.vtu", md.variable("u"), "u", mft, md.variable("T"), "T",
                  mft, md.variable("V"), "V", ascii=True)
