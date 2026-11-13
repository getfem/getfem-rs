#!/usr/bin/env python3
"""Convergence tests of the Newton driver, with GetFEM-rs, on the plate of
thermo_coupling.py: the residual of each variable's equations, the hint
when some equations are solved less well than the global test suggests,
a tolerance below the rounding floor (converged there, with a hint), and tolerances
per variable with explicit scales."""
import os

import numpy as np

import getfem_rs as gf

MESH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "../../../data/meshes/thermo_elasticity_electrical_coupling.mesh")


def thermo_plate():
  """The plate of thermo_coupling.py (the same model): model, integration
  method, multiplier names."""
  m = gf.Mesh("load", MESH)
  # regions of the mesh file: 1 right, 2 left, 3 top, 4 bottom
  right, left = 1, 2
  def space(q, k):
    mf = gf.MeshFem(m, q)
    mf.set_classical_finite_element(k)
    return mf
  mim = gf.MeshIm(m, 4)
  md = gf.Model()
  md.add_fem_variable("u", space(2, 2))
  md.add_fem_variable("T", space(1, 2))
  md.add_fem_variable("V", space(1, 2))
  for n, v in [("t", 1.), ("E", 21e6), ("nu", 0.3), ("alpha_th", 16.6e-6), ("T0", 20.),
               ("F", 100e2), ("rho_0", 1.754e-8), ("alpha", 0.0039), ("V_left", 0.1),
               ("kappa", 4.), ("D", 10.), ("T_air", 20.)]:
    md.add_initialized_data(n, v)
  md.add_macro("sigma",
               "E/(1+nu)*(nu/(1-nu)*(Div_u-2*alpha_th*T)*Id(2) + Sym(Grad_u)-alpha_th*T*Id(2))")
  md.add_macro("rho", "rho_0*(1+alpha*(T-T0))")
  md.add_linear_term(mim, "t*sigma:Grad_Test_u")
  md.add_linear_term(mim, "-t*F*Test_u(1)", right)
  mu = md.add_Dirichlet_condition_with_multiplier(mim, "u", space(2, 1), left)
  md.add_nonlinear_term(mim, "t/rho*Grad_V.Grad_Test_V")
  mv1 = md.add_Dirichlet_condition_with_multiplier(mim, "V", space(1, 1), right)
  mv2 = md.add_Dirichlet_condition_with_multiplier(mim, "V", space(1, 1), left, "V_left")
  md.add_nonlinear_term(mim, "t*kappa*Grad_T.Grad_Test_T + 2*D*(T-T_air)*Test_T"
                             " - t/rho*Norm_sqr(Grad_V)*Test_T")
  # convection on top and bottom, region 5
  m.set_region(5, m.region([3, 4]))
  md.add_linear_term(mim, "t*D*(T-T_air)*Test_T", 5)
  return md, mim, (mu, mv1, mv2)


def temperature(md, mim):
  return np.sqrt(gf.asm_generic(mim, 0, "sqr(T)", -1, md))


if __name__ == "__main__":
  # flush: Newton's verbose output comes from Rust
  print("1. global test at the default rtol = 1e-8, verbose:", flush=True)
  md, mim, _ = thermo_plate()
  nit, conv, info = md.solve(verbose=True)
  print("   converged: %s, L2 norm of T %.6f\n" % (conv, temperature(md, mim)))

  print("2. global rtol = 1e-20, below the rounding floor of this problem:")
  md, _, _ = thermo_plate()
  nit, conv, info = md.solve(rtol=1e-20)
  print("   %s in %i solves: %s\n" % (info["status"], nit, info["hint"]))

  print("3. a tolerance per physics, scales from the residuals of run 1:")
  md, mim, (mu, mv1, mv2) = thermo_plate()
  # only the variable tolerances count
  nit, conv, info = md.solve(atol=float("inf"), variable_tolerances=[
    (["u", mu], 1e-10*5e4), ("T", 1e-10*2e4), (["V", mv1, mv2], 1e-8*2e4)])
  for step, r in enumerate(info["residuals_per_tolerance"]):
    print("   after %i steps: %s" % (step, ", ".join("%s %.1e" % (n, x) for n, x in r.items())))
  print("   converged: %s, L2 norm of T %.6f" % (conv, temperature(md, mim)))
