#!/usr/bin/env python3
"""Phase field fracture of a plate with hydrogen: the GWFL paper's example
2 (crates/getfem-tests/generators/paper/getfem_gwfl_example2.py), with GetFEM-rs (as the Rust
example phase_field_hydrogen).
u Q2, the damage d Q2, the pressure p Q1, the hydrogen concentration C Q1
(Gc falls with C), inertia written out (u_prev, v_prev), an initial crack
in the history field psi0_max, the top and bottom faces pulled apart.
Time step control: Dt x 1.5 after a solve in at most 4 linear solves,
never above the initial time step (Newton with its damped methods rarely
fails, the halving alone would not keep Dt small), halved after a failed
one. v_prev from the old u_prev (the paper's script set u_prev first, its
v_prev always 0). Writes the reaction force per step to
phase_field_hydrogen_forces.dat and the final state to
phase_field_hydrogen.vtu (u, d, p, C; the Cauchy stress, psi0_max, psi0
and Gc on discontinuous Q2). Optional arguments: elements in x and y
(default 80 x 60, as the paper), steps (default 250)."""
import sys

import numpy as np

import getfem_rs as gf

# Input data
arg = lambda k, d: int(sys.argv[k]) if len(sys.argv) > k else d
NX, NY = arg(1, 80), arg(2, 60)  # elements in x and y
LX, LY = 1., 1.           # [mm] length, height
E, nu = 210e3, 0.3        # [N/mm^2], [-]
rho = 8e-9                # [ton/mm^3]
Gc0 = 2.7e0               # [N/mm]
chi = 0.89                # [-] hydrogen damage coefficient
Dgb0 = 30e6               # [N*mm/mol] Gibbs free energy jump
C0 = 0.5                  # [wt ppm] initial hydrogen concentration
VH = 2000.                # [mm^3/mol] molar volume of hydrogen in solution
R = 8.314e3               # [N*mm/(K*mol)] universal gas constant
T = 300.                  # [K] absolute temperature
ll = 0.015                # [mm] length scale
strain_rate = 2e-4
init_time_step = 0.4
steps = arg(3, 250)

# Mesh: graded, finer near x < 0 and y = 0
NX = 2*((NX+1)//2)
NY = 2*((NY+1)//2)
XX1 = np.linspace(-1., 0., NX//4+1)
XX2 = np.linspace(0., 1., (NX-NX//4)+1)[1:]
XX = np.concatenate((LX/2*(0.2*XX1+0.8*np.sign(XX1)*np.power(np.abs(XX1), 1.5)),
                     LX/2*XX2))
YY = np.linspace(-1., 1., NY+1)
YY = LY/2*(0.2*YY+0.8*np.sign(YY)*np.power(np.abs(YY), 1.5))
mesh = gf.Mesh("cartesian", XX, YY)
T_RG, TB_RG, EXT_RG = 6, 7, 8
e = 1e-5
mesh.set_region(T_RG, mesh.outer_faces_in_box([-LX/2-e, LY/2-e], [LX/2+e, LY/2+e]))
mesh.set_region(TB_RG, mesh.outer_faces_in_box([-LX/2-e, -LY/2-e], [LX/2+e, -LY/2+e]))
mesh.extend_region(TB_RG, mesh.outer_faces_in_box([-LX/2-e, LY/2-e], [LX/2+e, LY/2+e]))
mesh.set_region(EXT_RG, mesh.outer_faces())

# Spaces and integration
def space(qdim, k):
  mf = gf.MeshFem(mesh, qdim)
  mf.set_classical_finite_element(k)
  return mf
mfu, mfd, mfp, mfC = space(2, 2), space(1, 2), space(1, 1), space(1, 1)
mfout = gf.MeshFem(mesh)
mfout.set_classical_discontinuous_finite_element(2)
mim4 = gf.MeshIm(mesh, 3)
mim9 = gf.MeshIm(mesh, 5)
mimd9 = gf.MeshImData(mim9)

# Model
md = gf.Model()
md.add_fem_variable("u", mfu)
md.add_fem_variable("d", mfd)
md.add_fem_variable("p", mfp)
md.add_fem_variable("C", mfC)
md.set_variable("C", C0*np.ones(mfC.nb_dof()))
md.add_fem_data("u_prev", mfu)
md.add_fem_data("v_prev", mfu)
md.add_im_data("psi0_max", mimd9)
md.add_initialized_data("Dt", init_time_step)
md.add_initialized_data("kappa", E/(3.*(1.-2.*nu)))
md.add_initialized_data("mu", E/(2*(1+nu)))
md.add_initialized_data("rho", rho)
md.add_initialized_data("Gc0", Gc0)
md.add_initialized_data("l", ll)
md.add_initialized_data("C0", C0)
md.add_initialized_data("chi", chi)
md.add_initialized_data("c1", np.exp(-Dgb0/(R*T))/5.5e-5)
md.add_initialized_data("c2", VH/(R*T))
# the initial crack
md.set_variable("psi0_max",
  "1e4*Gc0*min(1,max(0,1-(abs(X(2))+pos_part(X(1)))/(l/2)))")

md.add_linear_term(mim9, "rho/Dt*((u-u_prev)/Dt-v_prev).Test_u")
md.add_macro("degradation", "sqr(1-d)+1e-7")
md.add_macro("deveps", "Sym(Grad(u))-Div(u)/3*Id(2)")
md.add_macro("psi0", "(0.5*kappa*sqr(Div(u))+mu*Norm_sqr(deveps))")
md.add_macro("Gc", "(1-chi*C/(C+c1))*Gc0")
md.add_nonlinear_term(mim9, "degradation*(kappa*Div(u)*Id(2)+2*mu*deveps):Grad(Test_u)")
md.add_nonlinear_term(mim9, "(-2*(1-d)*max(psi0_max,psi0)*Test_d"
                            "+Gc*(d/l*Test_d+l*Grad(d).Grad(Test_d)))")
md.add_nonlinear_term(mim4, "(p+degradation*kappa*Div(u))*Test_p")
md.add_nonlinear_term(mim4, "(Grad(C)+c2*C*Grad(p)).Grad(Test_C)"
                            "+1e3*pos_part(2*d-1)*(C-C0)*Test_C")
# loading and boundary conditions
md.add_fem_data("dirichlet_data", mfu)
mult = md.add_Dirichlet_condition_with_multiplier(mim9, "u", mfu, TB_RG, "dirichlet_data")
md.add_Dirichlet_condition_with_multiplier(mim4, "C", mfC, EXT_RG, "C0")
print("%i displacement DOFs, %i model DOFs" % (mfu.nb_dof(), md.nb_dof()))

eps = 0.
with open("phase_field_hydrogen_forces.dat", "w") as f:
  for step in range(steps):
    eps_old = eps
    while True:  # until a solve converges, Dt halved
      dt = md.variable("Dt")[0]
      eps = eps_old + (step > 0)*strain_rate*dt
      x = md.from_variables()
      md.set_variable("dirichlet_data", "%.15g*[0;X(2)]" % eps)
      nit, conv, info = md.solve(max_iter=20, atol=1e-9)
      if conv:
        break
      md.to_variables(x)
      md.set_variable("Dt", [0.5*dt])
    md.set_variables([("v_prev", "(u-u_prev)/Dt"), ("u_prev", "u"),
                      ("psi0_max", "max(psi0_max,psi0)")])
    if nit <= 4:
      md.set_variable("Dt", [min(1.5*dt, init_time_step)])
    force = gf.asm_generic(mim9, 0, "%s(2)" % mult, T_RG, md)
    print("step %i: eps %.4e, Dt %.3e, %i linear solves (%s), force %.3f, max d %.4f"
          % (step, eps, dt, nit, info["method"] or "-", force, max(md.variable("d"))))
    f.write("step=%i average strain=%e reaction force=%e\n" % (step, eps, force))
    f.flush()
out = (mfu, md.variable("u"), "Displacements", mfd, md.variable("d"), "Damage",
       mfp, md.variable("p"), "Pressure", mfC, md.variable("C"), "C")
for i, j in [[1, 1], [2, 2], [1, 2]]:
  sigma = md.interpolation("degradation*(kappa*Div(u)*Id(2)+2*mu*deveps)(%i,%i)" % (i, j),
                           mfout)
  out += (mfout, sigma, "Cauchy Stress %i%i" % (i, j))
out += (mfout, md.local_projection(mim9, "psi0_max", mfout), "psi0_max",
        mfout, md.interpolation("psi0", mfout), "psi0",
        mfout, md.interpolation("Gc", mfout), "Gc")
mfout.export_to_vtu("phase_field_hydrogen.vtu", *out)
print("wrote phase_field_hydrogen.vtu and phase_field_hydrogen_forces.dat")
