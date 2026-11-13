#!/usr/bin/env python3
"""A gas-filled crack pair in a thermo-elastic block at large strains
(GetFEM's demo_cracked_thermo_elastic_body.py), with GetFEM-rs, as the
Rust example gas_filled_crack. A 40 x 10 mm block, clamped left and
right, with two curved cracks y = +-2.5 tanh(x/7) inside the circle of
radius 16; the gas in them (ideal gas, mass mR, temperature T, pressure
P) pushes their lips apart, exchanges heat with the lips, and the block
exchanges heat with its top and bottom faces, whose temperatures are
theta_T, theta_B. The displacement and the temperature are Q2 enriched by
the Heaviside functions of the cracks and by the singular functions near
the tips (the nodes within one element of them; the demo: 0.5 mm). The
crack is first opened with the pressure fixed, then the top face is
heated in five steps from 30 C to 130 C. Writes the displacement and
temperature on the cut mesh of each step to gas_filled_crack_<step>.vtu.
Optional arguments: elements across the block (default 10, as the demo),
the temperature steps (default 6)."""
import sys

import numpy as np

import getfem_rs as gf

E0 = 5e2            # [N/mm^2] Young modulus at the reference temperature
nu = 0.45           # Poisson ratio
beta = 8e-5         # [1/K] expansion coefficient
gamma = 0.1         # [1/K] thermal weakening
theta0 = 30.        # [C] reference temperature
alpha = 0.1         # [mm^2/s] thermal diffusivity
alpha_env = 0.02    # [N/mm/s/K] heat transfer to the environment
alpha_crack = 0.01  # [N/mm/s/K] heat transfer to the gas
mR = 287e3*5e-9     # [N mm/K] gas constant times mass
room_temp = 30.     # [C]
jump_temp = 100.    # [C] final temperature difference
L, H = 40., 10.     # [mm]
R1, R2 = 2.5, 16.
LEFT, RIGHT, BOTTOM, TOP = 101, 102, 103, 104


def simulate(ny, nsteps=6):
  """yields (step, state) after each temperature step"""
  h = H/ny
  nx = int(np.floor(L/h))
  m = gf.Mesh("cartesian", -L/2 + np.arange(nx + 1)*h, -H/2 + np.arange(ny + 1)*h)
  for rg, n in [(LEFT, [-1., 0.]), (RIGHT, [1., 0.]), (BOTTOM, [0., -1.]), (TOP, [0., 1.])]:
    m.set_region(rg, m.outer_faces_with_direction(n, 0.5))
  ls1 = gf.LevelSet(m, 2, "y-%r*tanh(x/7)" % R1, "x*x+y*y-sqr(%r)" % R2)
  ls2 = gf.LevelSet(m, 2, "y+%r*tanh(x/7)" % R1, "x*x+y*y-sqr(%r)" % R2)
  mls = gf.MeshLevelSet(m)
  mls.add(ls1)
  mls.add(ls2)
  mls.adapt()

  mf_pre = gf.MeshFem(m)
  mf_pre.set_finite_element(gf.Fem("FEM_QK(2,2)"))
  mfls = gf.MeshFem("levelset", mls, mf_pre)
  # the singular functions near the tips, on the Q1 partition of unity of
  # the nodes within one element of them
  pu_mf = gf.MeshFem(m)
  pu_mf.set_classical_finite_element(1)
  pts = pu_mf.basic_dof_nodes()
  xtip, ytip = np.sqrt(R2*R2 - R1*R1), R1
  def near(*tips):
    """the nodes within h of one of the tips"""
    d = np.min([np.hypot(pts[0] - x, pts[1] - y) for x, y in tips], axis=0)
    return np.nonzero(d < h)[0]
  ck = [gf.GlobalFunction("crack", i) for i in range(4)]
  mf_sing = [gf.MeshFem("product", pu_mf, gf.MeshFem("global function", m, ls, ck), dofs)
             for ls, dofs in [(ls1, near([xtip, ytip], [-xtip, -ytip])),
                              (ls2, near([xtip, -ytip], [-xtip, ytip]))]]
  mf_u = gf.MeshFem("sum", mf_sing[0], mf_sing[1], mfls)
  mf_u.set_qdim(2)
  mf_theta = gf.MeshFem("sum", mf_sing[0], mf_sing[1], mfls)
  # GetFEM's rule on every sub-triangle (tau 0)
  mim = gf.MeshIm("levelset", mls, "all",
                  gf.Integ("IM_STRUCTURED_COMPOSITE(IM_TRIANGLE(6),5)"),
                  gf.Integ("IM_STRUCTURED_COMPOSITE(IM_GAUSS_PARALLELEPIPED(2,6),9)"),
                  gf.Integ("IM_GAUSS_PARALLELEPIPED(2,4)"), tau=0.)
  mim_bound = [gf.MeshIm("levelset", mls, b, gf.Integ("IM_STRUCTURED_COMPOSITE(IM_TRIANGLE(6),3)"),
                         tau=0.) for b in ["boundary(a)", "boundary(b)"]]

  md = gf.Model()
  md.add_fem_variable("u", mf_u)
  md.add_fem_variable("theta", mf_theta)
  md.add_variable("P", 1)
  md.add_variable("T", 1)
  md.set_variable("T", [273 + room_temp])
  surf_crack = sum(gf.asm_generic(mb, 0, "1", -1, md) for mb in mim_bound)
  for k, v in [("surf_crack", surf_crack), ("mu0", E0/(2*(1 + nu))),
               ("la0", E0*nu/((1 - 2*nu)*(1 + nu))), ("ka0", E0/(3*(1 - 2*nu))),
               ("alpha", alpha), ("alpha_env", alpha_env), ("alpha_crack", alpha_crack),
               ("beta", beta), ("gamma", gamma), ("mR", mR), ("theta0", theta0)]:
    md.add_initialized_data(k, v)
  # Lame coefficients and bulk modulus, weakened by the temperature
  for p1, p2 in [("la", "la0"), ("mu", "mu0"), ("ka", "ka0")]:
    md.add_macro(p1, "%s/(1+gamma*(theta-theta0))" % p2)
  # deformation gradients, in the block and on both lips
  md.add_macro("F", "Id(2)+Grad_u")
  md.add_macro("Fm", "Id(2)+Xfem_minus(Grad_u)")
  md.add_macro("Fp", "Id(2)+Xfem_plus(Grad_u)")
  md.add_macro("Nanson(FF)", "Det(FF)*Norm(Inv(FF')*Normal)")
  md.add_macro("th", "exp(beta*(theta-theta0))")
  # Green-Lagrange strain, second Piola-Kirchhoff stress (elastic and
  # thermal expansion parts)
  md.add_macro("E", "(Grad_u+Grad_u'+Grad_u'*Grad_u)*0.5")
  md.add_macro("S", "((la*Trace(E)*Id(2)+2*mu*E)/th-1.5*(th-1/th)*ka*Id(2))")
  # elasticity, heat diffusion in the deformed body
  md.add_nonlinear_term(mim, "(F*S):Grad_Test_u")
  md.add_nonlinear_term(mim, "alpha*Det(F)*(Inv(F')*Grad_theta).(Inv(F')*Grad_Test_theta)")
  for mb in mim_bound:
    # ideal gas law
    md.add_nonlinear_term(mb, "(P*((Xfem_plus(u)-Xfem_minus(u)).Normalized(Inv(Fm')*Normal)+1e-6)"
                              "-mR*T/surf_crack)*Test_P")
    # heat flux equilibrium of the gas
    md.add_nonlinear_term(mb, "(Nanson(Fm)*(T-273-Xfem_minus(theta))"
                              "+Nanson(Fp)*(T-273-Xfem_plus(theta)))*Test_T")
    # heat exchange of the lips with the gas
    md.add_nonlinear_term(mb, "-alpha_crack*(Nanson(Fm)*(T-273-Xfem_minus(theta))*Xfem_minus(Test_theta)"
                              "+Nanson(Fp)*(T-273-Xfem_plus(theta))*Xfem_plus(Test_theta))")
  # heat exchange at the bottom and the top
  for var, rg in [("theta_B", BOTTOM), ("theta_T", TOP)]:
    md.add_initialized_data(var, room_temp)
    md.add_nonlinear_term(mim, "-alpha_env*Nanson(F)*(%s-theta)*Test_theta" % var, rg)
  # follower pressure on the lips
  for mb in mim_bound:
    md.add_nonlinear_term(mb, "(P*Det(Fm)*Inv(Fm')*Normal).Xfem_minus(Test_u)")
    md.add_nonlinear_term(mb, "(-P*Det(Fp)*Inv(Fp')*Normal).Xfem_plus(Test_u)")
  # clamped left and right
  md.add_Dirichlet_condition_with_multiplier(mim, "u", 1, LEFT)
  md.add_Dirichlet_condition_with_multiplier(mim, "u", 1, RIGHT)

  # open the lips with the pressure fixed
  md.disable_variable("P")
  md.set_variable("P", [1e-2])
  nit, conv, info = md.solve(atol=5e-4, max_iter=100)
  assert conv, info["message"]
  md.enable_variable("P")

  # heat the top face
  for k, fact in enumerate([0., 0.2, 0.4, 0.6, 0.8, 1.][:nsteps]):
    md.set_variable("theta_T", [room_temp + fact*jump_temp])
    nit, conv, info = md.solve(atol=5e-8, max_iter=100)
    assert conv, "step %i: %s" % (k, info["message"])
    yield k, dict(t=md.variable("T")[0] - 273, p=md.variable("P")[0],
                  norm_u=np.sqrt(gf.asm_generic(mim, 0, "Norm_sqr(u)", -1, md)),
                  norm_theta=np.sqrt(gf.asm_generic(mim, 0, "Norm_sqr(theta)", -1, md)),
                  mf_u=mf_u, u=md.variable("u"), mf_theta=mf_theta,
                  theta=md.variable("theta"), mls=mls)


if __name__ == "__main__":
  ny = int(sys.argv[1]) if len(sys.argv) > 1 else 10
  nsteps = int(sys.argv[2]) if len(sys.argv) > 2 else 6
  for k, s in simulate(ny, nsteps):
    print("step %i: gas temperature %.4f C, pressure %.6f MPa; |u| %.6e, |theta| %.6e"
          % (k, s["t"], s["p"], s["norm_u"], s["norm_theta"]))
    # the solution on the cut mesh, discontinuous across the cracks
    # (alpha 0.001: the nodes just off the cracks, each side its value)
    cut = s["mls"].cut_mesh()
    mfv = gf.MeshFem(cut, 2)
    mfv.set_classical_discontinuous_finite_element(2, 0.001)
    mfw = gf.MeshFem(cut, 1)
    mfw.set_classical_discontinuous_finite_element(2, 0.001)
    mfv.export_to_vtu("gas_filled_crack_%i.vtu" % k,
                      gf.interpolation(s["mf_u"], s["u"], mfv), "u",
                      mfw, gf.interpolation(s["mf_theta"], s["theta"], mfw), "temperature")
  print("wrote gas_filled_crack_<step>.vtu")
