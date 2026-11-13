#!/usr/bin/env python3
"""The gas-filled crack pair in 3D (the 2D example gas_filled_crack
extruded), with GetFEM-rs, as the Rust example gas_filled_crack_3d: a 40
x 10 x 20 mm block, clamped left and right, with two curved cracks,
y = 2.5 tanh(x/7) (extruded along z) and y = -2.5 tanh((x - 0.3 z)/7) +
0.15 z (oblique), inside the ellipsoid (x/16)^2 + (y/6)^2 + (z/8)^2 < 1:
each crack is inside the block, its front a closed curve. The gas in the
cracks (ideal gas, mass mR, temperature T, pressure P) pushes their lips
apart and exchanges heat with them; the block exchanges heat with its top
and bottom faces. The displacement and the temperature are Q2 on
hexahedra, enriched by the Heaviside functions of the cracks and, on the
elements the fronts pass through, by the crack tip functions of the plane
normal to the front (functions of the level sets); the DOFs without
support dropped. The crack is first opened with the pressure fixed, then
the top face is heated in five steps from 30 C to 130 C. Writes the
displacement and temperature on the cut mesh of each step to
gas_filled_crack_3d_<step>.vtu. Optional arguments: element size [mm]
(default 2), the temperature steps (default 6)."""
import sys

import numpy as np

import getfem_rs as gf

E0 = 5e2            # [N/mm^2] Young modulus at the reference temperature
NU = 0.45           # Poisson ratio
BETA = 8e-5         # [1/K] expansion coefficient
GAMMA = 0.1         # [1/K] thermal weakening
THETA0 = 30.        # [C] reference temperature
ALPHA = 0.1         # [mm^2/s] thermal diffusivity
ALPHA_ENV = 0.02    # [N/mm/s/K] heat transfer to the environment
ALPHA_CRACK = 0.01  # [N/mm/s/K] heat transfer to the gas
MR = 287e3*5e-9     # [N mm/K] gas constant times mass
ROOM_TEMP = 30.     # [C]
JUMP_TEMP = 100.    # [C] final temperature difference
L, H, W = 40., 10., 20.  # [mm]
R1, TILT, SHEAR = 2.5, 0.15, 0.3
AX, AY, AZ = 16., 6., 8.  # semi-axes of the ellipsoid bounding the cracks
LEFT, RIGHT, BOTTOM, TOP = 101, 102, 103, 104


def crack(m, sign):
  """the crack surface (primary): extruded for sign > 0, oblique
  otherwise; the ellipsoid (secondary), scaled to a distance near the
  front (the crack tip functions take the level sets as coordinates
  around the front)"""
  shear, tilt = (0., 0.) if sign > 0 else (SHEAR, TILT)
  return gf.LevelSet(m, 2, "y-(%r)*tanh((x-%r*z)/7)-%r*z" % (sign*R1, shear, tilt),
                     "(sqrt(sqr(x/%r)+sqr(y/%r)+sqr(z/%r))-1)*%r" % (AX, AY, AZ, AX))


def simulate(h, nsteps=6):
  """yields (step, state) after each temperature step"""
  m = gf.Mesh("cartesian Q1", np.linspace(-L/2, L/2, round(L/h) + 1),
              np.linspace(-H/2, H/2, round(H/h) + 1), np.linspace(-W/2, W/2, round(W/h) + 1))
  for rg, nn in [(LEFT, [-1., 0., 0.]), (RIGHT, [1., 0., 0.]), (BOTTOM, [0., -1., 0.]),
                 (TOP, [0., 1., 0.])]:
    m.set_region(rg, m.outer_faces_with_direction(nn, 0.5))
  lss = [crack(m, 1.), crack(m, -1.)]
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
  base_u = gf.MeshFem("sum", mf_sing[0], mf_sing[1], mfls)
  base_theta = gf.MeshFem("sum", mf_sing[0], mf_sing[1], mfls)
  mim = gf.MeshIm("levelset", mls, "all", gf.Integ("IM_TETRAHEDRON(5)"),
                  gf.Integ("IM_PRODUCT(IM_TRIANGLE(3),IM_GAUSS1D(3))"),
                  gf.Integ("IM_GAUSS_PARALLELEPIPED(3,4)"))
  # the DOFs with support (enriched DOFs without are singular)
  kept = base_u.dofs_with_support(mim, 1e-8)
  mf_theta = gf.MeshFem("partial", base_theta, kept)
  base_u.set_qdim(3)
  mf_u = gf.MeshFem("partial", base_u, [3*i + c for i in kept for c in range(3)])
  mim_bound = [gf.MeshIm("levelset", mls, "boundary(%s)" % c, gf.Integ("IM_TETRAHEDRON(3)"))
               for c in "ab"]

  md = gf.Model()
  md.add_fem_variable("u", mf_u)
  md.add_fem_variable("theta", mf_theta)
  md.add_variable("P", 1)
  md.add_variable("T", 1)
  md.set_variable("T", [273 + ROOM_TEMP])
  surf_crack = sum(gf.asm_generic(mb, 0, "1", -1) for mb in mim_bound)
  for k, v in [("surf_crack", surf_crack), ("mu0", E0/(2*(1 + NU))),
               ("la0", E0*NU/((1 - 2*NU)*(1 + NU))), ("ka0", E0/(3*(1 - 2*NU))),
               ("alpha", ALPHA), ("alpha_env", ALPHA_ENV), ("alpha_crack", ALPHA_CRACK),
               ("beta", BETA), ("gamma", GAMMA), ("mR", MR), ("theta0", THETA0)]:
    md.add_initialized_data(k, v)
  # Lame coefficients and bulk modulus, weakened by the temperature
  for p1, p2 in [("la", "la0"), ("mu", "mu0"), ("ka", "ka0")]:
    md.add_macro(p1, "%s/(1+gamma*(theta-theta0))" % p2)
  # deformation gradients, in the block and on both lips
  md.add_macro("F", "Id(3)+Grad_u")
  md.add_macro("Fm", "Id(3)+Xfem_minus(Grad_u)")
  md.add_macro("Fp", "Id(3)+Xfem_plus(Grad_u)")
  md.add_macro("Nanson(FF)", "Det(FF)*Norm(Inv(FF')*Normal)")
  md.add_macro("th", "exp(beta*(theta-theta0))")
  # Green-Lagrange strain, second Piola-Kirchhoff stress (elastic and
  # thermal expansion parts)
  md.add_macro("E", "(Grad_u+Grad_u'+Grad_u'*Grad_u)*0.5")
  md.add_macro("S", "((la*Trace(E)*Id(3)+2*mu*E)/th-1.5*(th-1/th)*ka*Id(3))")
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
    md.add_initialized_data(var, ROOM_TEMP)
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
    md.set_variable("theta_T", [ROOM_TEMP + fact*JUMP_TEMP])
    nit, conv, info = md.solve(atol=5e-8, max_iter=100)
    assert conv, "step %i: %s" % (k, info["message"])
    yield k, dict(t=md.variable("T")[0] - 273, p=md.variable("P")[0],
                  norm_u=np.sqrt(gf.asm_generic(mim, 0, "Norm_sqr(u)", -1, md)),
                  norm_theta=np.sqrt(gf.asm_generic(mim, 0, "Norm_sqr(theta)", -1, md)),
                  mf_u=mf_u, u=md.variable("u"), mf_theta=mf_theta,
                  theta=md.variable("theta"), mls=mls)


if __name__ == "__main__":
  h = float(sys.argv[1]) if len(sys.argv) > 1 else 2.
  nsteps = int(sys.argv[2]) if len(sys.argv) > 2 else 6
  for k, s in simulate(h, nsteps):
    print("step %i: gas temperature %.4f C, pressure %.6f MPa" % (k, s["t"], s["p"]))
    # the solution on the cut mesh, discontinuous across the cracks
    # (alpha 0.001: the nodes just off the cracks, each side its value)
    cut = s["mls"].cut_mesh()
    mfv = gf.MeshFem(cut, 3)
    mfv.set_classical_discontinuous_finite_element(2, 0.001)
    mfw = gf.MeshFem(cut, 1)
    mfw.set_classical_discontinuous_finite_element(2, 0.001)
    mfv.export_to_vtu("gas_filled_crack_3d_%i.vtu" % k,
                      gf.interpolation(s["mf_u"], s["u"], mfv), "u",
                      mfw, gf.interpolation(s["mf_theta"], s["theta"], mfw), "temperature")
  print("wrote gas_filled_crack_3d_<step>.vtu")
