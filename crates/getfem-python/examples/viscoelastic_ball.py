#!/usr/bin/env python3
"""A viscoelastic disc falling on the ground, with GetFEM-rs. The disc
(centre (0, 20), radius 20, Q2), lifted by 0.5, falls under the load 0.1
onto the ground y = 0 (a penalty contact potential, r = 1000). Newmark
(beta = 1/4, gamma = 1/2) written as GWFL terms; plane strain
quasilinear viscoelasticity (Horgan-Murphy, gamma = 1/6, kappa = lambda +
2 mu/3 with lambda = mu = 200, Prony rH = rD = 0.8, tauH = tauD = 1), the
stresses of the previous step stored at the integration points. Prints
per step the kinetic energy, the lowest point, the mean vertical
displacement and the contact force; writes viscoelastic_ball_<step>.vtu
every 20 steps."""
import numpy as np

import getfem_rs as gf

lam = 200.; mu = 200.; kappa = lam + 2*mu/3.  # Lame, bulk moduli
load = 0.1                                    # body force (downwards)
r = 1000.                                     # contact penalty
tauH = tauD = 1.; rH = rD = 0.8               # Prony series
gam = 1./6.                                   # Horgan-Murphy
beta = 0.25; gamma = 0.5                      # Newmark
dt = 0.02; T = 8.; lift = 0.5
CONTACT = 1

m = gf.Mesh("structured_ball",
            "GT='GT_QK(2,2)';ORG=[0,20];SIZES=[20];NSUBDIV=[2,2];SYMMETRIES=0")
# the faces of n_y < -0.01
m.set_region(CONTACT, m.outer_faces_with_direction([0., -1.],
                                                   np.arctan2(np.sqrt(1-1e-4), 0.01)))
mf = gf.MeshFem(m, 2)
mf.set_classical_finite_element(2)
mim = gf.MeshIm(m, 4)
mimc = gf.MeshIm(m, gf.Integ("IM_STRUCTURED_COMPOSITE(IM_GAUSS_PARALLELEPIPED(2,4),4)"))
imd = gf.MeshImData(mim, -1, [3, 3])
nd = mf.nb_dof()

md = gf.Model()
md.add_fem_variable("u", mf)
U0 = np.zeros(nd); U0[1::2] = lift
A0 = np.zeros(nd); A0[1::2] = -load
md.add_initialized_fem_data("u0", mf, U0)
md.add_fem_data("v0", mf)
md.add_initialized_fem_data("a0", mf, A0)
md.set_variable("u", U0)
for k, v in [("kappa", kappa), ("mu", mu), ("gamma", gam), ("tauH", tauH),
             ("tauD", tauD), ("rH", rH), ("rD", rD), ("r", r), ("beta", beta),
             ("dt", dt), ("gamma_N", gamma)]:
  md.add_initialized_data(k, v)
md.add_initialized_data("load", [0., -load])
for k in ["SeHprev", "SeDprev", "SvHprev", "SvDprev"]:
  md.add_im_data(k, imd)
for k, v in [
    ("E", "[1,0;0,1;0,0]"),
    ("F", "E*(Id(2)+Grad_u)*E' + [0,0,0;0,0,0;0,0,1]"),
    ("Sqr(a)", "a*a"),
    ("WW1", "mu*(0.5+gamma)"),
    ("WW2", "mu*(0.5-gamma)"),
    ("WW3", "kappa*(1-1/Det(F)) - 2*WW2*pow(Det(F),-2/3) - WW1*pow(Det(F),-4/3)"),
    ("I1", "Norm_sqr(F)"),
    ("I2", "Sqr(Det(F))*Norm_sqr(Inv(F))"),
    ("I3", "Sqr(Det(F))"),
    ("SeH", "(1/3*WW1*I1 + 2/3*WW2*I2 + WW3*I3)*Inv(Right_Cauchy_Green(F))"),
    ("SeD", "WW1*Id(3) + 1/3*(WW2*I2-WW1*I1)*Inv(Right_Cauchy_Green(F))"
            " - WW2*I3*Sqr(Inv(Right_Cauchy_Green(F)))"),
    ("SvH", "exp(-dt/tauH)*SvHprev - (1-rH)*(tauH/dt-(dt+tauH)/dt*exp(-dt/tauH))*SeHprev"
            " - (1-rH)*((dt-tauH)/dt+tauH/dt*exp(-dt/tauH))*SeH"),
    ("SvD", "exp(-dt/tauD)*SvDprev - (1-rD)*(tauD/dt-(dt+tauD)/dt*exp(-dt/tauD))*SeDprev"
            " - (1-rD)*((dt-tauD)/dt+tauD/dt*exp(-dt/tauD))*SeD"),
    ("pred", "(u0 + dt*v0 + dt*dt*(0.5-beta)*a0)"),
    ("acc", "(u - pred)/(beta*dt*dt)")]:
  md.add_macro(k, v)
md.add_nonlinear_term(mim, "acc.Test_u"
                           " + (E'*(F*(SeH+SvH+SeD+SvD))*E):Grad_Test_u - load.Test_u")
md.add_nonlinear_term(mimc, "r*sqr_neg_part(X(2) + u(2))", CONTACT)

mfy = gf.MeshFem(m); mfy.set_classical_finite_element(2)  # y of the nodes
area = gf.asm_generic(mim, 0, "1", -1, md)
print("     t   kinetic       lowest point  mean u_y      contact force  Newton")
for k in range(int(round(T/dt)) + 1):
  if k > 0:
    nit, conv, info = md.solve(atol=1e-9, rtol=0., max_iter=100)
    # the stresses of this step stored, Newmark's update
    # all from the values of this step
    md.set_variables([("SeHprev", "SeH"), ("SeDprev", "SeD"),
                      ("SvHprev", "SvH"), ("SvDprev", "SvD"),
                      ("u0", "u"),
                      ("v0", "v0 + dt*((1-gamma_N)*a0 + gamma_N*acc)"),
                      ("a0", "acc")])
  u = md.variable("u")
  print("%6.2f  %.6e  %.6e  %.6e  %.6e  %s" % (
    k*dt, gf.asm_generic(mim, 0, "0.5*v0.v0", -1, md), md.interpolation("X(2)+u(2)", mfy).min(),
    gf.asm_generic(mim, 0, "u(2)", -1, md)/area,
    gf.asm_generic(mimc, 0, "2*r*neg_part(X(2) + u(2))", CONTACT, md),
    "%d (%s)" % (nit, info["method"]) if k > 0 else ""))
  if k % 20 == 0:
    mf.export_to_vtu("viscoelastic_ball_%i.vtu" % k, u, "Displacements")
