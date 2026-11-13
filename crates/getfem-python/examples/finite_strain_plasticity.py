#!/usr/bin/env python3
"""Necking of a plane strain block, with GetFEM-rs: finite strain
plasticity of Eterovic-Bathe type with linear and exponential
hardening/softening (GetFEM's finite_strain_plasticity_ksi.py), the plastic
multiplier increment ksi an internal variable, condensed. A quarter of the
block, pulled along x by a normal displacement on the right face (a
multiplier), the displacement 5 lf, lf the load factor from 0 to 1. In
load steps (u extrapolated from the two last steps), or by Moore-Penrose
continuation in lf (the script's numerical_continuation settings and its
non-smooth option) past the force peak, to lf = 1. Writes the force per step and the final state to
finite_strain_plasticity_2D_forces.dat and finite_strain_plasticity_2D.vtu.
Optional arguments: the number of steps (default 200), "continuation"."""
import math
import sys

import numpy as np

import getfem_rs as gf

L, H = 2*26.667, 2*6.413
LEFT, RIGHT, BOTTOM = 1, 2, 3

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
continuation = len(sys.argv) > 2
if continuation and sys.argv[2] != "continuation":
  sys.exit("%s: continuation or nothing" % sys.argv[2])
e, nu = 206.9e3, 0.29
s0, sinf, hp, delta = 0.45e3, 0.715e3, -0.012924e3, 16.93

# quarter of the block, graded along x, thinner at the center
m = gf.Mesh("structured", "GT='GT_QK(2,2)';SIZES=[1,1];NSUBDIV=[20,10]")
m.transform(np.array([[L/2, 0.], [0., H/2]]))
# the outer faces of n.(sgn e_c) > 0.95
angle = np.arctan2(np.sqrt(1 - 0.95**2), 0.95)
for rg, n in [(LEFT, [-1., 0.]), (RIGHT, [1., 0.]), (BOTTOM, [0., -1.])]:
  m.set_region(rg, m.outer_faces_with_direction(n, angle))
dy, y0, dh = 0.2, 0.04, 0.018*H
x0 = y0/dy
b2 = (1 - dy)/(x0 - 1)**2
b1 = dy - 2*b2*x0
b0 = 1 - b1 - b2
x, y = m.pts()
t = 2*np.abs(x)/L
x = np.where(t < x0, x*dy, (b0 + b1*t + b2*t*t)*np.sign(x)*L/2)
m.set_pts([x, y - (y*dh)/(2*H)*(1 + np.cos(2*np.pi*x/L))])

# displacements without the normal components on the symmetry faces
pf = gf.Fem("FEM_Q2_INCOMPLETE(2)")
mf_n = gf.MeshFem(m, 2)
mf_n.set_finite_element(pf)
removed = np.concatenate([[d for d in mf_n.basic_dof_on_region(rg) if d % 2 == c]
                          for rg, c in [(LEFT, 0), (BOTTOM, 1)]])
mfu = gf.MeshFem("partial", mf_n, np.setdiff1d(np.arange(mf_n.nb_dof()), removed))
mfdir = gf.MeshFem(m, 1)
mfdir.set_finite_element(pf)
mfout = gf.MeshFem(m, 1)
mfout.set_classical_discontinuous_finite_element(2)
mim = gf.MeshIm(m, 5)
mim_vol = gf.MeshIm(m, 3)
mimd1 = gf.MeshImData(mim)
mimd4 = gf.MeshImData(mim, -1, 4)
npts = mimd1.nb_points()

md = gf.Model()
md.add_fem_variable("u", mfu)
md.add_fem_data("u_prev", mfu)
md.add_internal_im_variable("ksi", mimd1)
# accumulated plastic strain and components 11, 22, 33, 12 of Cp^-1 at the
# previous step
md.add_im_data("gamma0", mimd1)
md.add_im_data("invCp0vec", mimd4)
md.set_variable("invCp0vec", np.tile([1., 1., 1., 0.], npts))
md.add_initialized_data("lf", 0.)
md.add_initialized_data("K", e/(3*(1 - 2*nu)))
md.add_initialized_data("G", e/(2*(1 + nu)))
c = math.sqrt(2/3)
for k, v in [
    ("Mat3x3ToVec4", "[[[1,0,0,0],[0,0,0,0.5],[0,0,0,0]],[[0,0,0,0.5],[0,1,0,0],[0,0,0,0]],"
                     "[[0,0,0,0],[0,0,0,0],[0,0,1,0]]]"),
    ("Vec4ToMat3x3", "[[[1,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,1,0],[0,0,0]],"
                     "[[0,0,0],[0,0,0],[0,0,1]],[[0,1,0],[1,0,0],[0,0,0]]]"),
    ("F", "Id(2)+Grad(u)"),
    ("F3d", "Id(3)+[1,0;0,1;0,0]*Grad(u)*[1,0,0;0,1,0]"),
    ("J", "Det(F)"),
    ("invCp0", "Vec4ToMat3x3.invCp0vec"),
    ("devlogbetr", "Deviator(Logm(F3d*invCp0*F3d'))"),
    ("devlogbe", "(1-2*ksi)*devlogbetr"),
    ("tauD", "G*devlogbe"),
    ("tauD2d", "G*([1,0,0;0,1,0]*devlogbe*[1,0;0,1;0,0])"),
    ("tauH", "K*log(J)"),
    ("tau", "tauH*Id(3)+tauD"),
    ("Y(gamma)", "%r+%r*(1-exp(-%r*gamma))+%r*gamma" % (c*s0, c*(sinf - s0), c*delta, 2/3*hp)),
    ("gamma", "gamma0+ksi*Norm(devlogbetr)"),
    ("tauDtr", "G*devlogbetr"),
    ("invCp", "(Inv(F3d)*Expm(-ksi*devlogbetr)*F3d)*invCp0*(Inv(F3d)*Expm(-ksi*devlogbetr)*F3d)'")]:
  md.add_macro(k, v)
md.add_nonlinear_term(mim, "(tauD2d*Inv(F)'):Grad(Test_u)")
md.add_nonlinear_term(mim_vol, "tauH*Inv(F)':Grad(Test_u)")
# yield condition; +1e-12 ksi: the derivative in ksi does not vanish at u = 0
md.add_nonlinear_term(mim, "(Norm(tauD)-min(Y(gamma),Norm(tauDtr)+1e-12*ksi))*Test_ksi")
md.add_filtered_fem_variable("dirmult", mfdir, RIGHT)
md.add_linear_term(mim, "(5*lf-u(1))*dirmult", RIGHT)
print("%i displacement DOFs, %i model DOFs" % (mfu.nb_dof(), md.nb_dof()))

# the continuation in lf
variables = [("u", 1.)]
options = dict(h_init=1e-1, h_max=1., h_min=1e-5, h_dec=1/np.sqrt(10),
               h_inc=np.sqrt(10), max_iter=12, thr_iter=6, max_res=1e-8, max_diff=1e-8,
               min_cos=0.995, non_smooth=True, gamma_max=1.)
T, h = None, 0.
if continuation:
  T, h, info = md.numerical_continuation_step("lf", variables, **options)
# gamma at the center: the dof of the one element there
icenter = np.argmin(np.linalg.norm(mfout.basic_dof_nodes(), axis=0))
peak = (0., 0, 0.)
with open("finite_strain_plasticity_2D_forces.dat", "w") as f:
  for step in range(1, steps + 1):
    iterations = 0
    if continuation:
      T, h, info = md.numerical_continuation_step("lf", variables, T, h, **options)
      assert info["status"] == "converged", "step %i: the continuation failed" % step
    else:
      # x stretching 10/L x, at x = L/2
      md.set_variable("lf", [step/steps])
      # start: u extrapolated from the two last steps
      md.set_variables([("u", "2*u-u_prev"), ("u_prev", "u")] if step > 1 else [("u_prev", "u")])
      # plasticity is non-smooth: NLEQ-RES after full Newton, no NLEQ-ERR
      iterations, conv, info = md.solve(methods=["newton", "nleq_res"], atol=1e-8, rtol=1e-10)
      assert conv, "step %i: %s" % (step, info["message"])
    lf = md.variable("lf")[0]
    fr = gf.asm_generic(mim, 0, "dirmult", RIGHT, md)
    if fr > peak[0]:
      peak = (fr, step, lf)
    # the script's update: the history with 0.99 ksi, the rest the start of
    # the next step
    cc = 0.99
    md.set_variable("ksi", "%r*ksi" % cc)
    md.set_variables([("gamma0", "gamma"), ("invCp0vec", "Mat3x3ToVec4:invCp")])
    md.set_variable("ksi", "%r*ksi" % (1/cc - 1))
    gc = md.local_projection(mim, "gamma0", mfout)[icenter]
    f.write("step=%i loadfactor=%.17g fR=(%.17g,0) sR=(%.17g,0) gamma_center=%.17g iterations=%i h=%.17g\n"
            % (step, lf, fr, fr/H, gc, iterations, h))
    if step % 10 == 0 or step == steps or lf == 1:
      print("step %i: load factor %.4f, force %.3f, gamma at the center %.4f" % (step, lf, fr, gc))
    if lf == 1:
      break
print("force peak %.3f at step %i (load factor %.4f)" % peak)

fields = []
for expr, name in [("sqrt(1.5)*Norm(tauD)/J", "Von Mises stress"), ("gamma0", "gamma"),
                   ("tau(1,1)/J", "Cauchy stress 11")]:
  fields += [mfout, md.local_projection(mim, expr, mfout), name]
mfu.export_to_vtu("finite_strain_plasticity_2D.vtu", md.variable("u"), "u", *fields)
print("wrote finite_strain_plasticity_2D.vtu and _forces.dat")
