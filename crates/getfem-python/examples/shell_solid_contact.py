#!/usr/bin/env python3
"""Shell to solid contact, with GetFEM-rs: two hemispheres of midsurface
radius R = 10 and thickness T = 0.25, neo-Hookean (E, nu), pressed
together off their poles, half models (symmetry y = 0):
- solid: dome up, its equator clamped in x and z; 27-node hexahedra
  (Hexa27), one through the thickness, mixed u-p;
- shell: dome down above it, shifted sideways by sx, its equator held in
  x and pushed down by delta; 9-node quadrilaterals (Quad9) on the
  midsurface, a 7-parameter shell (displacement u, director change d,
  thickness stretch gradient w), with Normal and element_curvature.

The shell at xi across its thickness, K = element_curvature,
integrated through the thickness by 3-point Gauss:
  x(xi) = X + u + xi (1 + xi w) (Normal + d),
  F(xi) = (I + Grad_u + xi (1 + xi w) (K + Grad_d) + xi^2 (Normal + d)@Grad_w
           + (d + 2 xi w (Normal + d))@Normal) (I + xi K)^-1.
Contact: the shell (slave) on the solid's outer surface (master), rays
along the deformed shell normal nx, the gap of the shell's surface facing
the solid
  gap = (Interpolate(X,ray) + Interpolate(ub,ray) - X - us - T/2 (1 + T/2 w) (Normal + d)).nx,
the contact pressure lc (Q1, one order below the displacements) by the
augmented Lagrangian lc = neg_part(r gap - lc), as a weak form; the
augmentation r low, of the order of the shells' own stiffness E T/R^2.

Arguments: shell and solid elements per 45 deg (8, 3), final delta (6),
steps (120). Writes both bodies per step as shell_<k>.vtu, solid_<k>.vtu
with shell.pvd, solid.pvd (time: delta) and prints the Newton iterations and the
contact force."""
import sys
import numpy as np

import getfem_rs as gf

T, R, gap0, sx = 0.25, 10., 0.02, 6.
nc = int(sys.argv[1]) if len(sys.argv) > 1 else 8
ncb = int(sys.argv[2]) if len(sys.argv) > 2 else 3
delta_max = float(sys.argv[3]) if len(sys.argv) > 3 else 6.
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 120
E, nu = 6.825e7, 0.3
YM, ZM, EXT, SHELL = 4, 5, 6, 7
# centre of the shell sphere: the surfaces gap0 apart along the line of
# the centres
zc = np.sqrt((2*R + T + gap0)**2 - sx**2)

# solid: dome up, radii R -/+ T/2, the half y >= 0
mb = gf.Mesh("structured_ball_shell", "GT='GT_QK(3,2)';ORG=[0,0,0];SIZES=[%r,%r];NSUBDIV=[%i,1];"
             "SYMMETRIES=2" % (R + T/2, T, ncb))
mb.set_region(YM, mb.outer_faces_in_box([-2*R, -1e-5, -1e-5], [2*R, 1e-5, 2*R]))
mb.set_region(ZM, mb.outer_faces_in_box([-2*R, -1e-5, -1e-5], [2*R, 2*R, 1e-5]))
mb.set_region(EXT, mb.outer_faces_with_direction([0., 0., 1.], 0.49*np.pi))   # upper outer faces

# shell: dome down above the solid, the half y >= 0 rotated by (x,y,z) ->
# (-x,y,-z) (a rotation: Normal stays outwards, down, towards the solid)
ms = gf.Mesh("structured_ball_shell", "GT='GT_QK(2,2)';ORG=[0,0,0];SIZES=[%r];NSUBDIV=[%i];SYMMETRIES=2"
             % (R, nc))
ms.transform([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
ms.translate([sx, 0, zc])
ms.set_region(YM, ms.outer_faces_in_box([sx - 2*R, -1e-5, zc - 2*R], [sx + 2*R, 1e-5, zc + 1e-5]))
ms.set_region(ZM, ms.outer_faces_in_box([sx - 2*R, -1e-5, zc - 1e-5], [sx + 2*R, 2*R, zc + 1e-5]))
ms.set_region(SHELL, list(range(ms.nb_convexes())))

mfb3 = gf.MeshFem(mb, 3)
mfb3.set_classical_finite_element(2)
mfub = gf.MeshFem("partial", mfb3, sorted(set(range(mfb3.nb_dof()))
                                          - set(mfb3.basic_dof_on_region(YM)[1::3])    # y on y = 0
                                          - set(mfb3.basic_dof_on_region(ZM)[0::3])    # x and z on
                                          - set(mfb3.basic_dof_on_region(ZM)[2::3])))  # the equator
mfp = gf.MeshFem(mb, 1)
mfp.set_classical_finite_element(1)
mfs3 = gf.MeshFem(ms, 3)
mfs3.set_classical_finite_element(2)
mfus = gf.MeshFem("partial", mfs3, sorted(set(range(mfs3.nb_dof()))
                                          - set(mfs3.basic_dof_on_region(YM)[1::3])))  # y on y = 0
mfw = gf.MeshFem(ms, 1)
mfw.set_classical_finite_element(2)
mimb = gf.MeshIm(mb, 5)
mims = gf.MeshIm(ms, 5)

md = gf.Model()
md.add_initialized_data("K", E/(3*(1 - 2*nu)))
md.add_initialized_data("G", E/(2*(1 + nu)))
md.add_initialized_data("T", T)
md.add_initialized_data("r", E*T/R**2)   # augmentation, low for thin shells
md.add_initialized_data("delta", 0.)
# solid
md.add_fem_variable("ub", mfub)
md.add_fem_variable("p", mfp)
md.add_macro("Fb", "Id(3)+Grad_ub")
md.add_macro("tau_b", "p*Id(3)+G*pow(Det(Fb),-2/3)*Deviator(Fb*Fb')")
md.add_nonlinear_term(mimb, "(tau_b*Inv(Fb)'):Grad_Test_ub")
md.add_nonlinear_term(mimb, "(p-K*log(Det(Fb)))*Test_p")
# shell
md.add_fem_variable("us", mfus)
md.add_fem_variable("d", mfus)
md.add_fem_variable("w", mfw)
md.add_macro("F0", "Id(3)+Grad_us+d@Normal")
md.add_macro("W0", "(K/2*sqr(log(Det(F0)))+G/2*(pow(Det(F0),-2/3)*Norm_sqr(F0)-3))")
md.add_macro("A(xx)", "(Id(3)+xx*element_curvature)")
md.add_macro("F(xx)", "(Id(3)+Grad_us+xx*(1+xx*w)*(element_curvature+Grad_d)"
                                   "+xx*xx*(Normal+d)@Grad_w"
                                   "+(d+2*xx*w*(Normal+d))@Normal)*Inv(A(xx))")
md.add_macro("J(xx)", "Det(F(xx))")
md.add_macro("W(xx)", "Det(A(xx))*(K/2*sqr(log(J(xx)))+G/2*(pow(J(xx),-2/3)*Norm_sqr(F(xx))-3))")
md.add_macro("tau(xx)", "K*log(J(xx))*Id(3)+G*pow(J(xx),-2/3)*Deviator(F(xx)*F(xx)')")
md.add_initialized_data("c0", T/2*8/9)
md.add_initialized_data("c", T/2*5/9)
md.add_initialized_data("xi", T/2*np.sqrt(3/5))
md.add_nonlinear_term(mims, "c0*W0+c*(W(-xi)+W(xi))")
# the shell's equator held in x and pushed down: us(1) = 0, us(3) = -delta
mfl = gf.MeshFem(ms, 2)
mfl.set_classical_finite_element(2)
md.add_filtered_fem_variable("lam", mfl, ZM)
md.add_nonlinear_term(mims, "lam(1)*us(1)+lam(2)*(us(3)+delta)", ZM)
# contact: rays from the shell along its deformed normal
md.add_raytracing_transformation("ray", 0.5)
md.add_master_contact_boundary_to_raytracing_transformation("ray", mb, "ub", EXT)
md.add_slave_contact_boundary_to_raytracing_transformation("ray", ms, "us", SHELL)
md.add_macro("nx", "Normalized(Inv(Id(3)+Grad_us)'*Normal)")
md.add_macro("gap", "(Interpolate(X,ray)+Interpolate(ub,ray)-X-us-T/2*(1+T/2*w)*(Normal+d)).nx")
md.add_macro("dgap", "(Interpolate(Test_ub,ray)-Test_us-T/2*(1+T/2*w)*Test_d-T/2*T/2*Test_w*(Normal+d)).nx")
mflc = gf.MeshFem(ms, 1)
mflc.set_classical_finite_element(1)
md.add_fem_variable("lc", mflc)
md.add_nonlinear_term(mims, "-lc/r*Test_lc"
                            "+Interpolate_filter(ray, neg_part(r*gap-lc)*(Test_lc/r-dgap), 1)", SHELL)
print("%i hexahedra, %i shell elements, %i dofs" % (mb.nb_convexes(), ms.nb_convexes(), md.nb_dof()))

mfob = gf.MeshFem(mb, 1)
mfob.set_classical_discontinuous_finite_element(2)
mfos = gf.MeshFem(ms, 1)
mfos.set_classical_discontinuous_finite_element(2)
names = ["ub", "p", "us", "d", "w", "lam", "lc"]
t, dt, deltas = 0., 1/steps, []
print("  delta  Newton  contact force")
while t < 1 - 1e-12:
  dt = min(dt, 1 - t)
  saved = [md.variable(v) for v in names]
  md.set_variable("delta", [(t + dt)*delta_max])
  nit, conv, info = md.solve(max_iter=20)
  if not conv:   # back, half the step
    for v, x in zip(names, saved):
      md.set_variable(v, x)
    dt /= 2
    assert dt > 1e-4/steps, info["message"]
    continue
  t += dt
  dt = min(2*dt, 1/steps)   # back to the nominal step
  print("%7.4f  %6i  %13.5e" % (t*delta_max, nit, 2*gf.asm_generic(mims, 0, "lc", SHELL, md)), flush=True)
  deltas.append(t*delta_max)
  step = len(deltas)
  mfub.export_to_vtu("solid_%i.vtu" % step, md.variable("ub"), "u",
                     mfob, md.interpolation("sqrt(3/2)*Norm(Deviator(tau_b))/Det(Fb)", mfob),
                     "von Mises")
  mfus.export_to_vtu("shell_%i.vtu" % step, md.variable("us"), "u", mfus, md.variable("d"), "d",
                     mflc, md.variable("lc"), "contact pressure",
                     mfos, md.interpolation("sqrt(3/2)*Norm(Deviator(tau(0)))/J(0)", mfos),
                     "von Mises (midsurface)")
for name in ["shell", "solid"]:
  with open(name + ".pvd", "w") as f:
    f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n')
    for s, delta in enumerate(deltas, 1):
      f.write('<DataSet timestep="%g" file="%s_%i.vtu"/>\n' % (delta, name, s))
    f.write("</Collection>\n</VTKFile>\n")
print("written: shell.pvd, solid.pvd")
