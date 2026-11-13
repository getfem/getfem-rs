#!/usr/bin/env python3
"""Two rubber rings (a GetFEM user's script), with GetFEM-rs: rings of
radius R = 50 and cross section radius r = 5, a structured disc
(GT_QK(2,2)) extruded into 26 layers of second order hexahedra around the
circumference, FEM_Q2_INCOMPLETE(3), a compressible neo-Hookean material
(E = 1e5, nu = 0.45):
  tau = K log(J) Id + G J^(-2/3) Deviator(F F'),  P = tau F^-T.
Each ring has two handles (the outer faces at x = -R and x = R), held by
multipliers q (u = p on the handle) and by elastic supports. Ring 1 is the
master, ring 2 the slave of a raytracing contact along the deformed normal
of ring 2, with the penalty c = 1e3:
  g = (Interpolate(X,raytrace) + Interpolate(u1,raytrace) - X - u2).n2.
Phases: ring 2 is pulled through ring 1 by both handles (30 steps), the
handle 2 of ring 1 and the handle 1 of ring 2 are released (8 steps, the
supports decreasing to 0), then ring 2 is pulled by its handle 2 alone
(16 steps; the script's later steps make Newton cycle between contact
states, in GetFEM too).

Writes both rings per step (u and the von Mises stress of tau) as
ring1_<step>.vtu, ring2_<step>.vtu with ring1.pvd, ring2.pvd and prints
the Newton iterations, the largest displacements and the contact force."""
import numpy as np

import getfem_rs as gf

EXTERIOR, HANDLE1, HANDLE2 = 1, 2, 3

rr, r = 50., 5.
nc, nr1, nr2 = 26, 2, 2
e, nu = 1e5, 0.45
c_penal = 1e-2*e

# ring 1: the extruded disc wrapped around the y axis, its ends merged
disc = gf.Mesh("structured_ball", "GT='GT_QK(2,2)';ORG=[0,0];SIZES=[%r];NSUBDIV=[%i,%i];"
               "SYMMETRIES=0" % (r, nr1, nr2))
m1 = gf.Mesh("prismatic", disc, nc, 2)
x, y, z = m1.pts()
m1.set_pts([(rr + x)*np.cos(2*np.pi*z), y, (rr + x)*np.sin(2*np.pi*z)])
m1.merge_points()
m1.set_region(EXTERIOR, m1.outer_faces())
for rid, x in [(HANDLE1, -rr), (HANDLE2, rr)]:
  m1.set_region(rid, m1.outer_faces_in_box([x - 1.1*r, -1.1*r, -0.3*rr],
                                           [x + 1.1*r, 1.1*r, 0.3*rr]))
m1.optimize_structure()
# ring 2: a copy, shifted to interlock
m2 = gf.Mesh("clone", m1)
m2.translate([-rr, 2.5*r, 0.])

pf = gf.Fem("FEM_Q2_INCOMPLETE(3)")
def space(m, q):
  mf = gf.MeshFem(m, q)
  mf.set_finite_element(pf)
  return mf
mf1, mf2 = space(m1, 3), space(m2, 3)
# von Mises stress (of tau) at the nodes, for the output
mfs1, mfs2 = space(m1, 1), space(m2, 1)
mim1, mim2 = gf.MeshIm(m1, 5), gf.MeshIm(m2, 5)
mimc2 = gf.MeshIm(m2, 7)

md = gf.Model()
md.add_fem_variable("u1", mf1)
md.add_fem_variable("u2", mf2)
md.add_initialized_data("K", e/(3*(1 - 2*nu)))
md.add_initialized_data("G", e/(2*(1 + nu)))
md.add_macro("F(u)", "Id(3)+Grad_u")
md.add_macro("tau(u)", "K*log(Det(F(u)))*Id(3)+G*pow(Det(F(u)),-2/3)*Deviator(F(u)*F(u)')")
md.add_nonlinear_term(mim1, "(tau(u1)*Inv(F(u1)')):Grad_Test_u1")
md.add_nonlinear_term(mim2, "(tau(u2)*Inv(F(u2)')):Grad_Test_u2")

# handles: multipliers q and elastic supports (against twisting)
for q in ["q11", "q12", "q21", "q22"]:
  md.add_variable(q, 3)
  md.add_initialized_data(q.replace("q", "p"), [0., 0., 0.])
md.add_initialized_data("penal", 0.)
for mim, u, h, q, support in [(mim1, "u1", HANDLE1, "11", "1e4"),
                              (mim1, "u1", HANDLE2, "12", "penal"),
                              (mim2, "u2", HANDLE1, "21", "penal"),
                              (mim2, "u2", HANDLE2, "22", "1e4")]:
  md.add_linear_term(mim, "(%s-p%s).q%s" % (u, q, q), h)
  md.add_linear_term(mim, "%s*(%s-p%s).Test_%s" % (support, u, q, u), h)

# contact: ring 2 (slave) on ring 1 (master)
md.add_initialized_data("c", c_penal)
md.add_raytracing_transformation("raytrace", r)
md.add_master_contact_boundary_to_raytracing_transformation("raytrace", m1, "u1", EXTERIOR)
md.add_slave_contact_boundary_to_raytracing_transformation("raytrace", m2, "u2", EXTERIOR)
md.add_macro("normal2", "Normalized(Inv(F(u2)').Normal)")
md.add_macro("gap2", "(Interpolate(X,raytrace)+Interpolate(u1,raytrace)-X-u2).normal2")
md.add_nonlinear_term(mimc2, "Interpolate_filter(raytrace, -c*neg_part(gap2)"
                             "*(Interpolate(Test_u1,raytrace)-Test_u2).normal2, 1)", EXTERIOR)
print("%i hexahedra per ring, %i dofs" % (m1.nb_convexes(), md.nb_dof()))

frames = []
def ramp(t):
  return 1.8*rr*max(0., (t - 0.4)/0.6)**2
t = 0.
print("step      t  Newton  max|u1|  max|u2|  contact force")
for step in range(55):
  if step <= 30:
    t = step/30
    y = -5*r*min(t/0.4, 1.)
    md.set_variable("p21", [1.8*rr*max(t, min(4*t, 0.2)), y, 0.])
    md.set_variable("p22", [ramp(t), y, 0.])
  elif step <= 38:
    if step == 31:
      for q in ["q12", "q21"]:
        md.disable_variable(q)
        md.set_variable(q, [0., 0., 0.])
    md.set_variable("penal", [[1e2, 3e1, 1e1, 3e0, 1e0, 1e-1, 1e-3, 0.][step - 31]])
  else:
    t = 1 + (step - 38)/100
    md.set_variable("p22", [ramp(t), -5*r, 0.])
  # |R| <= 1e-8 of its start, or of the force E r^2 (GetFEM: of the
  # external load; the release steps start at a residual too small for a
  # relative test)
  nit, conv, info = md.solve(atol=1e-8*e*r*r)
  assert conv, "step %i: %s %s" % (step, info["status"], info["message"])
  force = gf.asm_generic(mimc2, 0, "Interpolate_filter(raytrace, c*neg_part(gap2), 1)",
                         EXTERIOR, md)
  print("%4i  %5.3f  %6i  %7.3f  %7.3f  %13.5e" % (step, t, nit, np.abs(md.variable("u1")).max(),
                                                   np.abs(md.variable("u2")).max(), force))
  files = []
  for name, mf, mfs, u in [("ring1", mf1, mfs1, "u1"), ("ring2", mf2, mfs2, "u2")]:
    files.append("%s_%i.vtu" % (name, step))
    vm = md.interpolation("sqrt(3/2)*Norm(Deviator(tau(%s)))" % u, mfs)
    mf.export_to_vtu(files[-1], md.variable(u), "u", mfs, vm, "von Mises")
  frames.append((step, files))
for i, name in enumerate(["ring1", "ring2"]):
  with open(name + ".pvd", "w") as f:
    f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n')
    for s, files in frames:
      f.write('<DataSet timestep="%i" file="%s"/>\n' % (s, files[i]))
    f.write("</Collection>\n</VTKFile>\n")
print("written: ring1.pvd, ring2.pvd")
