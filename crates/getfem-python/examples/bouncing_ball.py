#!/usr/bin/env python3
"""Bouncing ball, with GetFEM-rs: an elastic disc (centre (0, 20), radius
20, Q2) lifted above a rigid ground y = 0 and dropped under a vertical
load, the 2D case of GetFEM's demo_dynamic_contact.m (its parameters:
lambda = mu = 20, the load 0.1, dt = 0.01, penalty 100). Newmark (beta =
1/4, gamma = 1/2) written out: with the previous u0, v0, a0 as data, the
mass term on the prediction,
  (u - u0 - dt v0 - dt^2 (1/2 - beta) a0)/(beta dt^2).Test_u,
then a = (u - u0 - dt v0 - dt^2 (1/2 - beta) a0)/(beta dt^2) and
v = v0 + dt ((1 - gamma) a0 + gamma a); the contact by the penalty
potential r neg_part(y + u_y)^2 on the lower half of the boundary.

Writes the disc with its displacement as ball_<i>.vtu with a ball.pvd
collection and prints the energy, the lowest point, the mean vertical
displacement and the contact force."""
import numpy as np

import getfem_rs as gf

CONTACT = 1


def simulate(nsubdiv, lift, dt, t_end, on_step):
  """Runs the drop; on_step(step, md) after each step (and at t = 0),
  step: t, energy (kinetic + elastic - work of the load), ymin (min of
  y + u_y over the nodes), uc (mean vertical displacement), force (total
  contact force), iterations"""
  clambda, cmu, load, penalty = 20., 20., 0.1, 100.
  beta, gamma = 0.25, 0.5
  m = gf.Mesh("structured_ball",
              "GT='GT_QK(2,2)';ORG=[0,20];SIZES=[20];NSUBDIV=[%i,%i];SYMMETRIES=0"
              % tuple(nsubdiv))
  # the faces of n_y < -0.01
  m.set_region(CONTACT, m.outer_faces_with_direction([0., -1.],
                                                     np.arctan2(np.sqrt(1-1e-4), 0.01)))
  mf = gf.MeshFem(m, 2)
  mf.set_classical_finite_element(2)
  mim = gf.MeshIm(m, 4)
  # the contact on a refined rule (the demo's)
  mimc = gf.MeshIm(m, gf.Integ("IM_STRUCTURED_COMPOSITE(IM_GAUSS_PARALLELEPIPED(2,4),4)"))
  n = mf.nb_dof()
  y = mf.basic_dof_nodes()[1, 1::2]

  md = gf.Model()
  md.add_fem_variable("u", mf)
  u0 = np.zeros(n); u0[1::2] = lift
  # at rest, the rigid lift: the acceleration of the load alone
  a0 = np.zeros(n); a0[1::2] = -load
  md.add_initialized_fem_data("u0", mf, u0)
  md.add_fem_data("v0", mf)
  md.add_initialized_fem_data("a0", mf, a0)
  md.set_variable("u", u0)
  for k, v in [("clambda", clambda), ("cmu", cmu), ("r", penalty),
               ("beta", beta), ("dt", dt), ("gamma", gamma)]:
    md.add_initialized_data(k, v)
  md.add_initialized_data("load", [0., -load])
  md.add_macro("pred", "(u0 + dt*v0 + dt*dt*(0.5-beta)*a0)")
  md.add_macro("acc", "(u - pred)/(beta*dt*dt)")
  md.add_nonlinear_term(mim, "acc.Test_u + (clambda*Div_u*Id(2) + 2*cmu*Sym(Grad_u)):Grad_Test_u"
                             " - load.Test_u")
  md.add_nonlinear_term(mimc, "r*sqr(neg_part(X(2) + u(2)))", CONTACT)

  area = gf.asm_generic(mim, 0, "1", -1, md)
  def record(t, iterations):
    on_step(dict(
      t=t,
      energy=gf.asm_generic(mim, 0, "0.5*v0.v0 + 0.5*(clambda*Div_u0*Id(2)"
                            " + 2*cmu*Sym(Grad_u0)):Grad_u0 - load.u0", -1, md),
      ymin=(y + md.variable("u")[1::2]).min(),
      uc=gf.asm_generic(mim, 0, "u(2)", -1, md)/area,
      force=gf.asm_generic(mimc, 0, "2*r*neg_part(X(2) + u(2))", CONTACT, md),
      iterations=iterations), md)
  record(0., 0)
  # |R| < 1e-8, the demo's (the default relative test stops each impact
  # step at 1e-8 of a large initial residual)
  for k in range(1, int(round(t_end/dt)) + 1):
    nit, conv, info = md.solve(atol=1e-8, rtol=0.)
    if not conv:
      raise RuntimeError("step %i: %s" % (k, info["message"]))
    # Newmark's update, all from the values of this step
    md.set_variables([("a0", "acc"),
                      ("v0", "v0 + dt*((1-gamma)*a0 + gamma*acc)"),
                      ("u0", "u")])
    record(k*dt, nit)


if __name__ == "__main__":
  frames = []
  next_frame = [0.]
  print("     t   energy      lowest point  mean u_y   contact force  Newton")
  def on_step(s, md):
    if s["t"] < next_frame[0] - 1e-9:
      return
    next_frame[0] += 0.5
    print("%6.2f  %.6e  %10.5f  %10.5f  %12.5e  %4i"
          % (s["t"], s["energy"], s["ymin"], s["uc"], s["force"], s["iterations"]))
    f = "ball_%i.vtu" % len(frames)
    md.mesh_fem_of_variable("u").export_to_vtu(f, md.variable("u"), "u")
    frames.append((s["t"], f))
  simulate([6, 3], 5., 0.01, 40., on_step)
  with open("ball.pvd", "w") as f:
    f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n')
    for t, name in frames:
      f.write('<DataSet timestep="%.17g" file="%s"/>\n' % (t, name))
    f.write("</Collection>\n</VTKFile>\n")
  print("written: ball.pvd")
