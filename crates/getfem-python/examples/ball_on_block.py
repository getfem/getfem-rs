#!/usr/bin/env python3
"""Ball on a block, with GetFEM-rs: the elastic disc of the example
bouncing_ball (centre (0, 20), radius 20, Q2, lifted and dropped under a
vertical load) falls on an elastic block [-40, 40] x [-20, 0] (Q2, fixed
at its bottom, no load), both lambda = mu = 20. Newmark (beta = 1/4,
gamma = 1/2) written out for both bodies, with the previous u0, v0, a0 as
data:
  (u - u0 - dt v0 - dt^2 (1/2 - beta) a0)/(beta dt^2).Test_u,
then a = (u - u0 - dt v0 - dt^2 (1/2 - beta) a0)/(beta dt^2) and
v = v0 + dt ((1 - gamma) a0 + gamma a). The contact between the lower
half of the disc (slave) and the top of the block (master) by a
raytracing transformation along the deformed normal of the disc,
n1 = Transformed_unit_vector(Grad_u1, Normal) (Poulios, Renard, Comput.
Struct. 153, 2015), the gap
  g = (Interpolate(X,ray) + Interpolate(u2,ray) - X - u1).n1,
and the penalty force r neg_part(g) n1.

Writes both bodies with their displacement as ball_<i>.vtu, block_<i>.vtu
with ball.pvd, block.pvd and prints the energy, the lowest point of the
disc, the deepest point of the block's top and the contact force."""
import numpy as np

import getfem_rs as gf

# regions: of the disc, of the block
CONTACT = 1
TOP, BOTTOM = 1, 2


def simulate(nsubdiv, block, lift, dt, t_end, on_step):
  """Runs the drop; on_step(step, md) after each step (and at t = 0),
  step: t, energy (kinetic + elastic energy of both bodies + penalty
  energy - work of the load), ymin (min of y + u_y over the nodes of the
  disc), dent (min of u_y over the nodes of the block's top), force (total
  contact force), iterations"""
  clambda, cmu, load, penalty = 20., 20., 0.1, 200.
  beta, gamma = 0.25, 0.5
  ball = gf.Mesh("structured_ball",
                 "GT='GT_QK(2,2)';ORG=[0,20];SIZES=[20];NSUBDIV=[%i,%i];SYMMETRIES=0"
                 % tuple(nsubdiv))
  # the faces of n_y < -0.01
  ball.set_region(CONTACT, ball.outer_faces_with_direction([0., -1.],
                                                           np.arctan2(np.sqrt(1-1e-4), 0.01)))
  blk = gf.Mesh("structured", "GT='GT_QK(2,1)';ORG=[-40,-20];SIZES=[80,20];NSUBDIV=[%i,%i]"
                % tuple(block))
  for n, rid in [([0., 1.], TOP), ([0., -1.], BOTTOM)]:
    blk.set_region(rid, blk.outer_faces_with_direction(n, 0.01))
  mf1, mf2 = gf.MeshFem(ball, 2), gf.MeshFem(blk, 2)
  mf1.set_classical_finite_element(2)
  mf2.set_classical_finite_element(2)
  mim1, mim2 = gf.MeshIm(ball, 4), gf.MeshIm(blk, 4)
  # the contact on a refined rule
  mimc = gf.MeshIm(ball, gf.Integ("IM_STRUCTURED_COMPOSITE(IM_GAUSS_PARALLELEPIPED(2,4),4)"))
  n1, n2 = mf1.nb_dof(), mf2.nb_dof()
  y1 = mf1.basic_dof_nodes()[1, 1::2]
  top = np.arange(1, n2, 2)[np.abs(mf2.basic_dof_nodes()[1, 1::2]) < 1e-9]

  md = gf.Model()
  md.add_fem_variable("u1", mf1)
  md.add_fem_variable("u2", mf2)
  # the disc at rest, lifted: the acceleration of the load alone
  u10 = np.zeros(n1); u10[1::2] = lift
  a10 = np.zeros(n1); a10[1::2] = -load
  for b, mf, u0, a0 in [("1", mf1, u10, a10), ("2", mf2, np.zeros(n2), np.zeros(n2))]:
    md.add_initialized_fem_data("u0_" + b, mf, u0)
    md.add_fem_data("v0_" + b, mf)
    md.add_initialized_fem_data("a0_" + b, mf, a0)
  md.set_variable("u1", u10)
  for k, v in [("clambda", clambda), ("cmu", cmu), ("r", penalty),
               ("beta", beta), ("dt", dt), ("gamma", gamma)]:
    md.add_initialized_data(k, v)
  md.add_initialized_data("load", [0., -load])
  for b, mim, f in [("1", mim1, " - load.Test_u1"), ("2", mim2, "")]:
    md.add_macro("acc_" + b, "(u%s - (u0_%s + dt*v0_%s + dt*dt*(0.5-beta)*a0_%s))/(beta*dt*dt)"
                 % (b, b, b, b))
    md.add_linear_term(mim, "acc_%s.Test_u%s + (clambda*Div_u%s*Id(2) + 2*cmu*Sym(Grad_u%s))"
                       ":Grad_Test_u%s%s" % (b, b, b, b, b, f))
  md.add_Dirichlet_condition_with_multiplier(mim2, "u2", mf2, BOTTOM)
  md.add_raytracing_transformation("ray", 2.)
  md.add_slave_contact_boundary_to_raytracing_transformation("ray", ball, "u1", CONTACT)
  md.add_master_contact_boundary_to_raytracing_transformation("ray", blk, "u2", TOP)
  # the deformed normal of the disc
  md.add_macro("n1", "Transformed_unit_vector(Grad_u1, Normal)")
  md.add_macro("g", "((Interpolate(X,ray)+Interpolate(u2,ray)-X-u1).n1)")
  md.add_nonlinear_term(
    mimc, "Interpolate_filter(ray, r*neg_part(g)*(n1.Test_u1 - n1.Interpolate(Test_u2,ray)), 1)",
    CONTACT)

  def record(t, iterations):
    md.set_variables([("u0_1", "u1"), ("u0_2", "u2")])
    e = lambda b, mim: gf.asm_generic(
      mim, 0, "0.5*v0_%s.v0_%s + 0.5*(clambda*Div_u0_%s*Id(2) + 2*cmu*Sym(Grad_u0_%s))"
              ":Grad_u0_%s" % (b, b, b, b, b), -1, md)
    on_step(dict(
      t=t,
      energy=e("1", mim1) + e("2", mim2)
             + gf.asm_generic(mimc, 0, "Interpolate_filter(ray, 0.5*r*sqr(neg_part(g)), 1)",
                              CONTACT, md)
             - gf.asm_generic(mim1, 0, "load.u0_1", -1, md),
      ymin=(y1 + md.variable("u1")[1::2]).min(),
      dent=min(0., md.variable("u2")[top].min()),
      force=gf.asm_generic(mimc, 0, "Interpolate_filter(ray, r*neg_part(g), 1)", CONTACT, md),
      iterations=iterations), md)
  record(0., 0)
  # |R| < 1e-8 (absolute: a relative test stops the impact steps at 1e-8
  # of a large initial residual)
  for k in range(1, int(round(t_end/dt)) + 1):
    nit, conv, info = md.solve(atol=1e-8, rtol=0.)
    if not conv:
      raise RuntimeError("step %i: %s %s" % (k, info["status"], info["message"]))
    # Newmark's update, all from the values of this step
    md.set_variables([(v + b, e % {"b": b}) for b in ["1", "2"] for v, e in [
      ("a0_", "acc_%(b)s"),
      ("v0_", "v0_%(b)s + dt*((1-gamma)*a0_%(b)s + gamma*acc_%(b)s)")]])
    record(k*dt, nit)


def write_pvd(name, frames):
  with open(name, "w") as f:
    f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n')
    for t, file in frames:
      f.write('<DataSet timestep="%.17g" file="%s"/>\n' % (t, file))
    f.write("</Collection>\n</VTKFile>\n")


if __name__ == "__main__":
  frames = []
  next_frame = [0.]
  print("     t   energy        disc lowest  block top  contact force  Newton")
  def on_step(s, md):
    if s["t"] < next_frame[0] - 1e-9:
      return
    next_frame[0] += 0.25
    print("%6.2f  %.6e  %10.5f  %10.5f  %12.5e  %4i"
          % (s["t"], s["energy"], s["ymin"], s["dent"], s["force"], s["iterations"]))
    files = []
    for name, u in [("ball", "u1"), ("block", "u2")]:
      files.append("%s_%i.vtu" % (name, len(frames)))
      md.mesh_fem_of_variable(u).export_to_vtu(files[-1], md.variable(u), "u")
    frames.append((s["t"], files))
  simulate([6, 3], [32, 8], 5., 0.01, 40., on_step)
  for i, name in enumerate(["ball", "block"]):
    write_pvd(name + ".pvd", [(t, files[i]) for t, files in frames])
  print("written: ball.pvd, block.pvd")
