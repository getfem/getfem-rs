#!/usr/bin/env python3
"""Topology optimization of a cantilever (a user's GetFEM script, gf_topopt),
with GetFEM-rs: a beam LX x LY clamped on the left, a vertical traction on
the middle of its right end, large strains (a compressible neo-Hookean
law, plane strain), the material density rho(x) = 1/(1+exp(-x)) of a
design variable x on uniform order-3 B-splines (one interval per
element), stiffness RAMP-interpolated. The compliance (the load's work) is
minimized at the volume fraction 0.5 (multiplier multvol), with the
adjoint displacements ADJu and a penalty on steep design gradients. The
optimality conditions are solved together in pseudo-time: a viscous
filter term on x of step dt, dt doubled after fast Newton solves and
halved after failed ones, then removed (inv_dt = 0) for the final step.

Writes the density, the displacements and the von Mises stress of each
design iteration as topopt_<i>.vtu with a topopt.pvd collection and
prints the volume fraction, the measure of non-discreteness MND and the
objective."""
import numpy as np

import getfem_rs as gf

LEFT, RIGHT = 3, 5


def optimize(nx, ny, itermax, on_iteration):
  """Runs the optimization; on_iteration(it, md) after each design
  iteration, it: i, t (pseudo-time), dt (the step of the next damped
  solve), volume, mnd (measure of non-discreteness), objective"""
  lx, ly = 100., 25.
  e, nu = 1e3, 0.4
  vol = 0.5*lx*ly
  q_ramp, contrast = 4, 1e-10
  li, rfilt = 0.03*lx, 0.06*lx
  dy, traction = ly/20., -e/30.
  k_l = 1e-4
  eps = 1e-8

  m = gf.Mesh("structured",
              "GT='GT_QK(2,2)';ORG=[0,0];SIZES=[%.17g,%.17g];NSUBDIV=[%i,%i]" % (lx, ly, nx, ny))
  a, b = 1e-5, ly + 1e-5
  m.set_region(LEFT, m.outer_faces_in_box([-a, -a], [a, b]))
  m.set_region(RIGHT, m.outer_faces_in_box([lx - a, -a], [lx + a, b]))

  # displacements: incomplete Q2, clamped on the left
  mfn = gf.MeshFem(m, 2)
  mfn.set_finite_element(gf.Fem("FEM_Q2_INCOMPLETE(2)"))
  clamped = set(mfn.basic_dof_on_region(LEFT))
  mfu = gf.MeshFem("partial", mfn, [d for d in range(mfn.nb_dof()) if d not in clamped])
  # design variable: uniform B-splines of order 3
  mfx = gf.MeshFem("bspline_uniform", m, nx, ny, 3)
  mim = gf.MeshIm(m, 5)

  md = gf.Model()
  gf.define_gwfl_function("rho_der3", 1, "(exp(t)*(exp(2*t) - 4*exp(t) + 1))/pow(exp(t)+1,4)")
  gf.define_gwfl_function("rho_der2", 1, "exp(t)*(1-exp(t))/pow(1+exp(t),3)", "rho_der3(t)")
  gf.define_gwfl_function("rho_der", 1, "exp(-t)/sqr(1+exp(-t))", "rho_der2(t)")
  gf.define_gwfl_function("rho", 1, "1/(1+exp(-t))", "rho_der(t)")
  gf.define_gwfl_function("RAMP", 1, f"{contrast}+(1-{contrast})*t/(1+{q_ramp}*(1-t))")

  md.add_fem_variable("x", mfx)
  # the L2 projection of 0.5 (the basis is a partition of unity)
  md.set_variable("x", np.full(mfx.nb_dof(), 0.5))
  md.add_initialized_fem_data("x0", mfx, np.full(mfx.nb_dof(), 0.5))
  dt = 1e2
  md.add_initialized_data("inv_dt", 1/dt)
  area = gf.asm_generic(mim, 0, "1", -1, md)
  for k, v in [("Li", li), ("Rfilt", rfilt), ("k_l", k_l),
               ("K", e/(3*(1-2*nu))), ("G", e/(2*(1+nu))), ("rhobar", vol/area),
               ("LY", ly), ("DY", dy), ("load", traction)]:
    md.add_initialized_data(k, v)
  md.add_variable("multvol", 1)
  md.add_fem_variable("u", mfu)
  md.add_fem_variable("ADJu", mfu)

  for k, v in [
      ("load_weight", "pos_part(1-abs(X(2)-LY/2)/DY)"),
      # the objective
      ("OBJu", "-load_weight*u(2)"),
      ("Dev33(A)", "A-Id(2)/3*(Trace(A)+1)"),
      ("F(u)", "Id(2)+Grad_u"),
      ("F3d(u)", "Id(3)+[1,0;0,1;0,0]*Grad_u*[1,0,0;0,1,0]"),
      ("tau(u)", "K*log(Det(F(u)))*Id(2)+G*pow(Det(F(u)),-2/3)*Dev33(F(u)*F(u)')"),
      ("tau3d(u)", "K*log(Det(F(u)))*Id(3)+G*pow(Det(F(u)),-2/3)*Deviator(F3d(u)*F3d(u)')"),
      ("sigma3d(u)", "tau3d(u)/Det(F(u))")]:
    md.add_macro(k, v)
  md.add_nonlinear_term(mim, "-load_weight*Test_ADJu(2)", RIGHT)
  md.add_linear_term(mim, "load_weight*[0;-load].Test_u", RIGHT)
  for t in [
      "RAMP(rho(x))*((tau(u)*Inv(F(u)')):Grad(Test_u))",
      "RAMP(rho(x))*Diff((tau(u)*Inv(F(u)')):Grad(ADJu),u,Test_ADJu)",
      "Diff(RAMP(rho(x)),x,Test_x)*((tau(u)*Inv(F(u)')):Grad(ADJu))",
      # the volume constraint
      "(rho(x)-rhobar)*Test_multvol",
      "Diff(rho(x),x,Test_x)*multvol",
      # the penalty on steep design gradients
      "k_l*pow(pos_part(Norm(Grad(x))-8/Li),5)*(Normalized(Grad(x)).Grad(Test_x))"]:
    md.add_nonlinear_term(mim, t)
  # the viscous filter of the pseudo-time steps
  md.add_linear_term(mim, "inv_dt*((x-x0)*Test_x+sqr(Rfilt)*(Grad_x-Grad_x0).Grad_Test_x)")
  print("Displacement dofs: %i\nTotal model dofs: %i" % (mfu.nb_dof(), md.nb_dof()))

  # the equilibrium and its adjoint for the initial design
  for v in ["x", "multvol", "ADJu"]:
    md.disable_variable(v)
  nit, conv, info = md.solve(atol=1e-8, rtol=0.)
  assert conv, "equilibrium: " + info["message"]
  md.enable_variable("ADJu")
  nit, conv, info = md.solve(atol=1e-8, rtol=0.)
  assert conv, "adjoint: " + info["message"]
  for v in ["x", "multvol"]:
    md.enable_variable(v)
  print("Mechanical equilibrium solved for the initial design\n\nStarting optimization...")

  final_dt = 1e14
  # 0: damped steps, 1: the final undamped step tried, 2: done
  fin = 0
  t = 0.
  ctx = gf.MumpsContext()
  ctx.set_matrix(md.tangent_matrix())
  for i in range(itermax):
    print("\nDesign iteration %i" % i)
    md.set_variable("x0", "x")
    saved = md.from_variables()
    while True:
      restot = np.nan
      j = 0
      while j < 40:
        restot_prev = restot
        md.assembly("all")
        ctx.analyze()
        ctx.factorize()
        md.to_variables(md.from_variables() + ctx.solve(md.rhs()))
        md.assembly("rhs")
        r = md.rhs()
        restot = np.linalg.norm(r)
        parts = []
        for n in ["x", "multvol", "u", "ADJu"]:
          i0, size = md.interval_of_variable(n)
          parts.append("%s: %4.1e" % (n, np.linalg.norm(r[i0:i0+size])))
        print("%i (dt=%s):%i: res=%8.2e = %s" % (i, "inf" if fin == 1 else "%g" % dt,
                                                 j + 1, restot, " + ".join(parts)))
        if np.isnan(restot) or restot < eps or (j > 5 and restot > restot_prev):
          break
        j += 1
      if not np.isnan(restot) and restot < eps:
        t += dt
        if fin == 1:
          fin = 2
        elif j < 5:
          if dt > final_dt:
            fin = 1
            md.set_variable("inv_dt", [0.])
            print("Disabling damping of optimality and intending final step")
          else:
            dt *= 2
            md.set_variable("inv_dt", [1/dt])
        break
      md.to_variables(saved)
      if fin == 1:
        final_dt *= 10
        fin = 0
        print("Reenabling damping of optimality")
      else:
        dt *= 0.5
      md.set_variable("inv_dt", [1/dt])
    it = dict(i=i, t=t, dt=dt,
              volume=gf.asm_generic(mim, 0, "rho(x)", -1, md)/(lx*ly),
              mnd=gf.asm_generic(mim, 0, "4*rho(x)*(1-rho(x))", -1, md)/(lx*ly),
              objective=gf.asm_generic(mim, 0, "OBJu", RIGHT, md))
    print("%i (t=%e): Volume fraction = %.6f, MND = %.6f, Objective value = %.6f"
          % (i, t, it["volume"], it["mnd"], it["objective"]))
    on_iteration(it, md)
    if fin == 2:
      break


if __name__ == "__main__":
  frames = []
  def on_iteration(it, md):
    mfx = md.mesh_fem_of_variable("x")
    mfu = md.mesh_fem_of_variable("u")
    mfout = gf.MeshFem(mfx.linked_mesh(), 1)
    mfout.set_classical_discontinuous_finite_element(2)
    fields = [mfx, md.variable("x"), "x"]
    for e, name in [("rho(x)", "rho"), ("RAMP(rho(x))", "ramp"),
                    ("Norm(Deviator(sigma3d(u)))", "Von Mises stress")]:
      fields += [mfout, md.interpolation(e, mfout), name]
    fields += [mfu, md.variable("u"), "Displacements"]
    f = "topopt_%i.vtu" % it["i"]
    mfout.export_to_vtu(f, *fields)
    frames.append((it["t"], f))
  optimize(80, 20, 150, on_iteration)
  with open("topopt.pvd", "w") as f:
    f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n')
    for t, name in frames:
      f.write('<DataSet timestep="%.17g" file="%s"/>\n' % (t, name))
    f.write("</Collection>\n</VTKFile>\n")
  print("written: topopt.pvd")
