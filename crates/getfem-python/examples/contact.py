#!/usr/bin/env python3
"""Contact between two elastic bodies, written in GWFL, with GetFEM-rs: a
punch with a curved bottom pressed into a block. The two meshes are
coupled by a raytracing transformation from the bottom of the punch
(slave) to the top of the block (master); the gap along the deformed
normal n1 of the punch (Poulios, Renard, Comput. Struct. 153, 2015) is
  g = (Interpolate(X,ray) + Interpolate(u2,ray) - X - u1).n1.
Two formulations:
- penalty: force r*neg_part(g);
- Lagrange multiplier p (contact pressure) with the augmented condition
  p = pos_part(p - r*g), solved by semismooth Newton.

The punch is pushed down in steps; the bodies are written with their
displacement as punch_<k>.vtu, base_<k>.vtu with punch.pvd, base.pvd."""
import numpy as np

import getfem_rs as gf

# regions of both meshes
CONTACT, OUTER = 1, 2
# load steps and the final downward displacement of the punch top
STEPS, PUSH = 5, 0.05


def block(x0, y0, lx, ly, nx, ny, contact):
  """a mesh of [x0,x0+lx]x[y0,y0+ly] with regions CONTACT and OUTER on the
  faces of normal (0, contact) and (0, -contact)"""
  m = gf.Mesh("structured", "GT='GT_PK(2,1)';ORG=[%r,%r];SIZES=[%r,%r];NSUBDIV=[%i,%i]"
              % (x0, y0, lx, ly, nx, ny))
  m.set_region(CONTACT, m.outer_faces_with_direction([0., contact], 0.01))
  m.set_region(OUTER, m.outer_faces_with_direction([0., -contact], 0.01))
  return m


def bodies():
  # punch [0.6,1.4]x[0,0.5], bottom bent to y = 0.01 + 0.4 (x - 1)^2
  punch = block(0.6, 0., 0.8, 0.5, 16, 8, -1.)
  x, y = punch.pts()
  b = 0.01 + 0.4*(x - 1)**2
  punch.set_pts([x, b + y*(0.5 - b)/0.5])
  base = block(0., -0.5, 2., 0.5, 32, 8, 1.)
  mf1, mf2 = gf.MeshFem(punch, 2), gf.MeshFem(base, 2)
  mf1.set_classical_finite_element(2)
  mf2.set_classical_finite_element(2)
  return punch, base, mf1, mf2


def run(b, method, out):
  """per step: Newton iterations, total contact force, mean penetration;
  frames written if out (a list of (punch, base) pvd entries)"""
  punch, base, mf1, mf2 = b
  mim1, mim2 = gf.MeshIm(punch, 4), gf.MeshIm(base, 4)
  md = gf.Model()
  md.add_fem_variable("u1", mf1)
  md.add_fem_variable("u2", mf2)
  md.add_initialized_data("lambda", 1.)
  md.add_initialized_data("mu", 1.)
  md.add_fem_data("top", mf1)
  md.add_raytracing_transformation("ray", 0.1)
  md.add_slave_contact_boundary_to_raytracing_transformation("ray", punch, "u1", CONTACT)
  md.add_master_contact_boundary_to_raytracing_transformation("ray", base, "u2", CONTACT)
  # the deformed normal of the slave (Poulios, Renard 2015)
  md.add_macro("n1", "Transformed_unit_vector(Grad_u1, Normal)")
  md.add_macro("g", "((Interpolate(X,ray)+Interpolate(u2,ray)-X-u1).n1)")
  md.add_macro("dg", "(n1.Test_u1 - n1.Interpolate(Test_u2,ray))")
  for mim, u in [(mim1, "u1"), (mim2, "u2")]:
    md.add_linear_term(mim, "lambda*Div_%s*Div_Test_%s + mu*(Grad_%s+Grad_%s'):Grad_Test_%s"
                       % (u, u, u, u, u))
  if method == "penalty":
    md.add_initialized_data("r", 1e3)
    md.add_nonlinear_term(mim1, "Interpolate_filter(ray, r*neg_part(g)*dg, 1)", CONTACT)
  else:
    mfp = gf.MeshFem(punch, 1)
    mfp.set_classical_finite_element(1)
    mfp = gf.MeshFem("partial", mfp, mfp.basic_dof_on_region(CONTACT))
    md.add_fem_variable("p", mfp)
    md.add_initialized_data("r", 10.)
    md.add_nonlinear_term(mim1, "Interpolate_filter(ray, p*dg, 1)", CONTACT)
    # p = pos_part(p - r g) where a target is found, p = 0 elsewhere
    md.add_nonlinear_term(
      mim1, "p/r*Test_p - Interpolate_filter(ray, pos_part(p - r*g)/r*Test_p, 1)", CONTACT)
  md.add_Dirichlet_condition_with_multiplier(mim1, "u1", mf1, OUTER, "top")
  md.add_Dirichlet_condition_with_multiplier(mim2, "u2", mf2, OUTER)
  force = "Interpolate_filter(ray, r*neg_part(g), 1)" if method == "penalty" else "p"
  rows = []
  for k in range(1, STEPS + 1):
    d = PUSH*k/STEPS
    top = np.zeros(mf1.nb_dof())
    top[1::mf1.qdim()] = -d
    md.set_variable("top", top)
    nit, conv, info = md.solve()
    assert conv, "step %i: %s %s" % (k, info["status"], info["message"])
    f = gf.asm_generic(mim1, 0, force, CONTACT, md)
    # mean penetration where the bodies overlap
    pen = gf.asm_generic(mim1, 0, "Interpolate_filter(ray, neg_part(g), 1)", CONTACT, md)
    area = gf.asm_generic(mim1, 0, "Interpolate_filter(ray, Heaviside(-g), 1)", CONTACT, md)
    rows.append((nit, f, pen/area if area > 0 else 0.))
    if out is not None:
      files = []
      for mf, u, name in [(mf1, "u1", "punch"), (mf2, "u2", "base")]:
        files.append("%s_%i.vtu" % (name, k))
        mf.export_to_vtu(files[-1], md.variable(u), "u")
      out.append((d, files))
  return rows


def write_pvd(name, frames):
  with open(name, "w") as f:
    f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n')
    for t, file in frames:
      f.write('<DataSet timestep="%.17g" file="%s"/>\n' % (t, file))
    f.write("</Collection>\n</VTKFile>\n")


if __name__ == "__main__":
  b = bodies()
  pen = run(b, "penalty", None)
  frames = []
  mul = run(b, "multiplier", frames)
  print("step  push   | penalty: iter  force      mean penetration"
        " | multiplier: iter  force      mean penetration")
  for k in range(STEPS):
    d = PUSH*(k + 1)/STEPS
    a, c = pen[k], mul[k]
    print("%4i  %.3f  | %13i %.4e %.3e   | %16i %.4e %.3e"
          % (k + 1, d, a[0], a[1], a[2], c[0], c[1], c[2]))
  for i, name in enumerate(["punch", "base"]):
    write_pvd(name + ".pvd", [(d, files[i]) for d, files in frames])
  print("written: punch.pvd, base.pvd")
