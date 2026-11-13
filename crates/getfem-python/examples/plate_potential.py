#!/usr/bin/env python3
"""Laplace equation on a plate with three holes (gmsh, curved P2
triangles), with GetFEM-rs: u = 0 on the outer boundary, u = 1 on the
holes. Writes plate.vtu."""
import os

import numpy as np

import getfem_rs as gf

OUTER, HOLES = 10, 20  # the file's physical tags
m = gf.Mesh("import", "gmsh", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "../../../data/examples/plate.msh"))
m.transform([[1, 0, 0], [0, 1, 0]])  # the file's mesh in the plane z = 0, made 2D
mf = gf.MeshFem(m, 1)
mf.set_classical_finite_element(2)
mim = gf.MeshIm(m, 6)
# u = ud + v: ud the prescribed values (0 outside, 1 on the holes), v the
# unknown on the other DOFs
outer, holes = mf.basic_dof_on_region(OUTER), mf.basic_dof_on_region(HOLES)
ud = np.zeros(mf.nb_dof()); ud[holes] = 1.
mfv = gf.MeshFem("partial", mf, sorted(set(range(mf.nb_dof())) - set(outer) - set(holes)))
md = gf.Model()
md.add_initialized_fem_data("ud", mf, ud)
md.add_fem_variable("v", mfv)
md.add_linear_term(mim, "(Grad_v + Grad_ud).Grad_Test_v")
md.solve()
u = md.interpolation("ud + v", mf)
print("%i elements, %i DOFs, u in [%.4f, %.4f]" % (m.nb_convexes(), mf.nb_dof(), u.min(), u.max()))
mf.export_to_vtu("plate.vtu", u, "u", ascii=True)
