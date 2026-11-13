#!/usr/bin/env python3
"""-Laplace u = f on the unit square, u = sin(pi x) sin(pi y), with
GetFEM-rs: L2 and H1 errors for P1-P3 triangles and Q1-Q3 quadrilaterals
under refinement. Writes convergence.csv."""
import numpy as np

import getfem_rs as gf

exact = "sin(pi*X(1))*sin(pi*X(2))"
grad = "pi*[cos(pi*X(1))*sin(pi*X(2)); sin(pi*X(1))*cos(pi*X(2))]"
csv = ["element,k,n,h,dofs,err_l2,err_h1"]
for gt, label in [("GT_PK(2,1)", "triangle"), ("GT_QK(2,1)", "quad")]:
  for k in range(1, 4):
    prev = None
    for n in [4, 8, 16, 32]:
      m = gf.Mesh("structured", "GT='%s';ORG=[0,0];SIZES=[1,1];NSUBDIV=[%i,%i]" % (gt, n, n))
      mf = gf.MeshFem(m, 1)
      mf.set_classical_finite_element(k)
      mim = gf.MeshIm(m, 2*k + 4)
      # u = 0 on the boundary: the unknown on the interior DOFs
      m.set_region(1, m.outer_faces())
      boundary = set(mf.basic_dof_on_region(1))
      mfu = gf.MeshFem("partial", mf, [d for d in range(mf.nb_dof()) if d not in boundary])
      md = gf.Model()
      md.add_fem_variable("u", mfu)
      md.add_linear_term(mim, "Grad_u.Grad_Test_u - 2*sqr(pi)*%s*Test_u" % exact)
      md.solve()
      e0 = np.sqrt(gf.asm_generic(mim, 0, "sqr(u - %s)" % exact, -1, md))
      e1 = np.sqrt(gf.asm_generic(mim, 0, "Norm_sqr(Grad_u - %s)" % grad, -1, md))
      rates = "  rates %.2f %.2f" % (np.log2(prev[0]/e0), np.log2(prev[1]/e1)) if prev else ""
      print("%s k=%i n=%2i dofs=%6i  L2 %.3e  H1 %.3e%s" % (label, k, n, mf.nb_dof(), e0, e1, rates))
      csv.append("%s,%i,%i,%r,%i,%e,%e" % (label, k, n, 1/n, mf.nb_dof(), e0, e1))
      prev = (e0, e1)
with open("convergence.csv", "w") as f:
  f.write("\n".join(csv) + "\n")
