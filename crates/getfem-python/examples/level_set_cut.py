#!/usr/bin/env python3
"""Two crossing spheres cut a tetrahedral mesh of [-1, 1]^3 (level sets of
degree 2), with GetFEM-rs, as the Rust example level_set_cut: the volumes
of their intersection, difference and union and the area of the first
sphere by the level set integration methods, against the exact values.
Writes the cut mesh (the cut tetrahedra replaced by their
sub-tetrahedra) with the zone of each cell (0 outside, 1 in a only, 2 in
b only, 3 in both) to level_set_cut.vtu. Optional argument: the cubes
per direction (default 6)."""
import sys

import numpy as np

import getfem_rs as gf

C1, R1 = np.array([-0.15, 0.02, 0.01]), 0.45
C2, R2 = np.array([0.25, 0.05, -0.02]), 0.4


def sphere(m, c, r):
  return gf.LevelSet(m, 2, "sqr(x-%r)+sqr(y-%r)+sqr(z-%r)-%r" % (tuple(c.tolist()) + (r*r,)))


def solve(n=6):
  x = np.linspace(-1, 1, n + 1)
  m = gf.Mesh("regular_simplices", x, x, x)
  mls = gf.MeshLevelSet(m)
  mls.add(sphere(m, C1, R1))
  mls.add(sphere(m, C2, R2))
  mls.adapt()

  def measure(where):
    im = gf.Integ("IM_TETRAHEDRON(5)")
    return gf.asm_generic(gf.MeshIm("levelset", mls, where, im, None, im), 0, "1", -1)

  # exact: the lens of the two balls, and the balls
  d = np.linalg.norm(C2 - C1)
  lens = (np.pi*(R1 + R2 - d)**2*(d*d + 2*d*R2 - 3*R2*R2 + 2*d*R1 + 6*R1*R2 - 3*R1*R1)
          /(12*d))
  va, vb = 4*np.pi*R1**3/3, 4*np.pi*R2**3/3
  r = dict(m=m, mls=mls, volumes=[(csg, measure("inside(%s)" % csg), exact)
                                  for csg, exact in [("a*b", lens), ("a-b", va - lens),
                                                     ("a+b", va + vb - lens)]],
           area=(measure("boundary(a)"), 4*np.pi*R1*R1))
  return r


if __name__ == "__main__":
  n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
  s = solve(n)
  for csg, v, exact in s["volumes"]:
    print("volume %s: %.8f (exact %.8f, error %.1e)" % (csg, v, exact, v/exact - 1))
  a, ae = s["area"]
  print("area of a: %.8f (exact %.8f, error %.1e)" % (a, ae, a/ae - 1))
  # the cut mesh with the zone of each cell (at its centroid)
  cut = s["mls"].cut_mesh()
  mfz = gf.MeshFem(cut)
  mfz.set_classical_discontinuous_finite_element(0)
  c = mfz.basic_dof_nodes()
  zone = ((np.linalg.norm(c - C1[:, None], axis=0) < R1)
          + 2*(np.linalg.norm(c - C2[:, None], axis=0) < R2)).astype(float)
  print("%i convexes, %i in the cut mesh" % (s["m"].nb_convexes(), cut.nb_convexes()))
  mfz.export_to_vtu("level_set_cut.vtu", zone, "zone")
  print("wrote level_set_cut.vtu")
