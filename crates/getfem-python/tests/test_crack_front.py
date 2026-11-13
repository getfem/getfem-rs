"""Crack fronts: ls.crack_fronts (2D tips, 3D fronts), the DOFs near a
front and on the tip convexes, front hats, tip functions on a crack and
their products, gf.MeshFem("global function") with and without a crack."""
import numpy as np
import pytest

import getfem_rs as gf


def test_crack_tips_2d():
  x = np.linspace(-1, 1, 9)
  m = gf.Mesh("regular_simplices", x, x)
  ls = gf.LevelSet(m, 1, "y-0.01", "abs(x)-0.5")
  fronts = ls.crack_fronts()
  assert [(f[1], f[2]) for f in fronts] == [(None, False)]*2
  tips = sorted(f[0][:, 0].tolist() for f in fronts)
  assert np.allclose(tips, [[-0.5, 0.01], [0.5, 0.01]])
  mls = gf.MeshLevelSet(m)
  mls.add(ls)
  mls.adapt()
  mf = gf.MeshFem(m)
  mf.set_classical_finite_element(1)
  # the nodes within 0.3 of a tip (the level set exact at the nodes)
  P = mf.basic_dof_nodes()
  near = np.nonzero((P[1] - 0.01)**2 + (np.abs(P[0]) - 0.5)**2 < 0.09)[0]
  assert list(ls.select_mesh_fem_basic_dofs_near_front(mf, 0.3)) == list(near)
  # the DOFs of the tip convexes, as by a region of them
  cvs = mls.crack_tip_convexes(ls)
  assert cvs == mls.crack_tip_convexes() and len(cvs) == 4
  m.set_region(7, cvs)
  assert list(mf.basic_dof_on_region(cvs)) == list(mf.basic_dof_on_region(7))
  # a tip function on the crack: a function of the point
  g = gf.GlobalFunction("crack", 1)
  assert g.val(1., 0.) == 1.
  gl = gf.GlobalFunction("levelset", ls, g)
  assert gf.MeshFem("global function", m, [gl]).nb_dof() == 1
  with pytest.raises(ValueError):
    gl*g
  with pytest.raises(ValueError):
    gf.MeshFem("global function", m, [g])


def test_crack_front_3d():
  x = np.linspace(-1, 1, 5)
  m = gf.Mesh("regular_simplices", x, x, x)
  ls = gf.LevelSet(m, 2, "z-0.01", "sqrt(x*x+y*y)-0.5")
  fronts = ls.crack_fronts(0.25)
  assert len(fronts) == 1
  P, T, closed = fronts[0]
  n = P.shape[1]
  assert closed and P.shape == T.shape == (3, n) and n > 10
  # on the circle (the level set of degree 2 on 4^3), unit tangents
  assert np.abs(np.hypot(P[0], P[1]) - 0.5).max() < 1e-2
  assert np.allclose(np.linalg.norm(T, axis=0), 1.)
  assert np.abs((P[:2]*T[:2]).sum(0)).max() < 0.05
  # the hats: 1 at their point, 0 at the others, summing to 1 near it
  hats = [gf.GlobalFunction("front hat", P, T, closed, k, 0.3) for k in range(n)]
  assert np.allclose([[h.val(*P[:, j]) for j in range(n)] for h in hats], np.eye(n))
  assert np.isclose(sum(h.val(*(P[:, 0] + [0.05, 0.02, 0.1])) for h in hats), 1.)
  # tip functions along the front
  gl = gf.GlobalFunction("levelset", ls, gf.GlobalFunction("crack", 1))
  assert gf.MeshFem("global function", m, [h*gl for h in hats], 3).nb_dof() == 3*n
  assert gf.MeshFem("global function", m, ls, [gf.GlobalFunction("crack", 1),
                                               hats[0]]).nb_dof() == 2
  with pytest.raises(ValueError):
    ls.crack_fronts()


def test_dofs_with_support():
  x = np.linspace(-1, 1, 9)
  m = gf.Mesh("regular_simplices", x, x)
  ls = gf.LevelSet(m, 2, "x*x+y*y-0.36")
  mls = gf.MeshLevelSet(m)
  mls.add(ls)
  mls.adapt()
  mim = gf.MeshIm("levelset", mls, "inside(a)", gf.Integ("IM_TRIANGLE(4)"))
  mf = gf.MeshFem(m)
  mf.set_classical_finite_element(1)
  r = np.linalg.norm(mf.basic_dof_nodes(), axis=0)
  d = mf.dofs_with_support(mim, 1e-8)
  # the nodes in the disk, none farther than an element (0.25 sqrt 2)
  assert set(np.nonzero(r < 0.6)[0]) <= set(d) and r[d].max() < 0.6 + 0.36
