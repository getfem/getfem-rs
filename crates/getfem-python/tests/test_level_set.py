import numpy as np
import pytest

import getfem_rs as gf


def test_regular_simplices_non_uniform():
  m = gf.Mesh("regular_simplices", [0., 0.1, 0.5, 1.], [0., 2.])
  assert m.nb_convexes() == 6
  x, y = m.pts()
  assert sorted(set(np.round(x, 12))) == [0., 0.1, 0.5, 1.]
  assert sorted(set(np.round(y, 12))) == [0., 2.]


def test_level_set_values_from_strings_and_arrays():
  m = gf.Mesh("regular_simplices", np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))
  ls = gf.LevelSet(m, 2, "x*x+y*y-0.25", "x")
  p = ls.mf().basic_dof_nodes()
  assert np.allclose(ls.values(0), p[0]**2 + p[1]**2 - 0.25)
  assert np.allclose(ls.values(1), p[0])
  ls2 = gf.LevelSet(m, 2, p[1])
  assert np.allclose(ls2.values(), p[1])
  ls3 = gf.LevelSet(m, 1, "with_secondary")
  ls3.set_values("x", "y-0.5")
  q = ls3.mf().basic_dof_nodes()
  assert np.allclose(ls3.values(0), q[0]) and np.allclose(ls3.values(1), q[1] - 0.5)


def test_levelset_mesh_im_keeps_its_cut():
  # the disc of radius 0.4: set_integration_method sets the method of the
  # elements not cut only (GetFEM set_integ)
  x = np.linspace(-0.5, 0.5, 11)
  m = gf.Mesh("cartesian", x, x)
  mls = gf.MeshLevelSet(m)
  mls.add(gf.LevelSet(m, 2, "x*x+y*y-0.16"))
  mls.adapt()
  mim = gf.MeshIm("levelset", mls, "inside(a)", gf.Integ("IM_TRIANGLE(6)"))
  md = gf.Model()
  area = lambda: gf.asm_generic(mim, 0, "1", -1, md)
  a1 = area()
  mim.set_integration_method(4)
  assert abs(area() - a1) < 1e-12 and abs(a1 - np.pi*0.16) < 1e-6
  with pytest.raises(ValueError, match="all the elements not cut"):
    mim.set_integration_method(4, [0])


def test_global_functions():
  # sqrt(r) sin(theta/2) and sqrt(r) cos(theta/2), theta = atan2(y, x)
  r, t = 2., np.arctan2(1., -1.)
  x, y = r*np.cos(t), r*np.sin(t)
  f0, f1 = gf.GlobalFunction("crack", 0), gf.GlobalFunction("crack", 1)
  assert np.isclose(f0.val(x, y), np.sqrt(r)*np.sin(t/2))
  assert np.isclose(f1.val(x, y), np.sqrt(r)*np.cos(t/2))
  c = gf.GlobalFunction("cutoff", "polynomial", 0.5, 1.)
  assert c.val(0.3, 0.) == 1. and c.val(1.2, 0.) == 0.
  assert np.isclose((f0*c).val(0.3, 0.1), f0.val(0.3, 0.1))
  assert np.isclose((f0 + f1).val(x, y), f0.val(x, y) + f1.val(x, y))
  with pytest.raises(ValueError, match="2 parameters"):
    gf.GlobalFunction("cutoff", "polynomial2", 0.5)
  with pytest.raises(ValueError, match="no crack function 16"):
    gf.GlobalFunction("crack", 16)


def test_product_pu_dofs():
  # all of pu_mf's DOFs by default; pu_dofs keeps their functions only
  m = gf.Mesh("cartesian", np.linspace(0, 1, 4), np.linspace(0, 1, 3))
  ls = gf.LevelSet(m, 1, "y-0.5", "x-0.5")
  pu_mf = gf.MeshFem(m, 1)
  pu_mf.set_classical_finite_element(1)
  global_mf = gf.MeshFem("global function", m, ls,
                         [gf.GlobalFunction("crack", 0), gf.GlobalFunction("crack", 1)])
  n = pu_mf.nb_dof()
  assert gf.MeshFem("product", pu_mf, global_mf).nb_dof() == 2*n
  assert gf.MeshFem("product", pu_mf, global_mf, list(range(n))).nb_dof() == 2*n
  assert gf.MeshFem("product", pu_mf, global_mf, [0, 5]).nb_dof() == 4
  assert gf.MeshFem("product", pu_mf, global_mf, []).nb_dof() == 0
