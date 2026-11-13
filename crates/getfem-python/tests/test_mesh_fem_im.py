import numpy as np
import pytest

import getfem_rs as gf


@pytest.fixture
def m():
  return gf.Mesh("cartesian", np.linspace(0, 1, 4), [0.0, 0.5, 2.0])


def test_mesh_fem(m):
  # nb_dof as GetFEM's
  for q, k, n in [((), 1, 12), ((2,), 2, 70), ((3, 3), 1, 108)]:
    mf = gf.MeshFem(m, *q)
    mf.set_classical_finite_element(k)
    assert mf.nb_dof() == n
  mf = gf.MeshFem(m, 2)
  mf.set_finite_element(gf.Fem("FEM_QK(2,3)"), [0, 1])
  assert (mf.qdim(), mf.nb_dof()) == (2, 56)
  for k, a, n in [(0, 0.0, 12), (1, 0.0, 48), (2, 0.1, 108)]:
    mf = gf.MeshFem(m, 2)
    mf.set_classical_discontinuous_finite_element(k, a)
    assert mf.nb_dof() == n


def test_mesh_im(m):
  gf.MeshIm(m, 4)
  mim = gf.MeshIm(m, gf.Integ("IM_GAUSS_PARALLELEPIPED(2,4)"))
  mim.set_integration_method(2, [0])
  with pytest.raises(ValueError):
    mim.set_integration_method("IM_GAUSS_PARALLELEPIPED(2,4)")


def test_bspline_uniform():
  # cubic B-splines on 4 x 2 intervals: free n+2 per direction, periodic n
  m = gf.Mesh("cartesian", np.linspace(0, 4, 5), np.linspace(0, 2, 3))
  assert gf.MeshFem("bspline_uniform", m, 4, 2, 3).nb_dof() == 6*4
  assert gf.MeshFem("bspline_uniform", m, 4, 2, 3, "periodic", "free").nb_dof() == 4*4
  assert gf.MeshFem("bspline_uniform", m, 4, 2, 3,
                    "free", "free", "symmetry", "symmetry").nb_dof() == 15
  for a in [(4, 3), (4, 2, 3, "free"), (4, 2, 3, "open", "free")]:
    with pytest.raises(ValueError):
      gf.MeshFem("bspline_uniform", m, *a)
