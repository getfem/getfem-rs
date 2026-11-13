import os

import numpy as np
import pytest

import getfem_rs as gf
EXODUS = os.path.join(os.path.dirname(__file__), "../../../data/exodus")


def test_import():
  # sets.exo: two quads in the plane z = 0 of a 3D file; side set 1 and
  # element set 1 in region 1, side set 2 in region 2, blocks 1, 7
  m = gf.Mesh("import", "exodus", os.path.join(EXODUS, "sets.exo"))
  assert (m.dim(), m.nb_points(), m.nb_convexes()) == (3, 6, 2)
  assert np.array_equal(m.region(1)[1] >= 0, [True, False])
  assert m.region(2).shape == (2, 1)
  assert m.region(100001).shape == (2, 1) and m.region(100007).shape == (2, 1)
  m.transform([[1, 0, 0], [0, 1, 0]])
  assert m.dim() == 2
  with pytest.raises(OSError):
    gf.Mesh("import", "exodus", os.path.join(EXODUS, "none.exo"))
  with pytest.raises(ValueError):
    gf.Mesh("import", "exodus", os.path.join(EXODUS, "make_refs.py"))


def test_export(tmp_path):
  m = gf.Mesh("cartesian", np.linspace(0, 1, 3), np.linspace(0, 1, 3))
  m.set_region(4, m.outer_faces())
  mf = gf.MeshFem(m, 2)
  mf.set_classical_finite_element(1)
  P = mf.basic_dof_nodes()
  U = P[0] + 2 * P[1]  # a vector field: x + 2 y in each component
  f = str(tmp_path / "u.exo")
  mf.export_to_exodus(f, U, "u", time=0.0)
  mf.export_to_exodus(f, 2 * U, "u", time=0.5, append=True)
  with pytest.raises(ValueError, match="fields u_x, u_y expected"):
    mf.export_to_exodus(f, U, "v", time=1.0, append=True)
  # export_to_vtu's style: one file per space
  with pytest.raises(ValueError, match="on this space only"):
    mf.export_to_exodus(f, mf, U, "u")
  with pytest.raises(ValueError, match="fields as U, name"):
    mf.export_to_exodus(f, U, "u", U)
  with pytest.raises(TypeError):
    mf.export_to_exodus(f, U, "u", ascii=True)
  m2 = gf.Mesh("import", "exodus", f)
  assert (m2.dim(), m2.nb_convexes()) == (2, 4)
  assert m2.region(4).shape == (2, 8)  # the 8 boundary edges
  m.export_to_exodus(str(tmp_path / "m.exo"))
  assert gf.Mesh("import", "exodus", str(tmp_path / "m.exo")).nb_convexes() == 4


def test_import_data(tmp_path):
  # T = x + 2 y written by meshio, read onto P1 (continuous and not)
  f = os.path.join(EXODUS, "meshio_2d.exo")
  m = gf.Mesh("import", "exodus", f)
  for disc in [False, True]:
    mf = gf.MeshFem(m, 1)
    if disc:
      mf.set_classical_discontinuous_finite_element(1)
    else:
      mf.set_classical_finite_element(1)
    P = mf.basic_dof_nodes()
    T, t = mf.import_from_exodus(f, "T")
    assert np.array_equal(T, P[0] + 2 * P[1]) and t == 0.0
  # a vector field over two steps, read back onto the exported space
  m = gf.Mesh("cartesian", np.linspace(0, 1, 3), np.linspace(0, 1, 3))
  mf = gf.MeshFem(m, 2)
  mf.set_classical_finite_element(2)
  U = np.arange(mf.nb_dof(), dtype=float)
  g = str(tmp_path / "u.exo")
  mf.export_to_exodus(g, U, "u", time=0.0)
  mf.export_to_exodus(g, -U, "u", time=1.0, append=True)
  V, t = mf.import_from_exodus(g, "u", 0)
  assert np.array_equal(V, U) and t == 0.0
  V, t = mf.import_from_exodus(g, "u")  # the last step
  assert np.array_equal(V, -U) and t == 1.0
  assert mf.import_from_exodus(g, "u", -2)[1] == 0.0
  with pytest.raises(ValueError, match="no time step 2"):
    mf.import_from_exodus(g, "u", 2)
  with pytest.raises(ValueError, match="no field v"):
    mf.import_from_exodus(g, "v")
