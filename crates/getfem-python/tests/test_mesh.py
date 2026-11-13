import os

import numpy as np
import pytest

import getfem_rs as gf


def test_cartesian():
  # as GetFEM: points in Fortran order
  m = gf.Mesh("cartesian", [0.0, 1.0, 3.0], [0.0, 2.0])
  assert (m.dim(), m.nb_points(), m.nb_convexes()) == (2, 6, 2)
  pts = [[0, 1, 3, 0, 1, 3], [0, 0, 0, 2, 2, 2]]
  assert np.array_equal(m.pts(), pts)
  m = gf.Mesh("cartesian", np.linspace(0, 1, 3), [0.0, 1.0], [0.0, 1.0])
  assert (m.dim(), m.nb_points(), m.nb_convexes()) == (3, 12, 2)


def test_empty_add_convex():
  # two bilinear quadrilaterals, the shared points merged
  m = gf.Mesh("empty", 2)
  gt = gf.GeoTrans("GT_QK(2,1)")
  pts = np.array([[0, 1, 0, 1, 1, 3, 1, 3], [0, 0, 1, 1, 0, 0, 1, 2.]])
  assert np.array_equal(m.add_convex(gt, pts), [0, 1])
  assert (m.nb_points(), m.nb_convexes()) == (6, 2)
  with pytest.raises(ValueError):
    m.add_convex(gt, pts[:, :3])


def test_regions():
  # as GetFEM's: convexes (row 0), faces (row 1)
  m = gf.Mesh("cartesian", np.linspace(0, 1, 4), [0.0, 0.5, 2.0])
  of = [[0, 0, 1, 2, 2, 3, 3, 4, 5, 5], [1, 3, 3, 0, 3, 1, 2, 2, 0, 2]]
  assert np.array_equal(m.outer_faces(), of)
  bottom = m.outer_faces_in_box([-0.1, -0.1], [1.1, 0.1])
  assert np.array_equal(bottom, [[0, 1, 2], [3, 3, 3]])
  m.set_region(1, bottom)
  m.set_region(2, [[0, 1]])
  with pytest.raises(ValueError):
    m.set_region(3, [[0, 1], [2]])
  # convexes as a vector, or face -1 (the convex itself, as GetFEM)
  mim = gf.MeshIm(m, 2)
  area = lambda rg: gf.asm_generic(mim, 0, "1", rg, gf.Model())
  m.set_region(4, [0, 1])
  m.set_region(5, np.array([[0, 1], [-1, -1]]))
  m.extend_region(5, np.array([2]))
  m.extend_region(4, [[2], [-1]])
  assert abs(area(4) - 0.5) < 1e-12 and area(4) == area(5)
  m.set_region(6, [[0, 1], [3, -1]])
  m.set_region(7, [[0], [3]])
  m.set_region(8, [1])
  assert abs(area(6) - area(7) - area(8)) < 1e-12
  with pytest.raises(ValueError):
    m.set_region(3, [[0], [-2]])
  # read back: two rows, face -1 for a whole convex (GetFEM's)
  assert np.array_equal(m.region(1), bottom)
  assert np.array_equal(m.region(6), [[0, 1], [3, -1]])
  assert np.array_equal(m.region([7, 8]), [[0, 1], [3, -1]])
  # GetFEM's region_subtract, region_merge, region_intersect (r1 changed)
  m.set_region(9, [0, 1, 2])
  m.region_subtract(9, 8)
  assert np.array_equal(m.region(9), [[0, 2], [-1, -1]])
  m.region_merge(9, 7)
  assert np.array_equal(m.region(9), [[0, 0, 2], [-1, 3, -1]])
  m.region_intersect(9, 6)
  assert np.array_equal(m.region(9), [[0], [3]])
  m.region_subtract(9, 100)  # a missing region: empty
  assert np.array_equal(m.region(9), [[0], [3]])
  m.region_merge(101, 9)
  assert np.array_equal(m.region(101), [[0], [3]])


def test_load():
  # a GetFEM mesh file
  meshes = os.path.join(os.path.dirname(__file__), "../../../data/meshes")
  m = gf.Mesh("load", os.path.join(meshes, "thermo_elasticity_electrical_coupling.mesh"))
  assert m.dim() == 2 and m.nb_convexes() > 0 and m.region(1).shape[0] == 2


def test_errors():
  with pytest.raises(ValueError):
    gf.Mesh("nonexistent")
  with pytest.raises(ValueError):
    gf.Mesh("cartesian", [0.0])
  with pytest.raises(ValueError):
    gf.Mesh("cartesian", [0.0, 0.0])


def test_prismatic_clone_translate_points_of_convex():
  m = gf.Mesh("cartesian", [0., 1., 2.])
  p = gf.Mesh("prismatic", m, 3, 2)
  assert (p.dim(), p.nb_convexes(), p.nb_points()) == (2, 6, 3*7)
  assert p.trans_of_convex(0).name() == "GT_PRODUCT(GT_PK(1,1),GT_PK(1,2))"
  # the points of convex 0: 2 x 3 nodes in one cell and one layer
  pts = p.points_of_convex(0)
  assert pts.shape == (2, 6)
  assert np.ptp(pts[0]) == 1. and np.ptp(pts[1]) == pytest.approx(1/3)
  c = gf.Mesh("clone", p)
  c.translate([10., 0.])
  assert p.pts()[0].min() == 0. and c.pts()[0].min() == 10.
  for f in [p.points_of_convex, p.trans_of_convex]:
    with pytest.raises(ValueError):
      f(6)


def test_merge_points():
  # a strip of 8 x 2 quads wrapped into an annulus: its ends coincide
  m = gf.Mesh("structured", "GT='GT_QK(2,1)';SIZES=[1,1];NSUBDIV=[8,2]")
  x, y = m.pts()
  m.set_pts([(1 + y)*np.cos(2*np.pi*x), (1 + y)*np.sin(2*np.pi*x)])
  assert (m.nb_points(), m.outer_faces().shape[1]) == (27, 20)
  # convex 1 does not touch the seam
  m.set_region(1, [1])
  assert m.merge_points(0., 1) == 0
  # a convex would get a point twice: an error, the mesh unchanged
  with pytest.raises(ValueError):
    m.merge_points(0.5)
  assert m.merge_points() == 3
  assert (m.nb_points(), m.outer_faces().shape[1]) == (24, 16)


def test_transform():
  # the unit square embedded in 3D, in the plane y = 0
  m = gf.Mesh("cartesian", [0., 1.], [0., 1.])
  m.transform(np.array([[1., 0.], [0., 0.], [0., 1.]]))
  assert m.dim() == 3
  assert np.array_equal(m.pts()[1], np.zeros(4)) and m.pts()[2].max() == 1.
  with pytest.raises(ValueError):
    m.transform(np.eye(2))
  # nested lists or tuples, integers: a line embedded in 2D
  m = gf.Mesh("cartesian", [0., 2.])
  m.transform([[1], [0]])
  assert m.dim() == 2 and np.array_equal(m.pts(), [[0., 2.], [0., 0.]])
  m.transform(((0., -1.), (1., 0.)))
  assert np.array_equal(m.pts(), [[0., 0.], [0., 2.]])
  with pytest.raises(TypeError):
    m.transform([1, 0])


def test_faces_of_convexes(tmp_path):
  # 3 x 2 quadrilaterals
  m = gf.Mesh("cartesian", np.linspace(0, 1, 4), np.linspace(0, 1, 3))
  assert m.outer_faces().shape == (2, 10) and m.inner_faces().shape == (2, 7)
  assert m.outer_faces(1).shape == (2, 0)  # no 1D convexes
  # the boundary of convexes 0, 1 (bottom left), the face between them
  assert m.outer_faces(2, [0, 1]).shape == (2, 6)
  assert m.inner_faces([0, 1]).shape == (2, 1)
  with pytest.raises(ValueError):
    m.inner_faces([0, 6])
  # a line in 2D: its ends
  m1 = gf.Mesh("cartesian", np.linspace(0, 1, 5))
  m1.transform([[1], [0]])
  assert np.array_equal(m1.outer_faces(1), [[0, 3], [1, 0]])
  m.export_to_vtu(str(tmp_path / "m.vtu"))
  assert (tmp_path / "m.vtu").read_text(errors="ignore").find("<VTKFile") >= 0
