import numpy as np
import pytest

import getfem_rs as gf

def test_structured():
  m = gf.Mesh("structured", "GT='GT_QK(2,1)';ORG=[0,0];SIZES=[1,2];NSUBDIV=[4,3]")
  assert (m.nb_points(), m.nb_convexes()) == (20, 12)
  assert np.array_equal(m.pts()[:, 19], [1.0, 2.0])
  # GetFEM's defaults: 2 subdivisions, the unit box; new lines as separators
  m = gf.Mesh("structured", "GT=\"GT_PK(3,1)\"\nORG=[1,0,0]")
  assert (m.nb_points(), m.nb_convexes()) == (27, 48)
  for param in ["NSUBDIV=[2,2]", "GT='GT_QK(2,1)';NSUBDIV=[2]",
                "GT='GT_QK(2,1)';SIZE=[1,1]", "GT='GT_QK(2,1)';ORG=x"]:
    with pytest.raises(ValueError):
      gf.Mesh("structured", param)
  with pytest.raises(ValueError):
    gf.Mesh("structured_box", "GT='GT_QK(2,1)'")


def test_extend_region():
  m = gf.Mesh("cartesian", [0.0, 1.0, 2.0], [0.0, 1.0])
  m.set_region(1, [[0], [1]])
  m.extend_region(1, [[1], [1]])
  m.extend_region(2, [[1]])
  mim = gf.MeshIm(m, 2)
  md = gf.Model()
  # GetFEM's values
  assert gf.asm_generic(mim, 0, "1+X(1)", 1, md) == pytest.approx(3.0)
  assert gf.asm_generic(mim, 0, "1+X(1)", 2, md) == pytest.approx(2.5)
