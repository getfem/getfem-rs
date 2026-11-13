import pytest

import getfem_rs as gf


def test_by_name():
  gt = gf.GeoTrans("GT_PK(2,1)")
  assert (gt.name(), gt.dim()) == ("GT_PK(2,1)", 2)
  pf = gf.Fem("FEM_QK(3,2)")
  assert (pf.name(), pf.dim()) == ("FEM_QK(3,2)", 3)
  im = gf.Integ("IM_STRUCTURED_COMPOSITE(IM_TRIANGLE(6),3)")
  assert im.name() == "IM_STRUCTURED_COMPOSITE(IM_TRIANGLE(6),3)"
  assert repr(pf) == "gf.Fem(\"FEM_QK(3,2)\")"


def test_unknown():
  for c, name in [(gf.GeoTrans, "GT_XX(2)"), (gf.Fem, "FEM_XX(2)"), (gf.Integ, "IM_XX(2)")]:
    with pytest.raises(ValueError):
      c(name)
