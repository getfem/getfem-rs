import os

import numpy as np
import pytest

import getfem_rs as gf
GMSH = os.path.join(os.path.dirname(__file__), "../../../data/gmsh")


def test_errors():
  with pytest.raises(OSError):
    gf.Mesh("import", "gmsh", os.path.join(GMSH, "none.msh"))
  with pytest.raises(ValueError):
    gf.Mesh("import", "gmsh", os.path.join(GMSH, "square.geo"))
  with pytest.raises(ValueError):
    gf.Mesh("import", "ansys", os.path.join(GMSH, "square_tri1_msh22.msh"))
