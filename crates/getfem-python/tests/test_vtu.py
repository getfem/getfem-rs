import xml.etree.ElementTree as ET

import numpy as np
import pytest

import getfem_rs as gf


def arrays(f):
  root = ET.parse(f)
  out = {a.get("Name"): np.array(a.text.split(), dtype=float)
         for a in root.iter("DataArray") if a.get("Name")}
  out["points"] = np.array(next(root.iter("Points")).find("DataArray").text.split(),
                           dtype=float)
  return out


def test_export_to_vtu(tmp_path):
  m = gf.Mesh("cartesian", np.linspace(0, 1, 4), [0.0, 0.5, 2.0])
  mfu, mfp, mfo = gf.MeshFem(m, 2), gf.MeshFem(m), gf.MeshFem(m)
  mfu.set_classical_finite_element(2)
  mfp.set_classical_finite_element(1)
  mfo.set_classical_discontinuous_finite_element(2)
  md = gf.Model()
  md.add_fem_variable("u", mfu)
  md.add_fem_variable("p", mfp)
  md.set_variable("u", md.interpolation("[X(1)*X(2);X(1)]", mfu))
  md.set_variable("p", md.interpolation("X(1)+2*X(2)", mfp))
  s = md.interpolation("X(1)-X(2)*X(2)", mfo)
  f = str(tmp_path / "a.vtu")
  mfo.export_to_vtu(f, mfu, md.variable("u"), "u", mfp, md.variable("p"), "p", s, "s",
                    ascii=True)
  a = arrays(f)
  x, y = a["points"].reshape(-1, 3)[:, :2].T
  # GetFEM's: 6 Q2 cells, 9 points each; fields exact on these spaces
  assert x.size == 54
  assert np.allclose(a["p"], x + 2*y, rtol=0, atol=1e-14)
  assert np.allclose(a["s"], x - y*y, rtol=0, atol=1e-14)
  # vectors with 3 components (VTK)
  assert np.allclose(a["u"].reshape(-1, 3), np.c_[x*y, x, 0*x], rtol=0, atol=1e-14)
  # binary by default
  mfo.export_to_vtu(str(tmp_path / "b.vtu"), s, "s")
  assert 'format="binary"' in open(tmp_path / "b.vtu").read()
  with pytest.raises(ValueError):
    mfo.export_to_vtu(str(tmp_path / "c.vtu"), s)
