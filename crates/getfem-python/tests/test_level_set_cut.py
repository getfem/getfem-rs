"""The example level_set_cut.py: the volumes and the area cut by two
spheres as the Rust example (and near the exact values)."""
from level_set_cut import solve


def test_level_set_cut():
  s = solve(6)
  # the Rust example's output
  for (csg, v, exact), want in zip(s["volumes"], [0.10936108, 0.27234115, 0.54042212]):
    assert abs(v - want) < 1e-8 and abs(v/exact - 1) < 1e-5, csg
  a, ae = s["area"]
  assert abs(a - 2.54469583) < 1e-8 and abs(a/ae - 1) < 1e-5
  assert s["mls"].cut_mesh().nb_convexes() == 59995
