"""The example crack (GetFEM's demo_crack.py): variant 4 (two cracks)."""
from crack import solve


def test_two_cracks_jump():
  # the jump across the second crack penalized away
  s = solve("two_cracks", 20)
  assert s["jump"] < 1e-5 * s["norms"][0]
