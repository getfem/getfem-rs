"""The example crossing_cracks_3d at h = 5 against the Rust example's
maximal displacements (2 Nov 2026: the same output, the level sets the
same GWFL expressions)."""
import numpy as np

from crossing_cracks_3d import points, simulate


def test_both_load_cases_as_rust():
  pts = points()
  assert pts.shape == (3, 30000)
  for u, umax in zip(simulate(5., 2, pts), [9.0515e-2, 8.9990e-2]):
    assert np.isclose(np.abs(u).max(), umax, rtol=1e-5, atol=0)
