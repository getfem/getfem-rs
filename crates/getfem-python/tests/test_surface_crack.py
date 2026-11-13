"""The example surface_crack (as the Rust one, the same output at h =
0.25, 2 Nov 2026): the crack through a 3D block in plane strain opens as
the 2D edge crack, with its stress intensity factor along the whole
front, and the semicircular surface crack less."""
import numpy as np

from surface_crack import simulate


def test_through_crack_as_the_edge_crack():
  h = 0.25
  edge = simulate("edge", h)
  through = simulate("through", h)
  scale = edge["delta"][0]
  assert scale > 2
  # measured: at most 0.8 % of the opening at the surface
  assert np.abs(edge["delta"] - through["delta"]).max() <= 0.015*scale
  # K_I along the front (ends too) as the edge crack's: measured 0.1 %
  assert edge["K"].shape == (3, 1)
  k = edge["K"][0, 0]
  K = through["K"]
  assert np.abs(K[0]/k - 1).max() < 5e-3
  assert np.abs(K[1:]).max() < 1e-2*k
  # the Rust example's value
  assert abs(k - 1.39575) < 5e-6
  semi = simulate("semi", h)
  assert 0.3*scale < semi["delta"][0] < 0.7*scale
