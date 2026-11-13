"""The example inclined_penny_crack (as the Rust one): K_I, K_II and
K_III along the front of a penny crack tilted to a uniaxial tension,
against the infinite body's, within the Rust test's bounds (h = 0.25:
6 % of the peaks), and as the Rust example (the same output, the level
set the same GWFL expressions, 2 Nov 2026)."""
import numpy as np

from inclined_penny_crack import exact, simulate


def test_mixed_mode_along_the_front():
  s = simulate(0.25)
  X, K = s["front"].pts(), s["K"]
  assert np.unique(np.round(X, 12), axis=1).shape[1] == 14
  peak = [exact(np.array([0., 1., 0.]))[1][0], abs(exact(np.array([-1., 0., 0.]))[1][1]),
          abs(exact(np.array([0., 1., 0.]))[1][2])]
  # the Rust example's K at the angle 13.85 (K_I, K_II, K_III)
  j = np.argmin(np.abs([exact(x)[0] - np.radians(13.85) for x in X.T]))
  assert np.allclose(K[:, j], [0.85638, 0.56154, -0.08819], rtol=0, atol=5e-6)
  for x, k in zip(X.T, K.T):
    w, ke = exact(x)
    # K_III not at the ends on the symmetry plane
    for i in range(2 if x[1] < 1e-6 else 3):
      assert abs(k[i] - ke[i]) < 0.06*peak[i], (np.degrees(w), i + 1, k[i], ke[i])
