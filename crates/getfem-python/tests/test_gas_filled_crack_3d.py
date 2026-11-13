"""The example gas_filled_crack_3d: the opening with the pressure fixed
and the first temperature step only (no full runs during XFEM
development), h = 5, against the Rust example's values (2 Nov 2026: the
same to all 11 digits printed)."""
import numpy as np

from gas_filled_crack_3d import simulate


def test_first_step_as_rust():
  (k, s), = list(simulate(5., 1))
  assert np.isclose(s["t"], 30., rtol=0, atol=1e-6)
  for key, v in [("p", 0.0846531348), ("norm_u", 4.2353773138e-1),
                 ("norm_theta", 2.6832816226e3)]:
    assert np.isclose(s[key], v, rtol=1e-9, atol=0), key
