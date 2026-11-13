"""The example gas_filled_crack (GetFEM's
demo_cracked_thermo_elastic_body.py, as the Rust example): the opening
with the pressure fixed and the first temperature step only (no full runs
during XFEM development), against the Rust example's values (the same to
all printed digits, 1 Nov 2026)."""
import numpy as np

from gas_filled_crack import simulate


def test_first_step_as_rust():
  (k, s), = list(simulate(10, 1))
  assert np.isclose(s["t"], 30., rtol=0, atol=1e-9)
  for key, v in [("p", 0.065395), ("norm_u", 1.780858), ("norm_theta", 600.)]:
    assert np.isclose(s[key], v, rtol=1e-5, atol=0), key
