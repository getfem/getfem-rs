"""The example penny_crack (enrichments "zone" and "front") at h = 0.25
against the Rust example's output (2 Nov 2026: the same to all printed
digits)."""
import numpy as np

from penny_crack import A, simulate


def rms(o):
  inner = o["r"] < 0.95*A
  return np.sqrt(((o["delta"] - o["exact"])[inner]**2).sum()/(o["exact"][inner]**2).sum())


def test_zone_and_front_as_rust():
  o = simulate(0.25, ["zone:0.25"])
  assert (o["dofs"], o["dropped"]) == (13506, 0)
  assert np.isclose(rms(o), 3.5270e-1, rtol=1e-4, atol=0)
  o = simulate(0.25, ["front:0.3:0.8:0.25"])
  assert (o["dofs"], o["dropped"]) == (13326, 0)
  assert np.isclose(rms(o), 1.4855e-1, rtol=1e-4, atol=0)
  # K_I/exact at 45 degrees
  X = o["front"].pts()
  j = np.argmin(np.abs(np.arctan2(X[1], X[0]) - np.pi/4))
  assert np.isclose(o["K"][0, j]/(2*np.sqrt(A/np.pi)), 0.93210, rtol=0, atol=5e-6)
