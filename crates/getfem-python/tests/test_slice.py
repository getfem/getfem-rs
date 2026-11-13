"""gf.Slice: the slice operations against GetFEM's Python (the same
numbers of simplexes), and GetFEM's check_levelset.py (a degree 1 mesh,
not noised): the level sets, crack tips and DOFs as GetFEM, the slice of
the cut mesh covering the domain, a field discontinuous across the crack
on it."""
import numpy as np

import getfem_rs as gf


def test_slice_operations(tmp_path):
  x = np.linspace(0, 1, 5)
  m = gf.Mesh("regular_simplices", x, x)
  sl = gf.Slice(("none",), m, 2)
  assert (sl.dim(), sl.nbpts(), list(sl.nbsplxs()), len(sl.cvs())) == (2, 192, [0, 0, 128], 32)
  assert sl.pts().shape == (2, 192)
  s, cv2s = sl.splxs(2)
  assert s.shape == (3, 128) and list(cv2s[:4]) == [0, 4, 8, 12] and cv2s[-1] == 128
  # GetFEM's numbers
  for op, args, n in [(("boundary",), (1,), [0, 16, 0]),
                      (("planar", 0, [0.5, 0.5], [1., 1.]), (1,), [0, 8, 0]),
                      (("union", ("ball", -1, [0., 0.], 0.3), ("ball", -1, [1., 1.], 0.3)), (2,),
                       [0, 0, 36]),
                      (("diff", ("none",), ("ball", -1, [0.5, 0.5], 0.3)), (2, m.outer_faces()),
                       [0, 32, 0])]:
    assert list(gf.Slice(op, m, *args).nbsplxs()) == n, op
  # deformed by a displacement
  mfu = gf.MeshFem(m, 2)
  mfu.set_classical_finite_element(1)
  d = np.zeros(mfu.nb_dof())
  d[0::2] = 0.1
  p = gf.Slice(("none",), mfu, d, 1).pts()
  assert np.isclose(p[0].min(), 0.1) and np.isclose(p[0].max(), 1.1)
  sl.export_to_vtu(str(tmp_path / "sl.vtu"), mfu, d, "d")


def test_check_levelset(tmp_path):
  eps = 1.0/10
  m = gf.Mesh("regular_simplices", np.arange(-1, 1 + eps, eps), np.arange(-1, 1 + eps, eps))
  ls1 = gf.LevelSet(m, 2, "y", "x")
  ls2 = gf.LevelSet(m, 2, "y+0.1", "x")
  ls3 = gf.LevelSet(m, 2, "y-0.1", "x")
  mls = gf.MeshLevelSet(m)
  mls.add(ls1)
  mls.sup(ls1)
  mls.add(ls1)
  mls.add(ls2)
  mls.add(ls2)
  mls.add(ls2)
  mls.add(ls3)
  mls.adapt()
  # GetFEM's numbers
  assert len(mls.levelsets()) == 3
  assert sorted(mls.crack_tip_convexes()) == [339, 340, 378, 379, 380, 381, 418, 419, 420, 421,
                                              459, 460]
  mf = gf.MeshFem(m)
  mf.set_classical_finite_element(1)
  mfls = gf.MeshFem("levelset", mls, mf)
  assert mfls.nb_dof() == 471

  sl = gf.Slice(("none",), mls, 2)
  p = sl.pts()
  s = sl.splxs(2)[0]
  e1, e2 = p[:, s[1]] - p[:, s[0]], p[:, s[2]] - p[:, s[0]]
  assert np.isclose(0.5*np.abs(e1[0]*e2[1] - e1[1]*e2[0]).sum(), 4.)
  assert np.allclose(gf.interpolation(mfls, np.ones(mfls.nb_dof()), sl), 1.)
  # a random field: discontinuous across the crack y = 0, x < 0 (the
  # nodes there in pairs, one per side), not the P1 one (but for GetFEM's
  # reference points 1% toward the simplex centre in the cut convexes)
  rng = np.random.default_rng(0)
  u = gf.interpolation(mfls, rng.random(mfls.nb_dof()), sl)
  v = gf.interpolation(mf, rng.random(mf.nb_dof()), sl)
  on = np.nonzero((np.abs(p[1]) < 1e-12) & (p[0] < -0.2))[0]
  key = np.round(p[:, on].T, 9)
  _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
  jump_u, jump_v = [], []
  for k in np.nonzero(cnt > 1)[0]:
    nodes = on[inv == k]
    jump_u.append(np.ptp(u[nodes]))
    jump_v.append(np.ptp(v[nodes]))
  assert len(jump_u) > 10
  assert np.median(jump_u) > 0.1 and max(jump_v) < 5e-3
  sl.export_to_vtu(str(tmp_path / "slU.vtu"), mfls, rng.random(mfls.nb_dof()), "U")
