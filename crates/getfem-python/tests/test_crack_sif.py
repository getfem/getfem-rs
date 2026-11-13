"""ls.crack_sifs: the inclined centre crack in a large plate under remote
tension (as tests/crack_sif.rs: K_I, K_II 1.6 % and 1.2 % above the
infinite plate's), one front selected; in 3D the fronts' points as a
gf.Slice, K at its points, exported."""
import numpy as np

import getfem_rs as gf


def test_inclined_centre_crack():
  L, h, a = 8., 0.2, 1.
  c, s = float(np.sqrt(3)/2), 0.5
  x = np.linspace(-L, L, int(2*L/h) + 1)
  m = gf.Mesh("cartesian", x, x)
  for rg, n in [(1, [0., 1.]), (2, [0., -1.])]:
    m.set_region(rg, m.outer_faces_with_direction(n, 0.5))
  ls = gf.LevelSet(m, 2, "-%r*x+%r*y" % (s, c), "abs(%r*x+%r*y)-%r" % (c, s, a))
  mls = gf.MeshLevelSet(m)
  mls.add(ls)
  mls.adapt()
  pu_mf = gf.MeshFem(m)
  pu_mf.set_classical_finite_element(1)
  global_mf = gf.MeshFem("global function", m, ls, [gf.GlobalFunction("crack", i) for i in range(4)])
  sing = gf.MeshFem("product", pu_mf, global_mf, pu_mf.basic_dof_on_region(mls.crack_tip_convexes(ls)))
  base = gf.MeshFem("sum", sing, gf.MeshFem("levelset", mls, pu_mf))
  mim = gf.MeshIm("levelset", mls, "all", gf.Integ("IM_TRIANGLE(6)"),
                  gf.Integ("IM_GAUSS_PARALLELEPIPED(2,6)"), gf.Integ("IM_GAUSS_PARALLELEPIPED(2,4)"))
  kept = base.dofs_with_support(mim, 1e-8)
  base.set_qdim(2)
  mf_u = gf.MeshFem("partial", base, np.ravel([[2*i, 2*i + 1] for i in kept]))
  E, nu = 1., 0.3
  md = gf.Model()
  md.add_fem_variable("u", mf_u)
  md.add_initialized_data("lambda", E*nu/((1 + nu)*(1 - 2*nu)))
  md.add_initialized_data("mu", E/(2*(1 + nu)))
  # a weak spring against the rigid motions
  md.add_linear_term(mim, "lambda*Div_u*Div_Test_u + mu*(Grad_u + Grad_u'):Grad_Test_u"
                     " + 1e-9*u.Test_u")
  for rg in [1, 2]:
    md.add_linear_term(mim, "-Normal.Test_u", rg)
  nit, conv, info = md.solve()
  assert conv, info["message"]
  u = md.variable("u")
  K, P = ls.crack_sifs(mim, mf_u, u, E, nu, 0.5)
  assert K.shape == (3, 2) and P.shape == (2, 2)
  k0 = np.sqrt(np.pi*a)
  for k, p in zip(K.T, P.T):
    assert np.allclose(np.abs(p), [c*a, s*a])
    # K_II relative to e2 = grad phi: opposite at the two tips
    sg = np.sign(p[0])
    assert abs(k[0]/(k0*c*c) - 1) < 0.025 and abs(sg*k[1]/(k0*s*c) - 1) < 0.025 and k[2] == 0
  K1, P1 = ls.crack_sifs(mim, mf_u, u, E, nu, 0.5, front=1)
  assert np.array_equal(K1, K[:, 1:]) and np.array_equal(P1, P[:, 1:])


def test_crack_front_slice(tmp_path):
  x = np.linspace(-1, 1, 5)
  m = gf.Mesh("regular_simplices", x, x, x)
  ls = gf.LevelSet(m, 2, "z-0.01", "sqrt(x*x+y*y)-0.5")
  mf_u = gf.MeshFem(m, 3)
  mf_u.set_classical_finite_element(1)
  P = ls.crack_fronts(0.25)[0][0]
  K, sl = ls.crack_sifs(gf.MeshIm(m, 2), mf_u, np.zeros(mf_u.nb_dof()), 1., 0.3, 0.3, 0.25)
  # every front point in the slice, K (0 here) at its points
  Q = sl.pts()
  assert K.shape == (3, sl.nbpts()) and np.all(K == 0)
  assert all(np.linalg.norm(Q - p[:, None], axis=0).min() < 1e-12 for p in P.T)
  assert list(sl.nbsplxs()) == [sl.nbpts(), 0, 0, 0]
  sl.export_to_vtu(str(tmp_path / "K.vtu"), K, "K", ascii=True)
  vtu = open(tmp_path / "K.vtu").read()
  assert 'Name="K"' in vtu and 'NumberOfComponents="3"' in vtu
