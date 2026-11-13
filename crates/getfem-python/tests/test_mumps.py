import numpy as np
import pytest

import getfem_rs as gf


def poisson():
  # a linear Poisson problem, Dirichlet by penalty (symmetric positive
  # definite): one Newton step from 0 is the solution
  m = gf.Mesh("cartesian", np.linspace(0, 1, 5), np.linspace(0, 1, 5))
  m.set_region(1, m.outer_faces())
  mf = gf.MeshFem(m)
  mf.set_classical_finite_element(2)
  mim = gf.MeshIm(m, 4)
  md = gf.Model()
  md.add_fem_variable("u", mf)
  md.add_initialized_data("f", 1.0)
  md.add_linear_term(mim, "Grad(u).Grad(Test_u) - f*Test_u")
  md.add_linear_term(mim, "1e6*u*Test_u", 1)
  return md


@pytest.mark.parametrize("sym", [0, 1, 2])
def test_model_tangent(sym):
  md = poisson()
  ctx = gf.MumpsContext(sym)
  ctx.set_matrix(md.tangent_matrix())
  with pytest.raises(ValueError, match="no tangent"):
    ctx.analyze()
  md.assembly()
  n = md.nb_dof()
  assert md.tangent_matrix().size() == (n, n)
  assert 0 < md.tangent_matrix().nnz() < n*n
  ctx.analyze()
  ctx.factorize()
  x = ctx.solve(md.rhs())
  md.solve(atol=1e-12, rtol=0.0)
  assert np.allclose(x, md.variable("u"), rtol=1e-12, atol=1e-14)


def test_view_not_copy():
  # the view follows the model: new values, factorize again
  md = poisson()
  md.assembly()
  ctx = gf.MumpsContext()
  ctx.set_matrix(md.tangent_matrix())
  ctx.analyze()
  ctx.factorize()
  x = ctx.solve(md.rhs())
  md.set_variable("f", [2.0])
  md.assembly()
  ctx.factorize()
  assert np.allclose(ctx.solve(md.rhs()), 2*x, rtol=1e-12)
  assert not hasattr(md.tangent_matrix(), "set")


def test_errors():
  ctx = gf.MumpsContext()
  with pytest.raises(ValueError, match="set_matrix first"):
    ctx.analyze()
  with pytest.raises(TypeError):
    ctx.set_matrix(np.eye(2))
  with pytest.raises(TypeError):
    ctx.set_matrix(poisson())
  with pytest.raises(ValueError):
    gf.MumpsContext(3)
  with pytest.raises(ValueError):
    ctx.set_icntl(0, 1)
  md = poisson()
  md.assembly()
  ctx.set_matrix(md.tangent_matrix())
  ctx.analyze()
  with pytest.raises(ValueError, match="solve before factorize"):
    ctx.solve(md.rhs())
  ctx.set_icntl(14, 40)  # more workspace
  ctx.set_cntl(1, 0.02)  # pivoting threshold
  ctx.factorize()
  x = ctx.solve(md.rhs())
  md.solve(atol=1e-12, rtol=0.0)
  assert np.allclose(x, md.variable("u"), rtol=1e-12, atol=1e-14)


def test_infog():
  md = poisson()
  md.assembly()
  n = md.nb_dof()
  ctx = gf.MumpsContext(1)
  ctx.set_matrix(md.tangent_matrix())
  ctx.analyze()
  ctx.factorize()
  # success, no negative pivots (SPD), entries of the factors, flops
  assert ctx.infog(1) == 0 and ctx.infog(12) == 0
  assert ctx.infog(9) >= md.tangent_matrix().nnz() // 2 and ctx.rinfog(3) > 0
  with pytest.raises(ValueError):
    ctx.infog(81)
  with pytest.raises(ValueError):
    ctx.rinfog(0)
  # also after a failure: a singular matrix
  K = gf.Spmat("empty", n)
  K.assign(0, 0, 1.)
  ctx = gf.MumpsContext()
  ctx.set_matrix(K)
  ctx.analyze()
  with pytest.raises(RuntimeError, match="singular"):
    ctx.factorize()
  assert ctx.infog(1) == -10  # numerically singular
