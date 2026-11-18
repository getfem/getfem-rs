import numpy as np
import pytest

import getfem_rs as gf


def dense(K):
  # a gf.Spmat as a dense array, from its CSC storage
  jc, ir = K.csc_ind()
  v = K.csc_val()
  a = np.zeros(K.size())
  for j in range(len(jc) - 1):
    a[ir[jc[j]:jc[j+1]], j] = v[jc[j]:jc[j+1]]
  return a


def unit_square():
  m = gf.Mesh("cartesian", np.linspace(0, 1, 5), np.linspace(0, 1, 5))
  mf = gf.MeshFem(m)
  mf.set_classical_finite_element(1)
  return m, mf, gf.MeshIm(m, 4)


def test_empty_assign():
  K = gf.Spmat("empty", 3, 2)
  assert K.size() == (3, 2) and K.nnz() == 0
  K.assign(2, 0, 1.)
  K.assign(0, 0, 4.)
  K.assign(2, 0, 3.) # overwritten
  K.assign(1, 1, -2.)
  assert K.nnz() == 3
  assert np.array_equal(dense(K), [[4., 0.], [0., -2.], [3., 0.]])
  jc, ir = K.csc_ind()
  assert list(jc) == [0, 2, 3] and list(ir) == [0, 2, 1]
  assert gf.Spmat("empty", 2).size() == (2, 2)
  with pytest.raises(IndexError):
    K.assign(3, 0, 1.)


def test_tangent_read_only():
  m, mf, mim = unit_square()
  md = gf.Model()
  md.add_fem_variable("u", mf)
  md.add_linear_term(mim, "Grad(u).Grad(Test_u)+u*Test_u")
  md.assembly()
  with pytest.raises(ValueError, match="read-only"):
    md.tangent_matrix().assign(0, 0, 1.)


def test_asm_generic_matrix():
  # order 2: the tangent of the same linear term; MUMPS on it
  m, mf, mim = unit_square()
  md = gf.Model()
  md.add_fem_variable("u", mf)
  md.add_linear_term(mim, "Grad(u).Grad(Test_u)+u*Test_u")
  md.assembly()
  K = gf.asm_generic(mim, 2, "Grad_Test2_u.Grad_Test_u+Test2_u*Test_u", -1, md)
  assert np.allclose(dense(K), dense(md.tangent_matrix()), rtol=0, atol=1e-14)
  # the mass matrix applied to 1: the integrals of the basis functions
  M = gf.asm_generic(mim, 2, "Test2_u*Test_u", -1, md)
  assert np.allclose(dense(M).sum(1), gf.asm_generic(mim, 1, "Test_u", -1, md))
  ctx = gf.MumpsContext()
  ctx.set_matrix(K)
  ctx.analyze()
  ctx.factorize()
  b = np.arange(mf.nb_dof(), dtype=float)
  assert np.allclose(dense(K) @ ctx.solve(b), b)


def test_reduction_matrices():
  # all DOFs tied to one: the constants
  m, mf, mim = unit_square()
  n = mf.nb_dof()
  R = gf.Spmat("empty", 1, n)
  E = gf.Spmat("empty", n, 1)
  R.assign(0, 0, 1.)
  for i in range(n):
    E.assign(i, 0, 1.)
  mf.set_reduction_matrices(R, E)
  assert mf.nb_dof() == 1
  md = gf.Model()
  md.add_fem_variable("u", mf)
  md.set_variable("u", [2.])
  assert np.isclose(gf.asm_generic(mim, 0, "u+Norm(Grad(u))", -1, md), 2.)
  with pytest.raises(ValueError):
    mf.set_reduction_matrices(E, R)
  # R E = I checked
  with pytest.raises(ValueError, match="not a left inverse"):
    mf.set_reduction_matrices(R, gf.Spmat("add", E, E))
  # R from E: E'E diagonal, R stored (the mean)
  mf.set_reduction_matrices("auto", E)
  assert mf.reduction_matrix().size() == (1, n)
  assert np.allclose(mf.reduction_matrix().csc_val(), 1./n)
  assert mf.extension_matrix().nnz() == n


def test_interpolate_transformation_from_expression():
  m, mf, mim = unit_square()
  md = gf.Model()
  md.add_fem_variable("u", mf)
  md.set_variable("u", "X(1)")
  md.add_interpolate_transformation_from_expression("mirror", m, m, "[1-X(1);X(2)]")
  assert np.isclose(gf.asm_generic(mim, 0, "Interpolate(u,mirror)*X(1)", -1, md), 1/6)
  # the transformation in a matrix: u(1-x) tested against u
  K = gf.asm_generic(mim, 2, "Interpolate(Test2_u,mirror)*Test_u", -1, md)
  assert np.isclose(dense(K).sum(), 1.)


def test_diag():
  K = gf.Spmat("empty", 4, 3)
  for i, j, v in [(0, 0, 1.), (1, 1, 2.), (2, 2, 3.), (0, 1, 4.), (1, 0, 5.), (2, 1, 7.)]:
    K.assign(i, j, v)
  assert np.array_equal(K.diag(), [1., 2., 3.])
  # E: a column per rank, (i, i+d) at row i
  assert np.array_equal(K.diag([1, -1]), [[4., 0.], [0., 5.], [0., 7.]])


def test_mult_diag():
  K = gf.Spmat("empty", 3, 2)
  for i, j, v in [(0, 0, 1.), (2, 0, 3.), (1, 1, -2.), (2, 1, 1.)]:
    K.assign(i, j, v)
  D = gf.Spmat("diag", np.array([2., 0.5]))
  assert D.size() == (2, 2) and np.array_equal(dense(D), np.diag([2., 0.5]))
  assert gf.Spmat("diag", [1., 0., 3.]).nnz() == 2  # zeros not stored
  # column scaling, the GetFEM idiom Spmat("mult", K, Spmat("diag", 1./B))
  KD = gf.Spmat("mult", K, D)
  assert np.array_equal(dense(KD), dense(K) @ np.diag([2., 0.5]))
  # a product of non-square factors, a cancellation not stored
  L = gf.Spmat("empty", 2, 3)
  for i, j, v in [(0, 0, 1.), (0, 2, 1.), (1, 1, 1.), (1, 2, 2.)]:
    L.assign(i, j, v)
  LK = gf.Spmat("mult", L, K)
  assert np.array_equal(dense(LK), dense(L) @ dense(K))
  N = gf.Spmat("empty", 1, 3)
  N.assign(0, 0, 3.)
  N.assign(0, 2, -1.)
  assert gf.Spmat("mult", N, K).nnz() == 1  # 3*1 - 3 = 0 dropped
  with pytest.raises(ValueError, match="mult"):
    gf.Spmat("mult", K, K)


def test_add_transpose():
  A = gf.Spmat("empty", 3, 2)
  for i, j, v in [(0, 0, 1.), (2, 0, 3.), (1, 1, -2.)]:
    A.assign(i, j, v)
  B = gf.Spmat("empty", 3, 2)
  for i, j, v in [(0, 0, -1.), (1, 0, 4.), (1, 1, 5.)]:
    B.assign(i, j, v)
  S = gf.Spmat("add", A, B)
  assert np.array_equal(dense(S), dense(A) + dense(B))
  assert S.nnz() == 3  # 1 - 1 = 0 not stored
  T = gf.Spmat("transpose", A)
  assert T.size() == (2, 3) and np.array_equal(dense(T), dense(A).T)
  # the operands unchanged; a tangent view as operand
  assert np.array_equal(dense(A), [[1., 0.], [0., -2.], [3., 0.]])
  m, mf, mim = unit_square()
  md = gf.Model()
  md.add_fem_variable("u", mf)
  md.add_linear_term(mim, "Grad(u).Grad(Test_u)")
  md.assembly()
  K = md.tangent_matrix()
  assert np.allclose(dense(gf.Spmat("transpose", K)), dense(K).T)
  assert gf.Spmat("add", K, gf.Spmat("transpose", K)).size() == K.size()
  with pytest.raises(ValueError, match="add: a 3 x 2 matrix plus a 2 x 3"):
    gf.Spmat("add", A, T)
  with pytest.raises(ValueError, match="two matrices"):
    gf.Spmat("add", A)


def test_linsolve_mumps():
  m, mf, mim = unit_square()
  md = gf.Model()
  md.add_fem_variable("u", mf)
  K = gf.asm_generic(mim, 2, "Grad_Test2_u.Grad_Test_u+Test2_u*Test_u", -1, md)
  b = np.arange(K.size()[0], dtype=float)
  x = gf.linsolve_mumps(K, b)
  assert x.shape == b.shape and np.allclose(dense(K) @ x, b, rtol=0, atol=1e-10)
  assert np.allclose(gf.linsolve_mumps(K, b, "sym"), x, rtol=0, atol=1e-10)
  B = np.stack([b, np.ones_like(b)], 1)
  X = gf.linsolve_mumps(K, B)
  assert X.shape == B.shape and np.allclose(dense(K) @ X, B, rtol=0, atol=1e-10)
  with pytest.raises(ValueError):
    gf.linsolve_mumps(K, b[1:])


def test_asm_generic_select_output_and_tangent():
  m, mf, mim = unit_square()
  md = gf.Model()
  md.add_fem_variable("u", mf)
  md.add_fem_variable("p", mf)
  md.set_variable("u", "X(1)+X(2)")
  n = mf.nb_dof()
  # order 2 of a residual: its tangent
  res = "sqr(u)*Test_p + Grad(u).Grad(Test_u)"
  tan = "2*u*Test2_u*Test_p + Grad(Test2_u).Grad(Test_u)"
  K = dense(gf.asm_generic(mim, 2, res, -1, md))
  assert np.allclose(K, dense(gf.asm_generic(mim, 2, tan, -1, md)), rtol=0, atol=1e-14)
  # order 2 of a potential: its Hessian
  H = dense(gf.asm_generic(mim, 2, "sqr(u)*p", -1, md))
  assert np.abs(H).sum() > 0 and np.allclose(H, H.T, rtol=0, atol=1e-14)
  # select_output: the rows of p, the block (p, u)
  iu, ip = md.interval_of_variable("u")[0], md.interval_of_variable("p")[0]
  F = gf.asm_generic(mim, 1, res, -1, md)
  Fp = gf.asm_generic(mim, 1, res, -1, md, select_output="p")
  assert np.array_equal(Fp, F[ip:ip+n])
  Kpu = gf.asm_generic(mim, 2, res, -1, md, select_output=("p", "u"))
  assert np.array_equal(dense(Kpu), K[ip:ip+n, iu:iu+n])
  with pytest.raises(ValueError, match="select_output"):
    gf.asm_generic(mim, 2, res, -1, md, select_output="p")
  with pytest.raises(ValueError, match="not an enabled variable"):
    gf.asm_generic(mim, 1, res, -1, md, select_output="q")


def test_add_data():
  m, mf, mim = unit_square()
  md = gf.Model()
  # a tensor, column-major
  md.add_data("C", [2, 3])
  assert gf.asm_generic(mim, 0, "Norm_sqr(C)", -1, md) == 0.
  md.set_variable("C", [1., 2., 3., 4., 5., 6.])
  assert np.isclose(gf.asm_generic(mim, 0, "C(1,2)+10*C(2,1)", -1, md), 23.)
  md.add_initialized_data("D", [1., 2., 3., 4.], [2, 2])
  assert np.isclose(gf.asm_generic(mim, 0, "D(1,2)", -1, md), 3.)
  md.add_initialized_data("s", 2., 1)  # size 1: a scalar
  assert np.isclose(gf.asm_generic(mim, 0, "s", -1, md), 2.)
