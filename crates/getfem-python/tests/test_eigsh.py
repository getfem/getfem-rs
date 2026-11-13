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


def tridiag(n, a, b):
  K = gf.Spmat("empty", n)
  for i in range(n):
    K.assign(i, i, a)
    if i > 0:
      K.assign(i, i - 1, b)
      K.assign(i - 1, i, b)
  return K


def unit_square_km():
  m = gf.Mesh("cartesian", np.linspace(0, 1, 9), np.linspace(0, 1, 9))
  mf = gf.MeshFem(m)
  mf.set_classical_finite_element(1)
  mim = gf.MeshIm(m, 4)
  md = gf.Model()
  md.add_fem_variable("u", mf)
  K = gf.asm_generic(mim, 2, "Grad_Test2_u.Grad_Test_u", -1, md)
  M = gf.asm_generic(mim, 2, "Test2_u*Test_u", -1, md)
  return K, M


def test_spmat_copy():
  K, _ = unit_square_km()
  a = dense(K)
  assert np.array_equal(dense(gf.Spmat("copy", K)), a)
  I = [5, 3, 40, 0]
  J = [7, 3, 80]
  assert np.array_equal(dense(gf.Spmat("copy", K, I)), a[np.ix_(I, I)])
  assert np.array_equal(dense(gf.Spmat("copy", K, I, J)), a[np.ix_(I, J)])
  with pytest.raises(IndexError):
    gf.Spmat("copy", K, [0, 81])
  with pytest.raises(IndexError):
    gf.Spmat("copy", K, [1, 2, 1])


def test_eigsh_laplacian():
  # smallest eigenvalues of tridiag(2, -1): 2 - 2 cos(k pi/(n+1))
  n = 100
  vals, modes = gf.eigsh(tridiag(n, 2., -1.), nev=4, sigma=0.)
  k = np.arange(1, 5)
  assert np.allclose(vals, 2 - 2*np.cos(k*np.pi/(n + 1)), rtol=1e-12, atol=0)
  assert modes.shape == (4, n)
  assert np.allclose(modes @ modes.T, np.eye(4), atol=1e-12)


def test_eigsh_generalized():
  # Neumann Laplacian on the unit square: 0 (constant mode), then pairs
  K, M = unit_square_km()
  k, m = dense(K), dense(M)
  vals, modes = gf.eigsh(K, M, nev=5, sigma=-1.)
  assert abs(vals[0]) < 1e-10
  assert np.allclose(vals[1], vals[2], rtol=1e-10)
  assert np.allclose(vals[1], np.pi**2, rtol=0.05)
  for lam, phi in zip(vals, modes):
    r = k @ phi - lam*(m @ phi)
    assert np.linalg.norm(r) <= 1e-10*np.linalg.norm(k @ phi) + 1e-12
  assert np.allclose(modes @ m @ modes.T, np.eye(5), atol=1e-10)


def test_eigsh_bad_arguments():
  K = tridiag(10, 2., -1.)
  with pytest.raises(ValueError):
    gf.eigsh(K, nev=10)
  with pytest.raises(ValueError):
    gf.eigsh(K, tridiag(9, 1., 0.))
