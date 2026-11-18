import numpy as np
import pytest

import getfem_rs as gf


def test_periodic_cell_with_macro_strain():
  # periodic cell with macro strain DOFs (GetFEM snippet periodicity.py,
  # crates/getfem-model/tests/reduction.rs): E binds each DOF on the + sides
  # to its - side master and adds the macro strain (XX, XY, YX, YY) times
  # the position on both; the new DOFs: the 4 strains and the kept basic
  # DOFs (the cell center removed); RR computed from EE ("auto").
  # Homogeneous material under a macro
  # stress: the solution is affine, u = eps X
  LX, LY = 1., 1.
  m = gf.Mesh("regular_simplices", np.linspace(-LX/2, LX/2, 5),
              np.linspace(-LY/2, LY/2, 5))
  mfu = gf.MeshFem(m, 2)
  mfu.set_classical_finite_element(2)

  prec = 5
  xy = mfu.basic_dof_nodes()[:, ::2]
  xdof_at_location = {}
  for i in range(xy.shape[1]):
    x, y = np.around(xy[:, i], prec)
    xdof_at_location[x, y] = 2*i
  kept_xdofs = []
  slave_master_pairs = []
  for i in range(xy.shape[1]):
    x, y = xy[:, i]
    if min(x-LX/2, y-LY/2) > -1e-5:
      shifts = [(LX, LY), (LX, 0), (0, LY)]
    elif x-LX/2 > -1e-5:
      shifts = [(LX, 0)]
    elif y-LY/2 > -1e-5:
      shifts = [(0, LY)]
    else:
      shifts = []
    master = None
    for xshift, yshift in shifts:
      key = tuple(np.around([x-xshift, y-yshift], prec))
      if key in xdof_at_location:
        master = xdof_at_location[key]
        break
    if master is not None:
      slave_master_pairs.append([2*i, master])
    elif round(x, prec) != 0. or round(y, prec) != 0.:  # center: removed
      kept_xdofs.append(2*i)

  fullsize = mfu.nb_basic_dof()
  newdofs = {}
  newsize = 4  # 0: XX, 1: XY, 2: YX, 3: YY
  for xdof in kept_xdofs:
    newdofs[xdof] = newsize
    newdofs[xdof+1] = newsize+1
    newsize += 2
  EE = gf.Spmat("empty", fullsize, newsize)
  for xdof in kept_xdofs:
    for q in range(2):
      EE.assign(xdof+q, newdofs[xdof+q], 1.)
  for slave_dof, master_dof in slave_master_pairs:
    for q in range(2):
      EE.assign(slave_dof+q, newdofs[master_dof+q], 1.)
  # the macro strain on both DOFs of each pair
  for slave_dof, master_dof in slave_master_pairs:
    for xdof in [slave_dof, master_dof]:
      x, y = xy[:, xdof//2]
      EE.assign(xdof, 0, x)  # XX
      EE.assign(xdof, 2, y)  # YX
      EE.assign(xdof+1, 1, x)  # XY
      EE.assign(xdof+1, 3, y)  # YY

  # RR computed from EE (least squares)
  mfu.set_reduction_matrices("auto", EE)
  assert mfu.nb_dof() == newsize
  assert mfu.extension_matrix().size() == (fullsize, newsize)
  with pytest.raises(ValueError, match="not stored"):
    mfu.reduction_matrix()

  lmbda, mu = 1., 1.
  sigma = np.array([[1., 0.3], [0.3, 0.5]])
  mim = gf.MeshIm(m, 4)
  md = gf.Model()
  md.add_fem_variable("u", mfu)
  md.add_linear_term(mim, f"({lmbda}*Div_u*Id(2)+{mu}*(Grad_u+Grad_u')):Grad_Test_u")
  # macro stress times the cell area on the strains; the rotation
  # (XY = -YX) has no energy: penalized by (XY - YX)^2
  L = np.zeros(newsize)
  L[:4] = [sigma[0, 0], sigma[0, 1], sigma[1, 0], sigma[1, 1]]
  # one explicit rhs and matrix: set (here zero), replaced, removed by None
  md.set_explicit_rhs("u", np.zeros(newsize))
  md.set_explicit_rhs("u", L)
  P = gf.Spmat("empty", newsize)
  md.set_explicit_matrix("u", "u", P)
  md.set_explicit_matrix("u", "u", None)
  with pytest.raises(ValueError, match="unknown variable v"):
    md.set_explicit_rhs("v", None)
  for i, j, v in [(1, 1, 1.), (1, 2, -1.), (2, 1, -1.), (2, 2, 1.)]:
    P.assign(i, j, v)
  md.set_explicit_matrix("u", "u", P)
  nit, conv, info = md.solve()
  assert conv

  # eps = C^-1 sigma (plane, isotropic)
  a, b = lmbda + 2*mu, lmbda
  det = a*a - b*b
  exx = (a*sigma[0, 0] - b*sigma[1, 1]) / det
  eyy = (a*sigma[1, 1] - b*sigma[0, 0]) / det
  exy = sigma[0, 1] / (2*mu)
  eps = np.array([[exx, exy], [exy, eyy]])
  assert np.allclose(md.variable("u")[:4], [exx, exy, exy, eyy], atol=1e-10)
  # u on the basic DOFs (extended by EE): eps X
  U = md.variable("u")
  u = mfu.extend_vector(U).reshape(-1, 2)
  assert np.allclose(u, (eps @ xy).T, atol=1e-10)
  # extend then reduce: back
  assert np.allclose(mfu.reduce_vector(mfu.extend_vector(U)), U, atol=1e-12)
  # an expression in the range of EE: reduced exactly
  md.set_variable("u", "[0.1*X(1);0.2*X(2)]")
  assert np.allclose(md.variable("u")[:4], [0.1, 0., 0., 0.2], atol=1e-12)
