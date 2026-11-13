# gf.mpi_init under mpirun (ci/mpi.sh): python3 mpi_poisson.py MODE
#   first: mpi4py imported, then gf.mpi_init (mpi4py's MPI adopted)
#   after: gf.mpi_init, then mpi4py imported (gf's MPI used by mpi4py)
#   alone: gf.mpi_init without mpi4py
#   farm:  mpi4py without gf.mpi_init: each rank its own problem
# A Poisson problem whose P2 solution is exact (x^2 + y^2): the same on
# any number of ranks.
import os
import sys

import numpy as np

mode = sys.argv[1]
if mode == "first":
  from mpi4py import MPI
import getfem_rs as gf
if mode in ("first", "after", "alone"):
  gf.mpi_init()
  gf.mpi_init()  # a second call does nothing
if mode in ("after", "farm"):
  from mpi4py import MPI
nranks = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("PMI_SIZE", 1)))
# the problem of this rank (farm: scaled by the rank)
c = 1.0 + (MPI.COMM_WORLD.Get_rank() if mode == "farm" else 0)

m = gf.Mesh("cartesian", np.linspace(0, 1, 9), np.linspace(0, 1, 9))
m.set_region(1, m.outer_faces())
mf = gf.MeshFem(m)
mf.set_classical_finite_element(2)
mim = gf.MeshIm(m, 4)
md = gf.Model()
md.add_fem_variable("u", mf)
md.add_initialized_data("c", c)
md.add_linear_term(mim, "Grad(u).Grad(Test_u) + 4*c*Test_u")
md.add_initialized_fem_data("g", mf, md.interpolation("c*(sqr(X(1))+sqr(X(2)))", mf))
md.add_Dirichlet_condition_with_multiplier(mim, "u", 2, 1, "g")
it, conv = md.solve(atol=1e-12, rtol=0.)[:2]
assert conv, "Newton"
err = gf.asm_generic(mim, 0, "sqr(u-c*(sqr(X(1))+sqr(X(2))))", -1, md)
assert err < 1e-20, err
# assembled scalars summed over the ranks, the tangent each rank's part
assert abs(gf.asm_generic(mim, 0, "1", -1, md) - 1) < 1e-14
n = md.nb_dof()
md.assembly()
part = md.tangent_matrix().nnz()
# a matrix assembled on several ranks: each rank's part (row sums of the
# mass matrix: summed, the assembled integrals of the basis functions)
Mm = gf.asm_generic(mim, 2, "Test2_u*Test_u", -1, md)
jc, ir = Mm.csc_ind()
rows = np.bincount(ir, weights=Mm.csc_val(), minlength=n)
whole = gf.asm_generic(mim, 1, "Test_u", -1, md)

# linsolve_mumps on the tangent (parts) and the rhs: the Newton step
x = gf.linsolve_mumps(md.tangent_matrix(), md.rhs())
assert np.abs(x).max() < 1e-10, np.abs(x).max()

# eigsh on parts: the lowest Dirichlet mode of the unit square ~ 2 pi^2
iu = md.interval_of_variable("u")
I = range(iu[0], iu[0] + iu[1])
K = gf.Spmat("copy", gf.asm_generic(mim, 2, "Grad(Test2_u).Grad(Test_u)", -1, md), I)
M = gf.Spmat("copy", gf.asm_generic(mim, 2, "Test2_u*Test_u", -1, md), I)
bd = set(mf.basic_dof_on_region(1))
inner = [i for i in range(mf.nb_dof()) if i not in bd]
vals, modes = gf.eigsh(gf.Spmat("copy", K, inner), gf.Spmat("copy", M, inner), nev=2, sigma=0.)
assert abs(vals[0] / (2*np.pi**2) - 1) < 1e-2, vals
# without M (the identity), shifted: K phi = lambda phi
Ki = gf.Spmat("copy", K, inner)
vk, phi = gf.eigsh(Ki, nev=1, sigma=-1.)
jc, ir = Ki.csc_ind()
kv = Ki.csc_val()
kphi = np.zeros(len(inner))  # this rank's part of K phi
for j in range(len(inner)):
  np.add.at(kphi, ir[jc[j]:jc[j+1]], kv[jc[j]:jc[j+1]] * phi[0][j])
# a MUMPS context alive at exit: released before MPI is finalized
ctx = gf.MumpsContext()
ctx.set_matrix(md.tangent_matrix())
ctx.analyze()
ctx.factorize()

if mode != "alone":
  comm = MPI.COMM_WORLD
  assert comm.Get_size() == nranks
  # the same eigenvalues on every rank
  assert comm.allreduce(vals[0], op=MPI.MAX) == comm.allreduce(vals[0], op=MPI.MIN)
  summed = np.zeros(n)
  comm.Allreduce(rows, summed)
  distributed = nranks > 1 and mode != "farm"
  assert np.allclose(summed if distributed else rows, whole, rtol=0, atol=1e-14)
  if distributed:
    assert not np.allclose(rows, whole, rtol=0, atol=1e-14)
    kphi = comm.allreduce(kphi)
  assert np.linalg.norm(kphi - vk[0]*phi[0]) < 1e-10 * vk[0], vk
print(f"{mode} rank ok: n {n}, nnz {part}, lambda1 {vals[0]:.6f}, K {vk[0]:.9f}", flush=True)
