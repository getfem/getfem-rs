"""The example phase_field (crates/getfem/examples/phase_field.rs) with
GetFEM-rs' Python: phase field fracture of a notched plate (GetFEM's
demo_phase_field.py), u and phi Q2, the history field psi0_max at the
integration points, pulled by a Dirichlet multiplier on the top and
bottom faces. The demo's step control (load step halving, then a
hand-tuned pseudo-dynamic term) is replaced by pseudo-viscous steps on phi
where Newton fails: every load step is reached. The mesh is bilinear (the
demo's affine GT_LINEAR_QK(2) ignores the fourth point of the cells the
notch distorts). simulate(nx, strains, shear, on_step); run as a script:
60 x 60 elements, 100 load steps, the reaction force per step to
phase_field_forces.dat, the final state to phase_field.vtu."""
import sys

import numpy as np

import getfem_rs as gf

LX, LY = 1., 1.
E, nu = 210e3, 0.3
Gc0, ll = 2.7e0, 0.015
B_BOUNDARY, T_BOUNDARY, TB_BOUNDARY = 4, 6, 7


def mesh(NX, NY):
  """the demo's graded mesh, notched by a gap of 1e-5 on y = 0, x < 0"""
  NX = 2*((NX+1)//2)
  NY = 2*((NY+1)//2)
  NX_seed1 = np.linspace(-1., 0., int(NX/3+1))
  NX_seed2 = np.linspace(0., 1., int((NX-NX/3)+1))[1:]
  NY_seed = np.linspace(-1., 1., NY+1)
  X_seed1 = LX/2*(0.2*NX_seed1+0.8*np.sign(NX_seed1)*np.power(np.abs(NX_seed1),1.5))
  X_seed2 = LX/2*(0.6*NX_seed2+0.4*np.sign(NX_seed2)*np.power(np.abs(NX_seed2),1.5))
  X_seed = np.concatenate((X_seed1,X_seed2))
  Y_seed = LY/2*(0.2*NY_seed+0.8*np.sign(NY_seed)*np.power(np.abs(NY_seed),1.5))
  gap = 1e-5
  m = gf.Mesh("empty", 2)
  gt = gf.GeoTrans("GT_QK(2,1)")
  for Y, sign in [(Y_seed[0:NY//2+1], -1.), (Y_seed[NY//2:], 1.)]:
    X, Y = np.meshgrid(X_seed, Y)
    Y = np.where(X < -1e-10, Y + sign*gap/2*(1-sign*Y/(LY/2)), Y)
    # the cells' corners (i, j), (i+1, j), (i, j+1), (i+1, j+1)
    c = [(slice(0, -1), slice(0, -1)), (slice(0, -1), slice(1, None)),
         (slice(1, None), slice(0, -1)), (slice(1, None), slice(1, None))]
    P = np.stack([np.stack([X[s].ravel(), Y[s].ravel()]) for s in c], axis=2)
    m.add_convex(gt, P.reshape(2, -1))
  bottom_faces = m.outer_faces_in_box([-LX/2-1e-5,-LY/2-1e-5],[LX/2+1e-5,-LY/2+1e-5])
  top_faces = m.outer_faces_in_box([-LX/2-1e-5,LY/2-1e-5],[LX/2+1e-5,LY/2+1e-5])
  m.set_region(B_BOUNDARY, bottom_faces)
  m.set_region(T_BOUNDARY, top_faces)
  m.set_region(TB_BOUNDARY, top_faces)
  m.extend_region(TB_BOUNDARY, bottom_faces)
  return m



def solve_step(md, scale):
  """a load step: Newton; where it fails, pseudo-viscous steps on phi from
  the start values (implicit Euler, visc = scale/dtau, phi_n the last
  accepted step), each by full Newton (at most 30 solves): dtau x8 after
  at most 3 solves, x2 after at most 6, /4 after a failure; above 1e4,
  Newton without the term. Returns the pseudo-viscous steps and the linear
  solves."""
  x0 = md.from_variables()
  nit, conv, _ = md.solve(atol=1e-9)
  solves = nit
  if conv:
    return 0, solves
  md.to_variables(x0)
  xn, dtau, k = x0, 1., 0
  while solves < 10000:
    last = dtau > 1e4
    md.set_variable("visc", [0. if last else scale/dtau])
    md.set_variable("phi_n", "phi")
    nit, conv, _ = md.solve(methods=["newton"], max_iter=30, atol=1e-9)
    solves += nit
    k += 1
    if conv and last:
      return k, solves
    if conv:
      xn = md.from_variables()
      dtau *= 8. if nit <= 3 else 2. if nit <= 6 else 1.
    else:
      md.to_variables(xn)
      dtau = 1e4 if last else dtau/4.
  raise RuntimeError("no convergence in %d linear solves" % solves)


def simulate(nx, strains, shear=False, on_step=lambda step, s: None):
  """the plate on nx x nx elements for the given strains, tension (top and
  bottom faces pulled apart) or shear (moved along them); on_step(step, s)
  after each step, s: eps, force (the reaction on the top face), phimax,
  viscous (pseudo-viscous steps, 0: Newton alone), solves. Returns mfu,
  mfphi and the model."""
  load = "eps*[X(2);0]" if shear else "eps*[0;X(2)]"
  m = mesh(nx, nx)
  mfu = gf.MeshFem(m, 2)
  mfu.set_classical_finite_element(2)
  mfphi = gf.MeshFem(m, 1)
  mfphi.set_classical_finite_element(2)
  mim = gf.MeshIm(m, 5)
  mimd1 = gf.MeshImData(mim, -1)
  md = gf.Model()
  md.add_fem_variable("u", mfu)
  md.add_fem_variable("phi", mfphi)
  md.add_im_data("psi0_max", mimd1)
  md.add_initialized_data("kappa", E/(3.*(1.-2.*nu)))
  md.add_initialized_data("mu", E/(2*(1+nu)))
  md.add_initialized_data("Gc0", Gc0)
  md.add_initialized_data("l", ll)
  md.add_initialized_data("eps", 0.)
  md.add_fem_data("dirichlet_data", mfu)
  md.add_macro("damage", "max(1e-7,sqr(1-phi))")
  md.add_macro("psi0", "(0.5*kappa*sqr(Trace(Grad_u))+mu*Norm_sqr((Sym(Grad_u)-Trace(Grad_u)/3*Id(2))))")
  md.add_macro("Gc", "Gc0")
  md.add_macro("sigma", "damage*(kappa*Trace(Grad_u)*Id(2)+2*mu*(Sym(Grad_u)-Trace(Grad_u)/3*Id(2)))")
  md.add_nonlinear_term(mim, "sigma:Grad_Test_u")
  md.add_nonlinear_term(mim, "(-2*(1-phi)*max(psi0_max,psi0)*Test_phi+Gc*(phi/l*Test_phi+l*Grad_phi.Grad_Test_phi))")
  mult = md.add_Dirichlet_condition_with_multiplier(mim, "u", mfu, TB_BOUNDARY, "dirichlet_data")
  # the pseudo-viscous term on the damage only (u stays in equilibrium),
  # off while visc = 0
  md.add_initialized_data("visc", 0.)
  md.add_fem_data("phi_n", mfphi)
  md.add_nonlinear_term(mim, "visc*(phi-phi_n)*Test_phi")
  print("%i displacement DOFs, %i model DOFs" % (mfu.nb_dof(), md.nb_dof()))
  for step, eps in enumerate(strains):
    md.set_variable("eps", [eps])
    md.set_variable("dirichlet_data", load)
    viscous, solves = solve_step(md, Gc0/ll)
    md.set_variable("psi0_max", "max(psi0_max,psi0)")
    force = [-gf.asm_generic(mim, 0, "%s(%i)" % (mult, c), T_BOUNDARY, md) for c in (1, 2)]
    on_step(step, dict(eps=eps, force=force, phimax=max(md.variable("phi")),
                       viscous=viscous, solves=solves))
  return mfu, mfphi, md


if __name__ == "__main__":
  nx = int(sys.argv[1]) if len(sys.argv) > 1 else 60
  steps = int(sys.argv[2]) if len(sys.argv) > 2 else 100
  shear = len(sys.argv) > 3 and sys.argv[3] == "shear"
  comp = 0 if shear else 1  # the reaction shown
  with open("phase_field_forces.dat", "w") as f:
    def on_step(step, s):
      f.write("step=%i eps=%e fR=(%e,%e) phimax=%e viscous=%i solves=%i\n"
              % (step, s["eps"], *s["force"], s["phimax"], s["viscous"], s["solves"]))
      if s["viscous"] > 0:
        print("step %i: %i pseudo-viscous steps (%i linear solves)"
              % (step, s["viscous"], s["solves"]))
      if step % 10 == 0 or step + 1 == steps:
        print("step %i: eps %.4e, force %.3f, max phi %.4f"
              % (step, s["eps"], s["force"][comp], s["phimax"]))
    mfu, mfphi, md = simulate(nx, [2e-4*k for k in range(steps)], shear, on_step)
  mfu.export_to_vtu("phase_field.vtu", mfu, md.variable("u"), "u",
                    mfphi, md.variable("phi"), "phi")
  print("wrote phase_field.vtu and phase_field_forces.dat")
