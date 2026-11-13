import numpy as np
import pytest

import getfem_rs as gf


def test_singular_max_iter():
  # pure Neumann, incompatible load: no convergence, at most max_iter solves
  m = gf.Mesh("cartesian", np.linspace(0, 1, 5), np.linspace(0, 1, 5))
  mf = gf.MeshFem(m)
  mf.set_classical_finite_element(1)
  md = gf.Model()
  md.add_fem_variable("u", mf)
  md.add_nonlinear_term(gf.MeshIm(m, 2), "Grad_u.Grad_Test_u - Test_u")
  nit, conv, info = md.solve(max_iter=30)
  assert not conv and nit <= 30 and info["status"] == "max_iterations"
  assert info["message"].startswith(f"no convergence in {nit} linear solves: full Newton (")


def test_nonlinear_poisson():
  m = gf.Mesh("cartesian", np.linspace(0, 1, 9), np.linspace(0, 2, 7))
  m.set_region(1, m.outer_faces())
  mf = gf.MeshFem(m)
  mf.set_classical_finite_element(2)
  mim = gf.MeshIm(m, 4)
  md = gf.Model()
  md.add_fem_variable("u", mf)
  md.add_initialized_data("f", 3.0)
  md.add_initialized_data("c", [1.0, 0.5])
  md.add_macro("k", "(1+u*u)")
  md.add_nonlinear_term(mim, "k*Grad(u).Grad(Test_u)")
  md.add_linear_term(mim, "-f*(1+c.X)*Test_u")
  md.add_linear_term(mim, "1e3*(u-X(1)*X(2))*Test_u", 1)
  nit, conv, info = md.solve(atol=1e-10, rtol=0.0)
  assert (nit, conv, info["method"], info["message"]) == (5, True, "newton", "")
  # info: Newton's report
  assert info["status"] == "converged" and info["hint"] is None
  assert info["tolerance"] == 0.0 and info["scale"] == info["residuals_d"][0]
  assert len(info["residuals"]) == nit + 1 and info["residuals"][-1] <= 1e-10
  assert list(info["residuals_per_variable"][0]) == ["u"]
  assert len(info["increments"]) == nit and info["residuals_per_tolerance"][0] == {}
  # GetFEM's Python (max_res 1e-10): 5 iterations, the same sums
  assert md.nb_dof() == 221
  assert md.variable("u").sum() == pytest.approx(166.33105638923928, rel=1e-13)
  v = md.interpolation("u*u+X(1)", mf)
  assert v.shape == (221,)
  # a GWFL expression, interpolated on u's space with the old values
  md.set_variable("u", "u*u+X(1)")
  assert np.array_equal(md.variable("u"), v)
  md.set_variable("u", "2*u")
  assert np.array_equal(md.variable("u"), 2*v)
  # several at once from the old values: a swap
  md.add_initialized_fem_data("z", mf, np.ones(221))
  md.set_variables([("u", "z"), ("z", "u")])
  assert np.array_equal(md.variable("z"), 2*v) and (md.variable("u") == 1).all()
  md.set_variable("u", np.zeros(221))
  assert md.variable("u").sum() == 0.0
  md.add_initialized_data("s", 1.0)
  with pytest.raises(ValueError, match="fixed size"):
    md.set_variable("s", "2")
  with pytest.raises(ValueError):
    md.set_variable("u", "[1;2]")
  with pytest.raises(KeyError):
    md.variable("w")
  with pytest.raises(ValueError):
    md.add_linear_term(mim, "Grad(w).Grad(Test_u)")
  with pytest.raises(TypeError):
    md.solve(max_res=1e-10)
  # the Rust Newton's options: methods, variable tolerances
  md.set_variable("u", np.zeros(221))
  _, conv, info = md.solve(methods=["nleq_err"], atol=float("inf"),
                           variable_tolerances=[("u", 1e-10)])
  assert conv and info["method"] == "nleq_err"
  assert list(info["residuals_per_tolerance"][-1]) == ["[u]"]
  assert info["residuals_per_tolerance"][-1]["[u]"] <= 1e-10
  assert md.variable("u").sum() == pytest.approx(166.33105638923928, rel=1e-10)
  for kw in [{"methods": ["pure"]}, {"variable_tolerances": [(["u"], 1e-8, 1.0)]},
             {"variable_tolerances": [{"u": 1e-8}]}]:
    with pytest.raises(ValueError):
      md.solve(**kw)


def test_data_multiplier_projection_asm():
  # GetFEM's Python gives the same values (its variables ordered by name)
  m = gf.Mesh("cartesian", np.linspace(0, 1, 7), np.linspace(0, 2, 5))
  m.set_region(1, m.outer_faces())
  m.set_region(2, m.outer_faces_in_box([-0.1, -0.1], [1.1, 0.1]))
  mf, mfd, mfdg = gf.MeshFem(m), gf.MeshFem(m), gf.MeshFem(m, 2)
  mf.set_classical_finite_element(2)
  mfd.set_classical_finite_element(1)
  mfdg.set_classical_discontinuous_finite_element(1)
  mim = gf.MeshIm(m, 4)
  imd = gf.MeshImData(mim, -1, [2])
  md = gf.Model()
  md.add_fem_variable("u", mf)
  md.add_initialized_fem_data("g", mfd, 0.1 * np.arange(mfd.nb_dof()))
  md.add_im_data("q", imd)
  md.add_fem_data("z", mfd)
  assert not md.variable("q").any() and not md.variable("z").any()
  md.set_variable("q", md.interpolation("[X(1);X(2)*X(2)]", imd))
  md.add_nonlinear_term(mim, "(1+u*u)*Grad(u).Grad(Test_u) - q.Grad(Test_u) - Test_u")
  mult = md.add_Dirichlet_condition_with_multiplier(mim, "u", 1, 1, "g")
  assert mult == "mult_on_u"
  nit, conv, info = md.solve(atol=1e-11, rtol=0.0)
  assert conv, info["message"]
  x = md.from_variables()
  # insertion order: u, then the multiplier
  assert x[:117].sum() == pytest.approx(209.82605413136176, rel=1e-12)
  assert x[117:].sum() == pytest.approx(-20.653947450258514, rel=1e-12)
  p = md.local_projection(mim, "Grad(u)", mfdg)
  assert p.sum() == pytest.approx(197.17384850606095, rel=1e-12)
  assert gf.asm_generic(mim, 0, "u*u+q(2)", -1, md) == pytest.approx(10.400710618920794, rel=1e-12)
  assert gf.asm_generic(mim, 0, mult, 2, md) == pytest.approx(1.5456218928346899, rel=1e-12)
  v = gf.asm_generic(mim, 1, "Grad(u).Grad(Test_u)", -1, md)
  assert v.shape == (137,)
  assert md.mesh_fem_of_variable(mult).nb_dof() == 137 - 117
  assert md.mesh_fem_of_variable("g").nb_dof() == mfd.nb_dof()
  with pytest.raises(ValueError):
    md.mesh_fem_of_variable("q")
  # u disabled: data, only the multiplier solved for (u's values kept)
  u = md.variable("u").copy()
  md.disable_variable("u")
  assert md.nb_dof() == 20
  md.enable_variable("u")
  assert md.nb_dof() == 137
  assert np.array_equal(md.variable("u"), u)
  md.to_variables(np.zeros(137))
  assert not md.from_variables().any()
  with pytest.raises(ValueError):
    md.to_variables(np.zeros(3))
  with pytest.raises(ValueError):
    gf.asm_generic(mim, 0, "Test_u", -1, md)
  with pytest.raises(ValueError, match="of order 2, more than 1"):
    gf.asm_generic(mim, 1, "Test_u*Test2_u", -1, md)
  with pytest.raises(ValueError, match="order 0, 1 or 2"):
    gf.asm_generic(mim, 3, "Test_u*Test2_u", -1, md)


def test_num_threads():
  assert gf.num_threads() == 1
  gf.set_num_threads(2)
  assert gf.num_threads() == 2
  gf.set_num_threads(1)


def test_set_variable_same_space():
  """a variable on the same mesh_fem copied by values: B-splines (no
  interpolation), all from the old values"""
  m = gf.Mesh("cartesian", np.linspace(0, 1, 5), np.linspace(0, 1, 3))
  mf = gf.MeshFem("bspline_uniform", m, 4, 2, 3)
  n = mf.nb_dof()
  md = gf.Model()
  md.add_initialized_fem_data("x", mf, np.arange(n, dtype=float))
  md.add_fem_data("x0", mf)
  with pytest.raises(ValueError):
    md.set_variable("x0", "2*x")
  md.set_variable("x0", " x ")
  assert (md.variable("x0") == np.arange(n)).all()
  md.set_variable("x0", np.ones(n))
  md.set_variables([("x", "x0"), ("x0", "x")])
  assert (md.variable("x0") == np.arange(n)).all()
  assert (md.variable("x") == 1).all()
  with pytest.raises(ValueError):
    md.set_variables([("x0", "x"), ("x0", "1")])


def test_interval_of_variable_linked_mesh():
  m = gf.Mesh("cartesian", np.linspace(0, 1, 3), np.linspace(0, 1, 3))
  mf = gf.MeshFem(m, 2)
  mf.set_classical_finite_element(1)
  assert mf.linked_mesh().nb_points() == m.nb_points()
  md = gf.Model()
  md.add_fem_variable("u", mf)
  md.add_variable("lam", 3)
  md.add_fem_data("d", mf)
  assert list(md.interval_of_variable("u")) == [0, 18]
  assert list(md.interval_of_variable("lam")) == [18, 3]
  md.disable_variable("u")
  assert list(md.interval_of_variable("lam")) == [0, 3]
  with pytest.raises(ValueError):
    md.interval_of_variable("u")


def test_dirichlet_simplification():
  # -Laplace u = 1, u = 0 (eliminated) and u = 1 by data on the boundary
  m = gf.Mesh("cartesian", np.linspace(0, 1, 5), np.linspace(0, 1, 5))
  m.set_region(1, m.outer_faces())
  mf = gf.MeshFem(m)
  mf.set_classical_finite_element(2)
  mim = gf.MeshIm(m, 4)
  md = gf.Model()
  md.add_fem_variable("u", mf)
  md.add_linear_term(mim, "Grad_u.Grad_Test_u - Test_u")
  md.add_Dirichlet_condition_with_simplification("u", 1)
  md.solve()
  b = mf.basic_dof_on_region(1)
  assert (md.variable("u")[b] == 0).all() and md.variable("u").max() > 0.07
  md2 = gf.Model()
  md2.add_fem_variable("u", mf)
  md2.add_initialized_fem_data("g", mf, np.ones(mf.nb_dof()))
  md2.add_linear_term(mim, "Grad_u.Grad_Test_u")
  md2.add_Dirichlet_condition_with_simplification("u", 1, "g")
  md2.solve()
  assert np.abs(md2.variable("u") - 1).max() < 1e-12


def test_raytracing_transformation():
  # two unit squares 0.01 apart: from the top of the lower one, rays find
  # the bottom of the upper one, the gap 0.01
  def square(y0):
    m = gf.Mesh("structured", "GT='GT_PK(2,1)';ORG=[0,%r];SIZES=[1,1];NSUBDIV=[2,2]" % y0)
    mf = gf.MeshFem(m, 2)
    mf.set_classical_finite_element(1)
    return m, mf
  (m1, mf1), (m2, mf2) = square(0.), square(1.01)
  m1.set_region(1, m1.outer_faces_with_direction([0., 1.], 0.01))
  m2.set_region(1, m2.outer_faces_with_direction([0., -1.], 0.01))
  md = gf.Model()
  md.add_fem_variable("u1", mf1)
  md.add_fem_variable("u2", mf2)
  md.add_raytracing_transformation("ray", 0.1)
  md.add_slave_contact_boundary_to_raytracing_transformation("ray", m1, "u1", 1)
  md.add_master_contact_boundary_to_raytracing_transformation("ray", m2, "u2", 1)
  mim = gf.MeshIm(m1, 2)
  found = gf.asm_generic(mim, 0, "Interpolate_filter(ray, 1, 1)", 1, md)
  gap = gf.asm_generic(mim, 0, "Interpolate_filter(ray, (Interpolate(X,ray)-X).Normal, 1)", 1, md)
  assert found == pytest.approx(1., rel=1e-14) and gap == pytest.approx(0.01, rel=1e-12)
  with pytest.raises(ValueError):
    md.add_master_contact_boundary_to_raytracing_transformation("nope", m2, "u2", 1)


def test_interpolation():
  m = gf.Mesh("cartesian", np.linspace(0, 1, 5), np.linspace(0, 2, 5))
  mf = gf.MeshFem(m)
  mf.set_classical_finite_element(2)
  U = gf.interpolation("X(1)*X(1)+X(2)", mf)
  P = mf.basic_dof_nodes()
  assert np.allclose(U, P[0]**2 + P[1])
  # at points, the fields given (exact: P2)
  pts = np.array([[0.1, 0.33, 0.9], [0.2, 1.7, 0.05]])
  v = gf.interpolation("2*u", pts, m, ("u", mf, U))
  assert np.allclose(v, 2*(pts[0]**2 + pts[1]))
  m.set_region(3, [0])
  assert np.allclose(gf.interpolation("u", mf, 3, ("u", mf, U))[[0, 1]], U[[0, 1]])
  # on im_data
  imd = gf.MeshImData(gf.MeshIm(m, 2), -1)
  assert gf.interpolation("u", imd, ("u", mf, U)).shape == (imd.nb_values(),)
  # a model: md.interpolation at points
  md = gf.Model()
  md.add_fem_variable("u", mf)
  md.set_variable("u", U)
  assert np.allclose(md.interpolation("u", pts, m), v/2)
  with pytest.raises(ValueError):
    gf.interpolation("u", pts, ("u", mf, U))
  # a field onto another space (exact: P2 into P3 on a finer mesh)
  m2 = gf.Mesh("cartesian", np.linspace(0, 1, 3), np.linspace(0, 2, 7))
  mf2 = gf.MeshFem(m2)
  mf2.set_classical_finite_element(3)
  P2 = mf2.basic_dof_nodes()
  assert np.allclose(gf.interpolation(mf, U, mf2), P2[0]**2 + P2[1])


def test_define_gwfl_function():
  m = gf.Mesh("cartesian", np.linspace(0, 1, 3), np.linspace(0, 1, 3))
  mim = gf.MeshIm(m, 4)
  gf.define_gwfl_function("cube_t", 1, "t*t*t")
  for body in ["t*t*t", "t*t"]:  # a redefinition, also the same
    with pytest.raises(ValueError):
      gf.define_gwfl_function("cube_t", 1, body)
  # global: without a model, and in two models
  assert np.isclose(gf.asm_generic(mim, 0, "cube_t(X(1))", -1), 0.25)
  mf = gf.MeshFem(m)
  mf.set_classical_finite_element(1)
  for c in [1., 2.]:
    md = gf.Model()
    md.add_fem_variable("u", mf)
    md.add_nonlinear_term(mim, "(cube_t(u)-%g)*Test_u" % c)
    md.set_variable("u", np.ones(mf.nb_dof()))  # u=0: zero tangent
    md.solve()
    assert np.allclose(md.variable("u"), c**(1/3))


def test_mpi_init_sequential():
  # the sequential build (or one process): nothing, twice; MPI runs in
  # tests/mpi (ci/mpi.sh)
  gf.mpi_init()
  gf.mpi_init()
  m = gf.Mesh("cartesian", [0., 1.], [0., 1.])
  assert gf.asm_generic(gf.MeshIm(m, 1), 0, "1", -1) == 1.
