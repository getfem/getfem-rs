# getfem_rs: Python API and GWFL reference

For LLMs writing `getfem_rs` scripts. `getfem_rs` 0.9.0 is the Python module
of GetFEM-rs, a Rust port of the C++ finite element library GetFEM. It
follows GetFEM's Python interface (`import getfem as gf`) but differs in
many places. Everything below was checked against the source
(crates/getfem-python/src, crates/getfem-gwfl/src) and the code examples
were run. When this file and your memory of GetFEM disagree, this file
wins.

## 1. The 20 rules that prevent most errors

1. `import getfem_rs as gf`. Real arithmetic only: `gf.Model()` takes no
   argument (`gf.Model("real")` is a TypeError).
2. No bricks. Every model term is a GWFL string:
   `md.add_linear_term(mim, expr[, region])` or
   `md.add_nonlinear_term(mim, expr[, region])`. No `add_Laplacian_brick`,
   `add_isotropic_linearized_elasticity_brick`, `add_source_term_brick`,
   `add_source_term`, `add_mass_brick`, ...
3. Only three Dirichlet helpers:
   `add_Dirichlet_condition_with_multiplier` (singular, GetFEM:
   `..._with_multipliers`), `add_Dirichlet_condition_with_penalization`,
   `add_Dirichlet_condition_with_simplification`.
4. Renamed methods: `set_classical_finite_element` (not
   `set_classical_fem`), `set_classical_discontinuous_finite_element`,
   `set_finite_element` (not `set_fem`), `set_integration_method` (not
   `set_integ`), `nb_dof()` (not `nbdof()`), `nb_points()` (not
   `nbpts()`), `nb_convexes()` (not `nbcvs()`).
5. Options are Python keywords; GetFEM's string option pairs fail:
   `md.solve(max_iter=50, rtol=1e-10)`, not
   `md.solve("max_res", 1e-10, "max_iter", 50)`. There is no `max_res`,
   no `noisy`, no `lsolver`.
6. `md.solve()` returns `(iterations, converged, info)`. It does not raise
   when Newton fails: test `converged`.
7. `md.assembly(version)` takes `"all"` (default), `"rhs"` or
   `"all_condensed"`, not `"build_all"`/`"build_rhs"`/`"build_matrix"`.
8. `gf.asm_generic(mim, order, expr, region, md=None, select_output=None)`:
   `region` is required (`-1` = all convexes); select_output is a keyword
   (`select_output="u"` or `("u", "p")`), not trailing strings. There is no
   `gf.asm("generic", ...)`.
9. User functions are global: `gf.define_gwfl_function(name, nargs,
   expr, der1="", der2="")` (GetFEM: `gf.asm_define_function`). Redefining
   a name, even identically, is an error (GetFEM ignores it).
10. GWFL has no time: no `timestep`, no `Previous_u`, `Dot_u`,
    `Previous2_u`, no `md.set_time_step`, no time dispatchers. Write the
    scheme yourself with data (`u0`, `dt`) and update it with
    `md.set_variable("u0", "u")`.
11. GWFL has no `^` operator: `pow(x, 2)` or `sqr(x)` or `x*x`. No `%`, no
    comparisons, no `if`; use `Heaviside`, `pos_part`, `max`, `min`.
12. Level set strings are GWFL in `x`, `y`, `z` (GetFEM: muParser):
    `"x*x + y*y - 0.25"`, never `"x^2 + y^2 - 0.25"`.
13. Indices are 0-based in Python (DOFs, convexes, faces, points) and
    1-based inside GWFL (`X(1)`, `u(2)`, `Grad_u(1,2)`).
14. Fixed-size tensor data and `Reshape` are column-major (Fortran order).
    Inside GWFL, `[1,2;3,4]` lists rows (row 1 = 1,2) and `[[1,2],[3,4]]`
    lists columns (column 1 = 1,2).
15. A vector field's DOFs are interleaved: `U[0::2]` x components,
    `U[1::2]` y components (qdim 2, Lagrange).
16. As GetFEM: `md.add_fem_data(name, mf)` and `md.add_im_data(name,
    imd)` start at zero; `md.add_initialized_fem_data(name, mf, value)`
    takes values, im data get theirs by `md.set_variable`.
17. Set up every `MeshFem` (element, qdim) before `md.add_fem_variable`.
    Changing the space afterwards invalidates the variable until
    `md.set_variable(name, new_values_array)`.
18. `md.tangent_matrix()` is a live read-only view (`gf.Spmat`), not a
    copy. Use `gf.Spmat("copy", K)` to keep a snapshot.
19. `MeshIm(m, k)` with an integer k picks an exact-to-degree-k rule per
    element type. Give enough degree: about 2*(fem degree) for stiffness
    and mass terms.
20. `gf.Mesh("regular_simplices", ...)` with an underscore (GetFEM also
    accepts `"regular simplices"`; getfem_rs does not); structured meshes
    are `gf.Mesh("structured", param)`, not `gf.Mesh("import",
    "structured", param)`.

## 2. Quick start

Poisson problem -Δu = 1 on the unit square, u = 0 on the boundary.

```python
import numpy as np
import getfem_rs as gf

# mesh: unit square, 8x8 bilinear quadrilaterals; region 1 = whole boundary
m = gf.Mesh("cartesian", np.linspace(0, 1, 9), np.linspace(0, 1, 9))
m.set_region(1, m.outer_faces())
# Q2 Lagrange, scalar; integration exact to degree 4
mf = gf.MeshFem(m)                 # qdim 1; gf.MeshFem(m, 2) for a 2D vector
mf.set_classical_finite_element(2)
mim = gf.MeshIm(m, 4)
md = gf.Model()                    # no "real" argument
md.add_fem_variable("u", mf)
md.add_initialized_data("f", 1.0)
# -Laplace(u) = f, weak form in GWFL (no bricks)
md.add_linear_term(mim, "Grad_u.Grad_Test_u - f*Test_u")
# u = 0 on region 1, multiplier on a degree 1 Lagrange space
mult = md.add_Dirichlet_condition_with_multiplier(mim, "u", 1, 1)
nit, conv, info = md.solve()       # tuple of 3, never raises on non-convergence
assert conv, info["message"]
u = md.variable("u")               # numpy copy, len mf.nb_dof()
print(nit, mult, mf.nb_dof(), md.nb_dof(), u.max())   # 1 mult_on_u 289 321 0.0737
print(gf.asm_generic(mim, 0, "u", -1, md))           # integral of u
mf.export_to_vtu("u.vtu", u, "u")
```

## 3. Conventions

### Indices, arrays, shapes

- Python side: 0-based. Arrays returned are numpy (`float64` values,
  `int64` indices). Inputs accept numpy arrays, lists or tuples.
- Points and nodes are columns: `m.pts()`, `mf.basic_dof_nodes()`,
  `sl.pts()` are `dim x n`. Point arrays passed in are `dim x n` too
  (`md.interpolation(expr, P, m)`, `m.add_convex(gt, P)`, `m.set_pts(P)`).
- Face lists are `2 x n` int arrays: row 0 convex ids, row 1 local face
  numbers (`m.outer_faces()`, `m.region(id)`; face `-1` in `m.region`
  means the whole convex).
- GWFL side: 1-based indices.
- Tensors (fixed-size data, `Reshape`, `MeshImData` values, interpolated
  tensors, `GlobalFunction.hess`) are column-major. A 2x2 tensor given as
  `[a11, a21, a12, a22]`.
- `md.interpolation(expr, ...)` and `gf.interpolation` return a flat
  vector: for each target point (DOF node, integration point or given
  point) the components of `expr` in column-major order. Interpolating a
  vector expression on a qdim-q space returns one value per DOF.

### Regions

- Region ids are integers >= 0, set with `m.set_region(id, faces_or_cvs)`
  or `m.extend_region`. A region holds whole convexes and/or faces.
- `-1` in `region` arguments means all convexes of the mesh (volume).
- A term on a face region integrates over those faces (`Normal` is the
  outward unit normal there).
- Gmsh import keeps the physical tags as region ids.

### DOFs

- `mf.nb_dof()`: DOFs after reduction (a `"partial"` space or
  `set_reduction_matrices`); basic DOFs before. `basic_dof_nodes`,
  `basic_dof_on_region` index basic DOFs.
- Vector field (qdim q, Lagrange): DOF `q*i + c` is component `c` of node
  `i`; `basic_dof_nodes` has one column per DOF (each component's DOF its
  node).
- Model unknowns: variables in insertion order (multipliers are added
  when their condition is added), internal (condensed) im variables last.
  `md.interval_of_variable(name)` returns `[first, size]`.
  `md.from_variables()`/`md.to_variables(x)` exclude internal variables.

### Data vs variables

- "variable" in method names means variables and data:
  `md.variable(name)` returns a copy of either, `md.set_variable(name,
  v)` sets either.
- Scalar data: `add_initialized_data("c", 2.0)`. A value given as a
  one-element list is a size-1 tensor that GWFL treats as a scalar (index
  it and you get "Bad number of indices").

### Errors

| Python exception | When |
|---|---|
| `ValueError` | unknown name, wrong size or shape, GWFL syntax/semantic error, unsupported feature |
| `KeyError` | `md.variable(name)` of an unknown name |
| `IndexError` | `Spmat.assign`/`Spmat("copy")` index out of range |
| `RuntimeError` | solver failure (singular MUMPS factorization, ...) |
| `OSError` | file not found |
| `TypeError` | wrong argument type, unknown keyword (e.g. `max_res=`) |

GWFL errors print the expression with a caret under the faulty token.

### Threads, GIL, output

- `gf.set_num_threads(n)`: assembly threads for the whole program
  (default 1). `gf.num_threads()`.
- Assembly and solves release the GIL.
- With `OMP_NUM_THREADS=1` set, MUMPS prints a harmless warning to stderr
  ("OMP_NUM_THREADS=1 overridden in MUMPS' calls").
- `md.solve(verbose=True)` prints from Rust: `print(..., flush=True)`
  before it to keep the order. Per residual: `|R| = ..., scaled |R| = ...
  (u ..., p ...)` (scaled |R| is |R|_D), then e.g. `converged by full
  Newton in 4 linear solves`, `full Newton abandoned: ...`, damped steps
  as `damping 1.0e-2, contraction 0.99`.

### MPI

- Needs a module built with the `mpi` feature, run under `mpirun`. The
  default build is sequential; there `gf.mpi_init()` does nothing.
- `gf.mpi_init()` (no GetFEM counterpart): assembly and MUMPS on all
  ranks of `MPI_COMM_WORLD`. Call it once, before building the problem;
  a second call does nothing. mpi4py may be imported before or after
  (one MPI either way). Without `mpi_init`, each rank runs its own
  sequential problem (task farm).
- After `mpi_init` on several ranks: scalars from `asm_generic` order 0
  and assembled vectors are whole on every rank; assembled matrices
  (`md.tangent_matrix()`, `asm_generic` order 2) are each rank's part
  (summed by MUMPS); `gf.linsolve_mumps`, `gf.eigsh` and
  `gf.MumpsContext` take those parts; `b` is whole; the solution is
  returned on all ranks. One assembly thread per rank.

```python
import getfem_rs as gf
from mpi4py import MPI          # optional, for the rank
rank = MPI.COMM_WORLD.Get_rank()
gf.mpi_init()
```

## 4. API reference

Signature notation: `[x]` optional positional, `x=v` keyword with
default, `*` keyword-only after it.

### 4.1 Module functions

| Function | Returns | Semantics |
|---|---|---|
| `gf.asm_generic(mim, order, expr, region, md=None, select_output=None)` | order 0: `float`; 1: 1-D array over all enabled model DOFs; 2: `gf.Spmat` (n x n) | Assemble GWFL `expr` on `region` (-1 all). With `md`, its variables, data, macros, transformations. An expression of lower order than `order` is differentiated (order 1 of a potential: its residual; order 2 of a residual: its tangent; order 2 of a potential: its Hessian). Order higher than `order`: ValueError. `select_output`: a variable name (order 1: its rows) or a pair (order 2: rows of the first, columns of the second). Without `md` only expressions without variables (`"1"`, `"X(1)"`). |
| `gf.interpolation(expr, target, *args)` | 1-D array | `expr` without a model. `target`: a `MeshFem` (values at its basic DOF nodes), a `MeshImData` (at its integration points) or points `P` (dim x n) followed by the mesh `m`. Then optional `region` (int), then fields as tuples `(name, mf_or_imd, U)`. Example: `gf.interpolation("2*w", P, m, ("w", mf, W))`. |
| `gf.interpolation(mf, U, dest)` | 1-D array | `U` on `mf` interpolated onto `dest`: a `MeshFem` (same or another mesh) or a `gf.Slice` (its points). |
| `gf.define_gwfl_function(name, nargs, expr, der1="", der2="")` | None | A GWFL scalar function of `t` (and `u` if nargs 2) for every model and assembly. Derivatives: expressions, function names, or "" (symbolic). A name in use (even the same definition): ValueError. |
| `gf.linsolve_mumps(M, b, sym=None)` | array of `b`'s shape | Solve `M x = b` with MUMPS. `b` a vector or `n x k` matrix (k right-hand sides). `sym="sym"`: M symmetric. |
| `gf.eigsh(K, M=None, nev=6, sigma=0.0)` | `(vals, modes)` | The `nev` eigenvalues nearest `sigma` of `K phi = lambda M phi` (K symmetric, M symmetric positive semi-definite; None: identity), by ARPACK shift-invert with MUMPS. `vals` ascending (nev,); `modes` (nev, n), modes as ROWS, M-orthonormal. SciPy's name and defaults. Needs `nev < n`. |
| `gf.mpi_init()` | None | See MPI above. |
| `gf.set_num_threads(n)`, `gf.num_threads()` | None, int | Assembly threads (default 1). |
| `gf.version()` | str | `"0.9.0"`. |

### 4.2 gf.Mesh

Constructors:

| Call | Result |
|---|---|
| `gf.Mesh("empty", dim)` | empty mesh of dimension dim |
| `gf.Mesh("cartesian", X[, Y[, Z, ...]])` | tensor grid of `GT_LINEAR_QK(d)` cells (1D: `GT_PK(1,1)`); points in Fortran order (x fastest); coordinates must be distinct |
| `gf.Mesh("cartesian Q1", X, Y[, Z])` | same with `GT_QK(d,1)` |
| `gf.Mesh("regular_simplices", X[, Y[, Z]])` | grid split into simplices (`GT_PK(d,1)`), coordinates mapped from X, Y, Z (non-uniform allowed); no "degree", no "noised" options |
| `gf.Mesh("structured", param)` | GetFEM's structured mesh, `param` a string `"GT='GT_QK(2,1)';ORG=[0,0];SIZES=[1,1];NSUBDIV=[4,4];NOISED=0"` (keys GT required, others optional; defaults ORG 0, SIZES 1, NSUBDIV 2; `;` or newline separated). Any GT: `GT_PK(2,1)`, `GT_QK(2,2)`, `GT_PK(3,2)`, ... |
| `gf.Mesh("structured_ball", param)` | disc/ball: keys GT, ORG, SIZES=[r], NSUBDIV=[n_core, n_peel], NOISED, SYMMETRIES (0: whole ball; k: the part kept by k symmetry planes, d: one sector) |
| `gf.Mesh("structured_ball_shell", param)` | shell: SIZES=[r, thickness] (default [1, 0.5]), NSUBDIV default [3, 2], SYMMETRIES |
| `gf.Mesh("import", "gmsh", filename)` | Gmsh ASCII `.msh` 2.x or 4.1; physical tags become region ids. The mesh is 3D (the format's coordinates; GetFEM drops z = 0): a planar mesh is made 2D by `m.transform([[1,0,0],[0,1,0]])` (an info message says so). Nodes are not merged (a crack with duplicated nodes stays open; `m.merge_points()` glues); unused nodes skipped. No access to group names |
| `gf.Mesh("import", "exodus", filename)` | Exodus II, netCDF 64-bit offset format only (other netCDF formats: `nccopy -k '64-bit offset' in.exo out.exo`). The file's dimension (a planar 3D file: `transform`, as gmsh); nodes not merged. Regions: side set k -> face region k, element set k -> convex region k (the same k: one region, a warning), element block k -> convex region 100000 + k (a warning); set ids from 0 to 99999. Node sets ignored (an info message). Element types BAR2/3, TRI3/6, QUAD4/8/9, TETRA4/10, HEX8/20/27, WEDGE6/15/18, PYRAMID5/13/14 (2D elements in a 3D file: shells, sides 1, 2 their faces, edges from side 3) |
| `gf.Mesh("load", filename)` | a GetFEM mesh file |
| `gf.Mesh("prismatic", m, nl[, degree])` | `m` extruded over [0, 1] in `nl` layers |
| `gf.Mesh("clone", m)` | deep copy |

Methods (meshes are shared objects; methods mutate in place):

| Method | Returns | Semantics |
|---|---|---|
| `dim()` | int | dimension of points |
| `nb_points()`, `nb_convexes()` | int | counts |
| `pts()` | (dim, n) | coordinates, one column per point (deleted points NaN) |
| `set_pts(P)` | None | move all points (P dim x n, as `pts()`) |
| `points_of_convex(cv)` | (dim, k) | the convex's geometric nodes |
| `trans_of_convex(cv)` | `GeoTrans` | |
| `add_convex(gt, P)` | int array | add convexes of `GeoTrans gt`, `P` dim x (k*nb_points(gt)), points merged when closer than the merge tolerance; returns the new ids |
| `merge_points(tol=0.0, region=-1)` | int | merge points closer than tol (0: mesh default) |
| `transform(M)` | None | x -> M x (M N' x N; may change the dimension, e.g. `[[1],[0]]` puts a 1D mesh in 2D) |
| `translate(v)` | None | |
| `optimize_structure()` | None | renumber without gaps |
| `outer_faces(dim=None, cvids=None)` | (2, n) | boundary faces of the convexes cvids (all) of dimension dim (all) |
| `inner_faces(cvids=None)` | (2, n) | faces shared by two convexes, each once |
| `outer_faces_in_box(pmin, pmax)` | (2, n) | outer faces with all nodes in the box |
| `outer_faces_with_direction(v, angle)` | (2, n) | outer faces whose outward normal is within `angle` (radians) of v (at one node at least) |
| `set_region(id, cvfids)` | None | region id := convexes (1-D list) or faces (2 rows: convexes, faces; face -1 the whole convex) |
| `extend_region(id, cvfids)` | None | add to region (created if new) |
| `region_subtract(r1, r2)`, `region_merge(r1, r2)`, `region_intersect(r1, r2)` | None | r1 replaced by r1 minus / union / intersection with r2 (a missing region is empty; a whole convex intersected with its faces keeps the faces) |
| `region(id_or_ids)` | (2, n) | the region(s) content, face -1 for whole convexes; union of a list of ids |
| `export_to_vtu(filename, ascii=False)` | None | the mesh alone |
| `export_to_exodus(filename)` | None | Exodus II, the mesh alone (its nodes, no time step); regions as element sets (convexes) and side sets (faces); regions from 100000 as the element blocks when they partition the convexes, one element type each (as imported), else one block per element type |

Missing compared with GetFEM: `nbpts`, `nbcvs`, `save`, `delete_region`,
`faces_from_cvid`, `convex_area`,
`curved`, `triangles grid`, `ptND`, `from string`, `generate`.

### 4.3 gf.GeoTrans, gf.Fem, gf.Integ (descriptors)

`gf.GeoTrans(name)`, `gf.Fem(name)`, `gf.Integ(name)`; methods `name()`,
`dim()` (GeoTrans, Fem). Names as GetFEM's. Checked to work:

- GeoTrans: `GT_PK(n,k)`, `GT_QK(n,k)`, `GT_LINEAR_QK(n)`,
  `GT_PRISM(n,k)`, `GT_PYRAMID(k)`, `GT_Q2_INCOMPLETE(n)`,
  `GT_PRISM_INCOMPLETE_P2`, `GT_PYRAMID_Q2_INCOMPLETE`, `GT_PRODUCT(a,b)`.
- Fem: `FEM_PK(n,k)`, `FEM_QK(n,k)`, `FEM_PK_DISCONTINUOUS(n,k[,alpha])`,
  `FEM_QK_DISCONTINUOUS(n,k)`, `FEM_PK_HIERARCHICAL(n,k)`,
  `FEM_QK_HIERARCHICAL`, `FEM_PK_WITH_CUBIC_BUBBLE(n,k)`,
  `FEM_Q2_INCOMPLETE(n)`, `FEM_PK_PRISM(n,k)`, `FEM_PYRAMID_QK(k)`,
  `FEM_PRODUCT(a,b)`, `FEM_HERMITE(n)`, `FEM_ARGYRIS`, `FEM_MORLEY(n)`,
  `FEM_HCT_TRIANGLE`, `FEM_REDUCED_HCT_TRIANGLE`, `FEM_QUADC1_COMPOSITE`,
  `FEM_P1_NONCONFORMING`, `FEM_RT0(n)`, `FEM_RTK(n,k)`, `FEM_NEDELEC(n)`,
  `FEM_BDMK`, `FEM_HHO`, `FEM_STRUCTURED_COMPOSITE`, ... (GetFEM's list).
- Integ: `IM_TRIANGLE(k)`, `IM_TETRAHEDRON(k)`, `IM_QUAD(k)`,
  `IM_HEXAHEDRON(k)`, `IM_GAUSS1D(k)`, `IM_GAUSS_PARALLELEPIPED(n,k)`,
  `IM_EXACT_SIMPLEX(n)`, `IM_NC(n,k)`, `IM_PRODUCT(a,b)`,
  `IM_STRUCTURED_COMPOSITE(im,k)`, `IM_PYRAMID(im)`,
  `IM_QUASI_POLAR(im,ip[,ip2])`, `IM_SIMPLEX4D(k)`, `IM_NONE`.

### 4.4 gf.MeshFem

Constructors:

| Call | Result |
|---|---|
| `gf.MeshFem(m[, q1[, q2, ...]])` | space on mesh m, q1 x q2 x ... components (default 1); no element yet |
| `gf.MeshFem("partial", mf, dofs)` | mf restricted to its basic DOFs `dofs` (others removed, i.e. zero) |
| `gf.MeshFem("sum", mf1, mf2, ...)` | union of the basis functions |
| `gf.MeshFem("product", pu_mf, global_mf[, pu_dofs])` | functions of pu_mf (at pu_dofs, all if omitted) times global_mf's (XFEM); GetFEM: `set_enriched_dofs` afterwards |
| `gf.MeshFem("levelset", mls, mf)` | mf (qdim 1) enriched by the Heaviside of the level sets of mls (cut) |
| `gf.MeshFem("global function", m[, ls], [f, ...][, q])` | the `gf.GlobalFunction`s f; ls (a crack `LevelSet`) needed for tip functions |
| `gf.MeshFem("bspline_uniform", m, NX[, NY[, NZ]], order[, low...[, high...]])` | uniform B-splines, one interval per element; boundary per direction `"free"` (default), `"periodic"`, `"symmetry"` |

Methods:

| Method | Returns | Semantics |
|---|---|---|
| `set_classical_finite_element(k)` | None | Lagrange degree k on every convex (PK/QK per element type) |
| `set_classical_discontinuous_finite_element(k, alpha=0.0)` | None | discontinuous Lagrange; nodes shrunk toward the centre by alpha |
| `set_finite_element(pf, cvids=None)` | None | `gf.Fem` on convexes cvids (all) |
| `nb_dof()` | int | DOFs (after reduction) |
| `nb_basic_dof()` | int | DOFs before reduction |
| `qdim()`, `set_qdim(*q)` | int, None | components; `set_qdim(2)`, `set_qdim(3, 3)` |
| `basic_dof_nodes()` | (dim, nbasic) | node of each basic DOF |
| `basic_dof_on_region(rg)` | sorted int array | basic DOFs on region id rg, or on convexes/faces given as for `set_region` |
| `dofs_with_support(mim, fraction)` | int array | basic DOFs whose mass diagonal (with mim) > fraction x median: those touching the integration domain (fictitious domain) |
| `set_reduction_matrices(R, E)` | None | reduction R (nb_dof x nbasic) and extension E (nbasic x nb_dof), `gf.Spmat`s; R E = I checked (ValueError). `R = "auto"`: R = (E'E)^-1 E' from E (least squares; exact for vectors in the range of E): stored when no two columns of E share a row (a selection, plain periodicity: the mean of the copies), otherwise not stored, each reduce solving E'E q = E'b by CG (an info line) |
| `reduction_matrix()`, `extension_matrix()` | `gf.Spmat` or None | R, E (None: not reduced); `reduction_matrix()` ValueError when R is not stored (`"auto"` with E'E not diagonal) |
| `extend_vector(V)`, `reduce_vector(V)` | 1-D array | E V (DOFs to basic DOFs), R V (basic DOFs to DOFs; least squares when R is not stored); V itself if not reduced. `reduce_vector(extend_vector(V))` is V |
| `linked_mesh()` | `Mesh` | shared, not a copy |
| `export_to_vtu(filename, *fields, ascii=False)` | None | fields as `[mf2,] U, name, ...` (U on mf2, else on this space; interpolated onto this space). Binary by default |
| `export_to_exodus(filename, *fields, time=0.0, append=False)` | None | Exodus II (netCDF 64-bit offset, not compressed) of the fields `U, name, ...` on this space at time step `time`: one space per file (no `mf2, U, name` as in `export_to_vtu`: ValueError). The file's nodes: continuous Lagrange, per convex the smallest of degree 1, incomplete degree 2, complete degree 2 holding the mesh and this space (a Q1 or incomplete Q2 space on a HEX20 mesh: HEX20, so the mesh reads back the same; a complete Q2 space: HEX27); a discontinuous space: ValueError (use `export_to_vtu`). A space richer than the mesh (Q2 on HEX8: HEX27) prints a warning: a reimport gives an isoparametric mesh of the written elements; a poorer one (Q1 on HEX27) keeps the mesh's nodes, mesh and data read back exactly. Regions as for `Mesh.export_to_exodus`; components `u_x`, `u_y`, `u_z`, tensors `u_xx`, `u_yx`, .... `append=True`: one more time step at the end of the file, only with the same mesh, space and field names (else ValueError, e.g. "fields u_x, u_y expected (got v_x, v_y)"); new fields need a new file |
| `import_from_exodus(filename, name, step=-1)` | `(U, t)` | The nodal field `name` of an Exodus file (components `name_x`, ... for qdim > 1, as exported) at time step `step` (from 0; negative from the end, -1 the last) on this space's DOFs (reduced if the space is), and that step's time `t`. No numbering relied on: each file element is matched to the convex with the same vertex coordinates, each DOF to the element's node at its reference position. The space: Lagrange (continuous or discontinuous) of the file's degree or lower on a mesh of the file's geometry (imported, or the one exported); other spaces: `gf.interpolation(mf, U, mf2)`. ValueError for a missing field or step, a DOF not at a node, an element without its convex |

Missing: `set_fem`, `set_classical_fem`, `nbdof`, `eval`,
`dof_on_region`, `export_to_vtk`, `export_to_pos`, `set_enriched_dofs`.

### 4.5 gf.MeshIm

| Call / method | Semantics |
|---|---|
| `gf.MeshIm(m[, im])` | im a `gf.Integ` or an int degree (an approximate rule exact to that degree for each element type) |
| `gf.MeshIm("levelset", mls, where, im[, im_tip[, im_set]], lattice=2, tau=1e-2)` | integrate on the part of a cut mesh. `where`: `"all"`, `"inside"` (level set < 0), `"outside"` (> 0), `"boundary"` (the zero level set), optionally followed by a set expression of the level sets named a, b, c... in the order added: `"inside(a*b)"`, `"boundary(a+b)"`, `"all(a-b)"` (`*` intersection, `+` union, `-` difference, `!` complement, parentheses; right to left). `im` (Integ) on the sub-simplices of cut elements; `im_tip` at crack tips (None: im); `im_set` on uncut elements (default degree 1; change it with `set_integration_method`). `lattice`: 2D lattice cut (n triangles per direction); `tau`: size tolerance of sub-simplex rules (0: GetFEM's behaviour). Built from the current cut of mls |
| `set_integration_method(im, cvids=None)` | im Integ or degree, on cvids (all). On a "levelset" MeshIm: the rule of the uncut elements, no cvids |

Missing: `set_integ`, `eval`, `convex_area`.

### 4.6 gf.MeshImData

| Call / method | Semantics |
|---|---|
| `gf.MeshImData(mim, region=-1, shape=None)` | one value of `shape` (None: scalar; int n: vector; list: tensor) per integration point of region |
| `nb_points()` | integration points |
| `nb_values()` | points x tensor size (length of the value vector) |
| `tensor_shape()` | list |

### 4.7 gf.Model

Construction: `md = gf.Model()`.

Variables and data:

| Method | Semantics |
|---|---|
| `add_fem_variable(name, mf)` | unknown field on mf |
| `add_filtered_fem_variable(name, mf, region)` | unknown on the basic DOFs of mf on region (typical multiplier) |
| `add_variable(name, sizes)` | fixed-size unknown, sizes an int or a shape list |
| `add_im_variable(name, imd)` | unknown at the integration points of imd |
| `add_internal_im_variable(name, imd)` | same, condensed out of the tangent system (eliminated at element level) |
| `add_initialized_data(name, value, sizes=None)` | scalar (`2.0`), vector (list/array), or with `sizes` a tensor (values column-major) |
| `add_data(name, sizes)` | fixed-size data, zero |
| `add_fem_data(name, mf)` | data on mf's DOFs, zero |
| `add_initialized_fem_data(name, mf, value)` | data on mf's DOFs; value array of length `mf.nb_dof()` |
| `add_im_data(name, imd)` | data at imd's points, zero (values by `set_variable`) |
| `variable(name)` | numpy copy of a variable's or data's values (KeyError if unknown) |
| `set_variable(name, v)` | v an array/list/number of the right length, or a GWFL string interpolated on the variable's space (fem: basic DOF nodes; im: integration points) evaluated with the current values. Fixed-size variables/data need arrays, not strings |
| `set_variables([(name, expr), ...])` | several GWFL updates evaluated with the old values, then set (swaps, Newmark updates) |
| `from_variables()` | all enabled non-internal variables, system order |
| `to_variables(x)` | inverse of `from_variables` |
| `nb_dof()` | size of the system (all enabled variables, internal included) |
| `interval_of_variable(name)` | `[first, size]` (int64 array) of an enabled variable |
| `mesh_fem_of_variable(name)` | `MeshFem` of a fem variable or data |
| `disable_variable(name)`, `enable_variable(name)` | disabled: kept as data with its values, terms testing it dropped; its multipliers untouched (unlike GetFEM) |

Terms and helpers:

| Method | Semantics |
|---|---|
| `add_linear_term(mim, expr, region=-1)` | a GWFL term that must be linear in the variables (ValueError otherwise); cached while the data do not change |
| `add_nonlinear_term(mim, expr, region=-1)` | any GWFL term, reassembled every iteration |
| `add_explicit_matrix(var1, var2, B)` | `B` (`gf.Spmat`, size of var1 x var2) added to the tangent block of var1's equations w.r.t. var2 (residual += B var2); GetFEM's `issymmetric`, `iscoercive` not taken |
| `add_explicit_rhs(var, L)` | `L` added to the right-hand side of var's equations |
| `add_macro(name, expr)` | `name` or `name(a,b)` with parameters |
| `add_Dirichlet_condition_with_multiplier(mim, var, mf_mult, region, data=None)` | var = data on region (data: any GWFL expression of var's shape, e.g. a data name, `"[1;0]"`, `"X"`; None: 0). mf_mult a `MeshFem` or an int degree (Lagrange of var's qdim, its DOFs on region). Returns the multiplier name (`"mult_on_u"`, then `"mult_on_u_1"`, ...). Dependent constraints are a ValueError (see `select_multiplier_dofs_for_variable`) |
| `add_Dirichlet_condition_with_penalization(mim, var, coef, region, data=None)` | term coef (var - data).Test_var on region; data as above |
| `add_Dirichlet_condition_with_simplification(var, region, data=None)` | eliminates var's DOFs on region; data must be fem data on var's own MeshFem (not a constant, not an expression). Not compatible with internal variables |
| `select_multiplier_dofs_for_variable(var, mim, mf_mult, region, constraint="value")` | int array of basic DOFs of mf_mult giving independent constraints (`"value"` or `"normal_derivative"`); use with `gf.MeshFem("partial", mf_mult, dofs)` |
| `add_interpolate_transformation_from_expression(name, source_mesh, target_mesh, expr)` | `Interpolate(.., name)` maps X to expr (GWFL, may use variables) |
| `add_raytracing_transformation(name, release_distance)` | contact transformation |
| `add_master_contact_boundary_to_raytracing_transformation(name, m, disp, region)` | target boundary, displaced by variable disp |
| `add_slave_contact_boundary_to_raytracing_transformation(name, m, disp, region)` | boundary rays start from |

Assembly, solve, post-processing:

| Method | Returns | Semantics |
|---|---|---|
| `solve(*, methods=None, max_iter=None, atol=None, rtol=None, variable_tolerances=[], verbose=False)` | `(iterations, converged, info)` | Newton; see 4.8 |
| `assembly(version="all")` | None | `"all"`: tangent and rhs at the current values; `"rhs"`; `"all_condensed"`: internal variables condensed out |
| `tangent_matrix()` | `Spmat` | read-only live view of the last tangent (n x n) |
| `rhs()` | 1-D array | -R (minus the residual) of the last assembly |
| `interpolation(expr, target[, m][, region])` | 1-D array | expr on a `MeshFem` (basic DOF nodes), `MeshImData` (points), or points P (dim x n) of mesh m; region restricts convexes (others 0). A point outside the mesh: ValueError |
| `local_projection(mim, expr, mf, region=-1)` | 1-D array | element-wise L2 projection on a discontinuous mf (ValueError if continuous) |
| `numerical_continuation_step(parameter, variables, T=None, h=0.0, *, direction=1.0, sing_indicators=None, parametrised_data=None, gamma_max=None, h_init=None, h_max=None, h_min=None, h_inc=None, h_dec=None, max_iter=None, thr_iter=None, max_res=None, max_diff=None, min_cos=None, singularities=None, non_smooth=None, delta_max=None, delta_min=None, thr_var=None, nb_dir=None, nb_span=None, seed=None, noisy=None)` | `(T, h, info)` | one Moore-Penrose continuation step; see 4.9 |

Missing (GetFEM): all `add_*_brick`, `add_source_term`,
`add_Dirichlet_condition_with_multipliers` (plural), the `sizes` of
`add_fem_data` and `add_initialized_fem_data`, `del_macro`,
`set_time_step`, `set_time`, `add_time_dispatcher`, `first_iter`,
`next_iter`, `add_standard_secondary_domain`,
`add_element_extrapolation_transformation`,
`add_elementary_transformation`, `add_assembly_assignments`,
`variable_list`, `brick_list`, `is_symmetric`/`is_coercive` arguments.

### 4.8 Newton (md.solve)

- `methods`: list among `"newton"`, `"nleq_err"`, `"nleq_res"`, tried in
  turn, each from the start values. Default all three. "newton": full steps;
  "nleq_err": Deuflhard error-oriented damping (smooth problems); "nleq_res":
  residual-based (non-smooth, contact). For plasticity and other non-smooth
  problems pass `methods=["newton", "nleq_res"]` (NLEQ-ERR crawls there).
- Convergence: `|R|_D <= rtol * |R0|_D` (relative, diagonally scaled,
  default `rtol=1e-8`), or `|R| <= atol` (default 0: off;
  `atol=float("inf")` makes only `variable_tolerances` count).
- `variable_tolerances`: list of `(variables, atol)`, variables a name or
  a list of names; all must hold: `[("u", 1e-8), (["T", "mult_on_T"],
  1e-6)]`.
- `max_iter`: cap on linear solves, all methods (default 1000).
- A linear problem converges in 1 iteration (also with internal
  variables).
- info dict keys: `method` (that converged, or None), `message` (why not
  converged), `status` (`"converged"`, `"max_iterations"`), `tolerance`,
  `scale`, `hint` (remark on a converged solve, e.g. "converged at
  round-off level: ...", else None), `residuals`, `residuals_d` (per residual
  evaluation), `residuals_per_variable`, `residuals_per_tolerance` (lists
  of dicts), `increments` (per full Newton step).
- Raises `ValueError` for an ill-posed model (e.g. a variable without
  equations) and `RuntimeError` if MUMPS fails.

### 4.9 Continuation (md.numerical_continuation_step)

GetFEM's `gf.ContStruct` + `init_Moore_Penrose_continuation` +
`Moore_Penrose_continuation` as one function. The model holds the point
(variables and the scalar data `parameter`).

- `variables`: the metric, `[(names, scale), ...]` (names a name or a
  list; scale a typical magnitude).
- First call `T=None`: computes the tangent at the model's point, oriented
  by `direction` (+1/-1), returns `h = h_init`, `info["status"] ==
  "initialized"`. Then pass the returned `T` and `h` back each step.
- `T` is `(T_X, T_gamma)`. On `"failed"` the model is back at the start
  point.
- Defaults: `h_init 1e-2, h_max 1e-1, h_min 1e-5, h_inc 1.3, h_dec 0.5,
  max_iter 10, thr_iter 4, max_res 1e-6, max_diff 1e-6, min_cos 0.9,
  singularities 0 (1 limit points, 2 also bifurcations), non_smooth False,
  gamma_max inf, noisy 0`.
- info: `status`, `h0`, `sing_label` (`""`, `"limit point"`, `"smooth
  bifurcation point"`, `"non-smooth bifurcation point"`), `x_sing`,
  `gamma_sing`, `t_sing`, `sing_indicators` (pass to the next step with
  `singularities > 0`), `tau_bp_graph`.
- `parametrised_data=(init, final, current)`: data names, current = (1 -
  gamma) init + gamma final.

```python
import numpy as np
import getfem_rs as gf

# x^3 - 3x - gamma = 0 (on an area-1 square): folds at gamma = +-2
m = gf.Mesh("cartesian", [0., 1.], [0., 1.])
mim = gf.MeshIm(m, 0)
md = gf.Model()
md.add_variable("x", 1)
md.add_initialized_data("gamma", -2.)
md.add_nonlinear_term(mim, "(x*x*x - 3*x - gamma)*Test_x")
md.set_variable("x", [-2.])
opts = dict(h_init=0.1, h_max=0.5, gamma_max=3.)
T, h, info = md.numerical_continuation_step("gamma", [("x", 1.)], **opts)
for k in range(40):
  T, h, info = md.numerical_continuation_step("gamma", [("x", 1.)], T, h, **opts)
  if info["status"] != "converged":
    break
print(md.variable("x"), md.variable("gamma"))   # past both folds
```

### 4.10 gf.Spmat and gf.MumpsContext

`gf.Spmat` (CSC storage; new matrices from the constructor, no in-place
algebra: no `K.add(I, J, V)`, `K.transpose()`, `full`, `to_csc`):

| Call / method | Semantics |
|---|---|
| `gf.Spmat("empty", m[, n])` | m x n (m x m) with no entries |
| `gf.Spmat("copy", K[, I[, J]])` | copy, or sub-matrix K(I, J) (J = I if omitted; distinct indices) |
| `gf.Spmat("mult", A, B)` | the product A B |
| `gf.Spmat("add", A, B)` | the sum A + B (same sizes; exact zeros not stored) |
| `gf.Spmat("transpose", A)` | A transposed (a new matrix) |
| `gf.Spmat("diag", D)` | the diagonal matrix of the vector D |
| `size()` | (rows, cols) |
| `nnz()` | stored entries |
| `assign(i, j, v)` | one entry (ints, not arrays); read-only on a tangent view |
| `csc_ind()` | `(JC, IR)` column pointers (n+1) and row indices |
| `csc_val()` | values in csc_ind order |
| `diag(E=None)` | diagonal (1-D); with a list E of ranks d, (n, len(E)) array of diagonals (entry (i, i+d) at row i) |

To SciPy: `jc, ir = K.csc_ind(); scipy.sparse.csc_matrix((K.csc_val(), ir,
jc), shape=K.size())`.

`gf.MumpsContext(sym=0)` (0 general, 1 SPD, 2 symmetric):
`set_matrix(spmat)` (a `gf.Spmat` only, read at analyze and factorize),
`analyze()`, `factorize()`, `solve(b)` -> array, `set_icntl(i, v)`,
`set_cntl(i, v)` (1-based as the MUMPS guide), `infog(i)`, `rinfog(i)` (of
the last call, also after a failure). A singular matrix:
`RuntimeError` at factorize, `infog(1) == -10`.

### 4.11 Level sets, XFEM

`gf.LevelSet(m, degree, f1=None, f2=None)`:
- values on a Lagrange space of `degree`; f1, f2 a GWFL string in `x`,
  `y`, `z` (= `X(1)`, `X(2)`, `X(3)`) or an array over `ls.mf()` DOFs.
- A secondary function (f2 or `"ws"`/`"with_secondary"`) makes a crack:
  primary = 0 on its plane, secondary < 0 on it.
- `gf.LevelSet(m, 2)` then `ls.set_values(...)`;
  `gf.LevelSet(m, 1, "ws")` then `ls.set_values(v1, v2)`.

| Method | Semantics |
|---|---|
| `values(nls=0)` | primary (0) or secondary (1) values |
| `set_values(v1, v2=None)` | strings or arrays |
| `mf()`, `degree()` | the values' space, its degree |
| `crack_fronts(spacing=None)` | list of `(P, T, closed)`: front points (dim x n), unit tangents (None in 2D: each tip a front of one point) |
| `select_mesh_fem_basic_dofs_near_front(mf, radius)` | basic DOFs of mf within radius of the front |
| `crack_sifs(mim, mf_u, u, E, nu, radius, spacing=None, front=None, lips=None)` | `(K, where)`: K 3 x n (K_I, K_II, K_III) by the interaction integral, linear elasticity, plane strain in 2D. 2D: where = tip points (dim x n); 3D: a `gf.Slice` of front points. lips: `(mim_lips, plus_expr, minus_expr)` lip tractions |

`gf.MeshLevelSet(m)`: `add(ls)`, `sup(ls)`, `adapt()` (cut; call after
adding), `levelsets()`, `linked_mesh()`, `cut_mesh()` (sub-elements of
cut convexes plus uncut convexes; region 0 the faces on a level set),
`crack_tip_convexes(ls=None)` (list of convex ids).

`gf.GlobalFunction(kind, ...)`:
- `("crack", l)`: GetFEM's singular tip function l (0..15; 0..3 the
  linear elastic crack functions).
- `("cutoff", kind, ...)`: `"none"`, `"exponential", radius`,
  `"polynomial", inner, outer` (C1), `"polynomial2", inner, outer` (C2),
  `"hat", left, centre, right` (GetFEM: an index and 3 numbers).
- `("angular_hat", n, m, periodic)`.
- `("levelset", ls, f)`: tip function f on crack ls, a function of the point.
- `("front hat", P, T, closed, k, radius[, "linear"|"cubic"])`.
- `("product", f, g)`, `("add", f, g)`, or `f*g`, `f+g` (same kinds).
- `val(*x)`, `grad(*x)`, `hess(*x)` (column-major); tip functions take (x, y)
  in the tip frame.

XFEM pattern (from the examples):

```python
import numpy as np
import getfem_rs as gf

nx = 8
x = -0.5 + np.arange(nx + 1)/nx
m = gf.Mesh("regular_simplices", x, x)
m.set_region(1, m.outer_faces())
ls = gf.LevelSet(m, 1, "y", "x")          # crack y = 0, x < 0
mls = gf.MeshLevelSet(m)
mls.add(ls)
mls.adapt()
pre = gf.MeshFem(m)
pre.set_classical_finite_element(1)
mfls = gf.MeshFem("levelset", mls, pre)    # Heaviside enrichment
ck = [gf.GlobalFunction("crack", i) for i in range(4)]
cutoff = gf.GlobalFunction("cutoff", "polynomial2", 0.01, 0.4)
sing = gf.MeshFem("global function", m, ls, [f*cutoff for f in ck])
mf_u = gf.MeshFem("sum", sing, mfls)
mf_u.set_qdim(2)
mim = gf.MeshIm("levelset", mls, "all", gf.Integ("IM_STRUCTURED_COMPOSITE(IM_TRIANGLE(6),3)"),
                gf.Integ("IM_STRUCTURED_COMPOSITE(IM_GAUSS_PARALLELEPIPED(2,6),9)"),
                gf.Integ("IM_STRUCTURED_COMPOSITE(IM_TRIANGLE(6),5)"))
md = gf.Model()
md.add_fem_variable("u", mf_u)
md.add_initialized_data("lambda", 1.)
md.add_initialized_data("mu", 1.)
md.add_linear_term(mim, "lambda*Div_u*Div_Test_u + mu*(Grad_u + Grad_u'):Grad_Test_u")
md.add_linear_term(mim, "-[0;1].Test_u*sign(X(2))", 1)   # pull the lips apart
md.add_linear_term(mim, "1e-6*u.Test_u")                  # fix rigid motions
nit, conv, info = md.solve()
assert conv, info["message"]
K, tips = ls.crack_sifs(mim, mf_u, md.variable("u"), 2.5, 0.25, 0.2)
print(K[:, 0], tips[:, 0])
# view: displacement on the cut mesh, discontinuous across the crack
mfv = gf.MeshFem(mls.cut_mesh(), 2)
mfv.set_classical_discontinuous_finite_element(1, 0.001)
mfv.export_to_vtu("crack.vtu", gf.interpolation(mf_u, md.variable("u"), mfv), "u")
```

### 4.12 gf.Slice

`gf.Slice(op, source, nrefine[, region])`:
- `source`: a `Mesh`; a `MeshFem` followed by `U` (mesh deformed by U
  first); or a `MeshLevelSet` (cut convexes along the level sets).
- `nrefine`: refinements per convex; region: an id or convexes/faces.
- `op` tuple: `("none",)`, `("planar", orient, p, n)`, `("ball", orient,
  c, r)`, `("cylinder", orient, p1, p2, r)`, `("isovalues", orient, mf, U,
  s)`, `("boundary"[, op])`, `("explode", coef)`, `("union", op1, ...)`,
  `("intersection", op1, ...)`, `("diff", op1, op2)`, `("comp", op)`.
  orient: -1 inside (planar: the half-space (x - p).n <= 0, n pointing
  outward; isovalues: U < s), 0 the boundary, 1 outside, 2 both (split). No `"mesh"`, `"points"`,
  `"streamlines"`, `"load"`, no slice of a slice.
- Methods: `dim()`, `nbpts()`, `pts()` (dim x nbpts), `nbsplxs(d=None)`,
  `splxs(d)` -> `(S, cv2s)`, `cvs()`, `linked_mesh()`,
  `export_to_vtu(filename, *fields, ascii=False)` with fields `mf, U,
  name` (interpolated) or `U, name` (U at the slice points, q x nbpts or
  flat).
- `gf.interpolation(mf, U, sl)`: U at the slice points.

## 5. Differences from GetFEM's Python interface

| GetFEM (`import getfem as gf`) | getfem_rs (`import getfem_rs as gf`) |
|---|---|
| `gf.Model("real")` | `gf.Model()` |
| `md.add_Laplacian_brick(mim, "u")` | `md.add_linear_term(mim, "Grad_u.Grad_Test_u")` |
| `md.add_isotropic_linearized_elasticity_brick(mim, "u", "lambda", "mu")` | `md.add_linear_term(mim, "lambda*Div_u*Div_Test_u + mu*(Grad_u + Grad_u'):Grad_Test_u")` |
| `md.add_source_term_brick(mim, "u", "f"[, region])` | `md.add_linear_term(mim, "-f.Test_u"[, region])` (minus: terms are residuals) |
| `md.add_mass_brick(mim, "u", "rho")` | `md.add_linear_term(mim, "rho*u.Test_u")` |
| `md.add_nonlinear_elasticity_brick(mim, "u", "SaintVenant Kirchhoff", "params")` | `md.add_nonlinear_term(mim, "Saint_Venant_Kirchhoff_potential(Grad_u, params)")` |
| `md.add_finite_strain_elasticity_brick(...)` | `add_nonlinear_term` with a `<law>_potential` |
| `md.add_linear_generic_assembly_brick(mim, expr)` / `add_linear_term(mim, expr, region, is_sym, is_coer)` | `md.add_linear_term(mim, expr, region)` |
| `md.add_nonlinear_generic_assembly_brick(...)` | `md.add_nonlinear_term(mim, expr, region)` |
| `md.add_Dirichlet_condition_with_multipliers(mim, "u", mf, rg[, data])` | `md.add_Dirichlet_condition_with_multiplier(mim, "u", mf, rg[, data])` |
| `md.solve("max_res", 1e-9, "max_iter", 100, "noisy")` | `md.solve(atol=1e-9, max_iter=100, verbose=True)` |
| `md.solve()` -> `(nit, conv)` | `md.solve()` -> `(nit, conv, info)` |
| `md.assembly("build_all")`, `"build_rhs"`, `"build_matrix"` | `md.assembly("all")`, `"rhs"`, (none: `"all"`) |
| `md.tangent_matrix()` (copy) | `md.tangent_matrix()` (live read-only view) |
| `md.nbdof()` | `md.nb_dof()` |
| `md.set_time_step(dt)`, `timestep`, `Previous_u`, `Dot_u`, time dispatchers | data `dt`, `u0`; `md.set_variable("u0", "u")` |
| `md.del_macro(name)` | none |
| `gf.asm_define_function(name, n, expr)` | `gf.define_gwfl_function(name, n, expr)` |
| `gf.asm("generic", mim, order, expr, rg, md, "select_output", "u")` / `gf.asm_generic(..., "select_output", "u")` | `gf.asm_generic(mim, order, expr, rg, md, select_output="u")` |
| `gf.asm_generic(mim, 0, expr, rg, md, "Secondary_domain", ...)` | not supported |
| `gf.compute_interpolate_on(mf, U, dest)` | `gf.interpolation(mf, U, dest)` |
| `gf.compute_gradient(mf, U, mf2)` / `mf.eval(expr)` | `md.interpolation("Grad_u", mf2)`, `gf.interpolation(expr, mf)` |
| `gf.ContStruct(md, ...)` + `cs.Moore_Penrose_continuation(...)` | `md.numerical_continuation_step(...)` |
| `gf.Mesh("regular simplices", X, Y)` | `gf.Mesh("regular_simplices", X, Y)` |
| `gf.Mesh("import", "structured", param)` | `gf.Mesh("structured", param)` |
| `m.nbpts()`, `m.nbcvs()` | `m.nb_points()`, `m.nb_convexes()` |
| `m.export_to_vtk(f)` | `m.export_to_vtu(f)` |
| `mf.set_fem(f)` | `mf.set_finite_element(f)` |
| `mf.set_classical_fem(k)` | `mf.set_classical_finite_element(k)` |
| `mf.set_classical_discontinuous_fem(k, a)` | `mf.set_classical_discontinuous_finite_element(k, a)` |
| `mf.nbdof()` | `mf.nb_dof()` |
| `mf.reduction_matrices(R, E)` | `mf.set_reduction_matrices(R, E)` |
| `ctx.set_ICNTL(i, v)`, `ctx.CNTL(i, v)` (MumpsContext) | `ctx.set_icntl(i, v)`, `ctx.set_cntl(i, v)` |
| `mf.export_to_vtk(f, U, "u")` | `mf.export_to_vtu(f, U, "u")` |
| `mf.set_enriched_dofs(dofs)` after `gf.MeshFem("product", pu, g)` | `gf.MeshFem("product", pu, g, dofs)` |
| `mim.set_integ(im)` | `mim.set_integration_method(im)` |
| `gf.LevelSet(m, 2, "x^2+y^2-1")` (muParser) | `gf.LevelSet(m, 2, "x*x+y*y-1")` (GWFL) |
| `gf.GlobalFunction("cutoff", 2, r0, r1, 0)` | `gf.GlobalFunction("cutoff", "polynomial2", r0, r1)` |
| `mls.crack_tip_convexes()` | same, plus `crack_tip_convexes(ls)` |
| `K.assign(I, J, V)` with arrays | `K.assign(i, j, v)` one entry |
| `gf.Spmat("mult", A, B)`, `gf.Spmat("add", A, B)`, `gf.Spmat("diag", D)` | same (`diag`: no sub-diagonal argument E) |
| `K.transpose()` (in place) | `K = gf.Spmat("transpose", K)` (a new matrix) |
| `K.add(I, J, V)`, `K.full()`, `K.to_csc()` | none: `csc_ind`/`csc_val` to numpy or SciPy |
| `gf.Mesh("import", "gmsh", f)`: z dropped when 0, coincident nodes merged | 3D always (`m.transform` for 2D), nodes not merged |
| Exodus (pull request 68, not merged): `mf.export_to_exodus(f, U, "u", "time", t, "append")` | `mf.export_to_exodus(f, U, "u", time=t, append=True)`, one space per file |
| Exodus (pull request 68): `m2.exodus_nodal_data(f, "u", step)` (per mesh point) | `U, t = mf.import_from_exodus(f, "u", step)` (on a space, with the step's time) |
| `md.add_variable(name, size)` | same; sizes may be a shape |
| errors: one `RuntimeError` | `ValueError`/`KeyError`/`RuntimeError`/... (3) |

Unchanged (same names and argument order): `add_fem_variable`,
`add_filtered_fem_variable`, `add_variable`, `add_im_variable`,
`add_internal_im_variable`, `add_initialized_data`, `add_data`,
`add_macro`, `add_Dirichlet_condition_with_penalization`,
`add_Dirichlet_condition_with_simplification`,
`add_interpolate_transformation_from_expression`, raytracing methods,
`variable`, `set_variable`, `from_variables`, `to_variables`,
`interval_of_variable`, `mesh_fem_of_variable`, `interpolation`,
`local_projection`, `rhs`,
`gf.linsolve_mumps`, mesh region methods, `basic_dof_nodes`,
`basic_dof_on_region`, `gf.Slice`, `gf.MeshImData`, descriptor names.

## 6. GWFL (Generic Weak Form Language)

A term is a GWFL string assembled over the integration points of `mim`
on `region`. Its order: 0 (no test function: a potential/scalar), 1
(`Test_u`: a residual/weak form), 2 (`Test_u` and `Test2_v`: a bilinear
form/tangent). In a model, any order is accepted: a potential is
differentiated once for the residual and twice for the tangent; a residual
once. Conventions: residual R(u) = 0, so loads enter with a minus sign:
`"Grad_u.Grad_Test_u - f*Test_u"`.

### 6.1 Lexical

- Numbers: `3`, `1.5`, `.5`, `2.`, `1e-6`, `1.5e+1`.
- Names: letter or `_` then letters, digits, `_`; case sensitive.
  Reserved: function and operator names, `X`, `Normal`, `pi`, `Id`, ...,
  and prefixes `Test_`, `Test2_`, `Grad_`, `Hess_`, `Div_`,
  `Derivative_` (cannot start a variable/data name).
- Whitespace and newlines are free. No comments.
- Not in GWFL: `^`, `**`, `%`, comparisons, `if`, `Print` (refused),
  `timestep`, `Previous_*`, `Dot_*`, `Secondary_domain`.

### 6.2 Operators and precedence

Lowest to highest: `+ -` (binary); then `* / : . .* ./ @` (same level,
left to right: `2/2*3 = 3`); then unary `-` and postfix `'`.
Parentheses as usual. Index access `expr(i, j)` binds to the preceding
term.

| Operator | Meaning | Shapes |
|---|---|---|
| `a + b`, `a - b` | sum | same shape |
| `a * b` | scalar product; matrix-vector; matrix-matrix; 4th-order-tensor * matrix | `A*v`, `A*B`; `v*A` is refused ("Unauthorized multiplication"): use `v.A` or `v'*A` |
| `a / s` | division by a scalar | |
| `a . b` | contract last index of a with first index of b | `v.w` scalar; `A.v` = `A*v`; `v.A` = A^T v; `T3.v` (n,n) |
| `a : b` | contract the last two indices of a with the first two of b | `A:B` Frobenius (scalar); `T4:A` matrix |
| `a .* b`, `a ./ b` | component-wise | same shape |
| `a @ b` | tensor product | `v@w` matrix; `A@v` order 3 |
| `-a` | negation | |
| `a'` | transpose; vector -> 1 x n row; order > 2: swap the first two indices | |
| `Cross_product(v, w)` | 3D vectors | (3) |
| `Contract(A, i, j)` | contract indices i, j of A | `Contract(A,1,2)` = `Trace(A)` |
| `Contract(A, i, B, j)` | contract A's i with B's j | `Contract(v,1,w,1)` = `v.w` |
| `Contract(A, i, j, B, k, l)` | double contraction | `Contract(A,1,2,B,1,2)` = `A:B` |
| `Swap_indices(A, i, j)` | | `Swap_indices(A,1,2)` = `A'` |
| `Index_move_last(A, i)` | index i moved last | |
| `Reshape(A, n1, n2, ...)` | same components, new shape (column-major) | |
| `Trace(M)`, `Deviator(M)`, `Sym(M)`, `Skew(M)` | linear matrix operators (allowed on test functions) | Deviator = M - Trace(M) Id/n |

### 6.3 Tensor literals and indexing

- Vector: `[a, b, c]` or `[a; b; c]` (both column vectors, shape (3)).
- Matrix by rows: `[a, b; c, d]` = rows (a b) and (c d).
- Matrix by columns: `[[a, c], [b, d]]` = the same matrix (inner brackets
  are columns).
- Higher order: nested brackets, or `Reshape([...], n1, n2, n3)`
  (column-major).
- Row vector: `[a, b]'` (1 x 2).
- Elements may be expressions; with test functions all elements must
  carry the same test function (`[Test_u(1); Test_p]` refused).
- Indexing: 1-based, constant indices: `v(2)`, `A(1,2)`, `T(1,2,1)`,
  `A(1,:)` (row), `A(:,2)` (column), `Grad_u(1,:)`. `[1,2;3,4](1,2)` = 2.

### 6.4 Variables, test functions, derivatives

For a variable or data `u` (fem, im, or fixed size) of value shape `S`
(scalar `()`, vector `(q)`, tensor `(q1,q2)`), mesh dimension `d`:

| Expression | Shape | Notes |
|---|---|---|
| `u` | S | value |
| `Grad_u` = `Grad(u)` | S + (d) | scalar: (d); vector: (q, d), `Grad_u(i,j)` = du_i/dx_j |
| `Hess_u` = `Hess(u)` | S + (d, d) | fem only |
| `Div_u` = `Div(u)` | () | vector field with q = d only |
| `Test_u`, `Test2_u` | S | first/second order test functions (variables only, not data) |
| `Grad_Test_u`, `Hess_Test_u`, `Div_Test_u` | as above | also with `Test2_` |
| `u(i)`, `Grad_u(i,j)` | | components |

Rules:
- An expression must be linear in its test functions; products of two
  `Test_u` are refused; use `Test2_v*Test_u` for bilinear forms.
- Test functions cannot be arguments of scalar functions or nonlinear
  operators (`Norm(Test_u)` refused); linear operators (`Trace`, `Sym`,
  `'`, `Reshape`, contractions) accept them.
- Fixed-size variables have no `Grad_`/`Hess_`. im data/variables have no
  `Grad_`.
- Disabled variables act as data (no `Test_`).

Explicit derivatives:
- `Diff(expr, u)`: derivative w.r.t. variable u in the direction `Test_u`
  (adds one test order). `Diff(u.u, u)` = `2*(u.Test_u)`.
- `Diff(expr, u, dir)`: directional derivative in direction `dir` (an
  expression of u's shape, may be `Test_u` or another variable).
- `Grad(expr)`: spatial gradient of an expression, shape + (d); limited to
  second derivatives of shape functions.
- `Derivative_f(t)`: derivative of a scalar function f (built-in, e.g.
  `Derivative_sin(0)` = 1, or a user function of the same model);
  `Derivative_2_f(t, u)` w.r.t. the second argument.

Simplification (not GetFEM's): in a product, scalar factors with the same
base are merged, also through `/`, `-`, `'`, `:`, `.`, `@`, `.*` (not
through sums or functions): `x*(1/x)` and `x/x` -> 1, `pow(x,a)*pow(x,b)`
-> `pow(x,a+b)` (an exponent within a few ulps of an integer rounded),
`pow(x,1)` -> x, `pow(x,0)` -> 1; before derivation, so `Diff` adds no
`1/x - x/x^2` terms. A 0/0 there gives 1, not NaN.
The derivatives of `log`, `sqrt`, `pow` are expressions (`1/x`,
`0.5/sqrt(x)`, `a*pow(x,a-1)`, `pow(x,a)*log(abs(x))`), so they take part
in this simplification and can be derived any number of times.

### 6.5 Predefined values

| Name | Shape | Meaning |
|---|---|---|
| `X` | (d) | current point (real element); `X(1)`, `X(2)`, `X(3)` |
| `Normal` | (d) | outward unit normal on a face region; on a level-set boundary, along the level set gradient; zero vector inside volumes (no error) |
| `element_size` | () | element diameter estimate |
| `element_K` | (d, P) (vector (d) when P = 1) | gradient of the geometric transformation, P reference dimension |
| `element_B` | (P, d) | transposed pseudo-inverse of element_K |
| `pi` | () | |
| `meshdim` | () | d (usable as a constant: `Id(meshdim)`) |
| `Id(n)` | (n, n) | identity, n a constant expression |
| `qdim(u)` | () | total size of u's value |
| `qdims(u)` | vector | sizes of u's value |

### 6.6 Scalar functions

Applied component-wise to tensors. Two-argument functions need arguments
of the same size: no scalar broadcast (`max(v, 0)` with a vector v is
refused, as GetFEM; use `max(v, [0;0])` or `pos_part(v)`).

| Function | Args | Value |
|---|---|---|
| `sqr(t)`, `sqrt(t)`, `exp(t)`, `log(t)`, `log10(t)` | 1 | |
| `pow(t, u)` | 2 | t^u |
| `sin cos tan asin acos atan` | 1 | |
| `atan2(t, u)` | 2 | angle of (u, t), i.e. atan2(y=t, x=u) |
| `sinh cosh tanh asinh acosh atanh` | 1 | |
| `sinc(t)` | 1 | sin(t)/t |
| `Heaviside(t)` | 1 | 0 for t < 0, 1 for t >= 0 |
| `sign(t)` | 1 | -1 for t < 0, 1 for t >= 0 |
| `abs(t)` | 1 | |
| `pos_part(t)`, `neg_part(t)` | 1 | max(t,0), max(-t,0) |
| `sqr_pos_part`, `sqr_neg_part` | 1 | squares of the above |
| `half_sqr_pos_part`, `half_sqr_neg_part` | 1 | halves of the squares |
| `reg_pos_part(t, eps)` | 2 | C1 regularized pos_part |
| `max(t, u)`, `min(t, u)` | 2 | |
| `erf`, `erfc` | 1 | NOT SUPPORTED in getfem_rs (error "unsupported") |

All have derivatives (symbolic); non-smooth ones have the usual
generalized ones. `sqrt(-1)` is NaN, `log(0)` -inf (no error); `1/0` in a
constant expression is an error.

### 6.7 Nonlinear operators

Arguments: scalars, vectors or matrices without test functions.
Derivatives are automatic. `Expm`, `Logm`, `Normalized`, `Normalized_reg`,
`Ball_projection`, `Transformed_unit_vector` have first derivatives only:
not usable inside a potential (order 0 term derived twice); use them in
residual (order 1) terms. `Inv`, `Det`, `Norm` and the others have both.

| Operator | Argument | Result |
|---|---|---|
| `Norm(v)`, `Norm_sqr(v)` | vector or matrix | () (Euclidean / Frobenius) |
| `Normalized(v)` | vector/matrix | same shape |
| `Normalized_reg(v, eps)` | vector/matrix, scalar | v/sqrt(|v|^2+eps^2) |
| `Ball_projection(v, r)` | vector/matrix, scalar | projection on the ball of radius r |
| `Det(M)`, `Matrix_i2(M)`, `Matrix_j1(M)`, `Matrix_j2(M)` | square | () |
| `Inv(M)`, `Expm(M)`, `Logm(M)` | square | same |
| `Right_Cauchy_Green(F)` = F'F, `Green_Lagrangian(F)` = (F'F - I)/2 | matrix (m, n) | (n, n) |
| `Left_Cauchy_Green(F)` = F F' | (m, n) | (m, m) |
| `Transformed_unit_vector(Grad_u, n)` | (d,d), (d) | deformed unit normal (I+Grad_u)^-T n normalized |

Not ported (error "not ported yet"): `Von_Mises_projection`,
`Cauchy_stress_from_PK2`, `Coulomb_friction_coupled_projection`.

Hyperelastic laws: `<Law>_potential(Grad_u, params)` (scalar strain
energy), `<Law>_PK2(Grad_u, params)` and its alias `<Law>_sigma` (second
Piola-Kirchhoff stress, (d,d)); first argument is `Grad_u` (not F, not E).
Prefix `Plane_Strain_` for 2D plane strain. Laws other than SVK and
Ciarlet_Geymonat exist only in 3D (2D through `Plane_Strain_`).

| Law | params |
|---|---|
| `Saint_Venant_Kirchhoff` | [lambda; mu] |
| `Ciarlet_Geymonat` | [lambda; mu; gamma] |
| `Generalized_Blatz_Ko` | [a; b; c; d; n] |
| `Incompressible_Mooney_Rivlin` | [C1; C2] |
| `Compressible_Mooney_Rivlin` | [C1; C2; D1] |
| `Incompressible_Neo_Hookean` | [C1] |
| `Compressible_Neo_Hookean` | [C1; D1] |
| `Compressible_Neo_Hookean_Bonet`, `Compressible_Neo_Hookean_Ciarlet` | [lambda; mu] |

Weak form from the stress: `"((Id(meshdim)+Grad_u)*Saint_Venant_Kirchhoff_PK2(Grad_u, params)):Grad_Test_u"`,
or simply the potential as the term.

### 6.8 Macros

- Model macros: `md.add_macro("sigma", "lambda*Div_u*Id(meshdim) +
  2*mu*Sym(Grad_u)")`; with parameters `md.add_macro("ps(a,b)", "a.b")`.
  Expanded textually: wrap bodies in parentheses when precedence matters.
- Inline: `"Def k(w):=(1+w*w); k(u)*Grad_u.Grad_Test_u"` (definitions
  before the expression, each ending with `;`).
- A parameter may follow the prefixes `Grad_`, `Hess_`, `Test_`,
  `Test2_`: `md.add_macro("psgrad(a,b)", "Grad_a.Grad_b")` then
  `psgrad(u, Test_u)`; `jump(Test_v)` with body `Grad_w` gives
  `Grad_Test_v`.
- Predefined: `Grad(u)`, `Hess(u)`, `Div(u)`.

### 6.9 User scalar functions

`gf.define_gwfl_function("phi", 1, "t + t*t*t/3")`; two arguments
`gf.define_gwfl_function("g", 2, "t*u", "u", "t")` (der1, der2: expressions
in t, u or function names; omitted: symbolic). Only GWFL bodies (no Python
callables), no model data inside the body. Global, as GetFEM's: `phi(expr)`
in any model term, `gf.asm_generic` and `gf.interpolation`.

### 6.10 Interpolate transformations

- `Interpolate(q, trans)` with q one of `u`, `Grad_u`, `Div_u`, `Hess_u`,
  `Test_u`, `Grad_Test_u`, `Div_Test_u`, `Hess_Test_u`, `X`, `Normal`,
  `element_K`, `element_B`: evaluated at the transformed point.
- Transformations: `md.add_interpolate_transformation_from_expression(name,
  source_mesh, target_mesh, expr)` (expr maps X; may use variables,
  e.g. `"X+u"`); raytracing (contact); the
  predefined `neighbor_element` (on inner faces only: the value on the
  other element; `neighbour_elt` not supported).
- `Interpolate_filter(trans, expr, i)`: expr only where the
  transformation returned identifier i (1: a point found on the target; 0:
  none). `Interpolate_filter(trans, expr)` = any nonzero identifier.
- Derivatives w.r.t. variables in the transformation are taken into
  account for the tangent.

DG jump on inner faces:
`m.set_region(INNER, m.inner_faces())` then
`"(u-Interpolate(u,neighbor_element))*(Test_u-Interpolate(Test_u,neighbor_element))"`
on INNER.

### 6.11 XFEM and elementary transformations

- `Xfem_plus(q)`, `Xfem_minus(q)` (q as for Interpolate, e.g. `u`,
  `Grad_u`, `Test_u`): the value on the side where the level set is
  positive/negative, on a `MeshIm("levelset", mls, "boundary(...)", ...)`.
  Jump: `Xfem_plus(u)-Xfem_minus(u)`.
- `Elementary_transformation(u, trans[, dest])` exists in the language
  but no Python method adds an elementary transformation: unusable.

### 6.12 GWFL elsewhere

| Where | Context |
|---|---|
| `md.set_variable(name, expr)`, `md.set_variables` | interpolated on the variable's space with the model's values |
| `md.interpolation`, `gf.interpolation`, `md.local_projection` | |
| Dirichlet `data` (multiplier/penalization) | any expression of var's shape |
| `add_interpolate_transformation_from_expression` | the map, shape (d) |
| `gf.LevelSet` values | `x`, `y`, `z` macros for `X(1..3)` |
| `lips` of `crack_sifs` | expressions of X, Normal, numbers |

### 6.13 Expression results (checked, A = data [1,2,3,4] sizes [2,2])

| Expression | Value |
|---|---|
| `A(1,2)`, `A(2,1)` | 3, 2 (column-major storage) |
| `[1,2;3,4](1,2)`, `[[1,2],[3,4]](1,2)` | 2, 3 |
| `(A*[1;2])(2)`, `([1;2].A)(1)` | 10, 5 |
| `A:A`, `Trace(A)`, `Det(A)`, `Inv(A)(1,1)` | 30, 5, -2, -2 |
| `Diff(p*p, p, 1)` | 2p |
| `Interpolate_filter(t, 1, 0)` where t always finds a point | 0 |

## 7. Worked examples (all run)

### 7.1 Linear elasticity, traction, von Mises output

```python
import numpy as np
import getfem_rs as gf

LEFT, RIGHT = 1, 2
E, nu = 210e3, 0.3
lam, mu = E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))
m = gf.Mesh("structured", "GT='GT_PK(2,1)';ORG=[0,0];SIZES=[10,1];NSUBDIV=[20,4]")
m.set_region(LEFT, m.outer_faces_with_direction([-1., 0.], 0.01))
m.set_region(RIGHT, m.outer_faces_with_direction([1., 0.], 0.01))
mfu = gf.MeshFem(m, 2)
mfu.set_classical_finite_element(2)
mim = gf.MeshIm(m, 4)
md = gf.Model()
md.add_fem_variable("u", mfu)
md.add_initialized_data("lambda", lam)
md.add_initialized_data("mu", mu)
md.add_initialized_data("t", [0., -1.])      # traction on RIGHT
md.add_macro("sigma", "(lambda*Div_u*Id(meshdim) + 2*mu*Sym(Grad_u))")
md.add_linear_term(mim, "sigma:Grad_Test_u")
md.add_linear_term(mim, "-t.Test_u", RIGHT)  # Neumann: minus the load
md.add_Dirichlet_condition_with_multiplier(mim, "u", 1, LEFT)   # u = 0
nit, conv, info = md.solve()
assert conv, info["message"]
U = md.variable("u")                         # [u1(n0), u2(n0), u1(n1), ...]
print("tip deflection", U[1::2].min())
mfvm = gf.MeshFem(m)
mfvm.set_classical_discontinuous_finite_element(1)
vm = md.local_projection(mim, "sqrt(3/2)*Norm(Deviator(sigma))", mfvm)
mfu.export_to_vtu("beam.vtu", U, "u", mfvm, vm, "von Mises")
```

Dirichlet by removing DOFs instead of a multiplier (`examples/cantilever.py`
idiom): `clamped = set(mf.basic_dof_on_region(LEFT))`,
`mfu = gf.MeshFem("partial", mf, [d for d in range(mf.nb_dof()) if d not in clamped])`,
then the variable on `mfu` and `mf.export_to_vtu(f, mfu, U, "u")`.

### 7.2 Hyperelasticity, load steps, Newton report

```python
import numpy as np
import getfem_rs as gf

m = gf.Mesh("cartesian", np.linspace(0, 10, 21), np.linspace(0, 1, 3))
m.set_region(1, m.outer_faces_in_box([-0.1, -0.1], [0.1, 1.1]))   # x = 0
mfu = gf.MeshFem(m, 2)
mfu.set_classical_finite_element(2)
mim = gf.MeshIm(m, 4)
md = gf.Model()
md.add_fem_variable("u", mfu)
md.add_initialized_data("params", [100., 50.])   # [lambda; mu]
md.add_initialized_data("f", [0., 0.])
# a potential (no Test_): residual and tangent derived automatically
md.add_nonlinear_term(mim, "Plane_Strain_Compressible_Neo_Hookean_Bonet_potential(Grad_u, params)"
                           " - f.u")
md.add_Dirichlet_condition_with_multiplier(mim, "u", 1, 1)
for step in range(1, 4):
  md.set_variable("f", [0., -0.1*step])
  nit, conv, info = md.solve(rtol=1e-10, max_iter=50)
  assert conv, (info["status"], info["message"])
  print(step, nit, info["method"], md.variable("u")[1::2].min())
# Cauchy stress from the PK2 operator
md.add_macro("F", "(Id(2)+Grad_u)")
cauchy = "F*Plane_Strain_Compressible_Neo_Hookean_Bonet_PK2(Grad_u, params)*F'/Det(F)"
mfs = gf.MeshFem(m, 2, 2)
mfs.set_classical_discontinuous_finite_element(1)
S = md.local_projection(mim, cauchy, mfs)        # 4 values per node, column-major
print(S.shape, mfs.nb_dof())
```

### 7.3 Time stepping (backward Euler written out)

```python
import numpy as np
import getfem_rs as gf

m = gf.Mesh("regular_simplices", np.linspace(0, 1, 11), np.linspace(0, 1, 11))
m.set_region(1, m.outer_faces())
mf = gf.MeshFem(m)
mf.set_classical_finite_element(1)
mim = gf.MeshIm(m, 2)
md = gf.Model()
md.add_fem_variable("T", mf)
md.add_fem_data("T0", mf)   # previous step (zero)
md.add_initialized_data("dt", 0.01)
md.add_initialized_data("kappa", 1.)
# no time keyword, no Previous_/Dot_ in GWFL
md.add_linear_term(mim, "(T-T0)/dt*Test_T + kappa*Grad_T.Grad_Test_T - Test_T")
md.add_Dirichlet_condition_with_simplification("T", 1)
md.set_variable("T", "16*X(1)*(1-X(1))*X(2)*(1-X(2))")   # initial state
for step in range(20):
  md.set_variable("T0", "T")      # GWFL expression of the current values
  md.solve()
print("T max at t=0.2:", md.variable("T").max())
```

Second-order schemes (Newmark): keep `u0`, `v0`, `a0` as fem data, write
the acceleration as a macro of `u`, and update all at once with
`md.set_variables([("a0", "acc"), ("v0", "v0+dt*((1-gamma)*a0+gamma*acc)"), ("u0", "u")])`
(all right-hand sides use the old values).

### 7.4 Matrices, eigenvalues, direct solves

```python
import numpy as np
import getfem_rs as gf

m = gf.Mesh("cartesian", np.linspace(0, 1, 11), np.linspace(0, 1, 11))
m.set_region(1, m.outer_faces())
mf = gf.MeshFem(m)
mf.set_classical_finite_element(2)
mim = gf.MeshIm(m, 4)
md = gf.Model()
md.add_fem_variable("u", mf)
md.add_fem_variable("p", mf)
n = mf.nb_dof()
# order 0: float; order 1: vector over all model DOFs; order 2: gf.Spmat
area = gf.asm_generic(mim, 0, "1", -1, md)
F = gf.asm_generic(mim, 1, "X(1)*Test_u", -1, md)                      # len 2n
Fu = gf.asm_generic(mim, 1, "X(1)*Test_u", -1, md, select_output="u")  # len n
K = gf.asm_generic(mim, 2, "Grad_Test2_u.Grad_Test_u", -1, md, select_output=("u", "u"))
M = gf.asm_generic(mim, 2, "Test2_u*Test_u", -1, md, select_output=("u", "u"))
print(area, F.shape, Fu.shape, K.size(), K.nnz())
# Spmat -> dense numpy via the CSC arrays
jc, ir = K.csc_ind()
val = K.csc_val()
Kd = np.zeros(K.size())
for j in range(K.size()[1]):
  Kd[ir[jc[j]:jc[j+1]], j] = val[jc[j]:jc[j+1]]
# Dirichlet eigenproblem: keep the interior DOFs
bd = set(mf.basic_dof_on_region(1).tolist())
inner = [i for i in range(n) if i not in bd]
Ki, Mi = gf.Spmat("copy", K, inner), gf.Spmat("copy", M, inner)
vals, modes = gf.eigsh(Ki, Mi, nev=3, sigma=0.)   # ascending; modes as ROWS
print(vals/np.pi**2, modes.shape)                  # ~ [2, 5, 5], (3, 361)
x = gf.linsolve_mumps(Ki, np.ones(len(inner)))
print(x.shape, K.diag().shape)
```

### 7.5 Own Newton loop with MumpsContext

```python
import numpy as np
import getfem_rs as gf

m = gf.Mesh("cartesian", np.linspace(0, 1, 9), np.linspace(0, 1, 9))
m.set_region(1, m.outer_faces())
mf = gf.MeshFem(m)
mf.set_classical_finite_element(2)
mim = gf.MeshIm(m, 4)
md = gf.Model()
md.add_fem_variable("u", mf)
md.add_initialized_data("f", 1.)
md.add_linear_term(mim, "Grad_u.Grad_Test_u - f*Test_u")
md.add_Dirichlet_condition_with_multiplier(mim, "u", 1, 1)
md.assembly()                        # tangent K and rhs -R at the current values
ctx = gf.MumpsContext(2)             # 0 general, 1 SPD, 2 symmetric
ctx.set_matrix(md.tangent_matrix())  # live view; inline it, no K variable
ctx.analyze()
ctx.factorize()
du = ctx.solve(md.rhs())             # increment over all model DOFs
md.to_variables(md.from_variables() + du)
print(md.variable("u").max(), ctx.infog(1))
md.set_variable("f", 2.)             # new data: reassemble, refactorize
md.assembly()
ctx.factorize()
print(np.abs(ctx.solve(md.rhs())).max())
```

### 7.6 Level set: fictitious domain, cut integration, slice

```python
import numpy as np
import getfem_rs as gf

x = np.linspace(-1, 1, 11)
m = gf.Mesh("cartesian", x, x)
ls = gf.LevelSet(m, 2, "x*x + y*y - 0.5")      # GWFL in x, y, z; < 0 inside
mls = gf.MeshLevelSet(m)
mls.add(ls)
mls.adapt()                                    # cut the elements
mim = gf.MeshIm("levelset", mls, "inside(a)", gf.Integ("IM_TRIANGLE(6)"))
mim.set_integration_method(4)                  # rule of the uncut elements
mimb = gf.MeshIm("levelset", mls, "boundary(a)", gf.Integ("IM_TRIANGLE(6)"))
print("area", gf.asm_generic(mim, 0, "1", -1), "exact", 0.5*np.pi)
print("perimeter", gf.asm_generic(mimb, 0, "1", -1))
mf = gf.MeshFem(m)
mf.set_classical_finite_element(2)
mfu = gf.MeshFem("partial", mf, mf.dofs_with_support(mim, 1e-6))  # DOFs touching the disc
mfm = gf.MeshFem(m)
mfm.set_classical_finite_element(1)
md = gf.Model()
md.add_fem_variable("u", mfu)
md.add_linear_term(mim, "Grad_u.Grad_Test_u - Test_u")
# u = 0 on the circle: independent multipliers only
dofs = md.select_multiplier_dofs_for_variable("u", mimb, mfm, -1)
md.add_Dirichlet_condition_with_multiplier(mimb, "u", gf.MeshFem("partial", mfm, dofs), -1)
nit, conv, info = md.solve()
U = md.interpolation("u", mf)                  # on mf's basic DOFs
print(conv, U.max(), "exact", 0.5/4)
sl = gf.Slice(("isovalues", -1, ls.mf(), ls.values(0), 0.), m, 2)  # the disc
sl.export_to_vtu("disc.vtu", mf, U, "u")
```

### 7.7 Periodic condition and DG (interpolate transformations)

```python
import numpy as np
import getfem_rs as gf

LEFT, BOTTOMTOP, INNER, BOUNDARY = 1, 3, 4, 5
m = gf.Mesh("cartesian", np.linspace(0, 1, 9), np.linspace(0, 1, 9))
m.set_region(LEFT, m.outer_faces_with_direction([-1., 0.], 0.01))
m.set_region(BOTTOMTOP, m.outer_faces_with_direction([0., 1.], 0.01))
m.extend_region(BOTTOMTOP, m.outer_faces_with_direction([0., -1.], 0.01))
m.set_region(INNER, m.inner_faces())
m.set_region(BOUNDARY, m.outer_faces())
mim = gf.MeshIm(m, 4)

# periodic in x: u(0, y) = u(1, y) by a multiplier on LEFT
mf = gf.MeshFem(m)
mf.set_classical_finite_element(2)
md = gf.Model()
md.add_fem_variable("u", mf)
md.add_linear_term(mim, "Grad_u.Grad_Test_u - cos(2*pi*X(1))*Test_u")
md.add_Dirichlet_condition_with_multiplier(mim, "u", 1, BOTTOMTOP)
md.add_interpolate_transformation_from_expression("per", m, m, "X+[1;0]")
md.add_filtered_fem_variable("lp", mf, LEFT)
md.add_linear_term(mim, "(Interpolate(u,per)-u)*lp", LEFT)  # potential: derived in u, lp
nit, conv, info = md.solve()
print(md.interpolation("u", np.array([[0., 1.], [0.5, 0.5]]), m))  # equal values

# symmetric interior penalty DG on inner faces
mfd = gf.MeshFem(m)
mfd.set_classical_discontinuous_finite_element(2)
md2 = gf.Model()
md2.add_fem_variable("v", mfd)
md2.add_initialized_data("alpha", 20.)
md2.add_macro("jump(w)", "(w-Interpolate(w,neighbor_element))")
md2.add_macro("mean_dn(w)", "(0.5*(Grad_w+Interpolate(Grad_w,neighbor_element)).Normal)")
md2.add_linear_term(mim, "Grad_v.Grad_Test_v - Test_v")
md2.add_linear_term(mim, "-mean_dn(v)*jump(Test_v) - mean_dn(Test_v)*jump(v)"
                         " + alpha/element_size*jump(v)*jump(Test_v)", INNER)
md2.add_linear_term(mim, "-Grad_v.Normal*Test_v - Grad_Test_v.Normal*v"
                         " + alpha/element_size*v*Test_v", BOUNDARY)
nit, conv, info = md2.solve()
print("DG", conv, md2.variable("v").max())
```

Periodicity by reduction matrices (GetFEM's `periodicity.py` snippet,
with macro strain DOFs; only E written, R `"auto"`):
`crates/getfem-python/tests/test_reduction.py`.

### 7.8 Integration point variables (condensed) and im data

```python
import numpy as np
import getfem_rs as gf

m = gf.Mesh("cartesian", np.linspace(0, 1, 3), np.linspace(0, 1, 3))
m.set_region(1, m.outer_faces())
mf = gf.MeshFem(m)
mf.set_classical_finite_element(1)
mim = gf.MeshIm(m, 2)
imd = gf.MeshImData(mim)                 # scalar per point; MeshImData(mim, -1, [2, 2]) tensors
md = gf.Model()
md.add_fem_variable("u", mf)
md.add_internal_im_variable("q", imd)    # condensed; listed last in the system
md.add_im_data("q0", imd)
md.add_linear_term(mim, "Grad_u.Grad_Test_u + q*Test_u - Test_u + (q - 2*u - q0)*Test_q")
md.add_Dirichlet_condition_with_multiplier(mim, "u", 1, 1)   # not simplification
nit, conv, info = md.solve()
print(conv, md.nb_dof(), md.from_variables().shape, md.interval_of_variable("q"))
md.set_variable("q0", "q")               # history update at the points
print(md.variable("q0")[:3])
```

### 7.9 Exodus: time steps written, mesh and data read back

```python
import numpy as np
import getfem_rs as gf

m = gf.Mesh("cartesian", np.linspace(0, 1, 5), np.linspace(0, 1, 5))
m.set_region(3, m.outer_faces())         # written as side set 3
mf = gf.MeshFem(m, 2)
mf.set_classical_finite_element(1)
P = mf.basic_dof_nodes()
for k, t in enumerate([0.0, 0.5, 1.0]):  # one file, one space, three steps
  mf.export_to_exodus("u.exo", t * P[0], "u", time=t, append=k > 0)
m2 = gf.Mesh("import", "exodus", "u.exo")  # side set 3 -> region 3, block 1 -> 100001
mf2 = gf.MeshFem(m2, 2)
mf2.set_classical_finite_element(1)
U, t = mf2.import_from_exodus("u.exo", "u")   # the last step, its time
print(t, np.allclose(U, mf2.basic_dof_nodes()[0]), m2.region(3).shape)
U0, t0 = mf.import_from_exodus("u.exo", "u", step=0)  # onto the original space
```

## 8. Pitfalls

| Wrong | Right | Why |
|---|---|---|
| `md.add_linear_term(mim, "u*u*Test_u")` | `md.add_nonlinear_term(mim, "u*u*Test_u")` | add_linear_term checks linearity (ValueError) |
| `"f*Test_u"` as a load | `"-f*Test_u"` | terms are residuals R(u) = 0 |
| `"x^2"`, `"pow(X,2)"` | `"x*x"`, `"sqr(X(1))"`, `"X.X"` | no `^`; two-arg functions need equal sizes |
| `md.add_initialized_data("w", [5.])` then `w(1)` | `w` | size-1 data is a scalar in GWFL |
| `md.add_initialized_fem_data("g", mf, 1.0)` | `md.add_initialized_fem_data("g", mf, np.ones(mf.nb_dof()))` | one value per DOF |
| `md.set_variable("g2", 3.0)` on fem data | `md.set_variable("g2", "3")` or an array | numbers need the full length; strings are interpolated |
| `md.set_variable("s", "2")` on fixed-size data | `md.set_variable("s", 2.)` | strings only for fem/im spaces |
| `add_Dirichlet_condition_with_simplification("u", 1, "g")` with g a constant | g fem data on u's MeshFem, or use the multiplier version with `data="[1;2]"` | simplification takes DOF values |
| Two multiplier conditions on shared boundary DOFs | `select_multiplier_dofs_for_variable` then `"partial"` | dependent constraints are an error |
| `mf.set_classical_finite_element(2)` after `md.add_fem_variable("u", mf)` | set the element first | the variable keeps the old size until `md.set_variable("u", array)` |
| `K = md.tangent_matrix()`, then `md.solve()`, then use K as the old matrix | `K = gf.Spmat("copy", md.tangent_matrix())` | the view follows the model |
| `gf.asm_generic(mim, 0, "u", -1)` | `gf.asm_generic(mim, 0, "u", -1, md)` or `gf.interpolation` with fields | without md no variables |
| `gf.asm_generic(mim, 0, "1")` | `gf.asm_generic(mim, 0, "1", -1)` | region is required |
| `md.solve()` and ignore the result | `nit, conv, info = md.solve(); assert conv, info["message"]` | no exception on failure |
| Pure Neumann problem | add a Dirichlet condition, a multiplier, or `1e-9*u.Test_u` | a singular system with an incompatible load does not converge (`converged=False`, values ~1e15) |
| `Normal` in a volume term | a face region or a level-set boundary MeshIm | `Normal` is silently zero in volumes |
| `md.local_projection(mim, e, mf_continuous)` | a discontinuous mf | ValueError otherwise |
| `md.interpolation(e, P, m)` with points outside | keep points inside the mesh | ValueError |
| `Interpolate(p, neighbor_element)` on a volume region | on `m.inner_faces()` region | faces only |
| `Expm`, `Logm`, `Normalized`, `Ball_projection` inside a potential | write the residual (order 1) term instead | first derivatives only |
| `modes[:, 0]` as the first mode | `modes[0]` | eigsh modes are rows |
| a 2D gmsh (or planar Exodus) mesh used as 2D right after import | `m.transform([[1, 0, 0], [0, 1, 0]])` first | imports keep the file's dimension (gmsh: 3D) |
| `mf.export_to_exodus(f, mf2, V, "v")` | `mf2.export_to_exodus(f2, V, "v")` | one space per Exodus file |
| `mf.set_reduction_matrices(RR, EE)` with RR picking the kept DOFs and EE adding macro DOFs on them (GetFEM's periodicity snippet) | `mf.set_reduction_matrices("auto", EE)` | R E = I is checked (ValueError); "auto" computes R from E |
| `md.interpolation("u", mf)` on a reduced `mf` | `mf.extend_vector(md.variable("u"))` (u on the basic DOFs) | interpolation on a reduced space is refused, as in GetFEM |
| `U = mf.import_from_exodus(f, "u")` | `U, t = mf.import_from_exodus(f, "u")` | it returns the values and the step's time |
| `gf.Slice(("mesh",), m, 1)` | `gf.Slice(("none",), m, 1)` | no "mesh" op |
| `mim.set_integration_method(4, [0])` on a levelset MeshIm | `mim.set_integration_method(4)` | only the uncut elements, all of them |
| `erf(u)` | not available | unsupported in getfem_rs |
| `Print(u)` for debugging | `md.interpolation("u", mf)`, `gf.asm_generic(mim, 0, ...)` | `Print` refused |
