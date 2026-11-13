# GetFEM-rs 0.9

A port of [GetFEM](https://getfem.org), the finite element library, to Rust.
It keeps GetFEM's strengths (the weak form language GWFL, a broad element
and integration catalogue, models for large strains, plasticity, contact and
continuation, level sets with integration on cut elements, XFEM in 2D and 3D
with stress intensity factors along crack fronts) on a new, lean codebase,
and goes beyond the original in speed and correctness:

- GWFL at the centre: a model is a set of weak-form terms written as text;
  residuals and tangents are derived symbolically and compiled in-process,
  with no code generation step.
- Performance: the tangent and residual of a 29 106-DOF nonlinear
  elasticity model on Q2 hexahedra assembled in 0.14 s on one core; fields
  at 110 000 points per second on curved meshes (see Performance).
- Rust: unsafe code only in the MUMPS, libxsmm and MPI bindings; GetFEM's
  crashes found by the port are errors here.
- Python: the module `getfem_rs`, close to GetFEM's Python interface.
- Parallel: assembly threads, and MPI ranks (a compile-time feature).
- Few layers: physics is added as GWFL terms, not through bricks; one
  representation per object and one mechanism per job.
- No hidden machinery: an element without an integration method is an
  error, Newton reports which of its phases converged.

The tests check results against exact solutions and invariances. The
tests against GetFEM's own results (reference data of GetFEM c5433d32) are
the crate `getfem-tests`: in the repository, and in a separate archive
`getfem-rs-tests-0.9.0.tar.gz` that unpacks into this tree. Where GetFEM
has a defect, GetFEM-rs does not reproduce it.

## Contents

| Crate | What it holds |
| --- | --- |
| `getfem` | facade re-exporting the crates below; examples and acceptance tests |
| `getfem-geom` | reference convexes, polynomials, geometric transformations, meshes, regions, structured meshes, GetFEM `.mesh` files |
| `getfem-fem` | finite elements, cubature, `MeshFem`, `MeshIm`, `ImData`, interpolation, mesh slices |
| `getfem-gwfl` | the weak form language: parser, analysis, derivation, GetFEM's compiled instructions, `Workspace`, interpolate and elementary transformations; stress intensity factors of cracks |
| `getfem-model` | `Model` (terms, Dirichlet conditions, multipliers), `Newton` |
| `getfem-linalg` | sparse matrices (gmm's column matrix of sorted sparse vectors, CSR, COO), the `LinearSolver` trait, the real Schur form (gmm's QR algorithm), the preconditioned conjugate gradient (gmm's `cg`) |
| `getfem-mumps` | MUMPS as a `LinearSolver`, sequential or with MPI (feature `mpi`), and its FFI |
| `getfem-arpack` | ARPACK-ng's symmetric eigensolver in shift-invert mode, `eigsh` (MUMPS factorizes the shifted matrix) |
| `getfem-xsmm` | small matrix products by libxsmm kernels (no BLAS) |
| `getfem-mpi` | MPI (feature `mpi`): rank, sums and broadcasts of data replicated on every rank; one rank without it |
| `getfem-io` | gmsh import; Exodus II import and export (meshes, regions, nodal fields at time steps, own netCDF reader and writer); VTU export as GetFEM's `vtk_export` (a mesh, a space or a slice; fields on any space; ASCII, binary, compressed), PVD collections |
| `getfem-python` | the Python module `getfem_rs` (PyO3): GetFEM's classes, methods and string constructors for what GetFEM-rs has; its examples and tests |
| `getfem-tests` | tests against GetFEM: its reference data, the programs that generated it, Rust and Python tests (a separate archive) |

`data/` holds the input files of the examples and tests (GetFEM meshes, gmsh
and Exodus II files), `docs/getfem_rs.md` the Python API and GWFL reference.

## Requirements

- Rust 1.95 or later.
- The sequential MUMPS library, with its BLAS and LAPACK (GetFEM-rs itself
  uses neither). On Debian: `apt install libmumps-seq-dev libopenblas0`.
- ARPACK-ng (Debian: `libarpack2-dev`).
- For MPI runs only (the feature `mpi`, off by default): an MPI with
  `mpicc` and the MPI MUMPS (Debian: `libopenmpi-dev libmumps-dev`); from
  Python, usually mpi4py (Debian: `python3-mpi4py`).
- For Python only: Python 3.12 or later with numpy; pip builds the module
  with maturin.
- Unsafe code only in the FFI crates (`getfem-arpack`, `getfem-mpi`,
  `getfem-mumps`, `getfem-xsmm`); the others forbid it.

MUMPS runs on `getfem::mumps::set_num_threads(n)` threads (its BLAS', and
its OpenMP's if built with it), independent of the assembly's
`getfem::set_num_threads`; 1 by default, as GetFEM: the reference data
were produced so, and threads change the rounding of the factorization.
`getfem::mumps::blas_info()` tells the BLAS found and its threads; a
warning if they cannot follow (serial, unknown BLAS), and if
`OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, ... are set: they are
overridden.

## Build and test

```sh
cargo build --release
cargo test --release
```

Some tests also check exported files with VTK when the Python `vtk` module
is available, and are skipped otherwise. With the tests archive unpacked,
`cargo test --release -p getfem-tests` runs the tests against GetFEM (and
`pytest crates/getfem-tests/python` their Python part).

MPI: build with `--features getfem/mpi`; a program calls
`getfem::mpi::init()` first and runs under `mpirun`. Each rank runs one
thread (assembly and MUMPS; `init()` says so); parallelism comes from the
ranks. Without the feature there is one rank and the MPI calls cost
nothing. In the repository, `ci/mpi.sh` runs the MPI tests on 1, 2 and 4
ranks.

Python, in a virtual environment:

```sh
pip install ./crates/getfem-python
python crates/getfem-python/examples/cantilever.py
pip install pytest && pytest crates/getfem-python/tests
```

Scripts written for GetFEM's Python interface need few changes (`import
getfem_rs as gf`; GetFEM-rs has no bricks: terms in GWFL);
`docs/getfem_rs.md` lists the differences (section 5).

Python with MPI: build the module with `maturin build --release
--features mpi` in `crates/getfem-python`, `pip install` the wheel, and
run under `mpirun`. A script whose problem is distributed over the
ranks calls `gf.mpi_init()` (mpi4py imported before or after it: one
MPI either way, finalized by whoever started it); without it each rank runs
its own sequential problem (e.g. a task farm with mpi4py), and without
`mpirun` the MPI build runs sequential jobs. On several ranks an
assembled matrix is each rank's part (summed by MUMPS, also in
`gf.linsolve_mumps`, `gf.eigsh`, `gf.MumpsContext`), scalars and vectors
are whole on every rank.

```python
import getfem_rs as gf
from mpi4py import MPI  # the rank, for output
rank = MPI.COMM_WORLD.Get_rank()
gf.mpi_init()
```

## A first program

The Poisson problem -Δu = 1 on the unit square, u = 0 on the boundary:

```rust
use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im::MeshIm;
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::region::outer_faces_of_mesh;
use getfem::geom::regular_mesh::regular_unit_mesh;
use getfem::model::{Model, Newton};

fn main() {
  let mesh = regular_unit_mesh(&[20, 20], &geotrans_descriptor("GT_PK(2,1)").unwrap()).unwrap();
  mesh.set_region(1, &outer_faces_of_mesh(&mesh)).unwrap();
  let mf = MeshFem::new(&mesh, 1).with_classical_finite_element(2).unwrap();
  let mim = MeshIm::new(&mesh).with_integration_method(4).unwrap();

  let mut md = Model::new();
  md.add_fem_variable("u", &mf).unwrap();
  md.add_linear_term(&mim, None, "Grad_u.Grad_Test_u - Test_u").unwrap();
  md.add_Dirichlet_condition_with_multiplier(&mim, "u", &mf, 1, None).unwrap();
  let report = Newton::default().solve(&mut md, None).unwrap();
  assert!(report.converged());

  let u = md.variable("u").unwrap();
  println!("max u = {:.6}", u.iter().fold(0.0f64, |a, b| a.max(*b)));
}
```

## Examples

In `crates/getfem/examples`, run with `cargo run --release -p getfem --example <name>`:

| Example | |
| --- | --- |
| `poisson_convergence` | convergence rates of P1 to P3 triangles and Q1 to Q3 quadrilaterals |
| `plate_potential` | potential in a plate with holes, curved P2 mesh from gmsh |
| `cantilever` | 3D linear elasticity, Q2 hexahedra |
| `hyperelastic_beam` | large deflection of a neo-Hookean cantilever, in load steps |
| `thermo_coupling` | thermo-electro-elastic plate with holes (GetFEM's `thermo_elasticity_electrical_coupling`) |
| `newton_tolerances` | convergence tests of the Newton solver on that plate |
| `contact` | two elastic bodies in contact (raytracing along the deformed normal), penalty and semismooth Newton |
| `bouncing_ball` | an elastic disc dropped on a rigid ground (GetFEM's `demo_dynamic_contact.m` in 2D): Newmark written out, penalty contact |
| `ball_on_block` | the disc dropped on an elastic block: two bodies by Newmark, contact by raytracing along the deformed normal, penalty |
| `two_rings` | two rubber rings of second order hexahedra (extruded and bent), a GetFEM user's script: one pulled through the other by handles held with multipliers until they hold like chain links, contact by raytracing along the deformed normal |
| `viscoelastic_ball` | the bouncing disc in a large strain quasilinear viscoelastic material (GetFEM's `QLV_viscoelasticity.py`): each bounce lower |
| `topology_optimization` | a cantilever's compliance minimized at half volume, large strains, the density on uniform B-splines, solved in pseudo-time (a user's GetFEM script) |
| `phase_field` | phase field fracture of a notched plate (GetFEM's `demo_phase_field.py`), every load step reached: pseudo-viscous steps on phi where Newton fails; tension, or shear (argument `shear`) |
| `phase_field_hydrogen` | the GWFL paper's phase field fracture with hydrogen: u, damage, pressure, concentration, inertia, adaptive time step |
| `membrane` | the GWFL paper's inflated membrane: a disc embedded in 3D, u and the thickness vector, continuation in the load past its limit (0.6912) |
| `finite_strain_plasticity` | necking of a plane strain block, Eterovic-Bathe plasticity with a condensed internal variable (GetFEM's `finite_strain_plasticity_ksi.py`); in load steps, or by continuation past the force peak (argument `continuation`) |
| `level_set_cut` | two crossing spheres cut a tetrahedral mesh: volumes of their intersection, difference and union, the area of a sphere, the cut mesh |
| `crack` | a crack in 2D linear elasticity by XFEM (GetFEM's `demo_crack.py`): enrichment with a cutoff, by products, two crossing cracks |
| `gas_filled_crack` | two crossing cracks filled with an ideal gas in a heated rubber block, large strains, XFEM for u and the temperature (a user's GetFEM script) |
| `gas_filled_crack_3d` | the same in 3D, the cracks bounded by an ellipsoid |
| `crossing_cracks_3d` | two crossing cracks in 3D, linearized elasticity: a traction on the block, a pressure on the lips |
| `penny_crack` | a penny-shaped crack under pressure against Sneddon's opening and K_I along the front; the crack front enrichments compared |
| `surface_crack` | cracks from a free face: an edge crack in 2D, a through crack and a semicircular surface crack in 3D; stress intensity factors along the fronts (VTU) |
| `inclined_penny_crack` | a penny crack tilted to a uniaxial tension: K_I, K_II and K_III along the front against the exact ones (VTU) |

Results are written to `target/showcase`. Each example is also in Python,
in `crates/getfem-python/examples`.

## Performance

Measured on 3 Nov 2026 on one i9-13900K, wall time, best of several runs.

The benchmark `cargo bench -p getfem --bench model` times what Newton
assembles at each iteration (a nonlinear elasticity term, mass and source
terms, Dirichlet multipliers; tangents, residuals, the global matrix),
after the first assembly, on one core:

| elements | DOFs | time [s] |
| --- | --- | --- |
| P2 tetrahedra | 29 106 | 0.149 |
| Q2 hexahedra | 29 106 | 0.137 |
| 27-node hexahedra, 14-point rule | 29 106 | 0.096 |
| 20-node hexahedra, 14-point rule | 15 906 | 0.059 |
| Q1 hexahedra | 15 606 | 0.048 |
| P2 isoparametric triangles | 81 204 | 0.071 |

On 4 threads (`getfem::set_num_threads`) the assembly is 2.0 to 3.5 times
faster; with MUMPS on 4 threads too (`getfem::mumps::set_num_threads`,
pthread OpenBLAS) a Newton iteration is 1.20 to 1.74 times faster. On 8 MPI
ranks (111 000 DOFs) the assembly is 4.9 to 5.7 times and MUMPS 3.1 to 3.3
times faster.

The benchmark `cargo bench -p getfem --bench interp3d` evaluates fields of
curved second order meshes at 20 000 points: 0.16 to 0.18 s, the point
location included. The benchmark `cargo bench -p getfem --bench cut3d`
builds the cut and the integration methods of 4^3 to 8^3 cubes of
tetrahedra cut by spheres in 0.5 to 1.4 s, the measures within 8e-6
(relative).

Against GetFEM bda44707 (built with serial OpenBLAS, one core), GetFEM-rs's
time over GetFEM's, geometric means: 0.63 on the 70 cases of the benchmark
`assembly` (GetFEM's `tests/test_assembly.cc`, the Ogden stress term,
curved meshes), 0.77 on the 50 of `opt_assembly` (GetFEM's
`contrib/opt_assembly`), 0.63 on the models, 0.15 on fields at points and
0.11 on cut cells.

## Documentation

- `cargo doc --open`: the API.
- `docs/getfem_rs.md`: the Python API and GWFL in one file, written
  for LLMs writing `getfem_rs` scripts (useful to people too).
- In the repository, not in the release archive: `CHANGELOG.md`,
  `KNOWN_ISSUES.md` and `DEVIATIONS.md` (the defects of GetFEM that
  GetFEM-rs does not reproduce, the behaviour that differs on purpose).

## License

GNU Lesser General Public License, version 3 or later (`COPYING.LESSER`,
`COPYING`), as GetFEM. GetFEM-rs is a port of GetFEM (Yves Renard, Julien
Pommier and the GetFEM contributors); the meshes in `data/meshes` and the
reference data of `getfem-tests` were produced with, or taken from, GetFEM.
