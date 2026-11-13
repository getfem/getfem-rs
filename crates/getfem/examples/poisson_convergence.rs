//! -Laplace u = f on the unit square, u = sin(pi x) sin(pi y): L2 and H1
//! errors for P1-P3 triangles and Q1-Q3 quadrilaterals under refinement.
//! Writes target/showcase/convergence.csv.

use getfem::fem::mesh_fem::MeshFem;
use getfem::fem::mesh_im::MeshIm;
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::region::outer_faces_of_mesh;
use getfem::geom::regular_mesh::regular_mesh;
use getfem::gwfl::workspace::Workspace;
use getfem::linalg::{LinearSolver, MatrixKind};
use getfem::mumps::Mumps;

fn main() {
  let out = format!("{}/../../target/showcase", env!("CARGO_MANIFEST_DIR"));
  std::fs::create_dir_all(&out).unwrap();
  let mut csv = String::from("element,k,n,h,dofs,err_l2,err_h1\n");
  let exact = "sin(pi*X(1))*sin(pi*X(2))";
  let grad = "pi*[cos(pi*X(1))*sin(pi*X(2)); sin(pi*X(1))*cos(pi*X(2))]";
  for (gname, label) in [("GT_PK(2,1)", "triangle"), ("GT_QK(2,1)", "quad")] {
    for k in 1..=3 {
      let mut prev: Option<(f64, f64)> = None;
      for n in [4, 8, 16, 32] {
        let mesh = regular_mesh(
          &geotrans_descriptor(gname).unwrap(),
          &[n, n],
          &[0.0; 2],
          &[1.0; 2],
        )
        .unwrap();
        let mf = MeshFem::new(&mesh, 1)
          .with_classical_finite_element(k)
          .unwrap();
        let mim = MeshIm::new(&mesh)
          .with_integration_method(2 * k + 4)
          .unwrap();
        // u = 0 on the boundary: the unknown on the interior DOFs
        let boundary = mf.basic_dof_on_region(outer_faces_of_mesh(&mesh)).unwrap();
        let kept = (0..mf.nb_dof().unwrap())
          .filter(|d| !boundary.contains(d))
          .collect();
        let mfu = MeshFem::partial(&mf, &kept).unwrap();
        let mut ws = Workspace::new();
        ws.add_fem_variable("u", &mfu).unwrap();
        let kk = {
          ws.clear_expressions();
          ws.add_expression("Grad_Test_u.Grad_Test2_u", &mim, None)
            .unwrap();
          ws.assembly(2, false).unwrap();
          ws.assembled_matrix()
        }
        .to_csr();
        let mut b = {
          ws.clear_expressions();
          ws.add_expression(&format!("2*sqr(pi)*{exact}*Test_u"), &mim, None)
            .unwrap();
          ws.assembly(1, false).unwrap();
          ws.assembled_vector()
        }
        .to_vec();
        Mumps::new()
          .solve_system(&kk, MatrixKind::SymmetricPositiveDefinite, &mut b)
          .unwrap();
        ws.set_variable("u", &b).unwrap();
        let mut err = |e: &str| {
          {
            ws.clear_expressions();
            ws.add_expression(e, &mim, None).unwrap();
            ws.assembly(0, false).unwrap();
            ws.assembled_potential()
          }
          .sqrt()
        };
        let (e0, e1) = (
          err(&format!("sqr(u - {exact})")),
          err(&format!("Norm_sqr(Grad_u - {grad})")),
        );
        let h = 1.0 / n as f64;
        let rates = prev
          .map(|(p0, p1)| format!("  rates {:.2} {:.2}", (p0 / e0).log2(), (p1 / e1).log2()))
          .unwrap_or_default();
        println!(
          "{label} k={k} n={n:2} dofs={:6}  L2 {e0:.3e}  H1 {e1:.3e}{rates}",
          mf.nb_basic_dof().unwrap()
        );
        csv.push_str(&format!(
          "{label},{k},{n},{h},{},{e0:e},{e1:e}\n",
          mf.nb_basic_dof().unwrap()
        ));
        prev = Some((e0, e1));
      }
    }
  }
  std::fs::write(format!("{out}/convergence.csv"), csv).unwrap();
}
