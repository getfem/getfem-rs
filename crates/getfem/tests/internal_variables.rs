//! After GetFEM's tests/test_internal_variables: the derivative of the
//! determinant of an im variable.

use getfem::fem::im_data::ImData;
use getfem::fem::mesh_im::MeshIm;
use getfem::geom::geotrans::geotrans_descriptor;
use getfem::geom::regular_mesh::regular_unit_mesh;
use getfem::model::{BuildVersion, Model};

#[test]
fn det_of_an_im_variable() {
  // md4 of the GetFEM test: d/df33 Det(diag(1, 1, f33)) = 1 at f33 = 1, so
  // the tangent is diagonal, the weights times |J|, summing to the area
  let m = regular_unit_mesh(&[2, 2], &geotrans_descriptor("GT_QK(2, 2)").unwrap()).unwrap();
  let mim = MeshIm::new(&m).with_integration_method(3).unwrap();
  let mimd = ImData::new(&mim, -1, &[]).unwrap();
  let mut md = Model::new();
  md.add_im_variable("f33", &mimd).unwrap();
  let n = mimd.nb_values().unwrap();
  md.set_variable("f33", &vec![1.0; n]).unwrap();
  md.add_term(&mim, None, "Det([1,0,0;0,1,0;0,0,f33])*Test_f33")
    .unwrap();
  md.assembly(BuildVersion::All).unwrap();
  let r: Vec<f64> = md.rhs().iter().map(|v| -v).collect();
  let k = md.tangent_matrix();
  assert_eq!((k.nrows, r.len()), (n, n));
  let mut diag = 0.0f64;
  for (j, col) in k.cols.iter().enumerate() {
    for &(i, v) in col {
      assert_eq!(i, j);
      diag += v;
    }
  }
  assert!((diag - 1.0).abs() < 1e-13 && (r.iter().sum::<f64>() - 1.0).abs() < 1e-13);
}
