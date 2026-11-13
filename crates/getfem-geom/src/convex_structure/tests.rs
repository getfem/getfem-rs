use super::*;

#[test]
fn interned() {
  assert!(Arc::ptr_eq(
    &simplex_structure_k(3, 2),
    &simplex_structure_k(3, 2)
  ));
  assert_eq!(*parallelepiped_structure(2, 1), *prism_p1_structure(2));
  assert_eq!(
    *parallelepiped_structure(3, 2).basic(),
    *parallelepiped_structure(3, 1)
  );
  assert!(q2_incomplete_structure(4).is_err());
  assert!(pyramid_qk_structure(3).is_err());
}
