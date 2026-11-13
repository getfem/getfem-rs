use super::*;

#[test]
fn errors() {
  let c = simplex_of_reference(2);
  assert!(c.is_in(&[0.1]).is_err());
  assert!(c.is_in_face(3, &[0.1, 0.1]).is_err());
  assert!(pyramid_qk_of_reference(3).is_err());
}
