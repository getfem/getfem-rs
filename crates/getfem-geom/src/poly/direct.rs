//! Direct evaluation of polynomials of 1 to 3 variables and degree 2 to 6,
//! as GetFEM's polynomial::eval (generated from bgeot_poly.h: the same
//! formulas, so the same rounding).

use super::Polynomial;
use getfem_linalg::Scalar;

impl<T: Scalar> Polynomial<T> {
  pub(super) fn eval_direct(&self, it: &[T]) -> Option<T> {
    Some(match (self.n, self.d) {
      (1, 2) => {
        let p: &[T; 3] = self.c[..3].try_into().expect("coefficients");
        let x = it[0];
        p[0] + x * (p[1] + x * (p[2]))
      }
      (1, 3) => {
        let p: &[T; 4] = self.c[..4].try_into().expect("coefficients");
        let x = it[0];
        p[0] + x * (p[1] + x * (p[2] + x * (p[3])))
      }
      (1, 4) => {
        let p: &[T; 5] = self.c[..5].try_into().expect("coefficients");
        let x = it[0];
        p[0] + x * (p[1] + x * (p[2] + x * (p[3] + x * (p[4]))))
      }
      (1, 5) => {
        let p: &[T; 6] = self.c[..6].try_into().expect("coefficients");
        let x = it[0];
        p[0] + x * (p[1] + x * (p[2] + x * (p[3] + x * (p[4] + x * (p[5])))))
      }
      (1, 6) => {
        let p: &[T; 7] = self.c[..7].try_into().expect("coefficients");
        let x = it[0];
        p[0] + x * (p[1] + x * (p[2] + x * (p[3] + x * (p[4] + x * (p[5] + x * (p[6]))))))
      }
      (2, 2) => {
        let p: &[T; 6] = self.c[..6].try_into().expect("coefficients");
        let x = it[0];
        let y = it[1];
        p[0] + x * (p[1] + x * (p[3])) + y * (p[2] + x * (p[4]) + y * (p[5]))
      }
      (2, 3) => {
        let p: &[T; 10] = self.c[..10].try_into().expect("coefficients");
        let x = it[0];
        let y = it[1];
        p[0]
          + x * (p[1] + x * (p[3] + x * (p[6])))
          + y * (p[2] + x * (p[4] + x * (p[7])) + y * (p[5] + x * (p[8]) + y * (p[9])))
      }
      (2, 4) => {
        let p: &[T; 15] = self.c[..15].try_into().expect("coefficients");
        let x = it[0];
        let y = it[1];
        p[0]
          + x * (p[1] + x * (p[3] + x * (p[6] + x * (p[10]))))
          + y
            * (p[2]
              + x * (p[4] + x * (p[7] + x * (p[11])))
              + y * (p[5] + x * (p[8] + x * (p[12])) + y * (p[9] + x * (p[13]) + y * (p[14]))))
      }
      (2, 5) => {
        let p: &[T; 21] = self.c[..21].try_into().expect("coefficients");
        let x = it[0];
        let y = it[1];
        p[0]
          + x * (p[1] + x * (p[3] + x * (p[6] + x * (p[10] + x * (p[15])))))
          + y
            * (p[2]
              + x * (p[4] + x * (p[7] + x * (p[11] + x * (p[16]))))
              + y
                * (p[5]
                  + x * (p[8] + x * (p[12] + x * (p[17])))
                  + y
                    * (p[9] + x * (p[13] + x * (p[18])) + y * (p[14] + x * (p[19]) + y * (p[20])))))
      }
      (2, 6) => {
        let p: &[T; 28] = self.c[..28].try_into().expect("coefficients");
        let x = it[0];
        let y = it[1];
        p[0]
          + x * (p[1] + x * (p[3] + x * (p[6] + x * (p[10] + x * (p[15] + x * (p[21]))))))
          + y
            * (p[2]
              + x * (p[4] + x * (p[7] + x * (p[11] + x * (p[16] + x * (p[22])))))
              + y
                * (p[5]
                  + x * (p[8] + x * (p[12] + x * (p[17] + x * (p[23]))))
                  + y
                    * (p[9]
                      + x * (p[13] + x * (p[18] + x * (p[24])))
                      + y
                        * (p[14]
                          + x * (p[19] + x * (p[25]))
                          + y * (p[20] + x * (p[26]) + y * (p[27]))))))
      }
      (3, 2) => {
        let p: &[T; 10] = self.c[..10].try_into().expect("coefficients");
        let x = it[0];
        let y = it[1];
        let z = it[2];
        p[0]
          + x * (p[1] + x * (p[4]))
          + y * (p[2] + x * (p[5]) + y * (p[7]))
          + z * (p[3] + x * (p[6]) + y * (p[8]) + z * (p[9]))
      }
      (3, 3) => {
        let p: &[T; 20] = self.c[..20].try_into().expect("coefficients");
        let x = it[0];
        let y = it[1];
        let z = it[2];
        p[0]
          + x * (p[1] + x * (p[4] + x * (p[10])))
          + y * (p[2] + x * (p[5] + x * (p[11])) + y * (p[7] + x * (p[13]) + y * (p[16])))
          + z
            * (p[3]
              + x * (p[6] + x * (p[12]))
              + y * (p[8] + x * (p[14]) + y * (p[17]))
              + z * (p[9] + x * (p[15]) + y * (p[18]) + z * (p[19])))
      }
      (3, 4) => {
        let p: &[T; 35] = self.c[..35].try_into().expect("coefficients");
        let x = it[0];
        let y = it[1];
        let z = it[2];
        p[0]
          + x * (p[1] + x * (p[4] + x * (p[10] + x * (p[20]))))
          + y
            * (p[2]
              + x * (p[5] + x * (p[11] + x * (p[21])))
              + y * (p[7] + x * (p[13] + x * (p[23])) + y * (p[16] + x * (p[26]) + y * (p[30]))))
          + z
            * (p[3]
              + x * (p[6] + x * (p[12] + x * (p[22])))
              + y * (p[8] + x * (p[14] + x * (p[24])) + y * (p[17] + x * (p[27]) + y * (p[31])))
              + z
                * (p[9]
                  + x * (p[15] + x * (p[25]))
                  + y * (p[18] + x * (p[28]) + y * (p[32]))
                  + z * (p[19] + x * (p[29]) + y * (p[33]) + z * (p[34]))))
      }
      (3, 5) => {
        let p: &[T; 56] = self.c[..56].try_into().expect("coefficients");
        let x = it[0];
        let y = it[1];
        let z = it[2];
        p[0]
          + x * (p[1] + x * (p[4] + x * (p[10] + x * (p[20] + x * (p[35])))))
          + y
            * (p[2]
              + x * (p[5] + x * (p[11] + x * (p[21] + x * (p[36]))))
              + y
                * (p[7]
                  + x * (p[13] + x * (p[23] + x * (p[38])))
                  + y
                    * (p[16]
                      + x * (p[26] + x * (p[41]))
                      + y * (p[30] + x * (p[45]) + y * (p[50])))))
          + z
            * (p[3]
              + x * (p[6] + x * (p[12] + x * (p[22] + x * (p[37]))))
              + y
                * (p[8]
                  + x * (p[14] + x * (p[24] + x * (p[39])))
                  + y
                    * (p[17]
                      + x * (p[27] + x * (p[42]))
                      + y * (p[31] + x * (p[46]) + y * (p[51]))))
              + z
                * (p[9]
                  + x * (p[15] + x * (p[25] + x * (p[40])))
                  + y
                    * (p[18]
                      + x * (p[28] + x * (p[43]))
                      + y * (p[32] + x * (p[47]) + y * (p[52])))
                  + z
                    * (p[19]
                      + x * (p[29] + x * (p[44]))
                      + y * (p[33] + x * (p[48]) + y * (p[53]))
                      + z * (p[34] + x * (p[49]) + y * (p[54]) + z * (p[55])))))
      }
      (3, 6) => {
        let p: &[T; 84] = self.c[..84].try_into().expect("coefficients");
        let x = it[0];
        let y = it[1];
        let z = it[2];
        p[0]
          + x * (p[1] + x * (p[4] + x * (p[10] + x * (p[20] + x * (p[35] + x * (p[56]))))))
          + y
            * (p[2]
              + x * (p[5] + x * (p[11] + x * (p[21] + x * (p[36] + x * (p[57])))))
              + y
                * (p[7]
                  + x * (p[13] + x * (p[23] + x * (p[38] + x * (p[59]))))
                  + y
                    * (p[16]
                      + x * (p[26] + x * (p[41] + x * (p[62])))
                      + y
                        * (p[30]
                          + x * (p[45] + x * (p[66]))
                          + y * (p[50] + x * (p[71]) + y * (p[77]))))))
          + z
            * (p[3]
              + x * (p[6] + x * (p[12] + x * (p[22] + x * (p[37] + x * (p[58])))))
              + y
                * (p[8]
                  + x * (p[14] + x * (p[24] + x * (p[39] + x * (p[60]))))
                  + y
                    * (p[17]
                      + x * (p[27] + x * (p[42] + x * (p[63])))
                      + y
                        * (p[31]
                          + x * (p[46] + x * (p[67]))
                          + y * (p[51] + x * (p[72]) + y * (p[78])))))
              + z
                * (p[9]
                  + x * (p[15] + x * (p[25] + x * (p[40] + x * (p[61]))))
                  + y
                    * (p[18]
                      + x * (p[28] + x * (p[43] + x * (p[64])))
                      + y
                        * (p[32]
                          + x * (p[47] + x * (p[68]))
                          + y * (p[52] + x * (p[73]) + y * (p[79]))))
                  + z
                    * (p[19]
                      + x * (p[29] + x * (p[44] + x * (p[65])))
                      + y
                        * (p[33]
                          + x * (p[48] + x * (p[69]))
                          + y * (p[53] + x * (p[74]) + y * (p[80])))
                      + z
                        * (p[34]
                          + x * (p[49] + x * (p[70]))
                          + y * (p[54] + x * (p[75]) + y * (p[81]))
                          + z * (p[55] + x * (p[76]) + y * (p[82]) + z * (p[83]))))))
      }
      _ => return None,
    })
  }
}
