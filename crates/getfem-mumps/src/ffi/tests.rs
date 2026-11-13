//! Field offsets and size of DMUMPS_STRUC_C against the C header
//! (ffi/layout.txt, from ffi/layout.c: gcc layout.c && ./a.out > layout.txt).

use super::DMUMPS_STRUC_C;
use std::mem::offset_of;

#[test]
fn layout_matches_c_header() {
  let expected: Vec<(String, usize)> = include_str!("layout.txt")
    .lines()
    .map(|l| {
      let mut w = l.split_whitespace();
      (
        w.next().unwrap().to_string(),
        w.next().unwrap().parse().unwrap(),
      )
    })
    .collect();
  let ours: Vec<(&str, usize)> = vec![
    ("size", std::mem::size_of::<DMUMPS_STRUC_C>()),
    ("sym", offset_of!(DMUMPS_STRUC_C, sym)),
    ("par", offset_of!(DMUMPS_STRUC_C, par)),
    ("job", offset_of!(DMUMPS_STRUC_C, job)),
    ("comm_fortran", offset_of!(DMUMPS_STRUC_C, comm_fortran)),
    ("icntl", offset_of!(DMUMPS_STRUC_C, icntl)),
    ("keep", offset_of!(DMUMPS_STRUC_C, keep)),
    ("cntl", offset_of!(DMUMPS_STRUC_C, cntl)),
    ("dkeep", offset_of!(DMUMPS_STRUC_C, dkeep)),
    ("keep8", offset_of!(DMUMPS_STRUC_C, keep8)),
    ("n", offset_of!(DMUMPS_STRUC_C, n)),
    ("nblk", offset_of!(DMUMPS_STRUC_C, nblk)),
    ("nz_alloc", offset_of!(DMUMPS_STRUC_C, nz_alloc)),
    ("nz", offset_of!(DMUMPS_STRUC_C, nz)),
    ("nnz", offset_of!(DMUMPS_STRUC_C, nnz)),
    ("irn", offset_of!(DMUMPS_STRUC_C, irn)),
    ("jcn", offset_of!(DMUMPS_STRUC_C, jcn)),
    ("a", offset_of!(DMUMPS_STRUC_C, a)),
    ("nz_loc", offset_of!(DMUMPS_STRUC_C, nz_loc)),
    ("nnz_loc", offset_of!(DMUMPS_STRUC_C, nnz_loc)),
    ("irn_loc", offset_of!(DMUMPS_STRUC_C, irn_loc)),
    ("jcn_loc", offset_of!(DMUMPS_STRUC_C, jcn_loc)),
    ("a_loc", offset_of!(DMUMPS_STRUC_C, a_loc)),
    ("nelt", offset_of!(DMUMPS_STRUC_C, nelt)),
    ("eltptr", offset_of!(DMUMPS_STRUC_C, eltptr)),
    ("eltvar", offset_of!(DMUMPS_STRUC_C, eltvar)),
    ("a_elt", offset_of!(DMUMPS_STRUC_C, a_elt)),
    ("blkptr", offset_of!(DMUMPS_STRUC_C, blkptr)),
    ("blkvar", offset_of!(DMUMPS_STRUC_C, blkvar)),
    ("perm_in", offset_of!(DMUMPS_STRUC_C, perm_in)),
    ("sym_perm", offset_of!(DMUMPS_STRUC_C, sym_perm)),
    ("uns_perm", offset_of!(DMUMPS_STRUC_C, uns_perm)),
    ("colsca", offset_of!(DMUMPS_STRUC_C, colsca)),
    ("rowsca", offset_of!(DMUMPS_STRUC_C, rowsca)),
    (
      "colsca_from_mumps",
      offset_of!(DMUMPS_STRUC_C, colsca_from_mumps),
    ),
    (
      "rowsca_from_mumps",
      offset_of!(DMUMPS_STRUC_C, rowsca_from_mumps),
    ),
    ("colsca_loc", offset_of!(DMUMPS_STRUC_C, colsca_loc)),
    ("rowsca_loc", offset_of!(DMUMPS_STRUC_C, rowsca_loc)),
    ("rowind", offset_of!(DMUMPS_STRUC_C, rowind)),
    ("colind", offset_of!(DMUMPS_STRUC_C, colind)),
    ("pivots", offset_of!(DMUMPS_STRUC_C, pivots)),
    ("rhs", offset_of!(DMUMPS_STRUC_C, rhs)),
    ("redrhs", offset_of!(DMUMPS_STRUC_C, redrhs)),
    ("rhs_sparse", offset_of!(DMUMPS_STRUC_C, rhs_sparse)),
    ("sol_loc", offset_of!(DMUMPS_STRUC_C, sol_loc)),
    ("rhs_loc", offset_of!(DMUMPS_STRUC_C, rhs_loc)),
    ("rhsintr", offset_of!(DMUMPS_STRUC_C, rhsintr)),
    ("irhs_sparse", offset_of!(DMUMPS_STRUC_C, irhs_sparse)),
    ("irhs_ptr", offset_of!(DMUMPS_STRUC_C, irhs_ptr)),
    ("isol_loc", offset_of!(DMUMPS_STRUC_C, isol_loc)),
    ("irhs_loc", offset_of!(DMUMPS_STRUC_C, irhs_loc)),
    ("glob2loc_rhs", offset_of!(DMUMPS_STRUC_C, glob2loc_rhs)),
    ("glob2loc_sol", offset_of!(DMUMPS_STRUC_C, glob2loc_sol)),
    ("nrhs", offset_of!(DMUMPS_STRUC_C, nrhs)),
    ("lrhs", offset_of!(DMUMPS_STRUC_C, lrhs)),
    ("lredrhs", offset_of!(DMUMPS_STRUC_C, lredrhs)),
    ("nz_rhs", offset_of!(DMUMPS_STRUC_C, nz_rhs)),
    ("lsol_loc", offset_of!(DMUMPS_STRUC_C, lsol_loc)),
    ("nloc_rhs", offset_of!(DMUMPS_STRUC_C, nloc_rhs)),
    ("lrhs_loc", offset_of!(DMUMPS_STRUC_C, lrhs_loc)),
    ("nsol_loc", offset_of!(DMUMPS_STRUC_C, nsol_loc)),
    ("schur_mloc", offset_of!(DMUMPS_STRUC_C, schur_mloc)),
    ("schur_nloc", offset_of!(DMUMPS_STRUC_C, schur_nloc)),
    ("schur_lld", offset_of!(DMUMPS_STRUC_C, schur_lld)),
    ("mblock", offset_of!(DMUMPS_STRUC_C, mblock)),
    ("nblock", offset_of!(DMUMPS_STRUC_C, nblock)),
    ("nprow", offset_of!(DMUMPS_STRUC_C, nprow)),
    ("npcol", offset_of!(DMUMPS_STRUC_C, npcol)),
    ("ld_rhsintr", offset_of!(DMUMPS_STRUC_C, ld_rhsintr)),
    ("info", offset_of!(DMUMPS_STRUC_C, info)),
    ("infog", offset_of!(DMUMPS_STRUC_C, infog)),
    ("rinfo", offset_of!(DMUMPS_STRUC_C, rinfo)),
    ("rinfog", offset_of!(DMUMPS_STRUC_C, rinfog)),
    ("pivnul_list", offset_of!(DMUMPS_STRUC_C, pivnul_list)),
    ("mapping", offset_of!(DMUMPS_STRUC_C, mapping)),
    (
      "singular_values",
      offset_of!(DMUMPS_STRUC_C, singular_values),
    ),
    (
      "nb_singular_values",
      offset_of!(DMUMPS_STRUC_C, nb_singular_values),
    ),
    ("size_schur", offset_of!(DMUMPS_STRUC_C, size_schur)),
    ("listvar_schur", offset_of!(DMUMPS_STRUC_C, listvar_schur)),
    ("schur", offset_of!(DMUMPS_STRUC_C, schur)),
    ("wk_user", offset_of!(DMUMPS_STRUC_C, wk_user)),
    ("version_number", offset_of!(DMUMPS_STRUC_C, version_number)),
    ("ooc_tmpdir", offset_of!(DMUMPS_STRUC_C, ooc_tmpdir)),
    ("ooc_prefix", offset_of!(DMUMPS_STRUC_C, ooc_prefix)),
    ("write_problem", offset_of!(DMUMPS_STRUC_C, write_problem)),
    ("lwk_user", offset_of!(DMUMPS_STRUC_C, lwk_user)),
    ("save_dir", offset_of!(DMUMPS_STRUC_C, save_dir)),
    ("save_prefix", offset_of!(DMUMPS_STRUC_C, save_prefix)),
    ("metis_options", offset_of!(DMUMPS_STRUC_C, metis_options)),
    (
      "instance_number",
      offset_of!(DMUMPS_STRUC_C, instance_number),
    ),
  ];
  assert_eq!(ours.len(), expected.len());
  for ((a, x), (b, y)) in ours.iter().zip(&expected) {
    assert_eq!((*a, *x), (b.as_str(), *y), "field {b}");
  }
}
