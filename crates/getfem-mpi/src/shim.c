/* The MPI calls of getfem-mpi: MPI's handles (MPI_COMM_WORLD, MPI_DOUBLE,
   MPI_SUM) differ between implementations, so they stay on the C side. */
#include <mpi.h>
#include <stdlib.h>

int getfem_mpi_initialized(int *flag) { return MPI_Initialized(flag); }

/* funneled: the threads' level asked provided (single-threaded ranks: MPI
   called by the thread of init only) */
int getfem_mpi_init(int *funneled) {
  int provided, e = MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
  *funneled = provided >= MPI_THREAD_FUNNELED;
  return e;
}

int getfem_mpi_finalize(void) { return MPI_Finalize(); }

int getfem_mpi_finalized(int *flag) { return MPI_Finalized(flag); }

/* funneled: an MPI started by someone else (mpi4py) provides at least
   MPI_THREAD_FUNNELED */
int getfem_mpi_query_funneled(int *funneled) {
  int provided, e = MPI_Query_thread(&provided);
  *funneled = provided >= MPI_THREAD_FUNNELED;
  return e;
}

int getfem_mpi_rank(int *rank) { return MPI_Comm_rank(MPI_COMM_WORLD, rank); }

int getfem_mpi_size(int *size) { return MPI_Comm_size(MPI_COMM_WORLD, size); }

int getfem_mpi_sum(double *v, int n) {
  return MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

int getfem_mpi_max(double *v, int n) {
  return MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
}

/* every rank's v (n values) concatenated in rank order: the counts of all
   ranks, then the values */
int getfem_mpi_allgather_counts(int n, int *counts) {
  return MPI_Allgather(&n, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);
}

int getfem_mpi_allgatherv(const double *v, int n, double *out, const int *counts,
                          const int *displs) {
  return MPI_Allgatherv(v, n, MPI_DOUBLE, out, counts, displs, MPI_DOUBLE,
                        MPI_COMM_WORLD);
}

int getfem_mpi_bcast(double *v, int n, int root) {
  return MPI_Bcast(v, n, MPI_DOUBLE, root, MPI_COMM_WORLD);
}


/* METIS k-way partition of a graph (CSR: xadj, adjncy) into nparts, as
   GetFEM mesh::compute_mpi_region; idx_t of the METIS installed */
#include <metis.h>
#include <stdint.h>

int getfem_metis_part(int64_t n, const int64_t *xadj, const int64_t *adjncy,
                      int64_t nparts, int64_t *part) {
  int64_t ne = xadj[n], i;
  idx_t nv = (idx_t)n, ncon = 1, np = (idx_t)nparts, edgecut, e;
  idx_t options[METIS_NOPTIONS];
  idx_t *x = malloc(sizeof(idx_t) * (n + 1));
  idx_t *a = malloc(sizeof(idx_t) * (ne > 0 ? ne : 1));
  idx_t *p = malloc(sizeof(idx_t) * (n > 0 ? n : 1));
  if (!x || !a || !p) { free(x); free(a); free(p); return -1; }
  for (i = 0; i <= n; ++i) x[i] = (idx_t)xadj[i];
  for (i = 0; i < ne; ++i) a[i] = (idx_t)adjncy[i];
  METIS_SetDefaultOptions(options);
  /* the same parts on every rank */
  options[METIS_OPTION_SEED] = 1;
  e = METIS_PartGraphKway(&nv, &ncon, x, a, NULL, NULL, NULL, &np, NULL,
                          NULL, options, &edgecut, p);
  for (i = 0; i < n; ++i) part[i] = (int64_t)p[i];
  free(x); free(a); free(p);
  return e == METIS_OK ? 0 : (int)e;
}

/* MUMPS' communicators (its comm_fortran): all ranks after init, or each
   process alone, MPI initialized here if needed (finalized at exit;
   serialized: such a program may call MUMPS from any thread, under its
   lock) */
int getfem_mpi_comm_world_fortran(void) { return (int)MPI_Comm_c2f(MPI_COMM_WORLD); }

static void getfem_mpi_finalize_at_exit(void) {
  int f;
  MPI_Finalized(&f);
  if (!f) MPI_Finalize();
}

int getfem_mpi_comm_self_fortran(int *comm) {
  int f, provided, e;
  MPI_Initialized(&f);
  if (!f) {
    e = MPI_Init_thread(NULL, NULL, MPI_THREAD_SERIALIZED, &provided);
    if (e) return e;
    atexit(getfem_mpi_finalize_at_exit);
  }
  *comm = (int)MPI_Comm_c2f(MPI_COMM_SELF);
  return 0;
}
