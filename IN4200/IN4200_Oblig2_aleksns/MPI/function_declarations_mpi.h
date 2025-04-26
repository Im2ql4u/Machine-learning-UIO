#ifndef FUNCTION_DECLARATIONS_MPI_H
#define FUNCTION_DECLARATIONS_MPI_H

/* ---------------------- Serial Routine Declarations ---------------------- */

/**
 * Allocate a 3D array of size kmax × jmax × imax.
 * The allocated array is returned via the quadruple pointer.
 */
void allocate_array3D(int kmax, int jmax, int imax, double ****array);

/**
 * Perform one Gauss–Seidel iteration over the full 3D domain.
 */
void GS_iteration_normal(int kmax, int jmax, int imax, double ***phi);

/**
 * Perform the two‐chunk (serial) Gauss–Seidel traversal.
 */
void GS_iteration_2_chunks(int kmax, int jmax, int imax, double ***phi);

/**
 * Compute the Euclidean distance between two kmax × jmax × imax arrays.
 */
double euclidean_distance(int kmax, int jmax, int imax,
                          double ***arr1, double ***arr2);


/* ----------------------- MPI Routine Declarations ----------------------- */

/**
 * Allocate a local 3D array of size kmax × my_jmax × imax
 * (where my_jmax = jmax/2 + 1, including one ghost row).
 */
void allocate_array3D_mpi(int kmax, int my_jmax, int imax, double ****array);

/**
 * Perform one Gauss–Seidel wavefront iteration using two MPI processes.
 * my_rank ∈ {0,1}, my_jmax = jmax/2 + 1.
 */
void GS_iteration_2_chunks_mpi(int my_rank,
                               int kmax,
                               int my_jmax,
                               int imax,
                               double ***my_phi);

#endif  /* FUNCTION_DECLARATIONS_MPI_H */

