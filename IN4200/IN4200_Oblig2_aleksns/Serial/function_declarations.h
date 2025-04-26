#ifndef FUNCTION_DECLARATIONS_H
#define FUNCTION_DECLARATIONS_H

/* Allocates a 3D array of size kmax × jmax × imax. */
void allocate_array3D(int kmax, int jmax, int imax, double ****array);

/* Computes the Euclidean distance between two 3D arrays. */
double euclidean_distance(int kmax, int jmax, int imax, double ***arr1, double ***arr2);

/* Standard serial Gauss-Seidel iteration (updates interior points). */
void GS_iteration_normal(int kmax, int jmax, int imax, double ***phi);

/* Gauss-Seidel iteration using a two-chunk wavefront traversal. */
void GS_iteration_2_chunks(int kmax, int jmax, int imax, double ***phi);

#endif // FUNCTION_DECLARATIONS_H


