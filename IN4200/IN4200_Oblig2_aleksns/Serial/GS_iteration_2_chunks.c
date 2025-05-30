#include "function_declarations.h"

void GS_iteration_2_chunks(int kmax, int jmax, int imax, double ***phi) {
    int k, j, i;

    /* First wavefront: process left chunk at level k = 1 */
    for (j = 1; j < jmax/2; j++) {
        for (i = 1; i < imax-1; i++) {
            phi[1][j][i] = (phi[0][j][i] + phi[1][j-1][i] + phi[1][j][i-1] +
                            phi[1][j][i+1] + phi[1][j+1][i] + phi[2][j][i]) / 6.0;
        }
    }

    /* Wavefront traversal */
    for (k = 2; k <= kmax-2; k++) {
        /* Compute left chunk at level k: j = 1 to (jmax/2 - 1) */
        for (j = 1; j < jmax/2; j++) {
            for (i = 1; i < imax-1; i++) {
                phi[k][j][i] = (phi[k-1][j][i] + phi[k][j-1][i] + phi[k][j][i-1] +
                                phi[k][j][i+1] + phi[k][j+1][i] + phi[k+1][j][i]) / 6.0;
            }
        }
        /* Compute right chunk for previous level (k-1): j = jmax/2 to jmax-2 */
        for (j = jmax/2; j < jmax-1; j++) {
            for (i = 1; i < imax-1; i++) {
                phi[k-1][j][i] = (phi[k-1][j-1][i] + phi[k-2][j][i] + phi[k-1][j][i-1] +
                                  phi[k-1][j][i+1] + phi[k-1][j+1][i] + phi[k][j][i]) / 6.0;
            }
        }
    }

    /* Last wavefront: process right chunk at level k = kmax-2 */
    for (j = jmax/2; j < jmax-1; j++) {
        for (i = 1; i < imax-1; i++) {
            phi[kmax-2][j][i] = (phi[kmax-2][j-1][i] + phi[kmax-3][j][i] + phi[kmax-2][j][i-1] +
                                 phi[kmax-2][j][i+1] + phi[kmax-2][j+1][i] + phi[kmax-1][j][i]) / 6.0;
        }
    }
}
