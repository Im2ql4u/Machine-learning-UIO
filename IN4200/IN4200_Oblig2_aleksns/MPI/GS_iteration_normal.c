#include "function_declarations_mpi.h"

void GS_iteration_normal(int kmax, int jmax, int imax, double ***phi) {
    /* Update only interior points: k=1..(kmax-2), j=1..(jmax-2), i=1..(imax-2) */
    for (int k = 1; k < kmax-1; k++) {
        for (int j = 1; j < jmax-1; j++) {
            for (int i = 1; i < imax-1; i++) {
                phi[k][j][i] = (phi[k-1][j][i] + phi[k][j-1][i] +
                                phi[k][j][i-1] + phi[k][j][i+1] +
                                phi[k][j+1][i] + phi[k+1][j][i]) / 6.0;
            }
        }
    }
}
