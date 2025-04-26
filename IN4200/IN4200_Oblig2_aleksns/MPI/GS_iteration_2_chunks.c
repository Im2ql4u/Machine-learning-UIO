#include "function_declarations_mpi.h"

void GS_iteration_2_chunks(int kmax, int jmax, int imax, double ***phi) {
    // first wavefront: only computation on left chunk at level k=1
    for (int j=1; j<jmax-1; j++) {
        for (int i=1; i<imax-1; i++) {
            phi[1][j][i] = (phi[0][j][i]   + phi[1][j-1][i]
                +phi[1][j][i-1] + phi[1][j][i+1]
                +phi[1][j+1][i] + phi[2][j][i])/6.0;
        }
    }
    for (int k=2; k<=kmax-2; k++) {
        // computation on left chunk at level k
        for (int j=1; j<jmax-1; j++) {
            for (int i=1; i<imax-1; i++) {
                phi[k][j][i] = (phi[k-1][j][i] + phi[k][j-1][i]
                               +phi[k][j][i-1] + phi[k][j][i+1]
                               +phi[k][j+1][i] + phi[k+1][j][i])/6.0;
            }
        }

        // computation on right chunk at level k-1
        for (int j=1; j<jmax-1; j++) {
            for (int i=1; i<imax-1; i++) {
                phi[k-1][j][i] = (phi[k-2][j][i]  + phi[k-1][j-1][i]
                                +phi[k-1][j][i-1] + phi[k-1][j][i+1]
                                +phi[k-1][j+1][i] + phi[k][j][i])/6.0;
            }
        }
    }
    // last wavefront: only computation on right chunk at level k=kmax-2
    for (int j=1; j<jmax-1; j++) {
        for (int i=1; i<imax-1; i++) {
            phi[kmax-2][j][i] = (phi[kmax-3][j][i]  + phi[kmax-2][j-1][i]
                               +phi[kmax-2][j][i-1] + phi[kmax-2][j][i+1]
                               +phi[kmax-2][j+1][i] + phi[kmax-1][j][i])/6.0;
        }
    }
}