#include "function_declarations_mpi.h"

void GS_iteration_2_chunks (int kmax, int jmax, int imax, double ***phi){
    // first wavefront: only computation on left chunck at level k=1
    int k = 1;
    int j, i;
    for (j=1; j<jmax/2; j++)
        for (i=1; i<imax-1; i++) {
            phi[k][j][i] = ( phi[k-1][j][i] + phi[k][j-1][i]
                            +phi[k][j][i-1] + phi[k][j][i+1]
                            +phi[k][j+1][i] + phi[k+1][j][i])/6.0;
                        }

    for (k=2; k <= kmax-2; k += 1){
        //computation on left chuck at level k
        for (j=1; j<jmax/2; j++)
            for (i=1; i<imax-1; i++){
                phi[k][j][i] = ( phi[k-1][j][i] + phi[k][j-1][i]
                                +phi[k][j][i-1] + phi[k][j][i+1]
                                +phi[k][j+1][i] + phi[k+1][j][i])/6.0;
                            }
        
        //computation on right chucnk at level k-1
        for (j=jmax/2; j<jmax-1; j++)
            for (i=1; i<imax-1; i++) {
                phi[k-1][j][i] = ( phi[k-2][j][i] + phi[k-1][j-1][i]
                                  +phi[k-1][j][i-1] + phi[k-1][j][i+1]
                                  +phi[k-1][j+1][i] + phi[k][j][i])/6.0;
                }
                        
    }

    //last wavefront: only computation on right chunk at level k=kmax-2
    k = kmax-2;
    for (j=jmax/2; j<jmax-1; j++)
        for (i=1; i<imax-1; i++){
            phi[k][j][i] = ( phi[k-1][j][i] + phi[k][j-1][i]
                            +phi[k][j][i-1] + phi[k][j][i+1]
                            +phi[k][j+1][i] + phi[k+1][j][i])/6.0;
                        }
}