#include <mpi.h>
#include "function_declarations_mpi.h"

void GS_iteration_2_chunks_mpi(int rank,
                               int kmax,
                               int my_jmax,
                               int imax,
                               double ***phi)
{
    int count = imax;
    int dest, source, tag;

    if (rank == 0) {
        dest = source = 1;
        // first wavefront k=1
        for (int j = 1; j < my_jmax-1; ++j)
            for (int i = 1; i < imax-1; ++i)
                phi[1][j][i] = (phi[0][j][i]
                              + phi[1][j-1][i]
                              + phi[1][j][i-1]
                              + phi[1][j][i+1]
                              + phi[1][j+1][i]
                              + phi[2][j][i]) / 6.0;
        tag = 2;
        MPI_Send(phi[1][my_jmax-2], count, MPI_DOUBLE,
                 dest, tag, MPI_COMM_WORLD);

        // wavefronts k=2..kmax-2
        for (int k = 2; k <= kmax-2; ++k) {
            tag = 3;
            MPI_Recv(phi[k-1][my_jmax-1], count, MPI_DOUBLE,
                     source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 1; j < my_jmax-1; ++j)
                for (int i = 1; i < imax-1; ++i)
                    phi[k][j][i] = (phi[k-1][j][i]
                                  + phi[k][j-1][i]
                                  + phi[k][j][i-1]
                                  + phi[k][j][i+1]
                                  + phi[k][j+1][i]
                                  + phi[k+1][j][i]) / 6.0;
            tag = 4;
            MPI_Send(phi[k][my_jmax-2], count, MPI_DOUBLE,
                     dest, tag, MPI_COMM_WORLD);
        }
        tag = 5;
        MPI_Recv(phi[kmax-2][my_jmax-1], count, MPI_DOUBLE,
                 source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        dest = source = 0;
        tag = 2;
        MPI_Recv(phi[1][0], count, MPI_DOUBLE,
                 source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int k = 2; k <= kmax-2; ++k) {
            for (int j = 1; j < my_jmax-1; ++j)
                for (int i = 1; i < imax-1; ++i)
                    phi[k-1][j][i] = (phi[k-2][j][i]
                                  + phi[k-1][j-1][i]
                                  + phi[k-1][j][i-1]
                                  + phi[k-1][j][i+1]
                                  + phi[k-1][j+1][i]
                                  + phi[k][j][i]) / 6.0;
            tag = 3;
            MPI_Send(phi[k-1][1], count, MPI_DOUBLE,
                     dest, tag, MPI_COMM_WORLD);
            tag = 4;
            MPI_Recv(phi[k][0], count, MPI_DOUBLE,
                     source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        for (int j = 1; j < my_jmax-1; ++j)
            for (int i = 1; i < imax-1; ++i)
                phi[kmax-2][j][i] = (phi[kmax-3][j][i]
                                  + phi[kmax-2][j-1][i]
                                  + phi[kmax-2][j][i-1]
                                  + phi[kmax-2][j][i+1]
                                  + phi[kmax-2][j+1][i]
                                  + phi[kmax-1][j][i]) / 6.0;
        tag = 5;
        MPI_Send(phi[kmax-2][1], count, MPI_DOUBLE,
                 dest, tag, MPI_COMM_WORLD);
    }
}