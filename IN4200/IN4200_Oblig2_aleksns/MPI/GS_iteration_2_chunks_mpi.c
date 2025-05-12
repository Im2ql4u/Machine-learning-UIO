#include <mpi.h>
#include "function_declarations_mpi.h"

void GS_iteration_2_chunks_mpi(int my_rank,
                                   int kmax,
                                   int my_jmax,
                                   int imax,
                                   double ***phi) {
    int partner    = (my_rank == 0 ? 1 : 0);
    int count      = imax;
    int tag1       = 1;
    int tag_prev   = 2;
    int tag_next   = 3;
    int tag4       = 4;
    MPI_Request reqs[2];
    MPI_Status  stats[2];

    // Initial wavefront at k=1
    if (my_rank == 0) {
        int k = 1;
        for (int j = 1; j < my_jmax - 1; j++) {
            for (int i = 1; i < imax - 1; i++) {
                phi[k][j][i] = (
                    phi[k-1][j][i]     +
                    phi[k][j-1][i]     +
                    phi[k][j][i-1]     +
                    phi[k][j][i+1]     +
                    phi[k][j+1][i]     +
                    phi[k+1][j][i]
                ) / 6.0;
            }
        }
        // Send boundary row to rank 1
        MPI_Isend(
            phi[1][my_jmax - 2],
            count,
            MPI_DOUBLE,
            partner,
            tag1,
            MPI_COMM_WORLD,
            &reqs[0]
        );
        MPI_Wait(&reqs[0], &stats[0]);
    } else {
        // Rank 1 receives initial ghost row for k=1
        MPI_Recv(
            phi[1][0],
            count,
            MPI_DOUBLE,
            partner,
            tag1,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );
    }

    // Iterative sweep for k = 2 ... kmax-2
    for (int k = 2; k <= kmax - 2; k++) {
        if (my_rank == 0) {
            // Post non-blocking receive of prev-layer halo from rank 1
            MPI_Irecv(
                phi[k-1][my_jmax - 1],
                count,
                MPI_DOUBLE,
                partner,
                tag_prev,
                MPI_COMM_WORLD,
                &reqs[0]
            );

            // Compute interior of slice k
            for (int j = 1; j < my_jmax - 1; j++) {
                for (int i = 1; i < imax - 1; i++) {
                    phi[k][j][i] = (
                        phi[k-1][j][i]     +
                        phi[k][j-1][i]     +
                        phi[k][j][i-1]     +
                        phi[k][j][i+1]     +
                        phi[k][j+1][i]     +
                        phi[k+1][j][i]
                    ) / 6.0;
                }
            }

            // Send current-layer boundary to rank 1
            MPI_Isend(
                phi[k][my_jmax - 2],
                count,
                MPI_DOUBLE,
                partner,
                tag_next,
                MPI_COMM_WORLD,
                &reqs[1]
            );

        } else {
            // Post non-blocking receive of next-layer halo from rank 0
            MPI_Irecv(
                phi[k][0],
                count,
                MPI_DOUBLE,
                partner,
                tag_next,
                MPI_COMM_WORLD,
                &reqs[0]
            );

            // Compute interior of slice k-1
            for (int j = 1; j < my_jmax - 1; j++) {
                for (int i = 1; i < imax - 1; i++) {
                    phi[k-1][j][i] = (
                        phi[k-2][j][i]     +
                        phi[k-1][j-1][i]   +
                        phi[k-1][j][i-1]   +
                        phi[k-1][j][i+1]   +
                        phi[k-1][j+1][i]   +
                        phi[k][j][i]
                    ) / 6.0;
                }
            }

            // Send prev-layer boundary to rank 0
            MPI_Isend(
                phi[k-1][1],
                count,
                MPI_DOUBLE,
                partner,
                tag_prev,
                MPI_COMM_WORLD,
                &reqs[1]
            );
        }

        // Wait for both non-blocking operations to complete
        MPI_Waitall(2, reqs, stats);
    }

    // Final wavefront at k = kmax - 2
    if (my_rank == 1) {
        int k = kmax - 2;
        for (int j = 1; j < my_jmax - 1; j++) {
            for (int i = 1; i < imax - 1; i++) {
                phi[k][j][i] = (
                    phi[k-1][j][i]     +
                    phi[k][j-1][i]     +
                    phi[k][j][i-1]     +
                    phi[k][j][i+1]     +
                    phi[k][j+1][i]     +
                    phi[k+1][j][i]
                ) / 6.0;
            }
        }
        MPI_Send(
            phi[kmax - 2][1],
            count,
            MPI_DOUBLE,
            partner,
            tag4,
            MPI_COMM_WORLD
        );
    } else {
        MPI_Recv(
            phi[kmax - 2][my_jmax - 1],
            count,
            MPI_DOUBLE,
            partner,
            tag4,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );
    }
}