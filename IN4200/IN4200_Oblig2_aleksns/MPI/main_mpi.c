// main_mpi.c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <mpi.h>
#include "function_declarations_mpi.h"

int main(int argc, char **argv) {
    int num_iters, kmax, jmax, imax;
    bool print_verbose = false;
    int my_rank, num_procs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    if (num_procs != 2) {
        if (my_rank == 0)
            fprintf(stderr, "This program requires exactly 2 processes.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // --- Process 0 reads arguments, broadcasts to process 1 ---
    if (my_rank == 0) {
        if (argc != 5) {
            fprintf(stderr, "Usage: %s <iters> <kmax> <jmax> <imax>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        num_iters     = atoi(argv[1]);
        kmax          = atoi(argv[2]);
        jmax          = atoi(argv[3]);
        imax          = atoi(argv[4]);
        print_verbose = true;
    }
    MPI_Bcast(&num_iters,     1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&kmax,          1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&jmax,          1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&imax,          1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&print_verbose, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    // Compute local dimension (j-direction)
    int left_rows = jmax / 2;
    int my_jmax   = left_rows + 1;  // interior rows plus 1 ghost

    // Allocate local phi array
    double ***my_phi = NULL;
    allocate_array3D_mpi(kmax, my_jmax, imax, &my_phi);

    // Process 0 also needs full-sized arrays for serial and global result
    double ***benchmark = NULL;
    double ***global_phi = NULL;
    if (my_rank == 0) {
        allocate_array3D(kmax, jmax, imax, &benchmark);
        allocate_array3D(kmax, jmax, imax, &global_phi);
    }

    // Initialize both my_phi and the serial benchmark
    for (int k = 0; k < kmax; k++) {
        for (int j = 0; j < my_jmax; j++) {
            int global_j = (my_rank == 0 ? j : (j + left_rows - 1));
            double val = pow((double)(k*jmax*imax + global_j*imax + 0), 2.0);
            for (int i = 0; i < imax; i++) {
                my_phi[k][j][i] = pow((double)(k*jmax*imax + global_j*imax + i), 2.0);
            }
        }
        if (my_rank == 0) {
            for (int j = 0; j < jmax; j++) {
                for (int i = 0; i < imax; i++) {
                    benchmark[k][j][i] = pow((double)(k*jmax*imax + j*imax + i), 2.0);
                }
            }
        }
    }

    // Header
    if (my_rank == 0) {
        printf(" Iter    Diff\n");
    }

    // Main iteration loop
    for (int it = 1; it <= num_iters; it++) {
        // Perform one MPI Gauss-Seidel chunk step
        GS_iteration_2_chunks_mpi(my_rank, kmax, my_jmax, imax, my_phi);

        if (my_rank == 0) {
            // Serial two-chunk GS on the full benchmark array
            GS_iteration_2_chunks(kmax, jmax, imax, benchmark);

            // Copy left chunk from my_phi into global_phi
            for (int k = 0; k < kmax; k++) {
                for (int j = 0; j < left_rows; j++) {
                    for (int i = 0; i < imax; i++) {
                        global_phi[k][j][i] = my_phi[k][j][i];
                    }
                }
                // Receive right chunk rows from process 1
                for (int j = left_rows; j < jmax; j++) {
                    MPI_Recv(global_phi[k][j], imax, MPI_DOUBLE,
                             1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }

            // Compute and print the Euclidean difference
            double diff = euclidean_distance(kmax, jmax, imax,
                                             benchmark, global_phi);
            printf("%4d   %g\n", it, diff);
            fflush(stdout);
        }
        else {
            // Process 1 sends its computed chunk back to process 0
            for (int k = 0; k < kmax; k++) {
                for (int j = 1; j < my_jmax; j++) {
                    MPI_Send(my_phi[k][j], imax, MPI_DOUBLE,
                             0, 0, MPI_COMM_WORLD);
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
