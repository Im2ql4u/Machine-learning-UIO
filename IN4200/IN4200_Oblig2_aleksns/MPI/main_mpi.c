#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "function_declarations_mpi.h"
#include <time.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_iter, kmax, jmax, imax;

    if (rank == 0) {
        printf("\n====================================\n");

        if (argc != 5) {
            printf("Usage: %s <num_iter> <kmax> <jmax> <imax>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        num_iter = atoi(argv[1]);
        kmax     = atoi(argv[2]);
        jmax     = atoi(argv[3]);
        imax     = atoi(argv[4]);

        if (num_iter <= 0 || kmax <= 0 || jmax <= 0 || imax <= 0) {
            printf("Error: All dimensions must be greater than 0.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&num_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&kmax,     1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&jmax,     1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&imax,     1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Rank %d: num_iter = %d, kmax = %d, jmax = %d, imax = %d\n", rank, num_iter, kmax, jmax, imax);

    double ***my_array;
    double ***my_local_array;
    int my_jmax = jmax / 2 + 1;

    allocate_array3D(kmax, jmax, imax, &my_array);
    allocate_array3D(kmax, my_jmax, imax, &my_local_array);

    for (int k = 0; k < kmax; k++) {
        for (int j = 0; j < jmax; j++) {
            for (int i = 0; i < imax; i++) {
                double value = (double)(k * (k + 1) + (j - 1) * (i + 2));
                my_array[k][j][i] = value;

                if (rank == 0 && j < my_jmax) {
                    my_local_array[k][j][i] = value;
                }

                if (rank == 1 && j >= my_jmax - 2) {
                    int j_shifted = j - my_jmax + 2;
                    my_local_array[k][j_shifted][i] = value;
                }
            }
        }
    }

    double serial_time = 0.0;
    if (rank == 0) {
        clock_t start = clock();
        for (int i = 0; i < num_iter; i++) {
            GS_iteration_2_chunks(kmax, jmax, imax, my_array);
        }
        clock_t end = clock();
        serial_time = (double)(end - start) / CLOCKS_PER_SEC;
    }


    MPI_Barrier(MPI_COMM_WORLD);
    double mpi_start = MPI_Wtime();
    for (int i = 0; i < num_iter; i++) {
        GS_iteration_2_chunks_mpi(rank, kmax, my_jmax, imax, my_local_array);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double mpi_end = MPI_Wtime();
    double mpi_time = mpi_end - mpi_start;

    if (rank == 0) {
        int tag;
        int count = imax;
        int source = 1;

        double ***global_phi = NULL;
        double ***incoming_array = NULL;

        allocate_array3D(kmax, jmax, imax, &global_phi);
        allocate_array3D(kmax, my_jmax, imax, &incoming_array);

        for (int k = 0; k < kmax; k++) {
            for (int j = 0; j < my_jmax; j++) {
                tag = k + j;
                MPI_Recv(incoming_array[k][j], count, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        for (int k = 0; k < kmax; k++) {
            for (int j = 0; j < jmax; j++) {
                for (int i = 0; i < imax; i++) {
                    if (j < my_jmax) {
                        global_phi[k][j][i] = my_local_array[k][j][i];
                    } else {
                        int j_shifted = j - my_jmax + 2;
                        global_phi[k][j][i] = incoming_array[k][j_shifted][i];
                    }
                }
            }
        }

        double distance = euclidean_distance(kmax, jmax, imax, my_array, global_phi);
        printf("num iters=%d, kmax=%d, jmax=%d, imax=%d, diff=%g\n",
            num_iter, kmax, jmax, imax, distance);
        printf("Serial time: %.6f seconds\n", serial_time);
        printf("MPI time: %.6f seconds\n", mpi_time);

        for (int k = 0; k < kmax; k++) {
            for (int j = 0; j < jmax; j++) {
                free(global_phi[k][j]);
            }
            free(global_phi[k]);
        }
        free(global_phi);

        for (int k = 0; k < kmax; k++) {
            for (int j = 0; j < my_jmax; j++) {
                free(incoming_array[k][j]);
            }
            free(incoming_array[k]);
        }
        free(incoming_array);
    }

    if (rank == 1) {
        int tag;
        int destination = 0;
        int count = imax;

        for (int k = 0; k < kmax; k++) {
            for (int j = 0; j < my_jmax; j++) {
                tag = k + j;
                MPI_Send(my_local_array[k][j], count, MPI_DOUBLE, destination, tag, MPI_COMM_WORLD);
            }
        }
    }

    for (int k = 0; k < kmax; k++) {
        for (int j = 0; j < jmax; j++) {
            free(my_array[k][j]);
        }
        free(my_array[k]);
    }
    free(my_array);

    for (int k = 0; k < kmax; k++) {
        for (int j = 0; j < my_jmax; j++) {
            free(my_local_array[k][j]);
        }
        free(my_local_array[k]);
    }
    free(my_local_array);

    MPI_Finalize();
    return 0;
}
