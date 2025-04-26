#include <stdio.h>
#include <stdlib.h>
#include "math.h"
#include "function_declarations_mpi.h"

void allocate_array3D_mpi(int kmax, int jmax, int imax, double ****array) {
    // Allocate memory for the 3D array
    *array = (double ***)malloc(kmax * sizeof(double **));
    if (*array == NULL) {
        fprintf(stderr, "Memory allocation failed for the first dimension of 3D array.\n");
        exit(EXIT_FAILURE);
    }
    for (int k = 0; k < kmax; k++) {
        (*array)[k] = (double **)malloc(jmax * sizeof(double *));
        if ((*array)[k] == NULL) {
            fprintf(stderr, "Memory allocation failed for the second dimension of 3D array.\n");
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < jmax; j++) {
            (*array)[k][j] = (double *)malloc(imax * sizeof(double));
            if ((*array)[k][j] == NULL) {
                fprintf(stderr, "Memory allocation failed for the third dimension of 3D array.\n");
                exit(EXIT_FAILURE);
            }
        }
    }

    // Initialize the array wiht each element set to its flattened index
    for (int k = 0; k < kmax; k++) {
        for (int j = 0; j < jmax; j++) {
            for (int i = 0; i < imax; i++) {
                (*array)[k][j][i] = (double)pow(k * jmax * imax + 
                                                j * imax + 
                                                i, 2);
            }
        }
    }
}