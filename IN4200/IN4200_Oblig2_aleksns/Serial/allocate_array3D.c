#include <stdio.h>
#include <stdlib.h>
#include "function_declarations.h"

void allocate_array3D(int kmax, int jmax, int imax, double ****array) {
    double ***arr = malloc(kmax * sizeof(double **));
    if (!arr) { perror("malloc arr"); exit(EXIT_FAILURE); }
    arr[0] = malloc(kmax * jmax * sizeof(double *));
    if (!arr[0]) { perror("malloc arr[0]"); exit(EXIT_FAILURE); }
    arr[0][0] = malloc(kmax * jmax * imax * sizeof(double));
    if (!arr[0][0]) { perror("malloc arr[0][0]"); exit(EXIT_FAILURE); }
    for (int k = 0; k < kmax; k++) {
        if (k > 0)
            arr[k] = arr[k-1] + jmax;
        for (int j = 0; j < jmax; j++) {
            arr[k][j] = arr[0][0] + (k * jmax * imax) + (j * imax);
        }
    }
    *array = arr;
}
