#include <stdio.h>
#include <stdlib.h>
#include "function_declarations.h"

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <num_iters> <kmax> <jmax> <imax>\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    int num_iters = atoi(argv[1]);
    int kmax = atoi(argv[2]);
    int jmax = atoi(argv[3]);
    int imax = atoi(argv[4]);

    double ***arr1, ***arr2;
    
    /* Allocate two 3D arrays using allocate_array3D */
    allocate_array3D(kmax, jmax, imax, &arr1);
    allocate_array3D(kmax, jmax, imax, &arr2);
    
    /* Initialize arr1 and arr2 with identical nonconstant values.
       Here we use the sum of indices as an example function. */
    for (int k = 0; k < kmax; k++) {
        for (int j = 0; j < jmax; j++) {
            for (int i = 0; i < imax; i++) {
                double val = (double)(k + j + i);  // Example nonconstant initialization
                arr1[k][j][i] = val;
                arr2[k][j][i] = val;
            }
        }
    }
    
    /* Perform GS iterations using both the standard and two-chunk methods. */
    for (int n = 0; n < num_iters; n++) {
        GS_iteration_normal(kmax, jmax, imax, arr1);
        GS_iteration_2_chunks(kmax, jmax, imax, arr2);
    }
    
    /* Compute the Euclidean distance between the two arrays to verify results. */
    double diff = euclidean_distance(kmax, jmax, imax, arr1, arr2);
    printf("num_iters=%d, kmax=%d, jmax=%d, imax=%d, diff=%g\n", num_iters, kmax, jmax, imax, diff);
    
    /* Correct deallocation:
         Since allocate_array3D allocates:
            - arr (top-level pointer array) via malloc,
            - arr[0] (2D pointer block) via malloc,
            - arr[0][0] (data block) via malloc,
         we must free them in reverse order:
    */
    free(arr1[0][0]);  // Free the contiguous data block.
    free(arr1[0]);     // Free the 2D pointer block.
    free(arr1);        // Free the top-level pointer array.

    free(arr2[0][0]);  // Same for arr2.
    free(arr2[0]);
    free(arr2);

    return EXIT_SUCCESS;
}
