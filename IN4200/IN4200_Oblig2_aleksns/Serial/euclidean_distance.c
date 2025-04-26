#include <math.h>
#include <stdio.h>
#include "function_declarations.h"

double euclidean_distance(int kmax, int jmax, int imax, double ***arr1, double ***arr2) {
    double sum = 0.0;
    for (int k = 0; k < kmax; k++) {
        for (int j = 0; j < jmax; j++) {
            for (int i = 0; i < imax; i++) {
                double diff = arr1[k][j][i] - arr2[k][j][i];
                sum += diff * diff;
            }
        }
    }
    return sqrt(sum);
}
