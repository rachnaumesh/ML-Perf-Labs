#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../kernel/matrix_ops.h"

float **generate_random_matrix(int rows, int cols) {
    float **matrix = (float **)malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (float *)malloc(cols * sizeof(float));
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = (float)rand() / RAND_MAX;
        }
    }
    return matrix;
}

void free_matrix(float **matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

int main() {
    srand(time(NULL));
    int A_rows = 1300, A_cols = 1300, B_rows = 1300, B_cols = 1300;

    if (A_cols != B_rows) {
        printf("Matrix dimensions incompatible for multiplication.\n");
        return 1;
    }

    float **A = generate_random_matrix(A_rows, A_cols);
    float **B = generate_random_matrix(B_rows, B_cols);

    printf("Performing blocking multiplication...\n");
    // float **C = matmul_blocking(A, B, A_rows, A_cols, B_rows, B_cols);
    // float **C = matmul_blocking_optimized(A, B, A_rows, A_cols, B_rows, B_cols);
    // float **C = matmul_blocking_transpose(A, B, A_rows, A_cols, B_rows, B_cols);
    // float **C = matmul_blocking_ikj(A, B, A_rows, A_cols, B_rows, B_cols);
    // float **C = matmul_blocking_jik(A, B, A_rows, A_cols, B_rows, B_cols);
    // float **C = matmul_blocking_kij(A, B, A_rows, A_cols, B_rows, B_cols);
    // float **C = matmul_blocking_dynamic(A, B, A_rows, A_cols, B_rows, B_cols);
    float **C = matmul_blocking_unrolled(A, B, A_rows, A_cols, B_rows, B_cols, 169, 32, 9);

    // Cleanup
    free_matrix(A, A_rows);
    free_matrix(B, B_rows);
    free_matrix(C, A_rows);

    printf("Blocking multiplications completed.\n");

    return 0;
}
