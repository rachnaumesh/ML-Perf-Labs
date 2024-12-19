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

// Function to generate a random sparse matrix with a given sparsity level
float **generate_sparse_matrix(int rows, int cols, float sparsity) {
    float **matrix = (float **)malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (float *)malloc(cols * sizeof(float));
        for (int j = 0; j < cols; j++) {
            // Generate a non-zero element based on sparsity
            if ((float)rand() / RAND_MAX > sparsity) {
                matrix[i][j] = (float)rand() / RAND_MAX; // Assign a random non-zero value
            } else {
                matrix[i][j] = 0; // Set element to zero to keep sparsity level
            }
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

    float sparsity = 0.1;
    float **A = generate_sparse_matrix(A_rows, A_cols, sparsity);
    float **B = generate_sparse_matrix(B_rows, B_cols, sparsity);

    printf("Performing matmul_sparse...\n");
    float **C = matmul_sparse(A, B, A_rows, A_cols, B_rows, B_cols);

    // Cleanup
    free_matrix(A, A_rows);
    free_matrix(B, B_rows);
    free_matrix(C, A_rows);

    printf("matmul_sparse completed.\n");

    return 0;
}
