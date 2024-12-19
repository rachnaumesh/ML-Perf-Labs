#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../kernel/matrix_ops.h"

#define REPEAT 20

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

void test_matmul(int A_size, int B_size, float sparsity) {
    if (A_size != B_size) {
        printf("Matrix multiplication is not possible with dimensions A: %dx%d and B: %dx%d\n", A_size, A_size, B_size, B_size);
        return;
    }

    //Generate sparse matrices A and B
    float **A = generate_sparse_matrix(A_size, A_size, sparsity);
    float **B = generate_sparse_matrix(B_size, B_size, sparsity);

    // Profiling loop
    for (int i = 0; i < REPEAT; i++) {
        float **C = matmul(A, B, A_size, A_size, B_size, B_size);
        if (C != NULL) {
            for (int r = 0; r < A_size; r++) {
                free(C[r]);
            }
            free(C);
        }
    }

    // Cleanup
    for (int i = 0; i < A_size; i++) {
        free(A[i]);
    }
    free(A);

    for (int i = 0; i < B_size; i++) {
        free(B[i]);
    }
    free(B);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <A_size> <B_size> <sparsity\n", argv[0]);
        return 1;
    }

    int A_size = atoi(argv[1]);
    int B_size = atoi(argv[2]);
    float sparsity = atof(argv[3]);

    srand(time(NULL));
    printf("Testing matrix multiplication with square matrices A: %dx%d and B: %dx%d and sparsity: %.2f\n", A_size, A_size, B_size, B_size, sparsity);
    test_matmul(A_size, B_size, sparsity);
    return 0;
}
