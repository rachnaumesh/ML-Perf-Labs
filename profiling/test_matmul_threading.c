#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../kernel/matrix_ops.h"

#define REPEAT 10

// Function to generate a random matrix
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

void test_matmul(int A_size, int B_size) {
    if (A_size != B_size) {
        printf("Matrix multiplication is not possible with dimensions A: %dx%d and B: %dx%d\n", A_size, A_size, B_size, B_size);
        return;
    }

    // Generate random matrices A and B
    float **A = generate_random_matrix(A_size, A_size);
    float **B = generate_random_matrix(B_size, B_size);

    // Profiling loop
    for (int i = 0; i < REPEAT; i++) {
        float **C = matmul_thread(A, B, A_size, A_size, B_size, B_size);
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
        printf("Usage: %s <A_size> <B_size> <num_threads>\n", argv[0]);
        return 1;
    }

    int A_size = atoi(argv[1]);
    int B_size = atoi(argv[2]);
    int num_threads = atoi(argv[3]);

    srand(time(NULL));
    printf("Testing matrix multiplication with square matrices A: %dx%d and B: %dx%d using %d threads\n", A_size, A_size, B_size, B_size, num_threads);
    test_matmul(A_size, B_size);
    return 0;
}