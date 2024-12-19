#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../kernel/matrix_ops.h"

#define REPEAT 100

void test_matmul_blocking(int A_size, int B_size, int BLOCK_SIZE_I, int BLOCK_SIZE_J, int BLOCK_SIZE_K) {
    if (A_size != B_size) {
        printf("Matrix multiplication is not possible with dimensions A: %dx%d and B: %dx%d\n", A_size, A_size, B_size, B_size);
        return;
    }

    // Allocate and initialize matrix A
    float **A = (float **)malloc(A_size * sizeof(float *));
    for (int i = 0; i < A_size; i++) {
        A[i] = (float *)malloc(A_size * sizeof(float));
        for (int j = 0; j < A_size; j++) {
            A[i][j] = (float)(rand() % 10);  // Random values from 0 to 9
        }
    }

    // Allocate and initialize matrix B
    float **B = (float **)malloc(B_size * sizeof(float *));
    for (int i = 0; i < B_size; i++) {
        B[i] = (float *)malloc(B_size * sizeof(float));
        for (int j = 0; j < B_size; j++) {
            B[i][j] = (float)(rand() % 10);  // Random values from 0 to 9
        }
    }

    // Profiling loop
    for (int i = 0; i < REPEAT; i++) {
        float **C = matmul_blocking_unrolled(A, B, A_size, A_size, B_size, B_size, BLOCK_SIZE_I, BLOCK_SIZE_J, BLOCK_SIZE_K);
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
    if (argc != 6) {
        printf("Usage: %s <A_size> <B_size> <BLOCK_SIZE_I> <BLOCK_SIZE_J> <BLOCK_SIZE_K>\n", argv[0]);
        return 1;
    }

    int A_size = atoi(argv[1]);
    int B_size = atoi(argv[2]);
    int BLOCK_SIZE_I = atoi(argv[3]);
    int BLOCK_SIZE_J = atoi(argv[4]);
    int BLOCK_SIZE_K = atoi(argv[5]);

    srand(time(NULL));
    printf("Testing matrix multiplication with blocking for matrices A: %dx%d and B: %dx%d with block sizes I: %d, J: %d, K: %d\n", A_size, A_size, B_size, B_size, BLOCK_SIZE_I, BLOCK_SIZE_J, BLOCK_SIZE_K);
    test_matmul_blocking(A_size, B_size, BLOCK_SIZE_I, BLOCK_SIZE_J, BLOCK_SIZE_K);
    return 0;
}