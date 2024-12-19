#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../kernel/matrix_ops.h"

#define REPEAT 10

void test_matmul(int A_size, int B_size) {
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
    if (argc != 3) {
        printf("Usage: %s <A_size> <B_size>\n", argv[0]);
        return 1;
    }

    int A_size = atoi(argv[1]);
    int B_size = atoi(argv[2]);

    srand(time(NULL));
    printf("Testing matrix multiplication with square matrices A: %dx%d and B: %dx%d\n", A_size, A_size, B_size, B_size);
    test_matmul(A_size, B_size);
    return 0;
}
