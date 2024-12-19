#include "matrix_ops.h"
#include "math.h"

// #ifndef BLOCK_SIZE_I
// #define BLOCK_SIZE_I 169
// #endif

// #ifndef BLOCK_SIZE_J
// #define BLOCK_SIZE_J 32
// #endif

// #ifndef BLOCK_SIZE_K
// #define BLOCK_SIZE_K 9
// #endif

#define min(a, b) ((a) < (b) ? (a) : (b))

float **transpose_matrix(float **K, int rows, int cols) {
    float **K_T = (float **)malloc(cols * sizeof(float *));
    for (int i = 0; i < cols; i++) {
        K_T[i] = (float *)malloc(rows * sizeof(float));
        for (int j = 0; j < rows; j++) {
            K_T[i][j] = K[j][i];
        }
    }
    return K_T;
}


float **matmul(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols)
{
    // TODO: Implement matrix multiplication
    if (A_cols != B_rows) {
        return NULL;
    }
    
    float **C = (float **)malloc(A_rows * sizeof(float *));
    for (int i = 0; i < A_rows; i++) {
        C[i] = (float *)malloc(B_cols * sizeof(float));
        for (int j = 0; j < B_cols; j++) {
            C[i][j] = 0;
        }
    }

    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            for (int k = 0; k < A_cols; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

// Matmul with blocking optimization
float **matmul_blocking(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols, int BLOCK_SIZE_I, int BLOCK_SIZE_J, int BLOCK_SIZE_K)
{
    // TODO: Implement matrix multiplication with blocking (loop tiling)
    if (A_cols != B_rows) {
        return NULL;
    }

    float **C = (float **)malloc(A_rows * sizeof(float *));
    for (int i = 0; i < A_rows; i++) {
        C[i] = (float *)malloc(B_cols * sizeof(float));
        for (int j = 0; j < B_cols; j++) {
            C[i][j] = 0;
        }
    }

    for (int i = 0; i < A_rows; i += BLOCK_SIZE_I) {
        for (int j = 0; j < B_cols; j += BLOCK_SIZE_J) {
            for (int k = 0; k < A_cols; k += BLOCK_SIZE_K) {
                
                for (int ii = i; ii < min(i + BLOCK_SIZE_I, A_rows); ii++) {
                    for (int jj = j; jj < min(j + BLOCK_SIZE_J, B_cols); jj++) {
                        for (int kk = k; kk < min(k + BLOCK_SIZE_K, A_cols); kk++) {
                            C[ii][jj] += A[ii][kk] * B[kk][jj];
                        }
                    }
                }
            }
        }
    }

    return C;
}


// Optimized matrix multiplication with blocking (caching to save repeated access)
float **matmul_blocking_optimized(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols, int BLOCK_SIZE_I, int BLOCK_SIZE_J, int BLOCK_SIZE_K) {
    if (A_cols != B_rows) {
        return NULL;  // Incompatible matrices for multiplication
    }

    // Allocate the result matrix C
    float **C = (float **)malloc(A_rows * sizeof(float *));
    for (int i = 0; i < A_rows; i++) {
        C[i] = (float *)malloc(B_cols * sizeof(float));
        for (int j = 0; j < B_cols; j++) {
            C[i][j] = 0;  // Initialize all values to zero
        }
    }

    // Blocked matrix multiplication
    for (int i = 0; i < A_rows; i += BLOCK_SIZE_I) {
        for (int j = 0; j < B_cols; j += BLOCK_SIZE_J) {
            for (int k = 0; k < A_cols; k += BLOCK_SIZE_K) {

                // Perform multiplication within the block
                for (int ii = i; ii < min(i + BLOCK_SIZE_I, A_rows); ii++) {
                    for (int kk = k; kk < min(k + BLOCK_SIZE_K, A_cols); kk++) {
                        float temp = A[ii][kk];
                        for (int jj = j; jj < min(j + BLOCK_SIZE_J, B_cols); jj++) {
                            C[ii][jj] += temp * B[kk][jj];
                        }
                    }
}
            }
        }
    }

    return C;
}


// Blocking with transposed B matrix for cache efficiency
float **matmul_blocking_transpose(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols, int BLOCK_SIZE_I, int BLOCK_SIZE_J, int BLOCK_SIZE_K) {
    if (A_cols != B_rows) return NULL;

    float **B_T = transpose_matrix(B, B_rows, B_cols);

    float **C = (float **)malloc(A_rows * sizeof(float *));
    for (int i = 0; i < A_rows; i++) {
        C[i] = (float *)malloc(B_cols * sizeof(float));
        for (int j = 0; j < B_cols; j++) C[i][j] = 0;
    }

    for (int i = 0; i < A_rows; i += BLOCK_SIZE_I) {
        for (int j = 0; j < B_cols; j += BLOCK_SIZE_J) {
            for (int k = 0; k < A_cols; k += BLOCK_SIZE_K) {
                
                for (int ii = i; ii < min(i + BLOCK_SIZE_I, A_rows); ii++) {
                    for (int jj = j; jj < min(j + BLOCK_SIZE_J, B_cols); jj++) {
                        for (int kk = k; kk < min(k + BLOCK_SIZE_K, A_cols); kk++) {
                            C[ii][jj] += A[ii][kk] * B_T[jj][kk];
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < B_cols; i++) free(B_T[i]);
    free(B_T);
    return C;
}


// Blocking with ikj loop order and caching to save repeated access
float **matmul_blocking_ikj(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols, int BLOCK_SIZE_I, int BLOCK_SIZE_J, int BLOCK_SIZE_K) {
    if (A_cols != B_rows) return NULL;

    float **C = (float **)malloc(A_rows * sizeof(float *));
    for (int i = 0; i < A_rows; i++) {
        C[i] = (float *)malloc(B_cols * sizeof(float));
        for (int j = 0; j < B_cols; j++) C[i][j] = 0;
    }

    for (int i = 0; i < A_rows; i += BLOCK_SIZE_I) {
        for (int k = 0; k < A_cols; k += BLOCK_SIZE_K) {
            for (int j = 0; j < B_cols; j += BLOCK_SIZE_J) {
                
                for (int ii = i; ii < min(i + BLOCK_SIZE_I, A_rows); ii++) {
                    for (int kk = k; kk < min(k + BLOCK_SIZE_K, A_cols); kk++) {
                        float temp = A[ii][kk];
                        for (int jj = j; jj < min(j + BLOCK_SIZE_J, B_cols); jj++) {
                            C[ii][jj] += temp * B[kk][jj];
                        }
                    }
                }
            }
        }
    }
    return C;
}



// Blocking with jik loop order
float **matmul_blocking_jik(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols, int BLOCK_SIZE_I, int BLOCK_SIZE_J, int BLOCK_SIZE_K) {
    if (A_cols != B_rows) return NULL;

    float **C = (float **)malloc(A_rows * sizeof(float *));
    for (int i = 0; i < A_rows; i++) {
        C[i] = (float *)malloc(B_cols * sizeof(float));
        for (int j = 0; j < B_cols; j++) C[i][j] = 0;
    }

    for (int j = 0; j < B_cols; j += BLOCK_SIZE_J) {
        for (int i = 0; i < A_rows; i += BLOCK_SIZE_I) {
            for (int k = 0; k < A_cols; k += BLOCK_SIZE_K) {
                
                for (int jj = j; jj < min(j + BLOCK_SIZE_J, B_cols); jj++) {
                    for (int ii = i; ii < min(i + BLOCK_SIZE_I, A_rows); ii++) {
                        float temp = 0;
                        for (int kk = k; kk < min(k + BLOCK_SIZE_K, A_cols); kk++) {
                            temp += A[ii][kk] * B[kk][jj];
                        }
                        C[ii][jj] += temp;
                    }
                }
            }
        }
    }
    return C;
}


// Blocking with kij loop order and caching to save repeated access
float **matmul_blocking_kij(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols, int BLOCK_SIZE_I, int BLOCK_SIZE_J, int BLOCK_SIZE_K) {
    if (A_cols != B_rows) return NULL;

    float **C = (float **)malloc(A_rows * sizeof(float *));
    for (int i = 0; i < A_rows; i++) {
        C[i] = (float *)malloc(B_cols * sizeof(float));
        for (int j = 0; j < B_cols; j++) C[i][j] = 0;
    }

    for (int k = 0; k < A_cols; k += BLOCK_SIZE_K) {
        for (int i = 0; i < A_rows; i += BLOCK_SIZE_I) {
            for (int j = 0; j < B_cols; j += BLOCK_SIZE_J) {
                
                for (int kk = k; kk < min(k + BLOCK_SIZE_K, A_cols); kk++) {
                    for (int ii = i; ii < min(i + BLOCK_SIZE_I, A_rows); ii++) {
                        float temp = A[ii][kk];
                        for (int jj = j; jj < min(j + BLOCK_SIZE_J, B_cols); jj++) {
                            C[ii][jj] += temp * B[kk][jj];
                        }
                    }
                }
            }
        }
    }
    return C;
}


// Dynamic block size based on input dimensions
float **matmul_blocking_dynamic(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols, int BLOCK_SIZE_I, int BLOCK_SIZE_J, int BLOCK_SIZE_K) {
    if (A_cols != B_rows) return NULL;

    int block_i = A_rows > 1000 ? 64 : 32;
    int block_j = B_cols > 1000 ? 64 : 32;
    int block_k = A_cols > 1000 ? 32 : 16;

    float **C = (float **)malloc(A_rows * sizeof(float *));
    for (int i = 0; i < A_rows; i++) {
        C[i] = (float *)malloc(B_cols * sizeof(float));
        for (int j = 0; j < B_cols; j++) C[i][j] = 0;
    }

    for (int i = 0; i < A_rows; i += block_i) {
        for (int j = 0; j < B_cols; j += block_j) {
            for (int k = 0; k < A_cols; k += block_k) {
                
                for (int ii = i; ii < min(i + block_i, A_rows); ii++) {
                    for (int jj = j; jj < min(j + block_j, B_cols); jj++) {
                        for (int kk = k; kk < min(k + block_k, A_cols); kk++) {
                            C[ii][jj] += A[ii][kk] * B[kk][jj];
                        }
                    }
                }
            }
        }
    }
    return C;
}


// Blocking with unrolling for the innermost loop
float **matmul_blocking_unrolled(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols, int BLOCK_SIZE_I, int BLOCK_SIZE_J, int BLOCK_SIZE_K) {
    if (A_cols != B_rows) return NULL;

    float **C = (float **)malloc(A_rows * sizeof(float *));
    for (int i = 0; i < A_rows; i++) {
        C[i] = (float *)malloc(B_cols * sizeof(float));
        for (int j = 0; j < B_cols; j++) {
            C[i][j] = 0;
        }
    }

    int unroll_factor = 4;

    for (int i = 0; i < A_rows; i += BLOCK_SIZE_I) {
        for (int j = 0; j < B_cols; j += BLOCK_SIZE_J) {
            for (int k = 0; k < A_cols; k += BLOCK_SIZE_K) {

                for (int ii = i; ii < min(i + BLOCK_SIZE_I, A_rows); ii++) {
                    for (int jj = j; jj < min(j + BLOCK_SIZE_J, B_cols); jj++) {
                        float sum = 0;

                        // Unrolled loop for the innermost k iterations
                        int kk;
                        for (kk = k; kk < min(k + BLOCK_SIZE_K - unroll_factor, A_cols - unroll_factor); kk += unroll_factor) {
                            sum += A[ii][kk] * B[kk][jj]
                                + A[ii][kk + 1] * B[kk + 1][jj]
                                + A[ii][kk + 2] * B[kk + 2][jj]
                                + A[ii][kk + 3] * B[kk + 3][jj];
                        }

                        // Handle the remaining iterations
                        for (; kk < min(k + BLOCK_SIZE_K, A_cols); kk++) {
                            sum += A[ii][kk] * B[kk][jj];
                        }

                        C[ii][jj] += sum;
                    }
                }
            }
        }
    }

    return C;
}
