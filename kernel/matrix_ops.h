#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include <stdio.h>
#include <stdlib.h>

float **transpose_matrix(float **K, int rows, int cols);
float **matmul(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols);
float **matmul_blocking(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols, int BLOCK_SIZE_I, int BLOCK_SIZE_J, int BLOCK_SIZE_K);
float **matmul_blocking_optimized(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols, int BLOCK_SIZE_I, int BLOCK_SIZE_J, int BLOCK_SIZE_K);
float **matmul_blocking_transpose(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols, int BLOCK_SIZE_I, int BLOCK_SIZE_J, int BLOCK_SIZE_K);
float **matmul_blocking_ikj(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols, int BLOCK_SIZE_I, int BLOCK_SIZE_J, int BLOCK_SIZE_K);
float **matmul_blocking_jik(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols, int BLOCK_SIZE_I, int BLOCK_SIZE_J, int BLOCK_SIZE_K);
float **matmul_blocking_kij(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols, int BLOCK_SIZE_I, int BLOCK_SIZE_J, int BLOCK_SIZE_K);
float **matmul_blocking_dynamic(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols, int BLOCK_SIZE_I, int BLOCK_SIZE_J, int BLOCK_SIZE_K);
float **matmul_blocking_unrolled(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols, int BLOCK_SIZE_I, int BLOCK_SIZE_J, int BLOCK_SIZE_K);

#endif /* MATRIX_OPS_H */