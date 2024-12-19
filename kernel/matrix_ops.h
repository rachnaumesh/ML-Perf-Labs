#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include <stdio.h>
#include <stdlib.h>

float **matmul(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols);
float **matmul_sparse(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols);
void CSRMatrix(float **A, int A_rows, int A_cols, float **values, int **rowPtr, int **colIdx, int *nnz);
float **matmul_sparse_part2(float *values, int *rowPtr, int *colIdx, int A_rows, int A_cols, float **B, int B_rows, int B_cols);

#endif /* MATRIX_OPS_H */