#include "matrix_ops.h"


void CSRMatrix(float **A, int rows, int cols, float **values, int **rowPtr, int **colIdx, int *nnz)
{
    *nnz = 0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (A[i][j] != 0) {
                (*nnz)++;
            }
        }
    }

    *values = (float *)malloc(*nnz * sizeof(float));
    *colIdx = (int *)malloc(*nnz * sizeof(int));
    *rowPtr = (int *)malloc((rows + 1) * sizeof(int));

    // Set the first element of rowPtr to 0
    (*rowPtr)[0] = 0;

    int index = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (A[i][j] != 0) {
                (*values)[index] = A[i][j];
                (*colIdx)[index] = j;
                index++;
            }
        }
        (*rowPtr)[i + 1] = index;
    }

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

float **matmul_sparse(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols)
{
    // TODO: Implement sparse matrix multiplication
    if (A_cols != B_rows) {
        return NULL;
    }

    float *values;
    int *rowPtr;
    int *colIdx;
    int nnz;

    CSRMatrix(A, A_rows, A_cols, &values, &rowPtr, &colIdx, &nnz);

    float **C = (float **)malloc(A_rows * sizeof(float *));
    for (int i = 0; i < A_rows; i++) {
        C[i] = (float *)malloc(B_cols * sizeof(float));
        for (int j = 0; j < B_cols; j++) {
            C[i][j] = 0;
        }
    }

    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            for (int k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
                C[i][j] += values[k] * B[colIdx[k]][j];
            }
        }
    } 
    return C;
}

float **matmul_sparse_part2(float *values, int *rowPtr, int *colIdx, int A_rows, int A_cols, float **B, int B_rows, int B_cols)
{
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
            for (int k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
                C[i][j] += values[k] * B[colIdx[k]][j];
            }
        }
    }
    return C;
}