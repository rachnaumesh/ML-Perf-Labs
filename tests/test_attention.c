#include "unity/unity.h"
#include "../kernel/kernel.h"
#include "../utils/data_utils.h"
#include "test_attention.h"

// Utility functions to allocate and free memory for matrices
float **allocate_matrix(int rows, int cols)
{
    float **matrix = (float **)malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++)
    {
        matrix[i] = (float *)malloc(cols * sizeof(float));
    }
    return matrix;
}

void free_matrix(float **matrix, int rows)
{
    for (int i = 0; i < rows; i++)
    {
        free(matrix[i]);
    }
    free(matrix);
}

// Test functions for scaled dot-product attention and multi-head attention
void test_scaled_dot_product_attention(void)
{
    // Setup input data
    float Q_data[2][3] = {{1, 0, 1}, {0, 2, 1}};
    float K_data[2][3] = {{1, 0, 2}, {0, 1, 2}};
    float V_data[2][3] = {{0, 1, 0}, {1, 0, 1}};
    int seqLength = 2;
    int depth = 3;
    int d_k = 3;

    // Convert data to pointers
    float **Q = allocate_matrix(seqLength, depth);
    float **K = allocate_matrix(seqLength, depth);
    float **V = allocate_matrix(seqLength, depth);

    for (int i = 0; i < seqLength; i++)
    {
        for (int j = 0; j < depth; j++)
        {
            Q[i][j] = Q_data[i][j];
            K[i][j] = K_data[i][j];
            V[i][j] = V_data[i][j];
        }
    }

    double expected_output[2][3] = {{0.3595, 0.6405, 0.3595}, {0.7604, 0.2396, 0.7604}}; // Expected attention output values

    // Run function under test
    float** output = scaled_dot_product_attention(Q, K, V, seqLength, depth);

    // Check expectations
    for (int i = 0; i < seqLength; i++)
    {
        for (int j = 0; j < depth; j++) {
            UNITY_TEST_ASSERT_FLOAT_WITHIN(0.0001, expected_output[i][j], output[i][j], __LINE__, "Scaled dot-product attention output mismatch");
        }
    }

    // Cleanup
    free_matrix(Q, seqLength);
    free_matrix(K, seqLength);
    free_matrix(V, seqLength);
    free(output);
}