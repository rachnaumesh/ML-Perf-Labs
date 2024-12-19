#include "attention.h"
#include "matrix_ops.h"
#include "functional.h"
#include <stdio.h>
#include <math.h>

// Scaled dot-product attention
float **scaled_dot_product_attention(float **Q, float **K, float **V, int seqLength, int depth)
{
    // TODO: Implement the attention algorithm
    float **K_T = transpose_matrix(K, seqLength, depth); // Transpose K
    float **QK_T = matmul(Q, K_T, seqLength, depth, depth, seqLength); // QK

    // Scale QK_T
    float scaleFactor = sqrt(depth);
    float **QK_T_scaled = (float **)malloc(seqLength * sizeof(float *));
    for (int i = 0; i < seqLength; i++) {
        QK_T_scaled[i] = (float *)malloc(seqLength * sizeof(float));
        for (int j = 0; j < seqLength; j++) {
            QK_T_scaled[i][j] = QK_T[i][j] / scaleFactor;
        }
    }

    // Softmax
    float **attention_weights = (float **)malloc(seqLength * sizeof(float *));
    for (int i = 0; i < seqLength; i++) {
        attention_weights[i] = softmax_without_log(QK_T_scaled[i], seqLength);  // Apply softmax row-wise
    }

    // Attention
    float **attention = matmul(attention_weights, V, seqLength, seqLength, seqLength, depth);
    
    return attention;
}