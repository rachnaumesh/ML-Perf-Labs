#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>  // Added for AVX intrinsics

#define EPSILON 1e-5
#define EMBEDDING_SIZE 768
#define NUM_BLOCKS 12
#define NUM_HEADS 12
#define HEAD_DIM (EMBEDDING_SIZE / NUM_HEADS)
#define VOCAB_SIZE 50257
#define MAX_POSITION_EMBEDDINGS 1024

// Original struct definitions remain the same
typedef struct {
    int batch_size;
    int sequence_length;
    int features;
    float *data;
} Tensor3D;

typedef struct {
    int rows;
    int cols;
    float *data;
} Tensor2D;

typedef struct {
    float **weights;
    float *biases;
    int fcInputSize;
    int fcOutputSize;
} LinearLayer;

typedef struct {
    LinearLayer q_mlp;
    LinearLayer k_mlp;
    LinearLayer v_mlp;
    LinearLayer first_block_MLP;
    LinearLayer second_block_MLP;
} BlockWeights;

typedef struct {
    float **wpe;
    float **wte;
    BlockWeights *blocks;
    LinearLayer logits_mlp;
} GPT2Weights;

// SIMD optimized linear function
float *linear(float *fcInput, float **weights, float *biases, int fcInputSize, int fcOutputSize) {
    float *output = (float *)malloc(fcOutputSize * sizeof(float));
    if (!output) {
        fprintf(stderr, "Memory allocation failed in linear function\n");
        exit(1);
    }

    // Process 8 elements at a time using AVX
    for (int i = 0; i < fcOutputSize; i++) {
        __m256 sum_vec = _mm256_setzero_ps();
        
        // Main loop - process 8 elements at once
        for (int j = 0; j < fcInputSize - 7; j += 8) {
            __m256 input_vec = _mm256_loadu_ps(&fcInput[j]);
            __m256 weight_vec = _mm256_loadu_ps(&weights[i][j]);
            sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(input_vec, weight_vec));
        }

        // Horizontal sum
        float sum_array[8];
        _mm256_storeu_ps(sum_array, sum_vec);
        float sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] +
                   sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];

        // Handle remaining elements
        for (int j = (fcInputSize/8)*8; j < fcInputSize; j++) {
            sum += fcInput[j] * weights[i][j];
        }

        output[i] = sum + biases[i];
    }

    return output;
}

// SIMD optimized scaled dot-product attention
float **scaled_dot_product_attention(float **Q, float **K, float **V, int seqLength, int depth) {
    float *scores = (float *)malloc(seqLength * sizeof(float *));
    float *attention_weights = (float *)malloc(seqLength * sizeof(float *));
    float *output = (float *)malloc(seqLength * sizeof(float *));

    float scale = 1.0f / sqrt(depth);
    __m256 scale_vec = _mm256_set1_ps(scale);

    for (int i = 0; i < seqLength; i++) {
        scores[i] = (float *)malloc(seqLength * sizeof(float));
        attention_weights[i] = (float *)malloc(seqLength * sizeof(float));
        output[i] = (float *)malloc(depth * sizeof(float));

        // Compute scores using SIMD
        for (int j = 0; j < seqLength; j++) {
            __m256 sum_vec = _mm256_setzero_ps();
            
            for (int k = 0; k < depth - 7; k += 8) {
                __m256 q_vec = _mm256_loadu_ps(&Q[i][k]);
                __m256 k_vec = _mm256_loadu_ps(&K[j][k]);
                sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(q_vec, k_vec));
            }

            // Horizontal sum
            float sum_array[8];
            _mm256_storeu_ps(sum_array, sum_vec);
            float sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] +
                       sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];

            // Handle remaining elements
            for (int k = (depth/8)*8; k < depth; k++) {
                sum += Q[i][k] * K[j][k];
            }

            scores[i][j] = sum * scale;
        }
    }

    // Apply softmax (keeping original implementation for numerical stability)
    for (int i = 0; i < seqLength; i++) {
        float max_val = scores[i][0];
        for (int j = 1; j < seqLength; j++) {
            if (scores[i][j] > max_val) max_val = scores[i][j];
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < seqLength; j++) {
            attention_weights[i][j] = exp(scores[i][j] - max_val);
            sum_exp += attention_weights[i][j];
        }

        for (int j = 0; j < seqLength; j++) {
            attention_weights[i][j] /= sum_exp;
        }
    }

    // Compute attention output using SIMD
    for (int i = 0; i < seqLength; i++) {
        for (int k = 0; k < depth - 7; k += 8) {
            __m256 sum_vec = _mm256_setzero_ps();
            
            for (int j = 0; j < seqLength; j++) {
                __m256 v_vec = _mm256_loadu_ps(&V[j][k]);
                __m256 weight_vec = _mm256_set1_ps(attention_weights[i][j]);
                sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(v_vec, weight_vec));
            }

            _mm256_storeu_ps(&output[i][k], sum_vec);
        }

        // Handle remaining elements
        for (int k = (depth/8)*8; k < depth; k++) {
            float sum = 0.0f;
            for (int j = 0; j < seqLength; j++) {
                sum += attention_weights[i][j] * V[j][k];
            }
            output[i][k] = sum;
        }
    }

    // Free intermediate allocations
    for (int i = 0; i < seqLength; i++) {
        free(scores[i]);
        free(attention_weights[i]);
    }
    free(scores);
    free(attention_weights);

    return output;
}

// SIMD optimized matrix addition
float **matrix_add(float **x, float **y, int numRow, int numCol) {
    float *result = (float *)malloc(numRow * sizeof(float *));
    for (int i = 0; i < numRow; i++) {
        result[i] = (float *)malloc(numCol * sizeof(float));
        
        // Process 8 elements at a time
        for (int j = 0; j < numCol - 7; j += 8) {
            __m256 x_vec = _mm256_loadu_ps(&x[i][j]);
            __m256 y_vec = _mm256_loadu_ps(&y[i][j]);
            __m256 sum_vec = _mm256_add_ps(x_vec, y_vec);
            _mm256_storeu_ps(&result[i][j], sum_vec);
        }

        // Handle remaining elements
        for (int j = (numCol/8)*8; j < numCol; j++) {
            result[i][j] = x[i][j] + y[i][j];
        }
    }
    return result;
}

// Rest of the original functions remain unchanged
float **norm(float **x, int seqLength, int features) {
    float *normalized = (float *)malloc(seqLength * sizeof(float *));
    for (int i = 0; i < seqLength; i++) {
        normalized[i] = (float *)malloc(features * sizeof(float));
        float mean = 0.0;
        for (int j = 0; j < features; j++) {
            mean += x[i][j];
        }
        mean /= features;
        float variance = 0.0;
        for (int j = 0; j < features; j++) {
            variance += (x[i][j] - mean) * (x[i][j] - mean);
        }
        variance /= features;
        for (int j = 0; j < features; j++) {
            normalized[i][j] = (x[i][j] - mean) / sqrt(variance + EPSILON);
        }
    }
    return normalized;
}

// Implement the GELU activation function
float *gelu(float *x, int size) {
    float *output = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        output[i] = 0.5 * x[i] * (1 + tanh(sqrt(2 / M_PI) * (x[i] + 0.044715 * x[i] * x[i] * x[i])));
    }
    return output;
}

// Function to compute positions
int *positions_for(int *tokens, int seqLength, int past_length) {
    int *positions = (int *)malloc(seqLength * sizeof(int));
    for (int i = 0; i < seqLength; i++) {
        positions[i] = past_length + i;
    }
    return positions;
}

// Implement the transformer block with multi-head attention
float **block(float **x, int seqLength, int embeddingSize, BlockWeights weights) {
    // Extract weights
    LinearLayer q_mlp = weights.q_mlp;
    LinearLayer k_mlp = weights.k_mlp;
    LinearLayer v_mlp = weights.v_mlp;
    LinearLayer first_block_MLP = weights.first_block_MLP;
    LinearLayer second_block_MLP = weights.second_block_MLP;

    // Apply layer normalization to x
    float **normalized_x = norm(x, seqLength, embeddingSize);

    // Allocate memory for Q, K, V
    float *Q = (float *)malloc(seqLength * sizeof(float *));
    float *K = (float *)malloc(seqLength * sizeof(float *));
    float *V = (float *)malloc(seqLength * sizeof(float *));
    for (int i = 0; i < seqLength; i++) {
        Q[i] = linear(normalized_x[i], q_mlp.weights, q_mlp.biases, q_mlp.fcInputSize, q_mlp.fcOutputSize);
        K[i] = linear(normalized_x[i], k_mlp.weights, k_mlp.biases, k_mlp.fcInputSize, k_mlp.fcOutputSize);
        V[i] = linear(normalized_x[i], v_mlp.weights, v_mlp.biases, v_mlp.fcInputSize, v_mlp.fcOutputSize);
    }

    // Reshape Q, K, V for multi-head attention
    // Q_heads[NUM_HEADS][seqLength][HEAD_DIM]
    float **Q_heads = (float *)malloc(NUM_HEADS * sizeof(float *));
    float **K_heads = (float *)malloc(NUM_HEADS * sizeof(float *));
    float **V_heads = (float *)malloc(NUM_HEADS * sizeof(float *));
    for (int h = 0; h < NUM_HEADS; h++) {
        Q_heads[h] = (float **)malloc(seqLength * sizeof(float *));
        K_heads[h] = (float **)malloc(seqLength * sizeof(float *));
        V_heads[h] = (float **)malloc(seqLength * sizeof(float *));
        for (int i = 0; i < seqLength; i++) {
            Q_heads[h][i] = (float *)malloc(HEAD_DIM * sizeof(float));
            K_heads[h][i] = (float *)malloc(HEAD_DIM * sizeof(float));
            V_heads[h][i] = (float *)malloc(HEAD_DIM * sizeof(float));
            // Copy the corresponding slice from Q, K, V
            memcpy(Q_heads[h][i], &Q[i][h * HEAD_DIM], HEAD_DIM * sizeof(float));
            memcpy(K_heads[h][i], &K[i][h * HEAD_DIM], HEAD_DIM * sizeof(float));
            memcpy(V_heads[h][i], &V[i][h * HEAD_DIM], HEAD_DIM * sizeof(float));
        }
    }

    // Apply attention on each head
    float **head_outputs = (float *)malloc(NUM_HEADS * sizeof(float *));
    for (int h = 0; h < NUM_HEADS; h++) {
        head_outputs[h] = scaled_dot_product_attention(Q_heads[h], K_heads[h], V_heads[h], seqLength, HEAD_DIM);
    }

    // Concatenate the outputs from all heads
    float *a = (float *)malloc(seqLength * sizeof(float *));
    for (int i = 0; i < seqLength; i++) {
        a[i] = (float *)malloc(embeddingSize * sizeof(float));
        for (int h = 0; h < NUM_HEADS; h++) {
            memcpy(&a[i][h * HEAD_DIM], head_outputs[h][i], HEAD_DIM * sizeof(float));
        }
    }

    // Add residual connection
    float **x_added = matrix_add(x, a, seqLength, embeddingSize);

    // Apply layer normalization
    float **normalized_x_added = norm(x_added, seqLength, embeddingSize);

    // Allocate memory for m
    float *m = (float *)malloc(seqLength * sizeof(float *));
    for (int i = 0; i < seqLength; i++) {
        float *first_mlp = linear(normalized_x_added[i], first_block_MLP.weights, first_block_MLP.biases, first_block_MLP.fcInputSize, first_block_MLP.fcOutputSize);
        float *gelu_out = gelu(first_mlp, first_block_MLP.fcOutputSize);
        m[i] = linear(gelu_out, second_block_MLP.weights, second_block_MLP.biases, second_block_MLP.fcInputSize, second_block_MLP.fcOutputSize);
        free(first_mlp);
        free(gelu_out);
    }

    // Add residual connection
    float **output = matrix_add(x_added, m, seqLength, embeddingSize);

    // Free allocated memory
    for (int i = 0; i < seqLength; i++) {
        free(normalized_x[i]);
        free(Q[i]);
        free(K[i]);
        free(V[i]);
        free(normalized_x_added[i]);
        free(m[i]);
        free(x_added[i]);
    }
    free(normalized_x);
    free(Q);
    free(K);
    free(V);
    free(normalized_x_added);
    free(m);
    free(x_added);

    // Free memory for heads
    for (int h = 0; h < NUM_HEADS; h++) {
        for (int i = 0; i < seqLength; i++) {
            free(Q_heads[h][i]);
            free(K_heads[h][i]);
            free(V_heads[h][i]);
            free(head_outputs[h][i]);
        }
        free(Q_heads[h]);
        free(K_heads[h]);
        free(V_heads[h]);
        free(head_outputs[h]);
    }
    free(Q_heads);
    free(K_heads);
    free(V_heads);
    free(head_outputs);

    return output;
}

// Implement the model function with positional embeddings
float *model(int *tokens, int seqLength, GPT2Weights weights) {
    // Compute positions
    int past_length = 0; // Assuming no past tokens for simplicity
    int *positions = positions_for(tokens, seqLength, past_length);

    // Initialize h with embeddings
    float *h = (float *)malloc(seqLength * sizeof(float *));
    for (int i = 0; i < seqLength; i++) {
        h[i] = (float *)malloc(EMBEDDING_SIZE * sizeof(float));
        // Get word embeddings and add positional embeddings
        for (int j = 0; j < EMBEDDING_SIZE; j++) {
            h[i][j] = weights.wte[tokens[i]][j] + weights.wpe[positions[i]][j];
        }
    }

    // Free positions
    free(positions);

    // Pass through transformer blocks
    for (int i = 0; i < NUM_BLOCKS; i++) {
        float **new_h = block(h, seqLength, EMBEDDING_SIZE, weights.blocks[i]);
        // Free previous h
        for (int j = 0; j < seqLength; j++) {
            free(h[j]);
        }
        free(h);
        h = new_h;
    }

    // Get logits for the last token
    LinearLayer logits_mlp = weights.logits_mlp;
    float *logits = linear(h[seqLength - 1], logits_mlp.weights, logits_mlp.biases, logits_mlp.fcInputSize, logits_mlp.fcOutputSize);

    // Free h
    for (int i = 0; i < seqLength; i++) {
        free(h[i]);
    }
    free(h);

    return logits;
}

void initialize_linear_layer(LinearLayer *layer, int inputSize, int outputSize) {
    layer->fcInputSize = inputSize;
    layer->fcOutputSize = outputSize;
    layer->weights = (float **)malloc(outputSize * sizeof(float *));
    layer->biases = (float *)malloc(outputSize * sizeof(float));
    for (int i = 0; i < outputSize; i++) {
        layer->weights[i] = (float *)malloc(inputSize * sizeof(float));
        layer->biases[i] = 0.0f; // Initialize biases to zero
        for (int j = 0; j < inputSize; j++) {
            layer->weights[i][j] = ((float)rand() / RAND_MAX) * 0.02f - 0.01f; // Random weights between -0.01 and 0.01
        }
    }
}

GPT2Weights initialize_weights() {
    // Initialize GPT2Weights
    GPT2Weights weights;

    // Initialize token embeddings (wte)
    weights.wte = (float **)malloc(VOCAB_SIZE * sizeof(float *));
    for (int i = 0; i < VOCAB_SIZE; i++) {
        weights.wte[i] = (float *)malloc(EMBEDDING_SIZE * sizeof(float));
        for (int j = 0; j < EMBEDDING_SIZE; j++) {
            weights.wte[i][j] = ((float)rand() / RAND_MAX) * 0.02f - 0.01f; // Random values between -0.01 and 0.01
        }
    }

    // Initialize positional embeddings (wpe)
    weights.wpe = (float **)malloc(MAX_POSITION_EMBEDDINGS * sizeof(float *));
    for (int i = 0; i < MAX_POSITION_EMBEDDINGS; i++) {
        weights.wpe[i] = (float *)malloc(EMBEDDING_SIZE * sizeof(float));
        for (int j = 0; j < EMBEDDING_SIZE; j++) {
            weights.wpe[i][j] = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;
        }
    }

    weights.blocks = (BlockWeights *)malloc(NUM_BLOCKS * sizeof(BlockWeights));
    for (int b = 0; b < NUM_BLOCKS; b++) {
        // Initialize Q, K, V linear layers using the helper function
        initialize_linear_layer(&weights.blocks[b].q_mlp, EMBEDDING_SIZE, EMBEDDING_SIZE);
        initialize_linear_layer(&weights.blocks[b].k_mlp, EMBEDDING_SIZE, EMBEDDING_SIZE);
        initialize_linear_layer(&weights.blocks[b].v_mlp, EMBEDDING_SIZE, EMBEDDING_SIZE);

        // Initialize MLP layers
        int mlpHiddenSize = EMBEDDING_SIZE * 4; // MLP hidden size is typically 4x the embedding size
        initialize_linear_layer(&weights.blocks[b].first_block_MLP, EMBEDDING_SIZE, mlpHiddenSize);
        initialize_linear_layer(&weights.blocks[b].second_block_MLP, mlpHiddenSize, EMBEDDING_SIZE);
    }

    // Initialize logits_mlp
    initialize_linear_layer(&weights.logits_mlp, EMBEDDING_SIZE, VOCAB_SIZE);

    printf("GPT-2 Weights initialization complete.\n");
    return weights;
}

// Function to free a LinearLayer
void free_linear_layer(LinearLayer *layer) {
    for (int i = 0; i < layer->fcOutputSize; i++) {
        free(layer->weights[i]);
    }
    free(layer->weights);
    free(layer->biases);
}

// Function to free GPT2Weights
void free_weights(GPT2Weights *weights) {
    // Free token embeddings
    for (int i = 0; i < VOCAB_SIZE; i++) {
        free(weights->wte[i]);
    }
    free(weights->wte);

    // Free positional embeddings
    for (int i = 0; i < MAX_POSITION_EMBEDDINGS; i++) {
        free(weights->wpe[i]);
    }
    free(weights->wpe);

    // Free transformer blocks
    for (int b = 0; b < NUM_BLOCKS; b++) {
        // Free Q, K, V linear layers
        free_linear_layer(&weights->blocks[b].q_mlp);
        free_linear_layer(&weights->blocks[b].k_mlp);
        free_linear_layer(&weights->blocks[b].v_mlp);

        // Free MLP layers
        free_linear_layer(&weights->blocks[b].first_block_MLP);
        free_linear_layer(&weights->blocks[b].second_block_MLP);
    }
    free(weights->blocks);

    // Free logits_mlp
    free_linear_layer(&weights->logits_mlp);
}

int main() {
    srand(42);
    int seqLength = 50;
    int tokens[] = { 10, 20, 30, 40, 50,10, 20, 30, 40, 50,10, 20, 30, 40, 50,10, 20, 30, 40, 50,10, 20, 30, 40, 50,10, 20, 30, 40, 50,10, 20, 30, 40, 50,10, 20, 30, 40, 50,10, 20, 30, 40, 50,10, 20, 30, 40, 50 };
    const int NUM_RUNS = 20;
    
    GPT2Weights weights = initialize_weights();
    
    // Timing variables
    clock_t start, end;
    double total_time = 0.0;
    
    // Run inference multiple times
    for (int i = 0; i < NUM_RUNS; i++) {
        start = clock();
        float *logits = model(tokens, seqLength, weights);
        end = clock();
        
        // Calculate time in milliseconds
        double time_taken = ((double)(end - start) * 1000.0) / CLOCKS_PER_SEC;
        total_time += time_taken;
        
        // Find max logit (keeping existing logic)
        int max_index = 0;
        float max_value = logits[0];
        for (int j = 1; j < VOCAB_SIZE; j++) {
            if (logits[j] > max_value) {
                max_value = logits[j];
                max_index = j;
            }
        }
        
        printf("Run %d - Inference time: %.2f ms, Predicted token: %d\n", 
               i + 1, time_taken, max_index);
        
        free(logits);
    }
    
    printf("\nAverage inference time over %d runs: %.2f ms\n", 
           NUM_RUNS, total_time / NUM_RUNS);
    
    free_weights(&weights);
    return 0;
}