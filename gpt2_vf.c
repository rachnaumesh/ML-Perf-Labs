#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#include <omp.h>


#define EPSILON 1e-5
#define EMBEDDING_SIZE 768 // GPT-2 base model embedding size
#define NUM_BLOCKS 12 // Number of transformer blocks in GPT-2 base model
#define NUM_HEADS 12 // Number of attention heads
#define HEAD_DIM (EMBEDDING_SIZE / NUM_HEADS) // Dimension of each attention head
#define VOCAB_SIZE 50257 // GPT-2 vocabulary size
#define MAX_POSITION_EMBEDDINGS 1024 // Maximum sequence length

// Assuming MatmulType is defined elsewhere
typedef enum { MATMUL_STANDARD, MATMUL_THREADED } MatmulType;

// Define the necessary data structures
typedef struct {
    int batch_size;
    int sequence_length;
    int features;
    float *data; // data[batch_size * sequence_length * features]
} Tensor3D;

typedef struct {
    int rows;
    int cols;
    float *data; // data[rows * cols]
} Tensor2D;

typedef struct {
    float **weights; // weights[fcOutputSize][fcInputSize]
    float *biases; // biases[fcOutputSize]
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
    float **wpe; // Positional embeddings
    float **wte; // Token embeddings
    BlockWeights *blocks;
    LinearLayer logits_mlp;
} GPT2Weights;

// Function prototypes
float *linear(float *fcInput, float **weights, float *biases, int fcInputSize, int fcOutputSize);
float **scaled_dot_product_attention(float **Q, float **K, float **V, int seqLength, int depth);
float **matrix_add(float **x, float **y, int numRow, int numCol);
float **norm(float **x, int seqLength, int features);
float *gelu(float *x, int size);
float **block(float **x, int seqLength, int embeddingSize, BlockWeights weights);
float *model(int *tokens, int seqLength, GPT2Weights weights);
int *positions_for(int *tokens, int seqLength, int past_length);


// Memory management functions
static inline void *aligned_malloc(size_t size) {
    void *ptr = _mm_malloc(size, 32);
    if (!ptr) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    return ptr;
}

static inline void aligned_free(void *ptr) {
    if (ptr) {
        _mm_free(ptr);
    }
}

static inline float **aligned_malloc_2d(int rows, int cols) {
    float **array = aligned_malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++) {
        array[i] = aligned_malloc(cols * sizeof(float));
    }
    return array;
}

static inline void aligned_free_2d(float **array, int rows) {
    if (array) {
        for (int i = 0; i < rows; i++) {
            _mm_free(array[i]);
        }
        _mm_free(array);
    }
}


// Core operations
static inline float horizontal_sum(__m256 sum_vec) {
    __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
    __m128 lo = _mm256_castps256_ps128(sum_vec);
    lo = _mm_add_ps(lo, hi);
    hi = _mm_movehl_ps(hi, lo);
    lo = _mm_add_ps(lo, hi);
    hi = _mm_shuffle_ps(lo, lo, 1);
    lo = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(lo);
}


// Optimized linear function with better SIMD
float *linear(float *fcInput, float **weights, float *biases, int fcInputSize, int fcOutputSize) {
    float *output = aligned_malloc(fcOutputSize * sizeof(float));
    
    #pragma omp parallel for
    for (int i = 0; i < fcOutputSize; i++) {
        __m256 sum_vec = _mm256_setzero_ps();
        for (int j = 0; j < fcInputSize - 7; j += 8) {
            __m256 input_vec = _mm256_load_ps(&fcInput[j]);
            __m256 weight_vec = _mm256_load_ps(&weights[i][j]);
            sum_vec = _mm256_fmadd_ps(input_vec, weight_vec, sum_vec);
        }
        
        float sum = horizontal_sum(sum_vec);
        for (int j = (fcInputSize/8)*8; j < fcInputSize; j++) {
            sum += fcInput[j] * weights[i][j];
        }
        output[i] = sum + biases[i];
    }
    return output;
}

// Optimized scaled dot-product attention with SIMD
float **scaled_dot_product_attention(float **Q, float **K, float **V, int seqLength, int depth) {
    float **scores = aligned_malloc_2d(seqLength, seqLength);
    float **attention_weights = aligned_malloc_2d(seqLength, seqLength);
    float **output = aligned_malloc_2d(seqLength, depth);
    float scale = 1.0f / sqrtf(depth);
    __m256 scale_vec = _mm256_set1_ps(scale);

    #pragma omp parallel for
    for (int i = 0; i < seqLength; i++) {
        for (int j = 0; j < seqLength; j++) {
            __m256 sum_vec = _mm256_setzero_ps();
            for (int k = 0; k < depth - 7; k += 8) {
                __m256 q_vec = _mm256_load_ps(&Q[i][k]);
                __m256 k_vec = _mm256_load_ps(&K[j][k]);
                sum_vec = _mm256_fmadd_ps(q_vec, k_vec, sum_vec);
            }
            scores[i][j] = horizontal_sum(sum_vec) * scale;

            for (int k = (depth/8)*8; k < depth; k++) {
                scores[i][j] += Q[i][k] * K[j][k] * scale;
            }
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < seqLength; i++) {
        float max_val = scores[i][0];
        for (int j = 1; j < seqLength; j++) {
            max_val = fmaxf(max_val, scores[i][j]);
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < seqLength; j++) {
            attention_weights[i][j] = expf(scores[i][j] - max_val);
            sum_exp += attention_weights[i][j];
        }

        __m256 sum_inv_vec = _mm256_set1_ps(1.0f / sum_exp);
        for (int j = 0; j < seqLength - 7; j += 8) {
            __m256 weight_vec = _mm256_load_ps(&attention_weights[i][j]);
            _mm256_store_ps(&attention_weights[i][j], _mm256_mul_ps(weight_vec, sum_inv_vec));
        }
        for (int j = (seqLength/8)*8; j < seqLength; j++) {
            attention_weights[i][j] /= sum_exp;
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < seqLength; i++) {
        for (int k = 0; k < depth - 7; k += 8) {
            __m256 sum_vec = _mm256_setzero_ps();
            for (int j = 0; j < seqLength; j++) {
                __m256 v_vec = _mm256_load_ps(&V[j][k]);
                __m256 weight_vec = _mm256_set1_ps(attention_weights[i][j]);
                sum_vec = _mm256_fmadd_ps(v_vec, weight_vec, sum_vec);
            }
            _mm256_store_ps(&output[i][k], sum_vec);
        }

        for (int k = (depth/8)*8; k < depth; k++) {
            float sum = 0.0f;
            for (int j = 0; j < seqLength; j++) {
                sum += attention_weights[i][j] * V[j][k];
            }
            output[i][k] = sum;
        }
    }

    aligned_free_2d(scores, seqLength);
    aligned_free_2d(attention_weights, seqLength);

    return output;
}


// Optimized matrix addition with SIMD
float **matrix_add(float **x, float **y, int numRow, int numCol) {
    float **result = aligned_malloc_2d(numRow, numCol);
    const int simd_width = 8;
    const int aligned_size = (numCol / simd_width) * simd_width;

    for (int i = 0; i < numRow; i++) {
        for (int j = 0; j < aligned_size; j += simd_width) {
            __m256 x_vec = _mm256_load_ps(&x[i][j]);
            __m256 y_vec = _mm256_load_ps(&y[i][j]);
            _mm256_store_ps(&result[i][j], _mm256_add_ps(x_vec, y_vec));
        }
        
        for (int j = aligned_size; j < numCol; j++) {
            result[i][j] = x[i][j] + y[i][j];
        }
    }
    return result;
}


// Optimized layer normalization
float **norm(float **x, int seqLength, int features) {
    float **normalized = aligned_malloc_2d(seqLength, features);
    
    #pragma omp parallel for
    for (int i = 0; i < seqLength; i++) {
        float mean = 0.0f, var = 0.0f;
        
        __m256 sum_vec = _mm256_setzero_ps();
        for (int j = 0; j < features - 7; j += 8) {
            __m256 x_vec = _mm256_load_ps(&x[i][j]);
            sum_vec = _mm256_add_ps(sum_vec, x_vec);
        }
        mean = horizontal_sum(sum_vec);
        
        for (int j = (features/8)*8; j < features; j++) {
            mean += x[i][j];
        }
        mean /= features;
        
        __m256 mean_vec = _mm256_set1_ps(mean);
        __m256 var_vec = _mm256_setzero_ps();
        for (int j = 0; j < features - 7; j += 8) {
            __m256 x_vec = _mm256_load_ps(&x[i][j]);
            __m256 diff = _mm256_sub_ps(x_vec, mean_vec);
            var_vec = _mm256_fmadd_ps(diff, diff, var_vec);
        }
        var = horizontal_sum(var_vec);
        
        for (int j = (features/8)*8; j < features; j++) {
            float diff = x[i][j] - mean;
            var += diff * diff;
        }
        
        var = sqrtf(var / features + EPSILON);
        __m256 scale_vec = _mm256_set1_ps(1.0f / var);
        
        for (int j = 0; j < features - 7; j += 8) {
            __m256 x_vec = _mm256_load_ps(&x[i][j]);
            __m256 normalized_vec = _mm256_mul_ps(_mm256_sub_ps(x_vec, mean_vec), scale_vec);
            _mm256_store_ps(&normalized[i][j], normalized_vec);
        }
        
        for (int j = (features/8)*8; j < features; j++) {
            normalized[i][j] = (x[i][j] - mean) / var;
        }
    }
    return normalized;
}


// Implement the GELU activation function
float *gelu(float *x, int size) {
    float *output = aligned_malloc(size * sizeof(float));
    
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        output[i] = 0.5f * x[i] * (1.0f + tanhf(0.797884f * (x[i] + 0.044715f * x[i] * x[i] * x[i])));
    }
    return output;
}


// Function to compute positions
int *positions_for(int *tokens, int seqLength, int past_length) {
    int *positions = aligned_malloc(seqLength * sizeof(int));
    for (int i = 0; i < seqLength; i++) {
        positions[i] = past_length + i;
    }
    return positions;
}

// Implement the transformer block with multi-head attention
float **block(float **x, int seqLength, int embeddingSize, BlockWeights weights) {
    // Apply layer normalization to x
    float **normalized_x = norm(x, seqLength, embeddingSize);

    // Allocate memory for Q, K, V with proper alignment
    float **Q = aligned_malloc_2d(seqLength, embeddingSize);
    float **K = aligned_malloc_2d(seqLength, embeddingSize);
    float **V = aligned_malloc_2d(seqLength, embeddingSize);
    
    // Process each sequence position
    #pragma omp parallel for
    for (int i = 0; i < seqLength; i++) {
        Q[i] = linear(normalized_x[i], weights.q_mlp.weights, weights.q_mlp.biases, 
                     weights.q_mlp.fcInputSize, weights.q_mlp.fcOutputSize);
        K[i] = linear(normalized_x[i], weights.k_mlp.weights, weights.k_mlp.biases,
                     weights.k_mlp.fcInputSize, weights.k_mlp.fcOutputSize);
        V[i] = linear(normalized_x[i], weights.v_mlp.weights, weights.v_mlp.biases,
                     weights.v_mlp.fcInputSize, weights.v_mlp.fcOutputSize);
    }

    // Properly allocate head arrays
    float ***Q_heads = (float ***)aligned_malloc(NUM_HEADS * sizeof(float **));
    float ***K_heads = (float ***)aligned_malloc(NUM_HEADS * sizeof(float **));
    float ***V_heads = (float ***)aligned_malloc(NUM_HEADS * sizeof(float **));
    
    for (int h = 0; h < NUM_HEADS; h++) {
        Q_heads[h] = aligned_malloc_2d(seqLength, HEAD_DIM);
        K_heads[h] = aligned_malloc_2d(seqLength, HEAD_DIM);
        V_heads[h] = aligned_malloc_2d(seqLength, HEAD_DIM);
        
        for (int i = 0; i < seqLength; i++) {
            memcpy(Q_heads[h][i], &Q[i][h * HEAD_DIM], HEAD_DIM * sizeof(float));
            memcpy(K_heads[h][i], &K[i][h * HEAD_DIM], HEAD_DIM * sizeof(float));
            memcpy(V_heads[h][i], &V[i][h * HEAD_DIM], HEAD_DIM * sizeof(float));
        }
    }

    // Process attention heads
    float ***head_outputs = (float ***)aligned_malloc(NUM_HEADS * sizeof(float **));
    for (int h = 0; h < NUM_HEADS; h++) {
        head_outputs[h] = scaled_dot_product_attention(Q_heads[h], K_heads[h], V_heads[h], 
                                                     seqLength, HEAD_DIM);
    }

    // Concatenate head outputs
    float **a = aligned_malloc_2d(seqLength, embeddingSize);
    for (int i = 0; i < seqLength; i++) {
        for (int h = 0; h < NUM_HEADS; h++) {
            memcpy(&a[i][h * HEAD_DIM], head_outputs[h][i], HEAD_DIM * sizeof(float));
        }
    }

    // Process residual connections and MLP
    float **x_added = matrix_add(x, a, seqLength, embeddingSize);
    float **normalized_x_added = norm(x_added, seqLength, embeddingSize);
    float **m = aligned_malloc_2d(seqLength, embeddingSize);

    #pragma omp parallel for
    for (int i = 0; i < seqLength; i++) {
        float *first_mlp_output = linear(normalized_x_added[i], 
                                       weights.first_block_MLP.weights,
                                       weights.first_block_MLP.biases, 
                                       weights.first_block_MLP.fcInputSize,
                                       weights.first_block_MLP.fcOutputSize);
        float *gelu_output = gelu(first_mlp_output, weights.first_block_MLP.fcOutputSize);
        m[i] = linear(gelu_output, weights.second_block_MLP.weights,
                     weights.second_block_MLP.biases,
                     weights.second_block_MLP.fcInputSize,
                     weights.second_block_MLP.fcOutputSize);
        aligned_free(first_mlp_output);
        aligned_free(gelu_output);
    }

    float **output = matrix_add(x_added, m, seqLength, embeddingSize);

    // Proper cleanup
    aligned_free_2d(normalized_x, seqLength);
    aligned_free_2d(Q, seqLength);
    aligned_free_2d(K, seqLength);
    aligned_free_2d(V, seqLength);
    aligned_free_2d(normalized_x_added, seqLength);
    aligned_free_2d(m, seqLength);
    aligned_free_2d(x_added, seqLength);
    aligned_free_2d(a, seqLength);

    for (int h = 0; h < NUM_HEADS; h++) {
        aligned_free_2d(Q_heads[h], seqLength);
        aligned_free_2d(K_heads[h], seqLength);
        aligned_free_2d(V_heads[h], seqLength);
        aligned_free_2d(head_outputs[h], seqLength);
    }
    aligned_free(Q_heads);
    aligned_free(K_heads);
    aligned_free(V_heads);
    aligned_free(head_outputs);

    return output;
}


// Implement the model function with positional embeddings
float *model(int *tokens, int seqLength, GPT2Weights weights) {
    float **h = aligned_malloc_2d(seqLength, EMBEDDING_SIZE);
    int past_length = 0;
    int *positions = positions_for(tokens, seqLength, past_length);
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < seqLength; i++) {
        for (int j = 0; j < EMBEDDING_SIZE; j++) {
            h[i][j] = weights.wte[tokens[i]][j] + weights.wpe[positions[i]][j];
        }
    }
    
    aligned_free(positions);
    
    for (int i = 0; i < NUM_BLOCKS; i++) {
        float **new_h = block(h, seqLength, EMBEDDING_SIZE, weights.blocks[i]);
        aligned_free_2d(h, seqLength);
        h = new_h;
    }
    
    float *logits = linear(h[seqLength - 1], weights.logits_mlp.weights, 
                          weights.logits_mlp.biases, EMBEDDING_SIZE, VOCAB_SIZE);
    
    aligned_free_2d(h, seqLength);
    return logits;
}


void initialize_linear_layer(LinearLayer *layer, int input_size, int output_size) {
    layer->fcInputSize = input_size;
    layer->fcOutputSize = output_size;
    layer->weights = aligned_malloc_2d(output_size, input_size);
    layer->biases = (float *)aligned_malloc(output_size * sizeof(float));
    
    #pragma omp parallel for
    for (int i = 0; i < output_size; i++) {
        layer->biases[i] = 0.0f;
        for (int j = 0; j < input_size; j++) {
            layer->weights[i][j] = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;
        }
    }
}

GPT2Weights initialize_weights(void) {
    GPT2Weights weights;
    
    weights.wte = aligned_malloc_2d(VOCAB_SIZE, EMBEDDING_SIZE);
    weights.wpe = aligned_malloc_2d(MAX_POSITION_EMBEDDINGS, EMBEDDING_SIZE);
    weights.blocks = aligned_malloc(NUM_BLOCKS * sizeof(BlockWeights));
    
    for (int b = 0; b < NUM_BLOCKS; b++) {
        initialize_linear_layer(&weights.blocks[b].q_mlp, EMBEDDING_SIZE, EMBEDDING_SIZE);
        initialize_linear_layer(&weights.blocks[b].k_mlp, EMBEDDING_SIZE, EMBEDDING_SIZE);
        initialize_linear_layer(&weights.blocks[b].v_mlp, EMBEDDING_SIZE, EMBEDDING_SIZE);
        initialize_linear_layer(&weights.blocks[b].first_block_MLP, EMBEDDING_SIZE, EMBEDDING_SIZE * 4);
        initialize_linear_layer(&weights.blocks[b].second_block_MLP, EMBEDDING_SIZE * 4, EMBEDDING_SIZE);
    }
    
    initialize_linear_layer(&weights.logits_mlp, EMBEDDING_SIZE, VOCAB_SIZE);

    printf("GPT-2 Weights initialization complete.\n");
    return weights;
}

// Function to free a LinearLayer
void free_linear_layer(LinearLayer *layer) {
    if (layer) {
        aligned_free_2d(layer->weights, layer->fcOutputSize);
        _mm_free(layer->biases);
    }
}

// Function to free GPT2Weights
void free_weights(GPT2Weights *weights) {
    aligned_free_2d(weights->wte, VOCAB_SIZE);
    aligned_free_2d(weights->wpe, MAX_POSITION_EMBEDDINGS);
    
    for (int b = 0; b < NUM_BLOCKS; b++) {
        free_linear_layer(&weights->blocks[b].q_mlp);
        free_linear_layer(&weights->blocks[b].k_mlp);
        free_linear_layer(&weights->blocks[b].v_mlp);
        free_linear_layer(&weights->blocks[b].first_block_MLP);
        free_linear_layer(&weights->blocks[b].second_block_MLP);
    }
    
    _mm_free(weights->blocks);
    free_linear_layer(&weights->logits_mlp);
}

// Test case
int main() {
    clock_t start = clock();
    srand(42);
    int seqLength = 5;
    int tokens[] = { 10, 20, 30, 40, 50 };
    
    GPT2Weights weights = initialize_weights();
    float *logits = model(tokens, seqLength, weights);
    
    int max_index = 0;
    float max_value = logits[0];
    for (int i = 1; i < VOCAB_SIZE; i++) {
        if (logits[i] > max_value) {
            max_value = logits[i];
            max_index = i;
        }
    }
    
    printf("Predicted next token ID: %d\n", max_index);
    aligned_free(logits);
    free_weights(&weights);
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Execution time: %f seconds\n", time_spent);
    return 0;
}