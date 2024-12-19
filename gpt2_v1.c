#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#include <stdint.h>
#include <cpuid.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define SUCCESS 0
#define ERROR -1

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


// Add CPU feature detection
static inline int check_avx_support() {
    unsigned int eax, ebx, ecx, edx;
    
    // Check for AVX support
    __cpuid(1, eax, ebx, ecx, edx);
    return (ecx & (1 << 28)) != 0;  // AVX bit
}


// Add fast exponential approximation for AVX
static inline __m256 exp256_ps(__m256 x) {
    const __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
    const __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);
    
    // Clamp the values to avoid overflow
    x = _mm256_min_ps(x, exp_hi);
    x = _mm256_max_ps(x, exp_lo);
    
    // exp(x) = 2^(x * log2(e))
    __m256 t = _mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f));
    
    // Split into integer and fractional parts
    __m256i i = _mm256_cvtps_epi32(t);
    __m256 f = _mm256_sub_ps(t, _mm256_cvtepi32_ps(i));
    
    // Approximate 2^f using polynomial
    __m256 p = _mm256_fmadd_ps(f, _mm256_set1_ps(0.693147180559945f),
                               _mm256_set1_ps(1.0f));
    
    // Combine parts
    __m256 result = _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_add_epi32(i, _mm256_set1_epi32(127)), 23)
    );
    
    return _mm256_mul_ps(result, p);
}

// Add a fast tanh approximation for AVX
static inline __m256 tanh256_ps(__m256 x) {
    // Clamp input to avoid overflow
    __m256 abs_x = _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
    __m256 sign_x = _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000)));
    
    // Approximate tanh using exp
    __m256 exp_2x = exp256_ps(_mm256_add_ps(x, x));
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 result = _mm256_div_ps(_mm256_sub_ps(exp_2x, one),
                                 _mm256_add_ps(exp_2x, one));
    
    return result;
}

// Helper function for aligned memory allocation
float* aligned_malloc(size_t size) {
    float* temp = (float *)malloc(size * sizeof(float));
    float* aligned = (float*)_mm_malloc(size * sizeof(float), 32);
    if (temp && aligned) {
        memcpy(aligned, temp, size * sizeof(float));
    }
    free(temp);
    return aligned;
}

// Helper function to free aligned memory
void aligned_free(void* ptr) {
    _mm_free(ptr);
}

void aligned_free_2d(float** matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        _mm_free(matrix[i]);
    }
    _mm_free(matrix);
}

// AVX-optimized matrix multiplication
void matrix_multiply_avx(float** A, float** B, float** C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            __m256 sum = _mm256_setzero_ps(); // Sum 8 floats at once
            
            // Process 8 elements at a time
            for (int k = 0; k <= K - 8; k += 8) {
                __m256 a = _mm256_load_ps(&A[i][k]); 
                __m256 b = _mm256_load_ps(&B[j][k]); 
                sum = _mm256_add_ps(sum, _mm256_mul_ps(a, b));
            }
            
            float temp[8] __attribute__((aligned(32)));
            _mm256_store_ps(temp, sum);
            C[i][j] = temp[0] + temp[1] + temp[2] + temp[3] + 
                      temp[4] + temp[5] + temp[6] + temp[7];
            
            // Handle remaining elements
            for (int k = (K/8)*8; k < K; k++) {
                C[i][j] += A[i][k] * B[j][k];
            }
        }
    }
}

// AVX-optimized layer normalization
void layer_norm_avx(float** x, float** output, int seqLength, int features) {
    for (int i = 0; i < seqLength; i++) {
        // Calculate mean using AVX
        __m256 sum_vec = _mm256_setzero_ps();
        for (int j = 0; j <= features - 8; j += 8) {
            __m256 x_vec = _mm256_load_ps(&x[i][j]);
            sum_vec = _mm256_add_ps(sum_vec, x_vec);
        }
        
        // Horizontal sum for mean
        float temp[8] __attribute__((aligned(32)));
        _mm256_store_ps(temp, sum_vec);
        float mean = (temp[0] + temp[1] + temp[2] + temp[3] + 
                     temp[4] + temp[5] + temp[6] + temp[7]) / features;
        
        // Handle remaining elements for mean
        for (int j = (features/8)*8; j < features; j++) {
            mean += x[i][j];
        }
        mean /= features;
        
        // Calculate variance using AVX
        __m256 mean_vec = _mm256_set1_ps(mean);
        __m256 var_vec = _mm256_setzero_ps();
        
        for (int j = 0; j <= features - 8; j += 8) {
            __m256 x_vec = _mm256_load_ps(&x[i][j]);
            __m256 diff = _mm256_sub_ps(x_vec, mean_vec);
            var_vec = _mm256_add_ps(var_vec, _mm256_mul_ps(diff, diff));
        }
        
        // Horizontal sum for variance
        _mm256_store_ps(temp, var_vec);
        float variance = (temp[0] + temp[1] + temp[2] + temp[3] + 
                         temp[4] + temp[5] + temp[6] + temp[7]) / features;
        
        // Handle remaining elements for variance
        for (int j = (features/8)*8; j < features; j++) {
            float diff = x[i][j] - mean;
            variance += diff * diff;
        }
        variance /= features;
        
        // Normalize using AVX
        __m256 var_eps = _mm256_set1_ps(sqrtf(variance + EPSILON));
        
        for (int j = 0; j <= features - 8; j += 8) {
            __m256 x_vec = _mm256_load_ps(&x[i][j]);
            __m256 normalized = _mm256_div_ps(
                _mm256_sub_ps(x_vec, mean_vec),
                var_eps
            );
            _mm256_store_ps(&output[i][j], normalized);
        }
        
        // Handle remaining elements
        for (int j = (features/8)*8; j < features; j++) {
            output[i][j] = (x[i][j] - mean) / sqrtf(variance + EPSILON);
        }
    }
}

// AVX-optimized GELU activation
void gelu_avx(float* input, float* output, int size) {
    const __m256 c1 = _mm256_set1_ps(0.044715f);
    const __m256 c2 = _mm256_set1_ps(sqrt(2.0f/M_PI));
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    
    for (int i = 0; i <= size - 8; i += 8) {
        __m256 x = _mm256_load_ps(&input[i]);
        
        // Calculate x³
        __m256 x_squared = _mm256_mul_ps(x, x);
        __m256 x_cubed = _mm256_mul_ps(x_squared, x);
        
        // Calculate 0.044715 * x³
        __m256 term1 = _mm256_mul_ps(c1, x_cubed);
        
        // Calculate x + 0.044715 * x³
        __m256 inner_term = _mm256_add_ps(x, term1);
        
        // Calculate sqrt(2/π) * (x + 0.044715 * x³)
        __m256 scaled_term = _mm256_mul_ps(c2, inner_term);
        
        // Calculate tanh(sqrt(2/π) * (x + 0.044715 * x³))
        // Using our previously defined tanh approximation
        __m256 tanh_val = tanh256_ps(scaled_term);
        
        // Calculate 1 + tanh(...)
        __m256 one_plus_tanh = _mm256_add_ps(one, tanh_val);
        
        // Calculate 0.5 * x
        __m256 half_x = _mm256_mul_ps(half, x);
        
        // Calculate final result: 0.5 * x * (1 + tanh(...))
        __m256 result = _mm256_mul_ps(half_x, one_plus_tanh);
        
        // Store the result
        _mm256_store_ps(&output[i], result);
    }
    
    // Handle remaining elements
    for (int i = (size/8)*8; i < size; i++) {
        float x3 = input[i] * input[i] * input[i];
        float inner = sqrt(2.0f/M_PI) * (input[i] + 0.044715f * x3);
        output[i] = 0.5f * input[i] * (1.0f + tanhf(inner));
    }
}


float *linear(float *fcInput, float **weights, float *biases, int fcInputSize, int fcOutputSize) {
    float *output = (float *)aligned_malloc(fcOutputSize);
    
    #pragma omp parallel for
    for (int i = 0; i < fcOutputSize; i++) {
        output[i] = biases[i];
        __m256 sum_vec = _mm256_setzero_ps();
        
        // Process 8 elements at a time
        for (int j = 0; j <= fcInputSize - 8; j += 8) {
            __m256 input_vec = _mm256_loadu_ps(&fcInput[j]);
            __m256 weight_vec = _mm256_loadu_ps(&weights[i][j]);
            sum_vec = _mm256_fmadd_ps(input_vec, weight_vec, sum_vec);
        }
        
        // Reduce sum in consistent order
        float temp[8] __attribute__((aligned(32)));
        _mm256_store_ps(temp, sum_vec);
        output[i] += temp[0] + temp[1] + temp[2] + temp[3] + 
                     temp[4] + temp[5] + temp[6] + temp[7];
        
        // Handle remaining elements exactly as in original
        for (int j = (fcInputSize/8)*8; j < fcInputSize; j++) {
            output[i] += fcInput[j] * weights[i][j];
        }
    }
    
    return output;
}


// Replace scaled_dot_product_attention with optimized version
float **scaled_dot_product_attention(float **Q, float **K, float **V, int seqLength, int depth) {
    float scale = 1.0f / sqrt(depth);
    __m256 scale_vec = _mm256_set1_ps(scale);
    
    // Allocate aligned memory but compute in same order as original
    float **scores = aligned_malloc_2d(seqLength, seqLength);
    float **attention_weights = aligned_malloc_2d(seqLength, seqLength);
    float **output = aligned_malloc_2d(seqLength, depth);
    
    // Compute Q * K^T with SIMD but maintain ordering
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < seqLength; i++) {
        for (int j = 0; j < seqLength; j++) {
            __m256 sum_vec = _mm256_setzero_ps();
            
            for (int k = 0; k <= depth - 8; k += 8) {
                __m256 q_vec = _mm256_load_ps(&Q[i][k]);
                __m256 k_vec = _mm256_load_ps(&K[j][k]);
                sum_vec = _mm256_fmadd_ps(q_vec, k_vec, scale_vec);
            }
            
            float temp[8] __attribute__((aligned(32)));
            _mm256_store_ps(temp, sum_vec);
            scores[i][j] = (temp[0] + temp[1] + temp[2] + temp[3] + 
                           temp[4] + temp[5] + temp[6] + temp[7]);
            
            // Handle remaining elements
            for (int k = (depth/8)*8; k < depth; k++) {
                scores[i][j] += Q[i][k] * K[j][k] * scale;
            }
        }
    }

    // Apply softmax maintaining numerical consistency
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
    
    // Compute output with SIMD but maintain consistency
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < seqLength; i++) {
        for (int k = 0; k < depth; k += 8) {
            __m256 sum_vec = _mm256_setzero_ps();
            
            for (int j = 0; j < seqLength; j++) {
                __m256 v_vec = _mm256_load_ps(&V[j][k]);
                __m256 weight_vec = _mm256_set1_ps(attention_weights[i][j]);
                sum_vec = _mm256_fmadd_ps(v_vec, weight_vec, sum_vec);
            }
            
            _mm256_store_ps(&output[i][k], sum_vec);
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
    
    // Free intermediate results
    aligned_free_2d(scores, seqLength);
    aligned_free_2d(attention_weights, seqLength);
    
    return output;
}

// AVX-optimized matrix addition
float **matrix_add(float **x, float **y, int numRow, int numCol) {
    float **result = aligned_malloc_2d(numRow, numCol);
    if (!result) return NULL;
    
    for (int i = 0; i < numRow; i++) {
        for (int j = 0; j <= numCol - 8; j += 8) {
            __m256 x_vec = _mm256_load_ps(&x[i][j]);
            __m256 y_vec = _mm256_load_ps(&y[i][j]);
            __m256 sum = _mm256_add_ps(x_vec, y_vec);
            _mm256_store_ps(&result[i][j], sum);
        }
        
        // Handle remaining elements
        for (int j = (numCol/8)*8; j < numCol; j++) {
            result[i][j] = x[i][j] + y[i][j];
        }
    }
    
    return result;
}


// Implement layer normalization
float **norm(float **x, int seqLength, int features) {
    float **normalized = aligned_malloc_2d(seqLength, features);
    if (!normalized) return NULL;
    
    // Use the existing layer_norm_avx function
    layer_norm_avx(x, normalized, seqLength, features);
    return normalized;
}

// Implement the GELU activation function
float *gelu(float *x, int size) {
    float *output = aligned_malloc(size);
    if (!output) return NULL;
    gelu_avx(x, output, size);  // Use the AVX version directly
    return output;
}

// Function to compute positions
int *positions_for(int *tokens, int seqLength, int past_length) {
    int *positions = (int *)aligned_malloc(seqLength * sizeof(int));
    if (!positions) return NULL;  // Add this line
    for (int i = 0; i < seqLength; i++) {
        positions[i] = past_length + i;
    }
    return positions;
}

// Implement the transformer block with multi-head attention
#define SUCCESS 0
#define ERROR -1

float **block(float **x, int seqLength, int embeddingSize, BlockWeights weights) {
    float **normalized_x = NULL, **Q = NULL, **K = NULL, **V = NULL;
    float ***Q_heads = NULL, ***K_heads = NULL, ***V_heads = NULL;
    float ***head_outputs = NULL;
    float **a = NULL, **x_added = NULL, **normalized_x_added = NULL;
    float **m = NULL, **output = NULL;
    int compute_qkv_status = SUCCESS;
    int process_heads_status = SUCCESS;
    int mlp_status = SUCCESS;

    // 1. Layer normalization
    normalized_x = norm(x, seqLength, embeddingSize);
    if (!normalized_x) {
        return NULL;
    }

    // 2. Allocate Q, K, V matrices
    Q = aligned_malloc_2d(seqLength, embeddingSize);
    K = aligned_malloc_2d(seqLength, embeddingSize);
    V = aligned_malloc_2d(seqLength, embeddingSize);
    if (!Q || !K || !V) {
        if (normalized_x) aligned_free_2d(normalized_x, seqLength);
        if (Q) aligned_free_2d(Q, seqLength);
        if (K) aligned_free_2d(K, seqLength);
        if (V) aligned_free_2d(V, seqLength);
        return NULL;
    }

    // 3. Compute Q, K, V transformations in parallel
    #pragma omp parallel for
    for (int i = 0; i < seqLength; i++) {
        if (compute_qkv_status == SUCCESS) {
            Q[i] = linear(normalized_x[i], weights.q_mlp.weights, weights.q_mlp.biases,
                         weights.q_mlp.fcInputSize, weights.q_mlp.fcOutputSize);
            K[i] = linear(normalized_x[i], weights.k_mlp.weights, weights.k_mlp.biases,
                         weights.k_mlp.fcInputSize, weights.k_mlp.fcOutputSize);
            V[i] = linear(normalized_x[i], weights.v_mlp.weights, weights.v_mlp.biases,
                         weights.v_mlp.fcInputSize, weights.v_mlp.fcOutputSize);
            if (!Q[i] || !K[i] || !V[i]) {
                #pragma omp atomic write
                compute_qkv_status = ERROR;
            }
        }
    }

    if (compute_qkv_status == ERROR) {
        aligned_free_2d(normalized_x, seqLength);
        aligned_free_2d(Q, seqLength);
        aligned_free_2d(K, seqLength);
        aligned_free_2d(V, seqLength);
        return NULL;
    }

    // 4. Allocate head arrays
    Q_heads = (float ***)aligned_malloc(NUM_HEADS * sizeof(float **));
    K_heads = (float ***)aligned_malloc(NUM_HEADS * sizeof(float **));
    V_heads = (float ***)aligned_malloc(NUM_HEADS * sizeof(float **));
    head_outputs = (float ***)aligned_malloc(NUM_HEADS * sizeof(float **));
    
    if (!Q_heads || !K_heads || !V_heads || !head_outputs) {
        if (normalized_x) aligned_free_2d(normalized_x, seqLength);
        if (Q) aligned_free_2d(Q, seqLength);
        if (K) aligned_free_2d(K, seqLength);
        if (V) aligned_free_2d(V, seqLength);
        if (Q_heads) aligned_free(Q_heads);
        if (K_heads) aligned_free(K_heads);
        if (V_heads) aligned_free(V_heads);
        if (head_outputs) aligned_free(head_outputs);
        return NULL;
    }

    // 5. Initialize head matrices
    for (int h = 0; h < NUM_HEADS; h++) {
        Q_heads[h] = aligned_malloc_2d(seqLength, HEAD_DIM);
        K_heads[h] = aligned_malloc_2d(seqLength, HEAD_DIM);
        V_heads[h] = aligned_malloc_2d(seqLength, HEAD_DIM);
        if (!Q_heads[h] || !K_heads[h] || !V_heads[h]) {
            for (int j = 0; j <= h; j++) {
                if (Q_heads[j]) aligned_free_2d(Q_heads[j], seqLength);
                if (K_heads[j]) aligned_free_2d(K_heads[j], seqLength);
                if (V_heads[j]) aligned_free_2d(V_heads[j], seqLength);
            }
            aligned_free(Q_heads);
            aligned_free(K_heads);
            aligned_free(V_heads);
            aligned_free(head_outputs);
            aligned_free_2d(normalized_x, seqLength);
            aligned_free_2d(Q, seqLength);
            aligned_free_2d(K, seqLength);
            aligned_free_2d(V, seqLength);
            return NULL;
        }
    }

    // 6. Split Q, K, V into heads in parallel
    #pragma omp parallel for collapse(2)
    for (int h = 0; h < NUM_HEADS; h++) {
        for (int i = 0; i < seqLength; i++) {
            memcpy(Q_heads[h][i], &Q[i][h * HEAD_DIM], HEAD_DIM * sizeof(float));
            memcpy(K_heads[h][i], &K[i][h * HEAD_DIM], HEAD_DIM * sizeof(float));
            memcpy(V_heads[h][i], &V[i][h * HEAD_DIM], HEAD_DIM * sizeof(float));
        }
    }

    // 7. Process attention heads in parallel
    #pragma omp parallel for
    for (int h = 0; h < NUM_HEADS; h++) {
        if (process_heads_status == SUCCESS) {
            head_outputs[h] = scaled_dot_product_attention(Q_heads[h], K_heads[h], V_heads[h], 
                                                         seqLength, HEAD_DIM);
            if (!head_outputs[h]) {
                #pragma omp atomic write
                process_heads_status = ERROR;
            }
        }
    }

    if (process_heads_status == ERROR) {
        for (int h = 0; h < NUM_HEADS; h++) {
            if (Q_heads[h]) aligned_free_2d(Q_heads[h], seqLength);
            if (K_heads[h]) aligned_free_2d(K_heads[h], seqLength);
            if (V_heads[h]) aligned_free_2d(V_heads[h], seqLength);
            if (head_outputs[h]) aligned_free_2d(head_outputs[h], seqLength);
        }
        aligned_free(Q_heads);
        aligned_free(K_heads);
        aligned_free(V_heads);
        aligned_free(head_outputs);
        aligned_free_2d(normalized_x, seqLength);
        aligned_free_2d(Q, seqLength);
        aligned_free_2d(K, seqLength);
        aligned_free_2d(V, seqLength);
        return NULL;
    }

    // 8. Concatenate head outputs
    a = aligned_malloc_2d(seqLength, embeddingSize);
    if (!a) {
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
        aligned_free_2d(normalized_x, seqLength);
        aligned_free_2d(Q, seqLength);
        aligned_free_2d(K, seqLength);
        aligned_free_2d(V, seqLength);
        return NULL;
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < seqLength; i++) {
        for (int h = 0; h < NUM_HEADS; h++) {
            memcpy(&a[i][h * HEAD_DIM], head_outputs[h][i], HEAD_DIM * sizeof(float));
        }
    }

    // 9. Add residual connection and apply normalization
    x_added = matrix_add(x, a, seqLength, embeddingSize);
    if (!x_added) {
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
        aligned_free_2d(normalized_x, seqLength);
        aligned_free_2d(Q, seqLength);
        aligned_free_2d(K, seqLength);
        aligned_free_2d(V, seqLength);
        aligned_free_2d(a, seqLength);
        return NULL;
    }

    normalized_x_added = norm(x_added, seqLength, embeddingSize);
    if (!normalized_x_added) {
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
        aligned_free_2d(normalized_x, seqLength);
        aligned_free_2d(Q, seqLength);
        aligned_free_2d(K, seqLength);
        aligned_free_2d(V, seqLength);
        aligned_free_2d(a, seqLength);
        aligned_free_2d(x_added, seqLength);
        return NULL;
    }

    // 10. Process MLP layers
    m = aligned_malloc_2d(seqLength, embeddingSize);
    if (!m) {
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
        aligned_free_2d(normalized_x, seqLength);
        aligned_free_2d(Q, seqLength);
        aligned_free_2d(K, seqLength);
        aligned_free_2d(V, seqLength);
        aligned_free_2d(a, seqLength);
        aligned_free_2d(x_added, seqLength);
        aligned_free_2d(normalized_x_added, seqLength);
        return NULL;
    }

    #pragma omp parallel for
    for (int i = 0; i < seqLength; i++) {
        if (mlp_status == SUCCESS) {
            float *first_mlp_output = linear(normalized_x_added[i], 
                                           weights.first_block_MLP.weights,
                                           weights.first_block_MLP.biases,
                                           weights.first_block_MLP.fcInputSize,
                                           weights.first_block_MLP.fcOutputSize);
            if (!first_mlp_output) {
                #pragma omp atomic write
                mlp_status = ERROR;
                continue;
            }

            float *gelu_output = gelu(first_mlp_output, weights.first_block_MLP.fcOutputSize);
            if (!gelu_output) {
                aligned_free(first_mlp_output);
                #pragma omp atomic write
                mlp_status = ERROR;
                continue;
            }

            m[i] = linear(gelu_output, 
                         weights.second_block_MLP.weights,
                         weights.second_block_MLP.biases,
                         weights.second_block_MLP.fcInputSize,
                         weights.second_block_MLP.fcOutputSize);

            aligned_free(first_mlp_output);
            aligned_free(gelu_output);

            if (!m[i]) {
                #pragma omp atomic write
                mlp_status = ERROR;
            }
        }
    }

    if (mlp_status == ERROR) {
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
        aligned_free_2d(normalized_x, seqLength);
        aligned_free_2d(Q, seqLength);
        aligned_free_2d(K, seqLength);
        aligned_free_2d(V, seqLength);
        aligned_free_2d(a, seqLength);
        aligned_free_2d(x_added, seqLength);
        aligned_free_2d(normalized_x_added, seqLength);
        aligned_free_2d(m, seqLength);
        return NULL;
    }

    // 11. Final residual connection
    output = matrix_add(x_added, m, seqLength, embeddingSize);

    // 12. Clean up all intermediate allocations
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
    aligned_free_2d(normalized_x, seqLength);
    aligned_free_2d(Q, seqLength);
    aligned_free_2d(K, seqLength);
    aligned_free_2d(V, seqLength);
    aligned_free_2d(a, seqLength);
    aligned_free_2d(x_added, seqLength);
    aligned_free_2d(normalized_x_added, seqLength);
    aligned_free_2d(m, seqLength);

    return output;
}

// Implement the model function with positional embeddings
// Update model function with aligned memory
float *model(int *tokens, int seqLength, GPT2Weights weights) {
    
    int past_length = 0;
    int *positions = positions_for(tokens, seqLength, past_length);
    
    float **h = aligned_malloc_2d(seqLength, EMBEDDING_SIZE);
    
    // Initialize embeddings
    for (int i = 0; i < seqLength; i++) {
        for (int j = 0; j < EMBEDDING_SIZE; j++) {
            h[i][j] = weights.wte[tokens[i]][j] + weights.wpe[positions[i]][j];
        }
    }
    
    aligned_free(positions);
    
    // Pass through transformer blocks
    for (int i = 0; i < NUM_BLOCKS; i++) {
        float **new_h = block(h, seqLength, EMBEDDING_SIZE, weights.blocks[i]);
        aligned_free_2d(h, seqLength);
        h = new_h;
    }
    
    // Get logits for the last token
    LinearLayer logits_mlp = weights.logits_mlp;
    float *logits = linear(h[seqLength - 1], logits_mlp.weights, logits_mlp.biases, 
                          logits_mlp.fcInputSize, logits_mlp.fcOutputSize);
    
    aligned_free_2d(h, seqLength);
    return logits;
}


void initialize_linear_layer(LinearLayer *layer, int inputSize, int outputSize) {
    layer->fcInputSize = inputSize;
    layer->fcOutputSize = outputSize;
    layer->weights = aligned_malloc_2d(outputSize, inputSize);
    layer->biases = aligned_malloc(outputSize);
    
    for (int i = 0; i < outputSize; i++) {
        layer->biases[i] = 0.0f;
        for (int j = 0; j < inputSize; j++) {
            layer->weights[i][j] = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;
        }
    }
}


GPT2Weights initialize_weights() {
    GPT2Weights weights;
    
    // Initialize using regular malloc first to maintain random pattern
    weights.wte = (float **)malloc(VOCAB_SIZE * sizeof(float *));
    float** temp_wte = (float **)malloc(VOCAB_SIZE * sizeof(float *));
    
    for (int i = 0; i < VOCAB_SIZE; i++) {
        temp_wte[i] = (float *)malloc(EMBEDDING_SIZE * sizeof(float));
        for (int j = 0; j < EMBEDDING_SIZE; j++) {
            temp_wte[i][j] = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;
        }
    }
    
    // Now copy to aligned memory
    weights.wte = aligned_malloc_2d(VOCAB_SIZE, EMBEDDING_SIZE);
    for (int i = 0; i < VOCAB_SIZE; i++) {
        memcpy(weights.wte[i], temp_wte[i], EMBEDDING_SIZE * sizeof(float));
        free(temp_wte[i]);
    }
    free(temp_wte);

    // Do the same for positional embeddings
    float** temp_wpe = (float **)malloc(MAX_POSITION_EMBEDDINGS * sizeof(float *));
    for (int i = 0; i < MAX_POSITION_EMBEDDINGS; i++) {
        temp_wpe[i] = (float *)malloc(EMBEDDING_SIZE * sizeof(float));
        for (int j = 0; j < EMBEDDING_SIZE; j++) {
            temp_wpe[i][j] = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;
        }
    }
    
    weights.wpe = aligned_malloc_2d(MAX_POSITION_EMBEDDINGS, EMBEDDING_SIZE);
    for (int i = 0; i < MAX_POSITION_EMBEDDINGS; i++) {
        memcpy(weights.wpe[i], temp_wpe[i], EMBEDDING_SIZE * sizeof(float));
        free(temp_wpe[i]);
    }
    free(temp_wpe);

    // Initialize blocks
    weights.blocks = (BlockWeights *)aligned_malloc(NUM_BLOCKS * sizeof(BlockWeights));
    for (int b = 0; b < NUM_BLOCKS; b++) {
        // Initialize Q, K, V linear layers
        initialize_linear_layer(&weights.blocks[b].q_mlp, EMBEDDING_SIZE, EMBEDDING_SIZE);
        initialize_linear_layer(&weights.blocks[b].k_mlp, EMBEDDING_SIZE, EMBEDDING_SIZE);
        initialize_linear_layer(&weights.blocks[b].v_mlp, EMBEDDING_SIZE, EMBEDDING_SIZE);

        // Initialize MLP layers
        int mlpHiddenSize = EMBEDDING_SIZE * 4;
        initialize_linear_layer(&weights.blocks[b].first_block_MLP, EMBEDDING_SIZE, mlpHiddenSize);
        initialize_linear_layer(&weights.blocks[b].second_block_MLP, mlpHiddenSize, EMBEDDING_SIZE);
    }

    // Initialize logits layer
    initialize_linear_layer(&weights.logits_mlp, EMBEDDING_SIZE, VOCAB_SIZE);

    printf("GPT-2 Weights initialization complete.\n");
    return weights;
}


void free_linear_layer(LinearLayer *layer) {
    aligned_free_2d(layer->weights, layer->fcOutputSize);
    aligned_free(layer->biases);
}

void free_weights(GPT2Weights *weights) {
    // Free token embeddings
    aligned_free_2d(weights->wte, VOCAB_SIZE);
    
    // Free positional embeddings
    aligned_free_2d(weights->wpe, MAX_POSITION_EMBEDDINGS);
    
    // Free transformer blocks
    for (int b = 0; b < NUM_BLOCKS; b++) {
        free_linear_layer(&weights->blocks[b].q_mlp);
        free_linear_layer(&weights->blocks[b].k_mlp);
        free_linear_layer(&weights->blocks[b].v_mlp);
        free_linear_layer(&weights->blocks[b].first_block_MLP);
        free_linear_layer(&weights->blocks[b].second_block_MLP);
    }
    aligned_free(weights->blocks);
    
    // Free logits_mlp
    free_linear_layer(&weights->logits_mlp);
}


void check_alignment() {
    if (!check_avx_support()) {
        fprintf(stderr, "Error: AVX instructions not supported on this CPU\n");
        exit(1);
    }
}

// Add alignment check helper
static inline int is_aligned(const void *ptr, size_t alignment) {
    return ((uintptr_t)ptr & (alignment - 1)) == 0;
}

// Add to memory-critical functions
static void verify_alignment(float **matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        if (!is_aligned(matrix[i], 32)) {
            fprintf(stderr, "Error: Memory not properly aligned for AVX\n");
            exit(1);
        }
    }
}

// Add these validation functions
static void validate_linear_layer(LinearLayer *layer, const char *name) {
    if (!layer->weights) {
        printf("ERROR: Null weights in %s\n", name);
        exit(1);
    }
    if (!layer->biases) {
        printf("ERROR: Null biases in %s\n", name);
        exit(1);
    }
    for (int i = 0; i < layer->fcOutputSize; i++) {
        if (!layer->weights[i]) {
            printf("ERROR: Null weights row %d in %s\n", i, name);
            exit(1);
        }
    }
}

static void validate_weights(GPT2Weights *weights) {
    
    if (!weights->wte) {
        printf("ERROR: Null token embeddings\n");
        exit(1);
    }
    if (!weights->wpe) {
        printf("ERROR: Null position embeddings\n");
        exit(1);
    }
    if (!weights->blocks) {
        printf("ERROR: Null blocks\n");
        exit(1);
    }
    
    for (int i = 0; i < VOCAB_SIZE; i++) {
        if (!weights->wte[i]) {
            printf("ERROR: Null token embedding row %d\n", i);
            exit(1);
        }
    }
    
    for (int i = 0; i < MAX_POSITION_EMBEDDINGS; i++) {
        if (!weights->wpe[i]) {
            printf("ERROR: Null position embedding row %d\n", i);
            exit(1);
        }
    }
    
    for (int b = 0; b < NUM_BLOCKS; b++) {
        char name[100];
        sprintf(name, "block %d q_mlp", b);
        validate_linear_layer(&weights->blocks[b].q_mlp, name);
        sprintf(name, "block %d k_mlp", b);
        validate_linear_layer(&weights->blocks[b].k_mlp, name);
        sprintf(name, "block %d v_mlp", b);
        validate_linear_layer(&weights->blocks[b].v_mlp, name);
        sprintf(name, "block %d first_block_MLP", b);
        validate_linear_layer(&weights->blocks[b].first_block_MLP, name);
        sprintf(name, "block %d second_block_MLP", b);
        validate_linear_layer(&weights->blocks[b].second_block_MLP, name);
    }
    
    validate_linear_layer(&weights->logits_mlp, "logits_mlp");
}


// Test case
int main() {
    // Add checks
    check_alignment();
    if (!check_avx_support()) {
        fprintf(stderr, "Error: AVX instructions not supported on this CPU\n");
        return 1;
    }

    // Time the execution
    clock_t start = clock();

    // Seed the random number generator
    srand(42);

    // Define sequence length and tokens
    int seqLength = 5;
    int tokens[] = { 10, 20, 30, 40, 50 }; // Example token IDs

    GPT2Weights weights = initialize_weights();
    // validate_weights(&weights);

    // Run the model
    float *logits = model(tokens, seqLength, weights);

    // Find the token with the highest logit value
    int max_index = 0;
    float max_value = logits[0];
    for (int i = 1; i < VOCAB_SIZE; i++) {
        if (logits[i] > max_value) {
            max_value = logits[i];
            max_index = i;
        }
    }

    // It should be 26146
    printf("Predicted next token ID: %d\n", max_index);

    aligned_free(logits); 
    free_weights(&weights);

    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Execution time: %f seconds\n", time_spent);

    return 0;
}