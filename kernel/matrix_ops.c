#include "matrix_ops.h"
#include <pthread.h>
#include <stdlib.h>

int num_threads = 8; // Default number of threads

typedef struct {
    float **A;
    float **B;
    float **C;
    int A_rows;
    int A_cols;
    int B_cols;
    int start_row;
    int end_row;
} thread_data_t;


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



void *matmul_thread_func(void *arg) {
    thread_data_t *data = (thread_data_t *)arg;
    float **A = data->A;
    float **B = data->B;
    float **C = data->C;
    int A_rows = data->A_rows;
    int A_cols = data->A_cols;
    int B_cols = data->B_cols;
    int start_row = data->start_row;
    int end_row = data->end_row;

    // Print thread ID and the range of rows it is processing
    // printf("Thread %ld processing rows %d to %d\n", pthread_self(), start_row, end_row);

    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < B_cols; j++) {
            C[i][j] = 0;
            for (int k = 0; k < A_cols; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return NULL;
}


float **matmul_thread(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols)
{
    // TODO: Implement multithreaded matrix multiplication
    if (A_cols != B_rows) {
        return NULL;
    }

    float **C = (float **)malloc(A_rows * sizeof(float *));
    for (int i = 0; i < A_rows; i++) {
        C[i] = (float *)malloc(B_cols * sizeof(float));
    }

    pthread_t threads[num_threads];
    thread_data_t thread_data[num_threads];

    int rows_per_thread = A_rows / num_threads;
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].A = A;
        thread_data[i].B = B;
        thread_data[i].C = C;
        thread_data[i].A_rows = A_rows;
        thread_data[i].A_cols = A_cols;
        thread_data[i].B_cols = B_cols;
        thread_data[i].start_row = i * rows_per_thread;
        thread_data[i].end_row = (i == num_threads - 1) ? A_rows : (i + 1) * rows_per_thread;

        pthread_create(&threads[i], NULL, matmul_thread_func, &thread_data[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    return C;
}

