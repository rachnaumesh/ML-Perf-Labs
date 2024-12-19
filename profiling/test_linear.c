#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../kernel/kernel.h"

#define REPEAT 100

void test_linear(int inputSize, int outputSize) {
    // Allocate input
    float *input = (float *)malloc(inputSize * sizeof(float));
    // Initialize input with random positive values
    for (int i = 0; i < inputSize; i++) {
        input[i] = (float)(rand() % 10);  // Random values from 0 to 9
    }

    // Allocate weights
    float **weights = (float **)malloc(outputSize * sizeof(float *));
    for (int i = 0; i < outputSize; i++) {
        weights[i] = (float *)malloc(inputSize * sizeof(float));
        for (int j = 0; j < inputSize; j++) {
            weights[i][j] = (float)(rand() % 10);  // Random values from 0 to 9
        }
    }

    // Allocate biases
    float *biases = (float *)malloc(outputSize * sizeof(float));
    for (int i = 0; i < outputSize; i++) {
        biases[i] = (float)(rand() % 10);  // Random values from 0 to 9
    }

    // Profiling loop
    for (int i = 0; i < REPEAT; i++) {
        float *output = linear(input, weights, biases, inputSize, outputSize);
        free(output);  // Free the output after use
    }

    // Cleanup
    free(input);
    for (int i = 0; i < outputSize; i++) {
        free(weights[i]);
    }
    free(weights);
    free(biases);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_size> <output_size>\n", argv[0]);
        return 1;
    }

    int inputSize = atoi(argv[1]);
    int outputSize = atoi(argv[2]);

    srand(time(NULL));
    printf("Testing linear with input size: %d and output size: %d\n", inputSize, outputSize);
    test_linear(inputSize, outputSize);
    return 0;
}
