#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../kernel/functional.h"


#define REPEAT 5000

void test_relu(int inputSize) {
    // Allocate and initialize input array
    float *input = (float *)malloc(inputSize * sizeof(float));
    for (int i = 0; i < inputSize; i++) {
        input[i] = (float)(rand() % 201 - 100); // Random values from -100 to 100
    }

    // Profiling loop
    for (int i = 0; i < REPEAT; i++) {
        // Create a copy of the input array to avoid modifying the original
        float *input_copy = (float *)malloc(inputSize * sizeof(float));
        for (int j = 0; j < inputSize; j++) {
            input_copy[j] = input[j];
        }

        applyRelu(input_copy, inputSize);
        free(input_copy);
    }

    // Cleanup
    free(input);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <input_size>\n", argv[0]);
        return 1;
    }

    int inputSize = atoi(argv[1]);

    srand(time(NULL));
    printf("Testing ReLU function with input size: %d\n", inputSize);
    test_relu(inputSize);
    return 0;
}
