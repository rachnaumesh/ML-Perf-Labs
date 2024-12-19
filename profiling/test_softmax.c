#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../kernel/kernel.h"

#define REPEAT 5000

void test_softmax(int size) {
    float *input = (float *)malloc(size * sizeof(float));
    
    // Initialize with random data
    for (int i = 0; i < size; i++) {
        input[i] = (float)rand() / RAND_MAX;
    }

    // Profiling loop
    for (int i = 0; i < REPEAT; i++) {
        float *output = softmax(input, size);
        free(output);
    }

    // Cleanup
    free(input);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <input_size>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);

    srand(time(NULL));
    printf("Testing softmax with input size: %d\n", size);
    test_softmax(size);
    return 0;
}