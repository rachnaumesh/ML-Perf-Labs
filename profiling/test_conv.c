#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../kernel/conv.h"

#define REPEAT 100


void test_convolution(int numChannels, int numFilters, int inputSize, int kernelSize) {
    // Allocate and initialize the input image
    float ***image = (float ***)malloc(numChannels * sizeof(float **));
    for (int l = 0; l < numChannels; l++) {
        image[l] = (float **)malloc(inputSize * sizeof(float *));
        for (int j = 0; j < inputSize; j++) {
            image[l][j] = (float *)malloc(inputSize * sizeof(float));
            for (int k = 0; k < inputSize; k++) {
                image[l][j][k] = (float)(rand() % 256); // Random values from 0 to 255
            }
        }
    }

    // Allocate and initialize the kernel
    float ****kernel = (float ****)malloc(numFilters * sizeof(float ***));
    for (int i = 0; i < numFilters; i++) {
        kernel[i] = (float ***)malloc(numChannels * sizeof(float **));
        for (int l = 0; l < numChannels; l++) {
            kernel[i][l] = (float **)malloc(kernelSize * sizeof(float *));
            for (int m = 0; m < kernelSize; m++) {
                kernel[i][l][m] = (float *)malloc(kernelSize * sizeof(float));
                for (int n = 0; n < kernelSize; n++) {
                    kernel[i][l][m][n] = (float)(rand() % 10); // Random values from 0 to 9
                }
            }
        }
    }

    // Allocate and initialize bias data
    float *biasData = (float *)malloc(numFilters * sizeof(float));
    for (int i = 0; i < numFilters; i++) {
        biasData[i] = (float)(rand() % 21 - 10); // Random values from -10 to 10
    }

    // Profiling loop
    for (int i = 0; i < REPEAT; i++) {
        float ***output = convolution(image, numChannels, kernel, biasData, numFilters, inputSize, kernelSize);

        // Cleanup output after usage
        for (int f = 0; f < numFilters; f++) {
            for (int j = 0; j < inputSize - kernelSize + 1; j++) {
                free(output[f][j]);
            }
            free(output[f]);
        }
        free(output);
    }

    // Cleanup
    for (int l = 0; l < numChannels; l++) {
        for (int j = 0; j < inputSize; j++) {
            free(image[l][j]);
        }
        free(image[l]);
    }
    free(image);

    for (int i = 0; i < numFilters; i++) {
        for (int l = 0; l < numChannels; l++) {
            for (int m = 0; m < kernelSize; m++) {
                free(kernel[i][l][m]);
            }
            free(kernel[i][l]);
        }
        free(kernel[i]);
    }
    free(kernel);

    free(biasData);
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        printf("Usage: %s <num_channels> <num_filters> <input_size> <kernel_size>\n", argv[0]);
        return 1;
    }

    int numChannels = atoi(argv[1]);
    int numFilters = atoi(argv[2]);
    int inputSize = atoi(argv[3]);
    int kernelSize = atoi(argv[4]);

    srand(time(NULL));
    printf("Testing convolution with %d channels, %d filters, input size: %d, kernel size: %d\n", numChannels, numFilters, inputSize, kernelSize);
    test_convolution(numChannels, numFilters, inputSize, kernelSize);
    return 0;
}
