#include "conv.h"
#include "functional.h"

// Basic convolution operation
float ***convolution(float ***image, int numChannels, float ****kernel, float *biasData, int numFilters, int inputSize, int kernelSize)
{
    // TODO: Implement the convolution operation
    int outputSize = inputSize - kernelSize + 1;

    float ***output = (float ***)malloc(numFilters * sizeof(float **));
    for (int i = 0; i < numFilters; i++)
    {
        output[i] = (float **)malloc(outputSize * sizeof(float *));
        for (int j = 0; j < outputSize; j++)
        {
            output[i][j] = (float *)malloc(outputSize * sizeof(float));
        }
    }

    for (int i = 0; i < numFilters; i++)
    {
        for (int j = 0; j < outputSize; j++)
        {
            for (int k = 0; k < outputSize; k++)
            {
                output[i][j][k] = 0;
                for (int l = 0; l < numChannels; l++)
                {
                    for (int m = 0; m < kernelSize; m++)
                    {
                        for (int n = 0; n < kernelSize; n++)
                        {
                            output[i][j][k] += image[l][j + m][k + n] * kernel[i][l][m][n];
                        }
                    }
                }
                output[i][j][k] += biasData[i];

                // Apply ReLU
                output[i][j][k] = relu(output[i][j][k]);
            }
        }
    }
    return output;
}


