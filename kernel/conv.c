#include "conv.h"

// Im2col algorithm
float **im2col(float ***image, int numChannels, int imageSize, int kernelSize, int stride, int *outputSize)
{
    // TODO: Implement the im2col algorithm
}

// Im2col algorithm's inverse
float ***col2im(float **result, int num_kernels, int conv_rows, int conv_cols)
{
    // TODO: Implement the col2im algorithm
}

float **kernel_flatten(float ****kernel, int num_kernels, int kernel_size)
{
    float **flattened_kernels = (float **)malloc(num_kernels * sizeof(float *));
    for (int i = 0; i < num_kernels; i++)
    {
        flattened_kernels[i] = (float *)malloc(kernel_size * kernel_size * sizeof(float));
    }

    for (int k = 0; k < num_kernels; k++)
    {
        for (int i = 0; i < kernel_size; i++)
        {
            for (int j = 0; j < kernel_size; j++)
            {
                flattened_kernels[k][i * kernel_size + j] = kernel[k][0][i][j];
            }
        }
    }

    return flattened_kernels;
}

// Basic convolution operation
float ***convolution(float ***image, int numChannels, float ****kernel, float *biasData, int numFilters, int inputSize, int kernelSize)
{
    // TODO: Implement the basic convolution operation
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

// Convolution with im2col algorithm
float ***convolution_im2col(float ***image, int numChannels, float ****kernel, float *biasData, int numFilters, int inputSize, int kernelSize, MatmulType matmul_type)
{
    // TODO: Implement the convolution operation with im2col algorithm
    // Flatten kernel

    // Apply im2col

    // Apply matmul

    // Apply col2im

    // Add bias and apply ReLU

    // Cleanup
}