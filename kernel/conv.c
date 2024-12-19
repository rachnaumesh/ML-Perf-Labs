#include "conv.h"
#include "matrix_ops.h"
#include "functional.h"

// Im2col algorithm
float **im2col(float ***image, int numChannels, int imageSize, int kernelSize, int stride, int *outputSize)
{
    int output_rows = (imageSize - kernelSize) / stride + 1;
    int output_cols = (imageSize - kernelSize) / stride + 1;
    *outputSize = output_rows * output_cols;

    int col_width = numChannels * kernelSize * kernelSize;
    float **result = (float **)malloc(col_width * sizeof(float *));
    for (int i = 0; i < col_width; i++)
    {
        result[i] = (float *)malloc(output_rows * output_cols * sizeof(float));
    }

    for (int c = 0; c < numChannels; c++)
    {
        for (int i = 0; i < output_rows; i++)
        {
            for (int j = 0; j < output_cols; j++)
            {
                for (int k = 0; k < kernelSize; k++)
                {
                    for (int l = 0; l < kernelSize; l++)
                    {
                        int row_index = c * kernelSize * kernelSize + k * kernelSize + l;
                        result[row_index][i * output_cols + j] = image[c][i * stride + k][j * stride + l];
                    }
                }
            }
        }
    }

    return result;
}


// Im2col algorithm's inverse
float ***col2im(float **result, int num_kernels, int conv_rows, int conv_cols)
{
    // TODO: Implement the col2im algorithm
    float ***output = (float ***)malloc(num_kernels * sizeof(float **));
    for (int i = 0; i < num_kernels; i++) {
        output[i] = (float **)malloc(conv_rows * sizeof(float *));
        for (int j = 0; j < conv_rows; j++) {
            output[i][j] = (float *)malloc(conv_cols * sizeof(float));
        }
    }
    for (int filter = 0; filter < num_kernels; filter++) {
        for (int i = 0; i < conv_rows; i++) {
            for (int j = 0; j < conv_cols; j++) {
                output[filter][i][j] = result[filter][i * conv_cols + j];
            }
        }
    }
    return output;
    
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

float **kernel_flatten_multiple_channels(float ****kernel, int num_kernels, int num_channels, int kernel_size)
{
    int flattened_size = kernel_size * kernel_size * num_channels;
    float **flattened_kernels = (float **)malloc(num_kernels * sizeof(float *));
    for (int i = 0; i < num_kernels; i++)
    {
        flattened_kernels[i] = (float *)malloc(flattened_size * sizeof(float));
    }

    for (int k = 0; k < num_kernels; k++)
    {
        for (int c = 0; c < num_channels; c++)
        {
            for (int i = 0; i < kernel_size; i++)
            {
                for (int j = 0; j < kernel_size; j++)
                {
                    int index = c * kernel_size * kernel_size + i * kernel_size + j;
                    flattened_kernels[k][index] = kernel[k][c][i][j];
                }
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
    float **flattened_kernels = kernel_flatten_multiple_channels(kernel, numFilters, numChannels, kernelSize);
    // float **flattened_kernels = kernel_flatten(kernel, numFilters, kernelSize);

    // Apply im2col
    int outputSize;
    float **im2col_result = im2col(image, numChannels, inputSize, kernelSize, 1, &outputSize);

    // Apply matmul
    float **matmul_result;
    if (matmul_type == MATMUL_BASE)
    {
        matmul_result = matmul(flattened_kernels, im2col_result, numFilters, numChannels * kernelSize * kernelSize, numChannels * kernelSize * kernelSize, outputSize);
    }
    else if (matmul_type == MATMUL_SPARSE)
    {
        matmul_result = matmul_sparse(flattened_kernels, im2col_result, numFilters, numChannels * kernelSize * kernelSize, numChannels * kernelSize * kernelSize, outputSize);
    }

    // Apply col2im
    float ***output = col2im(matmul_result, numFilters, inputSize - kernelSize + 1, inputSize - kernelSize + 1);

    // Add bias and apply ReLU
    for (int i = 0; i < numFilters; i++)
    {
        for (int j = 0; j < inputSize - kernelSize + 1; j++)
        {
            for (int k = 0; k < inputSize - kernelSize + 1; k++)
            {
                output[i][j][k] += biasData[i];
                output[i][j][k] = relu(output[i][j][k]);
            }
        }
    }

    // Cleanup
    for (int i = 0; i < numFilters; i++)
    {
        free(flattened_kernels[i]);
    }
    free(flattened_kernels);
    for (int i = 0; i < numFilters; i++)
    {
        free(matmul_result[i]);
    }
    free(matmul_result);
    for (int i = 0; i < numChannels * kernelSize * kernelSize; i++)
    {
        free(im2col_result[i]);
    }
    free(im2col_result);

    return output;
}