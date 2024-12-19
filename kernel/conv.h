#ifndef CONV_H
#define CONV_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "functional.h"
#include "matrix_ops.h"

typedef enum
{
    MATMUL_BASE,
    MATMUL_SPARSE
} MatmulType;

float ***convolution(float ***image, int numChannels, float ****kernel, float *biasData, int numFilters, int inputSize, int kernelSize);
float ***convolution_im2col(float ***image, int numChannels, float ****kernel, float *biasData, int numFilters, int inputSize, int kernelSize, MatmulType matmul_type);
float **im2col(float ***image, int numChannels, int imageSize, int kernelSize, int stride, int *outputSize);
float ***col2im(float **result, int num_kernels, int conv_rows, int conv_cols);
float **kernel_flatten(float ****kernel, int num_kernels, int kernel_size);
float **kernel_flatten_multiple_channels(float ****kernel, int num_kernels, int num_channels, int kernel_size);


#endif