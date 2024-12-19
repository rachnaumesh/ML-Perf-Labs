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
    MATMUL_BLOCKING,
} MatmulType;

float ***convolution(float ***image, int numChannels, float ****kernel, float *biasData, int numFilters, int inputSize, int kernelSize);
float ***convolution_im2col(float ***image, int numChannels, float ****kernel, float *biasData, int numFilters, int inputSize, int kernelSize, MatmulType matmul_type);

#endif