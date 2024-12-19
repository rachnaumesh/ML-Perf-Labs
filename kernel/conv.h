#ifndef CONV_H
#define CONV_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "functional.h"
#include "matrix_ops.h"

float ***convolution(float ***image, int numChannels, float ****kernel, float *biasData, int numFilters, int inputSize, int kernelSize);

#endif