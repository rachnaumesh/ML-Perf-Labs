#ifndef LINEAR_H
#define LINEAR_H

#include <float.h>
#include <stdlib.h>

float *linear(float *fcInput, float **weights, float *biases, int fcInputSize, int fcOutputSize);

#endif // LINEAR_H