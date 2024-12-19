#ifndef ATTENTION_H
#define ATTENTION_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "functional.h"
#include "matrix_ops.h"

// Function declarations
float **scaled_dot_product_attention(float **Q, float **K, float **V, int seqLength, int depth);

#endif