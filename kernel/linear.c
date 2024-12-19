#include "linear.h"

float *linear(float *input, float **weights, float *biases, int inputSize, int outputSize)
{
    // TODO: Implement the linear function (fully connected layer)
    float *output = (float *)malloc(outputSize * sizeof(float));

    for (int i = 0; i < outputSize; i++)
    {
        output[i] = 0;
        for (int j = 0; j < inputSize; j++)
        {
            output[i] += input[j] * weights[i][j];
        }
        output[i] += biases[i];
    }
    return output;
}
