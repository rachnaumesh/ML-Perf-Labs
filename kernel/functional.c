#include "functional.h"

float relu(float x)
{
    // TODO: Implement relu
    return x > 0 ? x : 0;
}

void applyRelu(float *input, int inputSize)
{
    for (int i = 0; i < inputSize; i++)
    {
        input[i] = relu(input[i]);
    }
}

float *softmax_without_log(float *input, int inputSize)
{
    // Find maximum of input vector (for numerical stability)
    float *output = (float *)malloc(inputSize * sizeof(float));
    float maxInput = input[0];
    for (int i = 1; i < inputSize; i++) {
        if (input[i] > maxInput) {
            maxInput = input[i];
        }
    }

    // Compute exp of input - maxInput to avoid overflow
    float sum = 0;
    for (int i = 0; i < inputSize; i++) {
        output[i] = exp(input[i] - maxInput);
        sum += output[i];
    }

    // Normalize the values
    for (int i = 0; i < inputSize; i++) {
        output[i] /= sum;
    }

    return output;
}


float *softmax(float *input, int inputSize)
{
    // TODO: Implement softmax
    // Find maximum of input vector
    float *output = (float *)malloc(inputSize * sizeof(float));
    int maxInput = input[0];
    for (int i = 1; i < inputSize; i++)
    {
        if (input[i] > maxInput)
        {
            maxInput = input[i];
        }
    }

    // Compute exp of input - maxInput to avoid underflow
    float sum = 0;
    for (int i = 0; i < inputSize; i++)
    {
        output[i] = exp(input[i] - maxInput);
        sum += output[i];
    }

    // Normalise and apply log
    for (int i = 0; i < inputSize; i++)
    {
        output[i] = output[i] / sum;
    }

    for (int i = 0; i < inputSize; i++)
    {
        output[i] = log(output[i]);
    }

    return output;
}
