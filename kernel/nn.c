#include "nn.h"

float *flatten(float ***input, int inputSize, int depth)
{
    // TODO: Implement the flatten function
    float *output = (float *)malloc(inputSize * inputSize * depth * sizeof(float));
    int index = 0;
    for (int i = 0; i < depth; i++)
    {
        for (int j = 0; j < inputSize; j++)
        {
            for (int k = 0; k < inputSize; k++)
            {
                output[index] = input[i][j][k];
                index++;
            }
        }
    }
    return output;
}

void destroyConvOutput(float ***convOutput, int convOutputSize)
{
    for (int i = 0; i < 32; i++)
    {
        for (int j = 0; j < convOutputSize; j++)
        {
            free(convOutput[i][j]);
        }
        free(convOutput[i]);
    }
    free(convOutput);
}

int forwardPass(float ***image, int numChannels, float ****conv1WeightsData, float **fc1WeightsData, float **fc2WeightsData, float *conv1BiasData, float *fc1BiasData, float *fc2BiasData)
{
    // 1. Perform the convolution operation

    // 2. Flatten the output

    // 3. Perform the fully connected operations

    // 4. Apply the final softmax activation

    // 5. Make predictions

    // Clean up the memory usage
}

int predict(float *probabilityVector, int numClasses)
{
    // TODO: Implement the prediction function
    int maxIndex = 0;
    float max = probabilityVector[0];
    for (int i = 1; i < numClasses; i++)
    {
        if (probabilityVector[i] > max)
        {
            max = probabilityVector[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}
