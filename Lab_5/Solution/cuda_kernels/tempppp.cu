#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

// Basic 1D convolution kernel without tiling
__global__ void convolution1D(float *input, float *mask, float *output, int inputLength, int maskLength) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

}


// normal convo
void convolution1D(float *input, float *mask, float *output, int inputLength, int maskLength) {
    
    for (size_t element_idx = 0; element_idx < inputLength; element_idx++)
    {
        float result = 0.0f;
        int maskCentered=mask[maskLength/2];
        
        //case where mask is out of bounds left
        if (element_idx < maskLength/2)
        {
            for (size_t i = 0; i < maskLength/2; i++)
            {
                result += mask[maskLength/2+ i]*input[i];
            }
        }

        //case where mask is out of bounds right
        else if (inputLength-element_idx <= maskLength/2)
        {
            for (size_t i = 0; i < maskLength/2; i++)
            {
                result += mask[i]*input[i];
            }
        }

        else
        {
            for (size_t i = 0; i < maskLength; i++)
            {
                result += mask[i]*input[i];
            }
        }
        output[element_idx]=result;
        
    }
    

}