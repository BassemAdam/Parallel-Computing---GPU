#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

// 1D convolution kernel with input tiling
__global__ void convolution1D_InputTiling(float *input, float *mask, float *output, 
    int inputLength, int maskLength, int outputTileSize) {
    extern __shared__ float sharedMem[];

    int tx = threadIdx.x;
    int maskRadius = maskLength / 2;
    
    // Calculate global input and output positions
    int outputIdx = blockIdx.x * outputTileSize + tx;
    int inputIdx = outputIdx - maskRadius;
    
    // Load input elements into shared memory (with boundary handling)
    //Each thread loads exactly one input element:
    if (inputIdx >= 0 && inputIdx < inputLength) {
        sharedMem[tx] = input[inputIdx];
    } else {
        sharedMem[tx] = 0.0f;  
    }
    
    __syncthreads();
    
    // Only threads with valid output positions compute results
    // Not all threads calculate output elements:
    if (tx < outputTileSize && outputIdx < inputLength) {
        float result = 0.0f;
        
        // Perform the convolution
        for (int j = 0; j < maskLength; j++) {
            int sharedIdx = tx + j;
            // Make sure we don't access out of bounds in shared memory
            if (sharedIdx < blockDim.x) {
                result += sharedMem[sharedIdx] * mask[j];
            }
        }
        
        // Write result to global memory
        output[outputIdx] = result;
    }
}
// Function to read input vector from file
void readInputFile(const char *filename, float **data, int *length) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening input file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    fscanf(file, "%d", length);
    *data = (float *)malloc(*length * sizeof(float));
    
    for (int i = 0; i < *length; i++) {
        fscanf(file, "%f", &(*data)[i]);
    }
    
    fclose(file);
}

// Function to read mask from file
void readMaskFile(const char *filename, float **data, int *length) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening mask file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    fscanf(file, "%d", length);
    *data = (float *)malloc(*length * sizeof(float));
    
    for (int i = 0; i < *length; i++) {
        fscanf(file, "%f", &(*data)[i]);
    }
    
    fclose(file);
}

// Function to write output data to file
void writeOutputFile(const char *filename, float *data, int length) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error opening output file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < length; i++) {
        fprintf(file, "%.3f", data[i]);
        if (i < length - 1) {
            fprintf(file, " ");
        }
    }
    fprintf(file, "\n");
    
    fclose(file);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <inputfile> <maskfile> <outputfile>\n", argv[0]);
        return -1;
    }

    const char *inputFile = argv[1];
    const char *maskFile = argv[2];
    const char *outputFile = argv[3];

    // Load input vector from file
    float *h_input;
    int inputLength;
    readInputFile(inputFile, &h_input, &inputLength);

    // Load mask from file
    float *h_mask;
    int maskLength;
    readMaskFile(maskFile, &h_mask, &maskLength);

    // Allocate device memory
    float *d_input, *d_mask, *d_output;
    cudaMalloc(&d_input, inputLength * sizeof(float));
    cudaMalloc(&d_mask, maskLength * sizeof(float));
    cudaMalloc(&d_output, inputLength * sizeof(float));

    // Copy input data and mask to device
    cudaMemcpy(d_input, h_input, inputLength * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, maskLength * sizeof(float), cudaMemcpyHostToDevice);


    //int blockSize = outputTileSize + maskLength - 1;
    int blockSize = 256;
    int outputTileSize = blockSize - maskLength + 1;
    int gridSize = (inputLength + outputTileSize - 1) / outputTileSize;
    
    // Calculate shared memory size
    int sharedMemSize = blockSize * sizeof(float);

    // Launch convolution kernel with input tiling
    convolution1D_InputTiling<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_mask, d_output, 
                                                                     inputLength, maskLength,outputTileSize);
    
  
    // Copy result back to host
    float *h_output = (float *)malloc(inputLength * sizeof(float));
    cudaMemcpy(h_output, d_output, inputLength * sizeof(float), cudaMemcpyDeviceToHost);

    // Save result to output file
    writeOutputFile(outputFile, h_output, inputLength);

    // Free memory
    cudaFree(d_input);
    cudaFree(d_mask);
    cudaFree(d_output);
    free(h_input);
    free(h_mask);
    free(h_output);

    return 0;
}