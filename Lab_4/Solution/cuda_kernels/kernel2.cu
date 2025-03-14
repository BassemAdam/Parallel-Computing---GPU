#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

// Define constant memory for the mask
__constant__ float c_mask[1024]; // Assuming mask won't be larger than 1024 elements

// 1D convolution kernel with output tiling where block size = tile size
__global__ void convolution1D_OutputTiling(float *input, float *output, 
                                         int inputLength, int maskLength, int tileSize) {
    int maskRadius = maskLength / 2;
    int threadId = threadIdx.x;
    int tileOffset = blockIdx.x * tileSize;
    
    // Shared memory for input tile + halo regions
    extern __shared__ float s_data[];
    
    // Each thread helps load the shared memory - both tile and halo elements
    // We need to load tileSize + maskLength - 1 elements in total
    // Some threads load more than one input element
    for (int i = threadId; i < tileSize + maskLength - 1; i += blockDim.x) {
        int globalIdx = tileOffset + i - maskRadius;
        
        // Apply boundary check with zero padding for halo regions
        if (globalIdx >= 0 && globalIdx < inputLength) {
            s_data[i] = input[globalIdx];
        } else {
            s_data[i] = 0.0f;
        }
    }
    
    // Make sure all threads finished loading shared memory
    __syncthreads();
    
    // Each thread computes one output element within the tile
    int outputIdx = tileOffset + threadId;
    
    if (threadId < tileSize && outputIdx < inputLength) {
        float result = 0.0f;
        
        // The position in shared memory for the current element
        // threadId is the position in the tile, maskRadius is the halo offset
        int s_idx = threadId + maskRadius;
        
        // Apply the convolution mask
        for (int j = 0; j < maskLength; j++) {
            result += s_data[s_idx - maskRadius + j] * c_mask[j];
        }
        
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
    float *d_input, *d_output;
    cudaMalloc(&d_input, inputLength * sizeof(float));
    cudaMalloc(&d_output, inputLength * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, inputLength * sizeof(float), cudaMemcpyHostToDevice);
    
    // Copy mask to constant memory ;since it does not change 
    // tried to use __restrict but it didnt work so i searched and i find this method
    cudaMemcpyToSymbol(c_mask, h_mask, maskLength * sizeof(float));

    // Define tiling parameters
    int tileSize = 256;  // Size of each output tile (matches the block size)
    int blockSize = tileSize;  // Block size equals tile size in this design
    
    // Calculate grid size (number of tiles needed)
    int gridSize = (inputLength + tileSize - 1) / tileSize;
    
    // Calculate shared memory size - need space for tile + halo regions
    int sharedMemSize = (tileSize + maskLength - 1) * sizeof(float);
    
    // Launch convolution kernel with output tiling
    convolution1D_OutputTiling<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, 
                                                                      inputLength, maskLength, tileSize);
    
    // Copy result back to host
    float *h_output = (float *)malloc(inputLength * sizeof(float));
    cudaMemcpy(h_output, d_output, inputLength * sizeof(float), cudaMemcpyDeviceToHost);

    // Save result to output file
    writeOutputFile(outputFile, h_output, inputLength);

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_mask);
    free(h_output);

    return 0;
}