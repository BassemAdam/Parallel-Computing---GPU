#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

// Basic 1D convolution kernel without tiling
__global__ void convolution1D(float *input, float *mask, float *output, int inputLength, int maskLength) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < inputLength) {
        float result = 0.0f;
        int maskRadius = maskLength / 2;
        
        for (int j = 0; j < maskLength; j++) {
            int inputIdx = idx - maskRadius + j;
            
            // Handle boundary conditions - zero padding
            if (inputIdx >= 0 && inputIdx < inputLength) {
                result += input[inputIdx] * mask[j];
            }
        }
        
        output[idx] = result;
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

    // Define block and grid sizes
    int blockSize = 256;
    int gridSize = (inputLength + blockSize - 1) / blockSize;

    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();

    // Launch convolution kernel
    convolution1D<<<gridSize, blockSize>>>(d_input, d_mask, d_output, inputLength, maskLength);
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // Copy result back to host
    float *h_output = (float *)malloc(inputLength * sizeof(float));
    cudaMemcpy(h_output, d_output, inputLength * sizeof(float), cudaMemcpyDeviceToHost);

    // Save result to output file
    writeOutputFile(outputFile, h_output, inputLength);

    printf("Kernel execution time: %.6f ms\n", duration / 1000.0);

    // Free memory
    cudaFree(d_input);
    cudaFree(d_mask);
    cudaFree(d_output);
    free(h_input);
    free(h_mask);
    free(h_output);

    return 0;
}