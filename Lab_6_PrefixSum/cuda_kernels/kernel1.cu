/* Work Inefficient PrefixSum*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

// Global counter for block execution order
__device__ int blockCounter;
// Global flags
__device__ unsigned int *flags;
__device__ const int BLOCK_SIZE = 256;
__global__ void EfficientPrefixSumUlt(float *input, float *output, int length)
{
    // Shared memory for scan operation
    __shared__ float s_data[BLOCK_SIZE];
    
    // Block ID tracking
    __shared__ int s_blk_id;
    if (threadIdx.x == 0)
        s_blk_id = atomicAdd(&blockCounter, 1);
    __syncthreads();
    
    // Load input into shared memory
    int idx = threadIdx.x + blockDim.x * s_blk_id;
    s_data[threadIdx.x] = idx < length ? input[idx] : 0;
    __syncthreads();
    
    if (idx >= length)
        return;
        
    // Up-sweep (Reduction) phase
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index < blockDim.x) {
            s_data[index] += s_data[index - stride];
        }
        __syncthreads();
    }
    
    // Clear the last element
    if (threadIdx.x == 0) {
        s_data[blockDim.x - 1] = 0;
    }
    __syncthreads();
    
    // Down-sweep (Distribution) phase
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index < blockDim.x) {
            float temp = s_data[index];
            s_data[index] += s_data[index - stride];
            s_data[index - stride] = temp;
        }
        __syncthreads();
    }
    
    // Handle cross-block dependencies
    __shared__ float s_previous_sum;
    if (threadIdx.x == 0) {
        // Wait for previous block to complete
        while (s_blk_id != 0 && atomicAdd(&flags[s_blk_id - 1], 0) == 0) {}
        
        s_previous_sum = s_blk_id > 0 ? output[idx - 1] : 0;
    }
    __syncthreads();
    
    // Convert to inclusive sum by adding the original value
    if (idx < length)
        output[idx] =  s_data[threadIdx.x] + s_previous_sum + input[idx];
    
    // Signal completion
    if (threadIdx.x == blockDim.x - 1) {
        __threadfence();
        atomicAdd(&flags[s_blk_id], 1);
    }
}

// Function to read input vector from file
void readInputFile(const char *filename, float **data, int *length)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        fprintf(stderr, "Error opening input file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    fscanf(file, "%d", length);
    *data = (float *)malloc(*length * sizeof(float));

    for (int i = 0; i < *length; i++)
    {
        fscanf(file, "%f", &(*data)[i]);
    }

    fclose(file);
}

// Function to write output data to file
void writeOutputFile(const char *filename, float *data, int length)
{
    FILE *file = fopen(filename, "w");
    if (!file)
    {
        fprintf(stderr, "Error opening output file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < length; i++)
    {
        fprintf(file, "%.3f", data[i]);
        if (i < length - 1)
        {
            fprintf(file, " ");
        }
    }
    fprintf(file, "\n");

    fclose(file);
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("ERROR Usage: %s <inputfile> <outputfile>\n", argv[0]);
        return -1;
    }

    const char *inputFile = argv[1];
    const char *outputFile = argv[2];

    // Load input vector from file
    float *h_input;
    int inputLength;
    readInputFile(inputFile, &h_input, &inputLength);

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, inputLength * sizeof(float));
    cudaMalloc(&d_output, inputLength * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, inputLength * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = BLOCK_SIZE;
    int gridSize = (inputLength + blockSize - 1) / blockSize;
    // Reset block counter before kernel launch
    int zero = 0;
    cudaMemcpyToSymbol(blockCounter, &zero, sizeof(int));

    // Allocate and initialize flags
    unsigned int *d_flags;
    cudaMalloc(&d_flags, (gridSize + 1) * sizeof(unsigned int));
    cudaMemset(d_flags, 0, (gridSize + 1) * sizeof(unsigned int));
    // Set first flag to 1 so first block doesn't wait
    unsigned int one = 1;
    cudaMemcpy(d_flags, &one, sizeof(unsigned int), cudaMemcpyHostToDevice);
    // Copy pointer to device symbol
    cudaMemcpyToSymbol(flags, &d_flags, sizeof(unsigned int *));

    size_t sharedMemSize = inputLength > 256 ? blockSize * sizeof(float) : inputLength * sizeof(float);

    // Launch convolution kernel with output tiling
    EfficientPrefixSumUlt<<<gridSize, blockSize,sharedMemSize>>>(d_input, d_output, inputLength);

    // Copy result back to host
    float *h_output = (float *)malloc(inputLength * sizeof(float));
    cudaMemcpy(h_output, d_output, inputLength * sizeof(float), cudaMemcpyDeviceToHost);

    // Save result to output file
    writeOutputFile(outputFile, h_output, inputLength);

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_flags);
    free(h_input);
    free(h_output);

    return 0;
}