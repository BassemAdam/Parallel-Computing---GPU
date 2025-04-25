/* Thread Coarsened PrefixSum */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

// Global counter for block execution order
__device__ int blockCounter;
// Global flags
__device__ unsigned int *flags;
__device__ const int BLOCK_SIZE = 256;
// Thread coarsening factor - each thread processes this many elements
#define COARSENING_FACTOR 4

__global__ void CoarsenedPrefixSumUlt(float *input, float *output, int length)
{
    // Shared memory for scan operation
    __shared__ float s_data[BLOCK_SIZE];
    
    // Block ID tracking
    __shared__ int s_blk_id;
    if (threadIdx.x == 0)
        s_blk_id = atomicAdd(&blockCounter, 1);
    __syncthreads();
    
    // Each thread now handles COARSENING_FACTOR elements
    int base_idx = (threadIdx.x + blockDim.x * s_blk_id) * COARSENING_FACTOR;
    
    // Step 1: Each thread computes local sum of its COARSENING_FACTOR elements
    float thread_sum = 0;
    float local_vals[COARSENING_FACTOR];
    
    // Load and sum the coarsened elements
    for (int i = 0; i < COARSENING_FACTOR; i++) {
        int global_idx = base_idx + i;
        if (global_idx < length) {
            local_vals[i] = input[global_idx];
            thread_sum += local_vals[i];
        } else {
            local_vals[i] = 0;
        }
    }
    
    // Load local sum into shared memory for block scan
    s_data[threadIdx.x] = thread_sum;
    __syncthreads();
    
    // Up-sweep (Reduction) phase
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index < blockDim.x) {
            s_data[index] += s_data[index - stride];
        }
        __syncthreads();
    }
    
    //setting the last element to 0
    if (threadIdx.x == 0) {
        s_data[blockDim.x - 1] = 0;
    }
    __syncthreads();
    
    // Down-sweep (Distribution) phase - exclusive scan
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index < blockDim.x) {
            float temp = s_data[index];
            s_data[index] += s_data[index - stride];
            s_data[index - stride] = temp;
        }
        __syncthreads();
    }
    
    // Get the exclusive prefix for this thread from the block scan
    float thread_prefix = s_data[threadIdx.x];
    
    // Handle cross-block dependencies
    __shared__ float s_previous_sum;
    if (threadIdx.x == 0) {
        // Wait for previous block to complete
        while (s_blk_id != 0 && atomicAdd(&flags[s_blk_id - 1], 0) == 0) {}
        
        s_previous_sum = s_blk_id > 0 ? output[(s_blk_id * blockDim.x * COARSENING_FACTOR) - 1] : 0;
    }
    __syncthreads();
    
    // Now calculate prefix sum for each element this thread handles
    float running_sum = thread_prefix + s_previous_sum;
    
    // Process first element separately (the prefix excludes this element)
    if (base_idx < length) {
        output[base_idx] = running_sum + local_vals[0];
        running_sum = output[base_idx];
    }
    
    // Process remaining elements (inclusive scan within thread)
    for (int i = 1; i < COARSENING_FACTOR; i++) {
        int global_idx = base_idx + i;
        if (global_idx < length) {
            running_sum += local_vals[i];
            output[global_idx] = running_sum;
        }
    }
    
    // Signal completion - last thread updates the flag
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
    // Adjust grid size for coarsening - each thread handles COARSENING_FACTOR elements
    int gridSize = (inputLength + (blockSize * COARSENING_FACTOR) - 1) / (blockSize * COARSENING_FACTOR);
    
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

    // Shared memory allocation (now just BLOCK_SIZE since we use local array)
    size_t sharedMemSize = BLOCK_SIZE * sizeof(float);

    // Time measurement
    auto start = std::chrono::high_resolution_clock::now();

    // Launch coarsened prefix sum kernel
    CoarsenedPrefixSumUlt<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, inputLength);
    
    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Thread coarsened execution time (factor=%d): %lld ms\n", COARSENING_FACTOR, duration);

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