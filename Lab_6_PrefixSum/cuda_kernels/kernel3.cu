/* Memory Performance Comparison for Prefix Sum */
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
    float original_value = idx < length ? input[idx] : 0;
    s_data[threadIdx.x] = original_value;
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
    
    // Clear the last element for exclusive scan within block
    if (threadIdx.x == 0) s_data[blockDim.x - 1] = 0;
     
    
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
    
    // Handle cross-block dependencies
    __shared__ float s_previous_sum;
    if (threadIdx.x == 0) {
        // Wait for previous block to complete
        while (s_blk_id != 0 && atomicAdd(&flags[s_blk_id - 1], 0) == 0) {}
        
        s_previous_sum = s_blk_id > 0 ? output[blockDim.x * s_blk_id - 1] : 0;
    }
    __syncthreads();
    
    // Convert from exclusive to inclusive scan
    if (idx < length)
        output[idx] = s_data[threadIdx.x] + original_value + s_previous_sum;
    
    // Signal completion - use the last computed value for this block
    if (threadIdx.x == blockDim.x - 1) {
        __threadfence();
        atomicAdd(&flags[s_blk_id], 1);
    }
}

// Helper functions
void readInputFile(const char *filename, float **data, int *length)
{
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

void writeOutputFile(const char *filename, float *data, int length)
{
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

// Helper function to check CUDA errors
#define checkCudaErrors(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Function to run prefix sum with pageable memory
double runWithPageableMemory(float *h_input, float *h_output, int inputLength) {
    float *d_input, *d_output;
    
    // Start timer
    auto start = std::chrono::high_resolution_clock::now();
    
    // Allocate device memory
    checkCudaErrors(cudaMalloc(&d_input, inputLength * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_output, inputLength * sizeof(float)));

    // Copy input data to device
    checkCudaErrors(cudaMemcpy(d_input, h_input, inputLength * sizeof(float), cudaMemcpyHostToDevice));

    int blockSize = BLOCK_SIZE;
    int gridSize = (inputLength + blockSize - 1) / blockSize;
    
    // Reset block counter
    int zero = 0;
    checkCudaErrors(cudaMemcpyToSymbol(blockCounter, &zero, sizeof(int)));

    // Allocate and initialize flags
    unsigned int *d_flags;
    checkCudaErrors(cudaMalloc(&d_flags, (gridSize + 1) * sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(d_flags, 0, (gridSize + 1) * sizeof(unsigned int)));
    
    // Set first flag to 1
    unsigned int one = 1;
    checkCudaErrors(cudaMemcpy(d_flags, &one, sizeof(unsigned int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(flags, &d_flags, sizeof(unsigned int *)));

    size_t sharedMemSize = blockSize * sizeof(float);

    // Launch kernel
    EfficientPrefixSumUlt<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, inputLength);
    
    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    checkCudaErrors(cudaMemcpy(h_output, d_output, inputLength * sizeof(float), cudaMemcpyDeviceToHost));
    
    // End timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_flags);
    
    return diff.count();
}

// Function to run prefix sum with pinned memory
double runWithPinnedMemory(float *h_input, float *h_output, int inputLength) {
    float *d_input, *d_output;
    float *pinned_input, *pinned_output;
    
    // Start timer
    auto start = std::chrono::high_resolution_clock::now();
    
    // Allocate pinned memory
    checkCudaErrors(cudaMallocHost(&pinned_input, inputLength * sizeof(float)));
    checkCudaErrors(cudaMallocHost(&pinned_output, inputLength * sizeof(float)));
    
    // Copy input data to pinned memory
    memcpy(pinned_input, h_input, inputLength * sizeof(float));
    
    // Allocate device memory
    checkCudaErrors(cudaMalloc(&d_input, inputLength * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_output, inputLength * sizeof(float)));

    // Copy input data to device
    checkCudaErrors(cudaMemcpy(d_input, pinned_input, inputLength * sizeof(float), cudaMemcpyHostToDevice));

    int blockSize = BLOCK_SIZE;
    int gridSize = (inputLength + blockSize - 1) / blockSize;
    
    // Reset block counter
    int zero = 0;
    checkCudaErrors(cudaMemcpyToSymbol(blockCounter, &zero, sizeof(int)));

    // Allocate and initialize flags
    unsigned int *d_flags;
    checkCudaErrors(cudaMalloc(&d_flags, (gridSize + 1) * sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(d_flags, 0, (gridSize + 1) * sizeof(unsigned int)));
    
    // Set first flag to 1
    unsigned int one = 1;
    checkCudaErrors(cudaMemcpy(d_flags, &one, sizeof(unsigned int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(flags, &d_flags, sizeof(unsigned int *)));

    size_t sharedMemSize = blockSize * sizeof(float);

    // Launch kernel
    EfficientPrefixSumUlt<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, inputLength);
    
    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    checkCudaErrors(cudaMemcpy(pinned_output, d_output, inputLength * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Copy from pinned memory to output
    memcpy(h_output, pinned_output, inputLength * sizeof(float));
    
    // End timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;
    
    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_flags);
    cudaFreeHost(pinned_input);
    cudaFreeHost(pinned_output);
    
    return diff.count();
}

// Function to run prefix sum with unified memory
double runWithUnifiedMemory(float *h_input, float *h_output, int inputLength) {
    float *unified_input, *unified_output;
    
    // Start timer
    auto start = std::chrono::high_resolution_clock::now();
    
    // Allocate unified memory
    checkCudaErrors(cudaMallocManaged(&unified_input, inputLength * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&unified_output, inputLength * sizeof(float)));
    
    // Copy input data to unified memory
    memcpy(unified_input, h_input, inputLength * sizeof(float));

    int blockSize = BLOCK_SIZE;
    int gridSize = (inputLength + blockSize - 1) / blockSize;
    
    // Reset block counter
    int zero = 0;
    checkCudaErrors(cudaMemcpyToSymbol(blockCounter, &zero, sizeof(int)));

    // Allocate and initialize flags
    unsigned int *d_flags;
    checkCudaErrors(cudaMalloc(&d_flags, (gridSize + 1) * sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(d_flags, 0, (gridSize + 1) * sizeof(unsigned int)));
    
    // Set first flag to 1
    unsigned int one = 1;
    checkCudaErrors(cudaMemcpy(d_flags, &one, sizeof(unsigned int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(flags, &d_flags, sizeof(unsigned int *)));

    // Prefetch to GPU
    int device = -1;
    cudaGetDevice(&device);
    // when i searched and asked llm i found that there are method that called prefetch it increases slighlty the performance by simply prefetching
    // but i wont use it because we didnt take it in lecture 
    //cudaMemPrefetchAsync(unified_input, inputLength * sizeof(float), device);

    size_t sharedMemSize = blockSize * sizeof(float);

    // Launch kernel
    EfficientPrefixSumUlt<<<gridSize, blockSize, sharedMemSize>>>(unified_input, unified_output, inputLength);
    
    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Prefetch to CPU
    //cudaMemPrefetchAsync(unified_output, inputLength * sizeof(float), cudaCpuDeviceId);
    
    // Copy result to host to print and validate output 
    memcpy(h_output, unified_output, inputLength * sizeof(float));
    
    // End timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;
    
    // Free memory
    cudaFree(unified_input);
    cudaFree(unified_output);
    cudaFree(d_flags);
    
    return diff.count();
}

// Function to run prefix sum with zero-copy memory
double runWithZeroCopyMemory(float *h_input, float *h_output, int inputLength) {
    float *zero_copy_input, *zero_copy_output;
    
    // Start timer
    auto start = std::chrono::high_resolution_clock::now();
    
    // Allocate zero-copy memory
    checkCudaErrors(cudaHostAlloc(&zero_copy_input, inputLength * sizeof(float), cudaHostAllocMapped));
    checkCudaErrors(cudaHostAlloc(&zero_copy_output, inputLength * sizeof(float), cudaHostAllocMapped));
    
    // Copy input data to zero-copy memory
    memcpy(zero_copy_input, h_input, inputLength * sizeof(float));
    
    // Get device pointers for zero-copy memory
    float *d_input, *d_output;
    checkCudaErrors(cudaHostGetDevicePointer(&d_input, zero_copy_input, 0));
    checkCudaErrors(cudaHostGetDevicePointer(&d_output, zero_copy_output, 0));

    int blockSize = BLOCK_SIZE;
    int gridSize = (inputLength + blockSize - 1) / blockSize;
    
    // Reset block counter
    int zero = 0;
    checkCudaErrors(cudaMemcpyToSymbol(blockCounter, &zero, sizeof(int)));

    // Allocate and initialize flags
    unsigned int *d_flags;
    checkCudaErrors(cudaMalloc(&d_flags, (gridSize + 1) * sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(d_flags, 0, (gridSize + 1) * sizeof(unsigned int)));
    
    // Set first flag to 1
    unsigned int one = 1;
    checkCudaErrors(cudaMemcpy(d_flags, &one, sizeof(unsigned int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(flags, &d_flags, sizeof(unsigned int *)));

    size_t sharedMemSize = blockSize * sizeof(float);

    // Launch kernel
    EfficientPrefixSumUlt<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, inputLength);
    
    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy result to host output
    memcpy(h_output, zero_copy_output, inputLength * sizeof(float));
    
    // End timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;
    
    // Free memory
    cudaFreeHost(zero_copy_input);
    cudaFreeHost(zero_copy_output);
    cudaFree(d_flags);
    
    return diff.count();
}

// Main function to compare all memory types
int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("ERROR Usage: %s <inputfile> <outputfile>\n", argv[0]);
        return -1;
    }

    const char *inputFile = argv[1];
    const char *outputFile = argv[2];

    // Load input vector from file
    float *h_input;
    int inputLength;
    readInputFile(inputFile, &h_input, &inputLength);
    
    // Allocate host output memory
    float *h_output = (float *)malloc(inputLength * sizeof(float));
    
    // Verify CUDA device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return -1;
    }
    
    // Print memory type benchmark comparison
    printf("=== Memory Type Performance Comparison ===\n");
    printf("Input size: %d elements\n\n", inputLength);
    
    // Run benchmarks multiple times to get more stable results
    const int NUM_RUNS = 1;
    double pageable_time = 0, pinned_time = 0, unified_time = 0, zerocopy_time = 0;
    
    printf("Running benchmarks (%d runs each)...\n", NUM_RUNS);
    
    for (int i = 0; i < NUM_RUNS; i++) {
        // Run with pageable memory
        //pageable_time += runWithPageableMemory(h_input, h_output, inputLength);
        
        // Run with pinned memory
        //pinned_time += runWithPinnedMemory(h_input, h_output, inputLength);
        
        // Run with unified memory
        //unified_time += runWithUnifiedMemory(h_input, h_output, inputLength);
        
        // Run with zero-copy memory
        zerocopy_time += runWithZeroCopyMemory(h_input, h_output, inputLength);
    }
    
    // Calculate averages
    pageable_time /= NUM_RUNS;
    pinned_time /= NUM_RUNS;
    unified_time /= NUM_RUNS;
    zerocopy_time /= NUM_RUNS;
    
    // Print results
    printf("\nResults (average time in milliseconds):\n");
    printf("Pageable memory: %.3f ms\n", pageable_time);
    printf("Pinned memory:   %.3f ms\n", pinned_time);
    printf("Unified memory:  %.3f ms\n", unified_time);
    printf("Zero-copy memory: %.3f ms\n\n", zerocopy_time);
    
    // Calculate and print speedups
    double base_time = pageable_time; // Use pageable as baseline
    printf("Speedups relative to pageable memory:\n");
    printf("Pinned memory:   %.2fx\n", base_time / pinned_time);
    printf("Unified memory:  %.2fx\n", base_time / unified_time);
    printf("Zero-copy memory: %.2fx\n", base_time / zerocopy_time);
    
    // Save final result to output file using the last run's output
    writeOutputFile(outputFile, h_output, inputLength);
    
    // Free host memory
    free(h_input);
    free(h_output);

    return 0;
}