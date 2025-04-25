/* Work Efficient PrefixSum with Streams*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

// Global counter for block execution order
__device__ int blockCounter;
// Global flags
__device__ unsigned int *flags;
__device__ const int BLOCK_SIZE = 256;

__global__ void EfficientPrefixSumUlt(float *input, float *output, int length, float *blockSums)
{
    // Shared memory for scan operation
    extern __shared__ float s_data[];
    
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
    
    // Store the block's total sum for later use in cross-stream adjustment
    if (threadIdx.x == 0 && blockIdx.x < gridDim.x) {
        blockSums[blockIdx.x] = s_data[blockDim.x - 1] + original_value;
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
    
    // Handle cross-block dependencies within the same stream
    __shared__ float s_previous_sum;
    if (threadIdx.x == 0) {
        // Wait for previous block to complete
        while (s_blk_id!= 0 && atomicAdd(&flags[s_blk_id- 1], 0) == 0) {}
        
        s_previous_sum = s_blk_id> 0 ? output[blockDim.x * s_blk_id- 1] : 0;
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

// Kernel to adjust values across streams
__global__ void AdjustStreamValues(float *output, float *streamSums, int length, int streamSize, int streamIdx)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < length && streamIdx > 0) {
        // Add sum from previous streams to the current stream's values
        output[idx + streamIdx * streamSize] += streamSums[streamIdx - 1];
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

    // Define stream count
    const int NUM_STREAMS = 4;
    cudaStream_t streams[NUM_STREAMS];
    
    // Create streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Calculate chunk size for each stream
    int streamSize = (inputLength + NUM_STREAMS - 1) / NUM_STREAMS;
    
    // Allocate device memory for full input and output
    float *d_input, *d_output;
    cudaMalloc(&d_input, inputLength * sizeof(float));
    cudaMalloc(&d_output, inputLength * sizeof(float));
    
    // Copy full input data to device
    cudaMemcpy(d_input, h_input, inputLength * sizeof(float), cudaMemcpyHostToDevice);
    
    // Allocate memory for stream sums
    float *d_streamSums;
    cudaMalloc(&d_streamSums, NUM_STREAMS * sizeof(float));
    cudaMemset(d_streamSums, 0, NUM_STREAMS * sizeof(float));
    
    // Allocate host memory for stream sums
    float *h_streamSums = (float *)malloc(NUM_STREAMS * sizeof(float));
    
    // Block and grid dimensions
    int blockSize = BLOCK_SIZE;
    
    // Process each stream
    for (int i = 0; i < NUM_STREAMS; i++) {
        // Calculate the size for this stream
        int currentSize = min(streamSize, inputLength - i * streamSize);
        if (currentSize <= 0) break;
        
        // Calculate grid size for this stream
        int gridSize = (currentSize + blockSize - 1) / blockSize;
        
        // Reset block counter for this stream
        int zero = 0;
        cudaMemcpyToSymbol(blockCounter, &zero, sizeof(int), 0, cudaMemcpyHostToDevice);
        
        // Allocate and initialize flags for this stream
        unsigned int *d_flags;
        cudaMalloc(&d_flags, (gridSize + 1) * sizeof(unsigned int));
        cudaMemset(d_flags, 0, (gridSize + 1) * sizeof(unsigned int));
        
        // Set first flag to 1 so first block doesn't wait
        unsigned int one = 1;
        cudaMemcpy(d_flags, &one, sizeof(unsigned int), cudaMemcpyHostToDevice);
        
        // Copy pointer to device symbol
        cudaMemcpyToSymbol(flags, &d_flags, sizeof(unsigned int *), 0, cudaMemcpyHostToDevice);
        
        // Allocate memory for block sums within this stream
        float *d_blockSums;
        cudaMalloc(&d_blockSums, gridSize * sizeof(float));
        
        // Calculate shared memory size
        size_t sharedMemSize = blockSize * sizeof(float);
        
        // Launch prefix sum kernel for this stream chunk
        EfficientPrefixSumUlt<<<gridSize, blockSize, sharedMemSize, streams[i]>>>(
            d_input + i * streamSize, 
            d_output + i * streamSize, 
            currentSize,
            d_blockSums
        );
        
        // Get the last sum from this stream to use for the next stream
        if (i < NUM_STREAMS - 1) {
            cudaMemcpyAsync(&h_streamSums[i], 
                           d_output + (i + 1) * streamSize - 1, 
                           sizeof(float), 
                           cudaMemcpyDeviceToHost, 
                           streams[i]);
        }
        
        // Clean up
        cudaFree(d_flags);
        cudaFree(d_blockSums);
    }
    
    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    // Calculate cumulative sums for stream adjustment
    float cumulativeSum = 0.0f;
    for (int i = 0; i < NUM_STREAMS - 1; i++) {
        cumulativeSum += h_streamSums[i];
        h_streamSums[i] = cumulativeSum;
    }
    
    // Copy cumulative sums back to device
    cudaMemcpy(d_streamSums, h_streamSums, (NUM_STREAMS - 1) * sizeof(float), cudaMemcpyHostToDevice);
    
    // Fix cross-stream dependencies
    for (int i = 1; i < NUM_STREAMS; i++) {
        int currentSize = min(streamSize, inputLength - i * streamSize);
        if (currentSize <= 0) break;
        
        int gridSize = (currentSize + blockSize - 1) / blockSize;
        AdjustStreamValues<<<gridSize, blockSize, 0, streams[i]>>>(
            d_output, d_streamSums, currentSize, streamSize, i);
    }
    
    // Synchronize all streams again
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    // Copy result back to host
    float *h_output = (float *)malloc(inputLength * sizeof(float));
    cudaMemcpy(h_output, d_output, inputLength * sizeof(float), cudaMemcpyDeviceToHost);

    // Save result to output file
    writeOutputFile(outputFile, h_output, inputLength);

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_streamSums);
    free(h_input);
    free(h_output);
    free(h_streamSums);
    
    // Destroy streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }

    return 0;
}