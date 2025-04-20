/* Work Inefficient PrefixSum*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

// Global counter for block execution order
__device__ int blockCounter;
// Global flags
__device__ unsigned int *flags;
__device__ const  int BLOCK_SIZE=256;
__global__ void EfficientPrefixSumUlt(float *input, float *output, int length)
{

    // Normal Prefix Sum
    __shared__ float s_accSum[2*BLOCK_SIZE];
    // Checking that Blocks are executed in order
    __shared__ int s_blk_id;
 
    if (threadIdx.x == 0) s_blk_id = atomicAdd(&blockCounter, 1);
    __syncthreads();

    int idx = threadIdx.x + blockDim.x * s_blk_id;
    s_accSum[threadIdx.x] = idx < length ? input[idx] : 0;     // why not return ? and must be zero ?


    if (idx >= length) return;

    // wa e7na nazlien lmien
    for (size_t stride = 1; stride < BLOCK_SIZE; stride *= 2)
    {
        int index = threadIdx.x*stride*2 + stride*2 - 1;
        if (index <=BLOCK_SIZE*2)
        {
            s_accSum[index] += s_accSum[index- stride];
        }
        __syncthreads();
    }

    // wa e7na nazlien 
    for (size_t stride = BLOCK_SIZE/2; stride > 0; stride /= 2)
    {
        __syncthreads();
        int index = threadIdx.x*stride*2 + stride*2 - 1;
        if (index+ stride<=BLOCK_SIZE*2)
        {
            s_accSum[index + stride] += s_accSum[index];
        }
    }
    __syncthreads();
    
    // Checking that Blocks are executed in order
    __shared__ float s_previous_sum;
    if (threadIdx.x == 0)
    {
        // wait for previous flag that the block before it is finished
        while (s_blk_id != 0 && atomicAdd(&flags[s_blk_id - 1], 0) == 0)
        {
        }

        s_previous_sum = s_blk_id > 0 ? output[idx - 1] : 0;
    }
    __syncthreads();

    if (idx < length)
        output[idx] = s_accSum[threadIdx.x] + s_previous_sum;


    // set flag
    if (threadIdx.x == blockDim.x - 1)
    {
        // wait for all cached values to be moving to global memory
        __threadfence();

        // set flag
        atomicAdd(&flags[s_blk_id], 1); // Changed from flags[s_blk_id + 1]
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

int main(int argc, char *argv[]){
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

    // Timing variables
    cudaEvent_t start, stop;
    float elapsedTime;

    int blockSize = BLOCK_SIZE;
    int gridSize = (inputLength + blockSize - 1) / blockSize;

    // a. Pageable memory (current implementation)
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    float *d_input, *d_output;
    cudaMalloc(&d_input, inputLength * sizeof(float));
    cudaMalloc(&d_output, inputLength * sizeof(float));
    cudaMemcpy(d_input, h_input, inputLength * sizeof(float), cudaMemcpyHostToDevice);

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

    EfficientPrefixSumUlt<<<gridSize, blockSize>>>(d_input, d_output, inputLength);
    float *h_output = (float *)malloc(inputLength * sizeof(float));
    cudaMemcpy(h_output, d_output, inputLength * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Pageable memory time: %.3f ms\n", elapsedTime);

    writeOutputFile(outputFile, h_output, inputLength);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_output);

    // b. Unified memory
    cudaEventRecord(start, 0);

    float *unified_input, *unified_output;
    cudaMallocManaged(&unified_input, inputLength * sizeof(float));
    cudaMallocManaged(&unified_output, inputLength * sizeof(float));
    memcpy(unified_input, h_input, inputLength * sizeof(float));

    EfficientPrefixSumUlt<<<gridSize, blockSize>>>(unified_input, unified_output, inputLength);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Unified memory time: %.3f ms\n", elapsedTime);

    writeOutputFile("unified_output.txt", unified_output, inputLength);

    cudaFree(unified_input);
    cudaFree(unified_output);

    // c. Zero-copy (mapped) memory
    cudaEventRecord(start, 0);

    float *host_mapped_input, *host_mapped_output;
    float *dev_mapped_input, *dev_mapped_output;
    cudaHostAlloc(&host_mapped_input, inputLength * sizeof(float), cudaHostAllocMapped);
    cudaHostAlloc(&host_mapped_output, inputLength * sizeof(float), cudaHostAllocMapped);
    memcpy(host_mapped_input, h_input, inputLength * sizeof(float));
    cudaHostGetDevicePointer(&dev_mapped_input, host_mapped_input, 0);
    cudaHostGetDevicePointer(&dev_mapped_output, host_mapped_output, 0);

    EfficientPrefixSumUlt<<<gridSize, blockSize>>>(dev_mapped_input, dev_mapped_output, inputLength);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Zero-copy (mapped) memory time: %.3f ms\n", elapsedTime);

    writeOutputFile("zerocopy_output.txt", host_mapped_output, inputLength);

    cudaFreeHost(host_mapped_input);
    cudaFreeHost(host_mapped_output);

    // d. Pinned memory
    cudaEventRecord(start, 0);

    float *h_pinned_input, *h_pinned_output;
    cudaHostAlloc(&h_pinned_input, inputLength * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&h_pinned_output, inputLength * sizeof(float), cudaHostAllocDefault);
    memcpy(h_pinned_input, h_input, inputLength * sizeof(float));
    cudaMalloc(&d_input, inputLength * sizeof(float));
    cudaMalloc(&d_output, inputLength * sizeof(float));
    cudaMemcpy(d_input, h_pinned_input, inputLength * sizeof(float), cudaMemcpyHostToDevice);

    EfficientPrefixSumUlt<<<gridSize, blockSize>>>(d_input, d_output, inputLength);
    cudaMemcpy(h_pinned_output, d_output, inputLength * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Pinned memory time: %.3f ms\n", elapsedTime);

    writeOutputFile("pinned_output.txt", h_pinned_output, inputLength);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFreeHost(h_pinned_input);
    cudaFreeHost(h_pinned_output);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_input);

    return 0;
}