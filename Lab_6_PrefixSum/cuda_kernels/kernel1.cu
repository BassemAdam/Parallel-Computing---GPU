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

    // Launch convolution kernel with output tiling
    EfficientPrefixSumUlt<<<gridSize, blockSize>>>(d_input, d_output, inputLength);

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