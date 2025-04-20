/* Work Inefficient PrefixSum*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

__global__ void InefficientNaivePrefixSum(float *input, float *output, const int length)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    extern __shared__ float s_accSum[];
    if (idx > length)
        return;
    for (size_t i = blockDim.x * blockIdx.x; i < idx; i++)
    {
        s_accSum[threadIdx.x] += input[i];
    }

    output[idx] = s_accSum[threadIdx.x];
}

__global__ void InefficientPrefixSum(float *input, float *output, int length)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    extern __shared__ float s_accSum[];
    if (idx < length)
    {
        s_accSum[threadIdx.x] = input[idx];
    }
    else
    {
        s_accSum[threadIdx.x] = 0;
        // why not return ? and must be zero ?
    }
    if (idx >= length)
        return;
    for (size_t stride = 1; stride < blockDim.x; stride *= 2)
    {
        __syncthreads();
        // this is easy we read and write from the same thing
        // so we should syncthread that all finished writing to start the next round of writing or so
        float accSum = 0;
        /*
         lets say blockdim.x = 4
         and iam thread.x=3 meaning iam at element number 4
         this for loop
         thread.x-stride = 3 -1 =2
         3 -2 = 1
         3 - 4 = -1 if condition block this
         so i will be able to accumulate sum my element 3 and 2 and 1 only
         the element at zero i wont be able to get it !
        */
        if (stride <= threadIdx.x)
        {
            accSum = s_accSum[threadIdx.x - stride];
        }
        __syncthreads();
        s_accSum[threadIdx.x] += accSum;
    }

    if (idx < length)
        output[idx] = s_accSum[threadIdx.x];
}

// Global counter for block execution order
__device__ int blockCounter;
// Global flags
__device__ unsigned int *flags;

__global__ void InefficientPrefixSumUlt(float *input, float *output, int length)
{

    // Normal Prefix Sum
    extern __shared__ float s_accSum[];
    // Checking that Blocks are executed in order
    __shared__ int s_blk_id;
 
    if (threadIdx.x == 0){
       // printf("blockCounter: %d\n",blockCounter);
        s_blk_id = atomicAdd(&blockCounter, 1);
        //printf("s_blk_id: %d\n",s_blk_id);
    }
    __syncthreads();
   //  if (threadIdx.x == 0) printf("s_blk_id: %d\n",s_blk_id);

    int idx = threadIdx.x + blockDim.x * s_blk_id;

    s_accSum[threadIdx.x] = idx < length ? input[idx] : 0;
    // why not return ? and must be zero ?

    if (idx >= length)
        return;

    for (size_t stride = 1; stride < blockDim.x; stride *= 2)
    {
        __syncthreads();
        // this is easy we read and write from the same thing
        // so we should syncthread that all finished writing to start the next round of writing or so
        float accSum = 0;
        if (stride <= threadIdx.x)
        {
            accSum = s_accSum[threadIdx.x - stride];
        }
        __syncthreads();
        s_accSum[threadIdx.x] += accSum;
    }

    // Checking that Blocks are executed in order
    __shared__ float s_previous_sum;
    if (threadIdx.x == 0)
    {
        //printf("Block %u: Waiting, blockCounter=%u, flag[%u]=%u\n",
         //      s_blk_id, blockCounter, s_blk_id - 1, atomicAdd(&flags[s_blk_id - 1], 0));

        // wait for previous flag that the block before it is finished
        while (s_blk_id != 0 && atomicAdd(&flags[s_blk_id - 1], 0) == 0)
        {
            // Add periodic debug output to avoid flooding
            if (clock() % 10000000 == 0)
            { // Only print occasionally
                printf("ahhhhh iam in deadlock bassem help me");
            }
        }

        s_previous_sum = s_blk_id > 0 ? output[idx - 1] : 0;
    }
    __syncthreads();

    if (idx < length)
        output[idx] = s_accSum[threadIdx.x] + s_previous_sum;

    // printf("Block %u, Thread %d: idx=%d, s_previous_sum=%.2f, s_accSum=%.2f, output=%.2f\n",
    //        s_blk_id, threadIdx.x, idx, s_previous_sum, s_accSum[threadIdx.x], output[idx]);

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

    int blockSize = 256;
    int gridSize = (inputLength + blockSize - 1) / blockSize;
    size_t sharedMemSize = inputLength > 256 ? blockSize * sizeof(float) : inputLength * sizeof(float);

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
    InefficientPrefixSumUlt<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, inputLength);

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