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

    // Parameters for streams
    int numStreams = 4; // You can adjust this for your GPU
    int chunkSize = (inputLength + numStreams - 1) / numStreams;

    // Allocate host output
    float *h_output = (float *)malloc(inputLength * sizeof(float));

    // Create streams
    cudaStream_t *streams = (cudaStream_t *)malloc(numStreams * sizeof(cudaStream_t));
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // Allocate device memory for each stream's chunk
    float **d_inputs = (float **)malloc(numStreams * sizeof(float *));
    float **d_outputs = (float **)malloc(numStreams * sizeof(float *));
    unsigned int **d_flags_arr = (unsigned int **)malloc(numStreams * sizeof(unsigned int *));
    for (int i = 0; i < numStreams; ++i) {
        int thisChunk = ((i+1)*chunkSize > inputLength) ? (inputLength - i*chunkSize) : chunkSize;
        cudaMalloc(&d_inputs[i], thisChunk * sizeof(float));
        cudaMalloc(&d_outputs[i], thisChunk * sizeof(float));
        int blockSize = BLOCK_SIZE;
        int gridSize = (thisChunk + blockSize - 1) / blockSize;
        cudaMalloc(&d_flags_arr[i], (gridSize + 1) * sizeof(unsigned int));
    }

    // Copy input chunks to device in all streams
    for (int i = 0; i < numStreams; ++i) {
        int offset = i * chunkSize;
        int thisChunk = ((i+1)*chunkSize > inputLength) ? (inputLength - i*chunkSize) : chunkSize;
        if (thisChunk <= 0) continue;
        cudaMemcpyAsync(d_inputs[i], h_input + offset, thisChunk * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
    }

    // Prepare and launch kernels in all streams
    for (int i = 0; i < numStreams; ++i) {
        int offset = i * chunkSize;
        int thisChunk = ((i+1)*chunkSize > inputLength) ? (inputLength - i*chunkSize) : chunkSize;
        if (thisChunk <= 0) continue;

        // Reset block counter before kernel launch
        int zero = 0;
        cudaMemcpyToSymbolAsync(blockCounter, &zero, sizeof(int), 0, cudaMemcpyHostToDevice, streams[i]);

        int blockSize = BLOCK_SIZE;
        int gridSize = (thisChunk + blockSize - 1) / blockSize;
        cudaMemsetAsync(d_flags_arr[i], 0, (gridSize + 1) * sizeof(unsigned int), streams[i]);
        unsigned int one = 1;
        cudaMemcpyAsync(d_flags_arr[i], &one, sizeof(unsigned int), cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyToSymbolAsync(flags, &d_flags_arr[i], sizeof(unsigned int *), 0, cudaMemcpyHostToDevice, streams[i]);

        EfficientPrefixSumUlt<<<gridSize, blockSize, 0, streams[i]>>>(d_inputs[i], d_outputs[i], thisChunk);
    }

    // Copy output chunks back to host in all streams
    for (int i = 0; i < numStreams; ++i) {
        int offset = i * chunkSize;
        int thisChunk = ((i+1)*chunkSize > inputLength) ? (inputLength - i*chunkSize) : chunkSize;
        if (thisChunk <= 0) continue;
        cudaMemcpyAsync(h_output + offset, d_outputs[i], thisChunk * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    // Synchronize all streams
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamSynchronize(streams[i]);
    }

    // Save result to output file
    writeOutputFile(outputFile, h_output, inputLength);

    // Free memory
    for (int i = 0; i < numStreams; ++i) {
        cudaFree(d_inputs[i]);
        cudaFree(d_outputs[i]);
        cudaFree(d_flags_arr[i]);
        cudaStreamDestroy(streams[i]);
    }
    free(d_inputs);
    free(d_outputs);
    free(d_flags_arr);
    free(streams);
    free(h_input);
    free(h_output);

    return 0;
}