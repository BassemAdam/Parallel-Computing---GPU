#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// First kernel - find max per block
__global__ void findMax(float* input, float* blockMaxes, int size) {
    extern __shared__ float sharedMem[];

    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory
    sharedMem[tid] = globalIdx < size ? input[globalIdx] : -INFINITY;

    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sharedMem[tid] = fmaxf(sharedMem[tid], sharedMem[tid + s]);
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0)
    {
        blockMaxes[blockIdx.x] = sharedMem[0];
    }
}

// Second kernel - find global max from block maxes
__global__ void findGlobalMax(float* blockMaxes, float* globalMax, int numBlocks) {
    extern __shared__ float sharedMem[];

    int tid = threadIdx.x;
    
    // Load block maximums into shared memory
    if (tid < numBlocks) {
        sharedMem[tid] = blockMaxes[tid];
    } else {
        sharedMem[tid] = -INFINITY;
    }
    
    __syncthreads();
    
    // Perform reduction to find global maximum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedMem[tid] = fmaxf(sharedMem[tid], sharedMem[tid + s]);
        }
        __syncthreads();
    }
    
    // Write global maximum
    if (tid == 0) {
        *globalMax = sharedMem[0];
    }
}

// Function to read 1D input data from file
void readInputFile(const char *filename, float **data, int *size) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Read the size of the array
    fscanf(file, "%d", size);

    // Allocate memory
    *data = (float *)malloc(*size * sizeof(float));

    // Read the array elements
    for (int i = 0; i < *size; i++) {
        fscanf(file, "%f", &(*data)[i]);
    }

    fclose(file);
    printf("Read %d elements from input file\n", *size);
}

// Function to write output data to file
void writeOutputFile(const char *filename, float result) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Round to 3 decimal places
    result = round(result * 1000.0) / 1000.0;

    fprintf(file, "%.3f\n", result);
    fclose(file);
    printf("Result %.3f written to output file\n", result);
}


int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <inputfile> <outputfile>\n", argv[0]);
        return -1;
    }

    const char *inputFile = argv[1];
    const char *outputFile = argv[2];

    // Load input data from file
    float *h_input;
    int size;
    readInputFile(inputFile, &h_input, &size);

    // Allocate device memory
    float *d_input;
    cudaMalloc(&d_input, size * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    // Allocate memory for block maximums
    float *d_blockMaxes;
    cudaMalloc(&d_blockMaxes, gridSize * sizeof(float));
    
    // Allocate memory for global maximum
    float *d_globalMax;
    cudaMalloc(&d_globalMax, sizeof(float));
    
    // Initialize global max with negative infinity
    float neg_inf = -INFINITY;
    cudaMemcpy(d_globalMax, &neg_inf, sizeof(float), cudaMemcpyHostToDevice);

    printf("Launching max element finder kernel with grid size %d, block size %d\n", gridSize, blockSize);
    
    // Launch first kernel - find max per block
    findMax<<<gridSize, blockSize, blockSize * sizeof(float)>>>(
        d_input, d_blockMaxes, size);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "First kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Launch second kernel to find global maximum if we have multiple blocks
    if (gridSize > 1) {
        // Use a single block for the second reduction
        int secondBlockSize = min(blockSize, gridSize);
        printf("Launching global max finder kernel with block size %d\n", secondBlockSize);
        
        findGlobalMax<<<1, secondBlockSize, secondBlockSize * sizeof(float)>>>(
            d_blockMaxes, d_globalMax, gridSize);
            
        // Check for kernel launch errors
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Second kernel launch failed: %s\n", cudaGetErrorString(err));
            return -1;
        }
    } else {
        // If only one block, block max is already the global max
        cudaMemcpy(d_globalMax, d_blockMaxes, sizeof(float), cudaMemcpyDeviceToDevice);
    }

    // Wait for kernel to complete
    cudaDeviceSynchronize();

    // Copy result back to host
    float h_output;
    cudaMemcpy(&h_output, d_globalMax, sizeof(float), cudaMemcpyDeviceToHost);

    // Save result to output file
    writeOutputFile(outputFile, h_output);

    // Free memory
    cudaFree(d_input);
    cudaFree(d_blockMaxes);
    cudaFree(d_globalMax);
    free(h_input);

    return 0;
}