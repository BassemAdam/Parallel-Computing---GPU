#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Custom atomicMax for float values
__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, 
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    
    return __int_as_float(old);
}


// Kernel to find maximum element using warp synchronization
__global__ void findMaxElement(float* input, float* output, int size) {
    // Shared memory for partial results from each warp
    __shared__ float warp_maxes[32]; // Max 32 warps per block
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % 32; // Lane ID within warp
    int warp_id = threadIdx.x / 32; // Warp ID within block
    
    // Each thread loads one element and finds local max
    float local_max = -INFINITY;
    if (tid < size) {
        local_max = input[tid];
    }
    
    // Perform warp-level reduction using shuffle operations
    for (int offset = 16; offset > 0; offset /= 2) {
        float neighbor = __shfl_down_sync(0xffffffff, local_max, offset);
        local_max = fmaxf(local_max, neighbor);
    }
    
    // First thread in each warp writes result to shared memory
    if (lane_id == 0) {
        warp_maxes[warp_id] = local_max;
    }
    
    __syncthreads();
    
    // First warp reduces all partial maxes from all warps
    if (warp_id == 0 && lane_id < (blockDim.x + 31) / 32) {
        local_max = warp_maxes[lane_id];
        
        // Final warp reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            float neighbor = __shfl_down_sync(0xffffffff, local_max, offset);
            local_max = fmaxf(local_max, neighbor);
        }
        
        // First thread in block writes result to global memory
        if (lane_id == 0) {
            atomicMaxFloat(output, local_max);
        }
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
void writeOutputFile(const char *filename, double result) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Round to 3 decimal places
    result = round(result * 1000.0) / 1000.0;

    fprintf(file, "%.3f\n", result);
    fclose(file);
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
    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(double));
    cudaMalloc(&d_output, sizeof(double));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size * sizeof(double), cudaMemcpyHostToDevice);

    // Initialize output with negative infinity
    double neg_inf = -INFINITY;
    cudaMemcpy(d_output, &neg_inf, sizeof(double), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    printf("Launching max element finder kernel with grid size %d, block size %d\n", gridSize, blockSize);
    
    // Launch kernel
    findMaxElement<<<gridSize, blockSize>>>(d_input, d_output, size);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Wait for kernel to complete
    cudaDeviceSynchronize();

    // Copy result back to host
    float h_output;
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Save result to output file
    writeOutputFile(outputFile, h_output);

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);

    return 0;
}