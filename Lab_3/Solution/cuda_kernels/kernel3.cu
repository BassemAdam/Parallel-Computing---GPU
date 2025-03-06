#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

// Kernel to sum across the z-dimension
__global__ void sumZDimension(double *input, double *output, size_t width, size_t height, size_t depth)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        double sum = 0.0;
        for (size_t z = 0; z < depth; ++z)
        {
            sum += input[(z * height + y) * width + x];
        }
        output[y * width + x] = sum;
    }
}

// Kernel to reduce 2D matrix to 1D vector
__global__ void reduce2DTo1D(double *input, double *output, size_t width, size_t height)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < width)
    {
        double sum = 0.0;
        for (size_t y = 0; y < height; ++y)
        {
            sum += input[y * width + x];
        }
        output[x] = sum;
    }
}

// kernel to reduce 1D vector to vector of length equal to number of blocks
__global__ void reduce1DTo1D(double *input, double *output, size_t length)
{
    extern __shared__ double sharedData[];

    // Calculate global and local indices
    size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t thread_Idx = threadIdx.x;

    // Load data into shared memory
    if (global_idx < length)
    {
        sharedData[thread_Idx] = input[global_idx];
    }else
    {
        sharedData[thread_Idx] = 0.0;
    }
    
    __syncthreads();

    // Reduce within the block using parallel reduction
    for (size_t stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (thread_Idx < stride)
        {
            sharedData[thread_Idx] += sharedData[thread_Idx + stride];
        }
        __syncthreads();
    }
   
    // The first thread in each block writes the block's sum to global memory    
    if (thread_Idx == 0)
    {
        printf("Block %d: sharedSum[0] = %f\n", blockIdx.x, sharedData[0]);
        output[blockIdx.x] = sharedData[0];
    }
}

// Function to reduce 1D vector to a single element on the CPU
double reduce1DToSingle(double *input, size_t length)
{
    double sum = 0.0;
    for (size_t i = 0; i < length; ++i)
    {
        sum += input[i];
    }
    return sum;
}

// Function to read input data from file
void readInputFile(const char *filename, double **data, size_t *width, size_t *height, size_t *depth)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    fscanf(file, "%zu %zu %zu", width, height, depth);

    size_t size = (*width) * (*height) * (*depth);
    *data = (double *)malloc(size * sizeof(double));

    for (size_t i = 0; i < size; ++i)
    {
        fscanf(file, "%lf", &(*data)[i]);
    }

    fclose(file);
}

// Function to write output data to file
void writeOutputFile(const char *filename, double result)
{
    FILE *file = fopen(filename, "w");
    if (!file)
    {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Round to 3 decimal places
    result = round(result * 1000.0) / 1000.0;

    fprintf(file, "%.3f\n", result);
    fclose(file);
}

// Function to write output array data to file
void writeOutputArrayToFile(const char *filename, double *data, size_t blocksNeededX)
{
    FILE *file = fopen(filename, "w");
    if (!file)
    {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // // Calculate the length of the array
    // size_t length = 0;
    // while (data[length] != '\0')
    // {
    //     length++;
    // }

    // Write each element of the array to the file, rounded to 3 decimal places
    for (size_t i = 0; i < blocksNeededX; ++i)
    {
        double rounded_value = round(data[i] * 1000.0) / 1000.0;
        fprintf(file, "%.3f\n", rounded_value);
    }

    fclose(file);
}

int main(int argc, char *argv[])
{

    // Handling Reading the arguments input and ouput file names from the command line
    if (argc != 3)
    {
        printf("Usage: %s <inputfile> <outputfile>\n", argv[0]);
        return -1;
    }
    const char *inputFile = argv[1];
    const char *outputFile = argv[2];
    // Load input data from file
    double *h_input;
    size_t width, height, depth;
    readInputFile(inputFile, &h_input, &width, &height, &depth);
    //-------------------------------------------------------------------------------------

    // Allocate memory for input and output
    double *d_input, *d_output2D, *d_output1D, *d_output1D_blockReduced;
    cudaMalloc(&d_input, width * height * depth * sizeof(double));
    cudaMalloc(&d_output2D, width * height * sizeof(double));
    cudaMalloc(&d_output1D, width * sizeof(double));
    cudaMalloc(&d_output1D_blockReduced, width * sizeof(double));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, width * height * depth * sizeof(double), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    size_t blocksNeededX = (width + blockSize.x - 1) / blockSize.x;
    size_t blocksNeededY = (height + blockSize.y - 1) / blockSize.y;
    dim3 gridSize(blocksNeededX, blocksNeededY);

    // Launch first kernel
    printf("Kernel sumZDimension Started\n");
    sumZDimension<<<gridSize, blockSize>>>(d_input, d_output2D, width, height, depth);

    // Launch second kernel
    printf("Kernel reduce2DTo1D Started\n");
    reduce2DTo1D<<<blocksNeededX, blockSize.x>>>(d_output2D, d_output1D, width, height);

    // Launch third kernel
    printf("Kernel reduce1DTo1D Started\n");
    size_t blocksNeeded_256 = (width + 256 - 1) / 256;
    size_t sharedMemSize = 256 * sizeof(double); // For 256 threads per block
    reduce1DTo1D<<<blocksNeeded_256,256,sharedMemSize>>>(d_output1D, d_output1D_blockReduced, width);

    // Copy result back to host
    double *h_output1D = (double *)malloc(blocksNeeded_256 * sizeof(double));
    cudaMemcpy(h_output1D, d_output1D_blockReduced, blocksNeeded_256 * sizeof(double), cudaMemcpyDeviceToHost);

    // Save result to output file
    // Save result to output file this is to see the output of the kernel that reduce to 1d
    // for example to make sure that the length of vector to the number of threads or blocks in req 3 is correct 
    // i will comment so that i dont lose marks 
    writeOutputArrayToFile("./Output_TestCases/1d_kernel3.txt", h_output1D,blocksNeeded_256);

    // Measure time in nanoseconds using chrono
    auto start = std::chrono::high_resolution_clock::now();

    double result = reduce1DToSingle(h_output1D, blocksNeeded_256);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Execution time of reduce1DToSingle: %lld ns (%.6f ms)\n", duration, duration / 1e6);

    // Save result to output file
    writeOutputFile(outputFile, result);

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output2D);
    cudaFree(d_output1D);
    cudaFree(d_output1D_blockReduced);
    free(h_input);
    free(h_output1D);

    return 0;
}
