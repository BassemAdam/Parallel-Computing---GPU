#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Kernel to sum across the z-dimension
__global__ void sumZDimension(double *input, double *output, size_t width, size_t height, size_t depth) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        double sum = 0.0;
        for (size_t z = 0; z < depth; ++z) {
            sum += input[(z * height + y) * width + x];
        }
        output[y * width + x] = sum;
    }
}

// Kernel to reduce 2D matrix to 1D vector
__global__ void reduce2DTo1D(double *input, double *output, size_t width, size_t height) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < width) {
        double sum = 0.0;
        for (size_t y = 0; y < height; ++y) {
            sum += input[y * width + x];
        }
        output[x] = sum;
    }
}

// Function to reduce 1D vector to a single element on the CPU
double reduce1DToSingle(double *input, size_t length) {
    double sum = 0.0;
    for (size_t i = 0; i < length; ++i) {
        sum += input[i];
    }
    return sum;
}

// Function to read input data from file
void readInputFile(const char *filename, double **data, size_t *width, size_t *height, size_t *depth) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    fscanf(file, "%zu %zu %zu", width, height, depth);

    size_t size = (*width) * (*height) * (*depth);
    *data = (double *)malloc(size * sizeof(double));

    for (size_t i = 0; i < size; ++i) {
        fscanf(file, "%lf", &(*data)[i]);
    }

    fclose(file);
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
    double *h_input;
    size_t width, height, depth;
    readInputFile(inputFile, &h_input, &width, &height, &depth);

    // Allocate memory for input and output
    double *d_input, *d_output2D, *d_output1D;
    cudaMalloc(&d_input, width * height * depth * sizeof(double));
    cudaMalloc(&d_output2D, width * height * sizeof(double));
    cudaMalloc(&d_output1D, width * sizeof(double));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, width * height * depth * sizeof(double), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch first kernel
    printf("Kernel sumZDimension Started\n");
    sumZDimension<<<gridSize, blockSize>>>(d_input, d_output2D, width, height, depth);
    cudaDeviceSynchronize();

    // Launch second kernel
    printf("Kernel reduce2DTo1D Started\n");
    reduce2DTo1D<<<(width + blockSize.x - 1) / blockSize.x, blockSize.x>>>(d_output2D, d_output1D, width, height);
    cudaDeviceSynchronize();

    // Copy result back to host
    double *h_output1D = (double *)malloc(width * sizeof(double));
    cudaMemcpy(h_output1D, d_output1D, width * sizeof(double), cudaMemcpyDeviceToHost);

    // Reduce 1D vector to a single element on the CPU
    double result = reduce1DToSingle(h_output1D, width);

    // Save result to output file
    writeOutputFile(outputFile, result);

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output2D);
    cudaFree(d_output1D);
    free(h_input);
    free(h_output1D);

    return 0;
}
