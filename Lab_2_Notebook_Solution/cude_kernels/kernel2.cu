#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <fstream>
#include <iostream>
#include <sstream>
#define MAX_ERR 1e-6
__global__ void matrixAddKernel1(double* C, double* A, double* B, size_t rows, size_t cols) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        for(size_t col =0 ; col < cols; col++){
              size_t idx = row * cols + col;
            C[idx] = A[idx] + B[idx];
        }
    }
}

cudaError_t addMatricesWithCuda(double* C, double* A, double* B, size_t rows, size_t cols) {
    double* dev_A = nullptr;
    double* dev_B = nullptr;
    double* dev_C = nullptr;
    cudaError_t cudaStatus;

    // Allocate GPU buffers
    size_t size = rows * cols * sizeof(double);  // Changed from size_t to double
    
    cudaStatus = cudaMalloc((void**)&dev_C, size);
    
    cudaStatus = cudaMalloc((void**)&dev_A, size);

    cudaStatus = cudaMalloc((void**)&dev_B, size);

    // Copy input matrices from host memory to GPU buffers
    cudaStatus = cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);    
    cudaStatus = cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16); // i just did what Ta and cuda said the best to use
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrixAddKernel1<<<numBlocks, threadsPerBlock>>>(dev_C, dev_A, dev_B, rows, cols);

    // Copy output matrix from GPU buffer to host memory
    cudaStatus = cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost);

    return cudaStatus;
}

int main(int argc, char* argv[]) {

    FILE *file_reading;
    int numberOfTests;
    size_t  rows, cols;
    // Open the file in read mode
    file_reading = fopen(argv[1], "r");
    if (file_reading == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    // Read number of tests
    fscanf(file_reading, "%d",&numberOfTests);
for(size_t i=0;i<numberOfTests;i++){
    
    // Read matrix dimensions
    fscanf(file_reading, "%zu %zu", &rows, &cols);

    // Allocate host matrices
    double* A = (double*)malloc(sizeof(double) * rows * cols);  // Changed from size_t to double
    double* B = (double*)malloc(sizeof(double) * rows * cols);
    double* C = (double*)malloc(sizeof(double) * rows * cols);

    // Read matrices A and B
    for (size_t i = 0; i < rows * cols; i++) {
        fscanf(file_reading, "%lf", &A[i]);
    }
    for (size_t i = 0; i < rows * cols; i++) {
        fscanf(file_reading, "%lf", &B[i]);
    }
    
 

    // Add matrices using CUDA
    cudaError_t cudaStatus = addMatricesWithCuda(C, A, B, rows, cols);

    // Verification
    for (size_t i = 0; i < rows * cols; i++) {
        assert(fabs(C[i] - A[i] - B[i]) < MAX_ERR);
    }

    printf("Vector addition completed successfully!\n");

    // Write results to output file
    FILE *file_writing;
    file_writing= fopen(argv[2], "a"); // Open file for writing
    if (file_writing == NULL) {
        perror("Error opening file");
        return 1;
    }


    // Write matrix C
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            //  printf("%.3f ", C[i * cols + j]);    
            fprintf(file_writing, "%.3f ", C[i * cols + j]); // Write double with 2 decimal places
        }
        // printf("\n");  
        fprintf(file_writing, "\n"); // New line after each row
    }
    fclose(file_writing);


    // Cleanup
    free(A);
    free(B);
    free(C);
}
    return 0;
}
