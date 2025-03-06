#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#define MAX_ERR 1e-6

// Function to perform vector addition
void vector_add(double *out, double *a, double *b, size_t  n) {
    for (size_t  i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

int main(int argc, char* argv[]) {
    FILE *file_reading;
    int numberOfTests;
    size_t rows, cols;
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
    fscanf(file_reading, "%zu %zu",&rows, &cols);
    // Allocate host matrices
    double* A = (double*)malloc(sizeof(double) * rows * cols);
    double* B = (double*)malloc(sizeof(double) * rows * cols);
    double* C = (double*)malloc(sizeof(double) * rows * cols);

    if (A == NULL || B == NULL || C == NULL) {
        printf("Memory allocation failed!\n");
        fclose(file_reading);
        return 1;
    }

    // Read matrices A and B
    for (size_t i = 0; i < rows * cols; i++) {
        fscanf(file_reading, "%lf", &A[i]);
    }
    for (size_t i = 0; i < rows * cols; i++) {
        fscanf(file_reading, "%lf", &B[i]);
    }



    // Start timing
    clock_t start = clock();

    // Perform vector addition
    vector_add(C, A, B, rows * cols);

    // End timing
    clock_t end = clock();

    // Calculate the elapsed time in seconds
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;

    printf("Time elapsed: %f ms\n", time_spent);

    // Verification
    for (size_t i = 0; i < rows * cols; i++) {
        assert(fabs(C[i] - A[i] - B[i]) < MAX_ERR);
    }

    printf("Vector addition completed successfully!\n");

  
    // Write results to output file
   // Write results to output file
    FILE *file_writing;
    file_writing= fopen(argv[2], "w"); // Open file for writing
    if (file_writing == NULL) {
        perror("Error opening file");
        return 1;
    }


    // Write matrix C
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
              // printf("%.3lf ", C[i * cols + j]);    
            fprintf(file_writing, "%.3lf ", C[i * cols + j]); // Write double with 2 decimal places
        }
         // printf("\n");  
        fprintf(file_writing, "\n"); // New line after each row
    }
    fclose(file_writing);


    // Free allocated memory
    free(A);
    free(B);
    free(C);
}
    return 0;
}
