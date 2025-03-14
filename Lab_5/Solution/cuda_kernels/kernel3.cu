#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <string>
#include <vector>
#include <cstring>

// Define constant memory for mask
__constant__ float c_mask[1024]; // Assuming mask won't exceed 1024 elements

// Structure to store RGB image data
typedef struct {
    unsigned char* data;
    int width;
    int height;
    int channels;
} RGBImage;

// 3D convolution kernel with output tiling
__global__ void convolution3D_OutputTiling(
    unsigned char* input,
    unsigned char* output,
    int width, 
    int height, 
    int channels,
    int maskSize,
    int stride
) {
    extern __shared__ unsigned char sharedMem[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int maskRadius = maskSize / 2;
    
    // Calculate output positions
    int outputX = bx * blockDim.x + tx;
    int outputY = by * blockDim.y + ty;
    
    // Skip threads outside the image bounds
    if (outputX >= width || outputY >= height) return;
    
    // Process each channel separately
    for (int c = 0; c < channels; c++) {
        // Define shared memory for this channel
        unsigned char* sharedData = &sharedMem[c * (blockDim.x + maskSize - 1) * (blockDim.y + maskSize - 1)];
        
        // Calculate shared memory dimensions
        int sharedWidth = blockDim.x + maskSize - 1;
        int sharedHeight = blockDim.y + maskSize - 1;
        
        // Each thread may need to load multiple elements into shared memory
        for (int dy = ty; dy < sharedHeight; dy += blockDim.y) {
            for (int dx = tx; dx < sharedWidth; dx += blockDim.x) {
                // Calculate global input position
                int inputX = bx * blockDim.x + dx - maskRadius;
                int inputY = by * blockDim.y + dy - maskRadius;
                
                // Check bounds and load or zero-pad
                if (inputX >= 0 && inputX < width && inputY >= 0 && inputY < height) {
                    sharedData[dy * sharedWidth + dx] = input[(inputY * width + inputX) * channels + c];
                } else {
                    sharedData[dy * sharedWidth + dx] = 0;
                }
            }
        }
        
        // Wait for all threads to finish loading shared memory
        __syncthreads();
        
        // Compute convolution result for this output pixel
        float sum = 0.0f;
        
        for (int ky = 0; ky < maskSize; ky++) {
            for (int kx = 0; kx < maskSize; kx++) {
                // Get position in shared memory - offset by our thread position
                int sharedX = tx + kx;
                int sharedY = ty + ky;
                
                // Get input pixel from shared memory
                unsigned char pixel = sharedData[sharedY * sharedWidth + sharedX];
                
                // Get the corresponding mask value
                float maskValue = c_mask[ky * maskSize + kx];
                
                // Accumulate weighted sum
                sum += static_cast<float>(pixel) * maskValue;
            }
        }
        
        // Clamp results to valid range [0, 255]
        sum = fminf(fmaxf(sum, 0.0f), 255.0f);
        
        // Write result to the output image
        output[(outputY * width + outputX) * channels + c] = static_cast<unsigned char>(sum);
        
        // Wait for all threads to finish using shared memory before next channel
        __syncthreads();
    }
}

// Function to read a 2D mask from file
void readMaskFile(const char* filename, float** mask, int* size) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening mask file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    
    // Read mask size (assuming square mask)
    fscanf(file, "%d", size);
    
    // Allocate memory for the mask
    *mask = (float*)malloc((*size) * (*size) * sizeof(float));
    
    // Read mask values
    for (int i = 0; i < (*size) * (*size); i++) {
        fscanf(file, "%f", &((*mask)[i]));
    }
    
    fclose(file);
}

// Load image data using stb_image.h (include this header at the top)
RGBImage loadImage(const char* filename) {
    RGBImage image;
    
    // Placeholder for loading image - you'll need to use a library like stb_image.h
    // For example:
    // image.data = stbi_load(filename, &image.width, &image.height, &image.channels, 3);
    
    printf("Loading image: %s\n", filename);
    
    // This is just a placeholder - replace with actual image loading code
    image.width = 512;   // Example width
    image.height = 512;  // Example height
    image.channels = 3;  // RGB image
    image.data = (unsigned char*)malloc(image.width * image.height * image.channels);
    
    if (!image.data) {
        fprintf(stderr, "Failed to allocate memory for image\n");
        exit(EXIT_FAILURE);
    }
    
    return image;
}

// Save image data using stb_image_write.h (include this header at the top)
void saveImage(const char* filename, RGBImage image) {
    // Placeholder for saving image - you'll need to use a library like stb_image_write.h
    // For example:
    // stbi_write_png(filename, image.width, image.height, image.channels, image.data, image.width * image.channels);
    
    printf("Saving image: %s (%dx%d, %d channels)\n", filename, image.width, image.height, image.channels);
    
    // This is just a placeholder - replace with actual image saving code
}

// Process a batch of images
void processBatch(const std::vector<std::string>& inputFiles, 
                 const std::string& outputFolder,
                 float* h_mask, int maskSize, int stride) {
    
    // Skip if batch is empty
    if (inputFiles.empty()) {
        return;
    }
    
    int batchSize = inputFiles.size();
    printf("Processing batch of %d images\n", batchSize);
    
    // First, load all images in the batch
    std::vector<RGBImage> images;
    std::vector<unsigned char*> d_inputs;
    std::vector<unsigned char*> d_outputs;
    
    for (const auto& inputFile : inputFiles) {
        RGBImage img = loadImage(inputFile.c_str());
        images.push_back(img);
        
        // Allocate device memory for this image
        unsigned char* d_input;
        unsigned char* d_output;
        cudaMalloc(&d_input, img.width * img.height * img.channels * sizeof(unsigned char));
        cudaMalloc(&d_output, img.width * img.height * img.channels * sizeof(unsigned char));
        
        // Copy image data to device
        cudaMemcpy(d_input, img.data, img.width * img.height * img.channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
        
        d_inputs.push_back(d_input);
        d_outputs.push_back(d_output);
    }
    
    // Copy mask to constant memory
    cudaMemcpyToSymbol(c_mask, h_mask, maskSize * maskSize * sizeof(float));
    
    // Process each image in the batch
    for (int i = 0; i < batchSize; i++) {
        // Define kernel launch parameters for output tiling
        // Block size equals output tile size (each thread produces one output)
        int blockWidth = 16;   // Output tile width
        int blockHeight = 16;  // Output tile height
        
        dim3 blockSize(blockWidth, blockHeight);
        
        // Calculate grid size based on output dimensions
        dim3 gridSize((images[i].width + blockWidth - 1) / blockWidth,
                     (images[i].height + blockHeight - 1) / blockHeight);
        
        // Calculate shared memory size (includes halo region)
        int sharedWidth = blockWidth + maskSize - 1;
        int sharedHeight = blockHeight + maskSize - 1;
        int sharedMemSize = images[i].channels * sharedWidth * sharedHeight * sizeof(unsigned char);
        
        // Launch convolution kernel with output tiling
        convolution3D_OutputTiling<<<gridSize, blockSize, sharedMemSize>>>(
            d_inputs[i], d_outputs[i],
            images[i].width, images[i].height, images[i].channels,
            maskSize, stride
        );
        
        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        }
    }
    
    // Wait for all kernels to complete
    cudaDeviceSynchronize();
    
    // Copy results back to host and save output images
    for (int i = 0; i < batchSize; i++) {
        // Get output filename
        std::string inputFile = inputFiles[i];
        size_t lastSlash = inputFile.find_last_of("/\\");
        std::string baseName = inputFile.substr(lastSlash + 1);
        std::string outputFile = outputFolder + "/" + baseName;
        
        // Copy processed image back to host
        cudaMemcpy(images[i].data, d_outputs[i], images[i].width * images[i].height * images[i].channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        
        // Save processed image
        saveImage(outputFile.c_str(), images[i]);
        
        // Free device memory
        cudaFree(d_inputs[i]);
        cudaFree(d_outputs[i]);
        
        // Free host memory
        free(images[i].data);
    }
}

// List all image files in a directory
std::vector<std::string> listImageFiles(const std::string& folderPath) {
    std::vector<std::string> files;
    
    // This is a placeholder - you should use platform-specific code to list files
    // For example, using dirent.h on Linux/macOS or FindFirstFile/FindNextFile on Windows
    
    printf("Listing image files in folder: %s\n", folderPath.c_str());
    
    return files;  // Return list of image files (empty in this placeholder)
}

int main(int argc, char* argv[]) {
    // Check command line arguments
    if (argc < 5 || argc > 6) {
        printf("Usage: %s <input_folder_path> <output_folder_path> <batch_size> <mask_file> [stride]\n", argv[0]);
        return -1;
    }
    
    // Parse command line arguments
    const char* inputFolder = argv[1];
    const char* outputFolder = argv[2];
    int batchSize = atoi(argv[3]);
    const char* maskFile = argv[4];
    int stride = (argc == 6) ? atoi(argv[5]) : 1;  // Default stride is 1
    
    // Print configuration
    printf("Input folder: %s\n", inputFolder);
    printf("Output folder: %s\n", outputFolder);
    printf("Batch size: %d\n", batchSize);
    printf("Mask file: %s\n", maskFile);
    printf("Stride: %d\n", stride);
    
    // Read mask file
    float* h_mask;
    int maskSize;
    readMaskFile(maskFile, &h_mask, &maskSize);
    printf("Mask size: %dx%d\n", maskSize, maskSize);
    
    // List all image files in the input folder
    std::vector<std::string> imageFiles = listImageFiles(inputFolder);
    
    // Process images in batches
    for (int i = 0; i < imageFiles.size(); i += batchSize) {
        // Create a batch of images (up to batchSize)
        std::vector<std::string> batch;
        for (int j = 0; j < batchSize && (i + j) < imageFiles.size(); j++) {
            batch.push_back(imageFiles[i + j]);
        }
        
        // Process this batch
        processBatch(batch, outputFolder, h_mask, maskSize, stride);
    }
    
    // Free host memory for mask
    free(h_mask);
    
    printf("Processing complete.\n");
    return 0;
}