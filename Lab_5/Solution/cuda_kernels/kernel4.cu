#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <string>
#include <vector>
#include <cstring>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#include "stb_image/stb_image_write.h"
#include <windows.h>
#include <algorithm>
// Define constant memory for mask
__constant__ float c_mask[1024]; // Assuming mask won't exceed 1024 elements

// Structure to store RGB image data
typedef struct
{
    unsigned char *data;
    int width;
    int height;
    int channels;
} RGBImage;

// New kernel that processes multiple images at once
__global__ void convolution3D_BatchedSingleChannel(
    unsigned char **inputs, // Array of pointers to input images
    float **outputs,        // Array of pointers to output images
    int * d_inputWidths,              // Assuming all images have same dimensions
    int * d_inputHeights,
    int channels,
    int maskSize,
    int stride,
    int batchSize) // Number of images in the batch
{
    // Calculate strided 2D position within the output image
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int imageIdx = blockIdx.z; // Each z-block handles one image in the batch

    // Calculate input image position by multiplying by stride
    int in_x = out_x * stride;
    int in_y = out_y * stride;
    int width = d_inputWidths[imageIdx];
    int height = d_inputHeights[imageIdx];
    // Calculate output dimensions based on stride
    int  outWidth = (width - maskSize) / stride + 1;
    int outHeight = (height - maskSize) / stride + 1;

    // Skip if we're outside the output image bounds or batch size
    if (out_x >=  outWidth || out_y >= outHeight || imageIdx >= batchSize)
        return;

    int maskRadius = maskSize / 2;
    float aggregateValue = 0.0f;

    // Get pointers for this specific image
    unsigned char *input = inputs[imageIdx];
    float *output = outputs[imageIdx];

    // Process each channel (R, G, B) and combine into a single output value
    for (int c = 0; c < channels; c++)
    {
        // Apply convolution mask for this channel
        for (int ky = 0; ky < maskSize; ky++)
        {
            for (int kx = 0; kx < maskSize; kx++)
            {
                // Calculate input image coordinates with padding handling
                int ix = in_x + kx - maskRadius;
                int iy = in_y + ky - maskRadius;

                // Skip positions outside the image - zero padding
                if (ix >= 0 && ix < width && iy >= 0 && iy < height)
                {
                    // Get input pixel value for this channel
                    unsigned char pixel = input[(iy * width + ix) * channels + c];

                    // Get the corresponding mask value
                    float maskValue = c_mask[ky * maskSize + kx];

                    // Accumulate weighted sum (one calculation)
                    aggregateValue += static_cast<float>(pixel) * maskValue;
                }
            }
        }
    }

    // Write the single-channel result to the output image
    output[out_y * outWidth + out_x] = aggregateValue;
}

// Kernel to find minimum and maximum values in the output image
__global__ void findMinMax(float *image, int size, float *minValue, float *maxValue)
{
    extern __shared__ float sharedMem[];
    float *sharedMin = sharedMem;
    float *sharedMax = sharedMem + blockDim.x;

    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory
    sharedMin[tid] = globalIdx < size ? static_cast<float>(image[globalIdx]) : 255.0f;
    sharedMax[tid] = globalIdx < size ? static_cast<float>(image[globalIdx]) : 0.0f;

    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sharedMin[tid] = fminf(sharedMin[tid], sharedMin[tid + s]);
            sharedMax[tid] = fmaxf(sharedMax[tid], sharedMax[tid + s]);
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0)
    {
        minValue[blockIdx.x] = sharedMin[0];
        maxValue[blockIdx.x] = sharedMax[0];
    }
}

// Kernel to normalize image values based on min/max
__global__ void normalizeImage(
    float *image,
    int size,
    float minValue,
    float maxValue)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;

    // // Get current pixel value
    // float pixelValue = static_cast<float>(image[idx]);

    // // Avoid division by zero
    // float range = maxValue - 0.001f;

    // Normalize to [0, 255] range with strict clamping
    // Handle positive and negative values differently
    float pixelValue = image[idx];
    if (pixelValue < 0.0f)
    {
        // For negative values, clamp to 0
        pixelValue = 0.0f;
    }

    image[idx] = ((pixelValue) / maxValue) * 255.0f;
    ;

    // // Ensure strict adherence to 0-255 range
    // if (normalized < 0.0f)
    //     normalized = 0.0f;
    // if (normalized > 255.0f)
    //     normalized = 255.0f;

    // image[idx] = static_cast<unsigned char>(normalized);
}

// Function to perform min/max normalization on a single-channel image
void normalizeImageMinMax(float *d_image, int width, int height)
{
    int size = width * height;

    // Calculate grid and block dimensions for min/max kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate device memory for intermediate min/max values
    float *d_minValues, *d_maxValues;
    cudaMalloc(&d_minValues, blocksPerGrid * sizeof(float));
    cudaMalloc(&d_maxValues, blocksPerGrid * sizeof(float));

    // Find min/max values
    findMinMax<<<blocksPerGrid, threadsPerBlock, 2 * threadsPerBlock * sizeof(float)>>>(
        d_image, size, d_minValues, d_maxValues);

    // Allocate host memory for min/max results
    float *h_minValues = (float *)malloc(blocksPerGrid * sizeof(float));
    float *h_maxValues = (float *)malloc(blocksPerGrid * sizeof(float));

    // Copy results back to host
    cudaMemcpy(h_minValues, d_minValues, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_maxValues, d_maxValues, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    // Find global min/max
    float globalMin = h_minValues[0];
    float globalMax = h_maxValues[0];
    for (int i = 1; i < blocksPerGrid; i++)
    {
        globalMin = fminf(globalMin, h_minValues[i]);
        globalMax = fmaxf(globalMax, h_maxValues[i]);
    }

    // printf("Image min: %f, max: %f\n", globalMin, globalMax);

    // Now normalize the image based on min/max
    normalizeImage<<<blocksPerGrid, threadsPerBlock>>>(d_image, size, globalMin, globalMax);

    // Free allocated memory
    free(h_minValues);
    free(h_maxValues);
    cudaFree(d_minValues);
    cudaFree(d_maxValues);
}

// Function to read a 2D mask from file
void readMaskFile(const char *filename, float **mask, int *size)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        fprintf(stderr, "Error opening mask file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Read mask size (assuming square mask)
    fscanf(file, "%d", size);

    // Allocate memory for the mask
    *mask = (float *)malloc((*size) * (*size) * sizeof(float));

    // Read mask values
    for (int i = 0; i < (*size) * (*size); i++)
    {
        fscanf(file, "%f", &((*mask)[i]));
    }

    fclose(file);
}

// Load image data using stb_image.h
RGBImage loadImage(const char *filename)
{
    RGBImage image;
    int width, height, channels;
    
    // Load image with stb_image
    unsigned char *data = stbi_load(filename, &width, &height, &channels, 3);
    
    if (!data)
    {
        fprintf(stderr, "Failed to load image '%s': %s\n", filename, stbi_failure_reason());
        exit(EXIT_FAILURE);
    }
    
    // Set image properties
    image.width = width;
    image.height = height;
    image.channels = 3; // Force 3 channels (RGB)
    image.data = data;  // Use the buffer directly instead of copying
    
    printf("Loaded %s (%dx%d with %d channels)\n", filename, width, height, channels);
    
    return image;
}

// Save image data as single-channel grayscale PNG
void saveImage(const char *filename, RGBImage image)
{
    // Make sure we're using exactly 1 channel
    const int channels = 1;

    // Calculate stride (bytes per row)
    const int stride = image.width * channels;

    // Save as PNG (which handles grayscale well)
    int success = stbi_write_png(filename, image.width, image.height, channels,
                                 image.data, stride);

    if (!success)
    {
        fprintf(stderr, "Failed to save image '%s'\n", filename);
    }
}

// Process a batch of images
void processBatch(const std::vector<std::string> &inputFiles,
                  const std::string &outputFolder,
                  float *h_mask, int maskSize, int stride)
{

    // Skip if batch is empty
    if (inputFiles.empty())
    {
        return;
    }

    int batchSize = inputFiles.size();
    printf("Processing batch of %d images\n", batchSize);

    // First, load all images in the batch
    std::vector<RGBImage> inputImages;
    std::vector<RGBImage> outputImages;
    std::vector<unsigned char *> d_inputs;
    std::vector<float *> d_outputs;
    // Arrays to store dimensions for each image
    std::vector<int> inputWidths;
    std::vector<int> inputHeights;
    std::vector<int> outputWidths;
    std::vector<int> outputHeights;
    // Load all images and store thier dimensions so that i can use different dimensions for each image
    for (const auto &inputFile : inputFiles)
    {
        RGBImage img = loadImage(inputFile.c_str());
        inputImages.push_back(img);

        // Store the dimensions
        inputWidths.push_back(img.width);
        inputHeights.push_back(img.height);

        // Calculate output dimensions based on stride
        int outWidth = (img.width - maskSize) / stride + 1;
        int outHeight = (img.height - maskSize) / stride + 1;
        outputWidths.push_back(outWidth);
        outputHeights.push_back(outHeight);
        // Create single-channel output image structure
        RGBImage outImg;
        outImg.width = outWidth;
        outImg.height = outHeight;
        outImg.channels = 1; // Grayscale output
        outImg.data = (unsigned char *)malloc(outWidth * outHeight * sizeof(unsigned char));
        outputImages.push_back(outImg);

        // Allocate device memory
        unsigned char *d_input;
        float *d_output;
        cudaMalloc(&d_input, img.width * img.height * img.channels * sizeof(unsigned char));
        cudaMalloc(&d_output, outWidth * outHeight * sizeof(float));

        // Copy image data to device
        cudaMemcpy(d_input, img.data, img.width * img.height * img.channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
        
        d_inputs.push_back(d_input);
        d_outputs.push_back(d_output);
    }
    // Allocate device memory for dimension arrays
    int *d_inputWidths, *d_inputHeights;
    cudaMalloc(&d_inputWidths, batchSize * sizeof(int));
    cudaMalloc(&d_inputHeights, batchSize * sizeof(int));
    // Copy dimension data to the device
    cudaMemcpy(d_inputHeights, inputHeights.data(), batchSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputWidths, inputWidths.data(), batchSize * sizeof(int), cudaMemcpyHostToDevice);
   

    // Create device arrays to hold pointers to all images
    unsigned char **d_inputPtrs;
    float **d_outputPtrs; // Changed to float**
    cudaMalloc(&d_inputPtrs, batchSize * sizeof(unsigned char *));
    cudaMalloc(&d_outputPtrs, batchSize * sizeof(float *)); // Changed to float*

    // Copy the arrays of pointers to the device
    cudaMemcpy(d_inputPtrs, d_inputs.data(), batchSize * sizeof(unsigned char *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outputPtrs, d_outputs.data(), batchSize * sizeof(float *), cudaMemcpyHostToDevice); // Changed to float*

    // Copy mask to constant memory
    cudaMemcpyToSymbol(c_mask, h_mask, maskSize * maskSize * sizeof(float));

    // Calculate output dimensions based on stride
    int outWidth = (inputImages[0].width - maskSize) / stride + 1;
    int outHeight = (inputImages[0].height - maskSize) / stride + 1;

    // Find maximum dimensions to size grid properly
    int maxOutHeight = *std::max_element(outputHeights.begin(), outputHeights.end());
    int maxOutWidth = *std::max_element(outputWidths.begin(), outputWidths.end());
    
    // Define kernel launch parameters
    dim3 blockSize(16, 16, 1);
    dim3 gridSize(
        (maxOutWidth + blockSize.x - 1) / blockSize.x,
        (maxOutHeight + blockSize.y - 1) / blockSize.y,
        batchSize);

    // Process all images with a single kernel launch
    convolution3D_BatchedSingleChannel<<<gridSize, blockSize>>>(
        d_inputPtrs, d_outputPtrs,
        d_inputWidths, d_inputHeights, inputImages[0].channels,
         
        maskSize, stride, batchSize);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    // Apply normalization to each image
    for (int i = 0; i < batchSize; i++)
    {
        // int outWidth = (inputImages[i].width - maskSize) / stride + 1;
        // int outHeight = (inputImages[i].height - maskSize) / stride + 1;
        normalizeImageMinMax(d_outputs[i],outputWidths[i], outputHeights[i]);
    }

    // Copy results back to host and save output images
    for (int i = 0; i < batchSize; i++)
    {
        // Get output filename
        std::string inputFile = inputFiles[i];
        size_t lastSlash = inputFile.find_last_of("/\\");
        std::string baseName = inputFile.substr(lastSlash + 1);
        std::string outputFile = outputFolder + "/" + baseName;

        float *h_floatOutput = (float *)malloc(outputWidths[i] * outputHeights[i] * sizeof(float));
        cudaMemcpy(h_floatOutput, d_outputs[i],
            outputWidths[i] * outputHeights[i]* sizeof(float),
                   cudaMemcpyDeviceToHost);

        //    // Print some statistics about the float values
        //    float minVal = h_floatOutput[0], maxVal = h_floatOutput[0];
        //    for (int j = 1; j < inputImages[i].width * inputImages[i].height; j++) {
        //        if (h_floatOutput[j] < minVal) minVal = h_floatOutput[j];
        //        if (h_floatOutput[j] > maxVal) maxVal = h_floatOutput[j];
        //    }
        //    printf("Image %d min value: %.2f, max value: %.2f\n", i, minVal, maxVal);

        //    // Convert float to unsigned char for saving
        //    for (int j = 0; j < inputImages[i].width * inputImages[i].height; j++) {
        //        // Normalize to 0-255 range
        //        float normalized = 255.0f * (h_floatOutput[j] - minVal) / (maxVal - minVal);
        //        outputImages[i].data[j] = static_cast<unsigned char>(fminf(fmaxf(normalized, 0.0f), 255.0f));
        //    }

        // Convert float values to unsigned char for output image
        for (int j = 0; j <  outputWidths[i] *  outputHeights[i]; j++)
        {
            float value = h_floatOutput[j];
            outputImages[i].data[j] = static_cast<unsigned char>(value);
        }
        // Save processed image (single channel grayscale)
        saveImage(outputFile.c_str(), outputImages[i]);

        // Free device and host memory
        cudaFree(d_inputs[i]);
        cudaFree(d_outputs[i]);
        free(h_floatOutput);
        free(inputImages[i].data);
        free(outputImages[i].data);
    }

    // Free device pointer arrays
    cudaFree(d_inputPtrs);
    cudaFree(d_outputPtrs);
    cudaFree(d_inputWidths);
    cudaFree(d_inputHeights);
}

// List all image files in a directory
std::vector<std::string> listImageFiles(const std::string &folderPath)
{
    std::vector<std::string> files;

    printf("Listing image files in folder: %s\n", folderPath.c_str());

    // Create search pattern for all files in the directory
    std::string searchPath = folderPath + "\\*";

    // Windows file finding structures
    WIN32_FIND_DATAA findFileData;
    HANDLE hFind = FindFirstFileA(searchPath.c_str(), &findFileData);

    if (hFind == INVALID_HANDLE_VALUE)
    {
        fprintf(stderr, "Error opening directory: %s (error code: %lu)\n",
                folderPath.c_str(), GetLastError());
        return files;
    }

    // List of supported image extensions
    const std::vector<std::string> supportedExtensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tga", ".gif"};

    do
    {
        // Skip "." and ".." directories
        if (strcmp(findFileData.cFileName, ".") == 0 || strcmp(findFileData.cFileName, "..") == 0)
        {
            continue;
        }

        // Skip directories
        if (findFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
        {
            continue;
        }

        // Check if the file has a supported image extension
        std::string filename = findFileData.cFileName;
        std::string fileExtension = "";

        size_t dotPos = filename.find_last_of(".");
        if (dotPos != std::string::npos)
        {
            fileExtension = filename.substr(dotPos);

            // Convert extension to lowercase for comparison
            for (char &c : fileExtension)
            {
                c = tolower(c);
            }

            // Check if this is a supported image file
            bool isSupported = false;
            for (const auto &ext : supportedExtensions)
            {
                if (fileExtension == ext)
                {
                    isSupported = true;
                    break;
                }
            }

            if (isSupported)
            {
                // Add full path to the list
                std::string fullPath = folderPath + "\\" + filename;
                files.push_back(fullPath);
                // printf("Found image file: %s\n", fullPath.c_str());
            }
        }

    } while (FindNextFileA(hFind, &findFileData) != 0);

    // Check for errors
    DWORD error = GetLastError();
    if (error != ERROR_NO_MORE_FILES)
    {
        fprintf(stderr, "Error reading directory: %s (error code: %lu)\n",
                folderPath.c_str(), error);
    }

    // Close the find handle
    FindClose(hFind);

    printf("Found %zu image files in total\n", files.size());
    return files;
}

int main(int argc, char *argv[])
{
    // Check command line arguments
    if (argc < 5 || argc > 6)
    {
        printf("Usage: %s <input_folder_path> <output_folder_path> <batch_size> <mask_file> [stride]\n", argv[0]);
        return -1;
    }

    // Parse command line arguments
    const char *inputFolder = argv[1];
    const char *outputFolder = argv[2];
    int batchSize = atoi(argv[3]);
    const char *maskFile = argv[4];
    int stride = (argc == 6) ? atoi(argv[5]) : 1; // Default stride is 1

    // Print configuration
    printf("Input folder: %s\n", inputFolder);
    printf("Output folder: %s\n", outputFolder);
    printf("Batch size: %d\n", batchSize);
    printf("Mask file: %s\n", maskFile);
    printf("Stride: %d\n", stride);

    // Read mask file
    float *h_mask;
    int maskSize;
    readMaskFile(maskFile, &h_mask, &maskSize);
    printf("Mask size: %dx%d\n", maskSize, maskSize);

    // List all image files in the input folder
    std::vector<std::string> imageFiles = listImageFiles(inputFolder);

    // Process images in batches
    for (int i = 0; i < imageFiles.size(); i += batchSize)
    {
        // Create a batch of images (up to batchSize)
        std::vector<std::string> batch;
        for (int j = 0; j < batchSize && (i + j) < imageFiles.size(); j++)
        {
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