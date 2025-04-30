// main.cu
#include "./gaussian_blur_kernel.cuh"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s:%d, %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// CPU implementation of Gaussian blur for verification
void cpuGaussianBlur(float* input, float* output, int width, int height, float sigma) {
    int kernelRadius = (int)(3.0f * sigma);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            float weightSum = 0.0f;
            
            for (int ky = -kernelRadius; ky <= kernelRadius; ky++) {
                for (int kx = -kernelRadius; kx <= kernelRadius; kx++) {
                    int nx = x + kx;
                    int ny = y + ky;
                    
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        float weight = expf(-(kx*kx + ky*ky) / (2.0f * sigma * sigma));
                        sum += input[ny * width + nx] * weight;
                        weightSum += weight;
                    }
                }
            }
            
            output[y * width + x] = sum / weightSum;
        }
    }
}

// Function to check GPU results against CPU results
void check(float* input, float* gpu_output, int width, int height, float sigma, float epsilon = 1e-3) {
    printf("Checking for errors...\n");
    
    // Allocate memory for CPU calculation
    float* cpu_output = (float*)malloc(width * height * sizeof(float));
    if (cpu_output == NULL) {
        printf("Host memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    
    // Run CPU implementation
    cpuGaussianBlur(input, cpu_output, width, height, sigma);
    
    // Compare results
    int errors = 0;
    for (int i = 0; i < width * height; i++) {
        if (fabs(gpu_output[i] - cpu_output[i]) >= epsilon) {
            if (errors < 10) { // Limit the number of error messages
                printf("Mismatch at position %d: GPU = %f, CPU = %f\n", 
                      i, gpu_output[i], cpu_output[i]);
            }
            errors++;
        }
    }
    
    if (errors > 0) {
        printf("Found %d errors!\n", errors);
        assert(0); // Fail the test
    } else {
        printf("No errors found!\n");
    }
    
    free(cpu_output);
}

void allocateMatrix(float*& matrix, int size, float init_value = 0) {
    matrix = (float*)malloc(size * sizeof(float));
    if (matrix == NULL) {
        printf("Host memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < size; ++i) {
        matrix[i] = init_value;
    }
}

int main() {
    const int width = 1024;
    const int height = 1024;
    const float sigma = 2.0f;
    const int numRuns = 100;
    const int bytes = width * height * sizeof(float);
    
    float *h_input, *h_output;
    float *d_input, *d_output;
    
    // Allocate and initialize host memory
    allocateMatrix(h_input, width * height, 0.0f);
    allocateMatrix(h_output, width * height, 0.0f);
    
    // Initialize input with sample data (simple gradient)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            h_input[y * width + x] = (float)(x + y) / (width + height);
        }
    }
    
    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));
    
    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    
    // Define block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Run kernel multiple times and measure average performance
    float totalTime = 0;
    for (int i = 0; i < numRuns; i++) {
        CHECK_CUDA(cudaEventRecord(start));
        gaussianBlur<<<gridSize, blockSize>>>(d_input, d_output, width, height, sigma);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        totalTime += milliseconds;
    }
    printf("Average execution time over %d runs: %f ms\n", numRuns, totalTime / numRuns);
    
    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    
    // Verify results
    check(h_input, h_output, width, height, sigma);
    
    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    free(h_input);
    free(h_output);
    
    printf("Gaussian blur completed successfully\n");
    
    return 0;
}