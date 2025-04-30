#include "./prefixScan.cuh"
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

void check_exclusive(float* input, float* output, int n, float epsilon = 1e-3) {
    printf("Checking for errors...\n");
    float sum = 0;
    for (int i = 0; i < n; ++i){
        if (fabs(output[i] - sum) >= epsilon) {
            printf("gpu output has sum of %f in index %d\n", output[i], i);
            printf("while it should be %f\n", sum);
        }
        assert(fabs(output[i] - sum) < epsilon);
        sum += input[i];
    }
}

void check_inclusive(float* input, float* output, int n, float epsilon = 1e-3) {
    printf("Checking for errors...\n");
    float sum = 0;
    for (int i = 0; i < n; ++i){
        sum += input[i];
        if (fabs(output[i] - sum) >= epsilon) {
            printf("gpu output has sum of %f in index %d\n", output[i], i);
            printf("while it should be %f\n", sum);
        }
        assert(fabs(output[i] - sum) < epsilon);
    }
}

void allocateArray(float*& array, int n, float init_value = 0) {
    array = (float*)malloc(n * sizeof(float));
    if (array == NULL) {
        printf("Host memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; ++i) {
        array[i] = init_value;
    }
}

int main() {
    int n = 1024; // the amount of values in the array
    int numRuns = 100;
    int bytes = sizeof(float) * n;
    
    float *h_input, *h_output;
    float *d_input, *d_output;
    
    allocateArray(h_input, n, 0.5);  // Initialize with some value
    allocateArray(h_output, n);
    
    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));
    
    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    
    // For naive implementation, use just 1 block
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    float totalTime = 0;
    for (int i = 0; i < numRuns; i++) {
        CHECK_CUDA(cudaEventRecord(start));
        
        // Shared memory size is 2 * n * sizeof(float)
        prefixScan(d_input, d_output, n);
        
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        totalTime += milliseconds;
    }
    
    printf("Average execution time over %d runs: %f ms\n", numRuns, totalTime / numRuns);
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    
    // our scan kernel is exclusive
    check_inclusive(h_input, h_output, n);
    
    free(h_input);
    free(h_output);
    
    printf("Prefix scan completed successfully\n");
    
    return 0;
}