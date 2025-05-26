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

void check_cumsumprod(const float* A, const float* B, const float* output, int n, float epsilon = 1e-3) {
    printf("Checking for errors...\n");
    
    // For sequential verification, we need to track running sums and products
    float runningSum = 0.0f;  // Equivalent to the combined sum across all elements
    
    for (int i = 0; i < n; ++i) {
        // Calculate the expected value:
        // B[i] (current value) + runningSum (sum of all previous elements)
        runningSum = B[i] + runningSum * A[i];
        
        if ((fabs(output[i] - runningSum) >= epsilon)) {
            printf("Error at index %d: GPU output = %f, Expected = %f\n", i, output[i], runningSum);
            assert(fabs(output[i] - runningSum) < epsilon);
        }
    }
    printf("All results match within epsilon=%f\n", epsilon);
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
    int n = 1000000; // the amount of values in the array
    int numRuns = 100;
    int bytes = sizeof(float) * n;
    
    float *h_A, *h_B, *h_output;
    float *d_A, *d_B, *d_output;
    
    // Initialize arrays
    allocateArray(h_A, n, 0.99f);  // Values for multiplication
    allocateArray(h_B, n, 0.5f);   // Values for addition
    allocateArray(h_output, n);
    
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));
    
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    float totalTime = 0;
    for (int i = 0; i < numRuns; i++) {
        CHECK_CUDA(cudaEventRecord(start));
        
        // Call the cumsumprod algorithm
        cumsumprodScan(d_A, d_B, d_output, n);
        
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
    
    // Verify results
    check_cumsumprod(h_A, h_B, h_output, n);
    
    // Free memory
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_output));
    
    free(h_A);
    free(h_B);
    free(h_output);
    
    printf("CumSumProd completed successfully\n");
    
    return 0;
}