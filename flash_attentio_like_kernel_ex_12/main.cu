#include "./flashAttention.cuh"
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

// CPU reference implementation for Flash Attention verification
void cpu_attention(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    const float scale = 1.0f / sqrtf(d);
    const float c = 5.0f;
    
    for (int i = 0; i < M; i++) {
        // Calculate attention scores for query i
        float* scores = (float*)malloc(N * sizeof(float));
        float max_score = -INFINITY;
        
        // Compute Q*K^T for this query
        for (int j = 0; j < N; j++) {
            scores[j] = 0.0f;
            for (int k = 0; k < d; k++) {
                scores[j] += Q[i * d + k] * K[j * d + k];
            }
            scores[j] *= scale;
            max_score = fmaxf(max_score, scores[j]);
        }
        
        // Apply softmax with numerical stability
        float sum_exp = 0.0f;
        for (int j = 0; j < N; j++) {
            scores[j] = expf(scores[j] - max_score + c);
            sum_exp += scores[j];
        }
        
        // Normalize and compute weighted sum with values
        for (int k = 0; k < d; k++) {
            output[i * d + k] = 0.0f;
            for (int j = 0; j < N; j++) {
                float attention_weight = scores[j] / sum_exp;
                output[i * d + k] += attention_weight * V[j * d + k];
            }
        }
        
        free(scores);
    }
}

void check_flash_attention(const float* Q, const float* K, const float* V, 
                          const float* gpu_output, int M, int N, int d, float epsilon = 1e-2) {
    printf("Checking Flash Attention results...\n");
    
    // Allocate memory for CPU reference
    float* cpu_output = (float*)malloc(M * d * sizeof(float));
    if (cpu_output == NULL) {
        printf("CPU memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    
    // Compute reference on CPU
    cpu_attention(Q, K, V, cpu_output, M, N, d);
    
    // Compare results
    int errors = 0;
    float max_error = 0.0f;
    for (int i = 0; i < M * d; i++) {
        float error = fabsf(gpu_output[i] - cpu_output[i]);
        max_error = fmaxf(max_error, error);
        if (error >= epsilon) {
            if (errors < 10) { // Only print first 10 errors
                printf("Error at index %d: GPU = %f, CPU = %f, diff = %f\n", 
                       i, gpu_output[i], cpu_output[i], error);
            }
            errors++;
        }
    }
    
    printf("Total errors: %d/%d, Max error: %f\n", errors, M * d, max_error);
    if (errors == 0) {
        printf("All results match within epsilon=%f\n", epsilon);
    } else {
        printf("Warning: %d values exceed tolerance\n", errors);
    }
    
    free(cpu_output);
}

void allocateMatrix(float*& matrix, int rows, int cols, float init_value = 0.0f) {
    int size = rows * cols;
    matrix = (float*)malloc(size * sizeof(float));
    if (matrix == NULL) {
        printf("Host memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    
    // Initialize with random values or specific pattern
    for (int i = 0; i < size; i++) {
        if (init_value == 0.0f) {
            // Random initialization between -1 and 1
            matrix[i] = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
        } else {
            matrix[i] = init_value;
        }
    }
}

int main() {
    // Problem dimensions
    int M = 128;  // Number of queries
    int N = 256;  // Number of keys/values  
    int d = 64;   // Embedding dimension
    int numRuns = 50;
    
    printf("Flash Attention dimensions: M=%d, N=%d, d=%d\n", M, N, d);
    
    // Host memory allocation
    float *h_Q, *h_K, *h_V, *h_output;
    float *d_Q, *d_K, *d_V, *d_output;
    
    // Allocate and initialize host matrices
    allocateMatrix(h_Q, M, d);  // Queries
    allocateMatrix(h_K, N, d);  // Keys
    allocateMatrix(h_V, N, d);  // Values
    allocateMatrix(h_output, M, d, 0.0f);  // Output (initialize to zero)
    
    // Allocate device memory
    int q_bytes = M * d * sizeof(float);
    int kv_bytes = N * d * sizeof(float);
    int output_bytes = M * d * sizeof(float);
    
    CHECK_CUDA(cudaMalloc(&d_Q, q_bytes));
    CHECK_CUDA(cudaMalloc(&d_K, kv_bytes));
    CHECK_CUDA(cudaMalloc(&d_V, kv_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, output_bytes));
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_Q, h_Q, q_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K, kv_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V, kv_bytes, cudaMemcpyHostToDevice));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Warm up
    solve(d_Q, d_K, d_V, d_output, M, N, d);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark
    float totalTime = 0;
    for (int i = 0; i < numRuns; i++) {
        CHECK_CUDA(cudaEventRecord(start));
        
        // Call Flash Attention
        solve(d_Q, d_K, d_V, d_output, M, N, d);
        
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        totalTime += milliseconds;
    }
    
    printf("Average execution time over %d runs: %f ms\n", numRuns, totalTime / numRuns);
    
    // Calculate theoretical FLOPS
    // Attention computation: M*N*d (QK^T) + M*N*d (softmax*V) ≈ 2*M*N*d
    long long flops = 2LL * M * N * d;
    double gflops = (double)flops / (totalTime / numRuns * 1e6); // Convert ms to seconds
    printf("Performance: %.2f GFLOPS\n", gflops);
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost));
    
    // Verify results against CPU implementation
    check_flash_attention(h_Q, h_K, h_V, h_output, M, N, d);
    
    // Free device memory
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_output));
    
    // Free host memory
    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_output);
    
    printf("Flash Attention completed successfully\n");
    
    return 0;
}