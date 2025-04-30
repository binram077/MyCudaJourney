#include "./matmul_kernel.cuh"
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

void check(float* a, float* b, float* c, int n, float epsilon = 1e-3) {
    printf("Checking for errors...\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float tmp = 0;
            for (int l = 0; l < n; ++l) {
                tmp += a[n * i + l] * b[n * l + j];
            }
            if (fabs(c[i * n + j] - tmp) >= epsilon) {
                printf("gpu c has %f\n", c[i * n + j]);
                printf("while it should be %f\n", tmp);
            }
            assert(fabs(c[i * n + j] - tmp) < epsilon);
        }
    }
}

void allocateMatrix(float*& matrix, int n, float init_value = 0) {
    matrix = (float*)malloc(n * n * sizeof(float));
    if (matrix == NULL) {
        printf("Host memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n * n; ++i) {
        matrix[i] = init_value;
    }
}

int main() {
    const int n = 1024;
    const int TileAWidth = 16, ThreadBlockHeight = 4, ThreadBlockWidth = 4;
    const int ThreadsPerRowOfRes = 16, ThreadsPerColOfRes = 16;
    const int numRuns = 100;
    int bytes = sizeof(float) * n * n;

    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    allocateMatrix(h_a, n, 0.03);
    allocateMatrix(h_b, n, 0.2);
    allocateMatrix(h_c, n);

    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // we should note that as we do a sort of block coalescing the x and y in the BlockID are transposed in the kernel
    dim3 gridSize(n / (ThreadBlockWidth * ThreadsPerColOfRes), n / (ThreadBlockHeight * ThreadsPerRowOfRes), 1);
    dim3 blockSize(ThreadsPerRowOfRes * ThreadsPerColOfRes, 1, 1);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float totalTime = 0;
    for (int i = 0; i < numRuns; i++) {
        CHECK_CUDA(cudaEventRecord(start));
        matmul<TileAWidth, ThreadBlockHeight, ThreadBlockWidth, ThreadsPerRowOfRes, ThreadsPerColOfRes><<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        totalTime += milliseconds;
    }
    printf("Average execution time over %d runs: %f ms\n", numRuns, totalTime / numRuns);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    check(h_a, h_b, h_c, n);

    free(h_a);
    free(h_b);
    free(h_c);

    printf("Matrix multiplication completed successfully\n");

    return 0;
}