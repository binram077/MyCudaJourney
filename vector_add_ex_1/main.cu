#include "./vector_add_kernel.cuh"
#include<device_launch_parameters.h>
#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<assert.h>

void check(int* a, int* b, int* c, int n) {
    for (int i = 0; i < n; ++i) {
        assert (a[i] + b[i] == c[i]);
    }
}

void allocateVector(int* &vector, int n, int init_value = 0) {
    vector = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; ++i) {
        vector[i] = init_value;
    }
}

int main() {
    int n = 256;

    int bytes = sizeof(int) * n;

    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;

    allocateVector(h_a, n, 3);
    allocateVector(h_b, n, 2);
    allocateVector(h_c, n);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int gridSize = 1;
    int blockSize = 256;

    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    check(h_a, h_b, h_c, n);
    
    free(h_a);
    free(h_b);
    free(h_c);

    printf("Good for you");

    return 0;
}