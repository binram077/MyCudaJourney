#include "./vector_add_kernel.cuh"

__global__ void vectorAdd(int* a, int* b, int* c, int n) {
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}