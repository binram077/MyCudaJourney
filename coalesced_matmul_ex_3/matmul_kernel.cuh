#ifndef MATMUL_KERNEL_H
#define MATMUL_KERNEL_H
#include <cuda_runtime.h>
__global__ void matmul(float* a, float* b, float* c, int n);
#endif