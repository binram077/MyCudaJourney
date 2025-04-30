#ifndef VECTOR_ADD_KERNEL_H
#define VECTOR_ADD_KERNEL_H
#include <cuda_runtime.h>
__global__ void vectorAdd(int* a, int* b, int* c, int n);
#endif