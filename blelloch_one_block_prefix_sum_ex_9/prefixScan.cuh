#pragma once
#include "cuda_runtime.h"
#include <cassert>

__global__ void prefixScan(float* input, float* output, int n) {
    const float identity = 0;
    int tid = threadIdx.x;

    // this assertion checks whether n is power of 2
    assert ((n & (n-1)) == 0 && n > 0);

    // We will assign the length of the buffer in the invokation of the kernel
    extern __shared__ float sharedMem[];
    
    if (tid < n){
        sharedMem[tid] = input[tid];
    }

    // first we would do the up sweep
    for (int offset = 1; offset * 2 <= n; offset *= 2){
        if ((tid - offset >= 0) && (tid < n) && ((tid + 1) % (2 * offset) == 0)){
            sharedMem[tid] += sharedMem[tid - offset];
        }
        __syncthreads();
    }

    // now we will set the last value to identity
    sharedMem[n - 1] = identity;

    // now down-sweep
    for (int offset = n / 2; offset > 0; offset >>= 1){
        if ((tid - offset >= 0) && (tid < n) && ((tid + 1) % (2 * offset) == 0)){
            float tmp = sharedMem[tid - offset];
            sharedMem[tid - offset] = sharedMem[tid];
            sharedMem[tid] += tmp;
        }
        __syncthreads();
    }

    if (tid < n){
        // remember that we swapped the last buffer out with buffer in
        output[tid] = sharedMem[tid];
    }
}