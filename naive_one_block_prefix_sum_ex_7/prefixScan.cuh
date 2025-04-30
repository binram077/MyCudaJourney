#pragma once
#include "cuda_runtime.h"
#include <cassert>

__global__ void prefixScan(float* input, float* output, int n) {
    int tid = threadIdx.x;

    // We will assign the length of the buffer in the invokation of the kernel
    extern __shared__ float doubleBuffer[];

    // the buffer will contain to buffers that will switch each iteration(one for reading and one for writing)
    // each iteration we will read from one and write into the other, idx 0 will be the first half and 1 the second
    int buffer_in_idx = 0, buffer_out_idx = 1; 

    // bound check
    if (tid < n){
        // as we are implementing exclusive scan we will shift the array on step to the right first
        doubleBuffer[buffer_in_idx * n + tid] = (tid > 0) ? input[tid - 1] : 0;
    }
    __syncthreads();

    for (int d = 1; d < n; d *= 2){
        doubleBuffer[buffer_out_idx * n + tid] =  doubleBuffer[buffer_in_idx * n + tid];
        if (tid >= d){
            doubleBuffer[buffer_out_idx * n + tid] += doubleBuffer[buffer_in_idx * n + tid - d];
        }

        // now we swap the buffers
        buffer_in_idx = 1 - buffer_in_idx;
        buffer_out_idx = 1 - buffer_out_idx;

        __syncthreads();
    }

    if (tid < n){
        // remember that we swapped the last buffer out with buffer in
        output[tid] = doubleBuffer[buffer_in_idx * n + tid];
    }
}