#pragma once
#include "cuda_runtime.h"

template <int BlockLength>
__global__ void matmul(float* a, float* b, float* c, int n) {
    // first we calculate what block do we need to calculate the positions of our block
    int bRow = blockIdx.x;
    int bCol = blockIdx.y;

    // then we calculate the thread x and y with coalesced access
    int threadRow = threadIdx.x / BlockLength;
    int threadCol = threadIdx.x % BlockLength;

    // then we calculate the base indexes for our block
    // a shift starts at bRows * BlockLength lines(the n is for the lines) 
    int aShift = n * bRow * BlockLength;
    // b shift starts at bCol * BlockLength
    int bShift = bCol * BlockLength;
    // c block location is the sum of the bases
    int cBlockLocation = aShift + bShift;

    // now we will define the shared memory arrays
    __shared__ float As[BlockLength * BlockLength];
    __shared__ float Bs[BlockLength * BlockLength];

    // tmp accumulate the result for the final c
    float tmp = 0;

    for (int i = 0; i < (n + 1) / BlockLength; ++i) {
        if ((i * BlockLength + threadCol < n) && (i * BlockLength + threadRow < n)) {
            // first we will load the data to the smem(both accesses are coalesced)
            As[BlockLength * threadRow + threadCol] = a[aShift + n * threadRow + threadCol];
            Bs[BlockLength * threadRow + threadCol] = b[bShift + n * threadRow + threadCol];

            __syncthreads();

            for (int j = 0; j < BlockLength; ++j) {
                tmp += As[threadRow * BlockLength + j] * Bs[j * BlockLength + threadCol];
            }
        }
        
        // then we advance the shifts
        // a shift advance BlockLength
        aShift += BlockLength;
        // b shift advance BlockLength lines
        bShift += n * BlockLength;

        __syncthreads();
    }

    if ((bRow * BlockLength + threadRow < n) && (bCol * BlockLength + threadCol < n)) {
        c[cBlockLocation + threadRow * n + threadCol] = tmp;
    }
}