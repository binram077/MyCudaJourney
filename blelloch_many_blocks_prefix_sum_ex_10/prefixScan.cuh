#pragma once
#include "cuda_runtime.h"
#include <stdio.h>
#include <cassert>

__global__ void scanKernel(const float* input, float* output, float* blockSums, int N, int numWorksPerThread){
    const int tid = threadIdx.x;
    const int smemShuffledTid = tid ^ (tid >> 5); // in order to solve bank conflict we shuffle the thread indexes
    extern __shared__ float smem[];
    const float identity = 0.0f;
    float sum = identity;
    int iterationBlockSize = blockDim.x * 2;

    for (int i = 0; i < numWorksPerThread; ++i){
        // define the starting point of the block we calculate
        int iterationBlockStart = blockIdx.x * (iterationBlockSize * numWorksPerThread) + i * iterationBlockSize;
        int iterationBlockEnd = iterationBlockStart + iterationBlockSize;
        
        // loading to smem
        int tIdx = iterationBlockStart + tid;

        // bound checking
        if (tIdx < N) {
            smem[tid] = input[tIdx];
        }
        else {
            smem[tid] = identity;
        }
        if (tIdx + blockDim.x < N) {
            smem[tid + blockDim.x] = input[tIdx + blockDim.x];
        }
        else {
            smem[tid + blockDim.x] = identity;
        }
        __syncthreads();

        // up sweep
        for (int d = iterationBlockSize >> 1, offset = 1; d > 0; d >>= 1, offset <<= 1){
            if (smemShuffledTid < d){
                int leftIdx = offset * (2 * smemShuffledTid + 1) - 1;
                int rightIdx = offset * (2 * smemShuffledTid + 2) - 1;
                smem[rightIdx] = smem[leftIdx] + smem[rightIdx];
            }
            __syncthreads();
        }

        // Save the total sum before setting last element to identity
        float blockTotal = smem[iterationBlockSize - 1];
        __syncthreads();

        // setting last element to identity
        if (tid == 0){
            smem[iterationBlockSize - 1] = 0;
        }
        __syncthreads();

        // down sweep
        for (int offset = iterationBlockSize >> 1, d = 1; offset > 0; offset >>= 1, d <<= 1){
            if (smemShuffledTid < d){
                int leftIdx = offset * (2 * smemShuffledTid + 1) - 1;
                int rightIdx = offset * (2 * smemShuffledTid + 2) - 1;
                float tmp = smem[leftIdx];
                smem[leftIdx] = smem[rightIdx];
                smem[rightIdx] = tmp + smem[rightIdx];
            }
            __syncthreads();
        }

        // offloading
        if (tIdx < N) {
            output[tIdx] = input[tIdx] + smem[tid] + sum;
        }
        if (tIdx + blockDim.x < N) {
            output[tIdx + blockDim.x] = input[tIdx + blockDim.x] + smem[tid + blockDim.x] + sum;
        }

        // adding the sum of the last block to the total sum
        sum += blockTotal;

        __syncthreads();
    }

    if (tid == 0) {
        blockSums[blockIdx.x] = sum;
    }
}

__global__ void scanKernelInplaceOneBlock(float* input, int N, int numWorksPerThread){
    extern __shared__ float smem[];
    const float identity = 0.0f;
    const int tid = threadIdx.x;
    float sum = identity;
    int iterationBlockSize = blockDim.x * 2;
    
    assert (blockIdx.x == 0);
    int numIteration = 0;
    for (int i = 0; i < numWorksPerThread; ++i){
        // define the starting point of the block we calculate
        int iterationBlockStart = blockIdx.x * (iterationBlockSize * numWorksPerThread) + i * iterationBlockSize;
        int iterationBlockEnd = iterationBlockStart + iterationBlockSize;
        
        // loading to smem
        int tIdx = iterationBlockStart + threadIdx.x;

        // bound checking
        if (tIdx < N) {
            smem[tid] = input[tIdx];
        }
        else {
            smem[tid] = identity;
        }
        if (tIdx + blockDim.x < N) {
            smem[tid + blockDim.x] = input[tIdx + blockDim.x];
        }
        else {
            smem[tid + blockDim.x] = identity;
        }
        __syncthreads();

        // up sweep
        for (int d = iterationBlockSize >> 1, offset = 1; d > 0; d >>= 1, offset <<= 1){
            if (tid < d){
                int leftIdx = offset * (2 * tid + 1) - 1;
                int rightIdx = offset * (2 * tid + 2) - 1;
                smem[rightIdx] = smem[leftIdx] + smem[rightIdx];
            }
            __syncthreads();
        }

        // Save the total sum before setting last element to identity
        float blockTotal = smem[iterationBlockSize - 1];
        __syncthreads();

        // setting last element to identity
        if (tid == 0){
            smem[iterationBlockSize - 1] = 0;
        }
        __syncthreads();

        // down sweep
        for (int offset = iterationBlockSize >> 1, d = 1; offset > 0; offset >>= 1, d <<= 1){
            if (tid < d){
                int leftIdx = offset * (2 * tid + 1) - 1;
                int rightIdx = offset * (2 * tid + 2) - 1;
                float tmp = smem[leftIdx];
                smem[leftIdx] = smem[rightIdx];
                smem[rightIdx] = tmp + smem[rightIdx];
            }
            __syncthreads();
        }

        // offloading
        if (tIdx < N) {
            input[tIdx] = input[tIdx] + smem[tid] + sum;
        }
        if (tIdx + blockDim.x < N) {
            input[tIdx + blockDim.x] = input[tIdx + blockDim.x] + smem[tid + blockDim.x] + sum;
        }

        // adding the sum of the last block to the total sum
        sum += blockTotal;

        __syncthreads();
    }
}

__global__ void addBlockSumsKernel(float* output, float* scannedBlockSums, int N, int numWorksPerThread) {
    int numValuesCalculatedByBlock = 2 * blockDim.x * numWorksPerThread;
    int idx = blockIdx.x * numValuesCalculatedByBlock + threadIdx.x;
    
    for (int i = 0; i < numWorksPerThread && idx < N && blockIdx.x > 0; ++i) {
        int currentIdx = idx + 2 * i * blockDim.x;
        if (currentIdx < N) {
            output[currentIdx] += scannedBlockSums[blockIdx.x - 1];
        }
        if (currentIdx + blockDim.x < N) {
            output[currentIdx + blockDim.x] += scannedBlockSums[blockIdx.x - 1];
        }
    }
}

void prefixScan(const float* input, float* output, int N) {
    if (N <= 0) return;

    const int THREADS_PER_BLOCK = 256;
    const int BLOCK_SIZE = THREADS_PER_BLOCK * 2;
    const int NUM_WORKS_PER_BLOCK = 1; // Tunable parameter
    const int ELEMENTS_PER_BLOCK_TOTAL = BLOCK_SIZE * NUM_WORKS_PER_BLOCK;
    const size_t sharedMemSize = BLOCK_SIZE * sizeof(float);

    const int numBlocks = (N + ELEMENTS_PER_BLOCK_TOTAL - 1) / ELEMENTS_PER_BLOCK_TOTAL;

    float* d_blockSums = nullptr;
    
    // TODO: Add error checking for cudaMalloc and kernel launches
    cudaMalloc(&d_blockSums, numBlocks * sizeof(float));

    // Step 1: Perform block-wise scan and calculate block sums
    scanKernel<<<numBlocks, THREADS_PER_BLOCK, sharedMemSize>>>(input, output, d_blockSums, N, NUM_WORKS_PER_BLOCK);
    cudaDeviceSynchronize();

    // Step 2: Scan the block sums array inplace using a single block kernel
    int bs_works = (numBlocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
    scanKernelInplaceOneBlock<<<1, THREADS_PER_BLOCK, sharedMemSize>>>(d_blockSums, numBlocks, bs_works);
    cudaDeviceSynchronize();

    // Step 3: Add the scanned block sums back to the output array
    addBlockSumsKernel<<<numBlocks, THREADS_PER_BLOCK>>>(output, d_blockSums, N, NUM_WORKS_PER_BLOCK);

    // Ensure all GPU work is done before freeing memory
    cudaDeviceSynchronize(); // TODO: Add error checking
    cudaFree(d_blockSums);
}
