#pragma once
#include "cuda_runtime.h"
#include <stdio.h>
#include <cassert>

__global__ void cumsumprodKernel(const float* A, const float* B, float* output, float* blockSums, int N, int numWorksPerThread){
    const int tid = threadIdx.x;
    const int smemShuffledTid = tid ^ (tid >> 5); // in order to solve bank conflict we shuffle the thread indexes
    extern __shared__ float smem[];
    const float identity[2] = {1.0f, 0.0f};
    __shared__ float sum;
    int iterationBlockSize = blockDim.x * 2;

    // this kernel is launched after the sums are already calculated
    if ((tid == 0)) {
        sum = (blockIdx.x > 0) ? blockSums[blockIdx.x - 1] : identity[1];
    }

    for (int i = 0; i < numWorksPerThread; ++i){
        // define the starting point of the block we calculate
        int iterationBlockStart = blockIdx.x * (iterationBlockSize * numWorksPerThread) + i * iterationBlockSize;
        int iterationBlockEnd = iterationBlockStart + iterationBlockSize;
        
        // loading to smem
        int tIdx = iterationBlockStart + tid;

        // bound checking
        if (tIdx < N) {
            smem[tid * 2] = A[tIdx];
            smem[tid * 2 + 1] = B[tIdx];
        }
        else {
            smem[tid * 2] = identity[0];
            smem[tid * 2 + 1] = identity[1];
        }
        if (tIdx + blockDim.x < N) {
            smem[2 * (tid + blockDim.x)] = A[tIdx + blockDim.x];
            smem[2 * (tid + blockDim.x) + 1] = B[tIdx + blockDim.x];
        }
        else {
            smem[2 * (tid + blockDim.x)] = identity[0];
            smem[2 * (tid + blockDim.x) + 1] = identity[1];
        }
        __syncthreads();

        // up sweep
        for (int d = iterationBlockSize >> 1, offset = 1; d > 0; d >>= 1, offset <<= 1){
            if (smemShuffledTid < d){
                int leftIdx = offset * (2 * smemShuffledTid + 1) - 1;
                int rightIdx = offset * (2 * smemShuffledTid + 2) - 1;
                // first we update the sum and then the prod because it uses it
                smem[2 * rightIdx + 1] = smem[2 * leftIdx + 1] * smem[2 * rightIdx] + smem[2 * rightIdx + 1];
                smem[2 * rightIdx] = smem[2 * leftIdx] * smem[2 * rightIdx];
            }
            __syncthreads();
        }

        // Save the total sum and prod before setting last element to identity
        float blockTotalProd = smem[2 * (iterationBlockSize - 1)];
        float blockTotalSum = smem[2 * (iterationBlockSize - 1) + 1];
        __syncthreads();

        // setting last element to identity
        if (tid == 0){
            smem[2 * (iterationBlockSize - 1)] = identity[0];
            smem[2 * (iterationBlockSize - 1) + 1] = identity[1];
        }
        __syncthreads();

        // down sweep
        for (int offset = iterationBlockSize >> 1, d = 1; offset > 0; offset >>= 1, d <<= 1){
            if (smemShuffledTid < d){
                int leftIdx = offset * (2 * smemShuffledTid + 1) - 1;
                int rightIdx = offset * (2 * smemShuffledTid + 2) - 1;
                float tmpProd = smem[2 * leftIdx];
                float tmpSum = smem[2 * leftIdx + 1];
                // left becomes right
                smem[2 * leftIdx + 1] = smem[2 * rightIdx + 1];
                smem[2 * leftIdx] = smem[2 * rightIdx];
                // note that we should multiply the sum of the sum of the left only by right prod alone
                // so we need to update the sum and then the prod
                smem[2 * rightIdx + 1] = tmpSum * smem[2 * rightIdx] + smem[2 * rightIdx + 1];
                smem[2 * rightIdx] = tmpProd * smem[2 * rightIdx];
            }
            __syncthreads();
        }

        // offloading
        if (tIdx < N) {
            // h_{t-1} = (smem[2 * tid + 1] + sum * smem[2 * tid]) so we multiply it by A_t and add B_t
            output[tIdx] = B[tIdx] + (smem[2 * tid + 1] + sum * smem[2 * tid]) * A[tIdx];
        }
        if (tIdx + blockDim.x < N) {
            // h_{t-1} = (smem[2 * tid + 1] + sum * smem[2 * tid]) so we multiply it by A_t and add B_t
            output[tIdx + blockDim.x] = B[tIdx + blockDim.x] + (smem[2 * (tid + blockDim.x) + 1] + sum * smem[2 * (tid + blockDim.x)]) * A[tIdx + blockDim.x];
        }

        // adding and multiplying with the block sum and prod
        sum = blockTotalSum + blockTotalProd * sum;

        __syncthreads();
    }
}

__global__ void reduceSumsAndProds(float* A, float* B, float* blockSums, float* blockProds, int N, int numWorksPerThread){
    const int tid = threadIdx.x;
    const int smemShuffledTid = tid ^ (tid >> 5); // in order to solve bank conflict we shuffle the thread indexes
    extern __shared__ float smem[];
    const float identity[2] = {1.0f, 0.0f};
    float sumAndProd[2] = {identity[0], identity[1]};
    int iterationBlockSize = blockDim.x * 2;

    for (int i = 0; i < numWorksPerThread; ++i){
        // define the starting point of the block we calculate
        int iterationBlockStart = blockIdx.x * (iterationBlockSize * numWorksPerThread) + i * iterationBlockSize;
        
        // loading to smem
        int tIdx = iterationBlockStart + tid;

        // bound checking
        if (tIdx < N) {
            smem[tid * 2] = A[tIdx];
            smem[tid * 2 + 1] = B[tIdx];
        }
        else {
            smem[tid * 2] = identity[0];
            smem[tid * 2 + 1] = identity[1];
        }
        if (tIdx + blockDim.x < N) {
            smem[2 * (tid + blockDim.x)] = A[tIdx + blockDim.x];
            smem[2 * (tid + blockDim.x) + 1] = B[tIdx + blockDim.x];
        }
        else {
            smem[2 * (tid + blockDim.x)] = identity[0];
            smem[2 * (tid + blockDim.x) + 1] = identity[1];
        }
        __syncthreads();

        // up sweep
        for (int d = iterationBlockSize >> 1, offset = 1; d > 0; d >>= 1, offset <<= 1){
            if (smemShuffledTid < d){
                int leftIdx = offset * (2 * smemShuffledTid + 1) - 1;
                int rightIdx = offset * (2 * smemShuffledTid + 2) - 1;
                // again sum before prod as it uses it
                smem[2 * rightIdx + 1] = smem[2 * leftIdx + 1] * smem[2 * rightIdx] + smem[2 * rightIdx + 1];
                smem[2 * rightIdx] = smem[2 * leftIdx] * smem[2 * rightIdx];
            }
            __syncthreads();
        }

        // adding and multiplying with the block sum and prod
        if (tid == 0){
            // again sum before prod as it uses it
            sumAndProd[1] = smem[2 * (iterationBlockSize - 1) + 1] + smem[2 * (iterationBlockSize - 1)] * sumAndProd[1];
            sumAndProd[0] = smem[2 * (iterationBlockSize - 1)] * sumAndProd[0];
        }
    }
    __syncthreads();

    if (tid == 0) {
        blockProds[blockIdx.x] = sumAndProd[0];
        blockSums[blockIdx.x] = sumAndProd[1];
    }
}

__global__ void accumulateBlockSumsInplace(float* blockSums, float* blockProds, int N, int numWorksPerThread) {
    // this kernel works 1 block only
    assert((blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0));
    
    // now we calculate the cumsumprod for the block of (blockSums, blockProds)
    const int tid = threadIdx.x;
    const int smemShuffledTid = tid ^ (tid >> 5); // in order to solve bank conflict we shuffle the thread indexes
    extern __shared__ float smem[];
    const float identity[2] = {1.0f, 0.0f};
    __shared__ float sum;
    int iterationBlockSize = blockDim.x * 2;

    // Initialize sum
    if (tid == 0) {
        sum = identity[1];
    }

    for (int i = 0; i < numWorksPerThread; ++i){
        // define the starting point of the block we calculate
        int iterationBlockStart = i * iterationBlockSize;
        int iterationBlockEnd = iterationBlockStart + iterationBlockSize;
        
        // loading to smem
        int tIdx = iterationBlockStart + tid;

        // bound checking
        if (tIdx < N) {
            smem[tid * 2] = blockProds[tIdx];
            smem[tid * 2 + 1] = blockSums[tIdx];
        }
        else {
            smem[tid * 2] = identity[0];
            smem[tid * 2 + 1] = identity[1];
        }
        if (tIdx + blockDim.x < N) {
            smem[2 * (tid + blockDim.x)] = blockProds[tIdx + blockDim.x];
            smem[2 * (tid + blockDim.x) + 1] = blockSums[tIdx + blockDim.x];
        }
        else {
            smem[2 * (tid + blockDim.x)] = identity[0];
            smem[2 * (tid + blockDim.x) + 1] = identity[1];
        }
        __syncthreads();

        // up sweep
        for (int d = iterationBlockSize >> 1, offset = 1; d > 0; d >>= 1, offset <<= 1){
            if (smemShuffledTid < d){
                int leftIdx = offset * (2 * smemShuffledTid + 1) - 1;
                int rightIdx = offset * (2 * smemShuffledTid + 2) - 1;
                // again sum before prod as it uses it
                smem[2 * rightIdx + 1] = smem[2 * leftIdx + 1] * smem[2 * rightIdx] + smem[2 * rightIdx + 1];
                smem[2 * rightIdx] = smem[2 * leftIdx] * smem[2 * rightIdx];
            }
            __syncthreads();
        }

        // Save the total sum and prod before setting last element to identity
        float blockTotalProd = smem[2 * (iterationBlockSize - 1)];
        float blockTotalSum = smem[2 * (iterationBlockSize - 1) + 1];
        __syncthreads();

        // setting last element to identity
        if (tid == 0){
            smem[2 * (iterationBlockSize - 1)] = identity[0];
            smem[2 * (iterationBlockSize - 1) + 1] = identity[1];
        }
        __syncthreads();

        // down sweep
        for (int offset = iterationBlockSize >> 1, d = 1; offset > 0; offset >>= 1, d <<= 1){
            if (smemShuffledTid < d){
                int leftIdx = offset * (2 * smemShuffledTid + 1) - 1;
                int rightIdx = offset * (2 * smemShuffledTid + 2) - 1;
                float tmpProd = smem[2 * leftIdx];
                float tmpSum = smem[2 * leftIdx + 1];
                // left becomes right
                smem[2 * leftIdx] = smem[2 * rightIdx];
                smem[2 * leftIdx + 1] = smem[2 * rightIdx + 1];
                // again sum before prod as it uses it
                smem[2 * rightIdx + 1] = tmpSum * smem[2 * rightIdx] + smem[2 * rightIdx + 1];
                smem[2 * rightIdx] = tmpProd * smem[2 * rightIdx];
            }
            __syncthreads();
        }

        // offloading
        if (tIdx < N) {
            // h_{t-1} = (smem[2 * tid + 1] + sum * smem[2 * tid]) so we multiply it by A_t and add B_t
            blockSums[tIdx] = blockSums[tIdx] + (smem[2 * tid + 1] + sum * smem[2 * tid]) * blockProds[tIdx];
        }
        if (tIdx + blockDim.x < N) {
            // h_{t-1} = (smem[2 * tid + 1] + sum * smem[2 * tid]) so we multiply it by A_t and add B_t
            blockSums[tIdx + blockDim.x] = blockSums[tIdx + blockDim.x] + (smem[2 * (tid + blockDim.x) + 1] + sum * smem[2 * (tid + blockDim.x)]) * blockProds[tIdx + blockDim.x];
        }

        // adding and multiplying with the block sum and prod
        if (tid == 0){
            sum = blockTotalSum + blockTotalProd * sum;
        }
        __syncthreads();
    }
}

void cumsumprodScan(float* A, float* B, float* output, int N) {
    if (N <= 0) return;

    const int THREADS_PER_BLOCK = 256;
    const int BLOCK_SIZE = THREADS_PER_BLOCK * 2;
    const int NUM_WORKS_PER_BLOCK = 1; // Tunable parameter
    const int ELEMENTS_PER_BLOCK_TOTAL = BLOCK_SIZE * NUM_WORKS_PER_BLOCK;
    const size_t sharedMemSize = 2 * BLOCK_SIZE * sizeof(float);

    const int numBlocks = (N + ELEMENTS_PER_BLOCK_TOTAL - 1) / ELEMENTS_PER_BLOCK_TOTAL;

    float* d_blockSums = nullptr;
    float* d_blockProds = nullptr;
    
    // TODO: Add error checking for cudaMalloc and kernel launches
    cudaMalloc(&d_blockSums, numBlocks * sizeof(float));
    cudaMalloc(&d_blockProds, numBlocks * sizeof(float));

    // Step 1: Perform block-wise scan and calculate block sums
    reduceSumsAndProds<<<numBlocks, THREADS_PER_BLOCK, sharedMemSize>>>(A, B, d_blockSums, d_blockProds, N, NUM_WORKS_PER_BLOCK);
    cudaDeviceSynchronize();

    // Step 2: Scan the block sums array inplace using a single block kernel
    int bs_works = (numBlocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
    accumulateBlockSumsInplace<<<1, THREADS_PER_BLOCK, sharedMemSize>>>(d_blockSums, d_blockProds, numBlocks, bs_works);
    cudaDeviceSynchronize();

    // Step 3: Add the scanned block sums back to the output array
    cumsumprodKernel<<<numBlocks, THREADS_PER_BLOCK, sharedMemSize>>>(A, B, output, d_blockSums, N, NUM_WORKS_PER_BLOCK);

    // Ensure all GPU work is done before freeing memory
    cudaDeviceSynchronize(); // TODO: Add error checking
    cudaFree(d_blockSums);
    cudaFree(d_blockProds);
}
