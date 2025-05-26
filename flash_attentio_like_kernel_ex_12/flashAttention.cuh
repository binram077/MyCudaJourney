#pragma once
#include "cuda_runtime.h"
#include <stdio.h>
#include <cassert>

__global__ void flashAttention(const float* Q, const float* K, const float* V, float* output, int M, int N, int d){
    const int numQueriesPerBlock = 4;
    const int numKeysPerBlock = 32;
    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    const float scaleFactor = 1.0f / sqrtf(d);
    const float c = 0.0f;

    // Shared memory allocations - these should be passed to the kernel launch configuration
    extern __shared__ float shared_mem[];
    // the scores will be a matrix that stores the local attention scores
    float* scores = shared_mem;                                      // numQueriesPerBlock * numKeysPerBlock
    // values will store the intermidiate sum values along the calculation of the row
    float* values = &shared_mem[numQueriesPerBlock * numKeysPerBlock]; // numQueriesPerBlock * d
    // max will store the maximum scores we encountered along the iteration
    float* max = &values[numQueriesPerBlock * d];                     // numQueriesPerBlock
    // maxFixFactors will keep the values which we need to multiply the accumulated values by.
    float* maxFixFactors = &max[numQueriesPerBlock];                     // numQueriesPerBlock
    // sum will store the sum of the exponent of the scores along the iteration
    float* sum = &maxFixFactors[numQueriesPerBlock];                            // numQueriesPerBlock
    
    // initialization
    for (int channelsIdx = tid; channelsIdx < d; channelsIdx += blockSize){
        for (int qIdx = 0; qIdx < numQueriesPerBlock; ++qIdx) {
            values[qIdx * d + channelsIdx] = 0.0f;
        }
    }

    for (int i = tid; i < numQueriesPerBlock; i += blockSize) {
        max[i] = -INFINITY;
        sum[i] = 0.0f;
    }
    __syncthreads();

    // qShift is used to tell us where do we take our queries from
    int qShift = blockIdx.x * numQueriesPerBlock;

    for (int kShift = 0; kShift < N; kShift += numKeysPerBlock){
        // first we would initiate scores to zeros
        for (int i = tid; i < numQueriesPerBlock * numKeysPerBlock; i += blockSize) {
            scores[i] = 0.0f;
        }
        __syncthreads();

        // now every thread will sum an attention chunk with one channel only
        // and then if we don't have enough threads to calc all the channels at once
        // the threads will proceed to calculate more channels.
        for (int channelsIdx = tid; channelsIdx < d; channelsIdx += blockSize){
            // now we load the chunk of the keys and queries into the registered and then we 
            // will sum it into the attentio chunk
            float qValues[numQueriesPerBlock];
            float kValues[numKeysPerBlock];
            for (int i = 0; i < numQueriesPerBlock; ++i) qValues[i] = (qShift + i < M) ? Q[(qShift + i) * d + channelsIdx] : 0;
            for (int i = 0; i < numKeysPerBlock; ++i) kValues[i] = (kShift + i < N) ? K[(kShift + i) * d + channelsIdx] : 0;
            for (int qIdx = 0; (qIdx < numQueriesPerBlock) && (qShift + qIdx < M); ++qIdx) {
                for (int kIdx = 0; (kIdx < numKeysPerBlock) && (kShift + kIdx < N); ++kIdx) {
                    atomicAdd(scores + qIdx * numKeysPerBlock + kIdx, scaleFactor * qValues[qIdx] * kValues[kIdx]);
                }
            }
        }
        __syncthreads();

        // now that scores is full we will calculate the max and then the sum
        // we begin with updating the max
        if (tid < numQueriesPerBlock && (qShift + tid < M)) {
            float oldMax = max[tid];
            float newMax = oldMax;

            for (int i = 0; (i < numKeysPerBlock) && (kShift + i < N); ++i) {
                newMax = fmaxf(newMax, scores[tid * numKeysPerBlock + i]);
            }
            
            max[tid] = newMax;

            float maxFixFactor = expf(oldMax - newMax);
            maxFixFactors[tid] = maxFixFactor;
            sum[tid] *= maxFixFactor;

            for (int i = 0; (i < numKeysPerBlock) && (kShift + i < N); ++i) {
                sum[tid] += (expf(scores[tid * numKeysPerBlock + i] - newMax + c));
            }
        }
        __syncthreads();

        // now using the attention scores we shall calculate the values
        for (int channelsIdx = tid; channelsIdx < d; channelsIdx += blockSize) {
            for (int qIdx = 0; (qIdx < numQueriesPerBlock) && (qShift + qIdx < M); ++qIdx) {
                // we would do the accumulation in the registers for better accuracy
                float accValue = values[qIdx * d + channelsIdx] * maxFixFactors[qIdx];
                
                // Add new values directly to the accumulator
                for (int kIdx = 0; (kIdx < numKeysPerBlock) && (kShift + kIdx < N); ++kIdx) {
                    float attnScore = expf(scores[qIdx * numKeysPerBlock + kIdx] - max[qIdx] + c);
                    accValue += attnScore * V[(kShift + kIdx) * d + channelsIdx];
                }
                
                // Write back once
                values[qIdx * d + channelsIdx] = accValue;
            }
        }
        __syncthreads();
    }

    // offloading
    for (int channelsIdx = tid; channelsIdx < d; channelsIdx += blockSize){
        for (int qIdx = 0; (qIdx < numQueriesPerBlock) && (qShift + qIdx < M); ++qIdx) {
            output[(qShift + qIdx) * d + channelsIdx] = values[qIdx * d + channelsIdx] / sum[qIdx];
        }
    }
    __syncthreads();
}

// Q, K, V, output are device pointers
void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    const int numQueriesPerBlock = 4, numKeysPerBlock = 32;
    const int THREADS_PER_BLOCK = 128;
    const int NUM_BLOCKS = (M + numQueriesPerBlock - 1) / numQueriesPerBlock;
    const int SMEM_SIZE = sizeof(float) * numQueriesPerBlock * (3 + d + numKeysPerBlock);

    flashAttention<<<NUM_BLOCKS, THREADS_PER_BLOCK, SMEM_SIZE>>>(Q, K, V, output, M, N, d);
}
