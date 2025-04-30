#pragma once
#include "cuda_runtime.h"
#include <cassert>

template <int TileHeight, int TileWidth>
__global__ void matmul(float* a, float* b, float* c, int n) {
    // first we calculate what block do we need to calculate the positions of our block
    // we transpose the block indexing as it increases the cache hit rate(similar to coalescing but for blocks)
    int bRow = blockIdx.y;
    int bCol = blockIdx.x;

    // the output of each Block is (TileHeight, TileWidth) * (TileWidth, TileHeight) => (TileHeight, TileHeight)
    // and each thread is caculating (TileHeight / TileWidth) columns in it(that how (TileHeight * TileWidth) calc (TileHeight * TileHeight) vals)
    // so the horizontal dim of the result is TileHeight, and using coalescing the calculation is:
    int threadRow = threadIdx.x / TileHeight;
    int threadCol = threadIdx.x % TileHeight;

    // then we calculate the base indexes for our Tiles
    // a shift starts at bRows * TileHeight lines(the n is for the lines) 
    int aShift = n * bRow * TileHeight;
    // b shift starts at bCol * TileHeight
    int bShift = bCol * TileHeight;
    // c block location is the sum of the bases
    int cBlockLocation = aShift + bShift;

    // (TileHeight * TileWidth) calc (TileHeight * TileHeight) vals
    // each thread calculates (TileHeight * TileHeight) / (TileHeight * TileWidth) rows in its column
    const int num_rows_per_thread = TileHeight / TileWidth;

    // now we will define the shared memory arrays
    __shared__ float As[TileHeight * TileWidth];
    __shared__ float Bs[TileWidth * TileHeight];

    // the idxs to loading are transposed for the A tile
    // as (threadRow, threadCol) calculate num_rows_per_thread from threadRow in column thread Col
    // meaning the thread indexes to results mapping is tranposed in relation to A
    int AThreadRow = threadIdx.x / TileWidth;
    int AThreadCol = threadIdx.x % TileWidth;

    // threadResults accumulate the result for the final num_rows_per_thread rows in c
    float threadResults[num_rows_per_thread] = {0.0};

    // assertion that demands Tile dims to divide the matrix size
    // this eliminates the need for checks and improve performance over flexabilty
    assert (n % TileHeight == 0);
    assert (n % TileWidth == 0);

    for (int i = 0; i < (n + 1) / TileWidth; ++i) {
        // first we will load the data to the smem(both accesses are coalesced)
        As[TileWidth * AThreadRow + AThreadCol] = a[aShift + n * AThreadRow + AThreadCol];
        // note the B is in shape (TileWidth, TileHeight)
        Bs[TileHeight * threadRow + threadCol] = b[bShift + n * threadRow + threadCol];

        __syncthreads();

        for (int j = 0; (j < TileWidth); ++j) {
            float BTmp = Bs[j * TileHeight + threadCol];
            for (int resIdx = 0; resIdx < num_rows_per_thread; ++resIdx){
                threadResults[resIdx] += As[(threadRow * num_rows_per_thread + resIdx) * TileWidth + j] * BTmp;
            }
        }
        
        // then we advance the shifts
        // a shift advance TileWidth
        aShift += TileWidth;
        // b shift advance TileWidth lines
        bShift += n * TileWidth;

        __syncthreads();
    }

    for (int resIdx = 0; resIdx < num_rows_per_thread; ++resIdx){
        int rowShift = (threadRow * num_rows_per_thread + resIdx) * n;
        c[cBlockLocation + rowShift + threadCol] = threadResults[resIdx];
    }
}