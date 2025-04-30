#pragma once
#include "cuda_runtime.h"
#include <cassert>

template <int TileAWidth, int ThreadBlockHeight, int ThreadBlockWidth, int ThreadsPerRowOfRes, int ThreadsPerColOfRes>
__global__ void matmul(float* a, float* b, float* c, int n) {
    // first we calculate what block do we need to calculate the positions of our block
    // we transpose the block indexing as it increases the cache hit rate(similar to coalescing but for blocks)
    const int bRow = blockIdx.y;
    const int bCol = blockIdx.x;

    // numThreads is the multiplication of the amount of thread per col and the amount per row
    const int numThreads = ThreadsPerRowOfRes * ThreadsPerColOfRes;

    // here we will calculate the sizes of TileA and TileB
    // TileAWidth is the width of TileA and the height of TileB
    const int TileAHeight = ThreadsPerColOfRes * ThreadBlockHeight;
    const int TileBHeight = TileAWidth;
    const int TileBWidth = ThreadsPerRowOfRes * ThreadBlockWidth;

    // Thread Idx is the idx of the result block
    const int threadCol = threadIdx.x % ThreadsPerRowOfRes;
    const int threadRow = threadIdx.x / ThreadsPerRowOfRes;

    // for the coalesced access to both A and B we will assign another four indexes 
    const int threadColLoadA = threadIdx.x % TileAWidth;
    const int threadRowLoadA = threadIdx.x / TileAWidth;
    const int threadColLoadB = threadIdx.x % TileBWidth;
    const int threadRowLoadB = threadIdx.x / TileBWidth;

    // then we calculate the base indexes for our Tiles
    // a shift starts at bRows * TileHeight lines(the n is for the lines) 
    int aShift = n * bRow * TileAHeight;
    // b shift starts at bCol * TileHeight
    int bShift = bCol * TileBWidth;
    // c block location is the sum of the bases
    int cBlockLocation = aShift + bShift;

    // now we will define the shared memory arrays
    __shared__ float As[TileAHeight * TileAWidth];
    __shared__ float Bs[TileBHeight * TileBWidth];

    // here we define the row and col buffers for the out product calculation that enters the resBlock
    // colBuffer can be an int because we will update it in the outer loop
    float colBuffer = 0.0;
    float rowBuffer[ThreadBlockWidth] = {0.0};

    // here we define the block where the thread saves the results of the block calculation
    float resBlock[ThreadBlockHeight * ThreadBlockWidth] = {0.0};

    // assertion that demands Tile dims to divide the matrix size
    // this eliminates the need for checks and improve performance over flexabilty
    assert (n % TileAHeight == 0);
    assert (n % TileBWidth == 0);
    assert (n % TileAWidth == 0);

    // we also check that the amount of threads we have divide the size of the SMEM tiles(the same idea but on the next level of mem)
    assert ((TileAHeight * TileAWidth) % numThreads == 0);
    assert ((TileBHeight * TileBWidth) % numThreads == 0);

    // we'll also check that each iteration loads a fixed number of lines from A and B
    // by making sure the length of the tiles divides the number of threads
    assert ((numThreads) % TileAWidth == 0);
    assert ((numThreads) % TileBWidth == 0);

    for (int tileIdx = 0; tileIdx < n / TileAWidth; ++tileIdx) {
        // first we run two loops to load TileA and TileB
        for (int base = 0; base < TileAHeight * TileAWidth; base+=numThreads){
		As[base + threadRowLoadA * TileAWidth + threadColLoadA] = a[aShift + base + threadRowLoadA * n + threadColLoadA];
	}

        for (int base = 0; base < TileBHeight * TileBWidth; base+=numThreads){
		Bs[base + threadRowLoadB * TileBWidth + threadColLoadB] = b[bShift + base + threadRowLoadB * n + threadColLoadB];
	}

        __syncthreads();

        for (int resIdx = 0; (resIdx < TileAWidth); ++resIdx) {
	    for (int i = 0; i < ThreadBlockWidth; ++i){
		rowBuffer[i] = Bs[resIdx * TileBWidth + threadCol * ThreadBlockWidth + i];
            }
	    for (int i = 0; i < ThreadBlockHeight; ++i){
		colBuffer = As[(threadRow * ThreadBlockHeight + i) * TileAWidth + resIdx];
	    	for (int j = 0; j < ThreadBlockWidth; ++j){
			resBlock[i * ThreadBlockWidth + j] += colBuffer * rowBuffer[j];
            	}
            }
        }
        
        // then we advance the shifts
        // a shift advance TileWidth
        aShift += TileAWidth;
        // b shift advance TileWidth lines
        bShift += n * TileBHeight;

        __syncthreads();
    }

    for (int i = 0; i < ThreadBlockHeight; ++i){
    	for (int j = 0; j < ThreadBlockWidth; ++j){
		c[cBlockLocation + (threadRow * ThreadBlockHeight + i) * n + (threadCol * ThreadBlockWidth) + j] = resBlock[i * ThreadBlockWidth + j];
        }
    }
}