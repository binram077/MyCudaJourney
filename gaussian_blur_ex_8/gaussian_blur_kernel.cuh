// gaussian_blur_kernel.cuh
#ifndef GAUSSIAN_BLUR_KERNEL_CUH
#define GAUSSIAN_BLUR_KERNEL_CUH

#include <cuda_runtime.h>

__global__ void gaussianBlur(float* d_input, float* d_output, int width, int height, float sigma){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x >= width) || (y >= height)) return;

    int kernelRadius = (int)(3.0f * sigma);

    float sum = 0;
    float weightSum = 0;
    float double_sigma_squared = (2 * sigma * sigma);

    for (int i = -kernelRadius; i <= kernelRadius; ++i){
        for (int j = -kernelRadius; j <= kernelRadius; ++j){
            int x_shifted = x + i;
            int y_shifted = y + j;
            if ((0 <= x_shifted) && (x_shifted < width) && (0 <= y_shifted) && (y_shifted < height)) {
                float weight = expf(-(i * i + j * j) / double_sigma_squared);
                sum += weight * d_input[y_shifted * width + x_shifted];
                weightSum += weight;
            }
        }
    }

    d_output[y * width + x] = sum / weightSum;
}

#endif // GAUSSIAN_BLUR_KERNEL_CUH