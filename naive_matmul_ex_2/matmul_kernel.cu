#include "./matmul_kernel.cuh"

__global__ void matmul(float* a, float* b, float* c, int n) {
    int id_x = (blockDim.x * blockIdx.x) + threadIdx.x;
    int id_y = (blockDim.y * blockIdx.y) + threadIdx.y;

    if (id_x < n && id_y < n) {
        float tmp = 0;
        for (int i = 0; i < n; ++i) {
            tmp += a[id_x * n + i] * b[i * n + id_y];
        }
        c[id_x * n + id_y] = tmp;
    }
}