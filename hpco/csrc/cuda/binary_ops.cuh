#pragma once
__global__ void muladd_kernel(int numel, const float *a, const float *b,
                              float c, float *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel)
        result[idx] = a[idx] * b[idx] + c;
}
__global__ void mul_kernel(int numel, const float *a, const float *b,
                           float *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel)
        result[idx] = a[idx] * b[idx];
}
__global__ void add_kernel(int numel, const float *a, const float *b,
                           float *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel)
        result[idx] = a[idx] * b[idx];
}
