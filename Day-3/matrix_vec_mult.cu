// Day 3: Matrix-Vector Multiplication using CUDA
// Author: DREAMER
// Each thread computes the dot product between one row of the matrix and the input vector.

#include <iostream>
#include <cuda_runtime.h>

#define N 4  // Matrix rows
#define M 4  // Matrix columns

__global__ void matVecMulKernel(float* mat, float* vec, float* result, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N) {
        float sum = 0.0f;
        for (int i = 0; i < cols; i++) {
            sum += mat[row * cols + i] * vec[i];
        }
        result[row] = sum;
    }
}

int main() {
    float h_mat[N * M] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9,10,11,12,
       13,14,15,16
    };
    float h_vec[M] = {1, 1, 1, 1};
    float h_result[N];

    float *d_mat, *d_vec, *d_result;

    cudaMalloc(&d_mat, N * M * sizeof(float));
    cudaMalloc(&d_vec, M * sizeof(float));
    cudaMalloc(&d_result, N * sizeof(float));

    cudaMemcpy(d_mat, h_mat, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, h_vec, M * sizeof(float), cudaMemcpyHostToDevice);

    matVecMulKernel<<<1, N>>>(d_mat, d_vec, d_result, M);

    cudaMemcpy(h_result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Matrix-Vector Multiplication Result:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << h_result[i] << std::endl;
    }

    cudaFree(d_mat);
    cudaFree(d_vec);
    cudaFree(d_result);

    return 0;
}
