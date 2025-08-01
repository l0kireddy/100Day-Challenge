#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel using 2D thread/block for matrix addition
__global__ void MatrixAdd_B(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // row
    int j = blockIdx.y * blockDim.y + threadIdx.y; // column

    if (i < N && j < N) {
        C[i * N + j] = A[i * N + j] + B[i * N + j];
    }
}

int main() {
    const int N = 10;
    const int SIZE = N * N * sizeof(float);

    float *A, *B, *C;

    // Allocate host memory
    A = (float *)malloc(SIZE);
    B = (float *)malloc(SIZE);
    C = (float *)malloc(SIZE);

    // Initialize input matrices
    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
        C[i] = 0.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, SIZE);
    cudaMalloc((void**)&d_B, SIZE);
    cudaMalloc((void**)&d_C, SIZE);

    // Copy input data from host to device
    cudaMemcpy(d_A, A, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, SIZE, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    MatrixAdd_B<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(C, d_C, SIZE, cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Matrix A:\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << A[i * N + j] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nMatrix B:\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << B[i * N + j] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nMatrix C = A + B:\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << "\n";
    }

    // Free device and host memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);

    return 0;
}