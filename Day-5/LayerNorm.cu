#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define EPSILON 1e-7f // Small value to avoid division by zero

// CUDA kernel for per-row Layer Normalization
__global__ void layerNormKernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float rowData[];

    if (row < rows) {
        // Load the row from global memory to shared memory
        for (int i = 0; i < cols; ++i) {
            rowData[i] = input[row * cols + i];
        }
        __syncthreads();

        // Compute mean
        float mean = 0.0f;
        for (int i = 0; i < cols; ++i) {
            mean += rowData[i];
        }
        mean /= cols;

        // Compute variance
        float variance = 0.0f;
        for (int i = 0; i < cols; ++i) {
            float diff = rowData[i] - mean;
            variance += diff * diff;
        }
        variance /= cols;
        float stddev = sqrtf(variance + EPSILON);

        // Normalize the row
        for (int i = 0; i < cols; ++i) {
            output[row * cols + i] = (rowData[i] - mean) / stddev;
        }
    }
}

// Optional: CUDA error checking macro
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    }

int main() {
    const int rows = 10;
    const int cols = 10;
    const int size = rows * cols * sizeof(float);

    // Allocate host memory
    float* h_input = (float*)malloc(size);
    float* h_output = (float*)malloc(size);

    // Initialize input with random values
    for (int i = 0; i < rows * cols; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float* d_input;
    float* d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 64;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMemSize = cols * sizeof(float);

    layerNormKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input, d_output, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    // Print input matrix
    std::cout << "Input Matrix (A):" << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << h_input[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }

    // Print normalized output
    std::cout << "\nNormalized Output (B):" << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << h_output[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output);

    return 0;
}
