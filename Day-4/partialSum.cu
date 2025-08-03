#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void reduce_sum(float* input, float* output, int N) {
    __shared__ float shared_data[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    shared_data[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // Write block result to output
    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

int main() {
    const int N = 1024;
    float* h_input = new float[N];
    float* h_output;

    for (int i = 0; i < N; ++i) h_input[i] = 1.0f;  // Initialize all elements to 1

    float *d_input, *d_intermediate, *d_output;

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaMalloc(&d_intermediate, num_blocks * sizeof(float));
    reduce_sum<<<num_blocks, BLOCK_SIZE>>>(d_input, d_intermediate, N);

    // Second reduction on CPU (could also be done in CUDA recursively)
    h_output = new float[num_blocks];
    cudaMemcpy(h_output, d_intermediate, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float total_sum = 0;
    for (int i = 0; i < num_blocks; ++i) {
        total_sum += h_output[i];
    }

    std::cout << "Total sum = " << total_sum << std::endl;

    cudaFree(d_input);
    cudaFree(d_intermediate);
    delete[] h_input;
    delete[] h_output;

    return 0;
}
