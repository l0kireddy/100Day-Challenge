#include <cstddef>
#include <algorithm>

void gemm_bias_relu_compute(const float* A, const float* W, const float* b,
                            float* C, size_t B, size_t N, size_t M)
{
    for (size_t i = 0; i < B; ++i) {
        for (size_t j = 0; j < M; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < N; ++k) {
                sum += A[i * N + k] * W[j * N + k];
            }
            float v = sum + b[j];
            C[i * M + j] = std::max(0.0f, v);
        }
    }
}

void solution(const float* A, const float* W, const float* b,
              float* C, size_t B, size_t N, size_t M)
{
    gemm_bias_relu_compute(A, W, b, C, B, N, M);
}
