#include <cstddef>
#include <algorithm>

void leaky_relu_compute(const float* input, float alpha, float* output, size_t M, size_t N) {
    size_t total = M * N;
    for (size_t i = 0; i < total; ++i) {
        float v = input[i];
        output[i] = std::max(v, alpha * v);
    }
}

void solution(const float* input, float alpha, float* output, size_t M, size_t N) {
    leaky_relu_compute(input, alpha, output, M, N);
}
