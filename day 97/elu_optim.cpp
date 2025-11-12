#include <cmath>
#include <cstddef>

void elu_compute(const float* input, float* output, size_t total, float alpha) {
    for (size_t i = 0; i < total; ++i) {
        float x = input[i];
        output[i] = x > 0.0f ? x : alpha * std::expm1(x);
    }
}

void solution(const float* input, float* output, size_t n, size_t m, float alpha) {
    size_t total = n * m;
    elu_compute(input, output, total, alpha);
}
