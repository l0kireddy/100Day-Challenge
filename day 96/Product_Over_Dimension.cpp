#include <vector>
#include <cstddef>

void prod_reduce(const float* input, float* output, size_t M, size_t S_d, size_t N)
{
    for (size_t out_idx = 0; out_idx < M * N; ++out_idx) {
        size_t m = out_idx / N;
        size_t n = out_idx % N;
        const float* base = input + (m * S_d) * N + n;
        
        double prod = 1.0;
        for (size_t k = 0; k < S_d; ++k) {
            prod *= static_cast<double>(base[k * N]);
        }
        output[out_idx] = static_cast<float>(prod);
    }
}

void solution(const float* input, int dim, float* output, size_t* shape, size_t ndim)
{
    size_t M = 1, N = 1;
    for (int i = 0; i < dim; ++i) M *= shape[i];
    for (int i = dim+1; i < (int)ndim; ++i) N *= shape[i];
    size_t S_d = shape[dim];
    
    size_t total_outputs = M * N;
    if (total_outputs == 0 || S_d == 0) return;
    
    prod_reduce(input, output, M, S_d, N);
}
