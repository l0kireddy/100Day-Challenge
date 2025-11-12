#include <cmath>
#include <cstddef>

static constexpr float EPS = 1e-10f;

void kl_divergence_compute(const float* predictions, const float* targets,
                           float* output, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        float t = targets[i] + EPS;
        float p = predictions[i] + EPS;
        float diff = std::log(t) - std::log(p);
        output[i] = t * diff;
    }
}

void solution(const float* predictions, const float* targets,
              float* output, size_t n)
{
    kl_divergence_compute(predictions, targets, output, n);
}
