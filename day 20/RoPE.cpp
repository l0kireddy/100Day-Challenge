#include <cmath>
#include <cstdio>

void apply_rotary_embedding_cpu(float* q, float* k, int head_dim, int position, float base = 10000.0f) {
    for (int i = 0; i < head_dim; i += 2) {
        float freq = 1.0f / std::pow(base, (float)(i) / head_dim);
        float theta = position * freq;
        
        float cos_theta = std::cos(theta);
        float sin_theta = std::sin(theta);
        
        float q_real = q[i];
        float q_img = q[i + 1];
        float k_real = k[i];
        float k_img = k[i + 1];
        
        q[i] = q_real * cos_theta - q_img * sin_theta;
        q[i + 1] = q_real * sin_theta + q_img * cos_theta;
        
        k[i] = k_real * cos_theta - k_img * sin_theta;
        k[i + 1] = k_real * sin_theta + k_img * cos_theta;
    }
}

void apply_rope_cpu(float* queries, float* keys, int batch_size, int seq_len, int num_heads, int head_dim) {
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            for (int h = 0; h < num_heads; h++) {
                int base_idx = b * (seq_len * num_heads * head_dim) + 
                              s * (num_heads * head_dim) + h * head_dim;
                apply_rotary_embedding_cpu(&queries[base_idx], &keys[base_idx], head_dim, s);
            }
        }
    }
}

int main() {
    const int batch_size = 2;
    const int seq_len = 128;
    const int num_heads = 8;
    const int head_dim = 64;

    float *queries = new float[batch_size * seq_len * num_heads * head_dim];
    float *keys = new float[batch_size * seq_len * num_heads * head_dim];

    for(size_t i = 0; i < batch_size * seq_len * num_heads * head_dim; i++) {
        queries[i] = static_cast<float>(rand()) / RAND_MAX;
        keys[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    apply_rope_cpu(queries, keys, batch_size, seq_len, num_heads, head_dim);

    printf("RoPE completed successfully\n");

    delete[] queries;
    delete[] keys;

    return 0;
}
