#include <iostream>
#include <cmath>
#include <cstdlib>

void flashAttentionForward(const float *Query, const float *Key, const float *Value, 
                          float *Output, int sequence_length, int embed_dimension) {
    float attention_scale = 1.0f / std::sqrt(static_cast<float>(embed_dimension));
    
    for (int i = 0; i < sequence_length; i++) {
        float row_max = -1e20f;
        float *scores = new float[sequence_length];
        
        for (int j = 0; j < sequence_length; j++) {
            float score = 0.0f;
            for (int d = 0; d < embed_dimension; d++) {
                score += Query[i * embed_dimension + d] * Key[j * embed_dimension + d];
            }
            score *= attention_scale;
            scores[j] = score;
            row_max = std::max(row_max, score);
        }
        
        float row_sum = 0.0f;
        for (int j = 0; j < sequence_length; j++) {
            scores[j] = std::exp(scores[j] - row_max);
            row_sum += scores[j];
        }
        
        for (int d = 0; d < embed_dimension; d++) {
            float weighted_sum = 0.0f;
            for (int j = 0; j < sequence_length; j++) {
                weighted_sum += scores[j] * Value[j * embed_dimension + d];
            }
            Output[i * embed_dimension + d] = (row_sum > 0) ? (weighted_sum / row_sum) : 0;
        }
        
        delete[] scores;
    }
}

int main() {
    const int sequence_length = 128;
    const int embed_dimension = 64;
    
    float *Query = new float[sequence_length * embed_dimension];
    float *Key = new float[sequence_length * embed_dimension];
    float *Value = new float[sequence_length * embed_dimension];
    float *Output = new float[sequence_length * embed_dimension];
    
    for (int i = 0; i < sequence_length * embed_dimension; i++) {
        Query[i] = 2.0f * rand() / RAND_MAX - 1.0f;
        Key[i] = 2.0f * rand() / RAND_MAX - 1.0f;
        Value[i] = 2.0f * rand() / RAND_MAX - 1.0f;
        Output[i] = 0.0f;
    }
    
    flashAttentionForward(Query, Key, Value, Output, sequence_length, embed_dimension);
    
    std::cout << "Flash attention forward computation complete" << std::endl;
    
    delete[] Query;
    delete[] Key;
    delete[] Value;
    delete[] Output;
    
    return 0;
}
