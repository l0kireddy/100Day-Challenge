#include <stdio.h>
#include <stdlib.h>
#include <cmath>

void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    int M = 128, N = 128, K = 128;
    float *h_A = (float *)malloc(M * K * sizeof(float));
    float *h_B = (float *)malloc(K * N * sizeof(float));
    float *h_C = (float *)malloc(M * N * sizeof(float));

    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            h_A[i * K + j] = i + j;

    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
            h_B[i * N + j] = i + j;

    matmul(h_A, h_B, h_C, M, N, K);

    printf("Matrix multiplication completed successfully\n");

    free(h_A); free(h_B); free(h_C);
    return 0;
}
