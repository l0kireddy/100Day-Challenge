#include <iostream>
#include <vector>

void sparse_matvec_mult(const float* A, const float* X, float* output, int N, int M) {
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        for (int j = 0; j < M; j++) {
            sum += A[i * M + j] * X[j];
        }
        output[i] = sum;
    }
}

int main() {
    const int N = 1000;
    const int M = 1000;

    float* A = new float[N * M];
    float* X = new float[M];
    float* output = new float[N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            A[i * M + j] = (i + j) % 3 == 0 ? i + j : 0;
        }
    }
    for (int i = 0; i < M; i++) {
        X[i] = 1.0f;
    }

    sparse_matvec_mult(A, X, output, N, M);

    std::cout << "Sparse matrix-vector multiplication completed successfully" << std::endl;

    delete[] A;
    delete[] X;
    delete[] output;

    return 0;
}
