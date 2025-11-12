#include <iostream>
#include <cstdlib>
#include <cmath>

void prefixSum(float* A, float* C, int N) {
    C[0] = A[0];
    for (int i = 1; i < N; i++) {
        C[i] = C[i-1] + A[i];
    }
}

int main() {
    int N = 1024;
    float *A = (float*)malloc(N * sizeof(float));
    float *C = (float*)malloc(N * sizeof(float));
    
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
    }
    
    prefixSum(A, C, N);

    bool success = true;
    float expected = 0.0f;
    for (int i = 0; i < N; i++) {
        expected += A[i];
        if (std::fabs(C[i] - expected) > 1e-4) {
            success = false;
            break;
        }
    }

    printf("Prefix sum %s\n", success ? "PASSED" : "FAILED");

    free(A);
    free(C);

    return 0;
}
