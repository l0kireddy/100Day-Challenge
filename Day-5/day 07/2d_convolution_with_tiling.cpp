#include <stdio.h>
#include <iostream>
#include <cstdlib>

#define Mask_width 5

void twod_convolution(const float* A, float* C, const float M[Mask_width][Mask_width], int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float result = 0.0f;
            for (int k = 0; k < Mask_width; k++) {
                for (int x = 0; x < Mask_width; x++) {
                    int row = i + k - Mask_width/2;
                    int col = j + x - Mask_width/2;
                    if (row >= 0 && row < n && col >= 0 && col < n) {
                        result += A[row*n+col] * M[k][x];
                    }
                }
            }
            C[i*n+j] = result;
        }
    }
}

int main() {
    int n = 1024;
    float *h_A = (float*)malloc(n * n * sizeof(float));
    float *h_C = (float*)malloc(n * n * sizeof(float));
    float M[Mask_width][Mask_width];

    for (int i = 0; i < Mask_width; i++) {
        for (int j = 0; j < Mask_width; j++) {
            M[i][j] = 1.0f;
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            h_A[i*n + j] = 1.0f;
        }
    }

    twod_convolution(h_A, h_C, M, n);

    bool success = (h_C[n/2*n + n/2] == 25.0f);

    printf("2D convolution %s\n", success ? "PASSED" : "FAILED");

    free(h_A);
    free(h_C);

    return 0;
}
