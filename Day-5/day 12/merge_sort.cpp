#include <stdio.h>
#include <algorithm>

void merge(const int* A, const int* B, int* C, int N, int M) {
    int i = 0, j = 0, k = 0;
    while (i < N && j < M) {
        if (A[i] <= B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }
    while (i < N) C[k++] = A[i++];
    while (j < M) C[k++] = B[j++];
}

int main() {
    const int N = 1024;
    const int M = 1024;
    int *A = new int[N];
    int *B = new int[M];
    int *C = new int[N+M];
    
    for(int i = 0; i < N; i++) A[i] = 2*i;
    for(int i = 0; i < M; i++) B[i] = 2*i + 1;

    merge(A, B, C, N, M);

    bool sorted = true;
    for(int i = 0; i < N+M-1; i++) {
        if(C[i] > C[i+1]) {
            sorted = false;
            break;
        }
    }
    
    printf("Parallel merge %s\n", sorted ? "PASSED" : "FAILED");

    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}
