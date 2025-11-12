#include <iostream>
#include <cmath>

int main() {
    const int N = 1024;
    float *A = new float[N];
    float *B = new float[N];
    float *C = new float[N];
    
    for(int i = 0; i < N; i++) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(i);
        C[i] = A[i] + B[i];
    }

    bool success = true;
    for(int i = 0; i < N; i++) {
        if(std::abs(C[i] - (A[i] + B[i])) > 1e-5) {
            success = false;
            break;
        }
    }
    std::cout << "Vector addition " << (success ? "PASSED" : "FAILED") << std::endl;

    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}
