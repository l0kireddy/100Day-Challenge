#include <cmath>
#include <iostream>

#define PI 3.14159265358979323846

void cmpFHd_cpu(float* rPhi, float* iPhi, float* phiMag,
                float* x, float* y, float* z,
                float* kx, float* ky, float* kz,
                float* rMu, float* iMu, int N, int M) {
    for (int n = 0; n < N; n++) {
        float rFhDn = rPhi[n];
        float iFhDn = iPhi[n];

        for (int m = 0; m < M; m++) {
            float expFhD = 2 * PI * (kx[m] * x[n] + ky[m] * y[n] + kz[m] * z[n]);
            float cArg = std::cos(expFhD);
            float sArg = std::sin(expFhD);
            rFhDn += rMu[m] * cArg - iMu[m] * sArg;
            iFhDn += iMu[m] * cArg + rMu[m] * sArg;
        }

        rPhi[n] = rFhDn;
        iPhi[n] = iFhDn;
        phiMag[n] = std::sqrt(rFhDn * rFhDn + iFhDn * iFhDn);
    }
}

int main() {
    int N = 1024;
    int M = 256;

    float *x = new float[N];
    float *y = new float[N];
    float *z = new float[N];
    float *kx = new float[M];
    float *ky = new float[M];
    float *kz = new float[M];
    float *rMu = new float[M];
    float *iMu = new float[M];
    float *rPhi = new float[N]();
    float *iPhi = new float[N]();
    float *phiMag = new float[N]();

    for (int i = 0; i < N; i++) {
        x[i] = (float)rand() / RAND_MAX;
        y[i] = (float)rand() / RAND_MAX;
        z[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < M; i++) {
        kx[i] = (float)rand() / RAND_MAX;
        ky[i] = (float)rand() / RAND_MAX;
        kz[i] = (float)rand() / RAND_MAX;
        rMu[i] = (float)rand() / RAND_MAX;
        iMu[i] = (float)rand() / RAND_MAX;
    }

    cmpFHd_cpu(rPhi, iPhi, phiMag, x, y, z, kx, ky, kz, rMu, iMu, N, M);

    std::cout << "Complex FHD computation completed" << std::endl;

    delete[] x; delete[] y; delete[] z;
    delete[] kx; delete[] ky; delete[] kz;
    delete[] rMu; delete[] iMu;
    delete[] rPhi; delete[] iPhi; delete[] phiMag;

    return 0;
}
