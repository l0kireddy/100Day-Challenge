#include <iostream>
#include <cstdlib>

void transposeMatrix(const float* input, float* output, int width, int height) {
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int inputIndex = y * width + x;
            int outputIndex = x * height + y;
            output[outputIndex] = input[inputIndex];
        }
    }
}

int main() {
    int width = 1024;
    int height = 1024;

    size_t size = width * height * sizeof(float);
    float* h_input = (float*)malloc(size);
    float* h_output = (float*)malloc(size);

    for (int i = 0; i < width * height; i++) {
        h_input[i] = static_cast<float>(i);
    }

    transposeMatrix(h_input, h_output, width, height);

    bool success = true;
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            if (h_output[i * height + j] != h_input[j * width + i]) {
                success = false;
                break;
            }
        }
    }

    std::cout << "Matrix transposition " << (success ? "PASSED" : "FAILED") << std::endl;

    free(h_input);
    free(h_output);

    return 0;
}
