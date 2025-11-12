#include <cfloat>
#include <algorithm>

void maxpool2d(const float* input, int H, int W,
               int kernel_size, int stride, int padding, int dilation,
               int H_out, int W_out, float* output)
{
    for (int out_y = 0; out_y < H_out; ++out_y) {
        for (int out_x = 0; out_x < W_out; ++out_x) {
            float max_val = -FLT_MAX;
            for (int m = 0; m < kernel_size; ++m) {
                int in_y = out_y * stride + m * dilation - padding;
                for (int n = 0; n < kernel_size; ++n) {
                    int in_x = out_x * stride + n * dilation - padding;
                    if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                        max_val = std::max(max_val, input[in_y * W + in_x]);
                    }
                }
            }
            output[out_y * W_out + out_x] = max_val;
        }
    }
}

void solution(const float* input, int kernel_size, int stride, int padding,
              int dilation, float* output, size_t H, size_t W)
{
    int H_out = (int)(((int)H + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1;
    int W_out = (int)(((int)W + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1;
    maxpool2d(input, (int)H, (int)W, kernel_size, stride, padding, dilation, H_out, W_out, output);
}
