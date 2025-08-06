# Day 5 – Layer Normalization (CUDA)

This program performs row-wise layer normalization using CUDA. Each thread handles one row, computes the mean and variance, and normalizes the values.

## Features
- Uses shared memory for per-row access
- Calculates mean and stddev per row
- Normalizes each element:  
  (x - mean) / sqrt(variance + epsilon)

## Compile & Run
```bash
nvcc LayerNorm.cu -o LayerNorm
./LayerNorm
