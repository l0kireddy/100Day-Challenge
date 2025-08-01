# CUDA Matrix Addition

This project demonstrates matrix addition using **NVIDIA CUDA** with C++. It performs element-wise addition of two `N x N` matrices on the GPU using a 2D grid of CUDA threads.

---

## 🚀 Features

- CUDA kernel (`MatrixAdd_B`) uses 2D thread and block indexing
- Dynamic memory allocation on host (CPU) and device (GPU)
- Matrix initialization, addition, and result printing
- Uses CUDA runtime API (`cudaMalloc`, `cudaMemcpy`, `cudaFree`)

---

## 📁 Files

- `matrix_add.cu` - Main CUDA C++ source file
- `README.md` - This documentation file

---

## 🧪 Matrix Description

- Matrix Size: `10 x 10`
- A[i][j] = 1.0  
- B[i][j] = 2.0  
- C[i][j] = A[i][j] + B[i][j] = 3.0

---

## 🛠️ Compilation

If you're using Google Colab or a CUDA-capable Linux system:

```bash
nvcc -o matrix_add matrix_add.cu
