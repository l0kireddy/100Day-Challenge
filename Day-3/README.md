# Day 3 – Matrix-Vector Multiplication (CUDA)

**Summary:**  
Implemented parallel matrix-vector multiplication using CUDA. Each GPU thread computes the dot product of one row of the matrix with the input vector.

**What I Learned:**  
- How to map matrix rows to CUDA threads.
- Thread indexing using `blockIdx`, `threadIdx`, and `blockDim`.
- Performing 1D dot products in parallel.

**Modifications I Made:**  
- Added fixed-size test data for easy verification.
- Used a 1D grid with N threads for simple row-to-thread mapping.
- Commented each section for clarity.

