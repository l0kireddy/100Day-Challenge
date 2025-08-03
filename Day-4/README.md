# Day 4 – Parallel Reduction (Partial Sum)

**Summary:**  
This CUDA program performs a reduction to compute the sum of a 1D array using shared memory. Each block computes a partial sum, and the final sum is calculated on the CPU.

**What I Learned:**  
- How to use shared memory for intra-block communication.
- How to implement tree-based reduction patterns in CUDA.
- The importance of avoiding warp divergence using power-of-2 block sizes.
- Launching multi-stage reductions when data doesn’t fit in one block.

**Modifications I Made:**  
- Added comments and simplified variable names.
- Handled edge cases where `N` is not a multiple of block size.
- Used CPU to compute the final reduction for simplicity.

