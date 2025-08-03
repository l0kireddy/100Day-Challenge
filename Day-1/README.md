## Day 1 - Vector Addition in CUDA



**Summary:**  

Implemented basic vector addition using CUDA, where each thread adds a pair of elements from two arrays. Learned about kernel launches, thread indexing, and GPU memory management (`cudaMalloc`, `cudaMemcpy`, `cudaFree`).



**What I Learned / Did Differently:**  

- Rewrote the kernel with extra checks for bounds.  

- Used `cudaEvent` to measure kernel execution time.  

- Added error checking to memory ops.  



