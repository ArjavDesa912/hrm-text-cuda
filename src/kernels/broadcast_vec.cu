#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Broadcast a vector of length `vec_len` across `n_copies` contiguous positions.
// dst: [n_copies, vec_len]  (n_copies = batch_size * seq_len)
// src: [vec_len]
extern "C" __global__ void broadcast_vec_kernel(
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ src,
    int n_copies,
    int vec_len
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_copies * vec_len;
    int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < total; idx += stride) {
        int d = idx % vec_len;
        dst[idx] = src[d];
    }
}
