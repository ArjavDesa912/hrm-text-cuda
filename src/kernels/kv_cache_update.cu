#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Batched KV cache update kernel.
// Replaces per-head CPU loops with a single GPU kernel launch.
//
// src_k/src_v: [batch, num_heads, seq_len, head_dim] (from scratch buffers)
// dst_k/dst_v: [batch, num_heads, max_len, head_dim] (cache layout)
//
// For prefill (is_decode=0): writes src at dst position 0..seq_len-1
// For decode   (is_decode=1): writes src at dst position cached_len..cached_len+seq_len-1
extern "C" __global__ void kv_cache_update_kernel(
    const __nv_bfloat16* __restrict__ src_k,
    const __nv_bfloat16* __restrict__ src_v,
    __nv_bfloat16* __restrict__ dst_k,
    __nv_bfloat16* __restrict__ dst_v,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int max_len,
    int cached_len,
    int is_decode
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_heads * seq_len * head_dim;
    int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < total_elements; idx += stride) {
        int tmp = idx;
        int d = tmp % head_dim;
        tmp /= head_dim;
        int s = tmp % seq_len;
        tmp /= seq_len;
        int h = tmp % num_heads;
        int b = tmp / num_heads;

        // src layout: [batch, num_heads, seq_len, head_dim]
        int src_idx = ((b * num_heads + h) * seq_len + s) * head_dim + d;

        // dst layout: [batch, num_heads, max_len, head_dim]
        int dst_seq_pos = is_decode ? (cached_len + s) : s;
        int dst_idx = ((b * num_heads + h) * max_len + dst_seq_pos) * head_dim + d;

        dst_k[dst_idx] = src_k[src_idx];
        dst_v[dst_idx] = src_v[src_idx];
    }
}
