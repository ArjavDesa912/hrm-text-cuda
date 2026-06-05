#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Split gqkv [batch, seq, 4*num_heads*head_dim] → gate, q, k, v each [batch, num_heads, seq_len, head_dim]
// PyTorch layout per token: [gate_heads..., query_heads..., key_heads..., value_heads...]
// where each group has num_heads heads of head_dim elements.
// Output layout: [batch, num_heads, seq_len, head_dim] for each tensor
// Vectorized: each thread processes 8 bf16 elements (one uint4).
// Requires head_dim to be a multiple of 8.
extern "C" __global__ void split_gqkv_kernel(
    const __nv_bfloat16* __restrict__ gqkv,
    __nv_bfloat16* __restrict__ gate_out,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim
) {
    int vec_dim = head_dim / 8;
    int total_vec = batch_size * num_heads * seq_len * vec_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_vec) return;

    int tmp = idx;
    int d_vec = tmp % vec_dim;
    tmp /= vec_dim;
    int s = tmp % seq_len;
    tmp /= seq_len;
    int h = tmp % num_heads;
    int b = tmp / num_heads;

    // PyTorch blocked layout: gate heads first, then query, key, value
    int token_base = ((b * seq_len + s) * 4 * num_heads) * vec_dim;

    int gqkv_gate_idx = token_base + h * vec_dim + d_vec;
    int gqkv_q_idx    = token_base + (num_heads + h) * vec_dim + d_vec;
    int gqkv_k_idx    = token_base + (2 * num_heads + h) * vec_dim + d_vec;
    int gqkv_v_idx    = token_base + (3 * num_heads + h) * vec_dim + d_vec;

    int gate_out_idx = ((b * seq_len + s) * num_heads + h) * vec_dim + d_vec;

    reinterpret_cast<uint4*>(gate_out)[gate_out_idx] = reinterpret_cast<const uint4*>(gqkv)[gqkv_gate_idx];
    reinterpret_cast<uint4*>(q_out)[idx]    = reinterpret_cast<const uint4*>(gqkv)[gqkv_q_idx];
    reinterpret_cast<uint4*>(k_out)[idx]    = reinterpret_cast<const uint4*>(gqkv)[gqkv_k_idx];
    reinterpret_cast<uint4*>(v_out)[idx]    = reinterpret_cast<const uint4*>(gqkv)[gqkv_v_idx];
}
