#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Split a [batch, seq_len, 3*hidden_size] tensor into three [batch, seq_len, hidden_size] tensors
// Q, K, V are interleaved in the input: for each position, the first hidden elements are Q,
// next hidden are K, last hidden are V.
extern "C" __global__ void split_qkv_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    int batch_size,
    int seq_len,
    int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_len * hidden_size;

    if (idx >= total) return;

    // Decompose idx into (batch, seq, d)
    int tmp = idx;
    int d = tmp % hidden_size;
    tmp /= hidden_size;
    int s = tmp % seq_len;
    int b = tmp / seq_len;

    int qkv_idx = ((b * seq_len + s) * 3 * hidden_size) + d;
    q_out[idx] = qkv[qkv_idx];
    k_out[idx] = qkv[qkv_idx + hidden_size];
    v_out[idx] = qkv[qkv_idx + 2 * hidden_size];
}
