#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

// Apply RoPE in-place to Q and K tensors.
// Tensor shape: [batch, heads, seq_len, head_dim=128]
// theta = 10000
//
// Each thread handles one (batch, head, pos, pair_index) instead of
// one (batch, head, pos) with a serial loop over pairs.
// Uses sincosf() for fused sin/cos computation.
extern "C" __global__ void rope_embed_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int position_offset,
    float theta
) {
    int half_dim = head_dim / 2;
    int total = batch_size * num_heads * seq_len * half_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int tmp = idx;
    int i = tmp % half_dim;
    tmp /= half_dim;
    int pos = tmp % seq_len;
    tmp /= seq_len;
    int head = tmp % num_heads;
    int batch = tmp / num_heads;

    int base = ((batch * num_heads + head) * seq_len + pos) * head_dim;
    int idx0 = base + i;
    int idx1 = base + i + half_dim;

    // Avoid expensive powf: angle = pos * theta^(-2i/head_dim)
    // = pos * exp(-2i * log(theta) / head_dim)
    float angle = (pos + position_offset) * expf(-2.0f * i * logf(theta) / (float)head_dim);
    float c, s;
    sincosf(angle, &s, &c);

    float x0 = __bfloat162float(q[idx0]);
    float x1 = __bfloat162float(q[idx1]);
    q[idx0] = __float2bfloat16(x0 * c - x1 * s);
    q[idx1] = __float2bfloat16(x1 * c + x0 * s);

    float k0 = __bfloat162float(k[idx0]);
    float k1 = __bfloat162float(k[idx1]);
    k[idx0] = __float2bfloat16(k0 * c - k1 * s);
    k[idx1] = __float2bfloat16(k1 * c + k0 * s);
}
