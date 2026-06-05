#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

// Multi-head attention kernel (prefill path).
// Layout: Q, K, V are [batch, num_heads, seq_len, head_dim]
// Mask is [batch, seq_len, seq_len] (already includes PrefixLM + causal)
// Output is [batch, num_heads, seq_len, head_dim]
//
// Each block = one (batch, head, q_pos).
// Optimizations:
// - Vectorized __nv_bfloat162 loads for Q, K, V (2x memory bandwidth)
// - Requires head_dim to be even
extern "C" __global__ void mha_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ mask,
    __nv_bfloat16* __restrict__ output,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    int bh = blockIdx.x;
    int q_pos = blockIdx.y;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    int warps_per_block = blockDim.x / 32;

    int batch = bh / num_heads;
    int head = bh % num_heads;

    if (batch >= batch_size) return;

    int vec_head_dim = head_dim / 2;
    int q_vec_offset = ((batch * num_heads + head) * seq_len + q_pos) * vec_head_dim;

    extern __shared__ float sbuf[];
    float* q_vec = sbuf;
    float* scores = &sbuf[head_dim];
    float* warp_buf = &sbuf[head_dim + seq_len];

    // Load Q vector for this q_pos (vectorized __nv_bfloat162)
    const __nv_bfloat162* q_vec2 = (const __nv_bfloat162*)q;
    for (int i = tid; i < vec_head_dim; i += blockDim.x) {
        __nv_bfloat162 val = q_vec2[q_vec_offset + i];
        q_vec[i * 2]     = __bfloat162float(val.x);
        q_vec[i * 2 + 1] = __bfloat162float(val.y);
    }
    __syncthreads();

    float scale = 1.0f / sqrtf((float)head_dim);

    // Compute attention scores using vectorized K loads
    const __nv_bfloat162* k_vec2 = (const __nv_bfloat162*)k;
    for (int k_pos = tid; k_pos < seq_len; k_pos += blockDim.x) {
        float dot = 0.0f;
        int k_vec_offset = ((batch * num_heads + head) * seq_len + k_pos) * vec_head_dim;
        for (int d = 0; d < vec_head_dim; ++d) {
            __nv_bfloat162 k_val = k_vec2[k_vec_offset + d];
            int q_idx = d * 2;
            dot += q_vec[q_idx]     * __bfloat162float(k_val.x);
            dot += q_vec[q_idx + 1] * __bfloat162float(k_val.y);
        }
        dot *= scale;
        int mask_idx = ((batch * seq_len + q_pos) * seq_len + k_pos);
        dot += __bfloat162float(mask[mask_idx]);
        scores[k_pos] = dot;
    }
    __syncthreads();

    // Softmax over seq_len
    float local_max = -1e30f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        local_max = fmaxf(local_max, scores[i]);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
        local_max = fmaxf(local_max, other);
    }
    if (lane == 0) warp_buf[warp_id] = local_max;
    __syncthreads();
    unsigned reduce_mask = (1u << warps_per_block) - 1;
    if (tid < warps_per_block) {
        float val = warp_buf[tid];
        for (int offset = warps_per_block / 2; offset > 0; offset /= 2) {
            val = fmaxf(val, __shfl_down_sync(reduce_mask, val, offset));
        }
        if (tid == 0) warp_buf[0] = val;
    }
    __syncthreads();
    float row_max = warp_buf[0];

    for (int i = tid; i < seq_len; i += blockDim.x) {
        scores[i] = expf(scores[i] - row_max);
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        local_sum += scores[i];
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }
    if (lane == 0) warp_buf[warp_id] = local_sum;
    __syncthreads();
    if (tid < warps_per_block) {
        float val = warp_buf[tid];
        for (int offset = warps_per_block / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(reduce_mask, val, offset);
        }
        if (tid == 0) warp_buf[0] = val;
    }
    __syncthreads();
    float row_sum = warp_buf[0];

    for (int i = tid; i < seq_len; i += blockDim.x) {
        scores[i] /= row_sum;
    }
    __syncthreads();

    // Compute weighted sum of V (vectorized)
    __nv_bfloat162* out_vec2 = (__nv_bfloat162*)output;
    int out_vec_offset = ((batch * seq_len + q_pos) * num_heads + head) * vec_head_dim;
    const __nv_bfloat162* v_vec2 = (const __nv_bfloat162*)v;

    for (int d = tid; d < vec_head_dim; d += blockDim.x) {
        float out_val_x = 0.0f;
        float out_val_y = 0.0f;
        for (int k_pos = 0; k_pos < seq_len; ++k_pos) {
            int v_vec_offset = ((batch * num_heads + head) * seq_len + k_pos) * vec_head_dim + d;
            __nv_bfloat162 v_val = v_vec2[v_vec_offset];
            out_val_x += scores[k_pos] * __bfloat162float(v_val.x);
            out_val_y += scores[k_pos] * __bfloat162float(v_val.y);
        }
        out_vec2[out_vec_offset + d] = __floats2bfloat162_rn(out_val_x, out_val_y);
    }
}
