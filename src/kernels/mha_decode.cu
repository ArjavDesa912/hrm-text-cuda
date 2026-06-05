#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

// Decode-optimized MHA kernel for q_len=1.
// Q: [batch, num_heads, 1, head_dim]
// K_cache: [batch, num_heads, max_len, head_dim]
// V_cache: [batch, num_heads, max_len, head_dim]
// mask: [batch, kv_len]
// output: [batch, num_heads, 1, head_dim]
//
// Optimizations:
// - Vectorized __nv_bfloat162 loads for Q, K, V (2x memory bandwidth)
// - Requires head_dim to be even
extern "C" __global__ void mha_decode_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const __nv_bfloat16* __restrict__ mask,
    __nv_bfloat16* __restrict__ output,
    int batch_size,
    int num_heads,
    int kv_len,
    int head_dim,
    int max_len
) {
    int bh = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    int warps_per_block = blockDim.x / 32;

    int batch = bh / num_heads;
    int head = bh % num_heads;
    if (batch >= batch_size) return;

    int vec_head_dim = head_dim / 2;
    int q_vec_base = ((batch * num_heads + head)) * vec_head_dim;
    int kv_vec_base = (batch * num_heads + head) * max_len * vec_head_dim;

    extern __shared__ float sbuf[];
    float* q_vec = sbuf;
    float* scores = &sbuf[head_dim];
    float* warp_buf = &sbuf[head_dim + kv_len];

    // Load Q vector to shared memory (vectorized __nv_bfloat162)
    const __nv_bfloat162* q_vec2 = (const __nv_bfloat162*)q;
    for (int i = tid; i < vec_head_dim; i += blockDim.x) {
        __nv_bfloat162 val = q_vec2[q_vec_base + i];
        q_vec[i * 2]     = __bfloat162float(val.x);
        q_vec[i * 2 + 1] = __bfloat162float(val.y);
    }
    __syncthreads();

    float scale = 1.0f / sqrtf((float)head_dim);

    // Compute QK^T scores using vectorized K loads
    const __nv_bfloat162* k_vec2 = (const __nv_bfloat162*)k_cache;
    for (int k_pos = tid; k_pos < kv_len; k_pos += blockDim.x) {
        float dot = 0.0f;
        int k_offset = kv_vec_base + k_pos * vec_head_dim;
        for (int d = 0; d < vec_head_dim; ++d) {
            __nv_bfloat162 k_val = k_vec2[k_offset + d];
            int q_idx = d * 2;
            dot += q_vec[q_idx]     * __bfloat162float(k_val.x);
            dot += q_vec[q_idx + 1] * __bfloat162float(k_val.y);
        }
        dot *= scale;
        int mask_idx = batch * kv_len + k_pos;
        dot += __bfloat162float(mask[mask_idx]);
        scores[k_pos] = dot;
    }
    __syncthreads();

    // Softmax — Step 1: find max
    float local_max = -1e30f;
    for (int i = tid; i < kv_len; i += blockDim.x) {
        local_max = fmaxf(local_max, scores[i]);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
    }
    if (lane == 0) warp_buf[warp_id] = local_max;
    __syncthreads();
    unsigned reduce_mask = (1u << warps_per_block) - 1;
    if (tid < warps_per_block) {
        float val = warp_buf[tid];
        for (int off = warps_per_block / 2; off > 0; off /= 2)
            val = fmaxf(val, __shfl_down_sync(reduce_mask, val, off));
        if (tid == 0) warp_buf[0] = val;
    }
    __syncthreads();
    float row_max = warp_buf[0];

    // Step 2: exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < kv_len; i += blockDim.x) {
        float e = __expf(scores[i] - row_max);
        scores[i] = e;
        local_sum += e;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }
    if (lane == 0) warp_buf[warp_id] = local_sum;
    __syncthreads();
    if (tid < warps_per_block) {
        float val = warp_buf[tid];
        for (int off = warps_per_block / 2; off > 0; off /= 2)
            val += __shfl_down_sync(reduce_mask, val, off);
        if (tid == 0) warp_buf[0] = val;
    }
    __syncthreads();
    float inv_sum = 1.0f / warp_buf[0];

    // Step 3: normalize
    for (int i = tid; i < kv_len; i += blockDim.x) {
        scores[i] *= inv_sum;
    }
    __syncthreads();

    // Weighted sum of V (vectorized)
    __nv_bfloat162* out_vec2 = (__nv_bfloat162*)output;
    int out_vec_base = ((batch * num_heads + head)) * vec_head_dim;
    const __nv_bfloat162* v_vec2 = (const __nv_bfloat162*)v_cache;

    for (int d = tid; d < vec_head_dim; d += blockDim.x) {
        float out_val_x = 0.0f;
        float out_val_y = 0.0f;
        for (int k_pos = 0; k_pos < kv_len; ++k_pos) {
            int v_offset = kv_vec_base + k_pos * vec_head_dim + d;
            __nv_bfloat162 v_val = v_vec2[v_offset];
            out_val_x += scores[k_pos] * __bfloat162float(v_val.x);
            out_val_y += scores[k_pos] * __bfloat162float(v_val.y);
        }
        out_vec2[out_vec_base + d] = __floats2bfloat162_rn(out_val_x, out_val_y);
    }
}
