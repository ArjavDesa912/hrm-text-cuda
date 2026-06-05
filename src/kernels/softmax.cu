#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Softmax over the last dimension (seq_len) for attention scores.
// Input:  [batch, heads, seq_len, seq_len] bf16
// Output: same shape, softmax applied per row over last dim
//
// Dynamic shared memory layout: [warps_per_block] warp_buf for cross-warp reduction
extern "C" __global__ void softmax_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int batch_size,
    int num_heads,
    int seq_len
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

    int row_offset = ((batch * num_heads + head) * seq_len + q_pos) * seq_len;

    extern __shared__ float warp_buf[];

    // Step 1: find max
    float local_max = -1e30f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        float v = __bfloat162float(input[row_offset + i]);
        if (v > local_max) local_max = v;
    }

    // Warp reduce max
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
    }
    if (lane == 0) warp_buf[warp_id] = local_max;
    __syncthreads();
    if (tid < warps_per_block) {
        float val = warp_buf[tid];
        for (int off = warps_per_block/2; off > 0; off /= 2)
            val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, off));
        if (tid == 0) warp_buf[0] = val;
    }
    __syncthreads();
    float row_max = warp_buf[0];

    // Step 2: compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        float v = __bfloat162float(input[row_offset + i]);
        float exp_v = expf(v - row_max);
        output[row_offset + i] = __float2bfloat16(exp_v);
        local_sum += exp_v;
    }

    // Warp reduce sum
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }
    if (lane == 0) warp_buf[warp_id] = local_sum;
    __syncthreads();
    if (tid < warps_per_block) {
        float val = warp_buf[tid];
        for (int off = warps_per_block/2; off > 0; off /= 2)
            val += __shfl_down_sync(0xFFFFFFFF, val, off);
        if (tid == 0) warp_buf[0] = val;
    }
    __syncthreads();
    float row_sum = warp_buf[0];

    // Step 3: normalize
    for (int i = tid; i < seq_len; i += blockDim.x) {
        float v = __bfloat162float(output[row_offset + i]);
        output[row_offset + i] = __float2bfloat16(v / row_sum);
    }
}
