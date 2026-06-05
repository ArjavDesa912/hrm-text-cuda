#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Parameterless Pre-RMSNorm over the last dimension (hidden_size = 1536).
// No learned weight/bias. Uses warp-level reductions.
// grid: [batch * seq_len], block: [threads_per_block]
// Each block processes one row of [hidden_size] elements.
extern "C" __global__ void rms_norm_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int hidden_size,
    float epsilon
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    const int warps_per_block = blockDim.x / 32;
    extern __shared__ float shared_sums[]; // size = warps_per_block

    const __nv_bfloat16* row_in = input + row * hidden_size;
    __nv_bfloat16* row_out = output + row * hidden_size;

    float local_sum = 0.0f;

    // Strided loop over hidden_size
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float v = __bfloat162float(row_in[i]);
        local_sum += v * v;
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }

    if (lane == 0) {
        shared_sums[warp_id] = local_sum;
    }
    __syncthreads();

    // Reduce across warps
    unsigned reduce_mask = (1u << warps_per_block) - 1;
    if (tid < warps_per_block) {
        float val = shared_sums[tid];
        #pragma unroll
        for (int offset = warps_per_block / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(reduce_mask, val, offset);
        }
        if (tid == 0) {
            shared_sums[0] = val;
        }
    }
    __syncthreads();

    float rms = sqrtf(shared_sums[0] / (float)hidden_size + epsilon);
    float inv_rms = 1.0f / rms;

    // Write normalized output
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float v = __bfloat162float(row_in[i]);
        row_out[i] = __float2bfloat16(v * inv_rms);
    }
}
