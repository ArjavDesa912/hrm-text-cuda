#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

// Apply per-head sigmoid gate to attention output (in-place).
// gate and attn_out are [batch, seq, num_heads*head_dim]
// Vectorized with 128-bit (uint4) loads for 8 bf16 values at once.
extern "C" __global__ void apply_attn_gate_kernel(
    const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ attn_out,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 8;

    if (vec_idx + 7 < total_elements) {
        const uint4 vg = reinterpret_cast<const uint4*>(gate)[idx];
        uint4 va = reinterpret_cast<const uint4*>(attn_out)[idx];
        const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&vg);
        __nv_bfloat162* a2 = reinterpret_cast<__nv_bfloat162*>(&va);
        __nv_bfloat162 out_arr[4];

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 fg = __bfloat1622float2(g2[i]);
            float2 fa = __bfloat1622float2(a2[i]);
            float2 fr;
            fr.x = (1.0f / (1.0f + __expf(-fg.x))) * fa.x;
            fr.y = (1.0f / (1.0f + __expf(-fg.y))) * fa.y;
            out_arr[i] = __float22bfloat162_rn(fr);
        }

        reinterpret_cast<uint4*>(attn_out)[idx] = *reinterpret_cast<const uint4*>(out_arr);
    } else {
        for (int i = vec_idx; i < total_elements && i < vec_idx + 8; ++i) {
            float g = __bfloat162float(gate[i]);
            float a = __bfloat162float(attn_out[i]);
            float sig = 1.0f / (1.0f + __expf(-g));
            attn_out[i] = __float2bfloat16(sig * a);
        }
    }
}
