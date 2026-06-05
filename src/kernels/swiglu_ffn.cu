#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

// Fused SwiGLU: output = silu(G) * U
// where silu(x) = x * sigmoid(x)
// G and U are of shape [batch, seq_len, ffn_dim=4096]
// Vectorized with 128-bit (uint4) loads for 8 bf16 values at once.
extern "C" __global__ void swiglu_ffn_kernel(
    const __nv_bfloat16* __restrict__ gate_proj,
    const __nv_bfloat16* __restrict__ up_proj,
    __nv_bfloat16* __restrict__ output,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 8;

    if (vec_idx + 7 < total_elements) {
        const uint4 vg = reinterpret_cast<const uint4*>(gate_proj)[idx];
        const uint4 vu = reinterpret_cast<const uint4*>(up_proj)[idx];
        const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&vg);
        const __nv_bfloat162* u2 = reinterpret_cast<const __nv_bfloat162*>(&vu);
        __nv_bfloat162 out_arr[4];

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 fg = __bfloat1622float2(g2[i]);
            float2 fu = __bfloat1622float2(u2[i]);
            float2 fr;
            // silu(g) * u = (g * sigmoid(g)) * u
            float sig_x = 1.0f / (1.0f + __expf(-fg.x));
            float sig_y = 1.0f / (1.0f + __expf(-fg.y));
            fr.x = fg.x * sig_x * fu.x;
            fr.y = fg.y * sig_y * fu.y;
            out_arr[i] = __float22bfloat162_rn(fr);
        }

        reinterpret_cast<uint4*>(output)[idx] = *reinterpret_cast<const uint4*>(out_arr);
    } else {
        for (int i = vec_idx; i < total_elements && i < vec_idx + 8; ++i) {
            float g = __bfloat162float(gate_proj[i]);
            float u = __bfloat162float(up_proj[i]);
            float sig = 1.0f / (1.0f + __expf(-g));
            output[i] = __float2bfloat16(g * sig * u);
        }
    }
}
