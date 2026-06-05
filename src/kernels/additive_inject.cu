#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Fused elementwise BF16 addition with 128-bit (8x bf16) vectorized loads.
// z_out = z_a + z_b for tensors of shape [batch, seq_len, hidden_size]
extern "C" __global__ void additive_inject_kernel_vec8(
    const __nv_bfloat16* __restrict__ z_a,
    const __nv_bfloat16* __restrict__ z_b,
    __nv_bfloat16* __restrict__ z_out,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 8;

    if (vec_idx + 7 < total_elements) {
        // uint4 = 128 bits. Reinterpret as 4 x __nv_bfloat162 (32 bits each = 8 bf16)
        const uint4 va_u = reinterpret_cast<const uint4*>(z_a)[idx];
        const uint4 vb_u = reinterpret_cast<const uint4*>(z_b)[idx];

        const __nv_bfloat162* va = reinterpret_cast<const __nv_bfloat162*>(&va_u);
        const __nv_bfloat162* vb = reinterpret_cast<const __nv_bfloat162*>(&vb_u);
        __nv_bfloat162 vo_arr[4];

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 fa = __bfloat1622float2(va[i]);
            float2 fb = __bfloat1622float2(vb[i]);
            float2 fr;
            fr.x = fa.x + fb.x;
            fr.y = fa.y + fb.y;
            vo_arr[i] = __float22bfloat162_rn(fr);
        }

        uint4 vo_u = *reinterpret_cast<const uint4*>(vo_arr);
        reinterpret_cast<uint4*>(z_out)[idx] = vo_u;
    } else {
        // Scalar tail loop for remaining elements
        for (int i = vec_idx; i < total_elements && i < vec_idx + 8; ++i) {
            float fa = __bfloat162float(z_a[i]);
            float fb = __bfloat162float(z_b[i]);
            z_out[i] = __float2bfloat16(fa + fb);
        }
    }
}
