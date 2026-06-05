#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Compute PrefixLM attention bias mask.
// Input: token_type_ids [batch, seq_len] (int32)
// Output: mask [batch, 1, seq_len, seq_len] (bf16)
//   type_id == 1 -> bidirectional (0.0)
//   otherwise -> causal (0.0 for j <= i, -inf for j > i)
extern "C" __global__ void prefixlm_mask_kernel(
    const int* __restrict__ token_type_ids,
    __nv_bfloat16* __restrict__ mask,
    int batch_size,
    int seq_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_len * seq_len;

    if (idx >= total) return;

    // Decompose idx into (batch, q_pos, k_pos)
    int tmp = idx;
    int k_pos = tmp % seq_len;
    tmp /= seq_len;
    int q_pos = tmp % seq_len;
    int batch = tmp / seq_len;

    int type_id = token_type_ids[batch * seq_len + q_pos];

    bool attend;
    if (type_id == 1) {
        // Prefix: bidirectional (attend to all prefix positions)
        int k_type = token_type_ids[batch * seq_len + k_pos];
        attend = (k_type == 1);
    } else {
        // Causal: attend only to past tokens
        attend = (k_pos <= q_pos);
    }

    float val = attend ? 0.0f : -65504.0f; // -inf proxy for bf16
    mask[idx] = __float2bfloat16(val);
}
