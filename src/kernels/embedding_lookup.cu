#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Embedding lookup: output[batch, seq, hidden] = embedding_table[input_ids[batch, seq]] * embedding_scale
// input_ids: [batch, seq_len] uint32
// embedding_table: [vocab_size, hidden_size] bf16
// output: [batch, seq_len, hidden_size] bf16
// Vectorized: each thread processes 8 bf16 elements (one uint4).
// Requires hidden_size to be a multiple of 8.
extern "C" __global__ void embedding_lookup_kernel(
    const unsigned int* __restrict__ input_ids,
    const __nv_bfloat16* __restrict__ embedding_table,
    __nv_bfloat16* __restrict__ output,
    int batch_size,
    int seq_len,
    int hidden_size,
    int vocab_size,
    float embedding_scale
) {
    int vec_hidden = hidden_size / 8;
    int total_vec = batch_size * seq_len * vec_hidden;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_vec) return;

    // Decompose idx into (batch, seq, h_vec)
    int tmp = idx;
    int h_vec = tmp % vec_hidden;
    tmp /= vec_hidden;
    int s = tmp % seq_len;
    int b = tmp / seq_len;

    unsigned int token_id = input_ids[b * seq_len + s];
    // Clamp to vocab_size-1 if out of bounds (shouldn't happen with valid data)
    if (token_id >= (unsigned int)vocab_size) token_id = vocab_size - 1;

    int emb_vec_offset = token_id * vec_hidden + h_vec;
    const uint4 v = reinterpret_cast<const uint4*>(embedding_table)[emb_vec_offset];
    const __nv_bfloat162* v2 = reinterpret_cast<const __nv_bfloat162*>(&v);
    __nv_bfloat162 out_arr[4];

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 f = __bfloat1622float2(v2[i]);
        f.x *= embedding_scale;
        f.y *= embedding_scale;
        out_arr[i] = __float22bfloat162_rn(f);
    }

    reinterpret_cast<uint4*>(output)[idx] = *reinterpret_cast<const uint4*>(out_arr);
}
