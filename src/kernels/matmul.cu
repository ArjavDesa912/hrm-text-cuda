#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Tiled kernel for m > 1 (prefill)
constexpr int TILE = 32;

extern "C" __global__ void matmul_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int m,
    int n,
    int k
) {
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    __shared__ float sA[TILE][TILE + 1];
    __shared__ float sB[TILE][TILE + 1];

    float sum = 0.0f;
    int tiles = (k + TILE - 1) / TILE;
    for (int t = 0; t < tiles; ++t) {
        int tile_k = t * TILE;

        if (row < m && tile_k + threadIdx.x < k) {
            sA[threadIdx.y][threadIdx.x] = __bfloat162float(A[row * k + tile_k + threadIdx.x]);
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < n && tile_k + threadIdx.y < k) {
            sB[threadIdx.y][threadIdx.x] = __bfloat162float(B[col * k + tile_k + threadIdx.y]);
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

#pragma unroll
        for (int i = 0; i < TILE; ++i) {
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * n + col] = __float2bfloat16(sum);
    }
}

// Fast decode kernel for m == 1 (batch_size=1, seq_len=1)
// Each block handles 32 output columns; 32 threads per column split the dot-product over k
extern "C" __global__ void matmul_m1_kernel(
    const __nv_bfloat16* __restrict__ A,  // [k]
    const __nv_bfloat16* __restrict__ B,  // [n, k]
    __nv_bfloat16* __restrict__ C,        // [n]
    int n,
    int k
) {
    __shared__ float sSum[32][33]; // [col_in_block][thread_y]

    int col = blockIdx.x * 32 + threadIdx.x;

    float sum = 0.0f;
    for (int i = threadIdx.y; i < k; i += 32) {
        sum += __bfloat162float(A[i]) * __bfloat162float(B[col * k + i]);
    }

    sSum[threadIdx.x][threadIdx.y] = sum;
    __syncthreads();

    if (threadIdx.y == 0 && col < n) {
        float total = 0.0f;
        for (int i = 0; i < 32; ++i) {
            total += sSum[threadIdx.x][i];
        }
        C[col] = __float2bfloat16(total);
    }
}
