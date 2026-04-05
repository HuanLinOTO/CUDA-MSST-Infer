// flash_attention.cuh — Flash Attention 2 kernel for SM 8.x
//
// Implements the Flash Attention algorithm (Dao et al.) that computes
// softmax(Q @ K^T / sqrt(d)) @ V in O(N) memory instead of O(N^2).
//
// This avoids materializing the full N×N attention matrix, drastically
// reducing memory bandwidth and improving performance.
//
// Supports: float32, head dims up to 128, arbitrary sequence lengths.

#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <cmath>

namespace cudasep {

// ============================================================================
// Configuration
// ============================================================================

// Block sizes for tiling in sequence dimension
// Br = rows of Q processed per block, Bc = columns of K/V processed per inner loop
// For head_dim <= 64: Br=64, Bc=64
// For head_dim <= 128: Br=32, Bc=64
// These are template parameters for the kernel.

static constexpr int FA_WARP_SIZE = 32;

// ============================================================================
// Flash Attention Forward Kernel
// ============================================================================
//
// Each thread block processes a tile of Br query rows.
// For each block of Bc key/value columns, it:
//   1. Loads Q tile [Br, d] and K tile [Bc, d] into shared memory
//   2. Computes S = Q @ K^T tile [Br, Bc]
//   3. Computes running softmax statistics (online softmax)
//   4. Loads V tile [Bc, d] and accumulates O += softmax(S) @ V
//
// After all K/V blocks, O is scaled by final softmax denominator.
//
// Template params:
//   Br: number of query rows per block
//   Bc: number of key cols per inner-loop tile
//   D:  head dimension (compile-time constant for register allocation)

template <int Br, int Bc, int D>
__global__ void flash_attention_forward_kernel(
    const float* __restrict__ Q,    // [B*H, N, D]
    const float* __restrict__ K,    // [B*H, N_k, D]
    const float* __restrict__ V,    // [B*H, N_k, D]
    float* __restrict__ O,          // [B*H, N, D]
    int N,                          // query sequence length
    int N_k,                        // key/value sequence length
    float scale                     // 1/sqrt(D)
) {
    const int bh = blockIdx.y;      // batch * head index
    const int block_row = blockIdx.x;  // which Br-tile of queries
    const int row_start = block_row * Br;

    if (row_start >= N) return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;  // typically 128 or 256

    // Pointers for this batch-head
    const float* q_bh = Q + (int64_t)bh * N * D;
    const float* k_bh = K + (int64_t)bh * N_k * D;
    const float* v_bh = V + (int64_t)bh * N_k * D;
    float* o_bh = O + (int64_t)bh * N * D;

    // Shared memory layout:
    // [Br * D] for Q tile
    // [Bc * D] for K tile (reused for V tile)
    // [Br * Bc] for S (attention scores tile)
    extern __shared__ float smem[];
    float* q_smem = smem;                          // [Br, D]
    float* kv_smem = smem + Br * D;                // [Bc, D]
    float* s_smem = smem + Br * D + Bc * D;        // [Br, Bc]

    // Register accumulators per row (each thread handles a subset of rows)
    // Each thread handles (Br / num_threads_per_row) rows
    // For simplicity: each thread processes elements across D dimension
    // and we iterate over rows in the thread block.

    // Per-row online softmax state: m (max), l (sum of exp), O accumulator
    // We store these in registers for the rows this thread is responsible for
    float row_m[Br];    // running max for each row
    float row_l[Br];    // running sum of exp for each row
    float row_o[Br * D / 1]; // We can't have Br*D registers... need a different approach

    // ---- Simplified approach: each thread handles specific elements ----
    // Thread layout: each thread handles certain (row, d) pairs
    // rows_per_thread = ceil(Br / (num_threads / D))
    // But D might not divide num_threads evenly.

    // Let's use a simpler approach: each thread handles multiple elements
    // and we use shared memory for intermediate results.

    // Initialize per-row accumulators in shared memory
    // We'll use a different strategy: process in warps

    // ---- Load Q tile into shared memory ----
    for (int i = tid; i < Br * D; i += num_threads) {
        int r = i / D;
        int d = i % D;
        int global_row = row_start + r;
        q_smem[r * D + d] = (global_row < N) ? q_bh[global_row * D + d] * scale : 0.0f;
    }
    __syncthreads();

    // Initialize output accumulator in shared memory
    // We repurpose a portion after Q is loaded, or use a separate region
    // Actually, let's keep O in registers per thread
    // Each thread will accumulate for specific (row, d) pairs

    // Output accumulator: use shared memory region after s_smem
    // Total smem so far: Br*D + Bc*D + Br*Bc
    float* o_smem = s_smem + Br * Bc;  // [Br, D]

    // Initialize O and softmax stats
    for (int i = tid; i < Br * D; i += num_threads) {
        o_smem[i] = 0.0f;
    }

    // m and l arrays in shared memory
    float* m_smem = o_smem + Br * D;    // [Br]
    float* l_smem = m_smem + Br;        // [Br]

    for (int i = tid; i < Br; i += num_threads) {
        m_smem[i] = -FLT_MAX;
        l_smem[i] = 0.0f;
    }
    __syncthreads();

    // ---- Iterate over K/V blocks ----
    int num_kv_blocks = (N_k + Bc - 1) / Bc;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int col_start = kv_block * Bc;

        // Load K tile [Bc, D] into kv_smem
        for (int i = tid; i < Bc * D; i += num_threads) {
            int r = i / D;
            int d = i % D;
            int global_col = col_start + r;
            kv_smem[r * D + d] = (global_col < N_k) ? k_bh[global_col * D + d] : 0.0f;
        }
        __syncthreads();

        // Compute S = Q_tile @ K_tile^T  -> S[Br, Bc]
        for (int idx = tid; idx < Br * Bc; idx += num_threads) {
            int r = idx / Bc;
            int c = idx % Bc;

            int global_row = row_start + r;
            int global_col = col_start + c;

            float sum = 0.0f;
            if (global_row < N && global_col < N_k) {
                // Q is already scaled
                for (int d = 0; d < D; d++) {
                    sum += q_smem[r * D + d] * kv_smem[c * D + d];
                }
            } else {
                sum = -FLT_MAX;
            }
            s_smem[r * Bc + c] = sum;
        }
        __syncthreads();

        // ---- Online softmax: update m and l per row ----
        // For each row: m_new = max(m_old, max(S_row))
        //               l_new = l_old * exp(m_old - m_new) + sum(exp(S_row - m_new))
        //               O_new = O_old * exp(m_old - m_new) / 1  + exp(S - m_new) @ V_tile
        // Actually: O_new = O_old * (l_old * exp(m_old - m_new) / l_new) + (exp(S - m_new) @ V) / l_new
        // We defer the /l_new to the very end.

        // Step 1: Find new max per row
        for (int r = tid; r < Br; r += num_threads) {
            if (row_start + r >= N) continue;
            float old_m = m_smem[r];
            float new_m = old_m;
            for (int c = 0; c < Bc; c++) {
                float v = s_smem[r * Bc + c];
                if (v > new_m) new_m = v;
            }

            // Step 2: Compute correction factor and new sum
            float correction = expf(old_m - new_m);
            float old_l = l_smem[r];
            float new_l = old_l * correction;

            // Compute exp(S - new_m) for this row and accumulate new_l
            float row_sum = 0.0f;
            for (int c = 0; c < Bc; c++) {
                float e = expf(s_smem[r * Bc + c] - new_m);
                s_smem[r * Bc + c] = e;  // store exp(S - m) for V accumulation
                row_sum += e;
            }
            new_l += row_sum;

            // Step 3: Rescale existing O accumulator
            for (int d = 0; d < D; d++) {
                o_smem[r * D + d] *= correction;
            }

            m_smem[r] = new_m;
            l_smem[r] = new_l;
        }
        __syncthreads();

        // Load V tile into kv_smem (reuse K tile memory)
        for (int i = tid; i < Bc * D; i += num_threads) {
            int r = i / D;
            int d = i % D;
            int global_col = col_start + r;
            kv_smem[r * D + d] = (global_col < N_k) ? v_bh[global_col * D + d] : 0.0f;
        }
        __syncthreads();

        // Accumulate O += exp(S - m) @ V_tile
        // O[r, d] += sum_c(s_smem[r, c] * v_smem[c, d])
        for (int idx = tid; idx < Br * D; idx += num_threads) {
            int r = idx / D;
            int d = idx % D;
            if (row_start + r >= N) continue;

            float sum = 0.0f;
            for (int c = 0; c < Bc; c++) {
                sum += s_smem[r * Bc + c] * kv_smem[c * D + d];
            }
            o_smem[r * D + d] += sum;
        }
        __syncthreads();
    }

    // ---- Final scaling: O /= l ----
    for (int idx = tid; idx < Br * D; idx += num_threads) {
        int r = idx / D;
        int d = idx % D;
        int global_row = row_start + r;
        if (global_row < N) {
            float l = l_smem[r];
            if (l > 0.0f) {
                o_bh[global_row * D + d] = o_smem[r * D + d] / l;
            } else {
                o_bh[global_row * D + d] = 0.0f;
            }
        }
    }
}

// ============================================================================
// Launch wrapper
// ============================================================================

inline void launch_flash_attention(
    const float* Q, const float* K, const float* V, float* O,
    int BH, int N, int N_k, int D, float scale
) {
    // Choose tile sizes based on head dimension
    // For typical D=64 or D=128 in music sep models

    if (D <= 64) {
        // Tile sizes chosen so shared memory fits within 48KB default limit
        // smem = (Br*D + Bc*D + Br*Bc + Br*D + 2*Br) * 4
        // = (32*64 + 64*64 + 32*64 + 32*64 + 64) * 4 = 41216 bytes < 48KB
        constexpr int Br = 32;
        constexpr int Bc = 64;
        constexpr int Dmax = 64;

        int num_blocks_x = (N + Br - 1) / Br;
        dim3 grid(num_blocks_x, BH);
        int threads = 256;

        size_t smem = sizeof(float) * (Br * Dmax + Bc * Dmax + Br * Bc + Br * Dmax + Br + Br);

        flash_attention_forward_kernel<Br, Bc, Dmax><<<grid, threads, smem>>>(
            Q, K, V, O, N, N_k, scale);
    } else {
        // D <= 128
        // smem = (16*128 + 32*128 + 16*32 + 16*128 + 32) * 4 = 34944 bytes < 48KB
        constexpr int Br = 16;
        constexpr int Bc = 32;
        constexpr int Dmax = 128;

        int num_blocks_x = (N + Br - 1) / Br;
        dim3 grid(num_blocks_x, BH);
        int threads = 128;

        size_t smem = sizeof(float) * (Br * Dmax + Bc * Dmax + Br * Bc + Br * Dmax + Br + Br);

        flash_attention_forward_kernel<Br, Bc, Dmax><<<grid, threads, smem>>>(
            Q, K, V, O, N, N_k, scale);
    }
}

} // namespace cudasep
