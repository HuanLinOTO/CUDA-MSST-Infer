// ops_norm.cu - Normalization operations for cudasep inference engine.
//
// Implements: rms_norm, layer_norm, group_norm, instance_norm, batch_norm.
// All ops work on Float32 internally. Float16 inputs are cast to f32, computed,
// then cast back.

#include "ops.h"
#include <cmath>
#include <cfloat>
#include <cassert>

namespace cudasep {

// ---------------------------------------------------------------------------
// Helpers (same conventions as ops_elementwise.cu)
// ---------------------------------------------------------------------------

static constexpr int kBlockSize = 256;

static inline int64_t ceildiv(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

static Tensor ensure_f32(const Tensor& x) {
    if (x.dtype() == DType::Float32) return x.contiguous();
    return x.to_f32().contiguous();
}

static Tensor maybe_cast_back(const Tensor& result, DType orig) {
    if (orig == DType::Float16) return result.to_f16();
    return result;
}

// ---------------------------------------------------------------------------
// Warp-level reduction utilities
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_sum_norm(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// ---------------------------------------------------------------------------
// Block-level reduction using shared memory
// ---------------------------------------------------------------------------

// Reduces across all threads in a block. smem must have size >= blockDim.x.
// After call, result is broadcast to all threads via smem[0].
__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int num_warps = (blockDim.x + 31) >> 5;

    // Intra-warp reduction
    val = warp_reduce_sum_norm(val);

    // Write warp results to shared memory
    if (lane == 0) smem[warp_id] = val;
    __syncthreads();

    // First warp reduces across warps
    if (warp_id == 0) {
        float v = (lane < num_warps) ? smem[lane] : 0.0f;
        v = warp_reduce_sum_norm(v);
        if (lane == 0) smem[0] = v;
    }
    __syncthreads();

    return smem[0];
}

// ===========================================================================
// 1. RMS Norm
// ===========================================================================
// x: [..., D], gamma: [D], scale: float
// result = x * rsqrt(mean(x^2, dim=-1)) * gamma * scale

// --- Warp-only kernel (D <= 1024, one warp per row) ---
__global__ void rms_norm_warp_kernel(const float* __restrict__ x,
                                     const float* __restrict__ gamma,
                                     float* __restrict__ out,
                                     int D, float scale, int64_t num_rows) {
    // One warp per row. Grid: ceil(num_rows / warps_per_block)
    int global_warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (global_warp >= num_rows) return;

    const float* row_in = x + global_warp * D;
    float* row_out = out + global_warp * D;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int j = lane; j < D; j += 32) {
        float v = row_in[j];
        sum_sq += v * v;
    }
    sum_sq = warp_reduce_sum_norm(sum_sq);

    // rsqrt(mean(x^2))
    float rms_inv = rsqrtf(sum_sq / (float)D + 1e-8f);

    // Apply normalization
    for (int j = lane; j < D; j += 32) {
        row_out[j] = row_in[j] * rms_inv * gamma[j] * scale;
    }
}

// --- Block-level kernel (D > 1024, one block per row) ---
__global__ void rms_norm_block_kernel(const float* __restrict__ x,
                                      const float* __restrict__ gamma,
                                      float* __restrict__ out,
                                      int D, float scale, int64_t num_rows) {
    int row = blockIdx.x;
    if (row >= num_rows) return;

    extern __shared__ float smem[];
    int tid = threadIdx.x;

    const float* row_in = x + (int64_t)row * D;
    float* row_out = out + (int64_t)row * D;

    // Compute partial sum of squares
    float sum_sq = 0.0f;
    for (int j = tid; j < D; j += blockDim.x) {
        float v = row_in[j];
        sum_sq += v * v;
    }

    sum_sq = block_reduce_sum(sum_sq, smem);
    float rms_inv = rsqrtf(sum_sq / (float)D + 1e-8f);

    // Apply normalization
    for (int j = tid; j < D; j += blockDim.x) {
        row_out[j] = row_in[j] * rms_inv * gamma[j] * scale;
    }
}

// ===========================================================================
// 2. Layer Norm
// ===========================================================================
// x: [..., D], gamma: [D], beta: [D]
// y = (x - mean) / sqrt(var + eps) * gamma + beta

// --- Warp-only kernel (D <= 1024) ---
__global__ void layer_norm_warp_kernel(const float* __restrict__ x,
                                       const float* __restrict__ gamma,
                                       const float* __restrict__ beta,
                                       float* __restrict__ out,
                                       int D, float eps, int64_t num_rows) {
    int global_warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (global_warp >= num_rows) return;

    const float* row_in = x + global_warp * D;
    float* row_out = out + global_warp * D;

    // Pass 1: compute mean
    float local_sum = 0.0f;
    for (int j = lane; j < D; j += 32) {
        local_sum += row_in[j];
    }
    float mean = warp_reduce_sum_norm(local_sum) / (float)D;

    // Pass 2: compute variance
    float local_var = 0.0f;
    for (int j = lane; j < D; j += 32) {
        float diff = row_in[j] - mean;
        local_var += diff * diff;
    }
    float var = warp_reduce_sum_norm(local_var) / (float)D;
    float inv_std = rsqrtf(var + eps);

    // Pass 3: normalize
    for (int j = lane; j < D; j += 32) {
        row_out[j] = (row_in[j] - mean) * inv_std * gamma[j] + beta[j];
    }
}

// --- Block-level kernel (D > 1024) ---
__global__ void layer_norm_block_kernel(const float* __restrict__ x,
                                        const float* __restrict__ gamma,
                                        const float* __restrict__ beta,
                                        float* __restrict__ out,
                                        int D, float eps, int64_t num_rows) {
    int row = blockIdx.x;
    if (row >= num_rows) return;

    extern __shared__ float smem[];
    int tid = threadIdx.x;

    const float* row_in = x + (int64_t)row * D;
    float* row_out = out + (int64_t)row * D;

    // Pass 1: compute mean
    float local_sum = 0.0f;
    for (int j = tid; j < D; j += blockDim.x) {
        local_sum += row_in[j];
    }
    float mean = block_reduce_sum(local_sum, smem) / (float)D;

    // Pass 2: compute variance
    float local_var = 0.0f;
    for (int j = tid; j < D; j += blockDim.x) {
        float diff = row_in[j] - mean;
        local_var += diff * diff;
    }
    float var = block_reduce_sum(local_var, smem) / (float)D;
    float inv_std = rsqrtf(var + eps);

    // Pass 3: normalize
    for (int j = tid; j < D; j += blockDim.x) {
        row_out[j] = (row_in[j] - mean) * inv_std * gamma[j] + beta[j];
    }
}

// ===========================================================================
// 3. Group Norm
// ===========================================================================
// x: [B, C, ...spatial...], gamma: [C], beta: [C]
// Split C into num_groups, normalize within each group.
// Each group covers (C / num_groups) channels and all spatial dims.
//
// One block per (batch, group). Each thread loops over the elements in
// that group to compute mean and variance, then normalizes.

__global__ void group_norm_kernel(const float* __restrict__ x,
                                  const float* __restrict__ gamma,
                                  const float* __restrict__ beta,
                                  float* __restrict__ out,
                                  int B, int C, int64_t spatial,
                                  int num_groups, float eps) {
    // blockIdx.x = batch * num_groups + group
    int bg = blockIdx.x;
    int b = bg / num_groups;
    int g = bg % num_groups;
    if (b >= B) return;

    extern __shared__ float smem[];
    int tid = threadIdx.x;

    int channels_per_group = C / num_groups;
    int c_start = g * channels_per_group;
    int c_end = c_start + channels_per_group;
    int64_t group_size = (int64_t)channels_per_group * spatial;

    // Pointer to batch start
    const float* batch_in = x + (int64_t)b * C * spatial;
    float* batch_out = out + (int64_t)b * C * spatial;

    // Pass 1: compute mean over the group
    float local_sum = 0.0f;
    for (int64_t idx = tid; idx < group_size; idx += blockDim.x) {
        int local_c = (int)(idx / spatial);  // channel within group
        int64_t s = idx % spatial;
        int c = c_start + local_c;
        local_sum += batch_in[(int64_t)c * spatial + s];
    }
    float mean = block_reduce_sum(local_sum, smem) / (float)group_size;

    // Pass 2: compute variance
    float local_var = 0.0f;
    for (int64_t idx = tid; idx < group_size; idx += blockDim.x) {
        int local_c = (int)(idx / spatial);
        int64_t s = idx % spatial;
        int c = c_start + local_c;
        float diff = batch_in[(int64_t)c * spatial + s] - mean;
        local_var += diff * diff;
    }
    float var = block_reduce_sum(local_var, smem) / (float)group_size;
    float inv_std = rsqrtf(var + eps);

    // Pass 3: normalize with per-channel affine
    for (int64_t idx = tid; idx < group_size; idx += blockDim.x) {
        int local_c = (int)(idx / spatial);
        int64_t s = idx % spatial;
        int c = c_start + local_c;
        int64_t linear = (int64_t)c * spatial + s;
        float normalized = (batch_in[linear] - mean) * inv_std;
        batch_out[linear] = normalized * gamma[c] + beta[c];
    }
}

// ===========================================================================
// 3b. Multi-block GroupNorm for num_groups=1 (large data)
// ===========================================================================
// When num_groups=1, the standard kernel uses only 1 CUDA block per batch,
// which is catastrophically slow for large C*spatial (e.g., 96*85995 = 8.3M).
// We split into 3 kernels using full GPU parallelism.

// Kernel 1: Each block computes partial sum and sum_sq over a chunk of elements.
// Grid: (reduce_blocks, B)
__global__ void gn1_reduce_kernel(const float* __restrict__ x,
                                   float* __restrict__ partials,
                                   int C, int64_t spatial,
                                   int reduce_blocks) {
    int bid = blockIdx.x;       // block within this batch's reduction
    int batch = blockIdx.y;
    int64_t N = (int64_t)C * spatial;

    const float* bx = x + (int64_t)batch * N;

    // Each block processes a contiguous chunk of the C*spatial elements
    int64_t chunk = (N + reduce_blocks - 1) / reduce_blocks;
    int64_t start = (int64_t)bid * chunk;
    int64_t end = start + chunk;
    if (end > N) end = N;

    extern __shared__ float smem[];

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (int64_t i = start + threadIdx.x; i < end; i += blockDim.x) {
        float v = bx[i];
        local_sum += v;
        local_sum_sq += v * v;
    }

    // Block-reduce sum
    local_sum = block_reduce_sum(local_sum, smem);

    // Block-reduce sum_sq (reuse smem after sync)
    local_sum_sq = block_reduce_sum(local_sum_sq, smem);

    if (threadIdx.x == 0) {
        int idx = batch * reduce_blocks + bid;
        partials[idx * 2]     = local_sum;
        partials[idx * 2 + 1] = local_sum_sq;
    }
}

// Kernel 2: Reduce partial sums and compute mean + inv_std per batch element.
// Grid: (B), Threads: 256
// Output: stats[batch * 2] = mean, stats[batch * 2 + 1] = inv_std
__global__ void gn1_stats_kernel(const float* __restrict__ partials,
                                  float* __restrict__ stats,
                                  int reduce_blocks, int64_t N, float eps) {
    int batch = blockIdx.x;
    float total_sum = 0.0f;
    float total_sum_sq = 0.0f;

    // Each thread processes some partial blocks
    for (int i = threadIdx.x; i < reduce_blocks; i += blockDim.x) {
        int idx = batch * reduce_blocks + i;
        total_sum += partials[idx * 2];
        total_sum_sq += partials[idx * 2 + 1];
    }

    extern __shared__ float smem[];
    total_sum = block_reduce_sum(total_sum, smem);
    total_sum_sq = block_reduce_sum(total_sum_sq, smem);

    if (threadIdx.x == 0) {
        float mean = total_sum / (float)N;
        float var = total_sum_sq / (float)N - mean * mean;
        stats[batch * 2]     = mean;
        stats[batch * 2 + 1] = rsqrtf(var + eps);
    }
}

// Kernel 3: Normalize using precomputed mean/inv_std with per-channel gamma/beta.
// Grid: (norm_blocks, B). Data layout: [B, C, spatial]
__global__ void gn1_normalize_kernel(const float* __restrict__ x,
                                      float* __restrict__ out,
                                      const float* __restrict__ gamma,
                                      const float* __restrict__ beta,
                                      const float* __restrict__ stats,
                                      int C, int64_t spatial) {
    int batch = blockIdx.y;
    int64_t N = (int64_t)C * spatial;

    float mean    = stats[batch * 2];
    float inv_std = stats[batch * 2 + 1];

    const float* bx = x + (int64_t)batch * N;
    float* bout = out + (int64_t)batch * N;

    // Grid-stride loop over elements
    int64_t global_tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = global_tid; i < N; i += grid_stride) {
        int c = (int)(i / spatial);
        float normalized = (bx[i] - mean) * inv_std;
        bout[i] = normalized * gamma[c] + beta[c];
    }
}

// Kernel 3b: Normalize for [B, T, C] layout (channel = last dim).
// Grid: (norm_blocks, B).
__global__ void gn1_normalize_btc_kernel(const float* __restrict__ x,
                                          float* __restrict__ out,
                                          const float* __restrict__ gamma,
                                          const float* __restrict__ beta,
                                          const float* __restrict__ stats,
                                          int C, int64_t total_per_batch) {
    int batch = blockIdx.y;

    float mean    = stats[batch * 2];
    float inv_std = stats[batch * 2 + 1];

    const float* bx = x + (int64_t)batch * total_per_batch;
    float* bout = out + (int64_t)batch * total_per_batch;

    int64_t global_tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = global_tid; i < total_per_batch; i += grid_stride) {
        int c = (int)(i % C);  // channel is the LAST dim for BTC layout
        float normalized = (bx[i] - mean) * inv_std;
        bout[i] = normalized * gamma[c] + beta[c];
    }
}

// ===========================================================================
// 4. Batch Norm (inference mode)
// ===========================================================================
// x: [B, C, ...spatial...], running_mean: [C], running_var: [C],
// gamma: [C], beta: [C]
// y = (x - running_mean[c]) / sqrt(running_var[c] + eps) * gamma[c] + beta[c]
// Simple element-wise kernel since we use running stats.

__global__ void batch_norm_kernel(const float* __restrict__ x,
                                  const float* __restrict__ running_mean,
                                  const float* __restrict__ running_var,
                                  const float* __restrict__ gamma,
                                  const float* __restrict__ beta,
                                  float* __restrict__ out,
                                  int C, int64_t spatial, float eps,
                                  int64_t total) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    // Determine channel index: x is [B, C, spatial...]
    // linear index = b * C * spatial + c * spatial + s
    int c = (int)((i / spatial) % C);

    float inv_std = rsqrtf(running_var[c] + eps);
    out[i] = (x[i] - running_mean[c]) * inv_std * gamma[c] + beta[c];
}

// ===========================================================================
// Public API
// ===========================================================================

namespace ops {

// ---------------------------------------------------------------------------
// rms_norm
// ---------------------------------------------------------------------------
Tensor rms_norm(const Tensor& x, const Tensor& gamma, float scale) {
    DType orig = x.dtype();
    Tensor xf = ensure_f32(x);
    Tensor gf = ensure_f32(gamma);

    int ndim = xf.ndim();
    assert(ndim >= 1);
    int D = (int)xf.size(ndim - 1);
    int64_t num_rows = xf.numel() / D;

    assert(gf.numel() == D);

    Tensor out = Tensor::empty(xf.shape(), DType::Float32);

    if (D <= 1024) {
        // Warp-based: one warp (32 threads) per row
        int warps_per_block = kBlockSize / 32; // 8 warps per block
        int threads = warps_per_block * 32;    // = kBlockSize
        int num_blocks = (int)ceildiv(num_rows, (int64_t)warps_per_block);

        rms_norm_warp_kernel<<<num_blocks, threads>>>(
            xf.data_f32(), gf.data_f32(), out.data_f32(),
            D, scale, num_rows);
    } else {
        // Block-based: one block per row
        int threads = kBlockSize;
        int num_warps = (threads + 31) / 32;
        size_t smem_bytes = num_warps * sizeof(float);

        rms_norm_block_kernel<<<(int)num_rows, threads, smem_bytes>>>(
            xf.data_f32(), gf.data_f32(), out.data_f32(),
            D, scale, num_rows);
    }
    CUDA_CHECK(cudaGetLastError());

    return maybe_cast_back(out, orig);
}

// ---------------------------------------------------------------------------
// layer_norm
// ---------------------------------------------------------------------------
Tensor layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps) {
    DType orig = x.dtype();
    Tensor xf = ensure_f32(x);
    Tensor gf = ensure_f32(gamma);
    Tensor bf = ensure_f32(beta);

    int ndim = xf.ndim();
    assert(ndim >= 1);
    int D = (int)xf.size(ndim - 1);
    int64_t num_rows = xf.numel() / D;

    assert(gf.numel() == D);
    assert(bf.numel() == D);

    Tensor out = Tensor::empty(xf.shape(), DType::Float32);

    if (D <= 1024) {
        // Warp-based: one warp per row
        int warps_per_block = kBlockSize / 32;
        int threads = warps_per_block * 32;
        int num_blocks = (int)ceildiv(num_rows, (int64_t)warps_per_block);

        layer_norm_warp_kernel<<<num_blocks, threads>>>(
            xf.data_f32(), gf.data_f32(), bf.data_f32(), out.data_f32(),
            D, eps, num_rows);
    } else {
        // Block-based: one block per row
        int threads = kBlockSize;
        int num_warps = (threads + 31) / 32;
        size_t smem_bytes = num_warps * sizeof(float);

        layer_norm_block_kernel<<<(int)num_rows, threads, smem_bytes>>>(
            xf.data_f32(), gf.data_f32(), bf.data_f32(), out.data_f32(),
            D, eps, num_rows);
    }
    CUDA_CHECK(cudaGetLastError());

    return maybe_cast_back(out, orig);
}

// ---------------------------------------------------------------------------
// group_norm
// ---------------------------------------------------------------------------
Tensor group_norm(const Tensor& x, int num_groups, const Tensor& gamma,
                  const Tensor& beta, float eps) {
    DType orig = x.dtype();
    Tensor xf = ensure_f32(x);
    Tensor gf = ensure_f32(gamma);
    Tensor bf = ensure_f32(beta);

    int ndim = xf.ndim();
    assert(ndim >= 2); // at least [B, C]
    int B = (int)xf.size(0);
    int C = (int)xf.size(1);
    assert(C % num_groups == 0);
    assert(gf.numel() == C);
    assert(bf.numel() == C);

    // Compute spatial dimensions: product of dims [2..ndim-1]
    int64_t spatial = 1;
    for (int i = 2; i < ndim; ++i) spatial *= xf.size(i);

    Tensor out = Tensor::empty(xf.shape(), DType::Float32);

    int64_t N = (int64_t)C * spatial;

    // Use multi-block path for num_groups=1 when data is large
    // Threshold: if each group has > 32K elements, multi-block is worthwhile
    if (num_groups == 1 && N > 32768) {
        // Multi-block GroupNorm(1): 3 kernel launches, full GPU parallelism
        int threads = kBlockSize;
        int num_warps = (threads + 31) / 32;
        size_t smem_bytes = num_warps * sizeof(float);

        // Choose number of reduction blocks to get good parallelism
        // Each block should process at least 1024 elements for efficiency
        int reduce_blocks = (int)std::min((int64_t)256, (N + 1023) / 1024);
        if (reduce_blocks < 1) reduce_blocks = 1;

        // Allocate temp buffers for partial sums and stats
        Tensor partials = Tensor::empty({(int64_t)B * reduce_blocks * 2}, DType::Float32);
        Tensor stats    = Tensor::empty({(int64_t)B * 2}, DType::Float32);

        // Kernel 1: Parallel reduction
        dim3 grid1(reduce_blocks, B);
        gn1_reduce_kernel<<<grid1, threads, smem_bytes>>>(
            xf.data_f32(), partials.data_f32(), C, spatial, reduce_blocks);

        // Kernel 2: Compute stats (mean, inv_std) per batch
        int stats_threads = std::min(reduce_blocks, threads);
        // Round to power of 2 for reduction
        int stats_threads_po2 = 1;
        while (stats_threads_po2 < stats_threads) stats_threads_po2 <<= 1;
        if (stats_threads_po2 > kBlockSize) stats_threads_po2 = kBlockSize;
        size_t stats_smem = ((stats_threads_po2 + 31) / 32) * sizeof(float);
        gn1_stats_kernel<<<B, stats_threads_po2, stats_smem>>>(
            partials.data_f32(), stats.data_f32(), reduce_blocks, N, eps);

        // Kernel 3: Normalize with per-channel gamma/beta
        int norm_blocks = (int)std::min((int64_t)1024, (N + threads - 1) / threads);
        dim3 grid3(norm_blocks, B);
        gn1_normalize_kernel<<<grid3, threads>>>(
            xf.data_f32(), out.data_f32(),
            gf.data_f32(), bf.data_f32(),
            stats.data_f32(), C, spatial);
    } else {
        // Standard single-block-per-group path
        int total_blocks = B * num_groups;
        int threads = kBlockSize;
        int num_warps = (threads + 31) / 32;
        size_t smem_bytes = num_warps * sizeof(float);

        group_norm_kernel<<<total_blocks, threads, smem_bytes>>>(
            xf.data_f32(), gf.data_f32(), bf.data_f32(), out.data_f32(),
            B, C, spatial, num_groups, eps);
    }
    CUDA_CHECK(cudaGetLastError());

    return maybe_cast_back(out, orig);
}

// ---------------------------------------------------------------------------
// instance_norm  (group_norm with num_groups = C)
// ---------------------------------------------------------------------------
Tensor instance_norm(const Tensor& x, const Tensor& gamma,
                     const Tensor& beta, float eps) {
    assert(x.ndim() >= 2);
    int C = (int)x.size(1);
    return group_norm(x, C, gamma, beta, eps);
}

// ---------------------------------------------------------------------------
// group_norm_1_btc — GroupNorm(1) directly on [B, T, C] layout
// ---------------------------------------------------------------------------
// Equivalent to: transpose(1,2).contiguous() → group_norm(1,...) → transpose(1,2).contiguous()
// but without the 2 transpose copies.
Tensor group_norm_1_btc(const Tensor& x, const Tensor& gamma,
                        const Tensor& beta, float eps) {
    DType orig = x.dtype();
    Tensor xf = ensure_f32(x);
    Tensor gf = ensure_f32(gamma);
    Tensor bf = ensure_f32(beta);

    assert(xf.ndim() == 3); // [B, T, C]
    int B = (int)xf.size(0);
    int64_t T = xf.size(1);
    int C = (int)xf.size(2);
    int64_t N = T * C;  // total elements per batch

    assert(gf.numel() == C);
    assert(bf.numel() == C);

    Tensor out = Tensor::empty(xf.shape(), DType::Float32);

    int threads = kBlockSize;
    int num_warps = (threads + 31) / 32;
    size_t smem_bytes = num_warps * sizeof(float);

    // Same reduce + stats as standard multi-block group_norm(1)
    // The reduce kernel doesn't care about data layout (just sums elements)
    int reduce_blocks = (int)std::min((int64_t)256, (N + 1023) / 1024);
    if (reduce_blocks < 1) reduce_blocks = 1;

    Tensor partials = Tensor::empty({(int64_t)B * reduce_blocks * 2}, DType::Float32);
    Tensor stats    = Tensor::empty({(int64_t)B * 2}, DType::Float32);

    // Kernel 1: Parallel reduction (layout-agnostic)
    dim3 grid1(reduce_blocks, B);
    gn1_reduce_kernel<<<grid1, threads, smem_bytes>>>(
        xf.data_f32(), partials.data_f32(), C, T, reduce_blocks);

    // Kernel 2: Compute stats
    int stats_threads_po2 = 1;
    while (stats_threads_po2 < reduce_blocks && stats_threads_po2 < kBlockSize) stats_threads_po2 <<= 1;
    size_t stats_smem = ((stats_threads_po2 + 31) / 32) * sizeof(float);
    gn1_stats_kernel<<<B, stats_threads_po2, stats_smem>>>(
        partials.data_f32(), stats.data_f32(), reduce_blocks, N, eps);

    // Kernel 3: Normalize with BTC layout (c = i % C)
    int norm_blocks = (int)std::min((int64_t)1024, (N + threads - 1) / threads);
    dim3 grid3(norm_blocks, B);
    gn1_normalize_btc_kernel<<<grid3, threads>>>(
        xf.data_f32(), out.data_f32(),
        gf.data_f32(), bf.data_f32(),
        stats.data_f32(), C, N);

    CUDA_CHECK(cudaGetLastError());
    return maybe_cast_back(out, orig);
}

// ---------------------------------------------------------------------------
// batch_norm (inference mode — uses running statistics)
// ---------------------------------------------------------------------------
Tensor batch_norm(const Tensor& x, const Tensor& mean, const Tensor& var,
                  const Tensor& gamma, const Tensor& beta, float eps) {
    DType orig = x.dtype();
    Tensor xf = ensure_f32(x);
    Tensor mf = ensure_f32(mean);
    Tensor vf = ensure_f32(var);
    Tensor gf = ensure_f32(gamma);
    Tensor bf = ensure_f32(beta);

    int ndim = xf.ndim();
    assert(ndim >= 2);
    int C = (int)xf.size(1);
    assert(mf.numel() == C);
    assert(vf.numel() == C);
    assert(gf.numel() == C);
    assert(bf.numel() == C);

    // spatial = product of dims [2..]
    int64_t spatial = 1;
    for (int i = 2; i < ndim; ++i) spatial *= xf.size(i);

    int64_t total = xf.numel();
    Tensor out = Tensor::empty(xf.shape(), DType::Float32);

    int grid = (int)ceildiv(total, (int64_t)kBlockSize);
    batch_norm_kernel<<<grid, kBlockSize>>>(
        xf.data_f32(), mf.data_f32(), vf.data_f32(),
        gf.data_f32(), bf.data_f32(), out.data_f32(),
        C, spatial, eps, total);
    CUDA_CHECK(cudaGetLastError());

    return maybe_cast_back(out, orig);
}

} // namespace ops
} // namespace cudasep
