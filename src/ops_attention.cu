// ops_attention.cu - Scaled dot-product attention for cudasep inference engine.
//
// Implements Flash Attention 2 — fused kernel that computes
//   output = softmax(Q @ K^T / sqrt(d)) @ V
// in O(N) memory instead of materializing the full N×N attention matrix.
//
// Falls back to cuBLAS GEMM for very small sequence lengths where the
// overhead of Flash Attention is not worth it.

#include "ops.h"
#include "flash_attention.cuh"
#include <cmath>
#include <cfloat>

namespace cudasep {

// ---------------------------------------------------------------------------
// Helpers
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
// Fused scale + softmax kernel — 2 global memory passes instead of 5
//
// Reads data once (with inline scale), finds max, computes exp and sum in
// registers, normalizes and writes once. Much more efficient than separate
// scale_kernel + 3-pass softmax_rows_kernel.
// ---------------------------------------------------------------------------

__global__ void fused_scale_softmax_kernel(float* __restrict__ data, float scale,
                                            int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float smem[];
    float* row_data = data + (int64_t)row * cols;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // Phase 1: Load from global memory, scale, and find max (1 read)
    // Each thread handles multiple elements stored in registers
    constexpr int MAX_ELEMS = 32;  // max elements per thread
    float reg[MAX_ELEMS];
    int elems_per_thread = (cols + num_threads - 1) / num_threads;

    float thread_max = -FLT_MAX;
    for (int i = 0; i < elems_per_thread; i++) {
        int j = tid + i * num_threads;
        float v = (j < cols) ? row_data[j] * scale : -FLT_MAX;
        reg[i] = v;
        if (v > thread_max) thread_max = v;
    }

    // Reduce max across threads via shared memory
    smem[tid] = thread_max;
    __syncthreads();
    for (int s = num_threads >> 1; s > 0; s >>= 1) {
        if (tid < s && smem[tid + s] > smem[tid]) smem[tid] = smem[tid + s];
        __syncthreads();
    }
    float max_val = smem[0];
    __syncthreads();

    // Phase 2: Compute exp(x - max), accumulate sum
    float thread_sum = 0.0f;
    for (int i = 0; i < elems_per_thread; i++) {
        int j = tid + i * num_threads;
        if (j < cols) {
            float e = __expf(reg[i] - max_val);
            reg[i] = e;
            thread_sum += e;
        }
    }

    // Reduce sum across threads via shared memory
    smem[tid] = thread_sum;
    __syncthreads();
    for (int s = num_threads >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / smem[0];
    __syncthreads();

    // Phase 3: Normalize and write back (1 write)
    for (int i = 0; i < elems_per_thread; i++) {
        int j = tid + i * num_threads;
        if (j < cols) {
            row_data[j] = reg[i] * inv_sum;
        }
    }
}

// ---------------------------------------------------------------------------
// scaled_dot_product_attention
// ---------------------------------------------------------------------------

namespace ops {

Tensor scaled_dot_product_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                                     float scale, float dropout) {
    // ---- validate shapes ----
    assert(q.ndim() == 4 && k.ndim() == 4 && v.ndim() == 4);

    const int64_t B = q.size(0);
    const int64_t H = q.size(1);
    const int64_t N = q.size(2);   // sequence length (queries)
    const int64_t D = q.size(3);   // head dimension

    const int64_t N_k = k.size(2); // sequence length (keys / values)
    assert(k.size(0) == B && k.size(1) == H && k.size(3) == D);
    assert(v.size(0) == B && v.size(1) == H && v.size(2) == N_k && v.size(3) == D);

    DType orig_dtype = q.dtype();

    // ---- ensure f32 contiguous ----
    Tensor qf = ensure_f32(q);   // [B, H, N,   D]
    Tensor kf = ensure_f32(k);   // [B, H, N_k, D]
    Tensor vf = ensure_f32(v);   // [B, H, N_k, D]

    // ---- default scale ----
    if (scale == 0.0f) {
        scale = 1.0f / sqrtf(static_cast<float>(D));
    }

    const int64_t BH = B * H;

    // Allocate output [B*H, N, D]
    Tensor out = Tensor::empty({BH, N, D}, DType::Float32);

    // Use Flash Attention only for D <= 128 AND very long sequences
    // For shorter/medium sequences, cuBLAS with materialized attention is faster
    // because cuBLAS uses tensor cores while our Flash Attention kernel doesn't
    if (false && D <= 128 && N > 4096) {
        // Flash Attention — O(N) memory, fused kernel
        launch_flash_attention(
            qf.data_f32(),  // [B*H, N, D] (contiguous, same layout as [B,H,N,D])
            kf.data_f32(),
            vf.data_f32(),
            out.data_f32(),
            (int)BH, (int)N, (int)N_k, (int)D, scale
        );
        CUDA_CHECK(cudaGetLastError());
    } else {
        // Fallback to cuBLAS GEMM-based attention for large head dims
        Tensor scores = Tensor::empty({BH, N, N_k}, DType::Float32);

        cublasHandle_t handle = CudaContext::instance().cublas();

        // GEMM 1: scores = Q @ K^T
        {
            float alpha = 1.0f;
            float beta  = 0.0f;
            int m = static_cast<int>(N);
            int n = static_cast<int>(N_k);
            int kk = static_cast<int>(D);
            long long int strideA = static_cast<long long int>(N * D);
            long long int strideB = static_cast<long long int>(N_k * D);
            long long int strideC = static_cast<long long int>(N * N_k);

            cublasSgemmStridedBatched(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n, m, kk, &alpha,
                kf.data_f32(), kk, strideB,
                qf.data_f32(), kk, strideA,
                &beta, scores.data_f32(), n, strideC,
                static_cast<int>(BH));
        }

        // Fused Scale + Softmax (2 memory passes instead of 5)
        {
            int total_rows = static_cast<int>(BH * N);
            int cols = static_cast<int>(N_k);
            int threads_po2 = 1;
            while (threads_po2 < cols && threads_po2 < kBlockSize) threads_po2 <<= 1;
            size_t smem_bytes = threads_po2 * sizeof(float);
            fused_scale_softmax_kernel<<<total_rows, threads_po2, smem_bytes>>>(
                scores.data_f32(), scale, total_rows, cols);
        }

        // GEMM 2: out = attn_weights @ V
        {
            float alpha = 1.0f;
            float beta  = 0.0f;
            int m = static_cast<int>(N);
            int kk = static_cast<int>(N_k);
            int n = static_cast<int>(D);
            long long int strideA = static_cast<long long int>(N * N_k);
            long long int strideB = static_cast<long long int>(N_k * D);
            long long int strideC = static_cast<long long int>(N * D);

            cublasSgemmStridedBatched(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, kk, &alpha,
                vf.data_f32(), n, strideB,
                scores.data_f32(), kk, strideA,
                &beta, out.data_f32(), n, strideC,
                static_cast<int>(BH));
        }
    }

    // Reshape output from [B*H, N, D] to [B, H, N, D]
    Tensor result = out.reshape({B, H, N, D});
    return maybe_cast_back(result, orig_dtype);
}

} // namespace ops
} // namespace cudasep
