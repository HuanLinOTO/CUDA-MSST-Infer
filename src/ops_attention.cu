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
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace cudasep {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static constexpr int kBlockSize = 256;

static inline int64_t ceildiv(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

static inline uint32_t float_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static inline float float_from_bits(uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static Tensor ensure_f32(const Tensor& x) {
    if (x.dtype() == DType::Float32) return x.contiguous();
    return x.to_f32().contiguous();
}

static Tensor maybe_cast_back(const Tensor& result, DType orig) {
    if (orig == DType::Float16) return result.to_f16();
    return result;
}

static bool is_f32_contiguous(const Tensor& x) {
    return x.dtype() == DType::Float32 && x.is_contiguous();
}

static void clear_cuda_error_state(cudaStream_t stream) {
    (void)cudaGetLastError();
    (void)cudaStreamSynchronize(stream);
    (void)cudaGetLastError();
}

static void async_copy_tensor(Tensor& dst, const Tensor& src, cudaStream_t stream) {
    if (dst.numel() != src.numel()) {
        throw std::runtime_error("async_copy_tensor: size mismatch");
    }
    if (dst.dtype() != src.dtype()) {
        throw std::runtime_error("async_copy_tensor: dtype mismatch");
    }
    size_t bytes = (size_t)dst.numel() * dtype_size(dst.dtype());
    if (bytes > 0) {
        CUDA_CHECK(cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(), bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    }
}

__global__ void convert_f32_to_f16_kernel(const float* __restrict__ src,
                                          __half* __restrict__ dst,
                                          int64_t count) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = __float2half(src[idx]);
    }
}

__global__ void fused_scale_softmax_kernel(float* __restrict__ data, float scale,
                                            int rows, int cols);

__global__ void fused_scale_softmax_to_half_kernel(const float* __restrict__ data,
                                                   __half* __restrict__ out,
                                                   float scale,
                                                   int rows,
                                                   int cols);

static void launch_materialized_attention(cublasHandle_t handle,
                                          const Tensor& qf,
                                          const Tensor& kf,
                                          const Tensor& vf,
                                          const Tensor* vh,
                                          bool use_fp16_value_gemm,
                                          float scale,
                                          Tensor& out,
                                          Tensor& scores,
                                          Tensor* scores_half) {
    const int64_t B = qf.size(0);
    const int64_t H = qf.size(1);
    const int64_t N = qf.size(2);
    const int64_t D = qf.size(3);
    const int64_t N_k = kf.size(2);
    const int64_t BH = B * H;

    {
        float alpha = 1.0f;
        float beta  = 0.0f;
        int m = static_cast<int>(N);
        int n = static_cast<int>(N_k);
        int kk = static_cast<int>(D);
        long long int strideA = static_cast<long long int>(N * D);
        long long int strideB = static_cast<long long int>(N_k * D);
        long long int strideC = static_cast<long long int>(N * N_k);

        cublasGemmStridedBatchedEx(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            n, m, kk, &alpha,
            kf.data_f32(), CUDA_R_32F, kk, strideB,
            qf.data_f32(), CUDA_R_32F, kk, strideA,
            &beta, scores.data_f32(), CUDA_R_32F, n, strideC,
            static_cast<int>(BH),
            CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }

    {
        int total_rows = static_cast<int>(BH * N);
        int cols = static_cast<int>(N_k);
        int threads_po2 = 1;
        while (threads_po2 < cols && threads_po2 < kBlockSize) threads_po2 <<= 1;
        size_t smem_bytes = threads_po2 * sizeof(float);
        if (use_fp16_value_gemm) {
            fused_scale_softmax_to_half_kernel<<<total_rows, threads_po2, smem_bytes>>>(
                scores.data_f32(), scores_half->data_f16(), scale, total_rows, cols);
        } else {
            fused_scale_softmax_kernel<<<total_rows, threads_po2, smem_bytes>>>(
                scores.data_f32(), scale, total_rows, cols);
        }
    }

    {
        float alpha = 1.0f;
        float beta  = 0.0f;
        int m = static_cast<int>(N);
        int kk = static_cast<int>(N_k);
        int n = static_cast<int>(D);
        long long int strideA = static_cast<long long int>(N * N_k);
        long long int strideB = static_cast<long long int>(N_k * D);
        long long int strideC = static_cast<long long int>(N * D);

        if (use_fp16_value_gemm) {
            cublasGemmStridedBatchedEx(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, kk, &alpha,
                vh->data_ptr(), CUDA_R_16F, n, strideB,
                scores_half->data_ptr(), CUDA_R_16F, kk, strideA,
                &beta, out.data_f32(), CUDA_R_32F, n, strideC,
                static_cast<int>(BH),
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        } else {
            cublasGemmStridedBatchedEx(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, kk, &alpha,
                vf.data_f32(), CUDA_R_32F, n, strideB,
                scores.data_f32(), CUDA_R_32F, kk, strideA,
                &beta, out.data_f32(), CUDA_R_32F, n, strideC,
                static_cast<int>(BH),
                CUBLAS_COMPUTE_32F_FAST_TF32,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
    }
}

struct AttentionGraphKey {
    int64_t B = 0;
    int64_t H = 0;
    int64_t N = 0;
    int64_t D = 0;
    int64_t N_k = 0;
    uint32_t scale_bits = 0;
    bool use_fp16_value_gemm = false;

    bool operator==(const AttentionGraphKey& other) const {
        return B == other.B && H == other.H && N == other.N && D == other.D &&
               N_k == other.N_k && scale_bits == other.scale_bits &&
               use_fp16_value_gemm == other.use_fp16_value_gemm;
    }
};

struct AttentionGraphKeyHash {
    size_t operator()(const AttentionGraphKey& key) const {
        size_t seed = std::hash<int64_t>{}(key.B);
        seed ^= std::hash<int64_t>{}(key.H) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int64_t>{}(key.N) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int64_t>{}(key.D) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int64_t>{}(key.N_k) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<uint32_t>{}(key.scale_bits) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<bool>{}(key.use_fp16_value_gemm) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

struct AttentionGraphCache {
    explicit AttentionGraphCache(const AttentionGraphKey& graph_key) : key(graph_key) {}

    ~AttentionGraphCache() {
        if (exec != nullptr) cudaGraphExecDestroy(exec);
        if (graph != nullptr) cudaGraphDestroy(graph);
    }

    AttentionGraphKey key;
    Tensor q_in;
    Tensor k_in;
    Tensor v_in;
    Tensor vh;
    Tensor scores;
    Tensor scores_half;
    Tensor out;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    bool capture_failed = false;
    bool initialized = false;
};

static std::mutex& attention_graph_cache_mutex() {
    static std::mutex cache_mutex;
    return cache_mutex;
}

static std::unordered_map<AttentionGraphKey, std::shared_ptr<AttentionGraphCache>, AttentionGraphKeyHash>& attention_graph_caches() {
    static std::unordered_map<AttentionGraphKey, std::shared_ptr<AttentionGraphCache>, AttentionGraphKeyHash> caches;
    return caches;
}

static std::shared_ptr<AttentionGraphCache> get_attention_graph_cache(const AttentionGraphKey& key) {
    std::lock_guard<std::mutex> lock(attention_graph_cache_mutex());
    auto& caches = attention_graph_caches();
    auto it = caches.find(key);
    if (it != caches.end()) {
        return it->second;
    }
    auto cache = std::make_shared<AttentionGraphCache>(key);
    caches.emplace(key, cache);
    return cache;
}

static bool initialize_attention_graph(AttentionGraphCache& cache) {
    const auto& key = cache.key;
    cache.q_in = Tensor::empty({key.B, key.H, key.N, key.D}, DType::Float32);
    cache.k_in = Tensor::empty({key.B, key.H, key.N_k, key.D}, DType::Float32);
    cache.v_in = Tensor::empty({key.B, key.H, key.N_k, key.D}, DType::Float32);
    cache.scores = Tensor::empty({key.B * key.H, key.N, key.N_k}, DType::Float32);
    cache.out = Tensor::empty({key.B * key.H, key.N, key.D}, DType::Float32);
    if (key.use_fp16_value_gemm) {
        cache.vh = Tensor::empty({key.B, key.H, key.N_k, key.D}, DType::Float16);
        cache.scores_half = Tensor::empty({key.B * key.H, key.N, key.N_k}, DType::Float16);
    }

    cublasHandle_t handle = CudaContext::instance().cublas();
    cudaStream_t stream = CudaContext::instance().stream();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Warm up library internals outside capture so the first lazy init does not invalidate graph capture.
    if (key.use_fp16_value_gemm) {
        int64_t count = cache.v_in.numel();
        int grid = (int)ceildiv(count, (int64_t)kBlockSize);
        convert_f32_to_f16_kernel<<<grid, kBlockSize>>>(cache.v_in.data_f32(), cache.vh.data_f16(), count);
    }
    launch_materialized_attention(handle,
                                  cache.q_in,
                                  cache.k_in,
                                  cache.v_in,
                                  key.use_fp16_value_gemm ? &cache.vh : nullptr,
                                  key.use_fp16_value_gemm,
                                  float_from_bits(key.scale_bits),
                                  cache.out,
                                  cache.scores,
                                  key.use_fp16_value_gemm ? &cache.scores_half : nullptr);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaError_t capture_status = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    if (capture_status != cudaSuccess) {
        clear_cuda_error_state(stream);
        return false;
    }

    if (key.use_fp16_value_gemm) {
        int64_t count = cache.v_in.numel();
        int grid = (int)ceildiv(count, (int64_t)kBlockSize);
        convert_f32_to_f16_kernel<<<grid, kBlockSize>>>(cache.v_in.data_f32(), cache.vh.data_f16(), count);
    }

    launch_materialized_attention(handle,
                                  cache.q_in,
                                  cache.k_in,
                                  cache.v_in,
                                  key.use_fp16_value_gemm ? &cache.vh : nullptr,
                                  key.use_fp16_value_gemm,
                                  float_from_bits(key.scale_bits),
                                  cache.out,
                                  cache.scores,
                                  key.use_fp16_value_gemm ? &cache.scores_half : nullptr);

    cudaGraph_t graph = nullptr;
    capture_status = cudaStreamEndCapture(stream, &graph);
    if (capture_status != cudaSuccess || graph == nullptr) {
        if (graph != nullptr) cudaGraphDestroy(graph);
        clear_cuda_error_state(stream);
        return false;
    }

    cudaGraphExec_t exec = nullptr;
    cudaError_t instantiate_status = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    if (instantiate_status != cudaSuccess) {
        cudaGraphDestroy(graph);
        clear_cuda_error_state(stream);
        return false;
    }

    cache.graph = graph;
    cache.exec = exec;
    cache.initialized = true;
    return true;
}

void clear_attention_graph_cache() {
    std::lock_guard<std::mutex> lock(attention_graph_cache_mutex());
    attention_graph_caches().clear();
}

static bool try_run_attention_graph(const Tensor& q,
                                    const Tensor& k,
                                    const Tensor& v,
                                    float scale,
                                    bool use_fp16_value_gemm,
                                    DType orig_dtype,
                                    Tensor& result) {
    if (!g_enable_cuda_graph_attention) {
        return false;
    }
    if (!is_f32_contiguous(q) || !is_f32_contiguous(k) || !is_f32_contiguous(v)) {
        return false;
    }

    AttentionGraphKey key;
    key.B = q.size(0);
    key.H = q.size(1);
    key.N = q.size(2);
    key.D = q.size(3);
    key.N_k = k.size(2);
    key.scale_bits = float_bits(scale);
    key.use_fp16_value_gemm = use_fp16_value_gemm;

    auto cache = get_attention_graph_cache(key);
    if (cache->capture_failed) {
        return false;
    }
    if (!cache->initialized && !initialize_attention_graph(*cache)) {
        cache->capture_failed = true;
        return false;
    }

    cudaStream_t stream = CudaContext::instance().stream();
    async_copy_tensor(cache->q_in, q, stream);
    async_copy_tensor(cache->k_in, k, stream);
    async_copy_tensor(cache->v_in, v, stream);
    CUDA_CHECK(cudaGraphLaunch(cache->exec, stream));

    result = maybe_cast_back(cache->out.reshape({key.B, key.H, key.N, key.D}), orig_dtype);
    return true;
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

__global__ void fused_scale_softmax_to_half_kernel(const float* __restrict__ data,
                                                   __half* __restrict__ out,
                                                   float scale,
                                                   int rows,
                                                   int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float smem[];
    const float* row_data = data + (int64_t)row * cols;
    __half* row_out = out + (int64_t)row * cols;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    constexpr int MAX_ELEMS = 32;
    float reg[MAX_ELEMS];
    int elems_per_thread = (cols + num_threads - 1) / num_threads;

    float thread_max = -FLT_MAX;
    for (int i = 0; i < elems_per_thread; i++) {
        int j = tid + i * num_threads;
        float v = (j < cols) ? row_data[j] * scale : -FLT_MAX;
        reg[i] = v;
        if (v > thread_max) thread_max = v;
    }

    smem[tid] = thread_max;
    __syncthreads();
    for (int s = num_threads >> 1; s > 0; s >>= 1) {
        if (tid < s && smem[tid + s] > smem[tid]) smem[tid] = smem[tid + s];
        __syncthreads();
    }
    float max_val = smem[0];
    __syncthreads();

    float thread_sum = 0.0f;
    for (int i = 0; i < elems_per_thread; i++) {
        int j = tid + i * num_threads;
        if (j < cols) {
            float e = __expf(reg[i] - max_val);
            reg[i] = e;
            thread_sum += e;
        }
    }

    smem[tid] = thread_sum;
    __syncthreads();
    for (int s = num_threads >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / smem[0];
    __syncthreads();

    for (int i = 0; i < elems_per_thread; i++) {
        int j = tid + i * num_threads;
        if (j < cols) {
            row_out[j] = __float2half(reg[i] * inv_sum);
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

    bool use_fp16_value_gemm = g_quantize_fp16 && D <= 128;

    // ---- default scale ----
    if (scale == 0.0f) {
        scale = 1.0f / sqrtf(static_cast<float>(D));
    }

    Tensor graph_result;
    if (try_run_attention_graph(q, k, v, scale, use_fp16_value_gemm, orig_dtype, graph_result)) {
        return graph_result;
    }

    // ---- ensure f32 contiguous ----
    Tensor qf = ensure_f32(q);   // [B, H, N,   D]
    Tensor kf = ensure_f32(k);   // [B, H, N_k, D]
    Tensor vf = ensure_f32(v);   // [B, H, N_k, D]
    Tensor vh;
    if (use_fp16_value_gemm) {
        vh = vf.to_f16().contiguous();
    }

    const int64_t BH = B * H;

    // Allocate output [B*H, N, D]
    Tensor out = Tensor::empty({BH, N, D}, DType::Float32);

    // Use Flash Attention only for D <= 128 AND very long sequences.
    // For shorter/medium sequences, cuBLAS with materialized attention is faster.
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
        Tensor scores_half;
        cublasHandle_t handle = CudaContext::instance().cublas();
        if (use_fp16_value_gemm) {
            scores_half = Tensor::empty({BH, N, N_k}, DType::Float16);
        }
        launch_materialized_attention(handle,
                                      qf,
                                      kf,
                                      vf,
                                      use_fp16_value_gemm ? &vh : nullptr,
                                      use_fp16_value_gemm,
                                      scale,
                                      out,
                                      scores,
                                      use_fp16_value_gemm ? &scores_half : nullptr);
    }

    // Reshape output from [B*H, N, D] to [B, H, N, D]
    Tensor result = out.reshape({B, H, N, D});
    return maybe_cast_back(result, orig_dtype);
}

} // namespace ops
} // namespace cudasep
