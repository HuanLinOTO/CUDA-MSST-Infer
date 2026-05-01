// ops_linear.cu - Linear algebra operations and CudaContext singleton
// for cudasep inference engine.
//
// Implements: CudaContext, linear, linear_no_bias, matmul, batched_matmul.
// All ops work on Float32. For Float16 inputs we cast to f32, compute, cast back.

#include "ops.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>

namespace cudasep {

// Global FP16 quantization flag
bool g_quantize_fp16 = false;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static constexpr int kBlockSize = 256;

static inline int64_t ceildiv(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

// Ensure tensor is f32-contiguous; returns possibly-converted tensor.
static Tensor ensure_f32(const Tensor& x) {
    if (x.dtype() == DType::Float32) return x.contiguous();
    return x.to_f32().contiguous();
}

// If the original tensor was f16, convert back.
static Tensor maybe_cast_back(const Tensor& result, DType orig) {
    if (orig == DType::Float16) return result.to_f16();
    return result;
}

static cublasStatus_t gemm_f32_fast(cublasHandle_t handle,
                                    cublasOperation_t transa,
                                    cublasOperation_t transb,
                                    int m, int n, int k,
                                    const float* alpha,
                                    const float* A, int lda,
                                    const float* B, int ldb,
                                    const float* beta,
                                    float* C, int ldc) {
    return cublasGemmEx(handle,
        transa, transb,
        m, n, k,
        alpha,
        A, CUDA_R_32F, lda,
        B, CUDA_R_32F, ldb,
        beta,
        C, CUDA_R_32F, ldc,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

static cublasStatus_t gemm_strided_batched_f32_fast(cublasHandle_t handle,
                                                     cublasOperation_t transa,
                                                     cublasOperation_t transb,
                                                     int m, int n, int k,
                                                     const float* alpha,
                                                     const float* A, int lda, long long int strideA,
                                                     const float* B, int ldb, long long int strideB,
                                                     const float* beta,
                                                     float* C, int ldc, long long int strideC,
                                                     int batch_count) {
    return cublasGemmStridedBatchedEx(handle,
        transa, transb,
        m, n, k,
        alpha,
        A, CUDA_R_32F, lda, strideA,
        B, CUDA_R_32F, ldb, strideB,
        beta,
        C, CUDA_R_32F, ldc, strideC,
        batch_count,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// ---------------------------------------------------------------------------
// CudaContext (singleton)
// ---------------------------------------------------------------------------

CudaContext::CudaContext() {
    cublasStatus_t stat = cublasCreate(&cublas_);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasCreate failed with status " +
                                 std::to_string((int)stat));
    }
    // Create cuDNN handle
    cudnnStatus_t dnn_stat = cudnnCreate(&cudnn_);
    if (dnn_stat != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error("cudnnCreate failed with status " +
                                 std::to_string((int)dnn_stat));
    }
    // Create a non-default stream for compute-memory overlap
    CUDA_CHECK(cudaStreamCreate(&stream_));
    // Set cuBLAS and cuDNN to use this stream
    cublasSetStream(cublas_, stream_);
    cudnnSetStream(cudnn_, stream_);
    // Enable tensor cores when available (TF32 on Ampere+)
    cublasSetMathMode(cublas_, CUBLAS_TF32_TENSOR_OP_MATH);
}

CudaContext::~CudaContext() {
    cublasDestroy(cublas_);
    cudnnDestroy(cudnn_);
    cudaStreamDestroy(stream_);
}

CudaContext& CudaContext::instance() {
    static CudaContext ctx;
    return ctx;
}

void CudaContext::sync() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// Bias addition kernel
// ---------------------------------------------------------------------------

__global__ void add_bias_kernel(float* __restrict__ out,
                                const float* __restrict__ bias,
                                int64_t rows, int64_t cols) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        out[idx] += bias[idx % cols];
    }
}

// ---------------------------------------------------------------------------
// linear(x, weight, bias)
//   x:      [..., in_features]
//   weight: [out_features, in_features]
//   bias:   [out_features]
//   output: [..., out_features]
//
// Equivalent to nn.Linear: out = x @ W^T + b
// ---------------------------------------------------------------------------
bool g_enable_cuda_graph_attention = true;

namespace ops {

Tensor linear(const Tensor& x, const Tensor& weight, const Tensor& bias) {
    DType orig = x.dtype();
    
    // Use FP16 GEMM only when weight is already FP16 (pre-converted at load time)
    if (weight.dtype() == DType::Float16) {
        Tensor xh = (x.dtype() == DType::Float16) ? x.contiguous() : x.to_f16().contiguous();
        Tensor wh = weight.contiguous();
        Tensor bf = ensure_f32(bias);
        
        int64_t out_features = wh.size(0);
        int64_t in_features = wh.size(1);
        int64_t total_batch = xh.numel() / in_features;
        
        auto orig_shape = xh.shape();
        std::vector<int64_t> out_shape(orig_shape.begin(), orig_shape.end());
        out_shape.back() = out_features;
        
        Tensor out = Tensor::empty({total_batch, out_features}, DType::Float32);
        
        cublasHandle_t handle = CudaContext::instance().cublas();
        float alpha = 1.0f, beta = 0.0f;
        
        int M = (int)total_batch;
        int K = (int)in_features;
        int N = (int)out_features;
        
        cublasGemmEx(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            wh.data_ptr(), CUDA_R_16F, K,
            xh.data_ptr(), CUDA_R_16F, K,
            &beta,
            out.data_f32(), CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        
        int64_t total = total_batch * out_features;
        int blocks = (int)ceildiv(total, (int64_t)kBlockSize);
        add_bias_kernel<<<blocks, kBlockSize>>>(out.data_f32(), bf.data_f32(), total_batch, out_features);
        CUDA_CHECK(cudaGetLastError());
        
        return out.reshape(out_shape);
    }
    
    Tensor xf = ensure_f32(x);
    Tensor wf = ensure_f32(weight);
    Tensor bf = ensure_f32(bias);

    // weight: [out_features, in_features]
    assert(wf.ndim() == 2);
    int64_t out_features = wf.size(0);
    int64_t in_features  = wf.size(1);
    assert(xf.size(xf.ndim() - 1) == in_features);
    assert(bf.ndim() == 1 && bf.size(0) == out_features);

    // Flatten x to [total_batch, in_features]
    int64_t total_batch = xf.numel() / in_features;

    // Save original shape for output reshape
    auto orig_shape = xf.shape();
    std::vector<int64_t> out_shape(orig_shape.begin(), orig_shape.end());
    out_shape.back() = out_features;

    // Allocate output [total_batch, out_features]
    Tensor out = Tensor::empty({total_batch, out_features}, DType::Float32);

    cublasHandle_t handle = CudaContext::instance().cublas();
    float alpha = 1.0f;
    float beta  = 0.0f;

    // Row-major: C = A @ B^T where A=[total_batch, in_features], B=[out_features, in_features]
    // We want out[total_batch, out_features] = xf[total_batch, in_features] @ wf^T[in_features, out_features]
    //
    // cuBLAS column-major trick:
    //   cublasSgemm(handle, transB, transA, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc)
    //   where M=total_batch, K=in_features, N=out_features
    //
    // C(col-major) = B^T @ A^T  =>  C(row-major) = A @ B
    // But we want A @ W^T, so B = W and we need B^T = W^T:
    //   transB = CUBLAS_OP_T (to get W^T)? No, let's think again.
    //
    // For row-major C = A @ W^T:
    //   Treat as col-major: C^T = W @ A^T
    //   cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, alpha, W, N, ... )
    //   Wait, let me be more systematic.
    //
    // We have row-major matrices:
    //   A: [M, K] = [total_batch, in_features]
    //   W: [N, K] = [out_features, in_features]
    //   C: [M, N] = [total_batch, out_features]
    //   C = A @ W^T
    //
    // In column-major view:
    //   A_col = A^T: [K, M], stored contiguously (since row-major A is col-major A^T)
    //   W_col = W^T: [K, N], stored contiguously
    //   C_col = C^T: [N, M]
    //
    //   C^T = (A @ W^T)^T = W @ A^T
    //   => C_col[N,M] = W_col_as_no_trans[N,K] @ A_col_as_no_trans[K,M]
    //   Wait: W row-major [N,K] is col-major W^T [K,N].
    //   So as col-major with no transpose, W is [K,N].
    //   To get W[N,K] in col-major we need to transpose: W_col with CUBLAS_OP_T.
    //
    // Let me just use the standard formula:
    //   For row-major C[M,N] = A[M,K] @ B[K,N]:
    //     cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
    //                 B_ptr, N, A_ptr, K, &beta, C_ptr, N)
    //
    //   Here we want C[M,N] = A[M,K] @ W^T[K,N], where W is [N,K].
    //   So B = W^T which is [K,N]. B_ptr = W.data_f32() but B is logically W^T.
    //   
    //   Actually for row-major, the pointer layout is:
    //   A[M,K] stored row-major = A^T[K,M] col-major, leading dim = K
    //   B[K,N] = W^T stored... but W is [N,K] row-major.
    //   W[N,K] row-major = W^T[K,N] col-major, leading dim = K.
    //   So col-major W^T has shape [K,N] and leading dim K.
    //   But we need B = W^T as [K,N] row-major = (W^T)^T[N,K] col-major = W[N,K] col-major with leading dim N.
    //   Hmm, this is getting confusing.
    //
    // Simplest correct approach for row-major C = A @ W^T:
    //   C^T = W @ A^T
    //   In col-major: need to compute C_cm[N,M] = W_cm[?,?] @ A_cm[?,?]
    //   A row-major [M,K] => col-major layout is A^T [K,M], ld=K
    //   W row-major [N,K] => col-major layout is W^T [K,N], ld=K
    //   C row-major [M,N] => col-major layout is C^T [N,M], ld=N
    //
    //   We want C^T[N,M] = W[N,K] @ A^T... wait no:
    //   C = A @ W^T
    //   C^T = W @ A^T  (take transpose of both sides)
    //
    //   col-major C^T is [N,M] with ld=N.
    //   W @ A^T: W is [N,K], A^T is [K,M], result is [N,M]. Good.
    //   
    //   In col-major land, W's memory (W row-major [N,K]) looks like W^T[K,N] with ld=K.
    //   So to get W[N,K] in col-major, we TRANSPOSE W_cm: cublas sees W_cm as [K,N] with ld=K,
    //   transposing gives [N,K].
    //   
    //   A^T: A's memory (row-major [M,K]) looks like [K,M] col-major with ld=K.
    //   A^T[K,M] with no transpose is what we need — that's [K,M]. Good.
    //
    //   cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
    //               W_ptr, K, A_ptr, K, &beta, C_ptr, N)
    //   
    //   transA=T on first matrix (W): takes col-major [K,N] ld=K and transposes -> [N,K]
    //   transB=N on second matrix (A): col-major [K,M] ld=K, no transpose -> [K,M]  
    //   Result: [N,K] @ [K,M] = [N,M] stored col-major ld=N = C^T row-major = C row-major [M,N]. Correct!
    //
    // BUT there's an even simpler way. Since cublasSgemm is:
    //   C_cm = op(A_cm) @ op(B_cm)
    //   where A_cm is the first pointer arg, B_cm is the second.
    //
    // For row-major C[M,N] = X[M,K] @ W^T[K,N]:
    //   Trick: swap order, i.e. compute C_col = W_col @ X_col
    //   W row-major [N,K] = col-major [K,N] ld=K. We want W[N,K], so OP_T on this gives [N,K].
    //   X row-major [M,K] = col-major [K,M] ld=K. We want X^T=[K,M], which is already [K,M] with OP_N.
    //   Hmm that's the same thing.
    //
    // Let me just use the well-known trick:

    int M = (int)total_batch;
    int K = (int)in_features;
    int N = (int)out_features;

    // cublasSgemm: C_cm = op(A_cm) * op(B_cm)
    // For row-major C[M,N] = X[M,K] * W^T[K,N]:
    //   We use: cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
    //                       &alpha, W, K, X, K, &beta, C, N)
    cublasStatus_t stat = gemm_f32_fast(handle,
        CUBLAS_OP_T,   // transpose W (col-major [K,N] -> [N,K])
        CUBLAS_OP_N,   // no transpose X (col-major [K,M] = X^T, which is what we want)
        N, M, K,
        &alpha,
        wf.data_f32(), K,   // W: row-major [N,K] = col-major [K,N], ld=K
        xf.data_f32(), K,   // X: row-major [M,K] = col-major [K,M], ld=K
        &beta,
        out.data_f32(), N); // C: col-major [N,M] = row-major [M,N], ld=N

    if (stat != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasSgemm failed in linear(), status=" +
                                 std::to_string((int)stat));
    }

    // Add bias
    int64_t total = total_batch * out_features;
    int blocks = (int)ceildiv(total, (int64_t)kBlockSize);
    add_bias_kernel<<<blocks, kBlockSize>>>(out.data_f32(), bf.data_f32(),
                                            total_batch, out_features);
    CUDA_CHECK(cudaGetLastError());

    // Reshape to [..., out_features]
    Tensor result = out.reshape(out_shape);
    return maybe_cast_back(result, orig);
}

// ---------------------------------------------------------------------------
// linear_no_bias(x, weight)
// ---------------------------------------------------------------------------

Tensor linear_no_bias(const Tensor& x, const Tensor& weight) {
    DType orig = x.dtype();
    
    if (weight.dtype() == DType::Float16) {
        Tensor xh = (x.dtype() == DType::Float16) ? x.contiguous() : x.to_f16().contiguous();
        Tensor wh = weight.contiguous();
        
        int64_t out_features = wh.size(0);
        int64_t in_features = wh.size(1);
        int64_t total_batch = xh.numel() / in_features;
        
        auto orig_shape = xh.shape();
        std::vector<int64_t> out_shape(orig_shape.begin(), orig_shape.end());
        out_shape.back() = out_features;
        
        Tensor out = Tensor::empty({total_batch, out_features}, DType::Float32);
        
        cublasHandle_t handle = CudaContext::instance().cublas();
        float alpha = 1.0f, beta = 0.0f;
        
        int M = (int)total_batch;
        int K = (int)in_features;
        int N = (int)out_features;
        
        cublasGemmEx(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            wh.data_ptr(), CUDA_R_16F, K,
            xh.data_ptr(), CUDA_R_16F, K,
            &beta,
            out.data_f32(), CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        
        return out.reshape(out_shape);
    }
    
    Tensor xf = ensure_f32(x);
    Tensor wf = ensure_f32(weight);

    assert(wf.ndim() == 2);
    int64_t out_features = wf.size(0);
    int64_t in_features  = wf.size(1);
    assert(xf.size(xf.ndim() - 1) == in_features);

    int64_t total_batch = xf.numel() / in_features;

    auto orig_shape = xf.shape();
    std::vector<int64_t> out_shape(orig_shape.begin(), orig_shape.end());
    out_shape.back() = out_features;

    Tensor out = Tensor::empty({total_batch, out_features}, DType::Float32);

    cublasHandle_t handle = CudaContext::instance().cublas();
    float alpha = 1.0f;
    float beta  = 0.0f;

    int M = (int)total_batch;
    int K = (int)in_features;
    int N = (int)out_features;

    cublasStatus_t stat = gemm_f32_fast(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        wf.data_f32(), K,
        xf.data_f32(), K,
        &beta,
        out.data_f32(), N);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasSgemm failed in linear_no_bias(), status=" +
                                 std::to_string((int)stat));
    }

    Tensor result = out.reshape(out_shape);
    return maybe_cast_back(result, orig);
}

// ---------------------------------------------------------------------------
// matmul(a, b)
//   Supports:
//     [M,K] @ [K,N] -> [M,N]               (2D x 2D)
//     [K]   @ [K,N] -> [N]                  (1D x 2D, vector-matrix)
//     [M,K] @ [K]   -> [M]                  (2D x 1D, matrix-vector)
//     [...,M,K] @ [...,K,N] -> [...,M,N]    (batched with broadcasting)
// ---------------------------------------------------------------------------

Tensor matmul(const Tensor& a, const Tensor& b) {
    DType orig = a.dtype();
    Tensor af = ensure_f32(a);
    Tensor bf = ensure_f32(b);

    int a_ndim = af.ndim();
    int b_ndim = bf.ndim();

    // --- 1D @ 2D: vector-matrix ---
    if (a_ndim == 1 && b_ndim == 2) {
        // a: [K], b: [K, N] -> result: [N]
        // Treat a as [1, K], multiply, squeeze
        Tensor a2 = af.reshape({1, af.size(0)});
        Tensor r = matmul(a2, bf);
        return maybe_cast_back(r.reshape({r.size(r.ndim() - 1)}), orig);
    }

    // --- 2D @ 1D: matrix-vector ---
    if (a_ndim == 2 && b_ndim == 1) {
        // a: [M, K], b: [K] -> result: [M]
        Tensor b2 = bf.reshape({bf.size(0), 1});
        Tensor r = matmul(af, b2);
        return maybe_cast_back(r.reshape({r.size(0)}), orig);
    }

    // --- 2D @ 2D ---
    if (a_ndim == 2 && b_ndim == 2) {
        int64_t M = af.size(0);
        int64_t K = af.size(1);
        int64_t N = bf.size(1);
        assert(bf.size(0) == K);

        Tensor out = Tensor::empty({M, N}, DType::Float32);

        cublasHandle_t handle = CudaContext::instance().cublas();
        float alpha = 1.0f;
        float beta  = 0.0f;

        // Row-major C[M,N] = A[M,K] @ B[K,N]
        // cublas col-major: swap args
        //   cublasSgemm(handle, N_op, N_op, N, M, K, alpha, B, N, A, K, beta, C, N)
        cublasStatus_t stat = gemm_f32_fast(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            (int)N, (int)M, (int)K,
            &alpha,
            bf.data_f32(), (int)N,  // B col-major [N,K] = B^T, ld=N
            af.data_f32(), (int)K,  // A col-major [K,M] = A^T, ld=K
            &beta,
            out.data_f32(), (int)N);

        if (stat != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasSgemm failed in matmul(2D), status=" +
                                     std::to_string((int)stat));
        }

        return maybe_cast_back(out, orig);
    }

    // --- Batched matmul with broadcasting ---
    // Both have >= 2 dims. Last two are the matrix dims.
    assert(a_ndim >= 2 && b_ndim >= 2);

    int64_t M = af.size(a_ndim - 2);
    int64_t K = af.size(a_ndim - 1);
    int64_t N = bf.size(b_ndim - 1);
    assert(bf.size(b_ndim - 2) == K);

    // Compute broadcast batch shape
    std::vector<int64_t> a_batch(af.shape().begin(), af.shape().end() - 2);
    std::vector<int64_t> b_batch(bf.shape().begin(), bf.shape().end() - 2);

    // Pad shorter batch dims with leading 1s
    int max_batch_ndim = std::max((int)a_batch.size(), (int)b_batch.size());
    while ((int)a_batch.size() < max_batch_ndim) a_batch.insert(a_batch.begin(), 1);
    while ((int)b_batch.size() < max_batch_ndim) b_batch.insert(b_batch.begin(), 1);

    std::vector<int64_t> out_batch(max_batch_ndim);
    for (int i = 0; i < max_batch_ndim; i++) {
        if (a_batch[i] == b_batch[i]) {
            out_batch[i] = a_batch[i];
        } else if (a_batch[i] == 1) {
            out_batch[i] = b_batch[i];
        } else if (b_batch[i] == 1) {
            out_batch[i] = a_batch[i];
        } else {
            throw std::runtime_error("matmul: incompatible batch dims");
        }
    }

    int64_t total_batch = 1;
    for (auto d : out_batch) total_batch *= d;

    int64_t a_total_batch = 1;
    for (auto d : a_batch) a_total_batch *= d;
    int64_t b_total_batch = 1;
    for (auto d : b_batch) b_total_batch *= d;

    // Expand and reshape both to [total_batch, ?, ?]
    std::vector<int64_t> a_expanded_shape = out_batch;
    a_expanded_shape.push_back(M);
    a_expanded_shape.push_back(K);
    std::vector<int64_t> b_expanded_shape = out_batch;
    b_expanded_shape.push_back(K);
    b_expanded_shape.push_back(N);

    // Reshape inputs to have proper batch dims (with leading 1s)
    std::vector<int64_t> a_reshape = a_batch;
    a_reshape.push_back(M);
    a_reshape.push_back(K);
    std::vector<int64_t> b_reshape = b_batch;
    b_reshape.push_back(K);
    b_reshape.push_back(N);

    Tensor ae = af.reshape(a_reshape).expand(a_expanded_shape).contiguous()
                   .reshape({total_batch, M, K});
    Tensor be = bf.reshape(b_reshape).expand(b_expanded_shape).contiguous()
                   .reshape({total_batch, K, N});

    // Use batched_matmul for the flat [total_batch, M, K] @ [total_batch, K, N]
    Tensor out_flat = batched_matmul(ae, be);

    // Reshape to [..., M, N]
    std::vector<int64_t> out_shape = out_batch;
    out_shape.push_back(M);
    out_shape.push_back(N);
    Tensor result = out_flat.reshape(out_shape);
    return maybe_cast_back(result, orig);
}

// ---------------------------------------------------------------------------
// batched_matmul(a, b)
//   a: [B, M, K], b: [B, K, N] -> [B, M, N]
//   Uses cublasSgemmStridedBatched
// ---------------------------------------------------------------------------

Tensor batched_matmul(const Tensor& a, const Tensor& b) {
    DType orig = a.dtype();
    Tensor af = ensure_f32(a);
    Tensor bf = ensure_f32(b);

    assert(af.ndim() == 3 && bf.ndim() == 3);
    int64_t B = af.size(0);
    int64_t M = af.size(1);
    int64_t K = af.size(2);
    assert(bf.size(0) == B);
    assert(bf.size(1) == K);
    int64_t N = bf.size(2);

    Tensor out = Tensor::empty({B, M, N}, DType::Float32);

    cublasHandle_t handle = CudaContext::instance().cublas();
    float alpha = 1.0f;
    float beta  = 0.0f;

    // Row-major C[M,N] = A[M,K] @ B[K,N] per batch
    // cuBLAS col-major trick: swap A and B
    //   cublasSgemmStridedBatched(handle, N_op, N_op, N, M, K,
    //       alpha, B, N, strideB, A, K, strideA, beta, C, N, strideC, B_count)
    long long int strideA = (long long int)(M * K);
    long long int strideB = (long long int)(K * N);
    long long int strideC = (long long int)(M * N);

    cublasStatus_t stat = gemm_strided_batched_f32_fast(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        (int)N, (int)M, (int)K,
        &alpha,
        bf.data_f32(), (int)N, strideB,   // "A" in cuBLAS = B in row-major
        af.data_f32(), (int)K, strideA,   // "B" in cuBLAS = A in row-major
        &beta,
        out.data_f32(), (int)N, strideC,
        (int)B);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasSgemmStridedBatched failed in batched_matmul(), status=" +
                                 std::to_string((int)stat));
    }

    return maybe_cast_back(out, orig);
}

} // namespace ops
} // namespace cudasep
