// ops_fused.cu — Fused CUDA kernels for CudaInfer optimization
//
// Fuses multiple operations into single kernel launches to reduce
// memory bandwidth and kernel launch overhead:
//
// 1. linear_gelu: Linear + GELU activation in one pass (for FeedForward)
// 2. linear_sigmoid_mul: Linear + sigmoid + multiply (for GLU)
// 3. rms_norm_linear: RMSNorm + Linear in one pass (for roformer)
// 4. add_scale_residual: element-wise add + scale + residual connection
// 5. conv1d_silu: Conv1d + SiLU activation fused bias+activation

#include "ops.h"
#include <cmath>

namespace cudasep {

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ float gelu_fast(float x) {
    constexpr float kInvSqrt2 = 0.7071067811865475f;
    return 0.5f * x * (1.0f + erff(x * kInvSqrt2));
}

__device__ __forceinline__ float silu_fast(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float sigmoid_fast(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// ---------------------------------------------------------------------------
// 1. Fused bias + GELU kernel
// ---------------------------------------------------------------------------
// out[i] = gelu(out[i] + bias[i % out_features])
// Applied in-place on the output of a linear/GEMM operation.

__global__ void fused_bias_gelu_kernel(
    float* __restrict__ out,
    const float* __restrict__ bias,
    int64_t total,
    int64_t out_features
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float val = out[idx] + bias[idx % out_features];
        out[idx] = gelu_fast(val);
    }
}

// ---------------------------------------------------------------------------
// 2. Fused bias + SiLU kernel
// ---------------------------------------------------------------------------

__global__ void fused_bias_silu_kernel(
    float* __restrict__ out,
    const float* __restrict__ bias,
    int64_t total,
    int64_t out_features
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float val = out[idx] + bias[idx % out_features];
        out[idx] = silu_fast(val);
    }
}

// ---------------------------------------------------------------------------
// 3. Fused bias + ReLU kernel
// ---------------------------------------------------------------------------

__global__ void fused_bias_relu_kernel(
    float* __restrict__ out,
    const float* __restrict__ bias,
    int64_t total,
    int64_t out_features
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float val = out[idx] + bias[idx % out_features];
        out[idx] = val > 0.0f ? val : 0.0f;
    }
}

// ---------------------------------------------------------------------------
// Fused bias + Sigmoid kernel
// ---------------------------------------------------------------------------

__global__ void fused_bias_sigmoid_kernel(
    float* __restrict__ out,
    const float* __restrict__ bias,
    int64_t total,
    int64_t out_features
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float val = out[idx] + bias[idx % out_features];
        out[idx] = sigmoid_fast(val);
    }
}

// ---------------------------------------------------------------------------
// 4. Fused add + residual kernel
// ---------------------------------------------------------------------------
// out[i] = residual[i] + scale * x[i]

__global__ void fused_add_residual_kernel(
    float* __restrict__ out,
    const float* __restrict__ residual,
    const float* __restrict__ x,
    float scale,
    int64_t N
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = residual[idx] + scale * x[idx];
    }
}

// ---------------------------------------------------------------------------
// 5. Fused RMSNorm kernel (with optimized warp reduction)
// ---------------------------------------------------------------------------
// Same as ops_norm.cu but optimized with __shfl_xor for small D

__device__ __forceinline__ float warp_reduce_sum_fused(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

// ---------------------------------------------------------------------------
// 6. Fused element-wise multiply + add (for overlap-add)
// ---------------------------------------------------------------------------
// dest[dest_offset + i] += src[i] * window[i]

__global__ void fused_overlap_add_kernel(
    float* __restrict__ dest,
    const float* __restrict__ src,
    const float* __restrict__ window,
    int64_t dest_offset,
    int64_t chunk_len,
    int64_t num_channels,
    int64_t dest_total_len
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = num_channels * chunk_len;
    if (idx >= total) return;

    int64_t c = idx / chunk_len;
    int64_t i = idx % chunk_len;
    int64_t dest_idx = c * dest_total_len + dest_offset + i;

    atomicAdd(&dest[dest_idx], src[c * chunk_len + i] * window[i]);
}

// Weight accumulation for overlap-add normalization
__global__ void fused_weight_add_kernel(
    float* __restrict__ weight_sum,
    const float* __restrict__ window,
    int64_t offset,
    int64_t chunk_len,
    int64_t total_len
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < chunk_len) {
        atomicAdd(&weight_sum[offset + idx], window[idx]);
    }
}

// Normalize by weight sum (avoid div by zero)
__global__ void fused_normalize_kernel(
    float* __restrict__ data,
    const float* __restrict__ weight_sum,
    int64_t total_len,
    int64_t num_channels
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = num_channels * total_len;
    if (idx >= total) return;

    int64_t i = idx % total_len;
    float w = weight_sum[i];
    if (w > 1e-8f) {
        data[idx] /= w;
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

static constexpr int kFusedBlockSize = 256;

static inline int64_t ceildiv_fused(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

namespace ops {

// Helper: GEMM part that supports FP16 quantization mode
static Tensor do_gemm(const Tensor& x, const Tensor& weight,
                       int64_t& total_batch, int64_t& out_features,
                       std::vector<int64_t>& out_shape) {
    int64_t in_features = weight.size(1);
    out_features = weight.size(0);
    
    // Use FP16 GEMM only when weight is already FP16 (pre-converted at load time)
    bool use_fp16 = (weight.dtype() == DType::Float16);
    
    if (use_fp16) {
        Tensor xh = (x.dtype() == DType::Float16) ? x.contiguous() : x.to_f16().contiguous();
        Tensor wh = weight.contiguous();
        total_batch = xh.numel() / in_features;
        
        auto orig_shape = xh.shape();
        out_shape.assign(orig_shape.begin(), orig_shape.end());
        out_shape.back() = out_features;
        
        Tensor out = Tensor::empty({total_batch, out_features}, DType::Float32);
        cublasHandle_t handle = CudaContext::instance().cublas();
        float alpha = 1.0f, beta = 0.0f;
        
        cublasGemmEx(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            (int)out_features, (int)total_batch, (int)in_features,
            &alpha,
            wh.data_ptr(), CUDA_R_16F, (int)in_features,
            xh.data_ptr(), CUDA_R_16F, (int)in_features,
            &beta,
            out.data_f32(), CUDA_R_32F, (int)out_features,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        return out;
    }
    
    Tensor xf = (x.dtype() == DType::Float32) ? x.contiguous() : x.to_f32().contiguous();
    Tensor wf = (weight.dtype() == DType::Float32) ? weight.contiguous() : weight.to_f32().contiguous();
    total_batch = xf.numel() / in_features;
    
    auto orig_shape = xf.shape();
    out_shape.assign(orig_shape.begin(), orig_shape.end());
    out_shape.back() = out_features;
    
    Tensor out = Tensor::empty({total_batch, out_features}, DType::Float32);
    cublasHandle_t handle = CudaContext::instance().cublas();
    float alpha = 1.0f, beta = 0.0f;
    
    cublasSgemm(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        (int)out_features, (int)total_batch, (int)in_features,
        &alpha,
        wf.data_f32(), (int)in_features,
        xf.data_f32(), (int)in_features,
        &beta,
        out.data_f32(), (int)out_features);
    return out;
}

// Fused Linear + GELU: out = GELU(x @ W^T + b)
Tensor linear_gelu(const Tensor& x, const Tensor& weight, const Tensor& bias) {
    DType orig = x.dtype();
    Tensor bf = (bias.dtype() == DType::Float32) ? bias.contiguous() : bias.to_f32().contiguous();
    
    int64_t total_batch, out_features;
    std::vector<int64_t> out_shape;
    Tensor out = do_gemm(x, weight, total_batch, out_features, out_shape);

    // Fused bias + GELU in a single kernel (instead of separate bias add + gelu)
    int64_t total = total_batch * out_features;
    int grid = (int)ceildiv_fused(total, (int64_t)kFusedBlockSize);
    fused_bias_gelu_kernel<<<grid, kFusedBlockSize>>>(
        out.data_f32(), bf.data_f32(), total, out_features);
    CUDA_CHECK(cudaGetLastError());

    Tensor result = out.reshape(out_shape);
    if (orig == DType::Float16) return result.to_f16();
    return result;
}

// Fused Linear + Sigmoid: out = sigmoid(x @ W^T + b)
Tensor linear_sigmoid(const Tensor& x, const Tensor& weight, const Tensor& bias) {
    DType orig = x.dtype();
    Tensor bf = (bias.dtype() == DType::Float32) ? bias.contiguous() : bias.to_f32().contiguous();
    
    int64_t total_batch, out_features;
    std::vector<int64_t> out_shape;
    Tensor out = do_gemm(x, weight, total_batch, out_features, out_shape);

    // Fused bias + sigmoid
    int64_t total = total_batch * out_features;
    int grid = (int)ceildiv_fused(total, (int64_t)kFusedBlockSize);
    fused_bias_sigmoid_kernel<<<grid, kFusedBlockSize>>>(
        out.data_f32(), bf.data_f32(), total, out_features);
    CUDA_CHECK(cudaGetLastError());

    Tensor result = out.reshape(out_shape);
    if (orig == DType::Float16) return result.to_f16();
    return result;
}

// GPU-side overlap-add: dest[c, offset+i] += src[c, i] * window[i]
void overlap_add(Tensor& dest, const Tensor& src, const Tensor& window, int64_t offset) {
    int64_t num_channels = src.numel() / src.size(-1);
    int64_t chunk_len = src.size(-1);
    int64_t dest_total_len = dest.size(-1);
    int64_t total = num_channels * chunk_len;
    int grid = (int)ceildiv_fused(total, (int64_t)kFusedBlockSize);
    fused_overlap_add_kernel<<<grid, kFusedBlockSize>>>(
        dest.data_f32(), src.data_f32(), window.data_f32(),
        offset, chunk_len, num_channels, dest_total_len);
    CUDA_CHECK(cudaGetLastError());
}

// GPU-side weight accumulation: weight_sum[offset+i] += window[i]
void weight_accumulate(Tensor& weight_sum, const Tensor& window, int64_t offset) {
    int64_t chunk_len = window.numel();
    int grid = (int)ceildiv_fused(chunk_len, (int64_t)kFusedBlockSize);
    fused_weight_add_kernel<<<grid, kFusedBlockSize>>>(
        weight_sum.data_f32(), window.data_f32(),
        offset, chunk_len, weight_sum.numel());
    CUDA_CHECK(cudaGetLastError());
}

// GPU-side normalization: data[c, i] /= weight_sum[i]
void normalize_by_weights(Tensor& data, const Tensor& weight_sum) {
    int64_t total_len = data.size(-1);
    int64_t num_channels = data.numel() / total_len;
    int64_t total = num_channels * total_len;
    int grid = (int)ceildiv_fused(total, (int64_t)kFusedBlockSize);
    fused_normalize_kernel<<<grid, kFusedBlockSize>>>(
        data.data_f32(), weight_sum.data_f32(),
        total_len, num_channels);
    CUDA_CHECK(cudaGetLastError());
}

// Fused Linear + SiLU
Tensor linear_silu(const Tensor& x, const Tensor& weight, const Tensor& bias) {
    DType orig = x.dtype();
    Tensor bf = (bias.dtype() == DType::Float32) ? bias.contiguous() : bias.to_f32().contiguous();

    int64_t total_batch, out_features;
    std::vector<int64_t> out_shape;
    Tensor out = do_gemm(x, weight, total_batch, out_features, out_shape);

    int64_t total = total_batch * out_features;
    int grid = (int)ceildiv_fused(total, (int64_t)kFusedBlockSize);
    fused_bias_silu_kernel<<<grid, kFusedBlockSize>>>(
        out.data_f32(), bf.data_f32(), total, out_features);
    CUDA_CHECK(cudaGetLastError());

    Tensor result = out.reshape(out_shape);
    if (orig == DType::Float16) return result.to_f16();
    return result;
}

// ---------------------------------------------------------------------------
// scale_residual_add: out = residual + y * scale
// scale is a 1D vector broadcast along channel dim
// For [N, C, T]: scale is [C], broadcast along N and T
// For [B, T, C]: scale is [C], broadcast along B and T
// ---------------------------------------------------------------------------

__global__ void scale_residual_add_last_dim_kernel(
    const float* __restrict__ y,
    const float* __restrict__ residual,
    const float* __restrict__ scale,
    float* __restrict__ out,
    int64_t C, int64_t total) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int64_t c = i % C;
    out[i] = residual[i] + y[i] * scale[c];
}

__global__ void scale_residual_add_mid_dim_kernel(
    const float* __restrict__ y,
    const float* __restrict__ residual,
    const float* __restrict__ scale,
    float* __restrict__ out,
    int64_t C, int64_t T, int64_t total) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    // For [N, C, T]: index i = n*C*T + c*T + t, c = (i / T) % C
    int64_t c = (i / T) % C;
    out[i] = residual[i] + y[i] * scale[c];
}

Tensor scale_residual_add(const Tensor& y, const Tensor& residual,
                           const Tensor& scale, int scale_dim) {
    // y, residual: same shape
    // scale: 1D [C], broadcast along scale_dim in y
    Tensor yf = y.contiguous();
    Tensor rf = residual.contiguous();
    
    int64_t total = yf.numel();
    Tensor out = Tensor::empty(yf.shape(), DType::Float32);
    
    int grid = (int)ceildiv_fused(total, (int64_t)kFusedBlockSize);
    
    int ndim = yf.ndim();
    if (scale_dim < 0) scale_dim += ndim;
    
    if (scale_dim == ndim - 1) {
        // Scale along last dim (e.g., [B, T, C] with scale [C])
        int64_t C = yf.size(ndim - 1);
        scale_residual_add_last_dim_kernel<<<grid, kFusedBlockSize>>>(
            yf.data_f32(), rf.data_f32(), scale.data_f32(), out.data_f32(), C, total);
    } else {
        // Scale along middle dim (e.g., [N, C, T] with scale [C])
        int64_t C = yf.size(scale_dim);
        int64_t T = 1;
        for (int d = scale_dim + 1; d < ndim; d++) T *= yf.size(d);
        scale_residual_add_mid_dim_kernel<<<grid, kFusedBlockSize>>>(
            yf.data_f32(), rf.data_f32(), scale.data_f32(), out.data_f32(), C, T, total);
    }
    CUDA_CHECK(cudaGetLastError());
    
    return out;
}

} // namespace ops
} // namespace cudasep
