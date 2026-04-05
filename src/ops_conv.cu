// ops_conv.cu - Convolution operations using cuDNN.
//
// Uses cuDNN for highly optimized convolution kernels with automatic
// algorithm selection and tensor core acceleration.

#include "ops.h"
#include <cudnn.h>
#include <cmath>
#include <stdexcept>
#include <string>

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

#define CUDNN_CHECK(expr)                                                       \
    do {                                                                         \
        cudnnStatus_t _status = (expr);                                          \
        if (_status != CUDNN_STATUS_SUCCESS) {                                   \
            throw std::runtime_error(std::string("cuDNN error: ") +              \
                                     cudnnGetErrorString(_status) +              \
                                     " at " + __FILE__ + ":" +                   \
                                     std::to_string(__LINE__));                  \
        }                                                                        \
    } while (0)

// RAII wrappers for cuDNN descriptors
struct TensorDesc {
    cudnnTensorDescriptor_t desc;
    TensorDesc() { CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc)); }
    ~TensorDesc() { cudnnDestroyTensorDescriptor(desc); }
    operator cudnnTensorDescriptor_t() const { return desc; }
};

struct FilterDesc {
    cudnnFilterDescriptor_t desc;
    FilterDesc() { CUDNN_CHECK(cudnnCreateFilterDescriptor(&desc)); }
    ~FilterDesc() { cudnnDestroyFilterDescriptor(desc); }
    operator cudnnFilterDescriptor_t() const { return desc; }
};

struct ConvDesc {
    cudnnConvolutionDescriptor_t desc;
    ConvDesc() { CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&desc)); }
    ~ConvDesc() { cudnnDestroyConvolutionDescriptor(desc); }
    operator cudnnConvolutionDescriptor_t() const { return desc; }
};

// ---------------------------------------------------------------------------
// Cached cuDNN workspace (pre-allocate 128MB)
// ---------------------------------------------------------------------------
static void* g_cudnn_workspace = nullptr;
static size_t g_cudnn_workspace_size = 0;

static void* get_cudnn_workspace(size_t required_size) {
    if (required_size <= g_cudnn_workspace_size && g_cudnn_workspace != nullptr) {
        return g_cudnn_workspace;
    }
    if (g_cudnn_workspace) {
        cudaFree(g_cudnn_workspace);
    }
    size_t alloc_size = std::max(required_size, (size_t)(128 * 1024 * 1024));
    cudaError_t err = cudaMalloc(&g_cudnn_workspace, alloc_size);
    if (err != cudaSuccess) {
        alloc_size = required_size;
        err = cudaMalloc(&g_cudnn_workspace, alloc_size);
        if (err != cudaSuccess) {
            g_cudnn_workspace = nullptr;
            g_cudnn_workspace_size = 0;
            throw std::runtime_error("Failed to allocate cuDNN workspace");
        }
    }
    g_cudnn_workspace_size = alloc_size;
    return g_cudnn_workspace;
}

// ---------------------------------------------------------------------------
// cuDNN Algorithm Cache — avoids repeated algorithm search overhead
// ---------------------------------------------------------------------------
#include <unordered_map>

struct ConvKey {
    int B, C_in, H, W, C_out, kH, kW;
    int stride_h, stride_w, pad_h, pad_w;
    int dilation_h, dilation_w, groups;
    bool is_transpose;

    bool operator==(const ConvKey& o) const {
        return B==o.B && C_in==o.C_in && H==o.H && W==o.W &&
               C_out==o.C_out && kH==o.kH && kW==o.kW &&
               stride_h==o.stride_h && stride_w==o.stride_w &&
               pad_h==o.pad_h && pad_w==o.pad_w &&
               dilation_h==o.dilation_h && dilation_w==o.dilation_w &&
               groups==o.groups && is_transpose==o.is_transpose;
    }
};

struct ConvKeyHash {
    size_t operator()(const ConvKey& k) const {
        size_t h = 0;
        auto combine = [&](int v) { h ^= std::hash<int>()(v) + 0x9e3779b9 + (h<<6) + (h>>2); };
        combine(k.B); combine(k.C_in); combine(k.H); combine(k.W);
        combine(k.C_out); combine(k.kH); combine(k.kW);
        combine(k.stride_h); combine(k.stride_w);
        combine(k.pad_h); combine(k.pad_w);
        combine(k.dilation_h); combine(k.dilation_w);
        combine(k.groups); combine(k.is_transpose ? 1 : 0);
        return h;
    }
};

struct CachedFwdAlgo {
    cudnnConvolutionFwdAlgo_t algo;
    size_t workspace_size;
};

struct CachedBwdDataAlgo {
    cudnnConvolutionBwdDataAlgo_t algo;
    size_t workspace_size;
};

static std::unordered_map<ConvKey, CachedFwdAlgo, ConvKeyHash> g_fwd_cache;
static std::unordered_map<ConvKey, CachedBwdDataAlgo, ConvKeyHash> g_bwd_cache;

static CachedFwdAlgo get_fwd_algo_cached(cudnnHandle_t handle,
    cudnnTensorDescriptor_t xDesc, cudnnFilterDescriptor_t wDesc,
    cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t yDesc,
    const ConvKey& key) {
    auto it = g_fwd_cache.find(key);
    if (it != g_fwd_cache.end()) return it->second;
    int cnt;
    cudnnConvolutionFwdAlgoPerf_t perf[8];
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(handle,
        xDesc, wDesc, convDesc, yDesc, 8, &cnt, perf));
    CachedFwdAlgo cached{perf[0].algo, 0};
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle,
        xDesc, wDesc, convDesc, yDesc, cached.algo, &cached.workspace_size));
    g_fwd_cache[key] = cached;
    return cached;
}

static CachedBwdDataAlgo get_bwd_algo_cached(cudnnHandle_t handle,
    cudnnFilterDescriptor_t wDesc, cudnnTensorDescriptor_t dyDesc,
    cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t dxDesc,
    const ConvKey& key) {
    auto it = g_bwd_cache.find(key);
    if (it != g_bwd_cache.end()) return it->second;
    int cnt;
    cudnnConvolutionBwdDataAlgoPerf_t perf[8];
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(handle,
        wDesc, dyDesc, convDesc, dxDesc, 8, &cnt, perf));
    CachedBwdDataAlgo cached{perf[0].algo, 0};
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(handle,
        wDesc, dyDesc, convDesc, dxDesc, cached.algo, &cached.workspace_size));
    g_bwd_cache[key] = cached;
    return cached;
}

// ---------------------------------------------------------------------------
// conv1d using cuDNN (via 2D with H=1)
// ---------------------------------------------------------------------------

namespace ops {

Tensor conv1d(const Tensor& x, const Tensor& weight, const Tensor& bias,
              int stride, int padding, int dilation, int groups) {
    DType orig = x.dtype();
    Tensor xf   = ensure_f32(x);
    Tensor wf   = ensure_f32(weight);
    Tensor bf;
    bool has_bias = (bias.numel() > 0);
    if (has_bias) bf = ensure_f32(bias);

    int B     = (int)xf.size(0);
    int C_in  = (int)xf.size(1);
    int L     = (int)xf.size(2);
    int C_out   = (int)wf.size(0);
    int C_in_g  = (int)wf.size(1);
    int K       = (int)wf.size(2);
    int L_out = (L + 2 * padding - dilation * (K - 1) - 1) / stride + 1;

    Tensor output = Tensor::empty({B, C_out, L_out}, DType::Float32);

    cudnnHandle_t handle = CudaContext::instance().cudnn();

    TensorDesc xDesc, yDesc;
    FilterDesc wDesc;
    ConvDesc convDesc;

    // Treat as 2D with H=1
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, B, C_in, 1, L));

    CUDNN_CHECK(cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW, C_out, C_in_g, 1, K));

    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(convDesc, 0, padding,
        1, stride, 1, dilation,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    if (groups > 1) {
        CUDNN_CHECK(cudnnSetConvolutionGroupCount(convDesc, groups));
    }
    CUDNN_CHECK(cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, B, C_out, 1, L_out));

    // Cached algorithm selection
    ConvKey key{B, C_in, 1, L, C_out, 1, K, 1, stride, 0, padding, 1, dilation, groups, false};
    auto cached = get_fwd_algo_cached(handle, xDesc, wDesc, convDesc, yDesc, key);
    void* workspace = (cached.workspace_size > 0) ? get_cudnn_workspace(cached.workspace_size) : nullptr;

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(handle,
        &alpha, xDesc, xf.data_f32(),
        wDesc, wf.data_f32(),
        convDesc, cached.algo, workspace, cached.workspace_size,
        &beta, yDesc, output.data_f32()));

    if (has_bias) {
        TensorDesc biasDesc;
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(biasDesc, CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT, 1, C_out, 1, 1));
        float ab = 1.0f, bb = 1.0f;
        CUDNN_CHECK(cudnnAddTensor(handle,
            &ab, biasDesc, bf.data_f32(),
            &bb, yDesc, output.data_f32()));
    }

    return maybe_cast_back(output, orig);
}

// ---------------------------------------------------------------------------
// conv2d using cuDNN
// ---------------------------------------------------------------------------

Tensor conv2d(const Tensor& x, const Tensor& weight, const Tensor& bias,
              int stride_h, int stride_w, int pad_h, int pad_w,
              int dilation_h, int dilation_w, int groups) {
    DType orig = x.dtype();
    Tensor xf   = ensure_f32(x);
    Tensor wf   = ensure_f32(weight);
    Tensor bf;
    bool has_bias = (bias.numel() > 0);
    if (has_bias) bf = ensure_f32(bias);

    int B     = (int)xf.size(0);
    int C_in  = (int)xf.size(1);
    int H     = (int)xf.size(2);
    int W     = (int)xf.size(3);
    int C_out   = (int)wf.size(0);
    int C_in_g  = (int)wf.size(1);
    int kH      = (int)wf.size(2);
    int kW      = (int)wf.size(3);
    int H_out = (H + 2 * pad_h - dilation_h * (kH - 1) - 1) / stride_h + 1;
    int W_out = (W + 2 * pad_w - dilation_w * (kW - 1) - 1) / stride_w + 1;

    Tensor output = Tensor::empty({B, C_out, H_out, W_out}, DType::Float32);

    cudnnHandle_t handle = CudaContext::instance().cudnn();

    TensorDesc xDesc, yDesc;
    FilterDesc wDesc;
    ConvDesc convDesc;

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, B, C_in, H, W));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW, C_out, C_in_g, kH, kW));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w,
        stride_h, stride_w, dilation_h, dilation_w,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    if (groups > 1) {
        CUDNN_CHECK(cudnnSetConvolutionGroupCount(convDesc, groups));
    }
    CUDNN_CHECK(cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, B, C_out, H_out, W_out));

    ConvKey key{B, C_in, H, W, C_out, kH, kW, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups, false};
    auto cached = get_fwd_algo_cached(handle, xDesc, wDesc, convDesc, yDesc, key);
    void* workspace = (cached.workspace_size > 0) ? get_cudnn_workspace(cached.workspace_size) : nullptr;

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(handle,
        &alpha, xDesc, xf.data_f32(),
        wDesc, wf.data_f32(),
        convDesc, cached.algo, workspace, cached.workspace_size,
        &beta, yDesc, output.data_f32()));

    if (has_bias) {
        TensorDesc biasDesc;
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(biasDesc, CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT, 1, C_out, 1, 1));
        float ab = 1.0f, bb = 1.0f;
        CUDNN_CHECK(cudnnAddTensor(handle,
            &ab, biasDesc, bf.data_f32(),
            &bb, yDesc, output.data_f32()));
    }

    return maybe_cast_back(output, orig);
}

// ---------------------------------------------------------------------------
// conv_transpose1d using cuDNN (via cudnnConvolutionBackwardData)
// ---------------------------------------------------------------------------

Tensor conv_transpose1d(const Tensor& x, const Tensor& weight, const Tensor& bias,
                        int stride, int padding, int output_padding) {
    DType orig = x.dtype();
    Tensor xf   = ensure_f32(x);
    Tensor wf   = ensure_f32(weight);
    Tensor bf;
    bool has_bias = (bias.numel() > 0);
    if (has_bias) bf = ensure_f32(bias);

    int B     = (int)xf.size(0);
    int C_in  = (int)xf.size(1);
    int L     = (int)xf.size(2);
    int C_out = (int)wf.size(1);
    int K     = (int)wf.size(2);
    int L_out = (L - 1) * stride - 2 * padding + K + output_padding;

    Tensor output = Tensor::empty({B, C_out, L_out}, DType::Float32);

    cudnnHandle_t handle = CudaContext::instance().cudnn();

    TensorDesc dyDesc, dxDesc;
    FilterDesc wDesc;
    ConvDesc convDesc;

    // dy (input to transposed conv): [B, C_in, 1, L]
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, B, C_in, 1, L));

    // Filter: [C_in, C_out, 1, K]
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW, C_in, C_out, 1, K));

    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(convDesc, 0, padding,
        1, stride, 1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    CUDNN_CHECK(cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH));

    // dx (output): [B, C_out, 1, L_out]
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, B, C_out, 1, L_out));

    ConvKey key{B, C_in, 1, L, C_out, 1, K, 1, stride, 0, padding, 1, 1, 1, true};
    auto cached = get_bwd_algo_cached(handle, wDesc, dyDesc, convDesc, dxDesc, key);
    void* workspace = (cached.workspace_size > 0) ? get_cudnn_workspace(cached.workspace_size) : nullptr;

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionBackwardData(handle,
        &alpha, wDesc, wf.data_f32(),
        dyDesc, xf.data_f32(),
        convDesc, cached.algo, workspace, cached.workspace_size,
        &beta, dxDesc, output.data_f32()));

    if (has_bias) {
        TensorDesc biasDesc, outDesc;
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(biasDesc, CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT, 1, C_out, 1, 1));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(outDesc, CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT, B, C_out, 1, L_out));
        float ab = 1.0f, bb = 1.0f;
        CUDNN_CHECK(cudnnAddTensor(handle,
            &ab, biasDesc, bf.data_f32(),
            &bb, outDesc, output.data_f32()));
    }

    return maybe_cast_back(output, orig);
}

// ---------------------------------------------------------------------------
// conv_transpose2d using cuDNN
// ---------------------------------------------------------------------------

Tensor conv_transpose2d(const Tensor& x, const Tensor& weight, const Tensor& bias,
                        int stride_h, int stride_w, int pad_h, int pad_w,
                        int output_pad_h, int output_pad_w) {
    DType orig = x.dtype();
    Tensor xf   = ensure_f32(x);
    Tensor wf   = ensure_f32(weight);
    Tensor bf;
    bool has_bias = (bias.numel() > 0);
    if (has_bias) bf = ensure_f32(bias);

    int B     = (int)xf.size(0);
    int C_in  = (int)xf.size(1);
    int H_in  = (int)xf.size(2);
    int W_in  = (int)xf.size(3);
    int C_out = (int)wf.size(1);
    int kH    = (int)wf.size(2);
    int kW    = (int)wf.size(3);
    int H_out = (H_in - 1) * stride_h - 2 * pad_h + kH + output_pad_h;
    int W_out = (W_in - 1) * stride_w - 2 * pad_w + kW + output_pad_w;

    Tensor output = Tensor::empty({B, C_out, H_out, W_out}, DType::Float32);

    cudnnHandle_t handle = CudaContext::instance().cudnn();

    TensorDesc dyDesc, dxDesc;
    FilterDesc wDesc;
    ConvDesc convDesc;

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, B, C_in, H_in, W_in));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW, C_in, C_out, kH, kW));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w,
        stride_h, stride_w, 1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    CUDNN_CHECK(cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, B, C_out, H_out, W_out));

    ConvKey key{B, C_in, H_in, W_in, C_out, kH, kW, stride_h, stride_w, pad_h, pad_w, 1, 1, 1, true};
    auto cached = get_bwd_algo_cached(handle, wDesc, dyDesc, convDesc, dxDesc, key);
    void* workspace = (cached.workspace_size > 0) ? get_cudnn_workspace(cached.workspace_size) : nullptr;

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionBackwardData(handle,
        &alpha, wDesc, wf.data_f32(),
        dyDesc, xf.data_f32(),
        convDesc, cached.algo, workspace, cached.workspace_size,
        &beta, dxDesc, output.data_f32()));

    if (has_bias) {
        TensorDesc biasDesc, outDesc;
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(biasDesc, CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT, 1, C_out, 1, 1));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(outDesc, CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT, B, C_out, H_out, W_out));
        float ab = 1.0f, bb = 1.0f;
        CUDNN_CHECK(cudnnAddTensor(handle,
            &ab, biasDesc, bf.data_f32(),
            &bb, outDesc, output.data_f32()));
    }

    return maybe_cast_back(output, orig);
}

} // namespace ops
} // namespace cudasep
