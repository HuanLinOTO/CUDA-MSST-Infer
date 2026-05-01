#pragma once
#include "tensor.h"
#include <cublas_v2.h>
#include <cufft.h>
#include <cudnn.h>

namespace cudasep {

// Global FP16 quantization flag for GEMM operations
extern bool g_quantize_fp16;
extern bool g_enable_cuda_graph_attention;
void clear_attention_graph_cache();

// Global context (cuBLAS/cuFFT/cuDNN handles + CUDA stream)
class CudaContext {
public:
    static CudaContext& instance();
    cublasHandle_t cublas() { return cublas_; }
    cudnnHandle_t cudnn() { return cudnn_; }
    cudaStream_t stream() { return stream_; }
    void sync(); // cudaDeviceSynchronize
private:
    CudaContext();
    ~CudaContext();
    cublasHandle_t cublas_;
    cudnnHandle_t cudnn_;
    cudaStream_t stream_;
};

namespace ops {

// === Element-wise activations ===
Tensor gelu(const Tensor& x);
Tensor relu(const Tensor& x);
Tensor sigmoid(const Tensor& x);
Tensor tanh_act(const Tensor& x);  // "tanh" conflicts with std::tanh
Tensor silu(const Tensor& x);      // SiLU / swish
Tensor softmax(const Tensor& x, int dim = -1);

// === Element-wise math ===
Tensor exp(const Tensor& x);
Tensor log(const Tensor& x);
Tensor sqrt(const Tensor& x);
Tensor abs(const Tensor& x);
Tensor cos(const Tensor& x);
Tensor sin(const Tensor& x);
Tensor rsqrt(const Tensor& x);
Tensor pow(const Tensor& x, float exponent);

// === Complex ops (last dim = 2 for real, imag) ===
Tensor complex_mul(const Tensor& a, const Tensor& b);
Tensor view_as_real(const Tensor& x); // stub: just reshapes complex representation
Tensor view_as_complex(const Tensor& x); // stub: just reshapes

// === Normalization ===
Tensor rms_norm(const Tensor& x, const Tensor& gamma, float scale);
Tensor layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps = 1e-5f);
Tensor group_norm(const Tensor& x, int num_groups, const Tensor& gamma, const Tensor& beta, float eps = 1e-5f);
Tensor group_norm_1_btc(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps = 1e-5f);
Tensor instance_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps = 1e-5f);
Tensor batch_norm(const Tensor& x, const Tensor& mean, const Tensor& var,
                  const Tensor& gamma, const Tensor& beta, float eps = 1e-5f);

// === Linear algebra ===
Tensor linear(const Tensor& x, const Tensor& weight, const Tensor& bias); // x @ W^T + b
Tensor linear_no_bias(const Tensor& x, const Tensor& weight);  // x @ W^T
Tensor matmul(const Tensor& a, const Tensor& b); // matrix multiply (supports batched)
Tensor batched_matmul(const Tensor& a, const Tensor& b); // (B,M,K) @ (B,K,N) -> (B,M,N)

// === Convolution (im2col + cuBLAS) ===
Tensor conv1d(const Tensor& x, const Tensor& weight, const Tensor& bias,
              int stride = 1, int padding = 0, int dilation = 1, int groups = 1);
Tensor conv2d(const Tensor& x, const Tensor& weight, const Tensor& bias,
              int stride_h = 1, int stride_w = 1, int pad_h = 0, int pad_w = 0,
              int dilation_h = 1, int dilation_w = 1, int groups = 1);
Tensor conv_transpose1d(const Tensor& x, const Tensor& weight, const Tensor& bias,
                        int stride = 1, int padding = 0, int output_padding = 0);
Tensor conv_transpose2d(const Tensor& x, const Tensor& weight, const Tensor& bias,
                        int stride_h = 1, int stride_w = 1, int pad_h = 0, int pad_w = 0,
                        int output_pad_h = 0, int output_pad_w = 0);

// === Attention ===
Tensor scaled_dot_product_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                                     float scale = 0.0f, float dropout = 0.0f);
// q,k,v: [B, H, N, D]

// === STFT / iSTFT ===
Tensor stft(const Tensor& signal, int n_fft, int hop_length, int win_length,
            const Tensor& window, bool center = true, bool normalized = false);
// Returns [B, n_fft/2+1, T, 2] (real, imag)

Tensor istft(const Tensor& complex_spec, int n_fft, int hop_length, int win_length,
             const Tensor& window, int64_t length = -1, bool center = true, bool normalized = false);
// complex_spec: [B, F, T, 2], returns [B, signal_length]

Tensor hann_window(int size); // returns [size] float32 on GPU

// === Rotary embedding ===
void apply_rotary_emb(Tensor& q, Tensor& k, const Tensor& cos_freqs, const Tensor& sin_freqs);
// q,k: [B, H, N, D], cos/sin: [N, D/2]

// === Embedding ===
Tensor embedding(const Tensor& indices, const Tensor& weight);
// indices: [*], weight: [V, D], returns [*, D]

// === LSTM ===
Tensor bilstm(const Tensor& x, const Tensor& w_ih, const Tensor& w_hh,
              const Tensor& b_ih, const Tensor& b_hh, int hidden_size);
// x: [B, T, input_size], returns [B, T, 2*hidden_size]

// === Scatter ops ===
void scatter_add(Tensor& dest, int dim, const Tensor& indices, const Tensor& src);
// dest[indices[i]] += src[i] along dim

// === GLU ===
Tensor glu(const Tensor& x, int dim = -1);

// === Misc ===
Tensor l2_normalize(const Tensor& x, int dim = -1);
Tensor dropout(const Tensor& x, float p = 0.0f); // inference mode: identity
Tensor index_fill(const Tensor& x, int dim, int64_t index, float value);

// === Mel filter bank ===
Tensor mel_filterbank(int sr, int n_fft, int n_mels, float fmin = 0.0f, float fmax = 0.0f);
// Returns [n_mels, n_fft/2+1] mel filter weights

// === Fused operations (kernel fusion for performance) ===
Tensor linear_gelu(const Tensor& x, const Tensor& weight, const Tensor& bias);
// Fused: out = GELU(x @ W^T + b) — single kernel for bias+activation

Tensor linear_silu(const Tensor& x, const Tensor& weight, const Tensor& bias);
// Fused: out = SiLU(x @ W^T + b)

Tensor linear_sigmoid(const Tensor& x, const Tensor& weight, const Tensor& bias);
// Fused: out = sigmoid(x @ W^T + b)

Tensor scale_residual_add(const Tensor& y, const Tensor& residual,
                           const Tensor& scale, int scale_dim);
// Fused: out = residual + y * scale (broadcast scale along scale_dim)

// === GPU-side overlap-add helpers ===
void overlap_add(Tensor& dest, const Tensor& src, const Tensor& window, int64_t offset);
// dest[c, offset+i] += src[c, i] * window[i]

void weight_accumulate(Tensor& weight_sum, const Tensor& window, int64_t offset);
// weight_sum[offset+i] += window[i]

void normalize_by_weights(Tensor& data, const Tensor& weight_sum);
// data[c, i] /= weight_sum[i]

} // namespace ops
} // namespace cudasep
