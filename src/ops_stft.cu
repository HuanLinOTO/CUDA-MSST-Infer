// ops_stft.cu - STFT, iSTFT and hann_window operations for cudasep inference engine.
//
// Uses cuFFT for FFT/IFFT. All ops work on Float32.
// STFT matches PyTorch's torch.stft behaviour; iSTFT matches torch.istft.

#include "ops.h"
#include <cufft.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// cuFFT error checking
// ---------------------------------------------------------------------------

#define CUFFT_CHECK(call) do { \
    cufftResult _err = (call); \
    if (_err != CUFFT_SUCCESS) { \
        throw std::runtime_error(std::string("cuFFT error: ") + \
            std::to_string((int)_err) + " at " + __FILE__ + ":" + \
            std::to_string(__LINE__)); \
    } \
} while(0)

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

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Extract overlapping frames from a signal.
// signal: [B, signal_length]  ->  frames: [B, T, n_fft]
__global__ void extract_frames_kernel(const float* __restrict__ signal,
                                      float* __restrict__ frames,
                                      int signal_length, int n_fft,
                                      int hop_length, int T, int B)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)B * T * n_fft;
    if (idx >= total) return;

    int n = idx % n_fft;
    int t = (idx / n_fft) % T;
    int b = idx / ((int64_t)n_fft * T);

    frames[idx] = signal[(int64_t)b * signal_length + (int64_t)t * hop_length + n];
}

// Multiply every frame element by the corresponding window value.
// frames: [B * T * n_fft]  (in-place)
__global__ void apply_window_kernel(float* __restrict__ frames,
                                    const float* __restrict__ window,
                                    int n_fft, int64_t total)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    frames[idx] *= window[idx % n_fft];
}

// Convert cuFFT R2C output ([B*T, F] cufftComplex) to [B, F, T, 2] float.
__global__ void complex_to_real_imag_kernel(const cufftComplex* __restrict__ complex_data,
                                            float* __restrict__ output,
                                            int B, int F, int T)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)B * F * T;
    if (idx >= total) return;

    int t_local = idx % T;
    int f       = (idx / T) % F;
    int b       = idx / ((int64_t)T * F);

    // complex_data layout: [B*T, F]
    cufftComplex val = complex_data[(int64_t)b * T * F + (int64_t)t_local * F + f];

    // output layout: [B, F, T, 2]
    int64_t out_base = ((int64_t)b * F * T + (int64_t)f * T + t_local) * 2;
    output[out_base]     = val.x;   // real
    output[out_base + 1] = val.y;   // imag
}

// Scale every element by a constant factor (in-place).
__global__ void stft_scale_kernel(float* __restrict__ data, float factor, int64_t N)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) data[i] *= factor;
}

// Convert [B, F, T, 2] float to [B*T, F] cufftComplex.
__global__ void real_imag_to_complex_kernel(const float* __restrict__ input,
                                            cufftComplex* __restrict__ complex_data,
                                            int B, int F, int T)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)B * F * T;
    if (idx >= total) return;

    int t_local = idx % T;
    int f       = (idx / T) % F;
    int b       = idx / ((int64_t)T * F);

    // input layout: [B, F, T, 2]
    int64_t in_base = ((int64_t)b * F * T + (int64_t)f * T + t_local) * 2;
    float re = input[in_base];
    float im = input[in_base + 1];

    // complex_data layout: [B*T, F]
    int64_t ci = (int64_t)b * T * F + (int64_t)t_local * F + f;
    complex_data[ci].x = re;
    complex_data[ci].y = im;
}

// Overlap-add: scatter windowed frames back into the output signal.
// frames: [B, T, n_fft]  ->  output: [B, signal_length]  (atomicAdd)
__global__ void overlap_add_kernel(const float* __restrict__ frames,
                                   float* __restrict__ output,
                                   int n_fft, int hop_length,
                                   int T, int signal_length, int B)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)B * T * n_fft;
    if (idx >= total) return;

    int n = idx % n_fft;
    int t = (idx / n_fft) % T;
    int b = idx / ((int64_t)n_fft * T);

    int out_pos = t * hop_length + n;
    if (out_pos < signal_length) {
        atomicAdd(&output[(int64_t)b * signal_length + out_pos], frames[idx]);
    }
}

// Compute the sum of squared window values at each output position (COLA normalizer).
// window_sum: [signal_length]
__global__ void window_sum_kernel(const float* __restrict__ window,
                                  float* __restrict__ window_sum,
                                  int n_fft, int hop_length,
                                  int T, int signal_length)
{
    int64_t pos = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= signal_length) return;

    float sum = 0.0f;
    for (int t = 0; t < T; t++) {
        int n = (int)pos - t * hop_length;
        if (n >= 0 && n < n_fft) {
            float w = window[n];
            sum += w * w;
        }
    }
    window_sum[pos] = sum;
}

// Divide signal by window_sum, per-batch.
// output: [B, signal_length], window_sum: [signal_length]
__global__ void normalize_by_window_kernel(float* __restrict__ output,
                                           const float* __restrict__ window_sum,
                                           int signal_length, int64_t N)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int pos = (int)(i % signal_length);
    float ws = window_sum[pos];
    if (ws > 1e-8f) {
        output[i] /= ws;
    }
}

// Match torch.hann_window(size) default behavior (periodic=True):
// w[i] = 0.5 * (1 - cos(2*pi*i / size))
__global__ void hann_window_kernel(float* __restrict__ out, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    float phase = 2.0f * (float)M_PI * (float)i / (float)size;
    out[i] = 0.5f * (1.0f - cosf(phase));
}

// ---------------------------------------------------------------------------
// hann_window
// ---------------------------------------------------------------------------

namespace ops {

Tensor hann_window(int size) {
    Tensor out = Tensor::empty({(int64_t)size}, DType::Float32);
    if (size <= 0) return out;

    if (size == 1) {
        out.fill_(1.0f);
        return out;
    }

    int grid = (int)ceildiv((int64_t)size, (int64_t)kBlockSize);
    hann_window_kernel<<<grid, kBlockSize>>>(out.data_f32(), size);
    CUDA_CHECK(cudaGetLastError());
    return out;
}

// ---------------------------------------------------------------------------
// stft
// ---------------------------------------------------------------------------

Tensor stft(const Tensor& signal, int n_fft, int hop_length, int win_length,
            const Tensor& window, bool center, bool normalized)
{
    // --- Validate inputs ---
    Tensor sig = ensure_f32(signal);
    Tensor win = ensure_f32(window);

    // Support 1-D input [signal_length] by treating as batch-1
    bool was_1d = (sig.ndim() == 1);
    if (was_1d) {
        sig = sig.reshape({1, sig.size(0)});
    }

    int B = (int)sig.size(0);
    int orig_len = (int)sig.size(1);

    // 1. Center padding
    if (center) {
        int pad_amount = n_fft / 2;
        sig = sig.pad_reflect({(int64_t)pad_amount, (int64_t)pad_amount});
    }

    int padded_length = (int)sig.size(1);

    // 2. Frame extraction
    int T = (padded_length - n_fft) / hop_length + 1;  // number of frames

    int64_t total_frame_elems = (int64_t)B * T * n_fft;
    Tensor frames = Tensor::empty({(int64_t)B, (int64_t)T, (int64_t)n_fft}, DType::Float32);

    {
        int grid = (int)ceildiv(total_frame_elems, (int64_t)kBlockSize);
        extract_frames_kernel<<<grid, kBlockSize>>>(
            sig.data_f32(), frames.data_f32(),
            padded_length, n_fft, hop_length, T, B);
        CUDA_CHECK(cudaGetLastError());
    }

    // 3. Apply window
    {
        int grid = (int)ceildiv(total_frame_elems, (int64_t)kBlockSize);
        apply_window_kernel<<<grid, kBlockSize>>>(
            frames.data_f32(), win.data_f32(), n_fft, total_frame_elems);
        CUDA_CHECK(cudaGetLastError());
    }

    // 4. cuFFT R2C  (batched: B*T transforms of size n_fft)
    int F = n_fft / 2 + 1;  // number of frequency bins

    // Allocate complex output: [B*T, F] cufftComplex
    int64_t complex_elems = (int64_t)B * T * F;
    cufftComplex* d_complex = nullptr;
    CUDA_CHECK(cudaMalloc(&d_complex, complex_elems * sizeof(cufftComplex)));

    {
        cufftHandle plan;
        CUFFT_CHECK(cufftPlan1d(&plan, n_fft, CUFFT_R2C, B * T));
        CUFFT_CHECK(cufftExecR2C(plan, frames.data_f32(), d_complex));
        CUFFT_CHECK(cufftDestroy(plan));
    }

    // 5. Reshape: [B*T, F] complex -> [B, F, T, 2] float
    Tensor output = Tensor::empty({(int64_t)B, (int64_t)F, (int64_t)T, 2}, DType::Float32);

    {
        int grid = (int)ceildiv(complex_elems, (int64_t)kBlockSize);
        complex_to_real_imag_kernel<<<grid, kBlockSize>>>(
            d_complex, output.data_f32(), B, F, T);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaFree(d_complex));

    // 6. Normalization
    if (normalized) {
        float norm_factor = 1.0f / sqrtf((float)n_fft);
        int64_t N = output.numel();
        int grid = (int)ceildiv(N, (int64_t)kBlockSize);
        stft_scale_kernel<<<grid, kBlockSize>>>(output.data_f32(), norm_factor, N);
        CUDA_CHECK(cudaGetLastError());
    }

    return output;
}

// ---------------------------------------------------------------------------
// istft
// ---------------------------------------------------------------------------

Tensor istft(const Tensor& complex_spec, int n_fft, int hop_length, int win_length,
             const Tensor& window, int64_t length, bool center, bool normalized)
{
    // --- Validate inputs ---
    Tensor spec = ensure_f32(complex_spec);
    Tensor win  = ensure_f32(window);

    // spec: [B, F, T, 2]
    int B = (int)spec.size(0);
    int F = (int)spec.size(1);
    int T = (int)spec.size(2);

    // If normalized, undo the normalization first (multiply by sqrt(n_fft))
    if (normalized) {
        float norm_factor = sqrtf((float)n_fft);
        int64_t N = spec.numel();
        // Need a mutable copy
        spec = spec.clone();
        int grid = (int)ceildiv(N, (int64_t)kBlockSize);
        stft_scale_kernel<<<grid, kBlockSize>>>(spec.data_f32(), norm_factor, N);
        CUDA_CHECK(cudaGetLastError());
    }

    // 1. Convert [B, F, T, 2] -> [B*T, F] cufftComplex
    int64_t complex_elems = (int64_t)B * F * T;
    cufftComplex* d_complex = nullptr;
    CUDA_CHECK(cudaMalloc(&d_complex, complex_elems * sizeof(cufftComplex)));

    {
        int grid = (int)ceildiv(complex_elems, (int64_t)kBlockSize);
        real_imag_to_complex_kernel<<<grid, kBlockSize>>>(
            spec.data_f32(), d_complex, B, F, T);
        CUDA_CHECK(cudaGetLastError());
    }

    // 2. cuFFT C2R  (batched: B*T transforms of size n_fft)
    // Output: [B*T, n_fft]
    Tensor frames = Tensor::empty({(int64_t)B * T, (int64_t)n_fft}, DType::Float32);

    {
        cufftHandle plan;
        CUFFT_CHECK(cufftPlan1d(&plan, n_fft, CUFFT_C2R, B * T));
        CUFFT_CHECK(cufftExecC2R(plan, d_complex, frames.data_f32()));
        CUFFT_CHECK(cufftDestroy(plan));
    }

    CUDA_CHECK(cudaFree(d_complex));

    // cuFFT C2R output is unnormalized: divide by n_fft
    {
        float inv_nfft = 1.0f / (float)n_fft;
        int64_t N = frames.numel();
        int grid = (int)ceildiv(N, (int64_t)kBlockSize);
        stft_scale_kernel<<<grid, kBlockSize>>>(frames.data_f32(), inv_nfft, N);
        CUDA_CHECK(cudaGetLastError());
    }

    // Reshape to [B, T, n_fft]
    frames = frames.reshape({(int64_t)B, (int64_t)T, (int64_t)n_fft});

    // 3. Apply window to each reconstructed frame
    {
        int64_t total = frames.numel();
        int grid = (int)ceildiv(total, (int64_t)kBlockSize);
        apply_window_kernel<<<grid, kBlockSize>>>(
            frames.data_f32(), win.data_f32(), n_fft, total);
        CUDA_CHECK(cudaGetLastError());
    }

    // 4. Overlap-add into output signal
    int signal_length = n_fft + (T - 1) * hop_length;  // full reconstructed length
    Tensor output = Tensor::zeros({(int64_t)B, (int64_t)signal_length}, DType::Float32);

    {
        int64_t total = (int64_t)B * T * n_fft;
        int grid = (int)ceildiv(total, (int64_t)kBlockSize);
        overlap_add_kernel<<<grid, kBlockSize>>>(
            frames.data_f32(), output.data_f32(),
            n_fft, hop_length, T, signal_length, B);
        CUDA_CHECK(cudaGetLastError());
    }

    // 5. Window normalization (COLA)
    //    Compute window_sum for one period, then divide each batch by it.
    Tensor window_sum = Tensor::empty({(int64_t)signal_length}, DType::Float32);

    {
        int grid = (int)ceildiv((int64_t)signal_length, (int64_t)kBlockSize);
        window_sum_kernel<<<grid, kBlockSize>>>(
            win.data_f32(), window_sum.data_f32(),
            n_fft, hop_length, T, signal_length);
        CUDA_CHECK(cudaGetLastError());
    }

    {
        int64_t N = output.numel();
        int grid = (int)ceildiv(N, (int64_t)kBlockSize);
        normalize_by_window_kernel<<<grid, kBlockSize>>>(
            output.data_f32(), window_sum.data_f32(),
            signal_length, N);
        CUDA_CHECK(cudaGetLastError());
    }

    // 6. Trim center padding
    if (center) {
        int pad = n_fft / 2;
        output = output.slice(1, pad, signal_length - pad);
        signal_length = (int)output.size(1);
    }

    // 7. Truncate or pad to requested length
    if (length > 0) {
        if (length < signal_length) {
            output = output.slice(1, 0, length);
        } else if (length > signal_length) {
            // Pad with zeros on the right
            int64_t pad_right = length - signal_length;
            output = output.pad({0, pad_right}, 0.0f);
        }
    }

    return output;
}

} // namespace ops
} // namespace cudasep
