// ops_misc.cu - Miscellaneous operations for cudasep inference engine.
//
// Implements: apply_rotary_emb, embedding, bilstm, scatter_add, mel_filterbank.
// All ops work on Float32. For Float16 inputs we cast to f32, compute, cast back.

#include "ops.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <vector>

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
// 1. Rotary Position Embedding
// ---------------------------------------------------------------------------

__global__ void rotary_emb_kernel(float* q, float* k,
                                  const float* cos_f, const float* sin_f,
                                  int B, int H, int N, int D) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int half_D = D / 2;
    int64_t total = (int64_t)B * H * N * half_D;
    if (idx >= total) return;

    int i = idx % half_D;
    int n = (idx / half_D) % N;
    // h and b from remaining
    int h = (idx / ((int64_t)half_D * N)) % H;
    int b = (int)(idx / ((int64_t)half_D * N * H));

    float c = cos_f[n * half_D + i];
    float s = sin_f[n * half_D + i];

    int64_t base = ((int64_t)((int64_t)b * H + h) * N + n) * D;

    // Apply to q
    float q0 = q[base + 2 * i];
    float q1 = q[base + 2 * i + 1];
    q[base + 2 * i]     = q0 * c - q1 * s;
    q[base + 2 * i + 1] = q0 * s + q1 * c;

    // Apply to k
    float k0 = k[base + 2 * i];
    float k1 = k[base + 2 * i + 1];
    k[base + 2 * i]     = k0 * c - k1 * s;
    k[base + 2 * i + 1] = k0 * s + k1 * c;
}

namespace ops {

void apply_rotary_emb(Tensor& q, Tensor& k,
                      const Tensor& cos_freqs, const Tensor& sin_freqs) {
    // q, k: [B, H, N, D]   cos_freqs, sin_freqs: [N, D/2]
    if (q.ndim() != 4 || k.ndim() != 4) {
        throw std::runtime_error("apply_rotary_emb: q and k must be 4-D [B,H,N,D]");
    }

    DType orig_dt = q.dtype();
    bool need_cast = (orig_dt == DType::Float16);

    // If f16, convert to f32, apply, convert back
    if (need_cast) {
        Tensor q32 = q.to_f32().contiguous();
        Tensor k32 = k.to_f32().contiguous();
        Tensor cos32 = ensure_f32(cos_freqs);
        Tensor sin32 = ensure_f32(sin_freqs);

        int B = (int)q32.size(0);
        int H = (int)q32.size(1);
        int N = (int)q32.size(2);
        int D = (int)q32.size(3);
        int half_D = D / 2;
        int64_t total = (int64_t)B * H * N * half_D;
        int grid = (int)ceildiv(total, (int64_t)kBlockSize);

        rotary_emb_kernel<<<grid, kBlockSize>>>(
            q32.data_f32(), k32.data_f32(),
            cos32.data_f32(), sin32.data_f32(),
            B, H, N, D);
        CUDA_CHECK(cudaGetLastError());

        q = q32.to_f16();
        k = k32.to_f16();
        return;
    }

    // Float32 path
    Tensor q_c = q.contiguous();
    Tensor k_c = k.contiguous();
    Tensor cos_c = ensure_f32(cos_freqs);
    Tensor sin_c = ensure_f32(sin_freqs);

    int B = (int)q_c.size(0);
    int H = (int)q_c.size(1);
    int N = (int)q_c.size(2);
    int D = (int)q_c.size(3);
    int half_D = D / 2;
    int64_t total = (int64_t)B * H * N * half_D;
    int grid = (int)ceildiv(total, (int64_t)kBlockSize);

    rotary_emb_kernel<<<grid, kBlockSize>>>(
        q_c.data_f32(), k_c.data_f32(),
        cos_c.data_f32(), sin_c.data_f32(),
        B, H, N, D);
    CUDA_CHECK(cudaGetLastError());

    q = q_c;
    k = k_c;
}

// ---------------------------------------------------------------------------
// 2. Embedding
// ---------------------------------------------------------------------------

__global__ void embedding_kernel(const int64_t* indices, const float* weight,
                                 float* output, int64_t N, int D) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * D) return;
    int64_t token = idx / D;
    int d = (int)(idx % D);
    output[idx] = weight[indices[token] * D + d];
}

Tensor embedding(const Tensor& indices, const Tensor& weight) {
    // indices: arbitrary shape [*], dtype Int64
    // weight: [V, D], dtype Float32
    // output: [*, D]
    if (indices.dtype() != DType::Int64) {
        throw std::runtime_error("embedding: indices must be Int64");
    }
    if (weight.ndim() != 2) {
        throw std::runtime_error("embedding: weight must be 2-D [V, D]");
    }

    Tensor idx_c = indices.contiguous();
    Tensor w_c = ensure_f32(weight);

    int64_t N = idx_c.numel();
    int D = (int)w_c.size(1);

    // Build output shape: indices.shape() + [D]
    std::vector<int64_t> out_shape = idx_c.shape();
    out_shape.push_back((int64_t)D);

    Tensor output = Tensor::empty(out_shape, DType::Float32);

    int64_t total = N * D;
    int grid = (int)ceildiv(total, (int64_t)kBlockSize);

    embedding_kernel<<<grid, kBlockSize>>>(
        idx_c.data_i64(), w_c.data_f32(), output.data_f32(), N, D);
    CUDA_CHECK(cudaGetLastError());

    return output;
}

// ---------------------------------------------------------------------------
// 3. Bidirectional LSTM
// ---------------------------------------------------------------------------

// Kernel to apply LSTM gate activations and produce new c and h.
// gates: [B, 4*H]  (already bias-added)
// prev_c: [B, H]
// new_c, new_h: [B, H]
__global__ void lstm_gates_kernel(const float* gates, const float* prev_c,
                                  float* new_c, float* new_h, int B, int H) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)B * H) return;

    int b = (int)(idx / H);
    int j = (int)(idx % H);

    float i_gate = gates[(int64_t)b * 4 * H + 0 * H + j];
    float f_gate = gates[(int64_t)b * 4 * H + 1 * H + j];
    float g_gate = gates[(int64_t)b * 4 * H + 2 * H + j];
    float o_gate = gates[(int64_t)b * 4 * H + 3 * H + j];

    // Activations
    i_gate = 1.0f / (1.0f + expf(-i_gate));  // sigmoid
    f_gate = 1.0f / (1.0f + expf(-f_gate));  // sigmoid
    g_gate = tanhf(g_gate);                    // tanh
    o_gate = 1.0f / (1.0f + expf(-o_gate));  // sigmoid

    float c = f_gate * prev_c[idx] + i_gate * g_gate;
    new_c[idx] = c;
    new_h[idx] = o_gate * tanhf(c);
}

// Kernel to add bias to gate matrix: gates[b, j] += b_ih[j] + b_hh[j]
__global__ void lstm_add_bias_kernel(float* gates, const float* b_ih, const float* b_hh,
                                     int B, int gate_size) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)B * gate_size) return;
    int j = (int)(idx % gate_size);
    gates[idx] += b_ih[j] + b_hh[j];
}

// Copy h into output at the correct direction offset
// h: [B, H]  ->  output[:, t, offset : offset+H]
// output: [B, T, 2*H]
__global__ void lstm_copy_h_kernel(const float* h, float* output,
                                   int B, int T, int H, int t, int offset) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)B * H) return;
    int b = (int)(idx / H);
    int j = (int)(idx % H);
    output[((int64_t)b * T + t) * 2 * H + offset + j] = h[idx];
}

Tensor bilstm(const Tensor& x, const Tensor& w_ih, const Tensor& w_hh,
              const Tensor& b_ih, const Tensor& b_hh, int hidden_size) {
    // x: [B, T, input_size]
    // w_ih: [2, 4*H, input_size]  (forward/backward stacked)
    // w_hh: [2, 4*H, H]
    // b_ih: [2, 4*H]
    // b_hh: [2, 4*H]
    // Returns: [B, T, 2*H]

    if (x.ndim() != 3) {
        throw std::runtime_error("bilstm: x must be 3-D [B, T, input_size]");
    }

    DType orig_dt = x.dtype();
    Tensor x_f = ensure_f32(x);
    Tensor wih = ensure_f32(w_ih);
    Tensor whh = ensure_f32(w_hh);
    Tensor bih = ensure_f32(b_ih);
    Tensor bhh = ensure_f32(b_hh);

    int B = (int)x_f.size(0);
    int T = (int)x_f.size(1);
    int input_size = (int)x_f.size(2);
    int H = hidden_size;
    int gate_size = 4 * H;

    cublasHandle_t handle = CudaContext::instance().cublas();

    // Allocate output [B, T, 2*H]
    Tensor output = Tensor::zeros({(int64_t)B, (int64_t)T, (int64_t)(2 * H)}, DType::Float32);

    // Temporary buffers
    Tensor h_state = Tensor::zeros({(int64_t)B, (int64_t)H}, DType::Float32);
    Tensor c_state = Tensor::zeros({(int64_t)B, (int64_t)H}, DType::Float32);
    Tensor gates = Tensor::empty({(int64_t)B, (int64_t)gate_size}, DType::Float32);
    Tensor new_c = Tensor::empty({(int64_t)B, (int64_t)H}, DType::Float32);
    Tensor new_h = Tensor::empty({(int64_t)B, (int64_t)H}, DType::Float32);

    // Temp for a single timestep of input: [B, input_size]
    // We'll slice from x_f directly.

    // Helper: run one direction
    // dir=0 forward, dir=1 backward
    for (int dir = 0; dir < 2; dir++) {
        // Extract weights for this direction
        // w_ih[dir]: [4*H, input_size]
        // w_hh[dir]: [4*H, H]
        // b_ih[dir]: [4*H]
        // b_hh[dir]: [4*H]
        Tensor dir_wih = wih.slice(0, dir, dir + 1).reshape({(int64_t)gate_size, (int64_t)input_size});
        Tensor dir_whh = whh.slice(0, dir, dir + 1).reshape({(int64_t)gate_size, (int64_t)H});
        Tensor dir_bih = bih.slice(0, dir, dir + 1).reshape({(int64_t)gate_size});
        Tensor dir_bhh = bhh.slice(0, dir, dir + 1).reshape({(int64_t)gate_size});

        // Make them contiguous
        dir_wih = dir_wih.contiguous();
        dir_whh = dir_whh.contiguous();
        dir_bih = dir_bih.contiguous();
        dir_bhh = dir_bhh.contiguous();

        // Reset states
        h_state.fill_(0.0f);
        c_state.fill_(0.0f);

        int h_offset = dir * H;  // where to write h in output

        for (int step = 0; step < T; step++) {
            int t = (dir == 0) ? step : (T - 1 - step);

            // x_t: [B, input_size]  = x_f[:, t, :]
            Tensor x_t = x_f.slice(1, t, t + 1).reshape({(int64_t)B, (int64_t)input_size}).contiguous();

            // gates = x_t @ W_ih^T   ->  [B, 4*H]
            // cuBLAS: C = alpha * A * B + beta * C
            // We want: gates[B, gate_size] = x_t[B, input_size] * dir_wih^T[input_size, gate_size]
            // cuBLAS column-major:
            //   C(gate_size, B) = W_ih(gate_size, input_size) * x_t^T(input_size, B)
            float alpha = 1.0f;
            float beta = 0.0f;

            cublasStatus_t stat = cublasSgemm(handle,
                CUBLAS_OP_N,     // W_ih not transposed (in col-major = transposed in row-major)
                CUBLAS_OP_T,     // x_t transposed (in col-major = not transposed in row-major)
                gate_size,       // m
                B,               // n
                input_size,      // k
                &alpha,
                dir_wih.data_f32(), gate_size,  // A: [gate_size x input_size] col-major
                x_t.data_f32(), input_size,     // B: [input_size x B] col-major (= x_t^T)
                &beta,
                gates.data_f32(), gate_size);   // C: [gate_size x B] col-major
            if (stat != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("bilstm: cublasSgemm (x_t @ W_ih^T) failed");
            }

            // gates += h @ W_hh^T
            // Same pattern: C(gate_size, B) += W_hh(gate_size, H) * h^T(H, B)
            beta = 1.0f;
            stat = cublasSgemm(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                gate_size,
                B,
                H,
                &alpha,
                dir_whh.data_f32(), gate_size,
                h_state.data_f32(), H,
                &beta,
                gates.data_f32(), gate_size);
            if (stat != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("bilstm: cublasSgemm (h @ W_hh^T) failed");
            }

            // gates += b_ih + b_hh
            {
                int64_t total_bias = (int64_t)B * gate_size;
                int grid = (int)ceildiv(total_bias, (int64_t)kBlockSize);
                lstm_add_bias_kernel<<<grid, kBlockSize>>>(
                    gates.data_f32(), dir_bih.data_f32(), dir_bhh.data_f32(),
                    B, gate_size);
                CUDA_CHECK(cudaGetLastError());
            }

            // Apply gate activations -> new_c, new_h
            {
                int64_t total_ch = (int64_t)B * H;
                int grid = (int)ceildiv(total_ch, (int64_t)kBlockSize);
                lstm_gates_kernel<<<grid, kBlockSize>>>(
                    gates.data_f32(), c_state.data_f32(),
                    new_c.data_f32(), new_h.data_f32(),
                    B, H);
                CUDA_CHECK(cudaGetLastError());
            }

            // Update states
            c_state.copy_from(new_c);
            h_state.copy_from(new_h);

            // Copy h to output[:, t, h_offset : h_offset + H]
            {
                int64_t total_copy = (int64_t)B * H;
                int grid = (int)ceildiv(total_copy, (int64_t)kBlockSize);
                lstm_copy_h_kernel<<<grid, kBlockSize>>>(
                    new_h.data_f32(), output.data_f32(),
                    B, T, H, t, h_offset);
                CUDA_CHECK(cudaGetLastError());
            }
        }
    }

    return maybe_cast_back(output, orig_dt);
}

// ---------------------------------------------------------------------------
// 4. Scatter Add
// ---------------------------------------------------------------------------

// scatter_add along a specified dimension.
// dest and src may differ in size along `dim`. indices has same shape as src.
// For each element in src at multi-index [i0, i1, ..., iN]:
//   dest[i0, ..., indices[i0,...,iN], ..., iN] += src[i0, ..., iN]
// where the indices value replaces the index at dimension `dim`.

__global__ void scatter_add_kernel(float* dest, const int64_t* indices, const float* src,
                                   int64_t total_elements, int64_t outer_stride_src,
                                   int64_t dim_size_src, int64_t inner_size,
                                   int64_t outer_stride_dest, int64_t dim_size_dest) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Decompose idx into (outer, d, inner) w.r.t. src shape
    int64_t inner = idx % inner_size;
    int64_t d = (idx / inner_size) % dim_size_src;
    int64_t outer = idx / (dim_size_src * inner_size);
    (void)d; // d is the source's index along dim, we don't use it for dest

    int64_t target_dim = indices[idx];

    // dest offset: outer * (dim_size_dest * inner_size) + target_dim * inner_size + inner
    int64_t dest_idx = outer * (dim_size_dest * inner_size) + target_dim * inner_size + inner;

    atomicAdd(&dest[dest_idx], src[idx]);
}

void scatter_add(Tensor& dest, int dim, const Tensor& indices, const Tensor& src) {
    if (indices.dtype() != DType::Int64) {
        throw std::runtime_error("scatter_add: indices must be Int64");
    }
    if (src.numel() != indices.numel()) {
        throw std::runtime_error("scatter_add: src and indices must have same number of elements");
    }

    // Normalize dim
    int ndim = src.ndim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("scatter_add: dim out of range");
    }

    // Ensure contiguous f32 for dest and src
    bool dest_was_f16 = (dest.dtype() == DType::Float16);
    Tensor dest_f32 = ensure_f32(dest);
    Tensor src_f32 = ensure_f32(src);
    Tensor idx_c = indices.contiguous();

    // Compute inner_size = product of dims after dim
    int64_t inner_size = 1;
    for (int i = dim + 1; i < ndim; i++) {
        inner_size *= src.size(i);
    }

    int64_t dim_size_src = src.size(dim);
    int64_t dim_size_dest = dest.size(dim);

    int64_t total = src_f32.numel();
    int grid = (int)ceildiv(total, (int64_t)kBlockSize);

    int64_t outer_stride_src = dim_size_src * inner_size;
    int64_t outer_stride_dest = dim_size_dest * inner_size;

    scatter_add_kernel<<<grid, kBlockSize>>>(
        dest_f32.data_f32(), idx_c.data_i64(), src_f32.data_f32(),
        total, outer_stride_src, dim_size_src, inner_size,
        outer_stride_dest, dim_size_dest);
    CUDA_CHECK(cudaGetLastError());

    if (dest_was_f16) {
        dest = dest_f32.to_f16();
    } else {
        dest = dest_f32;
    }
}

// ---------------------------------------------------------------------------
// 5. Mel Filterbank
// ---------------------------------------------------------------------------

Tensor mel_filterbank(int sr, int n_fft, int n_mels, float fmin, float fmax) {
    if (fmax <= 0.0f) fmax = (float)sr / 2.0f;
    int num_fft_bins = n_fft / 2 + 1;

    auto hz_to_mel = [](float f) -> float {
        return 2595.0f * std::log10(1.0f + f / 700.0f);
    };
    auto mel_to_hz = [](float m) -> float {
        return 700.0f * (std::pow(10.0f, m / 2595.0f) - 1.0f);
    };

    float mel_min = hz_to_mel(fmin);
    float mel_max = hz_to_mel(fmax);

    // n_mels + 2 evenly spaced points in mel scale
    std::vector<float> mel_points(n_mels + 2);
    for (int i = 0; i <= n_mels + 1; i++) {
        mel_points[i] = mel_min + (mel_max - mel_min) * (float)i / (float)(n_mels + 1);
    }

    // Convert to Hz
    std::vector<float> hz_points(n_mels + 2);
    for (int i = 0; i <= n_mels + 1; i++) {
        hz_points[i] = mel_to_hz(mel_points[i]);
    }

    // FFT bin center frequencies
    std::vector<float> bin_freqs(num_fft_bins);
    for (int i = 0; i < num_fft_bins; i++) {
        bin_freqs[i] = (float)sr * (float)i / (float)n_fft;
    }

    // Build triangular filterbank [n_mels, num_fft_bins]
    std::vector<float> fb(n_mels * num_fft_bins, 0.0f);
    for (int m = 0; m < n_mels; m++) {
        float left = hz_points[m];
        float center = hz_points[m + 1];
        float right = hz_points[m + 2];

        for (int f = 0; f < num_fft_bins; f++) {
            float freq = bin_freqs[f];
            if (freq >= left && freq <= center) {
                float denom = center - left;
                if (denom > 0.0f) {
                    fb[m * num_fft_bins + f] = (freq - left) / denom;
                }
            } else if (freq > center && freq <= right) {
                float denom = right - center;
                if (denom > 0.0f) {
                    fb[m * num_fft_bins + f] = (right - freq) / denom;
                }
            }
        }
    }

    return Tensor::from_cpu_f32(fb.data(), {(int64_t)n_mels, (int64_t)num_fft_bins});
}

} // namespace ops
} // namespace cudasep
