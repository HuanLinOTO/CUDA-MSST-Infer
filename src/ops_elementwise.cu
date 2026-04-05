// ops_elementwise.cu - Element-wise operations, softmax, GLU, complex_mul,
// l2_normalize, index_fill, dropout for cudasep inference engine.
//
// All ops work on Float32. For Float16 inputs we cast to f32, compute, cast back.

#include "ops.h"
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

// ---------------------------------------------------------------------------
// Generic unary kernel
// ---------------------------------------------------------------------------

template <typename OpFunc>
__global__ void unary_kernel(const float* __restrict__ in,
                             float* __restrict__ out,
                             int64_t N, OpFunc op) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = op(in[i]);
}

// Helper that applies an OpFunc element-wise.
template <typename OpFunc>
static Tensor apply_unary(const Tensor& x, OpFunc op) {
    DType orig = x.dtype();
    Tensor xf = ensure_f32(x);
    int64_t N = xf.numel();
    Tensor out = Tensor::empty(xf.shape(), DType::Float32);

    int grid = (int)ceildiv(N, (int64_t)kBlockSize);
    unary_kernel<<<grid, kBlockSize>>>(xf.data_f32(), out.data_f32(), N, op);
    CUDA_CHECK(cudaGetLastError());

    return maybe_cast_back(out, orig);
}

// ---------------------------------------------------------------------------
// Activation functor structs (device-side)
// ---------------------------------------------------------------------------

struct GeluOp {
    __device__ float operator()(float x) const {
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        constexpr float kSqrt2OverPi = 0.7978845608028654f;
        float cube = x * x * x;
        float inner = kSqrt2OverPi * (x + 0.044715f * cube);
        return 0.5f * x * (1.0f + tanhf(inner));
    }
};

struct ReluOp {
    __device__ float operator()(float x) const {
        return x > 0.0f ? x : 0.0f;
    }
};

struct SigmoidOp {
    __device__ float operator()(float x) const {
        return 1.0f / (1.0f + expf(-x));
    }
};

struct TanhOp {
    __device__ float operator()(float x) const {
        return tanhf(x);
    }
};

struct SiluOp {
    __device__ float operator()(float x) const {
        return x / (1.0f + expf(-x));
    }
};

struct ExpOp {
    __device__ float operator()(float x) const { return expf(x); }
};

struct LogOp {
    __device__ float operator()(float x) const { return logf(x); }
};

struct SqrtOp {
    __device__ float operator()(float x) const { return sqrtf(x); }
};

struct AbsOp {
    __device__ float operator()(float x) const { return fabsf(x); }
};

struct CosOp {
    __device__ float operator()(float x) const { return cosf(x); }
};

struct SinOp {
    __device__ float operator()(float x) const { return sinf(x); }
};

struct RsqrtOp {
    __device__ float operator()(float x) const { return rsqrtf(x); }
};

struct PowOp {
    float exponent;
    __device__ float operator()(float x) const { return powf(x, exponent); }
};

// ---------------------------------------------------------------------------
// Activation / math public API
// ---------------------------------------------------------------------------

namespace ops {

Tensor gelu(const Tensor& x) { return apply_unary(x, GeluOp{}); }
Tensor relu(const Tensor& x) { return apply_unary(x, ReluOp{}); }
Tensor sigmoid(const Tensor& x) { return apply_unary(x, SigmoidOp{}); }
Tensor tanh_act(const Tensor& x) { return apply_unary(x, TanhOp{}); }
Tensor silu(const Tensor& x) { return apply_unary(x, SiluOp{}); }

Tensor exp(const Tensor& x) { return apply_unary(x, ExpOp{}); }
Tensor log(const Tensor& x) { return apply_unary(x, LogOp{}); }
Tensor sqrt(const Tensor& x) { return apply_unary(x, SqrtOp{}); }
Tensor abs(const Tensor& x) { return apply_unary(x, AbsOp{}); }
Tensor cos(const Tensor& x) { return apply_unary(x, CosOp{}); }
Tensor sin(const Tensor& x) { return apply_unary(x, SinOp{}); }
Tensor rsqrt(const Tensor& x) { return apply_unary(x, RsqrtOp{}); }

Tensor pow(const Tensor& x, float exponent) {
    return apply_unary(x, PowOp{exponent});
}

// ---------------------------------------------------------------------------
// Softmax (numerically stable, along a given dim)
// ---------------------------------------------------------------------------

// For simplicity we handle softmax on the last dim.  The input is reshaped to
// [outer, inner_dim] where inner_dim = x.size(dim).  We assign one block per
// row and use shared-memory reductions for max and sum.

__global__ void softmax_kernel(const float* __restrict__ in,
                               float* __restrict__ out,
                               int64_t outer, int inner_dim) {
    int row = blockIdx.x;
    if (row >= outer) return;

    extern __shared__ float smem[]; // size = blockDim.x
    const float* row_in = in + (int64_t)row * inner_dim;
    float* row_out = out + (int64_t)row * inner_dim;
    int tid = threadIdx.x;

    // --- pass 1: max ---
    float local_max = -FLT_MAX;
    for (int j = tid; j < inner_dim; j += blockDim.x) {
        float v = row_in[j];
        if (v > local_max) local_max = v;
    }
    smem[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s && smem[tid + s] > smem[tid]) smem[tid] = smem[tid + s];
        __syncthreads();
    }
    float row_max = smem[0];
    __syncthreads();

    // --- pass 2: exp and sum ---
    float local_sum = 0.0f;
    for (int j = tid; j < inner_dim; j += blockDim.x) {
        float e = expf(row_in[j] - row_max);
        row_out[j] = e;          // store intermediate
        local_sum += e;
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / smem[0];
    __syncthreads();

    // --- pass 3: normalize ---
    for (int j = tid; j < inner_dim; j += blockDim.x) {
        row_out[j] *= inv_sum;
    }
}

Tensor softmax(const Tensor& x, int dim) {
    DType orig = x.dtype();
    Tensor xf = ensure_f32(x);

    int ndim = xf.ndim();
    if (dim < 0) dim += ndim;
    assert(dim >= 0 && dim < ndim);

    // Move target dim to the last position so kernel can work row-wise.
    // If dim is already last, no permute needed.
    Tensor inp = xf;
    bool permuted = false;
    std::vector<int> perm_fwd, perm_inv;
    if (dim != ndim - 1) {
        permuted = true;
        perm_fwd.resize(ndim);
        perm_inv.resize(ndim);
        // build forward permutation: move dim to end
        int idx = 0;
        for (int i = 0; i < ndim; ++i) {
            if (i == dim) continue;
            perm_fwd[idx++] = i;
        }
        perm_fwd[ndim - 1] = dim;
        // inverse
        for (int i = 0; i < ndim; ++i) perm_inv[perm_fwd[i]] = i;
        inp = xf.permute(perm_fwd).contiguous();
    }

    int64_t inner_dim = inp.size(ndim - 1);
    int64_t outer = inp.numel() / inner_dim;

    Tensor out = Tensor::empty(inp.shape(), DType::Float32);

    int threads = 256;
    if (inner_dim < threads) {
        threads = 1;
        while (threads < inner_dim) threads <<= 1;
        if (threads > 1024) threads = 1024;
    }
    size_t smem_bytes = threads * sizeof(float);

    softmax_kernel<<<(int)outer, threads, smem_bytes>>>(
        inp.data_f32(), out.data_f32(), outer, (int)inner_dim);
    CUDA_CHECK(cudaGetLastError());

    if (permuted) {
        out = out.permute(perm_inv).contiguous();
    }

    return maybe_cast_back(out, orig);
}

// ---------------------------------------------------------------------------
// complex_mul:  [..., 2] x [..., 2] -> [..., 2]
// (a_r*b_r - a_i*b_i,  a_r*b_i + a_i*b_r)
// ---------------------------------------------------------------------------

__global__ void complex_mul_kernel(const float* __restrict__ a,
                                   const float* __restrict__ b,
                                   float* __restrict__ out,
                                   int64_t num_complex) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_complex) {
        float ar = a[2 * i];
        float ai = a[2 * i + 1];
        float br = b[2 * i];
        float bi = b[2 * i + 1];
        out[2 * i]     = ar * br - ai * bi;
        out[2 * i + 1] = ar * bi + ai * br;
    }
}

Tensor complex_mul(const Tensor& a, const Tensor& b) {
    DType orig = a.dtype();
    Tensor af = ensure_f32(a);
    Tensor bf = ensure_f32(b);

    assert(af.ndim() >= 1 && af.size(af.ndim() - 1) == 2);
    assert(af.shape() == bf.shape());

    int64_t num_complex = af.numel() / 2;
    Tensor out = Tensor::empty(af.shape(), DType::Float32);

    int grid = (int)ceildiv(num_complex, (int64_t)kBlockSize);
    complex_mul_kernel<<<grid, kBlockSize>>>(
        af.data_f32(), bf.data_f32(), out.data_f32(), num_complex);
    CUDA_CHECK(cudaGetLastError());

    return maybe_cast_back(out, orig);
}

// ---------------------------------------------------------------------------
// view_as_real / view_as_complex  (shape stubs)
// ---------------------------------------------------------------------------

Tensor view_as_real(const Tensor& x) {
    // Assume x is already stored as [..., 2].  No-op at inference.
    return x;
}

Tensor view_as_complex(const Tensor& x) {
    // Assume last dim == 2; just return unchanged.  Caller is responsible for
    // interpreting last dim as (real, imag).
    assert(x.ndim() >= 1 && x.size(x.ndim() - 1) == 2);
    return x;
}

// ---------------------------------------------------------------------------
// GLU:  split along dim into two halves, return first * sigmoid(second)
// ---------------------------------------------------------------------------

__global__ void glu_fused_kernel(const float* __restrict__ x,
                                  float* __restrict__ out,
                                  int64_t half_dim,
                                  int64_t inner_size,
                                  int64_t full_dim_stride,  // = full_dim * inner_size
                                  int64_t total) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    
    // Decompose output index: out is [outer, half_dim, inner]
    int64_t inner_idx = i % inner_size;
    int64_t tmp = i / inner_size;
    int64_t dim_idx = tmp % half_dim;
    int64_t outer_idx = tmp / half_dim;
    
    // Read from original contiguous tensor
    int64_t base = outer_idx * full_dim_stride + inner_idx;
    float first = x[base + dim_idx * inner_size];
    float second = x[base + (dim_idx + half_dim) * inner_size];
    
    float gate = 1.0f / (1.0f + expf(-second));
    out[i] = first * gate;
}

Tensor glu(const Tensor& x, int dim) {
    DType orig = x.dtype();
    Tensor xf = ensure_f32(x);

    int ndim = xf.ndim();
    if (dim < 0) dim += ndim;
    assert(dim >= 0 && dim < ndim);
    assert(xf.size(dim) % 2 == 0);

    int64_t half = xf.size(dim) / 2;
    int64_t full_dim = xf.size(dim);

    // Compute output shape
    std::vector<int64_t> out_shape = xf.shape();
    out_shape[dim] = half;

    // Compute inner_size = product of dims after dim
    int64_t inner_size = 1;
    for (int d = dim + 1; d < ndim; d++) inner_size *= xf.size(d);

    int64_t N = 1;
    for (auto s : out_shape) N *= s;

    Tensor out = Tensor::empty(out_shape, DType::Float32);

    int grid = (int)ceildiv(N, (int64_t)kBlockSize);
    glu_fused_kernel<<<grid, kBlockSize>>>(
        xf.data_f32(), out.data_f32(),
        half, inner_size, full_dim * inner_size, N);
    CUDA_CHECK(cudaGetLastError());

    return maybe_cast_back(out, orig);
}

// ---------------------------------------------------------------------------
// l2_normalize:  x / ||x||_2  along last dim  (using warp-level reduction)
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void l2_normalize_kernel(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    int64_t outer, int inner_dim) {
    // One block per row.
    int row = blockIdx.x;
    if (row >= outer) return;

    const float* row_in = in + (int64_t)row * inner_dim;
    float* row_out = out + (int64_t)row * inner_dim;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    extern __shared__ float smem[];

    // Compute partial L2 squared sum.
    float partial = 0.0f;
    for (int j = tid; j < inner_dim; j += blockDim.x) {
        float v = row_in[j];
        partial += v * v;
    }

    // Warp reduce
    partial = warp_reduce_sum(partial);
    if (lane_id == 0) smem[warp_id] = partial;
    __syncthreads();

    // Final reduce across warps (first warp only).
    int num_warps = (blockDim.x + 31) / 32;
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) smem[0] = val;
    }
    __syncthreads();

    float inv_norm = rsqrtf(smem[0] + 1e-12f);

    for (int j = tid; j < inner_dim; j += blockDim.x) {
        row_out[j] = row_in[j] * inv_norm;
    }
}

Tensor l2_normalize(const Tensor& x, int dim) {
    DType orig = x.dtype();
    Tensor xf = ensure_f32(x);

    int ndim = xf.ndim();
    if (dim < 0) dim += ndim;
    assert(dim >= 0 && dim < ndim);

    // Move target dim to last position.
    Tensor inp = xf;
    bool permuted = false;
    std::vector<int> perm_fwd, perm_inv;
    if (dim != ndim - 1) {
        permuted = true;
        perm_fwd.resize(ndim);
        perm_inv.resize(ndim);
        int idx = 0;
        for (int i = 0; i < ndim; ++i) {
            if (i == dim) continue;
            perm_fwd[idx++] = i;
        }
        perm_fwd[ndim - 1] = dim;
        for (int i = 0; i < ndim; ++i) perm_inv[perm_fwd[i]] = i;
        inp = xf.permute(perm_fwd).contiguous();
    }

    int64_t inner_dim = inp.size(ndim - 1);
    int64_t outer = inp.numel() / inner_dim;

    Tensor out = Tensor::empty(inp.shape(), DType::Float32);

    int threads = 256;
    if (inner_dim < threads) {
        threads = 32; // at least one warp
        while (threads < inner_dim) threads <<= 1;
        if (threads > 1024) threads = 1024;
    }
    int num_warps = (threads + 31) / 32;
    size_t smem_bytes = num_warps * sizeof(float);

    l2_normalize_kernel<<<(int)outer, threads, smem_bytes>>>(
        inp.data_f32(), out.data_f32(), outer, (int)inner_dim);
    CUDA_CHECK(cudaGetLastError());

    if (permuted) {
        out = out.permute(perm_inv).contiguous();
    }

    return maybe_cast_back(out, orig);
}

// ---------------------------------------------------------------------------
// index_fill:  fill slice x[..., index, ...] along dim with value
// ---------------------------------------------------------------------------

__global__ void index_fill_kernel(float* __restrict__ data,
                                  int64_t dim_size, int64_t inner_size,
                                  int64_t target_idx, float value,
                                  int64_t total_outer) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_outer * inner_size) return;

    int64_t outer_idx = i / inner_size;
    int64_t in_idx    = i % inner_size;

    // Compute linear index: outer * dim_size * inner + target * inner + in
    int64_t linear = outer_idx * dim_size * inner_size + target_idx * inner_size + in_idx;
    data[linear] = value;
}

Tensor index_fill(const Tensor& x, int dim, int64_t index, float value) {
    DType orig = x.dtype();
    Tensor out = ensure_f32(x).clone(); // work on a copy

    int ndim = out.ndim();
    if (dim < 0) dim += ndim;
    assert(dim >= 0 && dim < ndim);
    int64_t dim_size = out.size(dim);
    assert(index >= 0 && index < dim_size);

    // outer = product of dims before dim
    int64_t outer = 1;
    for (int i = 0; i < dim; ++i) outer *= out.size(i);
    // inner = product of dims after dim
    int64_t inner = 1;
    for (int i = dim + 1; i < ndim; ++i) inner *= out.size(i);

    int64_t total = outer * inner;
    int grid = (int)ceildiv(total, (int64_t)kBlockSize);
    index_fill_kernel<<<grid, kBlockSize>>>(
        out.data_f32(), dim_size, inner, index, value, outer);
    CUDA_CHECK(cudaGetLastError());

    return maybe_cast_back(out, orig);
}

// ---------------------------------------------------------------------------
// dropout:  inference mode → identity (just clone)
// ---------------------------------------------------------------------------

Tensor dropout(const Tensor& x, float /*p*/) {
    return x;  // inference mode — identity
}

} // namespace ops
} // namespace cudasep
