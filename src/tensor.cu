#include "tensor.h"
#include "memory_pool.h"
#include <algorithm>
#include <sstream>
#include <cmath>

namespace cudasep {

// ============================================================================
// CUDA Kernels
// ============================================================================

static constexpr int BLOCK_SIZE = 256;

inline int grid_size(int64_t N) {
    return (int)((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

// --- Fill kernels ---
__global__ void fill_f32_kernel(float* dst, float val, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) dst[idx] = val;
}

__global__ void fill_f16_kernel(__half* dst, __half val, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) dst[idx] = val;
}

__global__ void fill_i64_kernel(int64_t* dst, int64_t val, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) dst[idx] = val;
}

// --- Arange kernel ---
__global__ void arange_f32_kernel(float* dst, int64_t start, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) dst[idx] = (float)(start + idx);
}

__global__ void arange_f16_kernel(__half* dst, int64_t start, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) dst[idx] = __float2half((float)(start + idx));
}

__global__ void arange_i64_kernel(int64_t* dst, int64_t start, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) dst[idx] = start + idx;
}

// --- Type conversion kernels ---
__global__ void f32_to_f16_kernel(const float* src, __half* dst, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) dst[idx] = __float2half(src[idx]);
}

__global__ void f16_to_f32_kernel(const __half* src, float* dst, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) dst[idx] = __half2float(src[idx]);
}

__global__ void i64_to_f32_kernel(const int64_t* src, float* dst, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) dst[idx] = (float)src[idx];
}

__global__ void f32_to_i64_kernel(const float* src, int64_t* dst, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) dst[idx] = (int64_t)src[idx];
}

__global__ void i64_to_f16_kernel(const int64_t* src, __half* dst, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) dst[idx] = __float2half((float)src[idx]);
}

__global__ void f16_to_i64_kernel(const __half* src, int64_t* dst, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) dst[idx] = (int64_t)__half2float(src[idx]);
}

// --- Scalar ops kernels ---
__global__ void add_scalar_f32_kernel(float* data, float val, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] += val;
}

__global__ void mul_scalar_f32_kernel(float* data, float val, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] *= val;
}

__global__ void add_scalar_f16_kernel(__half* data, __half val, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] = __hadd(data[idx], val);
}

__global__ void mul_scalar_f16_kernel(__half* data, __half val, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] = __hmul(data[idx], val);
}

__global__ void clamp_f32_kernel(float* data, float min_val, float max_val, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float v = data[idx];
        v = fmaxf(v, min_val);
        v = fminf(v, max_val);
        data[idx] = v;
    }
}

__global__ void clamp_f16_kernel(__half* data, __half min_val, __half max_val, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float v = __half2float(data[idx]);
        v = fmaxf(v, __half2float(min_val));
        v = fminf(v, __half2float(max_val));
        data[idx] = __float2half(v);
    }
}

__global__ void negate_f32_kernel(const float* src, float* dst, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) dst[idx] = -src[idx];
}

__global__ void negate_f16_kernel(const __half* src, __half* dst, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) dst[idx] = __hneg(src[idx]);
}

// --- Comparison kernel ---
__global__ void gt_scalar_f32_kernel(const float* src, float val, float* dst, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) dst[idx] = (src[idx] > val) ? 1.0f : 0.0f;
}

__global__ void gt_scalar_f16_kernel(const __half* src, __half val, __half* dst, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dst[idx] = (__half2float(src[idx]) > __half2float(val))
                       ? __float2half(1.0f)
                       : __float2half(0.0f);
    }
}

// --- Broadcasting element-wise binary kernels ---
//
// We pass shape and strides for both operands (with broadcast strides = 0
// for dims of size 1) plus the output shape. Max 8 dimensions supported.

static constexpr int MAX_DIMS = 8;

// Metadata for strided copy (permute/contiguous) - passed by value to avoid cudaMalloc
struct StridedCopyMeta {
    int64_t src_strides[MAX_DIMS];
    int64_t dst_shape[MAX_DIMS];
    int ndim;
};

struct BroadcastMeta {
    int64_t out_shape[MAX_DIMS];
    int64_t a_strides[MAX_DIMS];
    int64_t b_strides[MAX_DIMS];
    int ndim;
};

__device__ inline int64_t broadcast_offset(int64_t linear_idx,
                                            const int64_t* out_shape,
                                            const int64_t* src_strides,
                                            int ndim) {
    int64_t offset = 0;
    for (int d = ndim - 1; d >= 0; --d) {
        int64_t coord = linear_idx % out_shape[d];
        linear_idx /= out_shape[d];
        offset += coord * src_strides[d];
    }
    return offset;
}

enum class BinaryOp { Add, Sub, Mul, Div };

template <BinaryOp op>
__device__ inline float binary_apply_f32(float a, float b) {
    if constexpr (op == BinaryOp::Add) return a + b;
    if constexpr (op == BinaryOp::Sub) return a - b;
    if constexpr (op == BinaryOp::Mul) return a * b;
    if constexpr (op == BinaryOp::Div) return a / b;
    return 0.0f;
}

template <BinaryOp op>
__device__ inline __half binary_apply_f16(__half a, __half b) {
    if constexpr (op == BinaryOp::Add) return __hadd(a, b);
    if constexpr (op == BinaryOp::Sub) return __hsub(a, b);
    if constexpr (op == BinaryOp::Mul) return __hmul(a, b);
    if constexpr (op == BinaryOp::Div) return __hdiv(a, b);
    return __float2half(0.0f);
}

template <BinaryOp op>
__global__ void elementwise_binary_f32_kernel(const float* a, const float* b, float* out,
                                               BroadcastMeta meta, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int64_t a_off = broadcast_offset(idx, meta.out_shape, meta.a_strides, meta.ndim);
        int64_t b_off = broadcast_offset(idx, meta.out_shape, meta.b_strides, meta.ndim);
        out[idx] = binary_apply_f32<op>(a[a_off], b[b_off]);
    }
}

template <BinaryOp op>
__global__ void elementwise_binary_f16_kernel(const __half* a, const __half* b, __half* out,
                                               BroadcastMeta meta, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int64_t a_off = broadcast_offset(idx, meta.out_shape, meta.a_strides, meta.ndim);
        int64_t b_off = broadcast_offset(idx, meta.out_shape, meta.b_strides, meta.ndim);
        out[idx] = binary_apply_f16<op>(a[a_off], b[b_off]);
    }
}

// In-place binary (this += or *= other with broadcasting)
template <BinaryOp op>
__global__ void elementwise_binary_inplace_f32_kernel(float* a, const float* b,
                                                       BroadcastMeta meta, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int64_t b_off = broadcast_offset(idx, meta.out_shape, meta.b_strides, meta.ndim);
        a[idx] = binary_apply_f32<op>(a[idx], b[b_off]);
    }
}

template <BinaryOp op>
__global__ void elementwise_binary_inplace_f16_kernel(__half* a, const __half* b,
                                                       BroadcastMeta meta, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int64_t b_off = broadcast_offset(idx, meta.out_shape, meta.b_strides, meta.ndim);
        a[idx] = binary_apply_f16<op>(a[idx], b[b_off]);
    }
}

// --- Permute kernel ---
__global__ void permute_kernel(const char* src, char* dst,
                               const int64_t* src_strides,
                               const int64_t* dst_shape,
                               const int* perm,
                               int ndim, int elem_size, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Compute coordinates in destination tensor
    int64_t tmp = idx;
    int64_t src_offset = 0;
    for (int d = ndim - 1; d >= 0; --d) {
        int64_t coord = tmp % dst_shape[d];
        tmp /= dst_shape[d];
        // This coordinate in dst dimension d corresponds to src dimension perm[d]
        src_offset += coord * src_strides[perm[d]];
    }
    // Copy element
    const char* sp = src + src_offset * elem_size;
    char* dp = dst + idx * elem_size;
    for (int i = 0; i < elem_size; ++i) {
        dp[i] = sp[i];
    }
}

// --- Slice kernel ---
__global__ void slice_kernel(const char* src, char* dst,
                             const int64_t* src_shape,
                             const int64_t* src_strides,
                             const int64_t* dst_shape,
                             int slice_dim, int64_t slice_start,
                             int ndim, int elem_size, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Compute dst coordinates
    int64_t tmp = idx;
    int64_t src_offset = 0;
    for (int d = ndim - 1; d >= 0; --d) {
        int64_t coord = tmp % dst_shape[d];
        tmp /= dst_shape[d];
        int64_t src_coord = (d == slice_dim) ? (coord + slice_start) : coord;
        src_offset += src_coord * src_strides[d];
    }
    const char* sp = src + src_offset * elem_size;
    char* dp = dst + idx * elem_size;
    for (int i = 0; i < elem_size; ++i) {
        dp[i] = sp[i];
    }
}

// --- Index select kernel ---
__global__ void index_select_kernel(const char* src, char* dst,
                                     const int64_t* indices,
                                     StridedCopyMeta meta,
                                     int select_dim,
                                     int elem_size, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int64_t tmp = idx;
    int64_t src_offset = 0;
    for (int d = meta.ndim - 1; d >= 0; --d) {
        int64_t coord = tmp % meta.dst_shape[d];
        tmp /= meta.dst_shape[d];
        int64_t src_coord = (d == select_dim) ? indices[coord] : coord;
        src_offset += src_coord * meta.src_strides[d];
    }
    const char* sp = src + src_offset * elem_size;
    char* dp = dst + idx * elem_size;
    for (int i = 0; i < elem_size; ++i) {
        dp[i] = sp[i];
    }
}

// Metadata for padding kernels
struct PadMeta {
    int64_t src_shape[MAX_DIMS];
    int64_t dst_shape[MAX_DIMS];
    int64_t pad_before[MAX_DIMS];
    int ndim;
};

// --- Pad constant kernel ---
__global__ void pad_const_kernel(const char* src, char* dst,
                                  PadMeta meta,
                                  int elem_size,
                                  float pad_val, int dtype_code,
                                  int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Compute dst coordinates
    int64_t tmp = idx;
    int64_t coords[MAX_DIMS];
    for (int d = meta.ndim - 1; d >= 0; --d) {
        coords[d] = tmp % meta.dst_shape[d];
        tmp /= meta.dst_shape[d];
    }

    // Check if inside source region
    bool inside = true;
    int64_t src_linear = 0;
    int64_t src_stride = 1;
    for (int d = meta.ndim - 1; d >= 0; --d) {
        int64_t src_coord = coords[d] - meta.pad_before[d];
        if (src_coord < 0 || src_coord >= meta.src_shape[d]) {
            inside = false;
            break;
        }
        src_linear += src_coord * src_stride;
        src_stride *= meta.src_shape[d];
    }

    char* dp = dst + idx * elem_size;
    if (inside) {
        const char* sp = src + src_linear * elem_size;
        for (int i = 0; i < elem_size; ++i) dp[i] = sp[i];
    } else {
        // Fill with pad value
        if (dtype_code == 0) { // Float32
            *((float*)dp) = pad_val;
        } else if (dtype_code == 1) { // Float16
            *((__half*)dp) = __float2half(pad_val);
        } else { // Int64
            *((int64_t*)dp) = (int64_t)pad_val;
        }
    }
}

// --- Pad reflect kernel ---
__global__ void pad_reflect_kernel(const char* src, char* dst,
                                    PadMeta meta,
                                    int elem_size, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int64_t tmp = idx;
    int64_t coords[MAX_DIMS];
    for (int d = meta.ndim - 1; d >= 0; --d) {
        coords[d] = tmp % meta.dst_shape[d];
        tmp /= meta.dst_shape[d];
    }

    // Map each coordinate to source via reflection
    int64_t src_linear = 0;
    int64_t src_stride = 1;
    for (int d = meta.ndim - 1; d >= 0; --d) {
        int64_t src_coord = coords[d] - meta.pad_before[d];
        // Reflect
        if (src_coord < 0) {
            src_coord = -src_coord;
        } else if (src_coord >= meta.src_shape[d]) {
            src_coord = 2 * (meta.src_shape[d] - 1) - src_coord;
        }
        // Clamp (safety)
        if (src_coord < 0) src_coord = 0;
        if (src_coord >= meta.src_shape[d]) src_coord = meta.src_shape[d] - 1;
        src_linear += src_coord * src_stride;
        src_stride *= meta.src_shape[d];
    }

    const char* sp = src + src_linear * elem_size;
    char* dp = dst + idx * elem_size;
    for (int i = 0; i < elem_size; ++i) dp[i] = sp[i];
}

// --- Expand (broadcast copy) kernel ---
__global__ void expand_kernel(const char* src, char* dst,
                               const int64_t* src_shape,
                               const int64_t* dst_shape,
                               const int64_t* src_strides,
                               int ndim, int elem_size, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int64_t tmp = idx;
    int64_t src_offset = 0;
    for (int d = ndim - 1; d >= 0; --d) {
        int64_t coord = tmp % dst_shape[d];
        tmp /= dst_shape[d];
        // If src dim is 1, collapse coordinate to 0
        if (src_shape[d] == 1) {
            // stride is 0 effectively
        } else {
            src_offset += coord * src_strides[d];
        }
    }
    const char* sp = src + src_offset * elem_size;
    char* dp = dst + idx * elem_size;
    for (int i = 0; i < elem_size; ++i) dp[i] = sp[i];
}

// --- Reduction kernels ---
// Reduce along the innermost dimension using warp shuffle
__global__ void sum_reduce_last_dim_f32(const float* src, float* dst,
                                         int64_t inner_size,
                                         int64_t outer_size) {
    int64_t outer_idx = (int64_t)blockIdx.x;
    if (outer_idx >= outer_size) return;

    const float* row = src + outer_idx * inner_size;
    float val = 0.0f;
    for (int64_t i = threadIdx.x; i < inner_size; i += blockDim.x) {
        val += row[i];
    }
    // Warp reduce
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    // Use shared memory to reduce across warps
    __shared__ float shared[32]; // max 32 warps per block
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) shared[warp_id] = val;
    __syncthreads();
    int nwarps = (blockDim.x + 31) / 32;
    if (threadIdx.x < (unsigned)nwarps) {
        val = shared[threadIdx.x];
    } else {
        val = 0.0f;
    }
    if (threadIdx.x < 32) {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
    }
    if (threadIdx.x == 0) {
        dst[outer_idx] = val;
    }
}

__global__ void max_reduce_last_dim_f32(const float* src, float* dst,
                                         int64_t inner_size,
                                         int64_t outer_size) {
    int64_t outer_idx = (int64_t)blockIdx.x;
    if (outer_idx >= outer_size) return;

    const float* row = src + outer_idx * inner_size;
    float val = -INFINITY;
    for (int64_t i = threadIdx.x; i < inner_size; i += blockDim.x) {
        val = fmaxf(val, row[i]);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        float tmp = __shfl_down_sync(0xFFFFFFFF, val, offset);
        val = fmaxf(val, tmp);
    }
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) shared[warp_id] = val;
    __syncthreads();
    int nwarps = (blockDim.x + 31) / 32;
    if (threadIdx.x < (unsigned)nwarps) {
        val = shared[threadIdx.x];
    } else {
        val = -INFINITY;
    }
    if (threadIdx.x < 32) {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            float tmp = __shfl_down_sync(0xFFFFFFFF, val, offset);
            val = fmaxf(val, tmp);
        }
    }
    if (threadIdx.x == 0) {
        dst[outer_idx] = val;
    }
}

// General reduction kernel: reduce dimension `reduce_dim` which may not be the last.
// Reshape to [outer, reduce, inner] and reduce the middle dim.
__global__ void sum_reduce_general_f32(const float* src, float* dst,
                                        int64_t outer, int64_t reduce, int64_t inner,
                                        int64_t N_out) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_out) return;

    int64_t outer_idx = idx / inner;
    int64_t inner_idx = idx % inner;

    float val = 0.0f;
    for (int64_t r = 0; r < reduce; ++r) {
        val += src[outer_idx * reduce * inner + r * inner + inner_idx];
    }
    dst[idx] = val;
}

__global__ void max_reduce_general_f32(const float* src, float* dst,
                                        int64_t outer, int64_t reduce, int64_t inner,
                                        int64_t N_out) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_out) return;

    int64_t outer_idx = idx / inner;
    int64_t inner_idx = idx % inner;

    float val = -INFINITY;
    for (int64_t r = 0; r < reduce; ++r) {
        float v = src[outer_idx * reduce * inner + r * inner + inner_idx];
        val = fmaxf(val, v);
    }
    dst[idx] = val;
}

// --- Copy kernel (generic, for non-contiguous copies) using embedded metadata ---
__global__ void strided_copy_kernel(const char* src, char* dst,
                                     StridedCopyMeta meta,
                                     int elem_size, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int64_t tmp = idx;
    int64_t src_offset = 0;
    for (int d = meta.ndim - 1; d >= 0; --d) {
        int64_t coord = tmp % meta.dst_shape[d];
        tmp /= meta.dst_shape[d];
        src_offset += coord * meta.src_strides[d];
    }
    const char* sp = src + src_offset * elem_size;
    char* dp = dst + idx * elem_size;
    for (int i = 0; i < elem_size; ++i) dp[i] = sp[i];
}

// Specialized f32 strided copy — uses float load/store instead of byte memcpy
__global__ void strided_copy_f32_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                         StridedCopyMeta meta, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int64_t tmp = idx;
    int64_t src_offset = 0;
    for (int d = meta.ndim - 1; d >= 0; --d) {
        int64_t coord = tmp % meta.dst_shape[d];
        tmp /= meta.dst_shape[d];
        src_offset += coord * meta.src_strides[d];
    }
    dst[idx] = src[src_offset];
}

// Specialized f16 strided copy
__global__ void strided_copy_f16_kernel(const __half* __restrict__ src, __half* __restrict__ dst,
                                         StridedCopyMeta meta, int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int64_t tmp = idx;
    int64_t src_offset = 0;
    for (int d = meta.ndim - 1; d >= 0; --d) {
        int64_t coord = tmp % meta.dst_shape[d];
        tmp /= meta.dst_shape[d];
        src_offset += coord * meta.src_strides[d];
    }
    dst[idx] = src[src_offset];
}

// ============================================================================
// Helper: allocate device arrays and copy host data to them
// ============================================================================

struct DeviceArrays {
    int64_t* d_ptr;
    DeviceArrays(const int64_t* host, int count) {
        CUDA_CHECK(cudaMalloc(&d_ptr, count * sizeof(int64_t)));
        CUDA_CHECK(cudaMemcpy(d_ptr, host, count * sizeof(int64_t), cudaMemcpyHostToDevice));
    }
    ~DeviceArrays() { cudaFree(d_ptr); }
};

struct DeviceIntArray {
    int* d_ptr;
    DeviceIntArray(const int* host, int count) {
        CUDA_CHECK(cudaMalloc(&d_ptr, count * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_ptr, host, count * sizeof(int), cudaMemcpyHostToDevice));
    }
    ~DeviceIntArray() { cudaFree(d_ptr); }
};

// ============================================================================
// Tensor Implementation
// ============================================================================

Tensor::Tensor()
    : storage_(nullptr), data_(nullptr), numel_(0), dtype_(DType::Float32) {}

std::shared_ptr<void> Tensor::alloc_gpu(size_t bytes) {
    if (bytes == 0) return nullptr;
    void* ptr = CudaMemoryPool::instance().allocate(bytes);
    return std::shared_ptr<void>(ptr, [](void* p) {
        CudaMemoryPool::instance().deallocate(p);
    });
}

void Tensor::compute_strides() {
    strides_.resize(shape_.size());
    if (shape_.empty()) return;
    strides_.back() = 1;
    for (int i = (int)shape_.size() - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

// --- size ---
int64_t Tensor::size(int dim) const {
    if (dim < 0) dim += ndim();
    if (dim < 0 || dim >= ndim())
        throw std::out_of_range("Tensor::size: dim out of range");
    return shape_[dim];
}

// --- is_contiguous ---
bool Tensor::is_contiguous() const {
    if (numel_ <= 1) return true;
    int64_t expected = 1;
    for (int i = (int)shape_.size() - 1; i >= 0; --i) {
        if (shape_[i] != 1 && strides_[i] != expected) return false;
        expected *= shape_[i];
    }
    return true;
}

// --- Factory: empty ---
Tensor Tensor::empty(std::vector<int64_t> shape, DType dtype) {
    Tensor t;
    t.shape_ = std::move(shape);
    t.dtype_ = dtype;
    t.numel_ = 1;
    for (auto s : t.shape_) t.numel_ *= s;
    t.compute_strides();
    size_t bytes = t.numel_ * dtype_size(dtype);
    t.storage_ = alloc_gpu(bytes);
    t.data_ = t.storage_.get();
    return t;
}

// --- Factory: zeros ---
Tensor Tensor::zeros(std::vector<int64_t> shape, DType dtype) {
    Tensor t = empty(std::move(shape), dtype);
    if (t.numel_ > 0) {
        CUDA_CHECK(cudaMemset(t.data_, 0, t.numel_ * dtype_size(dtype)));
    }
    return t;
}

// --- Factory: ones ---
Tensor Tensor::ones(std::vector<int64_t> shape, DType dtype) {
    return full(std::move(shape), 1.0f, dtype);
}

// --- Factory: full ---
Tensor Tensor::full(std::vector<int64_t> shape, float value, DType dtype) {
    Tensor t = empty(std::move(shape), dtype);
    if (t.numel_ == 0) return t;
    int64_t N = t.numel_;
    int grid = grid_size(N);
    switch (dtype) {
        case DType::Float32:
            fill_f32_kernel<<<grid, BLOCK_SIZE>>>(t.data_f32(), value, N);
            break;
        case DType::Float16:
            fill_f16_kernel<<<grid, BLOCK_SIZE>>>(t.data_f16(), __float2half(value), N);
            break;
        case DType::Int64:
            fill_i64_kernel<<<grid, BLOCK_SIZE>>>(t.data_i64(), (int64_t)value, N);
            break;
    }
    CUDA_CHECK(cudaGetLastError());
    return t;
}

// --- Factory: from_cpu_f32 ---
Tensor Tensor::from_cpu_f32(const float* data, std::vector<int64_t> shape) {
    Tensor t = empty(shape, DType::Float32);
    if (t.numel_ > 0) {
        CUDA_CHECK(cudaMemcpy(t.data_, data, t.numel_ * sizeof(float), cudaMemcpyHostToDevice));
    }
    return t;
}

// --- Factory: from_cpu_f16 ---
Tensor Tensor::from_cpu_f16(const void* data, std::vector<int64_t> shape) {
    Tensor t = empty(shape, DType::Float16);
    if (t.numel_ > 0) {
        CUDA_CHECK(cudaMemcpy(t.data_, data, t.numel_ * sizeof(__half), cudaMemcpyHostToDevice));
    }
    return t;
}

// --- Factory: from_cpu_i64 ---
Tensor Tensor::from_cpu_i64(const int64_t* data, std::vector<int64_t> shape) {
    Tensor t = empty(shape, DType::Int64);
    if (t.numel_ > 0) {
        CUDA_CHECK(cudaMemcpy(t.data_, data, t.numel_ * sizeof(int64_t), cudaMemcpyHostToDevice));
    }
    return t;
}

// --- Factory: arange ---
Tensor Tensor::arange(int64_t start, int64_t end, DType dtype) {
    int64_t N = end - start;
    if (N <= 0) return Tensor();
    Tensor t = empty({N}, dtype);
    int grid = grid_size(N);
    switch (dtype) {
        case DType::Float32:
            arange_f32_kernel<<<grid, BLOCK_SIZE>>>(t.data_f32(), start, N);
            break;
        case DType::Float16:
            arange_f16_kernel<<<grid, BLOCK_SIZE>>>(t.data_f16(), start, N);
            break;
        case DType::Int64:
            arange_i64_kernel<<<grid, BLOCK_SIZE>>>(t.data_i64(), start, N);
            break;
    }
    CUDA_CHECK(cudaGetLastError());
    return t;
}

// --- to_cpu_f32 ---
std::vector<float> Tensor::to_cpu_f32() const {
    if (dtype_ == DType::Float32) {
        Tensor c = is_contiguous() ? *this : contiguous();
        std::vector<float> result(numel_);
        CUDA_CHECK(cudaMemcpy(result.data(), c.data_, numel_ * sizeof(float), cudaMemcpyDeviceToHost));
        return result;
    } else if (dtype_ == DType::Float16) {
        Tensor f32 = to_f32();
        return f32.to_cpu_f32();
    } else {
        // Int64 -> float
        Tensor f32 = to_dtype(DType::Float32);
        return f32.to_cpu_f32();
    }
}

// --- copy_from_cpu ---
void Tensor::copy_from_cpu(const void* src, size_t bytes) {
    CUDA_CHECK(cudaMemcpy(data_, src, bytes, cudaMemcpyHostToDevice));
}

// --- copy_to_cpu ---
void Tensor::copy_to_cpu(void* dst, size_t bytes) const {
    CUDA_CHECK(cudaMemcpy(dst, data_, bytes, cudaMemcpyDeviceToHost));
}

// --- reshape ---
Tensor Tensor::reshape(std::vector<int64_t> new_shape) const {
    // Resolve -1
    int neg_idx = -1;
    int64_t prod = 1;
    for (int i = 0; i < (int)new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            if (neg_idx >= 0) throw std::runtime_error("reshape: only one -1 allowed");
            neg_idx = i;
        } else {
            prod *= new_shape[i];
        }
    }
    if (neg_idx >= 0) {
        new_shape[neg_idx] = numel_ / prod;
    }
    // Verify
    int64_t new_numel = 1;
    for (auto s : new_shape) new_numel *= s;
    if (new_numel != numel_)
        throw std::runtime_error("reshape: incompatible sizes " +
                                 std::to_string(numel_) + " vs " + std::to_string(new_numel));

    if (!is_contiguous()) {
        // Must copy first
        Tensor c = contiguous();
        Tensor t;
        t.storage_ = c.storage_;
        t.data_ = c.data_;
        t.shape_ = std::move(new_shape);
        t.numel_ = numel_;
        t.dtype_ = dtype_;
        t.compute_strides();
        return t;
    }

    Tensor t;
    t.storage_ = storage_;
    t.data_ = data_;
    t.shape_ = std::move(new_shape);
    t.numel_ = numel_;
    t.dtype_ = dtype_;
    t.compute_strides();
    return t;
}

// --- permute (lazy - zero-copy stride reordering) ---
Tensor Tensor::permute(std::vector<int> dims) const {
    int nd = ndim();
    if ((int)dims.size() != nd)
        throw std::runtime_error("permute: dims size mismatch");

    Tensor out;
    out.storage_ = storage_;
    out.data_ = data_;
    out.dtype_ = dtype_;
    out.numel_ = numel_;
    out.shape_.resize(nd);
    out.strides_.resize(nd);
    for (int i = 0; i < nd; ++i) {
        out.shape_[i] = shape_[dims[i]];
        out.strides_[i] = strides_[dims[i]];
    }
    return out;
}

// --- transpose ---
Tensor Tensor::transpose(int dim0, int dim1) const {
    int nd = ndim();
    if (dim0 < 0) dim0 += nd;
    if (dim1 < 0) dim1 += nd;
    std::vector<int> dims(nd);
    std::iota(dims.begin(), dims.end(), 0);
    std::swap(dims[dim0], dims[dim1]);
    return permute(dims);
}

// --- contiguous ---
Tensor Tensor::contiguous() const {
    if (is_contiguous()) {
        return *this;  // Already contiguous - return shared view (no copy)
    }
    // Need to copy with strides
    int nd = ndim();
    Tensor out = Tensor::empty(shape_, dtype_);
    if (numel_ == 0) return out;

    int elem_size = (int)dtype_size(dtype_);
    StridedCopyMeta meta;
    meta.ndim = nd;
    for (int i = 0; i < nd; ++i) {
        meta.src_strides[i] = strides_[i];
        meta.dst_shape[i] = shape_[i];
    }

    int grid = grid_size(numel_);
    if (dtype_ == DType::Float32) {
        strided_copy_f32_kernel<<<grid, BLOCK_SIZE>>>(
            (const float*)data_, (float*)out.data_,
            meta, numel_);
    } else if (dtype_ == DType::Float16) {
        strided_copy_f16_kernel<<<grid, BLOCK_SIZE>>>(
            (const __half*)data_, (__half*)out.data_,
            meta, numel_);
    } else {
        int elem_size = (int)dtype_size(dtype_);
        strided_copy_kernel<<<grid, BLOCK_SIZE>>>(
            (const char*)data_, (char*)out.data_,
            meta, elem_size, numel_);
    }
    CUDA_CHECK(cudaGetLastError());

    return out;
}

// --- unsqueeze ---
Tensor Tensor::unsqueeze(int dim) const {
    if (dim < 0) dim += ndim() + 1;
    std::vector<int64_t> new_shape = shape_;
    new_shape.insert(new_shape.begin() + dim, 1);
    return reshape(new_shape);
}

// --- squeeze ---
Tensor Tensor::squeeze(int dim) const {
    if (dim < 0) dim += ndim();
    if (shape_[dim] != 1)
        throw std::runtime_error("squeeze: dimension is not 1");
    std::vector<int64_t> new_shape = shape_;
    new_shape.erase(new_shape.begin() + dim);
    return reshape(new_shape);
}

// --- expand (lazy view - zero-copy with broadcast strides) ---
Tensor Tensor::expand(std::vector<int64_t> new_shape) const {
    int nd = (int)new_shape.size();
    if (nd < ndim()) throw std::runtime_error("expand: cannot reduce dims");

    // Pad shape/strides on the left with 1s if needed
    std::vector<int64_t> src_shape = shape_;
    std::vector<int64_t> src_strides = strides_;
    while ((int)src_shape.size() < nd) {
        src_shape.insert(src_shape.begin(), 1);
        src_strides.insert(src_strides.begin(), 0);
    }

    // Validate and compute broadcast strides
    std::vector<int64_t> bcast_strides(nd);
    for (int d = 0; d < nd; ++d) {
        if (src_shape[d] == new_shape[d]) {
            bcast_strides[d] = src_strides[d];
        } else if (src_shape[d] == 1) {
            bcast_strides[d] = 0; // broadcast
        } else {
            throw std::runtime_error("expand: incompatible shape at dim " + std::to_string(d));
        }
    }

    Tensor out;
    out.storage_ = storage_;  // share ownership
    out.data_ = data_;
    out.dtype_ = dtype_;
    out.shape_ = std::move(new_shape);
    out.strides_ = std::move(bcast_strides);
    out.numel_ = 1;
    for (auto s : out.shape_) out.numel_ *= s;
    return out;
}

// --- slice (lazy view - zero-copy) ---
Tensor Tensor::slice(int dim, int64_t start, int64_t end) const {
    if (dim < 0) dim += ndim();
    if (start < 0) start += shape_[dim];
    if (end < 0) end += shape_[dim];
    if (start < 0) start = 0;
    if (end > shape_[dim]) end = shape_[dim];
    if (start >= end) return Tensor();

    Tensor out;
    out.storage_ = storage_;  // share ownership
    out.dtype_ = dtype_;
    out.shape_ = shape_;
    out.shape_[dim] = end - start;
    out.strides_ = strides_;
    out.numel_ = 1;
    for (auto s : out.shape_) out.numel_ *= s;
    // Offset data pointer by start * stride along the sliced dimension
    out.data_ = (char*)data_ + start * strides_[dim] * (int64_t)dtype_size(dtype_);
    return out;
}

// --- index_select ---
Tensor Tensor::index_select(int dim, const Tensor& indices) const {
    if (dim < 0) dim += ndim();
    if (indices.dtype() != DType::Int64)
        throw std::runtime_error("index_select: indices must be Int64");

    int64_t n_idx = indices.numel();
    std::vector<int64_t> out_shape = shape_;
    out_shape[dim] = n_idx;

    int64_t out_numel = 1;
    for (auto s : out_shape) out_numel *= s;
    Tensor out = Tensor::empty(out_shape, dtype_);
    if (out_numel == 0) return out;

    int nd = ndim();
    int elem_size = (int)dtype_size(dtype_);

    StridedCopyMeta meta;
    meta.ndim = nd;
    for (int i = 0; i < nd; ++i) {
        meta.src_strides[i] = strides_[i];
        meta.dst_shape[i] = out_shape[i];
    }

    int grid_sz = grid_size(out_numel);
    index_select_kernel<<<grid_sz, BLOCK_SIZE>>>(
        (const char*)data_, (char*)out.data_,
        indices.data_i64(), meta,
        dim, elem_size, out_numel);
    CUDA_CHECK(cudaGetLastError());

    return out;
}

// --- cat kernel ---
struct CatSrcInfo {
    const void* ptr;
    int64_t dim_size;
    int64_t dim_offset;
};

__global__ void cat_2d_kernel(
    char* __restrict__ out,
    const CatSrcInfo* __restrict__ infos,
    int64_t outer_size,
    int64_t total_cat_dim,
    int64_t inner_bytes,   // inner_size * elem_size
    int64_t src_elems_max  // max elements per tensor (for grid sizing)
) {
    int tensor_idx = blockIdx.y;
    const CatSrcInfo& info = infos[tensor_idx];

    int64_t src_bytes = outer_size * info.dim_size * inner_bytes;
    int64_t bid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    // Process 4 bytes at a time for alignment
    int64_t idx4 = bid * 4;
    if (idx4 >= src_bytes) return;

    // Decompose byte index to (outer, local_d_byte, inner_byte)
    int64_t dim_bytes = info.dim_size * inner_bytes;
    int64_t outer_idx = idx4 / dim_bytes;
    int64_t rem = idx4 % dim_bytes;

    // Destination byte offset
    int64_t dst_idx = outer_idx * total_cat_dim * inner_bytes + info.dim_offset * inner_bytes + rem;

    // Copy 4 bytes (handles float32 and float16 generically)
    if (idx4 + 4 <= src_bytes) {
        *reinterpret_cast<uint32_t*>(out + dst_idx) =
            *reinterpret_cast<const uint32_t*>((const char*)info.ptr + idx4);
    } else {
        // Tail bytes
        for (int64_t b = idx4; b < src_bytes; b++) {
            out[dst_idx + (b - idx4)] = ((const char*)info.ptr)[b];
        }
    }
}

// --- cat ---
Tensor Tensor::cat(const std::vector<Tensor>& tensors, int dim) {
    if (tensors.empty())
        throw std::runtime_error("cat: empty tensor list");

    int nd = tensors[0].ndim();
    if (dim < 0) dim += nd;

    DType dt = tensors[0].dtype();
    std::vector<int64_t> out_shape = tensors[0].shape();

    // Sum the concat dim
    int64_t total_dim = 0;
    for (auto& t : tensors) {
        if (t.ndim() != nd)
            throw std::runtime_error("cat: ndim mismatch");
        if (t.dtype() != dt)
            throw std::runtime_error("cat: dtype mismatch");
        total_dim += t.size(dim);
    }
    out_shape[dim] = total_dim;

    Tensor out = Tensor::empty(out_shape, dt);
    int elem_size = (int)dtype_size(dt);

    // Compute: outer_size = product of dims before `dim`
    //          inner_size = product of dims after `dim`
    int64_t outer_size = 1;
    for (int d = 0; d < dim; ++d) outer_size *= out_shape[d];
    int64_t inner_size = 1;
    for (int d = dim + 1; d < nd; ++d) inner_size *= out_shape[d];

    int64_t inner_bytes = inner_size * elem_size;
    int N = (int)tensors.size();

    // Ensure all tensors are contiguous and build metadata
    std::vector<Tensor> contig(N);
    std::vector<CatSrcInfo> h_infos(N);
    int64_t dim_offset = 0;
    int64_t max_src_bytes = 0;
    for (int i = 0; i < N; i++) {
        contig[i] = tensors[i].contiguous();
        h_infos[i].ptr = contig[i].data_;
        h_infos[i].dim_size = contig[i].size(dim);
        h_infos[i].dim_offset = dim_offset;
        int64_t sb = outer_size * h_infos[i].dim_size * inner_bytes;
        if (sb > max_src_bytes) max_src_bytes = sb;
        dim_offset += h_infos[i].dim_size;
    }

    // Upload metadata to GPU (small allocation, reuse static buffer)
    static CatSrcInfo* d_infos = nullptr;
    static size_t d_infos_cap = 0;
    size_t infos_bytes = N * sizeof(CatSrcInfo);
    if (infos_bytes > d_infos_cap) {
        if (d_infos) cudaFree(d_infos);
        CUDA_CHECK(cudaMalloc(&d_infos, infos_bytes));
        d_infos_cap = infos_bytes;
    }
    CUDA_CHECK(cudaMemcpyAsync(d_infos, h_infos.data(), infos_bytes, cudaMemcpyHostToDevice));

    // Launch 2D kernel: (blocks_per_tensor, N)
    // Each thread processes 4 bytes
    int64_t max_units = (max_src_bytes + 3) / 4;  // 4-byte units
    int threads = 256;
    int blocks_x = (int)((max_units + threads - 1) / threads);
    dim3 grid(blocks_x, N);

    cat_2d_kernel<<<grid, threads>>>(
        (char*)out.data_, d_infos,
        outer_size, total_dim, inner_bytes, max_units);
    CUDA_CHECK(cudaGetLastError());

    return out;
}

// --- stack ---
Tensor Tensor::stack(const std::vector<Tensor>& tensors, int dim) {
    if (tensors.empty())
        throw std::runtime_error("stack: empty tensor list");

    // Unsqueeze all tensors at dim, then cat
    std::vector<Tensor> unsqueezed;
    unsqueezed.reserve(tensors.size());
    for (auto& t : tensors) {
        unsqueezed.push_back(t.unsqueeze(dim));
    }
    return cat(unsqueezed, dim);
}

// --- split ---
std::vector<Tensor> Tensor::split(int dim, const std::vector<int64_t>& sizes) const {
    if (dim < 0) dim += ndim();
    int64_t total = 0;
    for (auto s : sizes) total += s;
    if (total != shape_[dim])
        throw std::runtime_error("split: sizes don't sum to dim size");

    std::vector<Tensor> result;
    result.reserve(sizes.size());
    int64_t offset = 0;
    for (auto sz : sizes) {
        result.push_back(slice(dim, offset, offset + sz));
        offset += sz;
    }
    return result;
}

// --- unbind ---
std::vector<Tensor> Tensor::unbind(int dim) const {
    if (dim < 0) dim += ndim();
    int64_t n = shape_[dim];
    std::vector<Tensor> result;
    result.reserve(n);
    for (int64_t i = 0; i < n; ++i) {
        Tensor s = slice(dim, i, i + 1);
        // Squeeze the dim
        result.push_back(s.squeeze(dim));
    }
    return result;
}

// --- pad ---
Tensor Tensor::pad(const std::vector<int64_t>& padding, float value) const {
    // padding is {left, right} for last dim, or {left, right, top, bottom} etc.
    // PyTorch convention: pairs from last dim backwards
    int n_pairs = (int)padding.size() / 2;
    int nd = ndim();

    // Build per-dim pad_before and pad_after
    std::vector<int64_t> pad_before(nd, 0);
    std::vector<int64_t> pad_after(nd, 0);
    for (int i = 0; i < n_pairs && i < nd; ++i) {
        int dim_idx = nd - 1 - i;
        pad_before[dim_idx] = padding[2 * i];
        pad_after[dim_idx] = padding[2 * i + 1];
    }

    std::vector<int64_t> out_shape(nd);
    for (int d = 0; d < nd; ++d) {
        out_shape[d] = shape_[d] + pad_before[d] + pad_after[d];
    }

    int64_t out_numel = 1;
    for (auto s : out_shape) out_numel *= s;
    Tensor out = Tensor::empty(out_shape, dtype_);
    if (out_numel == 0) return out;

    int elem_size = (int)dtype_size(dtype_);
    PadMeta pmeta;
    pmeta.ndim = nd;
    for (int i = 0; i < nd; ++i) {
        pmeta.src_shape[i] = shape_[i];
        pmeta.dst_shape[i] = out_shape[i];
        pmeta.pad_before[i] = pad_before[i];
    }

    int grid_sz = grid_size(out_numel);
    pad_const_kernel<<<grid_sz, BLOCK_SIZE>>>(
        (const char*)data_, (char*)out.data_,
        pmeta, elem_size, value, (int)dtype_, out_numel);
    CUDA_CHECK(cudaGetLastError());

    return out;
}

// --- pad_reflect ---
Tensor Tensor::pad_reflect(const std::vector<int64_t>& padding) const {
    int n_pairs = (int)padding.size() / 2;
    int nd = ndim();

    std::vector<int64_t> pad_before(nd, 0);
    std::vector<int64_t> pad_after(nd, 0);
    for (int i = 0; i < n_pairs && i < nd; ++i) {
        int dim_idx = nd - 1 - i;
        pad_before[dim_idx] = padding[2 * i];
        pad_after[dim_idx] = padding[2 * i + 1];
    }

    std::vector<int64_t> out_shape(nd);
    for (int d = 0; d < nd; ++d) {
        out_shape[d] = shape_[d] + pad_before[d] + pad_after[d];
    }

    int64_t out_numel = 1;
    for (auto s : out_shape) out_numel *= s;
    Tensor out = Tensor::empty(out_shape, dtype_);
    if (out_numel == 0) return out;

    int elem_size = (int)dtype_size(dtype_);
    PadMeta pmeta;
    pmeta.ndim = nd;
    for (int i = 0; i < nd; ++i) {
        pmeta.src_shape[i] = shape_[i];
        pmeta.dst_shape[i] = out_shape[i];
        pmeta.pad_before[i] = pad_before[i];
    }

    int grid_sz = grid_size(out_numel);
    pad_reflect_kernel<<<grid_sz, BLOCK_SIZE>>>(
        (const char*)data_, (char*)out.data_,
        pmeta, elem_size, out_numel);
    CUDA_CHECK(cudaGetLastError());

    return out;
}

// --- to_dtype ---
Tensor Tensor::to_dtype(DType new_dtype) const {
    if (new_dtype == dtype_) return *this;  // No-op when already correct dtype

    Tensor out = Tensor::empty(shape_, new_dtype);
    if (numel_ == 0) return out;

    int grid_sz = grid_size(numel_);

    if (dtype_ == DType::Float32 && new_dtype == DType::Float16) {
        f32_to_f16_kernel<<<grid_sz, BLOCK_SIZE>>>(data_f32(), out.data_f16(), numel_);
    } else if (dtype_ == DType::Float16 && new_dtype == DType::Float32) {
        f16_to_f32_kernel<<<grid_sz, BLOCK_SIZE>>>((const __half*)data_, out.data_f32(), numel_);
    } else if (dtype_ == DType::Int64 && new_dtype == DType::Float32) {
        i64_to_f32_kernel<<<grid_sz, BLOCK_SIZE>>>((const int64_t*)data_, out.data_f32(), numel_);
    } else if (dtype_ == DType::Float32 && new_dtype == DType::Int64) {
        f32_to_i64_kernel<<<grid_sz, BLOCK_SIZE>>>(data_f32(), out.data_i64(), numel_);
    } else if (dtype_ == DType::Int64 && new_dtype == DType::Float16) {
        i64_to_f16_kernel<<<grid_sz, BLOCK_SIZE>>>((const int64_t*)data_, out.data_f16(), numel_);
    } else if (dtype_ == DType::Float16 && new_dtype == DType::Int64) {
        f16_to_i64_kernel<<<grid_sz, BLOCK_SIZE>>>((const __half*)data_, out.data_i64(), numel_);
    } else {
        throw std::runtime_error("to_dtype: unsupported conversion");
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

// ============================================================================
// Broadcasting helpers
// ============================================================================

static BroadcastMeta compute_broadcast_meta(const std::vector<int64_t>& a_shape,
                                             const std::vector<int64_t>& a_strides,
                                             const std::vector<int64_t>& b_shape,
                                             const std::vector<int64_t>& b_strides,
                                             std::vector<int64_t>& out_shape) {
    int nd_a = (int)a_shape.size();
    int nd_b = (int)b_shape.size();
    int nd = std::max(nd_a, nd_b);
    if (nd > MAX_DIMS) throw std::runtime_error("broadcast: too many dims (max 8)");

    out_shape.resize(nd);
    BroadcastMeta meta;
    meta.ndim = nd;

    for (int d = 0; d < nd; ++d) {
        int a_idx = nd_a - nd + d;
        int b_idx = nd_b - nd + d;
        int64_t sa = (a_idx >= 0) ? a_shape[a_idx] : 1;
        int64_t sb = (b_idx >= 0) ? b_shape[b_idx] : 1;
        int64_t stride_a = (a_idx >= 0 && sa != 1) ? a_strides[a_idx] : 0;
        int64_t stride_b = (b_idx >= 0 && sb != 1) ? b_strides[b_idx] : 0;

        if (sa != sb && sa != 1 && sb != 1) {
            throw std::runtime_error("broadcast: incompatible shapes");
        }
        out_shape[d] = std::max(sa, sb);
        meta.out_shape[d] = out_shape[d];
        meta.a_strides[d] = (sa == 1) ? 0 : stride_a;
        meta.b_strides[d] = (sb == 1) ? 0 : stride_b;
    }
    return meta;
}

// For in-place: `a` is the output, so a_shape must match out_shape.
static BroadcastMeta compute_broadcast_meta_inplace(const std::vector<int64_t>& a_shape,
                                                     const std::vector<int64_t>& b_shape,
                                                     const std::vector<int64_t>& b_strides) {
    int nd_a = (int)a_shape.size();
    int nd_b = (int)b_shape.size();
    int nd = nd_a; // a is output, so nd == nd_a
    if (nd > MAX_DIMS) throw std::runtime_error("broadcast inplace: too many dims");

    BroadcastMeta meta;
    meta.ndim = nd;

    for (int d = 0; d < nd; ++d) {
        int b_idx = nd_b - nd + d;
        int64_t sb = (b_idx >= 0) ? b_shape[b_idx] : 1;
        int64_t stride_b = (b_idx >= 0 && sb != 1) ? b_strides[b_idx] : 0;

        if (a_shape[d] != sb && sb != 1) {
            throw std::runtime_error("broadcast inplace: incompatible shapes");
        }
        meta.out_shape[d] = a_shape[d];
        meta.a_strides[d] = 0; // not used for inplace (a is contiguous)
        meta.b_strides[d] = (sb == 1) ? 0 : stride_b;
    }
    return meta;
}

// ============================================================================
// Element-wise operators
// ============================================================================

template <BinaryOp op>
static Tensor binary_op(const Tensor& a, const Tensor& b) {
    std::vector<int64_t> out_shape;
    BroadcastMeta meta = compute_broadcast_meta(
        a.shape(), a.strides(), b.shape(), b.strides(), out_shape);

    int64_t out_numel = 1;
    for (auto s : out_shape) out_numel *= s;

    DType dt = a.dtype();
    if (dt != b.dtype()) throw std::runtime_error("binary op: dtype mismatch");

    Tensor out = Tensor::empty(out_shape, dt);
    if (out_numel == 0) return out;

    int grid_sz = grid_size(out_numel);
    if (dt == DType::Float32) {
        elementwise_binary_f32_kernel<op><<<grid_sz, BLOCK_SIZE>>>(
            a.data_f32(), b.data_f32(), out.data_f32(), meta, out_numel);
    } else if (dt == DType::Float16) {
        elementwise_binary_f16_kernel<op><<<grid_sz, BLOCK_SIZE>>>(
            (const __half*)a.data_ptr(), (const __half*)b.data_ptr(),
            (__half*)out.data_ptr(), meta, out_numel);
    } else {
        throw std::runtime_error("binary op: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor Tensor::operator+(const Tensor& other) const { return binary_op<BinaryOp::Add>(*this, other); }
Tensor Tensor::operator-(const Tensor& other) const { return binary_op<BinaryOp::Sub>(*this, other); }
Tensor Tensor::operator*(const Tensor& other) const { return binary_op<BinaryOp::Mul>(*this, other); }
Tensor Tensor::operator/(const Tensor& other) const { return binary_op<BinaryOp::Div>(*this, other); }

Tensor Tensor::operator-() const {
    Tensor out = Tensor::empty(shape_, dtype_);
    if (numel_ == 0) return out;
    int grid_sz = grid_size(numel_);
    if (dtype_ == DType::Float32) {
        negate_f32_kernel<<<grid_sz, BLOCK_SIZE>>>(data_f32(), out.data_f32(), numel_);
    } else if (dtype_ == DType::Float16) {
        negate_f16_kernel<<<grid_sz, BLOCK_SIZE>>>(
            (const __half*)data_, (__half*)out.data_, numel_);
    } else {
        throw std::runtime_error("negate: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

// --- In-place operations ---

template <BinaryOp op>
static void binary_op_inplace(Tensor& a, const Tensor& b) {
    DType dt = a.dtype();
    if (dt != b.dtype()) throw std::runtime_error("inplace binary op: dtype mismatch");

    BroadcastMeta meta = compute_broadcast_meta_inplace(a.shape(), b.shape(), b.strides());
    int64_t N = a.numel();
    if (N == 0) return;

    int grid_sz = grid_size(N);
    if (dt == DType::Float32) {
        elementwise_binary_inplace_f32_kernel<op><<<grid_sz, BLOCK_SIZE>>>(
            a.data_f32(), b.data_f32(), meta, N);
    } else if (dt == DType::Float16) {
        elementwise_binary_inplace_f16_kernel<op><<<grid_sz, BLOCK_SIZE>>>(
            a.data_f16(), (const __half*)b.data_ptr(), meta, N);
    } else {
        throw std::runtime_error("inplace binary op: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());
}

Tensor& Tensor::add_(const Tensor& other) {
    binary_op_inplace<BinaryOp::Add>(*this, other);
    return *this;
}

Tensor& Tensor::mul_(const Tensor& other) {
    binary_op_inplace<BinaryOp::Mul>(*this, other);
    return *this;
}

Tensor& Tensor::add_scalar_(float val) {
    if (numel_ == 0) return *this;
    int grid_sz = grid_size(numel_);
    if (dtype_ == DType::Float32) {
        add_scalar_f32_kernel<<<grid_sz, BLOCK_SIZE>>>(data_f32(), val, numel_);
    } else if (dtype_ == DType::Float16) {
        add_scalar_f16_kernel<<<grid_sz, BLOCK_SIZE>>>(data_f16(), __float2half(val), numel_);
    } else {
        throw std::runtime_error("add_scalar_: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());
    return *this;
}

Tensor& Tensor::mul_scalar_(float val) {
    if (numel_ == 0) return *this;
    int grid_sz = grid_size(numel_);
    if (dtype_ == DType::Float32) {
        mul_scalar_f32_kernel<<<grid_sz, BLOCK_SIZE>>>(data_f32(), val, numel_);
    } else if (dtype_ == DType::Float16) {
        mul_scalar_f16_kernel<<<grid_sz, BLOCK_SIZE>>>(data_f16(), __float2half(val), numel_);
    } else {
        throw std::runtime_error("mul_scalar_: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());
    return *this;
}

Tensor& Tensor::clamp_(float min_val, float max_val) {
    if (numel_ == 0) return *this;
    int grid_sz = grid_size(numel_);
    if (dtype_ == DType::Float32) {
        clamp_f32_kernel<<<grid_sz, BLOCK_SIZE>>>(data_f32(), min_val, max_val, numel_);
    } else if (dtype_ == DType::Float16) {
        clamp_f16_kernel<<<grid_sz, BLOCK_SIZE>>>(
            data_f16(), __float2half(min_val), __float2half(max_val), numel_);
    } else {
        throw std::runtime_error("clamp_: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());
    return *this;
}

Tensor& Tensor::fill_(float val) {
    if (numel_ == 0) return *this;
    int grid_sz = grid_size(numel_);
    if (dtype_ == DType::Float32) {
        fill_f32_kernel<<<grid_sz, BLOCK_SIZE>>>(data_f32(), val, numel_);
    } else if (dtype_ == DType::Float16) {
        fill_f16_kernel<<<grid_sz, BLOCK_SIZE>>>(data_f16(), __float2half(val), numel_);
    } else if (dtype_ == DType::Int64) {
        fill_i64_kernel<<<grid_sz, BLOCK_SIZE>>>(data_i64(), (int64_t)val, numel_);
    }
    CUDA_CHECK(cudaGetLastError());
    return *this;
}

// --- Comparison ---
Tensor Tensor::operator>(float val) const {
    Tensor out = Tensor::empty(shape_, dtype_);
    if (numel_ == 0) return out;
    int grid_sz = grid_size(numel_);
    if (dtype_ == DType::Float32) {
        gt_scalar_f32_kernel<<<grid_sz, BLOCK_SIZE>>>(data_f32(), val, out.data_f32(), numel_);
    } else if (dtype_ == DType::Float16) {
        gt_scalar_f16_kernel<<<grid_sz, BLOCK_SIZE>>>(
            (const __half*)data_, __float2half(val), (__half*)out.data_, numel_);
    } else {
        throw std::runtime_error("operator>: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

// ============================================================================
// Reduction
// ============================================================================

static void compute_reduce_dims(const std::vector<int64_t>& shape, int dim,
                                int64_t& outer, int64_t& reduce, int64_t& inner) {
    outer = 1;
    for (int d = 0; d < dim; ++d) outer *= shape[d];
    reduce = shape[dim];
    inner = 1;
    for (int d = dim + 1; d < (int)shape.size(); ++d) inner *= shape[d];
}

Tensor Tensor::sum(int dim, bool keepdim) const {
    if (dim < 0) dim += ndim();
    if (dtype_ != DType::Float32) {
        // Convert to f32 first
        return to_f32().sum(dim, keepdim);
    }

    int64_t outer, reduce, inner;
    compute_reduce_dims(shape_, dim, outer, reduce, inner);

    std::vector<int64_t> out_shape = shape_;
    if (keepdim) {
        out_shape[dim] = 1;
    } else {
        out_shape.erase(out_shape.begin() + dim);
    }
    if (out_shape.empty()) out_shape.push_back(1);

    int64_t out_numel = outer * inner;
    Tensor out = Tensor::empty(out_shape, DType::Float32);

    if (dim == ndim() - 1 && inner == 1) {
        // Use optimized warp-shuffle reduction for last dim
        sum_reduce_last_dim_f32<<<(int)outer, BLOCK_SIZE>>>(
            data_f32(), out.data_f32(), reduce, outer);
    } else {
        int grid_sz = grid_size(out_numel);
        sum_reduce_general_f32<<<grid_sz, BLOCK_SIZE>>>(
            data_f32(), out.data_f32(), outer, reduce, inner, out_numel);
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor Tensor::mean(int dim, bool keepdim) const {
    if (dim < 0) dim += ndim();
    Tensor s = sum(dim, keepdim);
    float n = (float)shape_[dim < 0 ? dim + ndim() : dim];
    s.mul_scalar_(1.0f / n);
    return s;
}

Tensor Tensor::max(int dim, bool keepdim) const {
    if (dim < 0) dim += ndim();
    if (dtype_ != DType::Float32) {
        return to_f32().max(dim, keepdim);
    }

    int64_t outer, reduce, inner;
    compute_reduce_dims(shape_, dim, outer, reduce, inner);

    std::vector<int64_t> out_shape = shape_;
    if (keepdim) {
        out_shape[dim] = 1;
    } else {
        out_shape.erase(out_shape.begin() + dim);
    }
    if (out_shape.empty()) out_shape.push_back(1);

    int64_t out_numel = outer * inner;
    Tensor out = Tensor::empty(out_shape, DType::Float32);

    if (dim == ndim() - 1 && inner == 1) {
        max_reduce_last_dim_f32<<<(int)outer, BLOCK_SIZE>>>(
            data_f32(), out.data_f32(), reduce, outer);
    } else {
        int grid_sz = grid_size(out_numel);
        max_reduce_general_f32<<<grid_sz, BLOCK_SIZE>>>(
            data_f32(), out.data_f32(), outer, reduce, inner, out_numel);
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

// ============================================================================
// Copy
// ============================================================================

Tensor Tensor::clone() const {
    Tensor out = Tensor::empty(shape_, dtype_);
    if (numel_ > 0) {
        if (is_contiguous()) {
            CUDA_CHECK(cudaMemcpy(out.data_, data_, numel_ * dtype_size(dtype_),
                                   cudaMemcpyDeviceToDevice));
        } else {
            // Strided copy
            int nd = ndim();
            int elem_size = (int)dtype_size(dtype_);
            StridedCopyMeta meta;
            meta.ndim = nd;
            for (int i = 0; i < nd; ++i) {
                meta.src_strides[i] = strides_[i];
                meta.dst_shape[i] = shape_[i];
            }
            int grid_sz = grid_size(numel_);
            if (dtype_ == DType::Float32) {
                strided_copy_f32_kernel<<<grid_sz, BLOCK_SIZE>>>(
                    (const float*)data_, (float*)out.data_,
                    meta, numel_);
            } else if (dtype_ == DType::Float16) {
                strided_copy_f16_kernel<<<grid_sz, BLOCK_SIZE>>>(
                    (const __half*)data_, (__half*)out.data_,
                    meta, numel_);
            } else {
                strided_copy_kernel<<<grid_sz, BLOCK_SIZE>>>(
                    (const char*)data_, (char*)out.data_,
                    meta, elem_size, numel_);
            }
            CUDA_CHECK(cudaGetLastError());
        }
    }
    return out;
}

void Tensor::copy_from(const Tensor& src) {
    if (numel_ != src.numel_)
        throw std::runtime_error("copy_from: size mismatch");
    if (dtype_ != src.dtype_)
        throw std::runtime_error("copy_from: dtype mismatch");
    if (numel_ > 0) {
        CUDA_CHECK(cudaMemcpy(data_, src.data_, numel_ * dtype_size(dtype_),
                               cudaMemcpyDeviceToDevice));
    }
}

// ============================================================================
// Debug
// ============================================================================

std::string Tensor::shape_str() const {
    std::ostringstream ss;
    ss << "(";
    for (int i = 0; i < ndim(); ++i) {
        if (i > 0) ss << ", ";
        ss << shape_[i];
    }
    ss << ")";
    return ss.str();
}

void Tensor::print(const std::string& name, int max_elements) const {
    std::string dtype_str;
    switch (dtype_) {
        case DType::Float32: dtype_str = "float32"; break;
        case DType::Float16: dtype_str = "float16"; break;
        case DType::Int64: dtype_str = "int64"; break;
    }
    std::cout << "Tensor";
    if (!name.empty()) std::cout << " \"" << name << "\"";
    std::cout << " shape=" << shape_str() << " dtype=" << dtype_str
              << " numel=" << numel_ << std::endl;

    if (numel_ == 0) {
        std::cout << "  (empty)" << std::endl;
        return;
    }

    // Print first few elements
    std::vector<float> vals = to_cpu_f32();
    int n = std::min((int)vals.size(), max_elements);
    std::cout << "  [";
    for (int i = 0; i < n; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << vals[i];
    }
    if ((int)vals.size() > max_elements) std::cout << ", ...";
    std::cout << "]" << std::endl;
}

} // namespace cudasep
