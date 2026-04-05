#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <memory>
#include <string>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <functional>

namespace cudasep {

enum class DType { Float32 = 0, Float16 = 1, Int64 = 2 };

inline size_t dtype_size(DType dt) {
    switch(dt) {
        case DType::Float32: return 4;
        case DType::Float16: return 2;
        case DType::Int64: return 8;
    }
    return 0;
}

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err) + \
            " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while(0)

class Tensor {
public:
    Tensor(); // empty

    // Factory methods
    static Tensor empty(std::vector<int64_t> shape, DType dtype = DType::Float32);
    static Tensor zeros(std::vector<int64_t> shape, DType dtype = DType::Float32);
    static Tensor ones(std::vector<int64_t> shape, DType dtype = DType::Float32);
    static Tensor from_cpu_f32(const float* data, std::vector<int64_t> shape);
    static Tensor from_cpu_f16(const void* data, std::vector<int64_t> shape);
    static Tensor from_cpu_i64(const int64_t* data, std::vector<int64_t> shape);
    static Tensor full(std::vector<int64_t> shape, float value, DType dtype = DType::Float32);
    static Tensor arange(int64_t start, int64_t end, DType dtype = DType::Float32); // on GPU

    // Properties
    int ndim() const { return (int)shape_.size(); }
    int64_t size(int dim) const;
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    int64_t numel() const { return numel_; }
    DType dtype() const { return dtype_; }
    bool is_contiguous() const;
    bool is_empty() const { return numel_ == 0; }

    // Raw data access
    void* data_ptr() { return data_; }
    const void* data_ptr() const { return data_; }
    float* data_f32() { assert(dtype_ == DType::Float32); return (float*)data_; }
    const float* data_f32() const { assert(dtype_ == DType::Float32); return (const float*)data_; }
    __half* data_f16() { assert(dtype_ == DType::Float16); return (__half*)data_; }
    int64_t* data_i64() { assert(dtype_ == DType::Int64); return (int64_t*)data_; }
    const int64_t* data_i64() const { assert(dtype_ == DType::Int64); return (const int64_t*)data_; }

    // CPU transfer
    std::vector<float> to_cpu_f32() const;
    void copy_from_cpu(const void* src, size_t bytes);
    void copy_to_cpu(void* dst, size_t bytes) const;

    // Shape operations (these create new tensors with copied data for simplicity)
    Tensor reshape(std::vector<int64_t> new_shape) const;
    Tensor permute(std::vector<int> dims) const;
    Tensor transpose(int dim0, int dim1) const;
    Tensor contiguous() const;
    Tensor unsqueeze(int dim) const;
    Tensor squeeze(int dim) const;
    Tensor expand(std::vector<int64_t> new_shape) const; // broadcast expand

    // Slicing
    Tensor slice(int dim, int64_t start, int64_t end) const;
    Tensor index_select(int dim, const Tensor& indices) const;

    // Concat/split
    static Tensor cat(const std::vector<Tensor>& tensors, int dim);
    static Tensor stack(const std::vector<Tensor>& tensors, int dim);
    std::vector<Tensor> split(int dim, const std::vector<int64_t>& sizes) const;
    std::vector<Tensor> unbind(int dim) const;

    // Padding
    Tensor pad(const std::vector<int64_t>& padding, float value = 0.0f) const; // last-dim padding: {left, right}
    Tensor pad_reflect(const std::vector<int64_t>& padding) const;

    // Type conversion
    Tensor to_dtype(DType new_dtype) const;
    Tensor to_f32() const { return to_dtype(DType::Float32); }
    Tensor to_f16() const { return to_dtype(DType::Float16); }

    // Element-wise operators (return new tensors)
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator-() const;

    // In-place
    Tensor& add_(const Tensor& other); // this += other
    Tensor& mul_(const Tensor& other);
    Tensor& add_scalar_(float val);
    Tensor& mul_scalar_(float val);
    Tensor& clamp_(float min_val, float max_val);
    Tensor& fill_(float val);

    // Reduction
    Tensor sum(int dim, bool keepdim = false) const;
    Tensor mean(int dim, bool keepdim = false) const;
    Tensor max(int dim, bool keepdim = false) const;

    // Comparison
    Tensor operator>(float val) const;

    // Copy
    Tensor clone() const;
    void copy_from(const Tensor& src);

    // Debug
    void print(const std::string& name = "", int max_elements = 10) const;
    std::string shape_str() const;

private:
    std::shared_ptr<void> storage_; // GPU memory (reference counted, CUDA deleter)
    void* data_;                     // pointer into storage
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    int64_t numel_;
    DType dtype_;

    void compute_strides(); // compute contiguous strides from shape
    static std::shared_ptr<void> alloc_gpu(size_t bytes);
};

} // namespace cudasep
