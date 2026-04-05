#pragma once
#include "tensor.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>

namespace cudasep {

// ============================================================================
// Minimal JSON value type (sufficient for model configs)
// ============================================================================
class JsonValue {
public:
    enum Type { Null, Bool, Number, String, Array, Object };

    JsonValue() : type_(Null), bool_val_(false), number_val_(0.0) {}

    // Type queries
    bool is_null()   const { return type_ == Null; }
    bool is_bool()   const { return type_ == Bool; }
    bool is_number() const { return type_ == Number; }
    bool is_string() const { return type_ == String; }
    bool is_array()  const { return type_ == Array; }
    bool is_object() const { return type_ == Object; }

    // Accessors (throw on type mismatch)
    bool                as_bool()   const;
    double              as_number() const;
    int                 as_int()    const { return (int)as_number(); }
    int64_t             as_int64()  const { return (int64_t)as_number(); }
    float               as_float()  const { return (float)as_number(); }
    const std::string&  as_string() const;

    // Container access
    size_t             size() const;                              // array or object size
    const JsonValue&   operator[](size_t index) const;            // array index
    const JsonValue&   operator[](const std::string& key) const;  // object key
    bool               has(const std::string& key) const;

    // Get with default (safe for missing keys)
    int         get_int(const std::string& key, int def) const;
    float       get_float(const std::string& key, float def) const;
    bool        get_bool(const std::string& key, bool def) const;
    std::string get_string(const std::string& key, const std::string& def) const;

    // Parse from JSON string
    static JsonValue parse(const std::string& json);

    // Factory helpers (used by the parser)
    static JsonValue make_bool(bool v);
    static JsonValue make_number(double v);
    static JsonValue make_string(const std::string& v);
    static JsonValue make_array(std::vector<JsonValue> v);
    static JsonValue make_object(std::unordered_map<std::string, JsonValue> v);

private:
    Type type_;
    bool bool_val_;
    double number_val_;
    std::string string_val_;
    std::vector<JsonValue> array_val_;
    std::unordered_map<std::string, JsonValue> object_val_;
};

// ============================================================================
// Binary weight file (.csm) reader
// ============================================================================
//
// File layout:
//   [4 bytes]  Magic "CSM\0"
//   [4 bytes]  Version (uint32_t, must be 1)
//   [4 bytes]  Config JSON length (uint32_t)
//   [config_len bytes]  Config JSON string
//   [4 bytes]  Number of tensors (uint32_t)
//   Per tensor:
//     [4 bytes]  Name length (uint32_t)
//     [name_len bytes]  Name string
//     [4 bytes]  Number of dimensions (uint32_t)
//     [ndim*8 bytes]  Shape (int64_t[])
//     [4 bytes]  DType (uint32_t: 0=float32, 1=float16)
//     [data_size bytes]  Raw tensor data (row-major, GPU-ready)
// ============================================================================

class ModelWeights {
public:
    /// Load all weights from a .csm file into GPU memory.
    static ModelWeights load(const std::string& path);

    /// Parsed model config.
    const JsonValue& config() const { return config_; }

    /// Look up a tensor by its fully-qualified name.
    /// Throws std::runtime_error if not found.
    const Tensor& get(const std::string& name) const;

    /// Check whether a tensor exists.
    bool has(const std::string& name) const;

    /// Return a list of all stored tensor names.
    std::vector<std::string> tensor_names() const;

    /// Total number of tensors.
    size_t num_tensors() const { return tensors_.size(); }

    /// Convert all 2D (linear) FP32 weight tensors to FP16 in-place.
    /// This avoids runtime FP32->FP16 conversion during GEMM.
    void convert_linear_weights_to_fp16();

private:
    JsonValue config_;
    std::unordered_map<std::string, Tensor> tensors_;
};

} // namespace cudasep
