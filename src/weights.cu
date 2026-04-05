#include "weights.h"
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <iostream>

namespace cudasep {

// ============================================================================
// JsonObjectData – the PIMPL body hidden from the header
// ============================================================================
struct JsonObjectData {
    std::unordered_map<std::string, JsonValue> data;
};

// ============================================================================
// JsonValue – Big-5 special members (need JsonObjectData to be complete)
// ============================================================================

JsonValue::JsonValue()
    : type_(Null), bool_val_(false), number_val_(0.0) {}

JsonValue::~JsonValue() = default;

JsonValue::JsonValue(const JsonValue& other)
    : type_(other.type_), bool_val_(other.bool_val_),
      number_val_(other.number_val_), string_val_(other.string_val_),
      array_val_(other.array_val_) {
    if (other.object_val_)
        object_val_ = std::make_unique<JsonObjectData>(*other.object_val_);
}

JsonValue::JsonValue(JsonValue&& other) noexcept = default;

JsonValue& JsonValue::operator=(const JsonValue& other) {
    if (this != &other) {
        type_       = other.type_;
        bool_val_   = other.bool_val_;
        number_val_ = other.number_val_;
        string_val_ = other.string_val_;
        array_val_  = other.array_val_;
        if (other.object_val_)
            object_val_ = std::make_unique<JsonObjectData>(*other.object_val_);
        else
            object_val_.reset();
    }
    return *this;
}

JsonValue& JsonValue::operator=(JsonValue&& other) noexcept = default;

// ============================================================================
// JsonValue – accessors
// ============================================================================

bool JsonValue::as_bool() const {
    if (type_ == Bool) return bool_val_;
    if (type_ == Number) return number_val_ != 0.0;  // coerce numeric to bool
    throw std::runtime_error("JsonValue: not a bool");
}

double JsonValue::as_number() const {
    if (type_ != Number) throw std::runtime_error("JsonValue: not a number");
    return number_val_;
}

const std::string& JsonValue::as_string() const {
    if (type_ != String) throw std::runtime_error("JsonValue: not a string");
    return string_val_;
}

size_t JsonValue::size() const {
    if (type_ == Array)  return array_val_.size();
    if (type_ == Object) return object_val_->data.size();
    throw std::runtime_error("JsonValue: size() called on non-container");
}

const JsonValue& JsonValue::operator[](size_t index) const {
    if (type_ != Array) throw std::runtime_error("JsonValue: not an array");
    if (index >= array_val_.size())
        throw std::runtime_error("JsonValue: array index out of range");
    return array_val_[index];
}

const JsonValue& JsonValue::operator[](const std::string& key) const {
    if (type_ != Object) throw std::runtime_error("JsonValue: not an object");
    auto it = object_val_->data.find(key);
    if (it == object_val_->data.end())
        throw std::runtime_error("JsonValue: key not found: " + key);
    return it->second;
}

bool JsonValue::has(const std::string& key) const {
    if (type_ != Object) return false;
    return object_val_->data.count(key) > 0;
}

int JsonValue::get_int(const std::string& key, int def) const {
    if (!has(key)) return def;
    return (*this)[key].as_int();
}

float JsonValue::get_float(const std::string& key, float def) const {
    if (!has(key)) return def;
    return (*this)[key].as_float();
}

bool JsonValue::get_bool(const std::string& key, bool def) const {
    if (!has(key)) return def;
    return (*this)[key].as_bool();
}

std::string JsonValue::get_string(const std::string& key,
                                  const std::string& def) const {
    if (!has(key)) return def;
    return (*this)[key].as_string();
}

// ============================================================================
// JsonValue – factory helpers
// ============================================================================

JsonValue JsonValue::make_bool(bool v) {
    JsonValue j;
    j.type_ = Bool;
    j.bool_val_ = v;
    return j;
}

JsonValue JsonValue::make_number(double v) {
    JsonValue j;
    j.type_ = Number;
    j.number_val_ = v;
    return j;
}

JsonValue JsonValue::make_string(const std::string& v) {
    JsonValue j;
    j.type_ = String;
    j.string_val_ = v;
    return j;
}

JsonValue JsonValue::make_array(std::vector<JsonValue> v) {
    JsonValue j;
    j.type_ = Array;
    j.array_val_ = std::move(v);
    return j;
}

JsonValue JsonValue::make_object(std::unordered_map<std::string, JsonValue> v) {
    JsonValue j;
    j.type_ = Object;
    j.object_val_ = std::make_unique<JsonObjectData>();
    j.object_val_->data = std::move(v);
    return j;
}

// ============================================================================
// JsonValue – recursive-descent parser
// ============================================================================

namespace {

class JsonParser {
public:
    explicit JsonParser(const std::string& src) : src_(src), pos_(0) {}

    JsonValue parse() {
        skip_ws();
        JsonValue v = parse_value();
        skip_ws();
        if (pos_ != src_.size())
            error("unexpected trailing content");
        return v;
    }

private:
    const std::string& src_;
    size_t pos_;

    // ---- helpers ----

    [[noreturn]] void error(const std::string& msg) const {
        throw std::runtime_error("JSON parse error at position " +
                                 std::to_string(pos_) + ": " + msg);
    }

    char peek() const {
        if (pos_ >= src_.size()) return '\0';
        return src_[pos_];
    }

    char advance() {
        if (pos_ >= src_.size()) error("unexpected end of input");
        return src_[pos_++];
    }

    void expect(char c) {
        char got = advance();
        if (got != c)
            error(std::string("expected '") + c + "' but got '" + got + "'");
    }

    void skip_ws() {
        while (pos_ < src_.size()) {
            char c = src_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
                ++pos_;
            else
                break;
        }
    }

    // ---- value dispatch ----

    JsonValue parse_value() {
        skip_ws();
        char c = peek();
        if (c == '"')  return parse_string_value();
        if (c == '{')  return parse_object();
        if (c == '[')  return parse_array();
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n')  return parse_null();
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
        error(std::string("unexpected character: '") + c + "'");
    }

    // ---- strings ----

    std::string parse_string() {
        expect('"');
        std::string result;
        while (true) {
            if (pos_ >= src_.size()) error("unterminated string");
            char c = src_[pos_++];
            if (c == '"') break;
            if (c == '\\') {
                if (pos_ >= src_.size()) error("unterminated escape");
                char esc = src_[pos_++];
                switch (esc) {
                    case '"':  result += '"';  break;
                    case '\\': result += '\\'; break;
                    case '/':  result += '/';  break;
                    case 'b':  result += '\b'; break;
                    case 'f':  result += '\f'; break;
                    case 'n':  result += '\n'; break;
                    case 'r':  result += '\r'; break;
                    case 't':  result += '\t'; break;
                    case 'u': {
                        // Parse 4 hex digits – emit as UTF-8 (basic BMP only)
                        if (pos_ + 4 > src_.size()) error("incomplete \\u escape");
                        std::string hex = src_.substr(pos_, 4);
                        pos_ += 4;
                        unsigned cp = 0;
                        for (char h : hex) {
                            cp <<= 4;
                            if (h >= '0' && h <= '9') cp |= (h - '0');
                            else if (h >= 'a' && h <= 'f') cp |= (h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') cp |= (h - 'A' + 10);
                            else error("invalid hex in \\u escape");
                        }
                        if (cp < 0x80) {
                            result += (char)cp;
                        } else if (cp < 0x800) {
                            result += (char)(0xC0 | (cp >> 6));
                            result += (char)(0x80 | (cp & 0x3F));
                        } else {
                            result += (char)(0xE0 | (cp >> 12));
                            result += (char)(0x80 | ((cp >> 6) & 0x3F));
                            result += (char)(0x80 | (cp & 0x3F));
                        }
                        break;
                    }
                    default:
                        error(std::string("unknown escape: \\") + esc);
                }
            } else {
                result += c;
            }
        }
        return result;
    }

    JsonValue parse_string_value() {
        return JsonValue::make_string(parse_string());
    }

    // ---- numbers ----

    JsonValue parse_number() {
        size_t start = pos_;

        // Optional minus
        if (peek() == '-') advance();

        // Integer part
        if (peek() == '0') {
            advance();
        } else if (peek() >= '1' && peek() <= '9') {
            while (peek() >= '0' && peek() <= '9') advance();
        } else {
            error("invalid number");
        }

        // Fractional part
        if (peek() == '.') {
            advance();
            if (!(peek() >= '0' && peek() <= '9'))
                error("expected digit after '.'");
            while (peek() >= '0' && peek() <= '9') advance();
        }

        // Exponent
        if (peek() == 'e' || peek() == 'E') {
            advance();
            if (peek() == '+' || peek() == '-') advance();
            if (!(peek() >= '0' && peek() <= '9'))
                error("expected digit in exponent");
            while (peek() >= '0' && peek() <= '9') advance();
        }

        std::string num_str = src_.substr(start, pos_ - start);
        double val = std::stod(num_str);
        return JsonValue::make_number(val);
    }

    // ---- booleans ----

    JsonValue parse_bool() {
        if (src_.compare(pos_, 4, "true") == 0) {
            pos_ += 4;
            return JsonValue::make_bool(true);
        }
        if (src_.compare(pos_, 5, "false") == 0) {
            pos_ += 5;
            return JsonValue::make_bool(false);
        }
        error("expected 'true' or 'false'");
    }

    // ---- null ----

    JsonValue parse_null() {
        if (src_.compare(pos_, 4, "null") == 0) {
            pos_ += 4;
            return JsonValue(); // Null
        }
        error("expected 'null'");
    }

    // ---- arrays ----

    JsonValue parse_array() {
        expect('[');
        std::vector<JsonValue> elems;
        skip_ws();
        if (peek() == ']') { advance(); return JsonValue::make_array({}); }

        while (true) {
            elems.push_back(parse_value());
            skip_ws();
            if (peek() == ']') { advance(); break; }
            expect(',');
        }
        return JsonValue::make_array(std::move(elems));
    }

    // ---- objects ----

    JsonValue parse_object() {
        expect('{');
        std::unordered_map<std::string, JsonValue> members;
        skip_ws();
        if (peek() == '}') { advance(); return JsonValue::make_object({}); }

        while (true) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            expect(':');
            JsonValue val = parse_value();
            members[std::move(key)] = std::move(val);
            skip_ws();
            if (peek() == '}') { advance(); break; }
            expect(',');
        }
        return JsonValue::make_object(std::move(members));
    }
};

} // anonymous namespace

JsonValue JsonValue::parse(const std::string& json) {
    JsonParser parser(json);
    return parser.parse();
}

// ============================================================================
// ModelWeights
// ============================================================================

ModelWeights ModelWeights::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("ModelWeights: cannot open file: " + path);

    auto read_u32 = [&]() -> uint32_t {
        uint32_t v;
        file.read(reinterpret_cast<char*>(&v), 4);
        if (!file) throw std::runtime_error("ModelWeights: unexpected EOF reading uint32");
        return v;
    };

    auto read_i64 = [&]() -> int64_t {
        int64_t v;
        file.read(reinterpret_cast<char*>(&v), 8);
        if (!file) throw std::runtime_error("ModelWeights: unexpected EOF reading int64");
        return v;
    };

    auto read_bytes = [&](char* dst, size_t n) {
        file.read(dst, (std::streamsize)n);
        if (!file) throw std::runtime_error("ModelWeights: unexpected EOF reading data");
    };

    // ---- magic ----
    char magic[4];
    read_bytes(magic, 4);
    if (magic[0] != 'C' || magic[1] != 'S' || magic[2] != 'M' || magic[3] != '\0')
        throw std::runtime_error("ModelWeights: invalid magic (expected CSM\\0)");

    // ---- version ----
    uint32_t version = read_u32();
    if (version != 1)
        throw std::runtime_error("ModelWeights: unsupported version " +
                                 std::to_string(version));

    // ---- config JSON ----
    uint32_t config_len = read_u32();
    std::string config_json(config_len, '\0');
    read_bytes(&config_json[0], config_len);

    ModelWeights weights;
    weights.config_ = JsonValue::parse(config_json);

    // ---- tensors ----
    uint32_t num_tensors = read_u32();
    std::cout << "[ModelWeights] Loading " << num_tensors << " tensors from "
              << path << std::endl;

    for (uint32_t i = 0; i < num_tensors; ++i) {
        // Name
        uint32_t name_len = read_u32();
        std::string name(name_len, '\0');
        read_bytes(&name[0], name_len);

        // Shape
        uint32_t ndim = read_u32();
        std::vector<int64_t> shape(ndim);
        for (uint32_t d = 0; d < ndim; ++d)
            shape[d] = read_i64();

        // DType
        uint32_t dtype_id = read_u32();
        DType dtype;
        switch (dtype_id) {
            case 0: dtype = DType::Float32; break;
            case 1: dtype = DType::Float16; break;
            case 2: dtype = DType::Int64;   break;
            default:
                throw std::runtime_error("ModelWeights: unknown dtype " +
                                         std::to_string(dtype_id) +
                                         " for tensor " + name);
        }

        // Compute data size
        int64_t numel = 1;
        for (int64_t s : shape) numel *= s;
        size_t data_bytes = (size_t)numel * dtype_size(dtype);

        // Read to CPU buffer then copy to GPU
        Tensor t = Tensor::empty(shape, dtype);

        if (data_bytes > 0) {
            std::vector<char> cpu_buf(data_bytes);
            read_bytes(cpu_buf.data(), data_bytes);
            t.copy_from_cpu(cpu_buf.data(), data_bytes);
        }

        weights.tensors_[std::move(name)] = std::move(t);
    }

    std::cout << "[ModelWeights] Loaded " << weights.tensors_.size()
              << " tensors, config keys: " << weights.config_.size()
              << std::endl;

    return weights;
}

void ModelWeights::convert_linear_weights_to_fp16() {
    int converted = 0;
    size_t saved_bytes = 0;
    for (auto& kv : tensors_) {
        Tensor& t = kv.second;
        // Only convert 2D FP32 tensors (linear weights)
        if (t.ndim() == 2 && t.dtype() == DType::Float32) {
            saved_bytes += t.numel() * 2;  // save 2 bytes per element
            kv.second = t.to_f16();
            converted++;
        }
    }
    std::cout << "  [FP16] Pre-converted " << converted << " linear weight tensors"
              << " (saved " << (saved_bytes / 1048576) << " MB VRAM)" << std::endl;
}

const Tensor& ModelWeights::get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end())
        throw std::runtime_error("ModelWeights: tensor not found: " + name);
    return it->second;
}

bool ModelWeights::has(const std::string& name) const {
    return tensors_.count(name) > 0;
}

std::vector<std::string> ModelWeights::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (auto& kv : tensors_)
        names.push_back(kv.first);
    std::sort(names.begin(), names.end());
    return names;
}

} // namespace cudasep
