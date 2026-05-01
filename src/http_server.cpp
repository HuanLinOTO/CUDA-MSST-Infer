#include "http_server.h"

#include "audio_io.h"
#include "inference_app.h"
#include "memory_pool.h"
#include "ops.h"
#include "webui_embedded.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace cudasep::app {
namespace {

#ifdef _WIN32
using SocketHandle = SOCKET;
static constexpr SocketHandle kInvalidSocket = INVALID_SOCKET;
#else
using SocketHandle = int;
static constexpr SocketHandle kInvalidSocket = -1;
#endif

struct SocketApi {
    SocketApi() {
#ifdef _WIN32
        WSADATA data;
        if (WSAStartup(MAKEWORD(2, 2), &data) != 0) {
            throw std::runtime_error("WSAStartup failed");
        }
#endif
    }
    ~SocketApi() {
#ifdef _WIN32
        WSACleanup();
#endif
    }
};

static void close_socket(SocketHandle sock) {
    if (sock == kInvalidSocket) return;
#ifdef _WIN32
    closesocket(sock);
#else
    close(sock);
#endif
}

struct SocketGuard {
    SocketHandle sock = kInvalidSocket;
    explicit SocketGuard(SocketHandle value = kInvalidSocket) : sock(value) {}
    ~SocketGuard() { close_socket(sock); }
    SocketGuard(const SocketGuard&) = delete;
    SocketGuard& operator=(const SocketGuard&) = delete;
};

struct HttpRequest {
    std::string method;
    std::string path;
    std::map<std::string, std::string> headers;
    std::string body;
};

struct HttpResponse {
    int status = 200;
    std::string content_type = "text/plain; charset=utf-8";
    std::vector<uint8_t> body;
    std::vector<std::pair<std::string, std::string>> headers;
};

struct ByteRange {
    size_t start = 0;
    size_t end = 0;
    bool valid = false;
};

struct UploadedFile {
    std::string field_name;
    std::string filename;
    std::string content_type;
    std::vector<uint8_t> data;
};

struct MultipartForm {
    std::map<std::string, std::string> fields;
    std::vector<UploadedFile> files;
};

struct ModelEntry {
    std::string relative_name;
    fs::path absolute_path;
};

struct TrackAsset {
    std::string name;
    bool derived = false;
    std::vector<uint8_t> wav_bytes;
};

struct JobState {
    std::string id;
    std::string model_name;
    std::string original_filename;
    std::string sanitized_base;
    bool quantize_fp16 = false;
    bool enable_cuda_graph = true;
    bool keep_model_loaded = true;
    int chunk_batch_size = 0;
    std::string status = "排队中";
    std::string error;
    int progress = 0;
    std::vector<std::string> logs;
    std::vector<TrackAsset> tracks;
    std::chrono::steady_clock::time_point created_at = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point started_at;
    std::chrono::steady_clock::time_point finished_at;
    bool started = false;
    bool finished = false;
    std::mutex mutex;
};

static std::atomic<unsigned long long> g_job_counter{0};

static std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

static std::string trim(std::string s) {
    while (!s.empty() && (s.back() == ' ' || s.back() == '\r' || s.back() == '\n' || s.back() == '\t')) s.pop_back();
    size_t start = 0;
    while (start < s.size() && (s[start] == ' ' || s[start] == '\t')) start++;
    return s.substr(start);
}

static bool parse_toggle_value(std::string value) {
    value = to_lower(trim(std::move(value)));
    return value == "1" || value == "true" || value == "on" || value == "yes" ||
           value == "enabled" || value == "mixed" || value == "keep";
}

static bool parse_precision_mode(std::string value, bool current) {
    value = to_lower(trim(std::move(value)));
    if (value.empty()) return current;
    if (value == "native" || value == "model" || value == "source" ||
        value == "0" || value == "false" || value == "off" || value == "fp32") {
        return false;
    }
    if (value == "mixed" || value == "accelerated" || value == "1" ||
        value == "true" || value == "on" || value == "fp16") {
        return true;
    }
    return current;
}

static std::string precision_mode_title(bool mixed_precision) {
    return mixed_precision ? "混精加速" : "遵循模型文件";
}

static std::string precision_mode_description(bool mixed_precision) {
    if (mixed_precision) {
        return "如果 .csm 里仍有 FP32 线性权重，加载后会转成 FP16，并启用 attention 混精路径。";
    }
    return "按 .csm 文件里的原始 dtype 推理，不额外把 FP32 线性权重转成 FP16。";
}

static std::string html_escape(const std::string& input) {
    std::string out;
    out.reserve(input.size());
    for (char c : input) {
        switch (c) {
            case '&': out += "&amp;"; break;
            case '<': out += "&lt;"; break;
            case '>': out += "&gt;"; break;
            case '"': out += "&quot;"; break;
            default: out.push_back(c); break;
        }
    }
    return out;
}

static std::string json_escape(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 8);
    for (char c : input) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out.push_back(c); break;
        }
    }
    return out;
}

static std::string sanitize_ascii(std::string name) {
    for (char& c : name) {
        unsigned char uc = (unsigned char)c;
        if (!(std::isalnum(uc) || c == '.' || c == '_' || c == '-')) c = '_';
    }
    while (!name.empty() && (name.front() == '.' || name.front() == '_')) name.erase(name.begin());
    if (name.empty()) name = "upload";
    return name;
}

static std::string base_name_of(const std::string& path) {
    size_t slash = path.find_last_of("/\\");
    std::string name = (slash == std::string::npos) ? path : path.substr(slash + 1);
    size_t dot = name.find_last_of('.');
    if (dot != std::string::npos) name = name.substr(0, dot);
    return sanitize_ascii(name);
}

static std::string extension_of(const std::string& path) {
    size_t slash = path.find_last_of("/\\");
    size_t dot = path.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) return ".bin";
    std::string ext = sanitize_ascii(to_lower(path.substr(dot)));
    if (ext.empty() || ext == ".") return ".bin";
    return ext;
}

static std::vector<ModelEntry> scan_models(const fs::path& model_dir) {
    std::vector<ModelEntry> models;
    if (!fs::exists(model_dir)) return models;
    for (const auto& entry : fs::recursive_directory_iterator(model_dir)) {
        if (!entry.is_regular_file()) continue;
        if (to_lower(entry.path().extension().string()) != ".csm") continue;
        ModelEntry model;
        model.absolute_path = entry.path();
        model.relative_name = fs::relative(entry.path(), model_dir).generic_string();
        models.push_back(std::move(model));
    }
    std::sort(models.begin(), models.end(), [](const ModelEntry& a, const ModelEntry& b) {
        return a.relative_name < b.relative_name;
    });
    return models;
}

static std::string header_value(const HttpRequest& req, const std::string& key) {
    auto it = req.headers.find(to_lower(key));
    return it == req.headers.end() ? std::string() : it->second;
}

static HttpResponse text_response(int status, const std::string& body, const std::string& content_type) {
    HttpResponse res;
    res.status = status;
    res.content_type = content_type;
    res.body.assign(body.begin(), body.end());
    return res;
}

static HttpResponse html_response(int status, const std::string& body) {
    return text_response(status, body, "text/html; charset=utf-8");
}

static HttpResponse json_response(int status, const std::string& body) {
    return text_response(status, body, "application/json; charset=utf-8");
}

static bool send_all(SocketHandle sock, const uint8_t* data, size_t size) {
    size_t sent = 0;
    while (sent < size) {
        int chunk = (int)std::min<size_t>(size - sent, 1 << 20);
        int rc = send(sock, reinterpret_cast<const char*>(data + sent), chunk, 0);
        if (rc <= 0) return false;
        sent += (size_t)rc;
    }
    return true;
}

static bool send_response(SocketHandle sock, const HttpResponse& response) {
    auto status_text = [&](int status) {
        switch (status) {
            case 200: return "OK";
            case 206: return "Partial Content";
            case 400: return "Bad Request";
            case 404: return "Not Found";
            case 405: return "Method Not Allowed";
            case 416: return "Range Not Satisfiable";
            case 413: return "Payload Too Large";
            case 500: return "Internal Server Error";
            default: return "OK";
        }
    };

    std::ostringstream oss;
    oss << "HTTP/1.1 " << response.status << " " << status_text(response.status) << "\r\n";
    oss << "Content-Type: " << response.content_type << "\r\n";
    oss << "Content-Length: " << response.body.size() << "\r\n";
    oss << "Connection: close\r\n";
    for (const auto& header : response.headers) {
        oss << header.first << ": " << header.second << "\r\n";
    }
    oss << "\r\n";
    std::string headers = oss.str();

    if (!send_all(sock, reinterpret_cast<const uint8_t*>(headers.data()), headers.size())) return false;
    if (!response.body.empty()) return send_all(sock, response.body.data(), response.body.size());
    return true;
}

static bool recv_request(SocketHandle sock, size_t max_body_size, HttpRequest& out_request, HttpResponse& out_error) {
    std::string buffer;
    buffer.reserve(8192);
    char temp[8192];
    size_t header_end = std::string::npos;

    while (header_end == std::string::npos) {
        int rc = recv(sock, temp, (int)sizeof(temp), 0);
        if (rc <= 0) return false;
        buffer.append(temp, temp + rc);
        header_end = buffer.find("\r\n\r\n");
        if (buffer.size() > 64 * 1024) {
            out_error = text_response(400, "Headers too large", "text/plain; charset=utf-8");
            return false;
        }
    }

    std::string header_block = buffer.substr(0, header_end);
    std::istringstream iss(header_block);
    std::string request_line;
    std::getline(iss, request_line);
    request_line = trim(request_line);
    std::istringstream line_stream(request_line);
    line_stream >> out_request.method >> out_request.path;
    if (out_request.method.empty() || out_request.path.empty()) {
        out_error = text_response(400, "Malformed request line", "text/plain; charset=utf-8");
        return false;
    }

    std::string line;
    while (std::getline(iss, line)) {
        line = trim(line);
        if (line.empty()) continue;
        size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        out_request.headers[to_lower(trim(line.substr(0, colon)))] = trim(line.substr(colon + 1));
    }

    size_t content_length = 0;
    std::string content_length_value = header_value(out_request, "content-length");
    if (!content_length_value.empty()) content_length = (size_t)std::stoull(content_length_value);
    if (content_length > max_body_size) {
        out_error = text_response(413, "Upload too large", "text/plain; charset=utf-8");
        return false;
    }

    out_request.body = buffer.substr(header_end + 4);
    while (out_request.body.size() < content_length) {
        int rc = recv(sock, temp, (int)sizeof(temp), 0);
        if (rc <= 0) break;
        out_request.body.append(temp, temp + rc);
    }
    if (out_request.body.size() < content_length) {
        out_error = text_response(400, "Request body truncated", "text/plain; charset=utf-8");
        return false;
    }
    if (out_request.body.size() > content_length) out_request.body.resize(content_length);
    return true;
}

static std::string get_disposition_attr(const std::string& value, const std::string& key) {
    std::string pattern = key + "=\"";
    size_t start = value.find(pattern);
    if (start == std::string::npos) return std::string();
    start += pattern.size();
    size_t end = value.find('"', start);
    return end == std::string::npos ? std::string() : value.substr(start, end - start);
}

static ByteRange parse_range_header(const std::string& header_value, size_t total_size) {
    ByteRange range;
    if (header_value.empty() || total_size == 0) return range;
    if (header_value.rfind("bytes=", 0) != 0) return range;
    std::string spec = trim(header_value.substr(6));
    size_t comma = spec.find(',');
    if (comma != std::string::npos) {
        spec = spec.substr(0, comma);
    }
    size_t dash = spec.find('-');
    if (dash == std::string::npos) return range;

    std::string left = trim(spec.substr(0, dash));
    std::string right = trim(spec.substr(dash + 1));
    try {
        if (left.empty()) {
            size_t suffix = (size_t)std::stoull(right);
            if (suffix == 0) return range;
            if (suffix >= total_size) {
                range.start = 0;
            } else {
                range.start = total_size - suffix;
            }
            range.end = total_size - 1;
        } else {
            range.start = (size_t)std::stoull(left);
            range.end = right.empty() ? (total_size - 1) : (size_t)std::stoull(right);
        }
    } catch (...) {
        return range;
    }

    if (range.start >= total_size) return range;
    if (range.end >= total_size) range.end = total_size - 1;
    if (range.end < range.start) return range;
    range.valid = true;
    return range;
}

static MultipartForm parse_multipart(const HttpRequest& request) {
    std::string content_type = header_value(request, "content-type");
    size_t boundary_pos = content_type.find("boundary=");
    if (boundary_pos == std::string::npos) throw std::runtime_error("缺少 multipart boundary");
    std::string boundary = content_type.substr(boundary_pos + 9);
    if (!boundary.empty() && boundary.front() == '"' && boundary.back() == '"') {
        boundary = boundary.substr(1, boundary.size() - 2);
    }

    std::string delimiter = "--" + boundary;
    MultipartForm form;
    size_t pos = 0;
    while (true) {
        size_t part_start = request.body.find(delimiter, pos);
        if (part_start == std::string::npos) break;
        part_start += delimiter.size();
        if (request.body.compare(part_start, 2, "--") == 0) break;
        if (request.body.compare(part_start, 2, "\r\n") == 0) part_start += 2;

        size_t part_headers_end = request.body.find("\r\n\r\n", part_start);
        if (part_headers_end == std::string::npos) break;
        std::string headers_block = request.body.substr(part_start, part_headers_end - part_start);
        size_t data_start = part_headers_end + 4;
        size_t next_delim = request.body.find("\r\n" + delimiter, data_start);
        if (next_delim == std::string::npos) break;
        std::string part_data = request.body.substr(data_start, next_delim - data_start);

        std::map<std::string, std::string> part_headers;
        std::istringstream iss(headers_block);
        std::string line;
        while (std::getline(iss, line)) {
            line = trim(line);
            size_t colon = line.find(':');
            if (colon == std::string::npos) continue;
            part_headers[to_lower(trim(line.substr(0, colon)))] = trim(line.substr(colon + 1));
        }

        std::string disposition = part_headers["content-disposition"];
        std::string field_name = get_disposition_attr(disposition, "name");
        std::string filename = get_disposition_attr(disposition, "filename");
        if (!filename.empty()) {
            UploadedFile file;
            file.field_name = field_name;
            file.filename = filename;
            file.content_type = part_headers["content-type"];
            file.data.assign(part_data.begin(), part_data.end());
            form.files.push_back(std::move(file));
        } else {
            form.fields[field_name] = part_data;
        }
        pos = next_delim + 2;
    }
    return form;
}

static std::string make_job_id() {
    unsigned long long counter = ++g_job_counter;
    auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::to_string((unsigned long long)stamp) + "_" + std::to_string(counter);
}

class ServerState {
public:
    explicit ServerState(ServerOptions opts)
        : options_(std::move(opts)), model_dir_(fs::weakly_canonical(options_.model_dir)) {
        models_ = scan_models(model_dir_);
        upload_dir_ = fs::current_path() / ".cudasep_http_uploads";
        fs::create_directories(upload_dir_);
    }

    const std::vector<ModelEntry>& models() const { return models_; }
    const fs::path& model_dir() const { return model_dir_; }
    const ServerOptions& options() const { return options_; }

    LoadedModel& get_model(const std::string& relative_name, bool quantize_fp16, LogCallback logger = nullptr) {
        fs::path resolved = resolve_model_path(relative_name);
        std::string key = resolved.generic_string() + (quantize_fp16 ? "|fp16" : "|fp32");
        if (!cached_model_ || cached_model_path_ != key) {
            cached_model_ = std::make_unique<LoadedModel>(load_model(resolved.string(), options_.device, quantize_fp16, logger));
            cached_model_path_ = key;
        } else if (logger) {
            logger("[模型] 复用已缓存模型");
        }
        return *cached_model_;
    }

    fs::path resolve_model_path(const std::string& relative_name) const {
        fs::path candidate = fs::weakly_canonical(model_dir_ / fs::path(relative_name));
        std::string candidate_str = candidate.generic_string();
        std::string base_str = model_dir_.generic_string();
        if (candidate_str.rfind(base_str, 0) != 0 || !fs::exists(candidate) || candidate.extension() != ".csm") {
            throw std::runtime_error("模型路径无效");
        }
        return candidate;
    }

    std::shared_ptr<JobState> create_job(const std::string& model_name, float overlap, bool quantize_fp16,
                                         bool enable_cuda_graph, bool keep_model_loaded,
                                         int chunk_batch_size, const UploadedFile& file) {
        auto job = std::make_shared<JobState>();
        job->id = make_job_id();
        job->model_name = model_name;
        job->original_filename = file.filename;
        job->sanitized_base = base_name_of(file.filename);
        job->quantize_fp16 = quantize_fp16;
        job->enable_cuda_graph = enable_cuda_graph;
        job->keep_model_loaded = keep_model_loaded;
        job->chunk_batch_size = chunk_batch_size;
        {
            std::lock_guard<std::mutex> lock(job->mutex);
            job->logs.push_back("任务已创建，等待进入推理队列");
        }
        {
            std::lock_guard<std::mutex> lock(jobs_mutex_);
            jobs_[job->id] = job;
        }
        std::thread([this, job, overlap, file]() { run_job(job, overlap, file); }).detach();
        return job;
    }

    std::shared_ptr<JobState> find_job(const std::string& id) const {
        std::lock_guard<std::mutex> lock(jobs_mutex_);
        auto it = jobs_.find(id);
        return it == jobs_.end() ? nullptr : it->second;
    }

    std::string job_json(const std::shared_ptr<JobState>& job) const {
        std::lock_guard<std::mutex> lock(job->mutex);
        auto now = std::chrono::steady_clock::now();
        auto end = job->finished ? job->finished_at : now;
        long long elapsed_ms = job->started ? std::chrono::duration_cast<std::chrono::milliseconds>(end - job->started_at).count() : 0;

        std::ostringstream json;
        json << "{";
        json << "\"id\":\"" << json_escape(job->id) << "\",";
        json << "\"model\":\"" << json_escape(job->model_name) << "\",";
        json << "\"fp16\":" << (job->quantize_fp16 ? "true" : "false") << ",";
        json << "\"precision_mode\":\"" << (job->quantize_fp16 ? "mixed" : "native") << "\",";
        json << "\"cuda_graph\":" << (job->enable_cuda_graph ? "true" : "false") << ",";
        json << "\"keep_model_loaded\":" << (job->keep_model_loaded ? "true" : "false") << ",";
        json << "\"chunk_batch_size\":" << job->chunk_batch_size << ",";
        json << "\"filename\":\"" << json_escape(job->original_filename) << "\",";
        json << "\"status\":\"" << json_escape(job->status) << "\",";
        json << "\"progress\":" << job->progress << ",";
        json << "\"elapsed_ms\":" << elapsed_ms << ",";
        json << "\"error\":\"" << json_escape(job->error) << "\",";
        json << "\"logs\":[";
        for (size_t i = 0; i < job->logs.size(); ++i) {
            if (i) json << ',';
            json << "\"" << json_escape(job->logs[i]) << "\"";
        }
        json << "],\"tracks\":[";
        for (size_t i = 0; i < job->tracks.size(); ++i) {
            if (i) json << ',';
            json << "{";
            json << "\"index\":" << i << ",";
            json << "\"name\":\"" << json_escape(job->tracks[i].name) << "\",";
            json << "\"derived\":" << (job->tracks[i].derived ? "true" : "false") << ",";
            json << "\"url\":\"/api/jobs/" << json_escape(job->id) << "/tracks/" << i << "\"";
            json << "}";
        }
        json << "]}";
        return json.str();
    }

    HttpResponse track_response(const std::shared_ptr<JobState>& job, size_t index, const HttpRequest& request) const {
        std::lock_guard<std::mutex> lock(job->mutex);
        if (!job->finished || job->status != "已完成") {
            return text_response(400, "任务尚未完成", "text/plain; charset=utf-8");
        }
        if (index >= job->tracks.size()) {
            return text_response(404, "找不到对应轨道", "text/plain; charset=utf-8");
        }

        const std::vector<uint8_t>& wav = job->tracks[index].wav_bytes;
        HttpResponse res;
        res.content_type = "audio/wav";
        res.headers.push_back({"Accept-Ranges", "bytes"});
        res.headers.push_back({"Content-Disposition", "inline; filename=\"" + sanitize_ascii(job->sanitized_base) + "_" + sanitize_ascii(job->tracks[index].name) + ".wav\""});

        ByteRange range = parse_range_header(header_value(request, "range"), wav.size());
        if (!header_value(request, "range").empty() && !range.valid) {
            res.status = 416;
            res.headers.push_back({"Content-Range", "bytes */" + std::to_string(wav.size())});
            res.body.clear();
            return res;
        }

        if (range.valid) {
            res.status = 206;
            res.body.assign(wav.begin() + (ptrdiff_t)range.start, wav.begin() + (ptrdiff_t)range.end + 1);
            res.headers.push_back({"Content-Range", "bytes " + std::to_string(range.start) + "-" + std::to_string(range.end) + "/" + std::to_string(wav.size())});
        } else {
            res.status = 200;
            res.body = wav;
        }
        return res;
    }

private:
    void trim_runtime_caches() {
        cudasep::clear_attention_graph_cache();
        CudaMemoryPool::instance().trim_cached_memory();
    }

    void unload_cached_model_and_trim() {
        CudaContext::instance().sync();
        cached_model_.reset();
        cached_model_path_.clear();
        trim_runtime_caches();
    }

    void set_state(const std::shared_ptr<JobState>& job, int progress, const std::string& status, const std::string& log_line) {
        std::lock_guard<std::mutex> lock(job->mutex);
        if (!job->started) {
            job->started = true;
            job->started_at = std::chrono::steady_clock::now();
        }
        job->progress = progress;
        job->status = status;
        if (!log_line.empty()) job->logs.push_back(log_line);
    }

    void finish_success(const std::shared_ptr<JobState>& job, std::vector<TrackAsset> tracks) {
        std::lock_guard<std::mutex> lock(job->mutex);
        job->tracks = std::move(tracks);
        job->status = "已完成";
        job->progress = 100;
        job->logs.push_back("所有轨道已经生成完成，可以试听和下载了");
        job->finished = true;
        job->finished_at = std::chrono::steady_clock::now();
    }

    void finish_error(const std::shared_ptr<JobState>& job, const std::string& error) {
        std::lock_guard<std::mutex> lock(job->mutex);
        job->status = "失败";
        job->error = error;
        job->logs.push_back(error);
        job->finished = true;
        job->finished_at = std::chrono::steady_clock::now();
    }

    void run_job(const std::shared_ptr<JobState>& job, float overlap, UploadedFile file) {
        fs::path upload_path;
        try {
            set_state(job, 5, "运行中", "[准备] 正在整理上传文件");
            upload_path = upload_dir_ / ("job_" + job->id + extension_of(file.filename));
            {
                std::ofstream out(upload_path, std::ios::binary);
                out.write(reinterpret_cast<const char*>(file.data.data()), (std::streamsize)file.data.size());
            }

            set_state(job, 15, "运行中", "[准备] 上传文件已写入临时目录");

            std::vector<TrackAsset> assets;
            {
                std::lock_guard<std::mutex> infer_lock(infer_mutex_);
                cudasep::g_enable_cuda_graph_attention = job->enable_cuda_graph;
                if (!job->enable_cuda_graph) {
                    trim_runtime_caches();
                }
                set_state(job, 25, "运行中", "[模型] 正在加载分离模型");
                LoadedModel& model = get_model(
                    job->model_name,
                    job->quantize_fp16,
                    [job](const std::string& line) {
                        std::lock_guard<std::mutex> lock(job->mutex);
                        job->logs.push_back(line);
                    }
                );
                cudasep::g_quantize_fp16 = job->quantize_fp16;
                model.chunk_batch_size = job->chunk_batch_size > 0 ? job->chunk_batch_size : (std::max)(1, model.chunk_batch_size);
                model.detailed_logger = nullptr;
                {
                    std::lock_guard<std::mutex> lock(job->mutex);
                    job->logs.push_back("[配置] 精度策略: " + precision_mode_title(job->quantize_fp16));
                    job->logs.push_back("[配置] 精度说明: " + precision_mode_description(job->quantize_fp16));
                    job->logs.push_back(std::string("[配置] CUDA Graph: ") + (job->enable_cuda_graph ? "开启（更快，但更占显存）" : "关闭（降低常驻显存）"));
                    job->logs.push_back(std::string("[配置] 模型保持加载: ") + (job->keep_model_loaded ? "开启（任务后继续驻留显存）" : "关闭（任务后释放模型与缓存）"));
                    job->logs.push_back("[配置] 分片批大小: " + std::to_string(model.chunk_batch_size));
                }

                set_state(job, 40, "运行中", "[音频] 正在解析输入音频");
                set_state(job, 55, "运行中", "[推理] 正在执行模型推理");
                InferenceResult result = run_inference(
                    model,
                    upload_path.string(),
                    overlap,
                    [job](const std::string& line) {
                        std::lock_guard<std::mutex> lock(job->mutex);
                        job->logs.push_back(line);
                    }
                );

                set_state(job, 80, "运行中", "[输出] 正在整理输出轨道并补算 other");
                auto collect_start = std::chrono::high_resolution_clock::now();
                std::vector<OutputTrack> tracks = collect_output_tracks(model, result.audio, result.output);
                {
                    std::lock_guard<std::mutex> lock(job->mutex);
                    job->logs.push_back("[耗时] 轨道整理: " + std::to_string((int)std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - collect_start).count()) + " ms");
                }

                auto encode_start = std::chrono::high_resolution_clock::now();
                for (const auto& track : tracks) {
                    TrackAsset asset;
                    asset.name = track.name;
                    asset.derived = track.derived;
                    asset.wav_bytes = encode_wav_bytes(track.audio, model.sample_rate);
                    assets.push_back(std::move(asset));
                }
                {
                    std::lock_guard<std::mutex> lock(job->mutex);
                    job->logs.push_back("[耗时] WAV 编码: " + std::to_string((int)std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - encode_start).count()) + " ms");
                }

                if (!job->keep_model_loaded) {
                    unload_cached_model_and_trim();
                    std::lock_guard<std::mutex> lock(job->mutex);
                    job->logs.push_back("[模型] 已按请求释放缓存模型、CUDA Graph 和空闲显存缓存");
                }
            }

            if (!upload_path.empty()) {
                std::error_code ec;
                fs::remove(upload_path, ec);
            }
            finish_success(job, std::move(assets));
        } catch (const std::exception& e) {
            if (!job->keep_model_loaded) {
                std::lock_guard<std::mutex> infer_lock(infer_mutex_);
                unload_cached_model_and_trim();
            }
            if (!upload_path.empty()) {
                std::error_code ec;
                fs::remove(upload_path, ec);
            }
            finish_error(job, std::string("推理失败: ") + e.what());
        }
    }

    ServerOptions options_;
    fs::path model_dir_;
    fs::path upload_dir_;
    std::vector<ModelEntry> models_;
    std::unique_ptr<LoadedModel> cached_model_;
    std::string cached_model_path_;
    mutable std::mutex jobs_mutex_;
    mutable std::mutex infer_mutex_;
    std::map<std::string, std::shared_ptr<JobState>> jobs_;
};

static std::string replace_all(std::string str, const std::string& from, const std::string& to) {
    size_t pos = 0;
    while ((pos = str.find(from, pos)) != std::string::npos) {
        str.replace(pos, from.length(), to);
        pos += to.length();
    }
    return str;
}

static std::string render_index(const ServerState& state) {
    const bool default_mixed_precision = state.options().quantize_fp16;
    const bool default_cuda_graph = state.options().enable_cuda_graph;
    const bool default_keep_model_loaded = state.options().keep_model_loaded;

    // Build model options
    std::ostringstream model_options;
    for (const auto& model : state.models()) {
        model_options << "<option value=\"" << html_escape(model.relative_name) << "\">" << html_escape(model.relative_name) << "</option>";
    }

    // Build precision options
    std::ostringstream precision_options;
    precision_options << "<option value=\"native\"" << (default_mixed_precision ? "" : " selected") << ">遵循模型文件原始 dtype</option>";
    precision_options << "<option value=\"mixed\"" << (default_mixed_precision ? " selected" : "") << ">启用混精加速</option>";

    // Build CUDA graph options
    std::ostringstream cuda_graph_options;
    cuda_graph_options << "<option value=\"0\"" << (default_cuda_graph ? "" : " selected") << ">关闭</option>";
    cuda_graph_options << "<option value=\"1\"" << (default_cuda_graph ? " selected" : "") << ">开启</option>";

    // Build keep model options
    std::ostringstream keep_model_options;
    keep_model_options << "<option value=\"0\"" << (default_keep_model_loaded ? "" : " selected") << ">任务后释放模型与缓存</option>";
    keep_model_options << "<option value=\"1\"" << (default_keep_model_loaded ? " selected" : "") << ">任务后继续驻留显存</option>";

    // Start with embedded HTML template - split into chunks to avoid MSVC C2026
    std::string html = webui::HTML_PART1;
    html += webui::HTML_PART2;
    html += webui::HTML_PART3;
    html += webui::HTML_PART4;
    html += webui::HTML_PART5;

    // Replace placeholders
    html = replace_all(html, "{{MODEL_DIR}}", html_escape(state.model_dir().generic_string()));
    html = replace_all(html, "{{HOST_PORT}}", html_escape(state.options().host) + ':' + std::to_string(state.options().port));
    html = replace_all(html, "{{DEVICE}}", std::to_string(state.options().device));
    html = replace_all(html, "{{PRECISION_TITLE}}", html_escape(precision_mode_title(default_mixed_precision)));
    html = replace_all(html, "{{PRECISION_DESC}}", html_escape(precision_mode_description(default_mixed_precision)));
    html = replace_all(html, "{{CUDA_GRAPH}}", default_cuda_graph ? "开启：更快，但更占显存" : "关闭：不保留额外 graph 工作区");
    html = replace_all(html, "{{KEEP_MODEL}}", default_keep_model_loaded ? "开启：任务后继续驻留显存" : "关闭：任务后释放模型与缓存");
    html = replace_all(html, "{{CHUNK_BATCH}}", state.options().chunk_batch_size > 0 ? std::to_string(state.options().chunk_batch_size) : std::string("模型默认"));
    html = replace_all(html, "{{MODEL_OPTIONS}}", model_options.str());
    html = replace_all(html, "{{PRECISION_OPTIONS}}", precision_options.str());
    html = replace_all(html, "{{CUDA_GRAPH_OPTIONS}}", cuda_graph_options.str());
    html = replace_all(html, "{{KEEP_MODEL_OPTIONS}}", keep_model_options.str());

    return html;
}

static HttpResponse handle_models(const ServerState& state) {
    std::ostringstream json;
    json << "{\"model_dir\":\"" << json_escape(state.model_dir().generic_string()) << "\",\"models\":[";
    for (size_t i = 0; i < state.models().size(); ++i) {
        if (i) json << ',';
        json << "\"" << json_escape(state.models()[i].relative_name) << "\"";
    }
    json << "]}";
    return json_response(200, json.str());
}

static HttpResponse handle_create_job(ServerState& state, const HttpRequest& request) {
    MultipartForm form = parse_multipart(request);
    if (!form.fields.count("model")) return json_response(400, "{\"error\":\"缺少模型字段\"}");

    const UploadedFile* audio_file = nullptr;
    for (const auto& file : form.files) {
        if (file.field_name == "audio") {
            audio_file = &file;
            break;
        }
    }
    if (!audio_file) return json_response(400, "{\"error\":\"缺少音频上传文件\"}");

    float overlap = state.options().overlap;
    bool quantize_fp16 = state.options().quantize_fp16;
    bool enable_cuda_graph = state.options().enable_cuda_graph;
    bool keep_model_loaded = state.options().keep_model_loaded;
    int chunk_batch_size = state.options().chunk_batch_size;
    if (form.fields.count("overlap") && !trim(form.fields["overlap"]).empty()) {
        overlap = std::stof(trim(form.fields["overlap"]));
    }
    if (form.fields.count("precision_mode") && !trim(form.fields["precision_mode"]).empty()) {
        quantize_fp16 = parse_precision_mode(form.fields["precision_mode"], quantize_fp16);
    } else if (form.fields.count("fp16") && !trim(form.fields["fp16"]).empty()) {
        quantize_fp16 = parse_toggle_value(form.fields["fp16"]);
    }
    if (form.fields.count("cuda_graph") && !trim(form.fields["cuda_graph"]).empty()) {
        enable_cuda_graph = parse_toggle_value(form.fields["cuda_graph"]);
    }
    if (form.fields.count("keep_model_loaded") && !trim(form.fields["keep_model_loaded"]).empty()) {
        keep_model_loaded = parse_toggle_value(form.fields["keep_model_loaded"]);
    }
    if (form.fields.count("chunk_batch_size") && !trim(form.fields["chunk_batch_size"]).empty()) {
        chunk_batch_size = (std::max)(1, std::stoi(trim(form.fields["chunk_batch_size"])));
    }

    auto job = state.create_job(form.fields["model"], overlap, quantize_fp16,
                                enable_cuda_graph, keep_model_loaded,
                                chunk_batch_size, *audio_file);
    return json_response(200, state.job_json(job));
}

static HttpResponse handle_job_status(ServerState& state, const std::string& job_id) {
    auto job = state.find_job(job_id);
    if (!job) return json_response(404, "{\"error\":\"任务不存在\"}");
    return json_response(200, state.job_json(job));
}

static HttpResponse handle_track(ServerState& state, const HttpRequest& request, const std::string& job_id, size_t index) {
    auto job = state.find_job(job_id);
    if (!job) return text_response(404, "任务不存在", "text/plain; charset=utf-8");
    return state.track_response(job, index, request);
}

static HttpResponse dispatch(ServerState& state, const HttpRequest& request) {
    if (request.method == "GET" && request.path == "/") {
        return html_response(200, render_index(state));
    }
    if (request.method == "GET" && request.path == "/healthz") {
        return json_response(200, "{\"ok\":true}");
    }
    if (request.method == "GET" && request.path == "/api/models") {
        return handle_models(state);
    }
    if (request.method == "POST" && request.path == "/api/jobs") {
        return handle_create_job(state, request);
    }
    if (request.method == "GET" && request.path.rfind("/api/jobs/", 0) == 0) {
        std::string tail = request.path.substr(10);
        size_t slash = tail.find('/');
        if (slash == std::string::npos) {
            return handle_job_status(state, tail);
        }
        std::string job_id = tail.substr(0, slash);
        std::string rest = tail.substr(slash + 1);
        if (rest.rfind("tracks/", 0) == 0) {
            std::string idx_text = rest.substr(7);
            size_t dot = idx_text.find('.');
            if (dot != std::string::npos) idx_text = idx_text.substr(0, dot);
            return handle_track(state, request, job_id, (size_t)std::stoul(idx_text));
        }
    }
    if (request.method != "GET" && request.method != "POST") {
        return text_response(405, "不支持的请求方法", "text/plain; charset=utf-8");
    }
    return text_response(404, "未找到请求资源", "text/plain; charset=utf-8");
}

static void handle_client(SocketHandle client_sock, ServerState& state, size_t max_upload_bytes) {
    SocketGuard client(client_sock);
    if (client.sock == kInvalidSocket) return;

    try {
        HttpRequest request;
        HttpResponse error_response;
        if (!recv_request(client.sock, max_upload_bytes, request, error_response)) {
            send_response(client.sock, error_response);
            return;
        }
        HttpResponse response = dispatch(state, request);
        send_response(client.sock, response);
    } catch (const std::exception& e) {
        send_response(client.sock, text_response(500, std::string("服务器错误: ") + e.what(), "text/plain; charset=utf-8"));
    }
}

}  // namespace

int run_http_server(const ServerOptions& options) {
    SocketApi socket_api;
    ServerState state(options);

    SocketGuard server(socket(AF_INET, SOCK_STREAM, IPPROTO_TCP));
    if (server.sock == kInvalidSocket) {
        throw std::runtime_error("Failed to create server socket");
    }

    int reuse = 1;
#ifdef _WIN32
    setsockopt(server.sock, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&reuse), sizeof(reuse));
#else
    setsockopt(server.sock, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
#endif

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)options.port);
    if (inet_pton(AF_INET, options.host.c_str(), &addr.sin_addr) != 1) {
        throw std::runtime_error("Invalid host address");
    }

    if (bind(server.sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        throw std::runtime_error("Failed to bind HTTP server socket");
    }
    if (listen(server.sock, 16) != 0) {
        throw std::runtime_error("Failed to listen on HTTP server socket");
    }

    std::cout << "HTTP server listening on http://" << options.host << ':' << options.port << std::endl;
    std::cout << "Model directory: " << state.model_dir().string() << std::endl;
    std::cout << "Found " << state.models().size() << " model(s)" << std::endl;

    while (true) {
        sockaddr_in client_addr{};
#ifdef _WIN32
        int client_len = sizeof(client_addr);
#else
        socklen_t client_len = sizeof(client_addr);
#endif
        SocketHandle client_sock = accept(server.sock, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_sock == kInvalidSocket) continue;
        std::thread([client_sock, &state, max_upload_bytes = options.max_upload_bytes]() {
            handle_client(client_sock, state, max_upload_bytes);
        }).detach();
    }

    return 0;
}

}  // namespace cudasep::app
