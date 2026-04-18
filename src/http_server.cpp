#include "http_server.h"

#include "audio_io.h"
#include "inference_app.h"

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
    if (boundary_pos == std::string::npos) throw std::runtime_error("Missing multipart boundary");
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

    LoadedModel& get_model(const std::string& relative_name) {
        fs::path resolved = resolve_model_path(relative_name);
        std::string key = resolved.generic_string();
        if (!cached_model_ || cached_model_path_ != key) {
            cached_model_ = std::make_unique<LoadedModel>(load_model(resolved.string(), options_.device, options_.quantize_fp16));
            cached_model_path_ = key;
        }
        return *cached_model_;
    }

    fs::path resolve_model_path(const std::string& relative_name) const {
        fs::path candidate = fs::weakly_canonical(model_dir_ / fs::path(relative_name));
        std::string candidate_str = candidate.generic_string();
        std::string base_str = model_dir_.generic_string();
        if (candidate_str.rfind(base_str, 0) != 0 || !fs::exists(candidate) || candidate.extension() != ".csm") {
            throw std::runtime_error("Invalid model path");
        }
        return candidate;
    }

    std::shared_ptr<JobState> create_job(const std::string& model_name, float overlap, const UploadedFile& file) {
        auto job = std::make_shared<JobState>();
        job->id = make_job_id();
        job->model_name = model_name;
        job->original_filename = file.filename;
        job->sanitized_base = base_name_of(file.filename);
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
                set_state(job, 25, "运行中", "[模型] 正在加载分离模型");
                LoadedModel& model = get_model(job->model_name);

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
                std::vector<OutputTrack> tracks = collect_output_tracks(model, result.audio, result.output);
                for (const auto& track : tracks) {
                    TrackAsset asset;
                    asset.name = track.name;
                    asset.derived = track.derived;
                    asset.wav_bytes = encode_wav_bytes(track.audio, model.sample_rate);
                    assets.push_back(std::move(asset));
                }
            }

            if (!upload_path.empty()) {
                std::error_code ec;
                fs::remove(upload_path, ec);
            }
            finish_success(job, std::move(assets));
        } catch (const std::exception& e) {
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

static std::string render_index(const ServerState& state) {
    std::ostringstream html;
    html << R"HTML(<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CudaInfer Server</title>
<style>
:root{--bg:#f5efe5;--ink:#1b2630;--muted:#5d6c78;--card:#fffaf4;--accent:#d16b3f;--accent2:#264653;--line:#dfd2c4}
*{box-sizing:border-box}
body{margin:0;font-family:'Segoe UI',Arial,sans-serif;background:radial-gradient(circle at top,#fff7ec,#f0e3d2 42%,#e6dfd7 100%);color:var(--ink)}
.shell{max-width:1100px;margin:0 auto;padding:28px 20px 40px}
.hero{display:grid;grid-template-columns:1.2fr .8fr;gap:18px}
.panel{background:rgba(255,250,244,.92);backdrop-filter:blur(8px);border:1px solid rgba(209,107,63,.12);border-radius:24px;box-shadow:0 20px 50px rgba(38,70,83,.10)}
.lead{padding:28px}.lead h1{margin:0 0 12px;font-size:38px;line-height:1.02}.lead p{margin:0;color:var(--muted);font-size:15px;line-height:1.6}
.stats{padding:24px;display:grid;gap:12px;align-content:start}.stat{padding:14px 16px;border-radius:16px;background:#fff;border:1px solid var(--line)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:18px}.section{padding:22px}.section h2{margin:0 0 14px;font-size:18px}
.field{display:grid;gap:8px;margin-bottom:14px}.field label{font-weight:700;font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
input,select,button{width:100%;padding:13px 14px;border-radius:14px;border:1px solid #d7c8b7;background:#fff;font-size:14px}
button{background:linear-gradient(135deg,var(--accent),#e08f52);color:#fff;border:none;font-weight:800;cursor:pointer}
button:disabled{opacity:.6;cursor:wait}
.progress{height:12px;background:#eadbcc;border-radius:999px;overflow:hidden;border:1px solid #deceb9}
.progress>div{height:100%;width:0;background:linear-gradient(90deg,var(--accent2),var(--accent));transition:width .35s ease}
.meta{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0 0}.pill{padding:8px 12px;border-radius:999px;background:#fff;border:1px solid var(--line);font-size:13px}
.logs{margin-top:14px;background:#18232b;color:#e8f0f4;border-radius:16px;padding:14px;height:260px;overflow:auto;font:12px/1.55 Consolas,monospace;display:grid;gap:8px}
.log-line{padding:8px 10px;border-radius:10px;white-space:pre-wrap;word-break:break-word;border:1px solid rgba(255,255,255,.06)}
.log-info{background:rgba(38,70,83,.45);border-color:rgba(120,180,200,.25)}
.log-stage{background:rgba(209,107,63,.16);border-color:rgba(240,180,120,.25);color:#ffe4c7}
.log-chunk{background:rgba(79,109,122,.22);border-color:rgba(160,220,255,.18);color:#d7f0ff}
.log-ok{background:rgba(46,125,90,.24);border-color:rgba(120,230,170,.18);color:#d7ffe9}
.log-warn{background:rgba(180,120,20,.18);border-color:rgba(255,220,120,.2);color:#fff0bf}
.log-error{background:rgba(140,48,48,.28);border-color:rgba(255,120,120,.25);color:#ffd8d8}
.tracks{display:grid;gap:14px}.track{padding:14px;border:1px solid var(--line);border-radius:16px;background:#fff}.track h3{display:flex;justify-content:space-between;gap:8px;margin:0 0 10px;font-size:16px}
.tag{font-size:12px;color:#8a4f2c;background:#fde9da;border:1px solid #f0c8ab;border-radius:999px;padding:3px 8px}.hint{font-size:13px;color:var(--muted)}audio{width:100%;margin:10px 0 8px}
@media (max-width:900px){.hero,.grid{grid-template-columns:1fr}.lead h1{font-size:30px}}
</style>
</head>
<body>
<div class="shell">
  <div class="hero">
    <section class="panel lead">
      <h1>CudaInfer Web Test Bench</h1>
      <p>上传音频后，前端会立刻开始计时，持续显示推理进度、阶段日志和分片日志，并在完成后直接展示全部输出轨道。如果模型本身没有 <code>other</code>，服务器会自动用 <code>原混音 - 所有已分离轨道之和</code> 补出一条 <code>other</code>。</p>
    </section>
    <aside class="panel stats">
      <div class="stat"><strong>Model Dir</strong><div class="hint">)HTML";
    html << html_escape(state.model_dir().generic_string());
    html << R"HTML(</div></div>
      <div class="stat"><strong>HTTP</strong><div class="hint">)HTML";
    html << html_escape(state.options().host) << ':' << state.options().port;
    html << R"HTML(</div></div>
      <div class="stat"><strong>GPU Device</strong><div class="hint">CUDA )HTML";
    html << state.options().device << (state.options().quantize_fp16 ? " - FP16" : " - FP32");
    html << R"HTML(</div></div>
    </aside>
  </div>
  <div class="grid">
    <section class="panel section">
      <h2>Start Inference</h2>
      <form id="infer-form">
        <div class="field">
          <label>Model</label>
          <select id="model" name="model">)HTML";

    for (const auto& model : state.models()) {
        html << "<option value=\"" << html_escape(model.relative_name) << "\">" << html_escape(model.relative_name) << "</option>";
    }

    html << R"HTML(</select>
        </div>
        <div class="field">
          <label>Audio File</label>
          <input id="audio" name="audio" type="file" accept="audio/*" required>
        </div>
        <div class="field">
          <label>Overlap Override</label>
          <input id="overlap" name="overlap" type="number" step="0.01" placeholder="Leave blank for model default">
        </div>
        <button id="submit" type="submit">开始推理</button>
      </form>
      <div class="meta">
        <div class="pill" id="status-pill">空闲</div>
        <div class="pill" id="timer-pill">00:00</div>
        <div class="pill" id="progress-pill">0%</div>
      </div>
      <div style="margin-top:14px" class="progress"><div id="progress-bar"></div></div>
      <div class="logs" id="logs"></div>
    </section>
    <section class="panel section">
      <h2>Output Tracks</h2>
      <div id="tracks" class="tracks"><div class="hint">还没有输出结果。任务完成后，这里会直接出现所有轨道的试听和下载按钮。</div></div>
    </section>
  </div>
</div>
<script>
const form = document.getElementById('infer-form');
const submit = document.getElementById('submit');
const logs = document.getElementById('logs');
const tracks = document.getElementById('tracks');
const statusPill = document.getElementById('status-pill');
const timerPill = document.getElementById('timer-pill');
const progressPill = document.getElementById('progress-pill');
const progressBar = document.getElementById('progress-bar');
let timer = null;
let poller = null;
let startedAt = 0;

function fmt(ms) {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  const r = s % 60;
  return String(m).padStart(2, '0') + ':' + String(r).padStart(2, '0');
}

function setState(status, progress) {
  statusPill.textContent = status;
  progressPill.textContent = progress + '%';
  progressBar.style.width = progress + '%';
}

function classifyLog(line) {
  if (line.includes('失败') || line.includes('error') || line.includes('Error')) return 'log-error';
  if (line.includes('完成') || line.includes('就绪')) return 'log-ok';
  if (line.includes('分片')) return 'log-chunk';
  if (line.startsWith('[') || line.startsWith('-')) return 'log-stage';
  return 'log-info';
}

function setLogs(lines) {
  const list = lines && lines.length ? lines : ['等待任务日志...'];
  logs.innerHTML = list.map(line => '<div class="log-line ' + classifyLog(line) + '">' + line.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;') + '</div>').join('');
  logs.scrollTop = logs.scrollHeight;
}

function renderTracks(data) {
  if (!data.tracks || !data.tracks.length) {
    tracks.innerHTML = '<div class="hint">任务完成了，但没有返回任何轨道。</div>';
    return;
  }
  tracks.innerHTML = '';
  for (const track of data.tracks) {
    const card = document.createElement('div');
    card.className = 'track';
    card.innerHTML = '<h3><span>' + track.name + '</span>' + (track.derived ? '<span class="tag">derived other</span>' : '') + '</h3>' +
      '<audio controls preload="none" src="' + track.url + '"></audio>' +
      '<a href="' + track.url + '" download>下载 WAV</a>';
    tracks.appendChild(card);
  }
}

function startTimer() {
  startedAt = Date.now();
  clearInterval(timer);
  timer = setInterval(() => {
    timerPill.textContent = fmt(Date.now() - startedAt);
  }, 250);
}

function stopTimer() {
  clearInterval(timer);
  timer = null;
}

async function pollJob(id) {
  clearInterval(poller);
  poller = setInterval(async () => {
    try {
      const res = await fetch('/api/jobs/' + id);
      const data = await res.json();
      setState(data.status, data.progress || 0);
      setLogs(data.logs || []);
      if (data.elapsed_ms != null) {
        timerPill.textContent = fmt(data.elapsed_ms);
      }
      if (data.status === '已完成') {
        stopTimer();
        clearInterval(poller);
        submit.disabled = false;
        submit.textContent = '开始推理';
        renderTracks(data);
      } else if (data.status === '失败') {
        stopTimer();
        clearInterval(poller);
        submit.disabled = false;
        submit.textContent = '开始推理';
        tracks.innerHTML = '<div class="hint">' + (data.error || '推理失败') + '</div>';
      }
    } catch (err) {
      stopTimer();
      clearInterval(poller);
      submit.disabled = false;
      submit.textContent = '开始推理';
      setLogs(['[轮询] 获取任务状态失败: ' + err.message]);
    }
  }, 700);
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(form);
  if (!fd.get('audio') || !fd.get('model')) return;
  submit.disabled = true;
  submit.textContent = '上传中...';
  tracks.innerHTML = '<div class="hint">正在等待输出轨道...</div>';
  setLogs(['[准备] 正在上传音频并创建任务...']);
  setState('上传中', 2);
  startTimer();
  try {
    const res = await fetch('/api/jobs', { method: 'POST', body: fd });
    const text = await res.text();
    let data = {};
    try { data = JSON.parse(text); } catch (_) {}
    if (!res.ok) {
      throw new Error(data.error || text || '请求失败');
    }
    submit.textContent = '推理中...';
    setState(data.status || '排队中', data.progress || 5);
    setLogs(data.logs || ['[队列] 任务已进入等待队列']);
    pollJob(data.id);
  } catch (err) {
    stopTimer();
    submit.disabled = false;
    submit.textContent = '开始推理';
    setState('失败', 0);
    setLogs(['[请求] 创建任务失败: ' + err.message]);
}
});

setLogs(['[空闲] 请选择模型并上传音频文件']);
</script>
</body>
</html>)HTML";

    return html.str();
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
    if (!form.fields.count("model")) return json_response(400, "{\"error\":\"Missing model field\"}");

    const UploadedFile* audio_file = nullptr;
    for (const auto& file : form.files) {
        if (file.field_name == "audio") {
            audio_file = &file;
            break;
        }
    }
    if (!audio_file) return json_response(400, "{\"error\":\"Missing audio upload\"}");

    float overlap = state.options().overlap;
    if (form.fields.count("overlap") && !trim(form.fields["overlap"]).empty()) {
        overlap = std::stof(trim(form.fields["overlap"]));
    }

    auto job = state.create_job(form.fields["model"], overlap, *audio_file);
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
        send_response(client.sock, text_response(500, std::string("Server error: ") + e.what(), "text/plain; charset=utf-8"));
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
