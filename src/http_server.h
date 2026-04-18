#pragma once

#include <cstddef>
#include <string>

namespace cudasep::app {

struct ServerOptions {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string model_dir;
    int device = 0;
    float overlap = -1.0f;
    bool quantize_fp16 = false;
    size_t max_upload_bytes = 200 * 1024 * 1024;
};

int run_http_server(const ServerOptions& options);

}  // namespace cudasep::app
