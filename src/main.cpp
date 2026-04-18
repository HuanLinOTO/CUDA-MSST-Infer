#include "http_server.h"
#include "inference_app.h"

#include <cuda_runtime.h>

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Args {
    std::string model_path;
    std::string input_path;
    std::string output_path = "output";
    int stem = 0;
    float overlap = -1.0f;
    int device = 0;
    bool list_stems = false;
    bool help = false;
    bool quantize_fp16 = false;
    bool deep_profile = false;
    int chunk_batch_size = 0;
    bool server_mode = false;
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string model_dir;
    int max_upload_mb = 200;
};

static Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--model" || a == "-m") && i + 1 < argc) {
            args.model_path = argv[++i];
        } else if ((a == "--input" || a == "-i") && i + 1 < argc) {
            args.input_path = argv[++i];
        } else if ((a == "--output" || a == "-o") && i + 1 < argc) {
            args.output_path = argv[++i];
        } else if ((a == "--stem" || a == "-s") && i + 1 < argc) {
            args.stem = std::stoi(argv[++i]);
        } else if (a == "--overlap" && i + 1 < argc) {
            args.overlap = std::stof(argv[++i]);
        } else if ((a == "--device" || a == "-d") && i + 1 < argc) {
            args.device = std::stoi(argv[++i]);
        } else if (a == "--list-stems") {
            args.list_stems = true;
        } else if (a == "--quantize" || a == "--fp16") {
            args.quantize_fp16 = true;
        } else if (a == "--no-fp16" || a == "--fp32") {
            args.quantize_fp16 = false;
        } else if (a == "--deep-profile") {
            args.deep_profile = true;
        } else if (a == "--chunk-batch-size" && i + 1 < argc) {
            args.chunk_batch_size = std::stoi(argv[++i]);
        } else if (a == "--serve") {
            args.server_mode = true;
        } else if (a == "--host" && i + 1 < argc) {
            args.host = argv[++i];
        } else if (a == "--port" && i + 1 < argc) {
            args.port = std::stoi(argv[++i]);
        } else if (a == "--model-dir" && i + 1 < argc) {
            args.model_dir = argv[++i];
        } else if (a == "--max-upload-mb" && i + 1 < argc) {
            args.max_upload_mb = std::stoi(argv[++i]);
        } else if (a == "--help" || a == "-h") {
            args.help = true;
        }
    }
    return args;
}

static void print_usage(const char* progname) {
    std::cout << "CudaInfer - GPU-accelerated music source separation\n\n"
              << "CLI mode:\n"
              << "  " << progname << " --model <path.csm> --input <audio> [options]\n\n"
              << "Server mode:\n"
              << "  " << progname << " --serve --model-dir <dir> [options]\n\n"
              << "Common options:\n"
              << "  --device, -d <int>     CUDA device ID (default: 0)\n"
              << "  --quantize, --fp16     Enable FP16 mixed-precision GEMM\n"
              << "  --no-fp16, --fp32      Disable FP16 mixed-precision GEMM\n"
              << "  --chunk-batch-size <n> Override chunk inference batch size\n"
              << "  --deep-profile         Print model-internal stage timings\n"
              << "  --help, -h             Show this help message\n\n"
              << "CLI options:\n"
              << "  --model, -m <path>     Path to .csm model weights file\n"
              << "  --input, -i <path>     Path to input audio file\n"
              << "  --output, -o <path>    Output directory or WAV file path (default: output)\n"
              << "  --stem, -s <int>       Stem index to extract (default: 0, -1 for all)\n"
              << "  --overlap <float>      Overlap override. (0,1)=ratio, >=1=num_overlap\n"
              << "  --list-stems           Print stem info from model config and exit\n\n"
              << "Server options:\n"
              << "  --serve                Start embedded HTTP server\n"
              << "  --model-dir <dir>      Directory scanned recursively for .csm files\n"
              << "  --host <ip>            HTTP bind address (default: 127.0.0.1)\n"
              << "  --port <int>           HTTP port (default: 8080)\n"
              << "  --max-upload-mb <int>  Max upload size in MB (default: 200)\n"
              << std::endl;
}

static void print_gpu_info(int device) {
    cudaSetDevice(device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "GPU: " << prop.name << " (SM " << prop.major << '.' << prop.minor
              << ", " << (prop.totalGlobalMem / (1024 * 1024)) << " MB)" << std::endl;
}

static std::string default_model_dir(const Args& args) {
    if (!args.model_dir.empty()) return args.model_dir;
    if (!args.model_path.empty()) return fs::path(args.model_path).parent_path().string();
    if (fs::exists("csm_models") && fs::is_directory("csm_models")) return "csm_models";
    return ".";
}

static int run_cli(const Args& args) {
    if (args.model_path.empty()) {
        std::cerr << "Error: --model is required" << std::endl;
        return 1;
    }

    std::cout << "Loading model: " << args.model_path << std::endl;
    cudasep::app::LogCallback cli_logger = [](const std::string& line) {
        std::cout << "  " << line << std::endl;
    };
    cudasep::app::LoadedModel model = cudasep::app::load_model(args.model_path, args.device, args.quantize_fp16, cli_logger);
    model.detailed_logger = args.deep_profile ? cli_logger : nullptr;
    if (args.chunk_batch_size > 0) {
        model.chunk_batch_size = args.chunk_batch_size;
        cli_logger("[配置] 分片批大小覆盖: " + std::to_string(model.chunk_batch_size));
    }
    std::cout << "  Model type: " << cudasep::app::model_type_name(model.type) << std::endl;
    std::cout << "  Sources: " << model.num_sources
              << ", Sample rate: " << model.sample_rate
              << ", Chunk size: " << model.chunk_size
              << ", Num overlap: " << model.num_overlap << std::endl;

    if (args.list_stems) {
        std::cout << "\nModel has " << model.num_sources << " source(s)." << std::endl;
        for (int i = 0; i < (int)model.stem_names.size(); ++i) {
            std::cout << "  [" << i << "] " << model.stem_names[i] << std::endl;
        }
        return 0;
    }

    if (args.input_path.empty()) {
        std::cerr << "Error: --input is required" << std::endl;
        return 1;
    }

    std::cout << "\nLoading audio: " << args.input_path << std::endl;
    cudasep::app::InferenceResult result = cudasep::app::run_inference(model, args.input_path, args.overlap, cli_logger);

    std::cout << "  " << result.audio.channels << "ch, " << result.audio.sample_rate << " Hz, "
              << result.audio.num_samples << " samples ("
              << std::fixed << std::setprecision(1)
              << (double)result.audio.num_samples / result.audio.sample_rate << "s)" << std::endl;
    if (result.audio.sample_rate != model.sample_rate) {
        std::cerr << "Warning: audio sample rate (" << result.audio.sample_rate
                  << ") != model sample rate (" << model.sample_rate << "). Results may be incorrect." << std::endl;
    }

    std::cout << "\nRunning inference..." << std::endl;
    std::cout << "  Inference time: " << std::fixed << std::setprecision(1)
              << result.infer_ms << " ms (RTF: " << std::setprecision(2) << result.rtf << "x)" << std::endl;

    std::cout << "  Output shape: [";
    for (int i = 0; i < result.output.ndim(); ++i) {
        if (i) std::cout << ", ";
        std::cout << result.output.size(i);
    }
    std::cout << "]" << std::endl;

    std::vector<fs::path> saved = cudasep::app::save_outputs(model, result.output, fs::path(args.output_path), args.stem);
    for (const auto& path : saved) {
        std::cout << "Saving: " << path.string() << std::endl;
    }
    std::cout << "Done." << std::endl;
    return 0;
}

static int run_server(const Args& args) {
    cudasep::app::ServerOptions options;
    options.host = args.host;
    options.port = args.port;
    options.model_dir = default_model_dir(args);
    options.device = args.device;
    options.overlap = args.overlap;
    options.quantize_fp16 = args.quantize_fp16;
    options.chunk_batch_size = args.chunk_batch_size;
    options.max_upload_bytes = (size_t)std::max(1, args.max_upload_mb) * 1024ull * 1024ull;
    return cudasep::app::run_http_server(options);
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    if (args.help) {
        print_usage(argv[0]);
        return 0;
    }

    try {
        if (args.server_mode) {
            return run_server(args);
        }
        print_gpu_info(args.device);
        return run_cli(args);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA: " << cudaGetErrorString(err) << std::endl;
        }
        return 1;
    }
}
