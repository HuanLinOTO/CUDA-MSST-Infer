// main.cpp — CudaInfer CLI entry point
//
// Usage:
//   cudasep_infer --model <path.csm> --input <audio> --output <output_dir> [options]
//
// Options:
//   --model, -m      Path to .csm model weights file (required)
//   --input, -i      Path to input audio file (required)
//   --output, -o     Output directory or file path (default: ./output)
//   --stem, -s       Stem index to extract (default: 0). For multi-source models, -1 = all.
//   --overlap        Overlap ratio for chunked processing (default: 0.25)
//   --device, -d     CUDA device ID (default: 0)
//   --list-stems     Print stem/source info from model config and exit

#include "tensor.h"
#include "weights.h"
#include "audio_io.h"
#include "ops.h"
#include "model_mel_band_roformer.h"
#include "model_bs_roformer.h"
#include "model_mdx23c.h"
#include "model_htdemucs.h"

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

namespace fs = std::filesystem;

// ============================================================================
// Model type detection from config JSON
// ============================================================================

enum class ModelType { Unknown, MelBandRoformer, BSRoformer, MDX23C, HTDemucs };

enum class ChunkMode { Generic, Demucs };

static const char* model_type_name(ModelType t) {
    switch (t) {
        case ModelType::MelBandRoformer: return "MelBandRoformer";
        case ModelType::BSRoformer:      return "BSRoformer";
        case ModelType::MDX23C:          return "MDX23C";
        case ModelType::HTDemucs:        return "HTDemucs";
        default:                         return "Unknown";
    }
}

static int default_num_overlap(const cudasep::JsonValue& config, ModelType t) {
    if (config.has("num_overlap")) {
        return std::max(1, config.get_int("num_overlap", 1));
    }

    switch (t) {
        case ModelType::MelBandRoformer:
        case ModelType::BSRoformer:
            return 2;
        case ModelType::MDX23C:
        case ModelType::HTDemucs:
            return 4;
        default:
            return 4;
    }
}

static int resolve_step(int chunk_size, float overlap_arg, int default_overlap, ChunkMode mode) {
    if (overlap_arg > 0.0f && overlap_arg < 1.0f) {
        int step = (int)std::lround(chunk_size * (1.0f - overlap_arg));
        return std::max(1, step);
    }

    int num_overlap = default_overlap;
    if (overlap_arg >= 1.0f) {
        num_overlap = std::max(1, (int)std::lround(overlap_arg));
    }
    return std::max(1, chunk_size / std::max(1, num_overlap));
}

static ModelType detect_model_type(const cudasep::JsonValue& config) {
    // Check for explicit model_type field
    if (config.has("model_type")) {
        std::string mt = config.get_string("model_type", "");
        if (mt == "mel_band_roformer" || mt == "MelBandRoformer") return ModelType::MelBandRoformer;
        if (mt == "bs_roformer" || mt == "BSRoformer") return ModelType::BSRoformer;
        if (mt == "mdx23c" || mt == "MDX23C" || mt == "tfc_tdf") return ModelType::MDX23C;
        if (mt == "htdemucs" || mt == "HTDemucs") return ModelType::HTDemucs;
    }

    // Heuristic detection based on config keys
    if (config.has("nfft") && config.has("depth") && config.has("growth")) {
        return ModelType::HTDemucs;
    }
    if (config.has("num_bands") || config.has("mel_scale")) {
        return ModelType::MelBandRoformer;
    }
    if (config.has("freqs_per_bands")) {
        return ModelType::BSRoformer;
    }
    if (config.has("num_scales") || config.has("bottleneck_factor") || config.has("num_subbands")) {
        // MDX23C uses num_scales and num_subbands in the model config
        // But HTDemucs also uses num_subbands (though it's typically 1)
        if (config.has("growth") && config.has("depth")) {
            return ModelType::HTDemucs; // growth + depth = HTDemucs
        }
        return ModelType::MDX23C;
    }
    if (config.has("dim") && config.has("depth") && config.has("dim_head")) {
        // Could be either Roformer variant
        if (config.has("num_bands")) return ModelType::MelBandRoformer;
        return ModelType::BSRoformer;
    }

    return ModelType::Unknown;
}

// ============================================================================
// Chunked processing with overlap
// ============================================================================

static cudasep::Tensor process_chunked(
    const cudasep::Tensor& audio,  // [1, channels, samples]
    int chunk_size,
    int step,
    ChunkMode mode,
    std::function<cudasep::Tensor(const cudasep::Tensor&)> model_forward
) {
    cudasep::Tensor mix = audio;
    int64_t length_init = audio.size(-1);

    int fade_size = 0;
    int border = 0;
    if (mode == ChunkMode::Generic) {
        fade_size = chunk_size / 10;
        border = chunk_size - step;
        if (length_init > 2LL * border && border > 0) {
            mix = mix.pad_reflect({border, border});
        }
    }

    int64_t total_samples = mix.size(-1);

    // Count chunks
    int num_chunks = 0;
    for (int64_t start = 0; start < total_samples; start += step) {
        num_chunks++;
    }

    std::cout << "  Processing " << num_chunks << " chunks "
              << "(chunk=" << chunk_size << ", step=" << step << ")" << std::endl;

    // First pass: determine output shape from first chunk
    cudasep::Tensor first_chunk = mix.slice(-1, 0, std::min((int64_t)chunk_size, total_samples));
    if (first_chunk.size(-1) < chunk_size) {
        if (mode == ChunkMode::Generic && first_chunk.size(-1) > chunk_size / 2) {
            first_chunk = first_chunk.pad_reflect({0, chunk_size - first_chunk.size(-1)});
        } else {
            first_chunk = first_chunk.pad({0, chunk_size - first_chunk.size(-1)}, 0.0f);
        }
    }
    cudasep::Tensor first_out = model_forward(first_chunk);

    // Output shape with full audio length
    std::vector<int64_t> out_shape = first_out.shape();
    out_shape.back() = total_samples;

    // Match MSST generic overlap-add: linear fade, ones in the middle.
    std::vector<float> fade(chunk_size, 1.0f);
    if (mode == ChunkMode::Generic && fade_size > 0) {
        if (fade_size == 1) {
            fade.front() = 0.0f;
            fade.back() = 0.0f;
        } else {
            for (int i = 0; i < fade_size; ++i) {
                float alpha = (float)i / (float)(fade_size - 1);
                fade[i] = alpha;
                fade[chunk_size - fade_size + i] = 1.0f - alpha;
            }
        }
    }
    cudasep::Tensor fade_tensor = cudasep::Tensor::from_cpu_f32(fade.data(), {(int64_t)chunk_size});

    // Allocate output and weight_sum on GPU (all zeros)
    cudasep::Tensor output = cudasep::Tensor::zeros(out_shape);
    cudasep::Tensor weight_sum = cudasep::Tensor::zeros({total_samples});

    // Flatten output for overlap-add: [num_channels, total_samples]
    int64_t num_channels = output.numel() / total_samples;
    output = output.reshape({num_channels, total_samples});

    int chunk_idx = 0;
    for (int64_t start = 0; start < total_samples; start += step) {
        int64_t end = std::min(start + chunk_size, total_samples);
        int64_t actual_len = end - start;

        cudasep::Tensor chunk_out;
        if (chunk_idx == 0) {
            chunk_out = first_out;
        } else {
            cudasep::Tensor chunk = mix.slice(-1, start, end);
            if (actual_len < chunk_size) {
                if (mode == ChunkMode::Generic && actual_len > chunk_size / 2) {
                    chunk = chunk.pad_reflect({0, chunk_size - actual_len});
                } else {
                    chunk = chunk.pad({0, chunk_size - actual_len}, 0.0f);
                }
            }
            chunk_out = model_forward(chunk);
        }

        // Trim chunk_out to actual_len if needed
        if (chunk_out.size(-1) > actual_len) {
            chunk_out = chunk_out.slice(-1, 0, actual_len);
        }

        // Flatten chunk_out to [num_channels, actual_len]
        chunk_out = chunk_out.reshape({num_channels, actual_len}).contiguous();

        // Get fade window cropped to actual_len
        cudasep::Tensor window = fade_tensor;
        if (mode == ChunkMode::Generic && fade_size > 0) {
            window = fade_tensor.clone();
            if (start == 0) {
                std::vector<float> window_cpu = window.to_cpu_f32();
                std::fill(window_cpu.begin(), window_cpu.begin() + fade_size, 1.0f);
                window = cudasep::Tensor::from_cpu_f32(window_cpu.data(), {(int64_t)chunk_size});
            } else if (start + step >= total_samples) {
                std::vector<float> window_cpu = window.to_cpu_f32();
                std::fill(window_cpu.end() - fade_size, window_cpu.end(), 1.0f);
                window = cudasep::Tensor::from_cpu_f32(window_cpu.data(), {(int64_t)chunk_size});
            }
        }

        cudasep::Tensor fade_crop = (actual_len < chunk_size)
            ? window.slice(0, 0, actual_len).contiguous()
            : window;

        // GPU-side overlap-add: output[:, start:start+actual_len] += chunk_out * fade
        cudasep::ops::overlap_add(output, chunk_out, fade_crop, start);

        // GPU-side weight accumulation: weight_sum[start:start+actual_len] += fade
        cudasep::ops::weight_accumulate(weight_sum, fade_crop, start);

        chunk_idx++;
        std::cout << "\r  Chunk " << chunk_idx << "/" << num_chunks << std::flush;
    }
    std::cout << std::endl;

    // GPU-side normalization: output /= weight_sum
    cudasep::ops::normalize_by_weights(output, weight_sum);

    // Reshape back to original output shape
    output = output.reshape(out_shape);

    if (mode == ChunkMode::Generic && length_init > 2LL * border && border > 0) {
        output = output.slice(output.ndim() - 1, border, border + length_init);
    }

    return output;
}

// ============================================================================
// Argument parser
// ============================================================================

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
};

static Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; i++) {
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
        } else if (a == "--help" || a == "-h") {
            args.help = true;
        }
    }
    return args;
}

static void print_usage(const char* progname) {
    std::cout << "CudaInfer — GPU-accelerated music source separation\n"
              << "\n"
              << "Usage: " << progname << " --model <path.csm> --input <audio> [options]\n"
              << "\n"
              << "Required:\n"
              << "  --model, -m <path>   Path to .csm model weights file\n"
              << "  --input, -i <path>   Path to input audio file\n"
              << "\n"
              << "Options:\n"
              << "  --output, -o <path>  Output directory or WAV file path (default: output)\n"
              << "  --stem, -s <int>     Stem index to extract (default: 0, -1 for all)\n"
              << "  --overlap <float>    Chunk overlap. Values in (0,1) are legacy ratios; values >= 1 are MSST num_overlap. Default: model config\n"
              << "  --device, -d <int>   CUDA device ID (default: 0)\n"
              << "  --quantize, --fp16   Enable FP16 mixed-precision GEMM\n"
              << "  --list-stems         Print stem info from model config and exit\n"
              << "  --help, -h           Show this help message\n"
              << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    if (args.help) {
        print_usage(argv[0]);
        return 0;
    }

    if (args.model_path.empty()) {
        std::cerr << "Error: --model is required" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    // Set CUDA device
    cudaSetDevice(args.device);

    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, args.device);
    std::cout << "GPU: " << prop.name << " (SM " << prop.major << "." << prop.minor
              << ", " << (prop.totalGlobalMem / (1024 * 1024)) << " MB)" << std::endl;

    // ========================================================================
    // Load model weights
    // ========================================================================
    std::cout << "Loading model: " << args.model_path << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();

    cudasep::ModelWeights weights = cudasep::ModelWeights::load(args.model_path);

    auto t1 = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "  Weights loaded: " << weights.num_tensors() << " tensors ("
              << (int)(load_ms) << " ms)" << std::endl;

    // ========================================================================
    // Detect model type
    // ========================================================================
    ModelType mtype = detect_model_type(weights.config());
    std::cout << "  Model type: " << model_type_name(mtype) << std::endl;

    if (mtype == ModelType::Unknown) {
        std::cerr << "Error: could not detect model type from config" << std::endl;
        return 1;
    }

    // ========================================================================
    // Initialize model
    // ========================================================================
    // Enable FP16 quantization if requested
    if (args.quantize_fp16) {
        cudasep::g_quantize_fp16 = true;
        std::cout << "  [FP16] Mixed-precision quantization enabled" << std::endl;
        // Pre-convert all 2D (linear) weights to FP16 to avoid runtime conversion
        weights.convert_linear_weights_to_fp16();
    }

    cudasep::MelBandRoformer mbr;
    cudasep::BSRoformer bsr;
    cudasep::MDX23C mdx;
    cudasep::HTDemucs htd;

    int num_sources = 1;
    int sample_rate = 44100;
    int chunk_size = 0;
    int num_overlap = 4;
    ChunkMode chunk_mode = ChunkMode::Generic;

    std::function<cudasep::Tensor(const cudasep::Tensor&)> model_forward;

    try {
    switch (mtype) {
        case ModelType::MelBandRoformer:
            mbr.load(weights);
            num_sources = mbr.config().num_stems;
            sample_rate = mbr.config().sample_rate;
            chunk_size = weights.config().get_int("chunk_size", mbr.config().stft_n_fft * 256);
            num_overlap = default_num_overlap(weights.config(), mtype);
            chunk_mode = ChunkMode::Generic;
            model_forward = [&](const cudasep::Tensor& x) { return mbr.forward(x); };
            break;
        case ModelType::BSRoformer:
            bsr.load(weights);
            num_sources = bsr.config().num_stems;
            sample_rate = bsr.config().sample_rate;
            chunk_size = weights.config().get_int("chunk_size", bsr.config().stft_n_fft * 256);
            num_overlap = default_num_overlap(weights.config(), mtype);
            chunk_mode = ChunkMode::Generic;
            model_forward = [&](const cudasep::Tensor& x) { return bsr.forward(x); };
            break;
        case ModelType::MDX23C:
            mdx.load(weights);
            num_sources = mdx.config().num_target_instruments;
            sample_rate = mdx.config().sample_rate;
            chunk_size = weights.config().get_int("chunk_size", mdx.config().chunk_size);
            num_overlap = default_num_overlap(weights.config(), mtype);
            chunk_mode = ChunkMode::Generic;
            model_forward = [&](const cudasep::Tensor& x) { return mdx.forward(x); };
            break;
        case ModelType::HTDemucs:
            htd.load(weights);
            num_sources = htd.config().num_sources;
            sample_rate = htd.config().samplerate;
            chunk_size = (int)(htd.config().segment * htd.config().samplerate);
            num_overlap = default_num_overlap(weights.config(), mtype);
            chunk_mode = ChunkMode::Demucs;
            model_forward = [&](const cudasep::Tensor& x) { return htd.forward(x); };
            break;
        default:
            break;
    }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception during model load: " << e.what() << std::endl;
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            std::cerr << "[CUDA] " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "  Sources: " << num_sources
              << ", Sample rate: " << sample_rate
              << ", Chunk size: " << chunk_size
              << ", Num overlap: " << num_overlap << std::endl;

    // Handle --list-stems
    if (args.list_stems) {
        std::cout << "\nModel has " << num_sources << " source(s)." << std::endl;
        if (weights.config().has("target_instrument")) {
            std::cout << "  Target: " << weights.config().get_string("target_instrument", "unknown") << std::endl;
        }
        if (weights.config().has("instruments")) {
            std::cout << "  Instruments:";
            const auto& inst = weights.config()["instruments"];
            for (size_t i = 0; i < inst.size(); i++) {
                std::cout << " [" << i << "]=" << inst[i].as_string();
            }
            std::cout << std::endl;
        }
        return 0;
    }

    if (args.input_path.empty()) {
        std::cerr << "Error: --input is required" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    // ========================================================================
    // Load audio
    // ========================================================================
    std::cout << "\nLoading audio: " << args.input_path << std::endl;
    auto t2 = std::chrono::high_resolution_clock::now();

    cudasep::AudioData audio = cudasep::load_audio(args.input_path);

    auto t3 = std::chrono::high_resolution_clock::now();
    double audio_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::cout << "  " << audio.channels << "ch, " << audio.sample_rate << " Hz, "
              << audio.num_samples << " samples ("
              << std::fixed << std::setprecision(1)
              << (double)audio.num_samples / audio.sample_rate << "s, "
              << (int)audio_ms << " ms)" << std::endl;

    // Resample if needed (simple skip for now — production should use proper resampling)
    if (audio.sample_rate != sample_rate) {
        std::cerr << "Warning: audio sample rate (" << audio.sample_rate
                  << ") != model sample rate (" << sample_rate
                  << "). Results may be incorrect." << std::endl;
    }

    // Prepare input tensor: [1, channels, samples]
    cudasep::Tensor input = audio.samples.unsqueeze(0);  // [1, C, L]

    // ========================================================================
    // Run inference
    // ========================================================================
    std::cout << "\nRunning inference..." << std::endl;

    // Warm up GPU
    cudaDeviceSynchronize();
    auto t4 = std::chrono::high_resolution_clock::now();

    cudasep::Tensor output;
    try {
        if (chunk_size > 0 && audio.num_samples > chunk_size) {
            int step = resolve_step(chunk_size, args.overlap, num_overlap, chunk_mode);
            std::cout << "  Chunk mode: "
                      << (chunk_mode == ChunkMode::Generic ? "generic" : "demucs")
                      << ", step=" << step << std::endl;
            output = process_chunked(input, chunk_size, step, chunk_mode, model_forward);
        } else {
            output = model_forward(input);
        }
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] Exception during inference: " << e.what() << std::endl;
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "[CUDA] " << cudaGetErrorString(err) << std::endl;
        }
        return 1;
    }

    cudaDeviceSynchronize();
    auto t5 = std::chrono::high_resolution_clock::now();
    double infer_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();
    double rtf = (double)audio.num_samples / audio.sample_rate / (infer_ms / 1000.0);

    std::cout << "  Inference time: " << std::fixed << std::setprecision(1)
              << infer_ms << " ms (RTF: " << std::setprecision(2) << rtf << "x)" << std::endl;
    std::cout << "  Output shape: [";
    for (int i = 0; i < output.ndim(); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << output.size(i);
    }
    std::cout << "]" << std::endl;

    // ========================================================================
    // Save output
    // ========================================================================
    fs::path out_path(args.output_path);

    // Determine output shape:
    // MBR/BSR: [B, C, L] (single stem) or [B, S, C, L] (multi-stem)
    // MDX23C: [B, S, C, L]
    // HTDemucs: [B, S, C, L]
    bool multi_source = (output.ndim() == 4);

    if (multi_source) {
        // Output is [B, S, C, L]
        int S = (int)output.size(1);
        int C = (int)output.size(2);

        if (args.stem >= 0 && args.stem < S) {
            // Save single stem
            cudasep::Tensor stem_audio = output.slice(1, args.stem, args.stem + 1).squeeze(0).squeeze(0);
            // stem_audio: [C, L]

            std::string out_file;
            if (out_path.extension() == ".wav") {
                out_file = out_path.string();
            } else {
                fs::create_directories(out_path);
                out_file = (out_path / ("stem_" + std::to_string(args.stem) + ".wav")).string();
            }

            std::cout << "\nSaving: " << out_file << std::endl;
            cudasep::save_wav(out_file, stem_audio, sample_rate);
        } else if (args.stem == -1) {
            // Save all stems
            fs::create_directories(out_path);
            for (int s = 0; s < S; s++) {
                cudasep::Tensor stem_audio = output.slice(1, s, s + 1).squeeze(0).squeeze(0);
                std::string out_file = (out_path / ("stem_" + std::to_string(s) + ".wav")).string();
                std::cout << "Saving: " << out_file << std::endl;
                cudasep::save_wav(out_file, stem_audio, sample_rate);
            }
        }
    } else {
        // Output is [B, C, L] — single stem
        cudasep::Tensor stem_audio = output.squeeze(0);  // [C, L]

        std::string out_file;
        if (out_path.extension() == ".wav") {
            out_file = out_path.string();
        } else {
            fs::create_directories(out_path);
            out_file = (out_path / "output.wav").string();
        }

        std::cout << "\nSaving: " << out_file << std::endl;
        cudasep::save_wav(out_file, stem_audio, sample_rate);
    }

    std::cout << "Done." << std::endl;
    return 0;
}
