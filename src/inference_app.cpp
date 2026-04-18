#include "inference_app.h"

#include "ops.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <utility>

namespace fs = std::filesystem;

namespace cudasep::app {

const char* model_type_name(ModelType t) {
    switch (t) {
        case ModelType::MelBandRoformer: return "MelBandRoformer";
        case ModelType::BSRoformer: return "BSRoformer";
        case ModelType::MDX23C: return "MDX23C";
        case ModelType::HTDemucs: return "HTDemucs";
        default: return "Unknown";
    }
}

static int default_num_overlap(const JsonValue& config, ModelType t) {
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

static int resolve_step(int chunk_size, float overlap_arg, int default_overlap) {
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

static std::string lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

ModelType detect_model_type(const JsonValue& config) {
    if (config.has("model_type")) {
        std::string mt = config.get_string("model_type", "");
        if (mt == "mel_band_roformer" || mt == "MelBandRoformer") return ModelType::MelBandRoformer;
        if (mt == "bs_roformer" || mt == "BSRoformer") return ModelType::BSRoformer;
        if (mt == "mdx23c" || mt == "MDX23C" || mt == "tfc_tdf") return ModelType::MDX23C;
        if (mt == "htdemucs" || mt == "HTDemucs") return ModelType::HTDemucs;
    }

    if (config.has("nfft") && config.has("depth") && config.has("growth")) return ModelType::HTDemucs;
    if (config.has("num_bands") || config.has("mel_scale")) return ModelType::MelBandRoformer;
    if (config.has("freqs_per_bands")) return ModelType::BSRoformer;
    if (config.has("num_scales") || config.has("bottleneck_factor") || config.has("num_subbands")) {
        if (config.has("growth") && config.has("depth")) return ModelType::HTDemucs;
        return ModelType::MDX23C;
    }
    if (config.has("dim") && config.has("depth") && config.has("dim_head")) {
        if (config.has("num_bands")) return ModelType::MelBandRoformer;
        return ModelType::BSRoformer;
    }
    return ModelType::Unknown;
}

static Tensor process_chunked(const Tensor& audio, int chunk_size, int step, ChunkMode mode,
                              std::function<Tensor(const Tensor&)> model_forward,
                              LogCallback logger) {
    Tensor mix = audio;
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
    Tensor first_chunk = mix.slice(-1, 0, std::min((int64_t)chunk_size, total_samples));
    if (first_chunk.size(-1) < chunk_size) {
        if (mode == ChunkMode::Generic && first_chunk.size(-1) > chunk_size / 2) {
            first_chunk = first_chunk.pad_reflect({0, chunk_size - first_chunk.size(-1)});
        } else {
            first_chunk = first_chunk.pad({0, chunk_size - first_chunk.size(-1)}, 0.0f);
        }
    }
    Tensor first_out = model_forward(first_chunk);

    std::vector<int64_t> out_shape = first_out.shape();
    out_shape.back() = total_samples;

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
    Tensor fade_tensor = Tensor::from_cpu_f32(fade.data(), {(int64_t)chunk_size});

    Tensor output = Tensor::zeros(out_shape);
    Tensor weight_sum = Tensor::zeros({total_samples});
    int64_t num_channels = output.numel() / total_samples;
    output = output.reshape({num_channels, total_samples});

    bool last_chunk = false;
    int64_t chunk_count = (total_samples + step - 1) / step;
    int64_t chunk_index = 0;
    if (logger) {
        logger("[分片] 开始分片推理");
        logger("[分片] 分片大小: " + std::to_string(chunk_size));
        logger("[分片] 步长: " + std::to_string(step));
        logger("[分片] 分片数量: " + std::to_string(chunk_count));
    }
    for (int64_t start = 0; start < total_samples; start += step) {
        chunk_index++;
        int64_t end = std::min(start + chunk_size, total_samples);
        int64_t actual_len = end - start;
        last_chunk = (end >= total_samples);

        Tensor chunk_out;
        if (start == 0) {
            chunk_out = first_out;
        } else {
            Tensor chunk = mix.slice(-1, start, end);
            if (actual_len < chunk_size) {
                if (mode == ChunkMode::Generic && actual_len > chunk_size / 2) {
                    chunk = chunk.pad_reflect({0, chunk_size - actual_len});
                } else {
                    chunk = chunk.pad({0, chunk_size - actual_len}, 0.0f);
                }
            }
            chunk_out = model_forward(chunk);
        }

        if (chunk_out.size(-1) > actual_len) {
            chunk_out = chunk_out.slice(-1, 0, actual_len);
        }
        chunk_out = chunk_out.reshape({num_channels, actual_len}).contiguous();

        Tensor window = fade_tensor;
        if (mode == ChunkMode::Generic && fade_size > 0) {
            window = fade_tensor.clone();
            if (start == 0 || last_chunk) {
                std::vector<float> window_cpu = window.to_cpu_f32();
                if (start == 0) {
                    std::fill(window_cpu.begin(), window_cpu.begin() + fade_size, 1.0f);
                }
                if (last_chunk) {
                    std::fill(window_cpu.end() - fade_size, window_cpu.end(), 1.0f);
                }
                window = Tensor::from_cpu_f32(window_cpu.data(), {(int64_t)chunk_size});
            }
        }

        Tensor fade_crop = (actual_len < chunk_size) ? window.slice(0, 0, actual_len).contiguous() : window;
        ops::overlap_add(output, chunk_out, fade_crop, start);
        ops::weight_accumulate(weight_sum, fade_crop, start);

        if (logger && (chunk_index == 1 || chunk_index == chunk_count || (chunk_index % 4) == 0)) {
            logger("[分片] 已完成分片 " + std::to_string(chunk_index) + "/" + std::to_string(chunk_count));
        }
    }

    if (logger) {
        logger("[分片] 分片拼接完成");
    }

    ops::normalize_by_weights(output, weight_sum);
    output = output.reshape(out_shape);
    if (mode == ChunkMode::Generic && length_init > 2LL * border && border > 0) {
        output = output.slice(output.ndim() - 1, border, border + length_init);
    }
    return output;
}

Tensor LoadedModel::forward(const Tensor& input) {
    switch (type) {
        case ModelType::MelBandRoformer: return mbr.forward(input);
        case ModelType::BSRoformer: return bsr.forward(input);
        case ModelType::MDX23C: return mdx.forward(input);
        case ModelType::HTDemucs: return htd.forward(input);
        default: throw std::runtime_error("Unknown model type");
    }
}

std::vector<std::string> collect_stem_names(const LoadedModel& model) {
    std::vector<std::string> names;
    const JsonValue& config = model.weights.config();
    if (config.has("instruments") && config["instruments"].is_array()) {
        const JsonValue& instruments = config["instruments"];
        for (size_t i = 0; i < instruments.size(); ++i) {
            names.push_back(instruments[i].as_string());
        }
    }
    if (names.empty() && config.has("sources") && config["sources"].is_array()) {
        const JsonValue& sources = config["sources"];
        for (size_t i = 0; i < sources.size(); ++i) {
            names.push_back(sources[i].as_string());
        }
    }
    if (names.empty() && config.has("target_instrument")) {
        names.push_back(config.get_string("target_instrument", "target"));
    }
    while ((int)names.size() < model.num_sources) {
        names.push_back("stem_" + std::to_string(names.size()));
    }
    if ((int)names.size() > model.num_sources) {
        names.resize(model.num_sources);
    }
    return names;
}

LoadedModel load_model(const std::string& model_path, int device, bool quantize_fp16) {
    cudaSetDevice(device);

    LoadedModel model;
    model.model_path = model_path;
    model.quantize_fp16 = quantize_fp16;
    model.weights = ModelWeights::load(model_path);
    model.type = detect_model_type(model.weights.config());
    if (model.type == ModelType::Unknown) {
        throw std::runtime_error("Could not detect model type from config");
    }

    if (quantize_fp16) {
        g_quantize_fp16 = true;
        model.weights.convert_linear_weights_to_fp16();
    }

    switch (model.type) {
        case ModelType::MelBandRoformer:
            model.mbr.load(model.weights);
            model.num_sources = model.mbr.config().num_stems;
            model.sample_rate = model.mbr.config().sample_rate;
            model.chunk_size = model.weights.config().get_int("chunk_size", model.mbr.config().stft_n_fft * 256);
            model.num_overlap = default_num_overlap(model.weights.config(), model.type);
            model.chunk_mode = ChunkMode::Generic;
            break;
        case ModelType::BSRoformer:
            model.bsr.load(model.weights);
            model.num_sources = model.bsr.config().num_stems;
            model.sample_rate = model.bsr.config().sample_rate;
            model.chunk_size = model.weights.config().get_int("chunk_size", model.bsr.config().stft_n_fft * 256);
            model.num_overlap = default_num_overlap(model.weights.config(), model.type);
            model.chunk_mode = ChunkMode::Generic;
            break;
        case ModelType::MDX23C:
            model.mdx.load(model.weights);
            model.num_sources = model.mdx.config().num_target_instruments;
            model.sample_rate = model.mdx.config().sample_rate;
            model.chunk_size = model.weights.config().get_int("chunk_size", model.mdx.config().chunk_size);
            model.num_overlap = default_num_overlap(model.weights.config(), model.type);
            model.chunk_mode = ChunkMode::Generic;
            break;
        case ModelType::HTDemucs:
            model.htd.load(model.weights);
            model.num_sources = model.htd.config().num_sources;
            model.sample_rate = model.htd.config().samplerate;
            model.chunk_size = (int)(model.htd.config().segment * model.htd.config().samplerate);
            model.num_overlap = default_num_overlap(model.weights.config(), model.type);
            model.chunk_mode = ChunkMode::Demucs;
            break;
        default:
            throw std::runtime_error("Unknown model type");
    }

    model.stem_names = collect_stem_names(model);
    return model;
}

InferenceResult run_inference(LoadedModel& model, const AudioData& audio, float overlap, LogCallback logger) {
    Tensor input = audio.samples.unsqueeze(0);

    if (logger) {
        logger("[音频] 采样率: " + std::to_string(audio.sample_rate));
        logger("[音频] 声道数: " + std::to_string(audio.channels));
        logger("[音频] 采样点: " + std::to_string(audio.num_samples));
    }

    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();

    Tensor output;
    if (model.chunk_size > 0 && audio.num_samples > model.chunk_size) {
        int step = resolve_step(model.chunk_size, overlap, model.num_overlap);
        if (logger) {
            logger("[推理] 音频长度超过 chunk_size，启用分片推理");
        }
        output = process_chunked(input, model.chunk_size, step, model.chunk_mode,
                                 [&](const Tensor& x) { return model.forward(x); }, logger);
    } else {
        if (logger) {
            logger("[推理] 音频长度较短，直接执行整段推理");
        }
        output = model.forward(input);
    }

    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    InferenceResult result;
    result.audio = audio;
    result.output = std::move(output);
    result.infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.rtf = (double)audio.num_samples / model.sample_rate / (result.infer_ms / 1000.0);
    if (logger) {
        logger("[推理] 推理耗时: " + std::to_string((int)result.infer_ms) + " ms");
        logger("[推理] 实时率: " + std::to_string(result.rtf));
    }
    return result;
}

InferenceResult run_inference(LoadedModel& model, const std::string& input_path, float overlap, LogCallback logger) {
    AudioData audio = load_audio(input_path);
    return run_inference(model, audio, overlap, logger);
}

Tensor extract_stem_audio(const Tensor& output, int stem) {
    if (output.ndim() == 4) {
        int sources = (int)output.size(1);
        if (stem < 0 || stem >= sources) {
            throw std::runtime_error("Stem index out of range");
        }
        return output.slice(1, stem, stem + 1).squeeze(0).squeeze(0);
    }
    return output.squeeze(0);
}

std::string stem_label(const LoadedModel& model, int stem) {
    if (stem >= 0 && stem < (int)model.stem_names.size()) {
        return model.stem_names[stem];
    }
    return "stem_" + std::to_string(stem);
}

std::vector<OutputTrack> collect_output_tracks(const LoadedModel& model, const AudioData& input, const Tensor& output) {
    std::vector<OutputTrack> tracks;
    if (output.ndim() == 4) {
        int sources = (int)output.size(1);
        Tensor sum;
        bool has_sum = false;
        bool has_other = false;
        for (int s = 0; s < sources; ++s) {
            OutputTrack track;
            track.name = stem_label(model, s);
            track.audio = extract_stem_audio(output, s);
            track.derived = false;
            if (!has_sum) {
                sum = track.audio.clone();
                has_sum = true;
            } else {
                sum = sum + track.audio;
            }
            if (lower_ascii(track.name) == "other") {
                has_other = true;
            }
            tracks.push_back(std::move(track));
        }
        if (!has_other && has_sum) {
            OutputTrack other;
            other.name = "other";
            other.audio = input.samples - sum;
            other.derived = true;
            tracks.push_back(std::move(other));
        }
        return tracks;
    }

    OutputTrack primary;
    primary.name = model.stem_names.empty() ? "target" : model.stem_names.front();
    primary.audio = output.squeeze(0);
    primary.derived = false;
    tracks.push_back(primary);

    if (lower_ascii(primary.name) != "other") {
        OutputTrack other;
        other.name = "other";
        other.audio = input.samples - primary.audio;
        other.derived = true;
        tracks.push_back(std::move(other));
    }
    return tracks;
}

std::vector<fs::path> save_outputs(const LoadedModel& model, const Tensor& output,
                                   const fs::path& out_path, int stem) {
    std::vector<fs::path> saved;
    bool multi_source = (output.ndim() == 4);

    if (multi_source) {
        int sources = (int)output.size(1);
        if (stem >= 0 && stem < sources) {
            Tensor stem_audio = extract_stem_audio(output, stem);
            fs::path file_path = out_path;
            if (out_path.extension() != ".wav") {
                fs::create_directories(out_path);
                file_path = out_path / (stem_label(model, stem) + ".wav");
            }
            save_wav(file_path.string(), stem_audio, model.sample_rate);
            saved.push_back(file_path);
            return saved;
        }

        if (stem != -1) {
            throw std::runtime_error("Stem index out of range");
        }

        fs::create_directories(out_path);
        for (int s = 0; s < sources; ++s) {
            Tensor stem_audio = extract_stem_audio(output, s);
            fs::path file_path = out_path / (stem_label(model, s) + ".wav");
            save_wav(file_path.string(), stem_audio, model.sample_rate);
            saved.push_back(file_path);
        }
        return saved;
    }

    Tensor stem_audio = output.squeeze(0);
    fs::path file_path = out_path;
    if (out_path.extension() != ".wav") {
        fs::create_directories(out_path);
        file_path = out_path / "output.wav";
    }
    save_wav(file_path.string(), stem_audio, model.sample_rate);
    saved.push_back(file_path);
    return saved;
}

}  // namespace cudasep::app
