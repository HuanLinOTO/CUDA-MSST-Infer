#pragma once

#include "audio_io.h"
#include "model_bs_roformer.h"
#include "model_htdemucs.h"
#include "model_mdx23c.h"
#include "model_mel_band_roformer.h"
#include "weights.h"

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace cudasep::app {

enum class ModelType { Unknown, MelBandRoformer, BSRoformer, MDX23C, HTDemucs };
enum class ChunkMode { Generic, Demucs };

struct LoadedModel {
    std::string model_path;
    ModelType type = ModelType::Unknown;
    ModelWeights weights;
    MelBandRoformer mbr;
    BSRoformer bsr;
    MDX23C mdx;
    HTDemucs htd;
    int num_sources = 1;
    int sample_rate = 44100;
    int chunk_size = 0;
    int num_overlap = 4;
    ChunkMode chunk_mode = ChunkMode::Generic;
    bool quantize_fp16 = false;
    std::vector<std::string> stem_names;

    Tensor forward(const Tensor& input);
};

struct InferenceResult {
    AudioData audio;
    Tensor output;
    double infer_ms = 0.0;
    double rtf = 0.0;
};

struct OutputTrack {
    std::string name;
    Tensor audio;
    bool derived = false;
};

using LogCallback = std::function<void(const std::string&)>;

const char* model_type_name(ModelType t);
ModelType detect_model_type(const JsonValue& config);
LoadedModel load_model(const std::string& model_path, int device, bool quantize_fp16);
InferenceResult run_inference(LoadedModel& model, const std::string& input_path, float overlap,
                              LogCallback logger = nullptr);
InferenceResult run_inference(LoadedModel& model, const AudioData& audio, float overlap,
                              LogCallback logger = nullptr);
std::vector<std::string> collect_stem_names(const LoadedModel& model);
Tensor extract_stem_audio(const Tensor& output, int stem);
std::string stem_label(const LoadedModel& model, int stem);
std::vector<OutputTrack> collect_output_tracks(const LoadedModel& model, const AudioData& input, const Tensor& output);
std::vector<std::filesystem::path> save_outputs(const LoadedModel& model, const Tensor& output,
                                                const std::filesystem::path& out_path, int stem);

}  // namespace cudasep::app
