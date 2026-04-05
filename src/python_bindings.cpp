// python_bindings.cpp — pybind11 wrapper for CudaInfer
//
// Exposes the CudaInfer inference engine to Python with a simple API:
//   import cudasep
//   model = cudasep.load_model("model.csm")
//   result = model.separate("input.wav")      # returns numpy array
//   result = model.separate_tensor(tensor)     # input/output as numpy

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "tensor.h"
#include "weights.h"
#include "audio_io.h"
#include "model_mel_band_roformer.h"
#include "model_bs_roformer.h"
#include "model_mdx23c.h"
#include "model_htdemucs.h"

#include <string>
#include <memory>
#include <stdexcept>
#include <cuda_runtime.h>

namespace py = pybind11;

// ============================================================================
// Model type detection (same as main.cpp)
// ============================================================================

enum class ModelType { Unknown, MelBandRoformer, BSRoformer, MDX23C, HTDemucs };

static ModelType detect_model_type(const cudasep::JsonValue& config) {
    if (config.has("model_type")) {
        std::string mt = config.get_string("model_type", "");
        if (mt == "mel_band_roformer" || mt == "MelBandRoformer") return ModelType::MelBandRoformer;
        if (mt == "bs_roformer" || mt == "BSRoformer") return ModelType::BSRoformer;
        if (mt == "mdx23c" || mt == "MDX23C" || mt == "tfc_tdf") return ModelType::MDX23C;
        if (mt == "htdemucs" || mt == "HTDemucs") return ModelType::HTDemucs;
    }
    if (config.has("nfft") && config.has("depth") && config.has("growth"))
        return ModelType::HTDemucs;
    if (config.has("num_bands") || config.has("mel_scale"))
        return ModelType::MelBandRoformer;
    if (config.has("freqs_per_bands"))
        return ModelType::BSRoformer;
    if (config.has("num_scales") || config.has("bottleneck_factor"))
        return ModelType::MDX23C;
    if (config.has("dim") && config.has("depth") && config.has("dim_head"))
        return ModelType::BSRoformer;
    return ModelType::Unknown;
}

// ============================================================================
// Helper: Tensor ↔ numpy conversion
// ============================================================================

static cudasep::Tensor numpy_to_tensor(py::array_t<float> arr) {
    auto info = arr.request();
    std::vector<int64_t> shape(info.shape.begin(), info.shape.end());
    int64_t numel = 1;
    for (auto s : shape) numel *= s;
    return cudasep::Tensor::from_cpu_f32((const float*)info.ptr, shape);
}

static py::array_t<float> tensor_to_numpy(const cudasep::Tensor& t) {
    std::vector<float> cpu_data = t.to_cpu_f32();
    auto shape = t.shape();
    std::vector<ssize_t> py_shape(shape.begin(), shape.end());
    py::array_t<float> result(py_shape);
    std::memcpy(result.mutable_data(), cpu_data.data(), cpu_data.size() * sizeof(float));
    return result;
}

// ============================================================================
// Separator class — unified Python API
// ============================================================================

class Separator {
public:
    Separator(const std::string& model_path, int device = 0) {
        cudaSetDevice(device);

        weights_ = std::make_unique<cudasep::ModelWeights>(
            cudasep::ModelWeights::load(model_path));

        type_ = detect_model_type(weights_->config());

        switch (type_) {
            case ModelType::MelBandRoformer:
                mbr_ = std::make_unique<cudasep::MelBandRoformer>();
                mbr_->load(*weights_);
                sample_rate_ = mbr_->config().sample_rate;
                num_sources_ = mbr_->config().num_stems;
                break;
            case ModelType::BSRoformer:
                bsr_ = std::make_unique<cudasep::BSRoformer>();
                bsr_->load(*weights_);
                sample_rate_ = bsr_->config().sample_rate;
                num_sources_ = bsr_->config().num_stems;
                break;
            case ModelType::MDX23C:
                mdx_ = std::make_unique<cudasep::MDX23C>();
                mdx_->load(*weights_);
                sample_rate_ = mdx_->config().sample_rate;
                num_sources_ = mdx_->config().num_target_instruments;
                break;
            case ModelType::HTDemucs:
                htd_ = std::make_unique<cudasep::HTDemucs>();
                htd_->load(*weights_);
                sample_rate_ = htd_->config().samplerate;
                num_sources_ = htd_->config().num_sources;
                break;
            default:
                throw std::runtime_error("Unknown model type in .csm file");
        }
    }

    // Run inference on file → returns numpy [S, C, L] or [C, L]
    py::array_t<float> separate_file(const std::string& audio_path, int stem = -1) {
        cudasep::AudioData audio = cudasep::load_audio(audio_path);
        cudasep::Tensor input = audio.samples.unsqueeze(0);  // [1, C, L]
        cudasep::Tensor output = run_forward(input);
        return extract_stem(output, stem);
    }

    // Run inference on numpy array [C, L] → returns numpy [S, C, L] or [C, L]
    py::array_t<float> separate_tensor(py::array_t<float> audio, int stem = -1) {
        auto info = audio.request();
        cudasep::Tensor t = numpy_to_tensor(audio);
        if (t.ndim() == 2) {
            t = t.unsqueeze(0);  // [1, C, L]
        }
        cudasep::Tensor output = run_forward(t);
        return extract_stem(output, stem);
    }

    int sample_rate() const { return sample_rate_; }
    int num_sources() const { return num_sources_; }

    std::string model_type() const {
        switch (type_) {
            case ModelType::MelBandRoformer: return "MelBandRoformer";
            case ModelType::BSRoformer:      return "BSRoformer";
            case ModelType::MDX23C:          return "MDX23C";
            case ModelType::HTDemucs:        return "HTDemucs";
            default:                         return "Unknown";
        }
    }

private:
    cudasep::Tensor run_forward(const cudasep::Tensor& input) {
        switch (type_) {
            case ModelType::MelBandRoformer: return mbr_->forward(input);
            case ModelType::BSRoformer:      return bsr_->forward(input);
            case ModelType::MDX23C:          return mdx_->forward(input);
            case ModelType::HTDemucs:        return htd_->forward(input);
            default: throw std::runtime_error("Unknown model type");
        }
    }

    py::array_t<float> extract_stem(const cudasep::Tensor& output, int stem) {
        if (output.ndim() == 4) {
            // [B, S, C, L]
            if (stem >= 0 && stem < (int)output.size(1)) {
                cudasep::Tensor s = output.slice(1, stem, stem + 1)
                                          .squeeze(0).squeeze(0);  // [C, L]
                return tensor_to_numpy(s);
            }
            // Return all stems: [S, C, L]
            return tensor_to_numpy(output.squeeze(0));
        }
        // [B, C, L] → [C, L]
        return tensor_to_numpy(output.squeeze(0));
    }

    ModelType type_;
    int sample_rate_ = 44100;
    int num_sources_ = 1;
    std::unique_ptr<cudasep::ModelWeights> weights_;
    std::unique_ptr<cudasep::MelBandRoformer> mbr_;
    std::unique_ptr<cudasep::BSRoformer> bsr_;
    std::unique_ptr<cudasep::MDX23C> mdx_;
    std::unique_ptr<cudasep::HTDemucs> htd_;
};

// ============================================================================
// Module definition
// ============================================================================

PYBIND11_MODULE(cudasep, m) {
    m.doc() = "CudaInfer — GPU-accelerated music source separation";

    py::class_<Separator>(m, "Separator")
        .def(py::init<const std::string&, int>(),
             py::arg("model_path"),
             py::arg("device") = 0,
             "Load a .csm model and prepare for inference")
        .def("separate_file", &Separator::separate_file,
             py::arg("audio_path"),
             py::arg("stem") = -1,
             "Separate an audio file. Returns numpy array [S, C, L] or [C, L] if stem specified.")
        .def("separate_tensor", &Separator::separate_tensor,
             py::arg("audio"),
             py::arg("stem") = -1,
             "Separate audio from numpy array [C, L]. Returns numpy array.")
        .def_property_readonly("sample_rate", &Separator::sample_rate)
        .def_property_readonly("num_sources", &Separator::num_sources)
        .def_property_readonly("model_type", &Separator::model_type);

    // Convenience function
    m.def("load_model", [](const std::string& path, int device) {
        return std::make_unique<Separator>(path, device);
    }, py::arg("model_path"), py::arg("device") = 0,
    "Load a .csm model. Returns a Separator object.");
}
