#pragma once
#include "tensor.h"
#include "weights.h"
#include <vector>
#include <string>

namespace cudasep {

// Configuration for MDX23C (TFC_TDF_net) — a U-Net with frequency-domain processing.
struct MDX23CConfig {
    // Audio parameters
    int chunk_size = 261120;
    int dim_f = 4096;
    int dim_t = 256;
    int hop_length = 1024;
    int n_fft = 8192;
    int num_channels = 2;       // audio channels (stereo)
    int sample_rate = 44100;

    // Model parameters
    std::string act_type = "gelu";       // gelu, relu, or elu{alpha}
    std::string norm_type = "BatchNorm"; // BatchNorm, InstanceNorm, or GroupNormN
    int bottleneck_factor = 4;           // bn — frequency bottleneck ratio in TDF
    int growth = 128;                    // g  — channel growth per encoder stage
    int num_blocks_per_scale = 2;        // l  — TFC_TDF blocks per stage
    int num_channels_model = 128;        // c  — base channel count
    int num_scales = 5;                  // n  — number of encoder/decoder stages
    int num_subbands = 4;                // k  — number of frequency subbands
    int scale_h = 2;                     // down/up-scale factor (time dimension)
    int scale_w = 2;                     // down/up-scale factor (freq dimension)
    int num_target_instruments = 1;      // number of output sources
    int group_norm_groups = 0;           // N for GroupNormN
    float elu_alpha = 1.0f;              // alpha for ELU activation

    static MDX23CConfig from_json(const JsonValue& j);
};

class MDX23C {
public:
    MDX23C() = default;

    /// Load all model weights from a .csm file.
    void load(const ModelWeights& weights);

    /// Run inference: input [B, channels, samples], output [B, num_targets, channels, samples].
    Tensor forward(const Tensor& audio);

    const MDX23CConfig& config() const { return cfg_; }

private:
    MDX23CConfig cfg_;

    // STFT window
    Tensor stft_window_;

    // Parsed normalization type
    enum class NormType { BatchNorm, InstanceNorm, GroupNorm };
    NormType norm_type_ = NormType::BatchNorm;

    // Empty tensor used as placeholder for no-bias convolutions
    Tensor no_bias_;

    // ---- Weight structures ------------------------------------------------

    struct NormWeights {
        Tensor weight;       // gamma  [C]
        Tensor bias;         // beta   [C]
        Tensor running_mean; // [C], BatchNorm only
        Tensor running_var;  // [C], BatchNorm only
    };

    // One sub-block inside TFC_TDF:
    //   tfc1: norm → act → Conv2d(in_c, c, 3, padding=1)
    //   tdf:  norm → act → Linear(f→f/bn) → norm → act → Linear(f/bn→f)
    //   tfc2: norm → act → Conv2d(c, c, 3, padding=1)
    //   shortcut: Conv2d(in_c, c, 1)
    struct TFC_TDF_Block {
        NormWeights tfc1_norm;
        Tensor tfc1_conv_w;       // [c, in_c, 3, 3]

        NormWeights tdf_norm1;
        Tensor tdf_linear1_w;     // [f/bn, f]
        NormWeights tdf_norm2;
        Tensor tdf_linear2_w;     // [f, f/bn]

        NormWeights tfc2_norm;
        Tensor tfc2_conv_w;       // [c, c, 3, 3]

        Tensor shortcut_w;        // [c, in_c, 1, 1]
    };

    struct TFC_TDF_Weights {
        std::vector<TFC_TDF_Block> blocks;   // l blocks
    };

    struct EncoderBlock {
        TFC_TDF_Weights tfc_tdf;
        NormWeights downscale_norm;
        Tensor downscale_conv_w;  // Conv2d(c, c+g, scale, stride=scale)
    };

    struct DecoderBlock {
        NormWeights upscale_norm;
        Tensor upscale_conv_w;    // ConvTranspose2d(c, c-g, scale, stride=scale)
        TFC_TDF_Weights tfc_tdf;
    };

    // ---- Stored weights ---------------------------------------------------

    Tensor first_conv_w_;                        // [c, dim_c, 1, 1]
    std::vector<EncoderBlock> encoder_blocks_;   // n stages
    TFC_TDF_Weights bottleneck_;
    std::vector<DecoderBlock> decoder_blocks_;   // n stages
    Tensor final_conv1_w_;                       // [c, c+dim_c, 1, 1]
    Tensor final_conv2_w_;                       // [num_targets*dim_c, c, 1, 1]

    // ---- Helper methods ---------------------------------------------------

    void load_norm(const ModelWeights& w, const std::string& prefix, NormWeights& nw);
    void load_tfc_tdf(const ModelWeights& w, const std::string& prefix,
                      int num_blocks, TFC_TDF_Weights& out);

    Tensor apply_norm(const Tensor& x, const NormWeights& nw);
    Tensor apply_act(const Tensor& x);
    Tensor apply_tfc_tdf(const Tensor& x, const TFC_TDF_Weights& w);

    Tensor cac2cws(const Tensor& x);   // subband split:    [B, c, f, t] → [B, c*k, f/k, t]
    Tensor cws2cac(const Tensor& x);   // subband merge:    [B, c*k, f/k, t] → [B, c, f, t]
};

} // namespace cudasep
