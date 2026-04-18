#pragma once
#include "tensor.h"
#include "weights.h"
#include <functional>
#include <vector>
#include <string>
#include <unordered_map>

namespace cudasep {

// Configuration for MelBandRoformer
struct MBRConfig {
    int dim = 384;
    int depth = 6;
    bool stereo = true;
    int num_stems = 1;
    int time_transformer_depth = 1;
    int freq_transformer_depth = 1;
    int num_bands = 60;
    int dim_head = 64;
    int heads = 8;
    int mask_estimator_depth = 2;
    int mlp_expansion_factor = 4;
    bool zero_dc = true;
    int sample_rate = 44100;
    int stft_n_fft = 2048;
    int stft_hop_length = 441;
    int stft_win_length = 2048;
    bool stft_normalized = false;
    bool match_input_audio_length = false;
    int dim_freqs_in = 1025;
    bool skip_connection = false;

    static MBRConfig from_json(const JsonValue& j);
};

class MelBandRoformer {
public:
    MelBandRoformer() = default;

    // Load model from weights file
    void load(const ModelWeights& weights);

    // Run inference: input [B, channels, samples], output [B, channels, samples] (or [B, num_stems, channels, samples])
    Tensor forward(const Tensor& audio);
    void set_profile_logger(std::function<void(const std::string&)> logger) { profile_logger_ = std::move(logger); }

    const MBRConfig& config() const { return cfg_; }

private:
    MBRConfig cfg_;

    // Precomputed mel band info
    std::vector<int64_t> freq_indices_;      // which freq bins belong to each mel band
    std::vector<int64_t> num_freqs_per_band_; // how many freqs per band
    std::vector<int64_t> num_bands_per_freq_; // how many bands overlap each freq
    std::vector<int64_t> band_freq_dims_;     // dim_input for each band (num_freqs * 2 * audio_channels)
    Tensor freq_indices_gpu_;                  // GPU tensor of freq_indices
    Tensor num_bands_per_freq_gpu_;            // GPU tensor for averaging

    // STFT window
    Tensor stft_window_;
    std::function<void(const std::string&)> profile_logger_;

    // Rotary embeddings cache (keyed by seq_len)
    std::unordered_map<int, std::pair<Tensor, Tensor>> rotary_cache_;

    // Model weights (references into ModelWeights)
    // Band split weights
    struct BandSplitLayer {
        Tensor norm_gamma;  // RMSNorm gamma
        Tensor linear_w;    // Linear weight
        Tensor linear_b;    // Linear bias
    };
    std::vector<BandSplitLayer> band_split_layers_;

    // Transformer layers
    struct AttentionWeights {
        Tensor norm_gamma;     // RMSNorm before attention
        Tensor to_qkv_w;      // [3*dim_inner, dim] no bias
        Tensor to_gates_w;    // [heads, dim]
        Tensor to_gates_b;    // [heads]
        Tensor to_out_w;      // [dim, dim_inner] no bias
    };

    struct FeedForwardWeights {
        Tensor norm_gamma;     // RMSNorm
        Tensor linear1_w;     // [dim_inner, dim]
        Tensor linear1_b;     // [dim_inner]
        Tensor linear2_w;     // [dim, dim_inner]
        Tensor linear2_b;     // [dim]
    };

    struct TransformerLayerWeights {
        std::vector<AttentionWeights> attn_layers;
        std::vector<FeedForwardWeights> ff_layers;
        Tensor final_norm_gamma;  // final RMSNorm
    };

    struct DepthBlock {
        TransformerLayerWeights time_transformer;
        TransformerLayerWeights freq_transformer;
    };
    std::vector<DepthBlock> depth_blocks_;

    // Mask estimator weights
    struct MaskEstimatorWeights {
        struct BandMLP {
            // MLP layers: pairs of (weight, bias) for each Linear in the MLP
            // For depth=1: 2 linears (indices 0, 2 in Sequential)
            // For depth=2: 3 linears (indices 0, 2, 4 in Sequential)
            std::vector<Tensor> linear_w;  // weights for each linear layer
            std::vector<Tensor> linear_b;  // biases for each linear layer
        };
        std::vector<BandMLP> band_mlps;
    };
    std::vector<MaskEstimatorWeights> mask_estimators_;

    // Helper methods
    void compute_mel_bands();
    Tensor compute_rotary_cos_sin(int seq_len, int dim);

    // Forward sub-steps
    Tensor apply_attention(const Tensor& x, const AttentionWeights& w,
                          const Tensor& cos_freqs, const Tensor& sin_freqs);
    Tensor apply_feedforward(const Tensor& x, const FeedForwardWeights& w);
    Tensor apply_transformer(const Tensor& x, const TransformerLayerWeights& w,
                            const Tensor& cos_freqs, const Tensor& sin_freqs);
    Tensor apply_band_split(const Tensor& x);
    Tensor apply_mask_estimator(const Tensor& x, const MaskEstimatorWeights& w);
};

} // namespace cudasep
