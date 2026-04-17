#pragma once
#include "tensor.h"
#include "weights.h"
#include <vector>
#include <string>
#include <unordered_map>

namespace cudasep {

// Configuration for BSRoformer
struct BSRConfig {
    int dim = 384;
    int depth = 12;
    bool stereo = true;
    int num_stems = 1;
    int time_transformer_depth = 1;
    int freq_transformer_depth = 1;
    int dim_head = 64;
    int heads = 8;
    int mask_estimator_depth = 1;
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

    std::vector<int> freqs_per_bands;  // number of frequency bins per band
    int num_bands = 0;                 // derived: freqs_per_bands.size()

    static BSRConfig from_json(const JsonValue& j);
};

class BSRoformer {
public:
    BSRoformer() = default;

    // Load model from weights file
    void load(const ModelWeights& weights);

    // Run inference: input [B, channels, samples], output [B, channels, samples]
    Tensor forward(const Tensor& audio);

    const BSRConfig& config() const { return cfg_; }

private:
    BSRConfig cfg_;

    // Precomputed band info (derived from freqs_per_bands)
    std::vector<int64_t> band_freq_dims_;  // input dim for each band: 2 * freqs_per_bands[b] * audio_channels

    // STFT window
    Tensor stft_window_;

    // Rotary embeddings cache (keyed by seq_len)
    std::unordered_map<int, std::pair<Tensor, Tensor>> rotary_cache_;

    // Model weights

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
        // No final_norm_gamma here (norm_output=False in BSRoformer)
    };

    struct DepthBlock {
        TransformerLayerWeights time_transformer;
        TransformerLayerWeights freq_transformer;
    };
    std::vector<DepthBlock> depth_blocks_;

    // Final norm applied after all transformer blocks (BSRoformer-specific)
    Tensor final_norm_gamma_;

    // Mask estimator weights
    struct MaskEstimatorWeights {
        struct BandMLP {
            // MLP layers: pairs of (weight, bias) for each Linear in the MLP
            // For depth=1: 2 linears (indices 0, 2 in Sequential)
            // For depth=2: 3 linears (indices 0, 2, 4 in Sequential)
            std::vector<Tensor> linear_w;
            std::vector<Tensor> linear_b;
        };
        std::vector<BandMLP> band_mlps;
    };
    std::vector<MaskEstimatorWeights> mask_estimators_;

    // Helper methods
    void compute_bands();
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
