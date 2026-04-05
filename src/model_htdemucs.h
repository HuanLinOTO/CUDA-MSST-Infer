#pragma once
#include "tensor.h"
#include "weights.h"
#include <vector>
#include <string>
#include <unordered_map>

namespace cudasep {

struct HTDemucsConfig {
    int num_sources = 4;
    int audio_channels = 2;
    int channels = 48;
    int growth = 2;
    int depth = 4;
    int nfft = 4096;
    int num_subbands = 1;
    bool cac = true;
    int kernel_size = 8;
    int stride = 4;
    int time_stride = 2;
    int context = 1;       // decoder rewrite context
    int context_enc = 0;   // encoder rewrite context
    int norm_starts = 4;
    int norm_groups = 4;
    int dconv_mode = 3;    // 1=enc only, 2=dec only, 3=both
    int dconv_depth = 2;
    int dconv_comp = 8;
    float dconv_init = 1e-3f;
    int bottom_channels = 512;
    int t_layers = 5;
    int t_heads = 8;
    float t_hidden_scale = 4.0f;
    float freq_emb = 0.2f;
    float emb_scale = 10.0f;
    bool rewrite = true;
    int samplerate = 44100;
    float segment = 11.0f;
    bool use_train_segment = true;
    
    int hop_length() const { return nfft / 4; }
    
    static HTDemucsConfig from_json(const JsonValue& j);
};

class HTDemucs {
public:
    HTDemucs() = default;
    void load(const ModelWeights& weights);
    Tensor forward(const Tensor& mix);
    const HTDemucsConfig& config() const { return cfg_; }
    
private:
    HTDemucsConfig cfg_;
    Tensor stft_window_;
    
    // DConv layer weights (shared structure for 1D processing)
    struct DConvLayer {
        // Each depth level: compressor conv + group_norm + expander conv + group_norm + layer_scale
        struct Level {
            Tensor conv1_w, conv1_b;     // Conv1d compressor: [C/comp, C, 3]
            Tensor gn1_w, gn1_b;         // GroupNorm(1, C/comp)
            Tensor conv2_w, conv2_b;     // Conv1d expander: [2*C, C/comp, 1]
            Tensor gn2_w, gn2_b;         // GroupNorm(1, 2*C)
            Tensor layer_scale;          // [C]
        };
        std::vector<Level> levels;  // dconv_depth levels
    };
    
    // Encoder layer weights
    struct EncLayerWeights {
        Tensor conv_w, conv_b;           // main convolution
        Tensor rewrite_w, rewrite_b;     // 1x1 rewrite
        DConvLayer dconv;
        bool use_norm = false;           // GroupNorm after conv
        bool is_freq = true;             // true=2D conv, false=1D conv
    };
    
    // Decoder layer weights
    struct DecLayerWeights {
        Tensor conv_tr_w, conv_tr_b;     // transposed convolution
        Tensor rewrite_w, rewrite_b;     // rewrite (3x3 for freq, 3 for time)
        DConvLayer dconv;
        bool use_norm = false;
        bool is_freq = true;
        bool is_last = false;            // last layer: no GELU
    };
    
    // CrossTransformer layer weights (self-attention or cross-attention)
    struct TransformerLayer {
        bool is_cross = false;
        // Attention weights (in_proj_weight combines Q,K,V)
        Tensor attn_in_proj_w, attn_in_proj_b;  // [3*dim, dim], [3*dim]
        Tensor attn_out_proj_w, attn_out_proj_b; // [dim, dim], [dim]
        // FFN
        Tensor linear1_w, linear1_b;     // [dim*hidden_scale, dim]
        Tensor linear2_w, linear2_b;     // [dim, dim*hidden_scale]
        // Norms
        Tensor norm1_w, norm1_b;         // LayerNorm for Q (or pre-attention)
        Tensor norm2_w, norm2_b;         // LayerNorm for K (cross) or pre-FFN (self)
        Tensor norm3_w, norm3_b;         // LayerNorm pre-FFN (cross only)
        Tensor norm_out_w, norm_out_b;   // GroupNorm(1) at end
        // LayerScale
        Tensor gamma_1, gamma_2;         // [dim]
    };
    
    // Frequency embedding
    Tensor freq_emb_weight_;  // [num_freqs, channels]
    
    // Channel adapters
    Tensor ch_up_w_, ch_up_b_;           // freq upsampler
    Tensor ch_down_w_, ch_down_b_;       // freq downsampler
    Tensor ch_up_t_w_, ch_up_t_b_;       // time upsampler
    Tensor ch_down_t_w_, ch_down_t_b_;   // time downsampler
    
    // Encoder/decoder storage
    std::vector<EncLayerWeights> encoder_;
    std::vector<EncLayerWeights> tencoder_;
    std::vector<DecLayerWeights> decoder_;
    std::vector<DecLayerWeights> tdecoder_;
    
    // CrossTransformer
    Tensor ct_norm_in_w_, ct_norm_in_b_;     // LayerNorm for freq input
    Tensor ct_norm_in_t_w_, ct_norm_in_t_b_; // LayerNorm for time input
    std::vector<TransformerLayer> ct_layers_;  // freq branch layers
    std::vector<TransformerLayer> ct_layers_t_; // time branch layers
    
    // Helper methods
    Tensor apply_dconv(const Tensor& x, const DConvLayer& dconv);
    Tensor apply_enc_layer(const Tensor& x, const EncLayerWeights& w, const Tensor& inject = Tensor());
    std::pair<Tensor, Tensor> apply_dec_layer(const Tensor& x, const Tensor& skip, 
                                               const DecLayerWeights& w, int64_t target_length);
    
    Tensor apply_self_attention(const Tensor& x, const TransformerLayer& w);
    Tensor apply_cross_attention(const Tensor& q_input, const Tensor& kv_input, const TransformerLayer& w);
    
    Tensor spec(const Tensor& x);           // STFT with padding
    Tensor ispec(const Tensor& z, int64_t length); // iSTFT with unpadding
    Tensor magnitude(const Tensor& z);       // CaC conversion
    Tensor mask(const Tensor& z, const Tensor& m); // CaC mask application
    
    Tensor create_sin_embedding_1d(int length, int dim);
    Tensor create_sin_embedding_2d(int dim, int height, int width);
    Tensor pad1d(const Tensor& x, int64_t pad_left, int64_t pad_right);

    // Cached position embeddings (computed once per unique shape)
    std::unordered_map<int64_t, Tensor> pos_emb_1d_cache_;
    std::unordered_map<int64_t, Tensor> pos_emb_2d_cache_;
    Tensor freq_arange_cache_;
};

} // namespace cudasep
