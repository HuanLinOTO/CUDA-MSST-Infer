// model_bs_roformer.cu - Complete BSRoformer inference implementation.
//
// BSRoformer (Band-Split Roformer) for music source separation.
// Very similar to MelBandRoformer but uses fixed frequency bands (disjoint)
// instead of overlapping mel-scale bands. Key differences:
//   1. freqs_per_bands config instead of num_bands + mel filterbank
//   2. No scatter_add (bands are disjoint, no overlap)
//   3. final_norm after all transformer blocks (not per-transformer)
//   4. norm_output=False in each transformer (no internal final RMSNorm)
//   5. FeedForward net has linear2 at index 3 (no Dropout module)

#include "model_bs_roformer.h"
#include "ops.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cassert>

namespace cudasep {

// ============================================================================
// BSRConfig
// ============================================================================

BSRConfig BSRConfig::from_json(const JsonValue& j) {
    BSRConfig c;
    c.dim                    = j.get_int("dim", 384);
    c.depth                  = j.get_int("depth", 12);
    c.stereo                 = j.get_bool("stereo", true);
    c.num_stems              = j.get_int("num_stems", 1);
    c.time_transformer_depth = j.get_int("time_transformer_depth", 1);
    c.freq_transformer_depth = j.get_int("freq_transformer_depth", 1);
    c.dim_head               = j.get_int("dim_head", 64);
    c.heads                  = j.get_int("heads", 8);
    c.mask_estimator_depth   = j.get_int("mask_estimator_depth", 1);
    c.mlp_expansion_factor   = j.get_int("mlp_expansion_factor", 4);
    c.zero_dc                = j.get_bool("zero_dc", true);
    c.sample_rate            = j.get_int("sample_rate", 44100);
    c.stft_n_fft             = j.get_int("stft_n_fft", 2048);
    c.stft_hop_length        = j.get_int("stft_hop_length", 441);
    c.stft_win_length        = j.get_int("stft_win_length", 2048);
    c.stft_normalized        = j.get_bool("stft_normalized", false);
    c.match_input_audio_length = j.get_bool("match_input_audio_length", false);
    c.dim_freqs_in           = j.get_int("dim_freqs_in", 1025);
    c.skip_connection        = j.get_bool("skip_connection", false);

    // Parse freqs_per_bands from JSON array
    if (j.has("freqs_per_bands") && j["freqs_per_bands"].is_array()) {
        const JsonValue& arr = j["freqs_per_bands"];
        c.freqs_per_bands.resize(arr.size());
        for (size_t i = 0; i < arr.size(); i++) {
            c.freqs_per_bands[i] = arr[i].as_int();
        }
    }
    c.num_bands = (int)c.freqs_per_bands.size();

    return c;
}

// ============================================================================
// compute_bands
// ============================================================================

void BSRoformer::compute_bands() {
    int audio_channels = cfg_.stereo ? 2 : 1;
    int num_bands = cfg_.num_bands;

    // Compute band_freq_dims: for each band, the input dimension is
    // 2 * freqs_per_bands[b] * audio_channels (real+imag * stereo channels * freq bins)
    band_freq_dims_.resize(num_bands);
    int64_t total = 0;
    for (int b = 0; b < num_bands; b++) {
        band_freq_dims_[b] = 2 * (int64_t)cfg_.freqs_per_bands[b] * audio_channels;
        total += band_freq_dims_[b];
    }

    // Verify: total should equal F * audio_channels * 2
    int64_t expected = (int64_t)(cfg_.stft_n_fft / 2 + 1) * audio_channels * 2;
    if (total != expected) {
        std::cerr << "[BSRoformer] WARNING: sum(band_freq_dims)=" << total
                  << " != expected " << expected
                  << " (sum(freqs_per_bands)=" << std::accumulate(
                         cfg_.freqs_per_bands.begin(), cfg_.freqs_per_bands.end(), 0)
                  << ", F=" << (cfg_.stft_n_fft / 2 + 1)
                  << ", channels=" << audio_channels << ")" << std::endl;
    }
}

// ============================================================================
// load
// ============================================================================

void BSRoformer::load(const ModelWeights& weights) {
    // 1. Parse config
    cfg_ = BSRConfig::from_json(weights.config());

    // 2. Compute band dimensions from freqs_per_bands
    compute_bands();

    // 3. Create STFT window
    stft_window_ = ops::hann_window(cfg_.stft_win_length);

    int num_bands = cfg_.num_bands;
    int dim = cfg_.dim;
    int heads = cfg_.heads;
    int dim_head = cfg_.dim_head;
    int dim_inner = heads * dim_head;
    int ff_dim = dim * cfg_.mlp_expansion_factor;

    // 4. Load band_split weights
    band_split_layers_.resize(num_bands);
    for (int b = 0; b < num_bands; b++) {
        std::string prefix = "band_split.to_features." + std::to_string(b);
        band_split_layers_[b].norm_gamma = weights.get(prefix + ".0.gamma");
        band_split_layers_[b].linear_w   = weights.get(prefix + ".1.weight");
        band_split_layers_[b].linear_b   = weights.get(prefix + ".1.bias");
    }

    // 5. Load transformer weights
    // layers.{d}.0 = time_transformer
    // layers.{d}.1 = freq_transformer
    // Each transformer has norm_output=False, so no final RMSNorm per transformer.
    // FeedForward: net.0=RMSNorm, 1=Linear, 2=GELU, 3=Dropout, 4=Linear.

    depth_blocks_.resize(cfg_.depth);
    for (int d = 0; d < cfg_.depth; d++) {
        auto load_transformer = [&](const std::string& tprefix, int depth,
                                    TransformerLayerWeights& tw) {
            tw.attn_layers.resize(depth);
            tw.ff_layers.resize(depth);
            for (int l = 0; l < depth; l++) {
                std::string lprefix = tprefix + ".layers." + std::to_string(l);

                // Attention: lprefix.0.*
                std::string aprefix = lprefix + ".0";
                tw.attn_layers[l].norm_gamma = weights.get(aprefix + ".norm.gamma");
                tw.attn_layers[l].to_qkv_w   = weights.get(aprefix + ".to_qkv.weight");
                tw.attn_layers[l].to_gates_w  = weights.get(aprefix + ".to_gates.weight");
                tw.attn_layers[l].to_gates_b  = weights.get(aprefix + ".to_gates.bias");
                tw.attn_layers[l].to_out_w    = weights.get(aprefix + ".to_out.0.weight");

                // FeedForward: lprefix.1.net.*
                // BSRoformer: net.0=RMSNorm, net.1=Linear, net.2=GELU, net.3=Dropout, net.4=Linear
                std::string fprefix = lprefix + ".1";
                tw.ff_layers[l].norm_gamma = weights.get(fprefix + ".net.0.gamma");
                tw.ff_layers[l].linear1_w  = weights.get(fprefix + ".net.1.weight");
                tw.ff_layers[l].linear1_b  = weights.get(fprefix + ".net.1.bias");
                tw.ff_layers[l].linear2_w  = weights.get(fprefix + ".net.4.weight");
                tw.ff_layers[l].linear2_b  = weights.get(fprefix + ".net.4.bias");
            }
            // No final_norm_gamma per transformer (norm_output=False)
        };

        std::string dprefix = "layers." + std::to_string(d);
        load_transformer(dprefix + ".0", cfg_.time_transformer_depth,
                         depth_blocks_[d].time_transformer);
        load_transformer(dprefix + ".1", cfg_.freq_transformer_depth,
                         depth_blocks_[d].freq_transformer);
    }

    // 6. Load final_norm (BSRoformer has a separate final_norm after all depth blocks)
    final_norm_gamma_ = weights.get("final_norm.gamma");

    // 7. Load mask estimator weights
    // Same structure as MBR:
    //   mask_estimators.{s}.to_freqs.{b}.0.{2*i}.weight/bias
    // For depth=1: 2 linears at indices 0, 2
    // For depth=2: 3 linears at indices 0, 2, 4

    int mlp_depth = cfg_.mask_estimator_depth;
    int num_mlp_linears = mlp_depth;  // BSR MLP: dims=(in, *(hidden*(depth-1)), out)

    mask_estimators_.resize(cfg_.num_stems);
    for (int s = 0; s < cfg_.num_stems; s++) {
        mask_estimators_[s].band_mlps.resize(num_bands);
        for (int b = 0; b < num_bands; b++) {
            auto& mlp = mask_estimators_[s].band_mlps[b];
            mlp.linear_w.resize(num_mlp_linears);
            mlp.linear_b.resize(num_mlp_linears);
            for (int i = 0; i < num_mlp_linears; i++) {
                std::string prefix = "mask_estimators." + std::to_string(s) +
                                     ".to_freqs." + std::to_string(b) +
                                     ".0." + std::to_string(i * 2);
                mlp.linear_w[i] = weights.get(prefix + ".weight");
                mlp.linear_b[i] = weights.get(prefix + ".bias");
            }
        }
    }

    std::cout << "[BSRoformer] Model loaded: dim=" << dim
              << " depth=" << cfg_.depth
              << " bands=" << num_bands
              << " heads=" << heads
              << " stems=" << cfg_.num_stems
              << " stereo=" << cfg_.stereo
              << " mask_est_depth=" << mlp_depth
              << std::endl;
}

// ============================================================================
// compute_rotary_cos_sin
// ============================================================================

Tensor BSRoformer::compute_rotary_cos_sin(int seq_len, int dim) {
    // Compute rotary embedding frequencies:
    // freqs_base = 1.0 / (10000 ^ (arange(0, dim, 2) / dim))
    // theta = outer(arange(seq_len), freqs_base) -> [seq_len, dim/2]
    // Returns stacked [2, seq_len, dim/2]: row 0 = cos(theta), row 1 = sin(theta)

    int half_dim = dim / 2;
    float theta_base = 10000.0f;

    std::vector<float> cos_data(seq_len * half_dim);
    std::vector<float> sin_data(seq_len * half_dim);

    for (int n = 0; n < seq_len; n++) {
        for (int i = 0; i < half_dim; i++) {
            float freq = 1.0f / std::pow(theta_base, (float)(2 * i) / (float)dim);
            float angle = (float)n * freq;
            cos_data[n * half_dim + i] = std::cos(angle);
            sin_data[n * half_dim + i] = std::sin(angle);
        }
    }

    Tensor cos_t = Tensor::from_cpu_f32(cos_data.data(), {(int64_t)seq_len, (int64_t)half_dim});
    Tensor sin_t = Tensor::from_cpu_f32(sin_data.data(), {(int64_t)seq_len, (int64_t)half_dim});

    return Tensor::stack({cos_t, sin_t}, 0);  // [2, seq_len, dim/2]
}

// ============================================================================
// apply_attention
// ============================================================================

Tensor BSRoformer::apply_attention(const Tensor& x, const AttentionWeights& w,
                                    const Tensor& cos_freqs, const Tensor& sin_freqs) {
    // x: [B, N, dim]
    int dim = cfg_.dim;
    int heads = cfg_.heads;
    int dim_head = cfg_.dim_head;
    int dim_inner = heads * dim_head;
    float scale = std::sqrt((float)dim);

    // 1. RMS Norm
    Tensor normed = ops::rms_norm(x, w.norm_gamma, scale);

    // 2. QKV projection: [B, N, 3*dim_inner]
    Tensor qkv = ops::linear_no_bias(normed, w.to_qkv_w);

    // 3. Reshape to q, k, v: each [B, heads, N, dim_head]
    int64_t B = qkv.size(0);
    int64_t N = qkv.size(1);
    Tensor qkv_r = qkv.reshape({B, N, 3, (int64_t)heads, (int64_t)dim_head});
    // [B, N, 3, H, D] -> permute to [3, B, H, N, D]
    qkv_r = qkv_r.permute({2, 0, 3, 1, 4});
    Tensor q = qkv_r.slice(0, 0, 1).squeeze(0).contiguous();  // [B, H, N, D]
    Tensor k = qkv_r.slice(0, 1, 2).squeeze(0).contiguous();  // [B, H, N, D]
    Tensor v = qkv_r.slice(0, 2, 3).squeeze(0).contiguous();  // [B, H, N, D]

    // 4. Apply rotary embedding to q and k
    ops::apply_rotary_emb(q, k, cos_freqs, sin_freqs);

    // 5. Scaled dot-product attention
    float attn_scale = 1.0f / std::sqrt((float)dim_head);
    Tensor out = ops::scaled_dot_product_attention(q, k, v, attn_scale);
    // out: [B, H, N, D]

    // 6. Gating
    Tensor gates = ops::linear_sigmoid(normed, w.to_gates_w, w.to_gates_b);
    gates = gates.permute({0, 2, 1}).unsqueeze(-1);  // [B, H, N, 1]

    // 7. Apply gates
    out = out * gates;  // [B, H, N, D]

    // 8. Reshape to [B, N, dim_inner]
    out = out.permute({0, 2, 1, 3}).contiguous();  // [B, N, H, D]
    out = out.reshape({B, N, (int64_t)dim_inner});

    // 9. Output projection
    out = ops::linear_no_bias(out, w.to_out_w);  // [B, N, dim]

    return out;
}

// ============================================================================
// apply_feedforward
// ============================================================================

Tensor BSRoformer::apply_feedforward(const Tensor& x, const FeedForwardWeights& w) {
    // x: [B, N, dim]
    float scale = std::sqrt((float)cfg_.dim);

    // 1. RMS Norm
    Tensor h = ops::rms_norm(x, w.norm_gamma, scale);

    // 2. Fused Linear1 + GELU: [B, N, dim] -> [B, N, dim*expansion]
    h = ops::linear_gelu(h, w.linear1_w, w.linear1_b);

    // 3. Linear2: [B, N, dim*expansion] -> [B, N, dim]
    h = ops::linear(h, w.linear2_w, w.linear2_b);

    return h;
}

// ============================================================================
// apply_transformer
// ============================================================================

Tensor BSRoformer::apply_transformer(const Tensor& x,
                                      const TransformerLayerWeights& w,
                                      const Tensor& cos_freqs,
                                      const Tensor& sin_freqs) {
    // x: [B, N, dim]
    // BSRoformer: norm_output=False, so NO final RMSNorm per transformer
    Tensor out = x;

    int depth = (int)w.attn_layers.size();
    for (int i = 0; i < depth; i++) {
        // Attention + residual
        Tensor attn_out = apply_attention(out, w.attn_layers[i], cos_freqs, sin_freqs);
        out = out + attn_out;

        // FeedForward + residual
        Tensor ff_out = apply_feedforward(out, w.ff_layers[i]);
        out = out + ff_out;
    }

    // No final norm here (norm_output=False)
    return out;
}

// ============================================================================
// apply_band_split
// ============================================================================

Tensor BSRoformer::apply_band_split(const Tensor& x) {
    // x: [B, T, total_freq_complex]
    // Split along last dim according to band_freq_dims_, apply per-band RMSNorm + Linear
    // Output: [B, T, num_bands, dim]

    int64_t B = x.size(0);
    int64_t T = x.size(1);
    int num_bands = cfg_.num_bands;
    int dim = cfg_.dim;

    // Split x along last dimension by band_freq_dims_
    std::vector<int64_t> split_sizes(band_freq_dims_.begin(), band_freq_dims_.end());
    std::vector<Tensor> band_splits = x.split(2, split_sizes);

    // Process each band and collect results
    std::vector<Tensor> band_outputs;
    band_outputs.reserve(num_bands);

    for (int b = 0; b < num_bands; b++) {
        // band_splits[b]: [B, T, band_freq_dim]
        float band_scale = std::sqrt((float)band_freq_dims_[b]);
        Tensor band_normed = ops::rms_norm(band_splits[b], band_split_layers_[b].norm_gamma,
                                            band_scale);
        Tensor band_out = ops::linear(band_normed, band_split_layers_[b].linear_w,
                                       band_split_layers_[b].linear_b);
        // band_out: [B, T, dim]
        band_outputs.push_back(band_out);
    }

    // Stack along new dim: [B, T, num_bands, dim]
    Tensor result = Tensor::stack(band_outputs, 2);
    return result;
}

// ============================================================================
// apply_mask_estimator
// ============================================================================

Tensor BSRoformer::apply_mask_estimator(const Tensor& x,
                                         const MaskEstimatorWeights& w) {
    // x: [B, T, num_bands, dim]
    // For each band: MLP layers -> Tanh (between linears) -> GLU at end
    // Output: [B, T, total_freq_complex]

    int64_t B = x.size(0);
    int64_t T = x.size(1);
    int num_bands = cfg_.num_bands;

    std::vector<Tensor> band_outputs;
    band_outputs.reserve(num_bands);

    for (int b = 0; b < num_bands; b++) {
        // Extract band features: x[:, :, b, :] -> [B, T, dim]
        Tensor band_x = x.slice(2, b, b + 1).squeeze(2).contiguous();

        const auto& mlp = w.band_mlps[b];
        int num_linears = (int)mlp.linear_w.size();

        Tensor h = band_x;
        for (int i = 0; i < num_linears; i++) {
            h = ops::linear(h, mlp.linear_w[i], mlp.linear_b[i]);
            // Apply Tanh activation between linear layers (not after the last one)
            if (i < num_linears - 1) {
                h = ops::tanh_act(h);
            }
        }

        // Apply GLU: splits last dim in half, applies sigmoid gate
        // h: [B, T, band_freq_dim * 2] -> [B, T, band_freq_dim]
        h = ops::glu(h, -1);

        band_outputs.push_back(h);
    }

    // Concatenate all bands along last dim
    Tensor result = Tensor::cat(band_outputs, -1);
    return result;  // [B, T, total_freq_complex]
}

// ============================================================================
// forward
// ============================================================================

Tensor BSRoformer::forward(const Tensor& audio) {
    // audio: [B, channels, samples] or [B, samples]
    int audio_channels = cfg_.stereo ? 2 : 1;
    int num_stems = cfg_.num_stems;
    int num_bands = cfg_.num_bands;
    int dim = cfg_.dim;

    // ---- 1. Prepare audio ----
    Tensor raw_audio = audio;
    if (raw_audio.ndim() == 2) {
        raw_audio = raw_audio.unsqueeze(1);
    }

    int64_t batch = raw_audio.size(0);
    int64_t channels = raw_audio.size(1);
    int64_t raw_audio_length = raw_audio.size(2);
    int64_t istft_length = cfg_.match_input_audio_length ? raw_audio_length : -1;

    // ---- 2. STFT ----
    // Reshape for STFT: [B, channels, T] -> [B*channels, T]
    Tensor audio_flat = raw_audio.reshape({batch * channels, raw_audio_length});

    // STFT: [B*channels, F, T_stft, 2]
    Tensor stft_repr = ops::stft(audio_flat, cfg_.stft_n_fft, cfg_.stft_hop_length,
                                  cfg_.stft_win_length, stft_window_, true,
                                  cfg_.stft_normalized);

    int64_t F = stft_repr.size(1);   // num frequency bins (n_fft/2 + 1)
    int64_t T = stft_repr.size(2);   // num time frames

    // ---- 3. Reshape back to [B, channels, F, T, 2] ----
    stft_repr = stft_repr.reshape({batch, channels, F, T, 2});

    // ---- 4. Merge stereo into frequency ----
    // Python: rearrange(stft_repr, 'b s f t c -> b (f s) t c')
    // Interleaves: (f0_ch0, f0_ch1, f1_ch0, f1_ch1, ...)
    // Result: [B, F*channels, T, 2]
    if (channels > 1) {
        // [B, S, F, T, 2] -> [B, F, S, T, 2] -> [B, F*S, T, 2]
        stft_repr = stft_repr.permute({0, 2, 1, 3, 4}).contiguous();
        stft_repr = stft_repr.reshape({batch, F * channels, T, 2});
    }

    int64_t total_freq = F * channels;

    // ---- 5. Fold complex into frequency dimension ----
    // Python: x = rearrange(x, 'b f t c -> b t (f c)')
    // No index_select needed (BSRoformer uses all freqs sequentially, no overlap)
    Tensor x = stft_repr.permute({0, 2, 1, 3}).contiguous();  // [B, T, total_freq, 2]
    x = x.reshape({batch, T, total_freq * 2});
    // x: [B, T, total_freq * 2]

    // ---- 6. Band split ----
    x = apply_band_split(x);
    // x: [B, T, num_bands, dim]

    // ---- 7. Transformer layers (depth iterations) ----
    std::vector<Tensor> skip_store;
    if (cfg_.skip_connection) {
        skip_store.resize(cfg_.depth);
    }

    for (int d = 0; d < cfg_.depth; d++) {
        // Skip connections: sum all previous
        if (cfg_.skip_connection) {
            for (int j = 0; j < d; j++) {
                x = x + skip_store[j];
            }
        }

        // Time attention: [B, T, F, D] -> [B, F, T, D] -> [B*F, T, D]
        x = x.permute({0, 2, 1, 3}).contiguous();  // [B, num_bands, T, dim]
        int64_t BF = batch * num_bands;
        x = x.reshape({BF, T, (int64_t)dim});      // [B*num_bands, T, dim]

        // Compute rotary embeddings for time dimension (cached)
        auto& time_cached = rotary_cache_[(int)T];
        if (time_cached.first.numel() == 0) {
            Tensor time_rot = compute_rotary_cos_sin((int)T, cfg_.dim_head);
            time_cached.first = time_rot.slice(0, 0, 1).squeeze(0).contiguous();
            time_cached.second = time_rot.slice(0, 1, 2).squeeze(0).contiguous();
        }
        const Tensor& time_cos = time_cached.first;
        const Tensor& time_sin = time_cached.second;

        x = apply_transformer(x, depth_blocks_[d].time_transformer, time_cos, time_sin);

        // Reshape back: [B*F, T, D] -> [B, F, T, D] -> [B, T, F, D]
        x = x.reshape({batch, (int64_t)num_bands, T, (int64_t)dim});
        x = x.permute({0, 2, 1, 3}).contiguous();  // [B, T, num_bands, dim]

        // Freq attention: [B, T, F, D] -> [B*T, F, D]
        int64_t BT = batch * T;
        x = x.reshape({BT, (int64_t)num_bands, (int64_t)dim});

        // Compute rotary embeddings for freq dimension (cached)
        auto& freq_cached = rotary_cache_[num_bands];
        if (freq_cached.first.numel() == 0) {
            Tensor freq_rot = compute_rotary_cos_sin(num_bands, cfg_.dim_head);
            freq_cached.first = freq_rot.slice(0, 0, 1).squeeze(0).contiguous();
            freq_cached.second = freq_rot.slice(0, 1, 2).squeeze(0).contiguous();
        }
        const Tensor& freq_cos = freq_cached.first;
        const Tensor& freq_sin = freq_cached.second;

        x = apply_transformer(x, depth_blocks_[d].freq_transformer, freq_cos, freq_sin);

        // Reshape back: [B*T, F, D] -> [B, T, F, D]
        x = x.reshape({batch, T, (int64_t)num_bands, (int64_t)dim});

        // Store for skip connections
        if (cfg_.skip_connection) {
            skip_store[d] = x;
        }
    }

    // ---- 8. Final norm (BSRoformer-specific) ----
    // x: [B, T, num_bands, dim] - apply RMSNorm over last dimension
    float norm_scale = std::sqrt((float)dim);
    x = ops::rms_norm(x, final_norm_gamma_, norm_scale);

    // ---- 9. Mask estimation ----
    // x: [B, T, num_bands, dim]
    std::vector<Tensor> stem_masks;
    stem_masks.reserve(num_stems);
    for (int s = 0; s < num_stems; s++) {
        Tensor mask = apply_mask_estimator(x, mask_estimators_[s]);
        // mask: [B, T, total_freq * 2]
        stem_masks.push_back(mask);
    }

    // Stack masks: [B, num_stems, T, total_freq * 2]
    Tensor masks = Tensor::stack(stem_masks, 1);

    // Reshape masks: [B, N, T, total_freq*2] -> [B, N, T, total_freq, 2] -> [B, N, total_freq, T, 2]
    masks = masks.reshape({batch, (int64_t)num_stems, T, total_freq, 2});
    masks = masks.permute({0, 1, 3, 2, 4}).contiguous();
    // masks: [B, num_stems, total_freq, T, 2]

    // ---- 10. Complex multiply (no scatter needed - bands are disjoint) ----
    // stft_repr: [B, total_freq, T, 2] -> expand to [B, N, total_freq, T, 2]
    stft_repr = stft_repr.unsqueeze(1);
    Tensor stft_expanded = stft_repr.expand({batch, (int64_t)num_stems,
                                              total_freq, T, 2}).contiguous();

    // Complex multiply: stft_repr * masks
    int64_t spatial = batch * num_stems * total_freq * T;
    Tensor stft_flat = stft_expanded.reshape({spatial, 2});
    Tensor mask_flat = masks.reshape({spatial, 2});
    Tensor result_flat = ops::complex_mul(stft_flat, mask_flat);
    Tensor stft_result = result_flat.reshape({batch, (int64_t)num_stems, total_freq, T, 2});

    // ---- 11. Prepare for iSTFT ----
    // Deinterleave stereo: [B, N, F*S, T, 2] -> [(B*N*S), F, T, 2]
    if (audio_channels > 1) {
        // [B, N, F*S, T, 2] -> [B, N, F, S, T, 2]
        stft_result = stft_result.reshape({batch, (int64_t)num_stems, F,
                                           (int64_t)audio_channels, T, 2});
        // -> [B, N, S, F, T, 2]
        stft_result = stft_result.permute({0, 1, 3, 2, 4, 5}).contiguous();
        // -> [(B*N*S), F, T, 2]
        stft_result = stft_result.reshape({batch * num_stems * audio_channels, F, T, 2});
    } else {
        // [B, N, F, T, 2] -> [(B*N), F, T, 2]
        stft_result = stft_result.reshape({batch * num_stems, F, T, 2});
    }

    // Zero DC component if requested
    if (cfg_.zero_dc) {
        stft_result = ops::index_fill(stft_result, 1, 0, 0.0f);
    }

    // ---- 12. iSTFT ----
    Tensor recon_audio = ops::istft(stft_result, cfg_.stft_n_fft, cfg_.stft_hop_length,
                                     cfg_.stft_win_length, stft_window_, istft_length,
                                     true, cfg_.stft_normalized);
    // recon_audio: [(B*N*S), signal_length]

    int64_t out_length = recon_audio.size(1);

    // ---- 13. Reshape output ----
    recon_audio = recon_audio.reshape({batch, (int64_t)num_stems,
                                       (int64_t)audio_channels, out_length});

    if (num_stems == 1) {
        recon_audio = recon_audio.squeeze(1);  // [B, channels, T]
    }

    return recon_audio;
}

} // namespace cudasep
