// model_mel_band_roformer.cu - Complete MelBandRoformer inference implementation.
//
// This implements the MelBandRoformer model for music source separation,
// mirroring the Python forward pass exactly.

#include "model_mel_band_roformer.h"
#include "ops.h"
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cassert>

namespace cudasep {

namespace {

static double profile_elapsed_ms(const std::chrono::high_resolution_clock::time_point& start,
                                 const std::chrono::high_resolution_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static std::string profile_format_ms(double ms) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(ms >= 100.0 ? 1 : 3) << ms << " ms";
    return ss.str();
}

}

// ============================================================================
// MBRConfig
// ============================================================================

MBRConfig MBRConfig::from_json(const JsonValue& j) {
    MBRConfig c;
    c.dim                    = j.get_int("dim", 384);
    c.depth                  = j.get_int("depth", 6);
    c.stereo                 = j.get_bool("stereo", true);
    c.num_stems              = j.get_int("num_stems", 1);
    c.time_transformer_depth = j.get_int("time_transformer_depth", 1);
    c.freq_transformer_depth = j.get_int("freq_transformer_depth", 1);
    c.num_bands              = j.get_int("num_bands", 60);
    c.dim_head               = j.get_int("dim_head", 64);
    c.heads                  = j.get_int("heads", 8);
    c.mask_estimator_depth   = j.get_int("mask_estimator_depth", 2);
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
    return c;
}

// ============================================================================
// compute_mel_bands
// ============================================================================

void MelBandRoformer::compute_mel_bands() {
    int n_fft = cfg_.stft_n_fft;
    int num_bands = cfg_.num_bands;
    int num_fft_bins = n_fft / 2 + 1;  // e.g. 1025
    int audio_channels = cfg_.stereo ? 2 : 1;

    // 1. Create mel filterbank on GPU, copy to CPU for processing
    Tensor mel_fb_gpu = ops::mel_filterbank(cfg_.sample_rate, n_fft, num_bands);
    // mel_fb_gpu: [num_bands, num_fft_bins] float32
    std::vector<float> mel_fb = mel_fb_gpu.to_cpu_f32();

    // 2. Force mel_filter_bank[0][0] = 1 and mel_filter_bank[-1][-1] = 1
    mel_fb[0 * num_fft_bins + 0] = 1.0f;
    mel_fb[(num_bands - 1) * num_fft_bins + (num_fft_bins - 1)] = 1.0f;

    // 3. Binary threshold: freqs_per_band = (mel_fb > 0)
    // freqs_per_band[band][freq] = true/false
    std::vector<std::vector<bool>> freqs_per_band(num_bands, std::vector<bool>(num_fft_bins, false));
    for (int b = 0; b < num_bands; b++) {
        for (int f = 0; f < num_fft_bins; f++) {
            freqs_per_band[b][f] = (mel_fb[b * num_fft_bins + f] > 0.0f);
        }
    }

    // 4. Build freq_indices: for each band, collect active frequency indices
    // Python: repeated_freq_indices = repeat(arange(freqs), 'f -> b f', b=num_bands)
    //         freq_indices = repeated_freq_indices[freqs_per_band]
    std::vector<int64_t> freq_indices_mono;
    num_freqs_per_band_.resize(num_bands);
    for (int b = 0; b < num_bands; b++) {
        int count = 0;
        for (int f = 0; f < num_fft_bins; f++) {
            if (freqs_per_band[b][f]) {
                freq_indices_mono.push_back((int64_t)f);
                count++;
            }
        }
        num_freqs_per_band_[b] = count;
    }

    // 5. If stereo: duplicate indices as (f*2, f*2+1) pairs
    // Python: freq_indices = repeat(freq_indices, 'f -> f s', s=2)
    //         freq_indices = freq_indices * 2 + torch.arange(2)
    //         freq_indices = rearrange(freq_indices, 'f s -> (f s)')
    if (cfg_.stereo) {
        std::vector<int64_t> stereo_indices;
        stereo_indices.reserve(freq_indices_mono.size() * 2);
        for (int64_t f : freq_indices_mono) {
            stereo_indices.push_back(f * 2);
            stereo_indices.push_back(f * 2 + 1);
        }
        freq_indices_ = std::move(stereo_indices);
    } else {
        freq_indices_ = std::move(freq_indices_mono);
    }

    // 6. Build num_bands_per_freq
    // Python: num_bands_per_freq = reduce(freqs_per_band, 'b f -> f', 'sum')
    int total_freq = num_fft_bins * audio_channels;  // after stereo interleave
    num_bands_per_freq_.resize(total_freq);

    // First compute per-FFT-bin counts
    std::vector<int64_t> bands_per_fft_bin(num_fft_bins, 0);
    for (int b = 0; b < num_bands; b++) {
        for (int f = 0; f < num_fft_bins; f++) {
            if (freqs_per_band[b][f]) {
                bands_per_fft_bin[f]++;
            }
        }
    }

    // Expand to stereo if needed: 'b s f t c -> b (f s) t c'
    // After merging stereo into frequency: index pattern is (f0_ch0, f0_ch1, f1_ch0, f1_ch1, ...)
    if (cfg_.stereo) {
        for (int f = 0; f < num_fft_bins; f++) {
            num_bands_per_freq_[f * 2]     = bands_per_fft_bin[f];
            num_bands_per_freq_[f * 2 + 1] = bands_per_fft_bin[f];
        }
    } else {
        num_bands_per_freq_ = bands_per_fft_bin;
    }

    // 7. Compute band_freq_dims = 2 * num_freqs * audio_channels (for complex real+imag)
    // Python: freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in num_freqs_per_band)
    band_freq_dims_.resize(num_bands);
    for (int b = 0; b < num_bands; b++) {
        band_freq_dims_[b] = 2 * num_freqs_per_band_[b] * audio_channels;
    }

    // 8. Upload freq_indices and num_bands_per_freq to GPU
    freq_indices_gpu_ = Tensor::from_cpu_i64(freq_indices_.data(),
                                             {(int64_t)freq_indices_.size()});

    // num_bands_per_freq as float for division later
    // Python: denom = repeat(self.num_bands_per_freq, 'f -> (f r) 1', r=channels)
    // But the denom is used with the actual freq dimension after stereo merge
    std::vector<float> nbpf_float(num_bands_per_freq_.begin(), num_bands_per_freq_.end());
    num_bands_per_freq_gpu_ = Tensor::from_cpu_f32(nbpf_float.data(),
                                                    {(int64_t)total_freq});
}

// ============================================================================
// load
// ============================================================================

void MelBandRoformer::load(const ModelWeights& weights) {
    // 1. Parse config
    cfg_ = MBRConfig::from_json(weights.config());

    // 2. Load precomputed mel band data (from converter)
    if (weights.has("__precomputed__.freq_indices")) {
        // Use precomputed mel bands from Python (librosa Slaney scale)
        freq_indices_gpu_ = weights.get("__precomputed__.freq_indices");
        Tensor num_freqs_t = weights.get("__precomputed__.num_freqs_per_band");
        Tensor num_bands_f = weights.get("__precomputed__.num_bands_per_freq");

        // Copy to CPU vectors
        int num_bands = cfg_.num_bands;
        int audio_channels = cfg_.stereo ? 2 : 1;
        int num_fft_bins = cfg_.stft_n_fft / 2 + 1;

        // num_freqs_per_band (int64 on GPU -> copy to CPU)
        std::vector<int64_t> nfpb_cpu(num_bands);
        num_freqs_t.copy_to_cpu(nfpb_cpu.data(), num_bands * sizeof(int64_t));
        num_freqs_per_band_.resize(num_bands);
        for (int i = 0; i < num_bands; i++) num_freqs_per_band_[i] = nfpb_cpu[i];

        // num_bands_per_freq (int64 -> float for division)
        int total_freq = num_fft_bins * audio_channels;
        std::vector<int64_t> nbpf_cpu(num_fft_bins);
        num_bands_f.copy_to_cpu(nbpf_cpu.data(), num_fft_bins * sizeof(int64_t));

        // Expand to stereo
        num_bands_per_freq_.resize(total_freq);
        if (cfg_.stereo) {
            for (int f = 0; f < num_fft_bins; f++) {
                num_bands_per_freq_[f * 2]     = nbpf_cpu[f];
                num_bands_per_freq_[f * 2 + 1] = nbpf_cpu[f];
            }
        } else {
            for (int f = 0; f < num_fft_bins; f++) {
                num_bands_per_freq_[f] = nbpf_cpu[f];
            }
        }

        // band_freq_dims
        band_freq_dims_.resize(num_bands);
        for (int b = 0; b < num_bands; b++) {
            band_freq_dims_[b] = 2 * num_freqs_per_band_[b] * audio_channels;
        }

        // freq_indices CPU copy
        int64_t n_idx = freq_indices_gpu_.numel();
        freq_indices_.resize(n_idx);
        freq_indices_gpu_.copy_to_cpu(freq_indices_.data(), n_idx * sizeof(int64_t));

        // num_bands_per_freq as float GPU tensor
        std::vector<float> nbpf_float(num_bands_per_freq_.begin(), num_bands_per_freq_.end());
        num_bands_per_freq_gpu_ = Tensor::from_cpu_f32(nbpf_float.data(), {(int64_t)total_freq});

        std::cout << "[MelBandRoformer] Using precomputed mel bands from .csm file" << std::endl;
    } else {
        // Fallback: compute mel bands (may not match Python model)
        std::cerr << "[MelBandRoformer] WARNING: No precomputed mel bands, computing from scratch" << std::endl;
        compute_mel_bands();
    }

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
    // Model has cfg_.depth depth blocks, each with time and freq transformers.
    // No linear_transformer in standard configs (linear_transformer_depth=0).
    // Weight naming:
    //   layers.{d}.0 = time_transformer
    //   layers.{d}.1 = freq_transformer
    // Each transformer has:
    //   layers.{l}.0 = Attention
    //   layers.{l}.1 = FeedForward
    //   norm.gamma   = final RMSNorm

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
                std::string fprefix = lprefix + ".1";
                tw.ff_layers[l].norm_gamma = weights.get(fprefix + ".net.0.gamma");
                tw.ff_layers[l].linear1_w  = weights.get(fprefix + ".net.1.weight");
                tw.ff_layers[l].linear1_b  = weights.get(fprefix + ".net.1.bias");
                tw.ff_layers[l].linear2_w  = weights.get(fprefix + ".net.4.weight");
                tw.ff_layers[l].linear2_b  = weights.get(fprefix + ".net.4.bias");
            }
            tw.final_norm_gamma = weights.get(tprefix + ".norm.gamma");
        };

        std::string dprefix = "layers." + std::to_string(d);
        load_transformer(dprefix + ".0", cfg_.time_transformer_depth,
                         depth_blocks_[d].time_transformer);
        load_transformer(dprefix + ".1", cfg_.freq_transformer_depth,
                         depth_blocks_[d].freq_transformer);
    }

    // 6. Load mask estimator weights
    // MaskEstimator for each stem. Each has num_bands band MLPs.
    // MLP structure depends on mask_estimator_depth:
    //   depth=1: dims=(dim, dim_hidden, dim_in*2) -> 2 linears at Sequential indices 0, 2
    //   depth=2: dims=(dim, dim_hidden, dim_hidden, dim_in*2) -> 3 linears at indices 0, 2, 4
    //   depth=k: (k+1) linears at indices 0, 2, 4, ..., 2k
    // Outer MaskEstimator Sequential: [MLP(index 0), GLU(index 1)]
    // So weight prefix: mask_estimators.{s}.to_freqs.{b}.0.{2*i}.weight/bias

    int mlp_depth = cfg_.mask_estimator_depth;
    int num_mlp_linears = mlp_depth + 1;  // depth hidden layers + 1 output layer

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

    std::cout << "[MelBandRoformer] Model loaded: dim=" << dim
              << " depth=" << cfg_.depth
              << " bands=" << num_bands
              << " heads=" << heads
              << " stems=" << cfg_.num_stems
              << " stereo=" << cfg_.stereo
              << " freq_indices=" << freq_indices_.size()
              << std::endl;
}

// ============================================================================
// compute_rotary_cos_sin
// ============================================================================

Tensor MelBandRoformer::compute_rotary_cos_sin(int seq_len, int dim) {
    // Compute rotary embedding frequencies:
    // freqs_base = 1.0 / (10000 ^ (arange(0, dim, 2) / dim))
    // theta = outer(arange(seq_len), freqs_base) -> [seq_len, dim/2]
    // Returns stacked [2, seq_len, dim/2]: row 0 = cos(theta), row 1 = sin(theta)

    int half_dim = dim / 2;
    float theta_base = 10000.0f;

    // Compute on CPU then upload
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

Tensor MelBandRoformer::apply_attention(const Tensor& x, const AttentionWeights& w,
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
    // qkv: [B, N, 3*heads*dim_head] -> rearrange 'b n (qkv h d) -> qkv b h n d'
    int64_t B = qkv.size(0);
    int64_t N = qkv.size(1);
    Tensor qkv_r = qkv.reshape({B, N, 3, (int64_t)heads, (int64_t)dim_head});
    // [B, N, 3, H, D] -> permute to [3, B, H, N, D]
    qkv_r = qkv_r.permute({2, 0, 3, 1, 4});
    // Split into q, k, v
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
    // gates = sigmoid(linear(normed, to_gates_w, to_gates_b)) -> [B, N, heads]
    Tensor gates = ops::linear_sigmoid(normed, w.to_gates_w, w.to_gates_b);

    // gates: [B, N, H] -> permute to [B, H, N, 1] for broadcasting with out [B, H, N, D]
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

Tensor MelBandRoformer::apply_feedforward(const Tensor& x, const FeedForwardWeights& w) {
    // x: [B, N, dim]
    float scale = std::sqrt((float)cfg_.dim);

    // 1. RMS Norm
    Tensor h = ops::rms_norm(x, w.norm_gamma, scale);

    // 2. Fused Linear1 + GELU: [B, N, dim] -> [B, N, dim*4]
    h = ops::linear_gelu(h, w.linear1_w, w.linear1_b);

    // 3. Linear2: [B, N, dim*4] -> [B, N, dim]
    h = ops::linear(h, w.linear2_w, w.linear2_b);

    return h;
}

// ============================================================================
// apply_transformer
// ============================================================================

Tensor MelBandRoformer::apply_transformer(const Tensor& x,
                                            const TransformerLayerWeights& w,
                                            const Tensor& cos_freqs,
                                            const Tensor& sin_freqs) {
    // x: [B, N, dim]
    Tensor out = x;
    float scale = std::sqrt((float)cfg_.dim);

    int depth = (int)w.attn_layers.size();
    for (int i = 0; i < depth; i++) {
        // Attention + residual
        Tensor attn_out = apply_attention(out, w.attn_layers[i], cos_freqs, sin_freqs);
        out = out + attn_out;

        // FeedForward + residual
        Tensor ff_out = apply_feedforward(out, w.ff_layers[i]);
        out = out + ff_out;
    }

    // Final RMS norm
    out = ops::rms_norm(out, w.final_norm_gamma, scale);

    return out;
}

// ============================================================================
// apply_band_split
// ============================================================================

Tensor MelBandRoformer::apply_band_split(const Tensor& x) {
    // x: [B, T, total_freq_complex]
    // Split along last dim according to band_freq_dims_, apply per-band RMSNorm + Linear
    // Output: [B, T, num_bands, dim]

    int64_t B = x.size(0);
    int64_t T = x.size(1);
    int num_bands = cfg_.num_bands;
    int dim = cfg_.dim;
    float scale_rms = std::sqrt((float)band_freq_dims_[0]);  // will recompute per band

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

Tensor MelBandRoformer::apply_mask_estimator(const Tensor& x,
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

Tensor MelBandRoformer::forward(const Tensor& audio) {
    using Clock = std::chrono::high_resolution_clock;
    // audio: [B, channels, samples] or [B, samples]
    int audio_channels = cfg_.stereo ? 2 : 1;
    int num_stems = cfg_.num_stems;
    int num_bands = cfg_.num_bands;
    int dim = cfg_.dim;

    // ---- 1. Prepare audio ----
    Tensor raw_audio = audio;
    if (raw_audio.ndim() == 2) {
        // [B, T] -> [B, 1, T]
        raw_audio = raw_audio.unsqueeze(1);
    }

    int64_t batch = raw_audio.size(0);
    int64_t channels = raw_audio.size(1);
    int64_t raw_audio_length = raw_audio.size(2);
    int64_t istft_length = cfg_.match_input_audio_length ? raw_audio_length : -1;

    double stft_ms = 0.0;
    double gather_fold_ms = 0.0;
    double band_split_ms = 0.0;
    double time_transform_ms = 0.0;
    double freq_transform_ms = 0.0;
    double mask_estimator_ms = 0.0;
    double mask_merge_ms = 0.0;
    double istft_ms = 0.0;

    // ---- 2. STFT ----
    // Reshape for STFT: [B, channels, T] -> [B*channels, T]
    Tensor audio_flat = raw_audio.reshape({batch * channels, raw_audio_length});

    // STFT: [B*channels, F, T_stft, 2]
    auto stft_start = Clock::now();
    Tensor stft_repr = ops::stft(audio_flat, cfg_.stft_n_fft, cfg_.stft_hop_length,
                                 cfg_.stft_win_length, stft_window_, true,
                                 cfg_.stft_normalized);
    cudaDeviceSynchronize();
    stft_ms += profile_elapsed_ms(stft_start, Clock::now());

    int64_t F = stft_repr.size(1);   // num frequency bins (n_fft/2 + 1)
    int64_t T = stft_repr.size(2);   // num time frames

    // ---- 3. Reshape back to [B, channels, F, T, 2] ----
    stft_repr = stft_repr.reshape({batch, channels, F, T, 2});

    // ---- 4. Merge stereo into frequency ----
    // Python: rearrange(stft_repr, 'b s f t c -> b (f s) t c')
    // This interleaves: for freq f, put channel 0 then channel 1
    // Result: [B, F*channels, T, 2] where dim1 = (f0_ch0, f0_ch1, f1_ch0, f1_ch1, ...)
    if (channels > 1) {
        // [B, S, F, T, 2] -> [B, F, S, T, 2] -> [B, F*S, T, 2]
        stft_repr = stft_repr.permute({0, 2, 1, 3, 4}).contiguous();
        // Now [B, F, S, T, 2], reshape to [B, F*S, T, 2]
        stft_repr = stft_repr.reshape({batch, F * channels, T, 2});
    }

    int64_t total_freq = F * channels;

    // ---- 5. Gather frequency bands ----
    // Python: x = stft_repr[batch_arange, self.freq_indices]
    // Index select along dim=1 using freq_indices
    auto gather_fold_start = Clock::now();
    Tensor x = stft_repr.index_select(1, freq_indices_gpu_);
    // x: [B, total_band_freqs, T, 2]

    int64_t total_band_freqs = x.size(1);

    // ---- 6. Fold complex into frequency dimension ----
    // Python: x = rearrange(x, 'b f t c -> b t (f c)')
    x = x.permute({0, 2, 1, 3}).contiguous();  // [B, T, total_band_freqs, 2]
    x = x.reshape({batch, T, total_band_freqs * 2});
    cudaDeviceSynchronize();
    gather_fold_ms += profile_elapsed_ms(gather_fold_start, Clock::now());
    // x: [B, T, total_band_freqs * 2]

    // ---- 7. Band split ----
    auto band_split_start = Clock::now();
    x = apply_band_split(x);
    cudaDeviceSynchronize();
    band_split_ms += profile_elapsed_ms(band_split_start, Clock::now());
    // x: [B, T, num_bands, dim]

    // ---- 8. Transformer layers (depth iterations) ----
    // Store for skip connections
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
        // Python: x = rearrange(x, 'b t f d -> b f t d')
        auto time_start = Clock::now();
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
        cudaDeviceSynchronize();
        time_transform_ms += profile_elapsed_ms(time_start, Clock::now());

        // Freq attention: [B, T, F, D] -> [B*T, F, D]
        auto freq_start = Clock::now();
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
        cudaDeviceSynchronize();
        freq_transform_ms += profile_elapsed_ms(freq_start, Clock::now());

        // Store for skip connections
        if (cfg_.skip_connection) {
            skip_store[d] = x;
        }
    }

    // ---- 9. Mask estimation ----
    // x: [B, T, num_bands, dim]
    // Apply each mask estimator and stack
    std::vector<Tensor> stem_masks;
    stem_masks.reserve(num_stems);
    for (int s = 0; s < num_stems; s++) {
        auto mask_start = Clock::now();
        Tensor mask = apply_mask_estimator(x, mask_estimators_[s]);
        // mask: [B, T, total_freq_complex]
        cudaDeviceSynchronize();
        mask_estimator_ms += profile_elapsed_ms(mask_start, Clock::now());
        stem_masks.push_back(mask);
    }

    // Stack masks: [B, num_stems, T, total_freq_complex]
    auto merge_start = Clock::now();
    Tensor masks = Tensor::stack(stem_masks, 1);

    // Reshape masks: 'b n t (f c) -> b n f t c' where c=2
    // masks: [B, N, T, total_band_freqs * 2] -> [B, N, total_band_freqs, T, 2]
    masks = masks.reshape({batch, (int64_t)num_stems, T, total_band_freqs, 2});
    masks = masks.permute({0, 1, 3, 2, 4}).contiguous();
    // masks: [B, num_stems, total_band_freqs, T, 2]

    // ---- 10. Complex multiply and scatter for overlapping bands ----
    // stft_repr: [B, total_freq, T, 2] -> [B, 1, total_freq, T, 2]
    stft_repr = stft_repr.unsqueeze(1);

    // Complex multiply stft_repr * masks using scatter_add for overlapping bands
    // Python:
    //   stft_repr = view_as_complex(stft_repr)       -> [B, 1, total_freq, T] complex
    //   masks = view_as_complex(masks)               -> [B, N, total_band_freqs, T] complex
    //   stft_repr_exp = expand(stft_repr, n=N)       -> [B, N, total_freq, T] complex
    //   masks_summed = zeros_like(stft_repr_exp).scatter_add_(2, scatter_indices, masks)
    //   masks_averaged = masks_summed / denom
    //   stft_repr = stft_repr * masks_averaged
    //
    // We work in real representation. Complex multiply via ops::complex_mul.

    // Expand stft_repr to match stems: [B, 1, total_freq, T, 2] -> [B, N, total_freq, T, 2]
    Tensor stft_expanded = stft_repr.expand({batch, (int64_t)num_stems,
                                              total_freq, T, 2}).contiguous();

    // Create scatter destination (zeros): [B, N, total_freq, T, 2]
    Tensor masks_summed = Tensor::zeros({batch, (int64_t)num_stems, total_freq, T, 2});

    // Build scatter indices for the [B, N, total_band_freqs, T, 2] tensor along dim=2
    // Each element of freq_indices_ maps band_freq -> original_freq
    // We need indices tensor of shape [B, N, total_band_freqs, T, 2]
    // where indices[b, n, f, t, c] = freq_indices_[f]
    {
        // Create base indices [total_band_freqs] on GPU
        // Expand to [B, N, total_band_freqs, T, 2]
        Tensor idx_base = freq_indices_gpu_.reshape({1, 1, total_band_freqs, 1, 1});
        Tensor scatter_idx = idx_base.expand({batch, (int64_t)num_stems,
                                               total_band_freqs, T, 2}).contiguous();
        // scatter_idx must be Int64
        ops::scatter_add(masks_summed, 2, scatter_idx, masks);
    }

    // Average overlapping bands
    // Python: denom = repeat(self.num_bands_per_freq, 'f -> (f r) 1', r=channels)
    // num_bands_per_freq_gpu_: [total_freq]
    // We need denom: [total_freq, 1] -> broadcast to [B, N, total_freq, T, 2]
    // But since clamping and dividing, we'll reshape for broadcasting
    Tensor denom = num_bands_per_freq_gpu_.reshape({1, 1, total_freq, 1, 1})
                   .expand({batch, (int64_t)num_stems, total_freq, T, 2}).contiguous();
    denom.clamp_(1e-8f, 1e30f);  // clamp(min=1e-8)
    Tensor masks_averaged = masks_summed / denom;

    // Complex multiply: stft_repr * masks_averaged
    // For complex multiply in real representation:
    // stft_expanded: [B, N, total_freq, T, 2], masks_averaged: [B, N, total_freq, T, 2]
    // Merge the middle dims for complex_mul: reshape to [..., 2]
    int64_t spatial = batch * num_stems * total_freq * T;
    Tensor stft_flat = stft_expanded.reshape({spatial, 2});
    Tensor mask_flat = masks_averaged.reshape({spatial, 2});
    Tensor result_flat = ops::complex_mul(stft_flat, mask_flat);
    Tensor stft_result = result_flat.reshape({batch, (int64_t)num_stems, total_freq, T, 2});
    cudaDeviceSynchronize();
    mask_merge_ms += profile_elapsed_ms(merge_start, Clock::now());

    // ---- 11. Prepare for iSTFT ----
    // Python: stft_repr = rearrange(stft_repr, 'b n (f s) t -> (b n s) f t', s=audio_channels)
    // stft_result: [B, N, total_freq, T, 2] where total_freq = F * channels
    // Need to deinterleave stereo: total_freq = F * S, reshape to [B, N, F, S, T, 2]
    // then to [(B*N*S), F, T, 2]
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
    auto istft_start = Clock::now();
    Tensor recon_audio = ops::istft(stft_result, cfg_.stft_n_fft, cfg_.stft_hop_length,
                                    cfg_.stft_win_length, stft_window_, istft_length,
                                    true, cfg_.stft_normalized);
    cudaDeviceSynchronize();
    istft_ms += profile_elapsed_ms(istft_start, Clock::now());

    int64_t out_length = recon_audio.size(1);

    // ---- 13. Reshape output ----
    // Python: rearrange(recon_audio, '(b n s) t -> b n s t', b=batch, s=audio_channels, n=num_stems)
    recon_audio = recon_audio.reshape({batch, (int64_t)num_stems,
                                       (int64_t)audio_channels, out_length});

    if (num_stems == 1) {
        // Python: rearrange(recon_audio, 'b 1 s t -> b s t')
        recon_audio = recon_audio.squeeze(1);  // [B, channels, T]
    }

    if (profile_logger_) {
        profile_logger_("[基层][MBR] STFT: " + profile_format_ms(stft_ms));
        profile_logger_("[基层][MBR] 频带收集折叠: " + profile_format_ms(gather_fold_ms));
        profile_logger_("[基层][MBR] Band Split: " + profile_format_ms(band_split_ms));
        profile_logger_("[基层][MBR] Time Transformer: " + profile_format_ms(time_transform_ms));
        profile_logger_("[基层][MBR] Freq Transformer: " + profile_format_ms(freq_transform_ms));
        profile_logger_("[基层][MBR] Mask Estimator: " + profile_format_ms(mask_estimator_ms));
        profile_logger_("[基层][MBR] Mask Merge: " + profile_format_ms(mask_merge_ms));
        profile_logger_("[基层][MBR] iSTFT: " + profile_format_ms(istft_ms));
    }

    return recon_audio;
}

} // namespace cudasep
