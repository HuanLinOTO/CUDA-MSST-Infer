// model_mdx23c.cu — Complete MDX23C (TFC_TDF_net) inference implementation.
//
// MDX23C is a U-Net style model operating in the frequency domain:
//   STFT → first_conv → encoder (TFC_TDF + downscale) → bottleneck →
//   decoder (upscale + skip + TFC_TDF) → final_conv → iSTFT
//
// Each TFC_TDF block:  tfc1 → tdf residual → tfc2 → shortcut residual
//   tfc1/tfc2 = Norm → Act → Conv2d(3×3)
//   tdf        = Norm → Act → Linear(f→f/bn) → Norm → Act → Linear(f/bn→f)

#include "model_mdx23c.h"
#include "ops.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cassert>

namespace cudasep {

// ============================================================================
// MDX23CConfig
// ============================================================================

MDX23CConfig MDX23CConfig::from_json(const JsonValue& j) {
    MDX23CConfig c;

    // Audio
    c.chunk_size   = j.get_int("chunk_size", 261120);
    c.dim_f        = j.get_int("dim_f", 4096);
    c.dim_t        = j.get_int("dim_t", 256);
    c.hop_length   = j.get_int("hop_length", 1024);
    c.n_fft        = j.get_int("n_fft", 8192);
    c.sample_rate  = j.get_int("sample_rate", 44100);

    // Audio channels: _flatten_config may overwrite "num_channels" with the
    // model value, so prefer an explicit "audio_num_channels" key; fall back
    // to 2 (stereo) otherwise.
    c.num_channels = j.get_int("audio_num_channels", 2);

    // Model
    c.act_type             = j.get_string("act", "gelu");
    c.norm_type            = j.get_string("norm", "BatchNorm");
    c.bottleneck_factor    = j.get_int("bottleneck_factor", 4);
    c.growth               = j.get_int("growth", 128);
    c.num_blocks_per_scale = j.get_int("num_blocks_per_scale", 2);
    c.num_channels_model   = j.get_int("num_channels", 128);
    c.num_scales           = j.get_int("num_scales", 5);
    c.num_subbands         = j.get_int("num_subbands", 4);
    c.num_target_instruments = j.get_int("num_target_instruments", 1);

    // Scale: array [h, w] or separate scalar keys
    if (j.has("scale") && j["scale"].is_array()) {
        c.scale_h = j["scale"][0].as_int();
        c.scale_w = j["scale"][1].as_int();
    } else {
        int s = j.get_int("scale", 2);
        c.scale_h = j.get_int("scale_h", s);
        c.scale_w = j.get_int("scale_w", s);
    }

    // Parse GroupNorm groups from e.g. "GroupNorm4"
    if (c.norm_type.size() > 9 &&
        c.norm_type.substr(0, 9) == "GroupNorm") {
        c.group_norm_groups = std::stoi(c.norm_type.substr(9));
    }

    // Parse ELU alpha from e.g. "elu1.0" or plain "elu"
    if (c.act_type.size() >= 3 && c.act_type.substr(0, 3) == "elu") {
        std::string a = c.act_type.substr(3);
        c.elu_alpha = a.empty() ? 1.0f : std::stof(a);
    }

    return c;
}

// ============================================================================
// load_norm — load weight/bias (and running stats for BatchNorm)
// ============================================================================

void MDX23C::load_norm(const ModelWeights& w,
                       const std::string& prefix,
                       NormWeights& nw) {
    nw.weight = w.get(prefix + ".weight");
    nw.bias   = w.get(prefix + ".bias");
    if (norm_type_ == NormType::BatchNorm) {
        nw.running_mean = w.get(prefix + ".running_mean");
        nw.running_var  = w.get(prefix + ".running_var");
    }
}

// ============================================================================
// load_tfc_tdf — load l TFC_TDF sub-blocks
// ============================================================================

void MDX23C::load_tfc_tdf(const ModelWeights& w,
                           const std::string& prefix,
                           int num_blocks,
                           TFC_TDF_Weights& out) {
    out.blocks.resize(num_blocks);
    for (int j = 0; j < num_blocks; j++) {
        std::string bp = prefix + ".blocks." + std::to_string(j);
        auto& b = out.blocks[j];

        // tfc1: Sequential[0=norm, 1=act, 2=Conv2d]
        load_norm(w, bp + ".tfc1.0", b.tfc1_norm);
        b.tfc1_conv_w = w.get(bp + ".tfc1.2.weight");

        // tdf: Sequential[0=norm, 1=act, 2=Linear, 3=norm, 4=act, 5=Linear]
        load_norm(w, bp + ".tdf.0", b.tdf_norm1);
        b.tdf_linear1_w = w.get(bp + ".tdf.2.weight");
        load_norm(w, bp + ".tdf.3", b.tdf_norm2);
        b.tdf_linear2_w = w.get(bp + ".tdf.5.weight");

        // tfc2: Sequential[0=norm, 1=act, 2=Conv2d]
        load_norm(w, bp + ".tfc2.0", b.tfc2_norm);
        b.tfc2_conv_w = w.get(bp + ".tfc2.2.weight");

        // shortcut: Conv2d (no bias)
        b.shortcut_w = w.get(bp + ".shortcut.weight");
    }
}

// ============================================================================
// apply_norm
// ============================================================================

Tensor MDX23C::apply_norm(const Tensor& x, const NormWeights& nw) {
    switch (norm_type_) {
    case NormType::BatchNorm:
        return ops::batch_norm(x, nw.running_mean, nw.running_var,
                               nw.weight, nw.bias);
    case NormType::InstanceNorm:
        return ops::instance_norm(x, nw.weight, nw.bias);
    case NormType::GroupNorm:
        return ops::group_norm(x, cfg_.group_norm_groups,
                               nw.weight, nw.bias);
    }
    throw std::runtime_error("MDX23C: unknown norm type");
}

// ============================================================================
// apply_act
// ============================================================================

Tensor MDX23C::apply_act(const Tensor& x) {
    if (cfg_.act_type == "gelu") return ops::gelu(x);
    if (cfg_.act_type == "relu") return ops::relu(x);

    // ELU(x, α) = relu(x) + α * (exp(min(x, 0)) − 1)
    if (cfg_.act_type.size() >= 3 && cfg_.act_type.substr(0, 3) == "elu") {
        float alpha = cfg_.elu_alpha;
        Tensor pos = ops::relu(x);
        Tensor neg = x.clone();
        neg.clamp_(-1e30f, 0.0f);             // min(x, 0)
        Tensor neg_part = ops::exp(neg);
        neg_part.add_scalar_(-1.0f);           // exp(min(x,0)) − 1
        neg_part.mul_scalar_(alpha);
        return pos + neg_part;
    }

    throw std::runtime_error("MDX23C: unsupported activation: " + cfg_.act_type);
}

// ============================================================================
// apply_tfc_tdf — forward through l TFC_TDF sub-blocks
// ============================================================================

Tensor MDX23C::apply_tfc_tdf(const Tensor& x, const TFC_TDF_Weights& w) {
    Tensor out = x;

    for (const auto& blk : w.blocks) {
        // ---- shortcut: Conv2d(in_c → c, 1×1) ----
        Tensor s = ops::conv2d(out, blk.shortcut_w, no_bias_);

        // ---- tfc1: Norm → Act → Conv2d(in_c → c, 3×3, pad=1) ----
        Tensor h = apply_norm(out, blk.tfc1_norm);
        h = apply_act(h);
        h = ops::conv2d(h, blk.tfc1_conv_w, no_bias_,
                        /*stride_h=*/1, /*stride_w=*/1,
                        /*pad_h=*/1,    /*pad_w=*/1);

        // ---- tdf (residual): Norm → Act → Linear → Norm → Act → Linear ----
        Tensor t = apply_norm(h, blk.tdf_norm1);
        t = apply_act(t);
        t = ops::linear_no_bias(t, blk.tdf_linear1_w);   // f → f/bn
        t = apply_norm(t, blk.tdf_norm2);
        t = apply_act(t);
        t = ops::linear_no_bias(t, blk.tdf_linear2_w);   // f/bn → f

        h = h + t;

        // ---- tfc2: Norm → Act → Conv2d(c → c, 3×3, pad=1) ----
        h = apply_norm(h, blk.tfc2_norm);
        h = apply_act(h);
        h = ops::conv2d(h, blk.tfc2_conv_w, no_bias_,
                        /*stride_h=*/1, /*stride_w=*/1,
                        /*pad_h=*/1,    /*pad_w=*/1);

        // ---- residual ----
        out = h + s;
    }

    return out;
}

// ============================================================================
// cac2cws / cws2cac — subband rearrangement
// ============================================================================

Tensor MDX23C::cac2cws(const Tensor& x) {
    int k = cfg_.num_subbands;
    if (k == 1) return x;
    // [B, c, f, t] → [B, c*k, f/k, t]
    int64_t b = x.size(0), c = x.size(1), f = x.size(2), t = x.size(3);
    return x.reshape({b, c * k, f / k, t});
}

Tensor MDX23C::cws2cac(const Tensor& x) {
    int k = cfg_.num_subbands;
    if (k == 1) return x;
    // [B, c*k, f/k, t] → [B, c, f*k, t]  where c = c_orig*k, c_orig = c_total/k
    int64_t b = x.size(0), ck = x.size(1), fk = x.size(2), t = x.size(3);
    int64_t c_orig = ck / k;
    return x.reshape({b, c_orig, fk * k, t});
}

// ============================================================================
// load
// ============================================================================

void MDX23C::load(const ModelWeights& weights) {
    // ---- 1. Parse config ---------------------------------------------------
    cfg_ = MDX23CConfig::from_json(weights.config());

    // ---- 2. Determine norm type --------------------------------------------
    if (cfg_.norm_type == "BatchNorm") {
        norm_type_ = NormType::BatchNorm;
    } else if (cfg_.norm_type == "InstanceNorm") {
        norm_type_ = NormType::InstanceNorm;
    } else if (cfg_.norm_type.find("GroupNorm") != std::string::npos) {
        norm_type_ = NormType::GroupNorm;
    } else {
        throw std::runtime_error("MDX23C: unknown norm type: " + cfg_.norm_type);
    }

    // ---- 3. Create STFT window ---------------------------------------------
    stft_window_ = ops::hann_window(cfg_.n_fft);

    // ---- 4. Load first_conv and infer audio channels / num_targets ----------
    first_conv_w_ = weights.get("first_conv.weight");
    // first_conv.weight: [c, dim_c, 1, 1]  where dim_c = num_subbands * num_channels * 2
    int dim_c_inferred = (int)first_conv_w_.size(1);
    cfg_.num_channels = dim_c_inferred / (cfg_.num_subbands * 2);
    if (cfg_.num_channels < 1) cfg_.num_channels = 2;

    final_conv2_w_ = weights.get("final_conv.2.weight");
    // final_conv.2.weight: [num_targets * dim_c, c, 1, 1]
    int dim_c = cfg_.num_subbands * cfg_.num_channels * 2;
    int final_out = (int)final_conv2_w_.size(0);
    cfg_.num_target_instruments = final_out / dim_c;
    if (cfg_.num_target_instruments < 1) cfg_.num_target_instruments = 1;

    int n = cfg_.num_scales;
    int l = cfg_.num_blocks_per_scale;

    // ---- 5. Load encoder blocks --------------------------------------------
    encoder_blocks_.resize(n);
    for (int i = 0; i < n; i++) {
        std::string ep = "encoder_blocks." + std::to_string(i);
        load_tfc_tdf(weights, ep + ".tfc_tdf", l, encoder_blocks_[i].tfc_tdf);
        load_norm(weights, ep + ".downscale.conv.0", encoder_blocks_[i].downscale_norm);
        encoder_blocks_[i].downscale_conv_w = weights.get(ep + ".downscale.conv.2.weight");
    }

    // ---- 6. Load bottleneck ------------------------------------------------
    load_tfc_tdf(weights, "bottleneck_block", l, bottleneck_);

    // ---- 7. Load decoder blocks --------------------------------------------
    decoder_blocks_.resize(n);
    for (int i = 0; i < n; i++) {
        std::string dp = "decoder_blocks." + std::to_string(i);
        load_norm(weights, dp + ".upscale.conv.0", decoder_blocks_[i].upscale_norm);
        decoder_blocks_[i].upscale_conv_w = weights.get(dp + ".upscale.conv.2.weight");
        load_tfc_tdf(weights, dp + ".tfc_tdf", l, decoder_blocks_[i].tfc_tdf);
    }

    // ---- 8. Load final conv ------------------------------------------------
    final_conv1_w_ = weights.get("final_conv.0.weight");
    // final_conv2_w_ was already loaded above for target-count inference.
    // final_conv.1 is the activation layer (no learned parameters for gelu/relu).

    // ---- done --------------------------------------------------------------
    std::cout << "[MDX23C] Model loaded:"
              << " scales=" << cfg_.num_scales
              << " blocks=" << cfg_.num_blocks_per_scale
              << " channels=" << cfg_.num_channels_model
              << " growth=" << cfg_.growth
              << " subbands=" << cfg_.num_subbands
              << " targets=" << cfg_.num_target_instruments
              << " dim_f=" << cfg_.dim_f
              << " scale=[" << cfg_.scale_h << "," << cfg_.scale_w << "]"
              << " norm=" << cfg_.norm_type
              << " act=" << cfg_.act_type
              << std::endl;
}

// ============================================================================
// forward
// ============================================================================

Tensor MDX23C::forward(const Tensor& audio) {
    // audio: [B, channels, T]
    int64_t batch    = audio.size(0);
    int64_t channels = audio.size(1);
    int64_t raw_len  = audio.size(2);
    int num_targets  = cfg_.num_target_instruments;

    // ========================================================================
    // 1. STFT → [B, C*2, dim_f, T']
    // ========================================================================

    // Flatten channels into batch: [B*C, T]
    Tensor flat_audio = audio.reshape({batch * channels, raw_len});

    // STFT: [B*C, F_full, T', 2]  (F_full = n_fft/2+1)
    Tensor spec = ops::stft(flat_audio, cfg_.n_fft, cfg_.hop_length,
                            cfg_.n_fft, stft_window_,
                            /*center=*/true, /*normalized=*/false);

    int64_t F_full  = spec.size(1);   // n_fft/2 + 1
    int64_t T_stft  = spec.size(2);   // number of STFT frames

    // Move complex dim before frequency: [B*C, 2, F_full, T']
    spec = spec.permute({0, 3, 1, 2}).contiguous();

    // Merge channels × complex:
    //   [B*C, 2, F_full, T'] → [B, C, 2, F_full, T'] → [B, C*2, F_full, T']
    spec = spec.reshape({batch, channels * 2, F_full, T_stft});

    // Trim frequency to dim_f: [B, C*2, dim_f, T']
    Tensor x = spec.slice(2, 0, (int64_t)cfg_.dim_f).contiguous();

    // ========================================================================
    // 2. Subband rearrangement: [B, C*2, dim_f, T'] → [B, dim_c, f_sub, T']
    // ========================================================================

    Tensor mix = cac2cws(x);
    x = mix;

    // ========================================================================
    // 3. First conv: [B, dim_c, f_sub, T'] → [B, c, f_sub, T']
    // ========================================================================

    Tensor first_conv_out = ops::conv2d(x, first_conv_w_, no_bias_);
    x = first_conv_out;

    // ========================================================================
    // 4. Transpose F ↔ T: [B, c, f_sub, T'] → [B, c, T', f_sub]
    //    All TFC_TDF / down / up operate on (T', f_sub) spatial dims.
    // ========================================================================

    x = x.transpose(-1, -2).contiguous();

    // ========================================================================
    // 5. Encoder
    // ========================================================================

    std::vector<Tensor> encoder_outputs;
    encoder_outputs.reserve(cfg_.num_scales);

    for (int i = 0; i < cfg_.num_scales; i++) {
        // TFC_TDF
        x = apply_tfc_tdf(x, encoder_blocks_[i].tfc_tdf);
        encoder_outputs.push_back(x);

        // Downscale: Norm → Act → Conv2d(c, c+g, kernel=scale, stride=scale)
        x = apply_norm(x, encoder_blocks_[i].downscale_norm);
        x = apply_act(x);
        x = ops::conv2d(x, encoder_blocks_[i].downscale_conv_w, no_bias_,
                        cfg_.scale_h, cfg_.scale_w);
    }

    // ========================================================================
    // 6. Bottleneck
    // ========================================================================

    x = apply_tfc_tdf(x, bottleneck_);

    // ========================================================================
    // 7. Decoder (symmetric with encoder; skip connections via concat)
    // ========================================================================

    for (int i = 0; i < cfg_.num_scales; i++) {
        // Upscale: Norm → Act → ConvTranspose2d(c, c-g, kernel=scale, stride=scale)
        x = apply_norm(x, decoder_blocks_[i].upscale_norm);
        x = apply_act(x);
        x = ops::conv_transpose2d(x, decoder_blocks_[i].upscale_conv_w, no_bias_,
                                  cfg_.scale_h, cfg_.scale_w);

        // Concatenate with encoder skip connection (LIFO order)
        Tensor skip = encoder_outputs.back();
        encoder_outputs.pop_back();
        x = Tensor::cat({x, skip}, 1);

        // TFC_TDF (first sub-block has in_c = 2*c due to concatenation)
        x = apply_tfc_tdf(x, decoder_blocks_[i].tfc_tdf);
    }

    // ========================================================================
    // 8. Transpose back: [B, c, T', f_sub] → [B, c, f_sub, T']
    // ========================================================================

    x = x.transpose(-1, -2).contiguous();

    // ========================================================================
    // 9. Artifact reduction: element-wise multiply with first conv output
    // ========================================================================

    x = x * first_conv_out;

    // ========================================================================
    // 10. Final conv
    //     cat([mix, x], 1) → Conv2d(c+dim_c → c) → Act → Conv2d(c → targets*dim_c)
    // ========================================================================

    x = Tensor::cat({mix, x}, 1);

    // Conv2d 1×1 (no bias)
    x = ops::conv2d(x, final_conv1_w_, no_bias_);
    x = apply_act(x);

    // Conv2d 1×1 (no bias)
    x = ops::conv2d(x, final_conv2_w_, no_bias_);

    // ========================================================================
    // 11. Subband inverse: [B, targets*dim_c, f_sub, T'] →
    //                      [B, targets*C*2, dim_f, T']
    // ========================================================================

    x = cws2cac(x);

    // ========================================================================
    // 12. Reshape for targets
    //     [B, targets*C*2, dim_f, T'] → [B, targets, C*2, dim_f, T']
    // ========================================================================

    int64_t c_total = x.size(1);   // targets * num_channels * 2
    int64_t f_out   = x.size(2);   // dim_f
    int64_t t_out   = x.size(3);   // T'
    int64_t c_per_target = c_total / num_targets;   // num_channels * 2

    x = x.reshape({batch, (int64_t)num_targets, c_per_target, f_out, t_out});

    // ========================================================================
    // 13. Inverse STFT
    //     [B, targets, C*2, dim_f, T'] → [B, targets, C, T_out]
    // ========================================================================

    int64_t C = c_per_target / 2;            // num_channels (audio channels)
    int64_t n_fft_bins = cfg_.n_fft / 2 + 1; // full frequency dimension

    // Pad frequency back to n_fft/2+1
    if (f_out < n_fft_bins) {
        Tensor f_pad = Tensor::zeros(
            {batch, (int64_t)num_targets, c_per_target,
             n_fft_bins - f_out, t_out});
        x = Tensor::cat({x, f_pad}, 3);
    }

    // Split channels and complex:
    //   [B, targets, C*2, n, T'] → [B, targets, C, 2, n, T']
    x = x.reshape({batch, (int64_t)num_targets, C, 2, n_fft_bins, t_out});

    // Flatten for iSTFT:
    //   [B, targets, C, 2, n, T'] → [B*targets*C, 2, n, T']
    int64_t B_flat = batch * num_targets * C;
    x = x.reshape({B_flat, 2, n_fft_bins, t_out});

    // Permute to iSTFT format: [B_flat, n, T', 2]
    x = x.permute({0, 2, 3, 1}).contiguous();

    // iSTFT: [B_flat, n, T', 2] → [B_flat, T_out]
    x = ops::istft(x, cfg_.n_fft, cfg_.hop_length, cfg_.n_fft,
                   stft_window_, /*length=*/-1,
                   /*center=*/true, /*normalized=*/false);

    int64_t out_length = x.size(1);

    // Reshape to [B, num_targets, num_channels, T_out]
    x = x.reshape({batch, (int64_t)num_targets, C, out_length});

    return x;
}

} // namespace cudasep
