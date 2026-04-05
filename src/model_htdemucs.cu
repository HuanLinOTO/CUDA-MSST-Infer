// model_htdemucs.cu — Complete HTDemucs inference implementation.
//
// HTDemucs is a hybrid time–frequency model for music source separation.
// Architecture:
//   mix ──► STFT ──► CaC ──► Normalize ──► Freq Encoder ──► CrossTransformer ──► Freq Decoder ──► CaC mask ──► iSTFT ─┐
//   mix ──► Normalize ──────────────────► Time Encoder ──► CrossTransformer ──► Time Decoder ─────────────────────────────┤► sum ──► output
//
// Frequency branch uses 2D convolutions (freq × time), time branch uses 1D convolutions.
// DConv residual blocks provide temporal context at each encoder/decoder layer.
// A CrossTransformer exchanges information between the two branches at the bottleneck.

#include "model_htdemucs.h"
#include "ops.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cassert>

namespace cudasep {

// ============================================================================
// HTDemucsConfig
// ============================================================================

HTDemucsConfig HTDemucsConfig::from_json(const JsonValue& j) {
    HTDemucsConfig c;
    c.num_sources      = j.get_int("num_sources", 4);
    c.audio_channels   = j.get_int("audio_channels", 2);
    c.channels         = j.get_int("channels", 48);
    c.growth           = j.get_int("growth", 2);
    c.depth            = j.get_int("depth", 4);
    c.nfft             = j.get_int("nfft", 4096);
    c.num_subbands     = j.get_int("num_subbands", 1);
    c.cac              = j.get_bool("cac", true);
    c.kernel_size      = j.get_int("kernel_size", 8);
    c.stride           = j.get_int("stride", 4);
    c.time_stride      = j.get_int("time_stride", 2);
    c.context          = j.get_int("context", 1);
    c.context_enc      = j.get_int("context_enc", 0);
    c.norm_starts      = j.get_int("norm_starts", 4);
    c.norm_groups      = j.get_int("norm_groups", 4);
    c.dconv_mode       = j.get_int("dconv_mode", 3);
    c.dconv_depth      = j.get_int("dconv_depth", 2);
    c.dconv_comp       = j.get_int("dconv_comp", 8);
    c.dconv_init       = j.get_float("dconv_init", 1e-3f);
    c.bottom_channels  = j.get_int("bottom_channels", 512);
    c.t_layers         = j.get_int("t_layers", 5);
    c.t_heads          = j.get_int("t_heads", 8);
    c.t_hidden_scale   = j.get_float("t_hidden_scale", 4.0f);
    c.freq_emb         = j.get_float("freq_emb", 0.2f);
    c.emb_scale        = j.get_float("emb_scale", 10.0f);
    c.rewrite          = j.get_bool("rewrite", true);
    c.samplerate       = j.get_int("samplerate", 44100);
    c.segment          = j.get_float("segment", 11.0f);
    c.use_train_segment = j.get_bool("use_train_segment", true);
    return c;
}

// ============================================================================
// pad1d — reflect padding with edge-case handling
// ============================================================================

Tensor HTDemucs::pad1d(const Tensor& x, int64_t pad_left, int64_t pad_right) {
    // x: [..., T]
    int64_t length = x.size(-1);
    int64_t max_pad = std::max(pad_left, pad_right);

    if (length <= max_pad) {
        // Need to zero-pad first to make reflect safe
        int64_t extra_pad = max_pad - length + 1;
        int64_t extra_pad_right = std::min(pad_right, extra_pad);
        int64_t extra_pad_left = extra_pad - extra_pad_right;
        // First zero-pad
        Tensor x_zp = x.pad({extra_pad_left, extra_pad_right}, 0.0f);
        // Then reflect pad with reduced amounts
        return x_zp.pad_reflect({pad_left - extra_pad_left, pad_right - extra_pad_right});
    }
    return x.pad_reflect({pad_left, pad_right});
}

// ============================================================================
// spec — STFT with custom padding
// ============================================================================

Tensor HTDemucs::spec(const Tensor& x) {
    // x: [B, C, L]
    // Returns: [B, C, F, T_out, 2]  where F = nfft/2
    int hl = cfg_.hop_length();
    int nfft = cfg_.nfft;
    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t L = x.size(-1);

    int le = (int)std::ceil((double)L / hl);
    int pad = hl / 2 * 3;  // 3/8 * nfft
    int64_t right_pad = pad + le * hl - L;

    // Flatten to [B*C, L]
    Tensor xf = x.reshape({B * C, L});

    // Reflect pad
    xf = pad1d(xf, pad, right_pad);

    // STFT with center=true and normalized=true
    // Python spectro uses center=True, normalized=True, win_length=nfft, hann_window(nfft)
    Tensor z = ops::stft(xf, nfft, hl, nfft, stft_window_,
                         /*center=*/true, /*normalized=*/true);
    // z: [B*C, F_full, T_full, 2]  where F_full = nfft/2 + 1

    int64_t F_full = z.size(1);  // nfft/2 + 1
    int64_t T_full = z.size(2);

    // Trim last freq bin: [B*C, nfft/2, T_full, 2]
    z = z.slice(1, 0, F_full - 1);

    // Verify and trim time: z[..., 2:2+le]
    // After the padding and center mode, T_full should be le + 4
    z = z.slice(2, 2, 2 + le);

    // Reshape to [B, C, F, le, 2]
    int64_t F = nfft / 2;
    z = z.reshape({B, C, F, (int64_t)le, 2});

    return z;
}

// ============================================================================
// ispec — inverse STFT with padding removal
// ============================================================================

Tensor HTDemucs::ispec(const Tensor& z, int64_t length) {
    // z: [*, F, T, 2]  where F = nfft/2
    // Returns: [*, length]
    int hl = cfg_.hop_length();
    int nfft = cfg_.nfft;

    // Collect all leading dims
    std::vector<int64_t> leading;
    for (int i = 0; i < z.ndim() - 3; i++) {
        leading.push_back(z.size(i));
    }
    int64_t batch_flat = 1;
    for (auto d : leading) batch_flat *= d;

    int64_t F = z.size(-3);   // nfft/2
    int64_t T = z.size(-2);

    // Flatten to [batch_flat, F, T, 2]
    Tensor zf = z.reshape({batch_flat, F, T, 2});

    // Pad frequency: add 1 zero bin at the end → [batch_flat, F+1, T, 2]
    Tensor zpad_freq = Tensor::zeros({batch_flat, 1, T, 2});
    zf = Tensor::cat({zf, zpad_freq}, 1);

    // Pad time: add 2 frames on each side → [batch_flat, F+1, T+4, 2]
    Tensor zpad_time_l = Tensor::zeros({batch_flat, F + 1, 2, 2});
    Tensor zpad_time_r = Tensor::zeros({batch_flat, F + 1, 2, 2});
    zf = Tensor::cat({zpad_time_l, zf, zpad_time_r}, 2);

    // Compute expected length for iSTFT
    int pad = hl / 2 * 3;
    int64_t le_target = hl * (int64_t)std::ceil((double)length / hl) + 2 * pad;

    // iSTFT: [batch_flat, F+1, T+4, 2] → [batch_flat, signal_len]
    Tensor x = ops::istft(zf, nfft, hl, nfft, stft_window_,
                          /*length=*/le_target,
                          /*center=*/true, /*normalized=*/true);

    // Trim padding: x[..., pad : pad + length]
    x = x.slice(-1, pad, pad + length);

    // Reshape to [*leading, length]
    leading.push_back(length);
    x = x.reshape(leading);
    return x;
}

// ============================================================================
// magnitude — CaC (Complex as Channels) conversion
// ============================================================================

Tensor HTDemucs::magnitude(const Tensor& z) {
    // z: [B, C, F, T, 2] → [B, C*2, F, T]
    assert(cfg_.cac);
    int64_t B = z.size(0);
    int64_t C = z.size(1);
    int64_t F = z.size(2);
    int64_t T = z.size(3);

    // permute(0,1,4,2,3): [B, C, F, T, 2] → [B, C, 2, F, T]
    Tensor m = z.permute({0, 1, 4, 2, 3}).contiguous();
    // reshape: [B, C*2, F, T]
    m = m.reshape({B, C * 2, F, T});
    return m;
}

// ============================================================================
// mask — CaC mask application
// ============================================================================

Tensor HTDemucs::mask(const Tensor& z, const Tensor& m) {
    // z: [B, C, F, T, 2] (original complex STFT — ignored in CaC mode)
    // m: [B, S, C_cac, F, T] where C_cac = audio_channels * 2
    // Returns: [B, S, audio_channels, F, T, 2]
    assert(cfg_.cac);
    int64_t B  = m.size(0);
    int64_t S  = m.size(1);
    int64_t Cc = m.size(2);  // C_cac = audio_channels * 2
    int64_t F  = m.size(3);
    int64_t T  = m.size(4);
    int64_t C  = Cc / 2;     // audio_channels

    // Reshape: [B, S, C, 2, F, T]
    Tensor out = m.reshape({B, S, C, 2, F, T});
    // Permute to [B, S, C, F, T, 2]
    out = out.permute({0, 1, 2, 4, 5, 3}).contiguous();
    return out;
}

// ============================================================================
// apply_dconv — DConv residual branch
// ============================================================================

Tensor HTDemucs::apply_dconv(const Tensor& x, const DConvLayer& dconv) {
    // x: [N, C, T]
    Tensor y = x;

    for (int d = 0; d < (int)dconv.levels.size(); d++) {
        const auto& lev = dconv.levels[d];
        Tensor residual = y;

        int dilation = 1 << d;   // 2^d
        int kernel = 3;
        int padding = dilation * (kernel / 2);

        // Conv1d compressor: [N, C, T] → [N, C/comp, T]
        y = ops::conv1d(y, lev.conv1_w, lev.conv1_b,
                        /*stride=*/1, /*padding=*/padding, /*dilation=*/dilation);

        // GroupNorm(1, hidden)
        y = ops::group_norm(y, 1, lev.gn1_w, lev.gn1_b);

        // GELU activation
        y = ops::gelu(y);

        // Conv1d expander (1×1): [N, C/comp, T] → [N, 2*C, T]
        y = ops::conv1d(y, lev.conv2_w, lev.conv2_b,
                        /*stride=*/1, /*padding=*/0);

        // GroupNorm(1, 2*C)
        y = ops::group_norm(y, 1, lev.gn2_w, lev.gn2_b);

        // GLU on channel dim → [N, C, T]
        y = ops::glu(y, 1);

        // LayerScale + residual (fused: out = residual + y * scale)
        y = ops::scale_residual_add(y, residual, lev.layer_scale, 1);
    }
    return y;
}

// ============================================================================
// apply_enc_layer — encoder layer (freq 2D or time 1D)
// ============================================================================

Tensor HTDemucs::apply_enc_layer(const Tensor& x, const EncLayerWeights& w,
                                  const Tensor& inject) {
    Tensor y = x;

    if (w.is_freq) {
        // Frequency branch: x is [B, C_in, F, T]
        // Conv2d with kernel [K, 1], stride [S, 1], pad [K/4, 0]
        int pad_h = w.conv_w.size(2) / 4;
        int stride_h = (int)w.conv_w.size(2) == cfg_.kernel_size ? cfg_.stride : (int)w.conv_w.size(2);
        // Determine stride: if kernel == freqs (last_freq case), stride = kernel, pad = 0
        // The conv weight shape tells us kernel: conv_w is [C_out, C_in, kH, kW=1]
        int kH = (int)w.conv_w.size(2);
        // Check if this is a pad=False (last_freq) layer: weight has kH small & special
        // We can detect this by checking if pad_h was meant to be 0
        // Actually, pad is embedded in the weight shapes. Let's use the conv_w directly.
        // The Python code uses pad=kernel//4 when pad=True, pad=0 when pad=False.
        // When pad=False (last_freq), the kernel size equals the remaining freq dimension.
        // We detect: if kH <= stride for a normal layer, OR if kH != kernel_size → special case
        // The simplest: store pad in the conv or check if conv_b leads to freqs=1.
        // Since we always have stride = kH for the last-freq case and stride != kH/2 normally...
        // Normal layers: kH = kernel_size, stride = cfg_.stride, pad = kernel_size/4
        // Last-freq: kH = remaining_freqs, stride = remaining_freqs, pad = 0
        // We detect by checking kH vs kernel_size
        if (kH != cfg_.kernel_size) {
            // last_freq layer: kernel covers all remaining freqs
            stride_h = kH;
            pad_h = 0;
        } else {
            stride_h = cfg_.stride;
            pad_h = cfg_.kernel_size / 4;
        }

        y = ops::conv2d(y, w.conv_w, w.conv_b,
                        stride_h, /*stride_w=*/1,
                        pad_h, /*pad_w=*/0);

        // Empty layer: just return conv output (used for time<->freq merge)
        if (w.rewrite_w.is_empty()) {
            return y;
        }

        // Inject from time branch (when merging)
        if (!inject.is_empty()) {
            Tensor inj = inject;
            if (inj.ndim() == 3 && y.ndim() == 4) {
                // [B, C, T] → [B, C, 1, T]
                inj = inj.unsqueeze(2);
            }
            y = y + inj;
        }

        // Norm + GELU
        // Note: norm_starts=4 and depth=4 by default means norm is always False.
        // If norm were True, separate GroupNorm weights would need to be loaded.
        // Currently unsupported for simplicity.
        if (w.use_norm) {
            // TODO: load and apply GroupNorm weights if norm_starts < depth
            std::cerr << "[HTDemucs] WARNING: GroupNorm in encoder not implemented, skipping" << std::endl;
        }
        y = ops::gelu(y);

        // DConv: reshape [B,C,F,T] → [B*F, C, T], apply, reshape back
        if (!w.dconv.levels.empty()) {
            int64_t B = y.size(0), C_ch = y.size(1), Fr = y.size(2), T_t = y.size(3);
            Tensor yd = y.permute({0, 2, 1, 3}).contiguous();  // [B, F, C, T]
            yd = yd.reshape({B * Fr, C_ch, T_t});
            yd = apply_dconv(yd, w.dconv);
            yd = yd.reshape({B, Fr, C_ch, T_t});
            y = yd.permute({0, 2, 1, 3}).contiguous();  // [B, C, F, T]
        }

        // Rewrite (1×1 conv, or 1+2*context_enc)
        if (!w.rewrite_w.is_empty()) {
            int ctx = cfg_.context_enc;
            Tensor z = ops::conv2d(y, w.rewrite_w, w.rewrite_b,
                                   /*stride_h=*/1, /*stride_w=*/1,
                                   /*pad_h=*/ctx, /*pad_w=*/ctx);
            z = ops::glu(z, 1);  // GLU on channel dim
            y = z;
        }
    } else {
        // Time branch: x is [B, C_in, T]

        // Pad to stride-aligned length
        int64_t le = y.size(-1);
        if (le % cfg_.stride != 0) {
            y = y.pad({0, (int64_t)(cfg_.stride - (le % cfg_.stride))}, 0.0f);
        }

        // Conv1d
        int ksize = cfg_.kernel_size;
        int stride_t = cfg_.stride;
        int pad_t = ksize / 4;

        y = ops::conv1d(y, w.conv_w, w.conv_b, stride_t, pad_t);

        // Empty layer: just return conv output
        if (w.rewrite_w.is_empty()) {
            return y;
        }

        // Inject (not used for time branch normally)
        if (!inject.is_empty()) {
            y = y + inject;
        }

        y = ops::gelu(y);

        // DConv
        if (!w.dconv.levels.empty()) {
            y = apply_dconv(y, w.dconv);
        }

        // Rewrite (1×1 conv, or 1+2*context_enc for 1D)
        if (!w.rewrite_w.is_empty()) {
            int ctx = cfg_.context_enc;
            Tensor z = ops::conv1d(y, w.rewrite_w, w.rewrite_b,
                                   /*stride=*/1, /*padding=*/ctx);
            z = ops::glu(z, 1);
            y = z;
        }
    }

    return y;
}

// ============================================================================
// apply_dec_layer — decoder layer (freq 2D or time 1D)
// ============================================================================

std::pair<Tensor, Tensor> HTDemucs::apply_dec_layer(const Tensor& x, const Tensor& skip,
                                                     const DecLayerWeights& w,
                                                     int64_t target_length) {
    Tensor y = x;

    if (w.is_freq) {
        // x: [B, C, F, T]  — if x comes in as 3D, reshape
        if (y.ndim() == 3) {
            int64_t Bb = y.size(0), Cc = y.size(1), Tt = y.size(2);
            int64_t chin = w.conv_tr_w.size(0);
            int64_t Fr = Cc / chin;
            y = y.reshape({Bb, chin, Fr, Tt});
        }

        // Determine kernel / stride / pad for this layer's transposed conv
        int kH = (int)w.conv_tr_w.size(2);
        int stride_h, pad_h_val;
        if (kH != cfg_.kernel_size) {
            // last_freq layer
            stride_h = kH;
            pad_h_val = 0;
        } else {
            stride_h = cfg_.stride;
            pad_h_val = cfg_.kernel_size / 4;
        }

        Tensor pre = y;

        if (!w.rewrite_w.is_empty()) {
            // Non-empty decoder layer
            // Skip connection
            if (!skip.is_empty()) {
                y = y + skip;
            }

            // Rewrite: Conv2d kernel detected from weight shape
            int rwH = (int)w.rewrite_w.size(2);
            int rwW = (int)w.rewrite_w.size(3);
            int pH = rwH / 2;
            int pW = rwW / 2;
            Tensor z = ops::conv2d(y, w.rewrite_w, w.rewrite_b,
                                   /*stride_h=*/1, /*stride_w=*/1, pH, pW);
            z = ops::glu(z, 1);  // GLU on channel dim
            y = z;

            // DConv
            if (!w.dconv.levels.empty()) {
                int64_t Bb = y.size(0), Cc = y.size(1), Fr = y.size(2), Tt = y.size(3);
                Tensor yd = y.permute({0, 2, 1, 3}).contiguous();  // [B, F, C, T]
                yd = yd.reshape({Bb * Fr, Cc, Tt});
                yd = apply_dconv(yd, w.dconv);
                yd = yd.reshape({Bb, Fr, Cc, Tt});
                y = yd.permute({0, 2, 1, 3}).contiguous();  // [B, C, F, T]
            }

            pre = y;
        } else {
            // Empty decoder layer (no rewrite/dconv, skip is None)
        }

        // Transposed convolution (Python ConvTranspose2d has padding=0)
        // Python: z = self.norm2(self.conv_tr(y))
        // norm2 is GroupNorm that we skip in default config (norm_starts >= depth)
        y = ops::conv_transpose2d(y, w.conv_tr_w, w.conv_tr_b,
                                  stride_h, /*stride_w=*/1,
                                  /*pad_h=*/0, /*pad_w=*/0);

        // Trim frequency padding: z[..., pad:-pad, :]
        if (pad_h_val > 0) {
            y = y.slice(2, pad_h_val, y.size(2) - pad_h_val);
        }

        if (!w.is_last) {
            y = ops::gelu(y);
        }
        return {y, pre};

    } else {
        // Time branch: x is [B, C, T]
        Tensor pre = y;

        if (!w.rewrite_w.is_empty()) {
            // Non-empty decoder layer
            if (!skip.is_empty()) {
                y = y + skip;
            }

            // Rewrite
            int k = (int)w.rewrite_w.size(2);
            int p = k / 2;
            Tensor z = ops::conv1d(y, w.rewrite_w, w.rewrite_b,
                                   /*stride=*/1, /*padding=*/p);
            z = ops::glu(z, 1);
            y = z;

            // DConv
            if (!w.dconv.levels.empty()) {
                y = apply_dconv(y, w.dconv);
            }

            pre = y;
        } else {
            // Empty layer: no skip, no rewrite, no dconv
        }

        // Transposed convolution
        int ksize = cfg_.kernel_size;
        int stride_t = cfg_.stride;
        int pad_val = ksize / 4;
        // Python ConvTranspose1d has padding=0, output is larger, then trimmed
        y = ops::conv_transpose1d(y, w.conv_tr_w, w.conv_tr_b, stride_t, /*padding=*/0);

        // Trim to target_length: z[..., pad:pad+length]
        y = y.slice(-1, pad_val, pad_val + target_length);

        if (!w.is_last) {
            y = ops::gelu(y);
        }
        return {y, pre};
    }
}

// ============================================================================
// apply_self_attention — Transformer encoder layer (self-attention)
// ============================================================================

Tensor HTDemucs::apply_self_attention(const Tensor& x, const TransformerLayer& w) {
    // x: [B, T, dim] (batch_first=True)
    int64_t B = x.size(0);
    int64_t T = x.size(1);
    int64_t dim = x.size(2);
    int heads = cfg_.t_heads;
    int dim_head = (int)(dim / heads);

    Tensor out = x;

    // norm_first = True path
    // Self-attention block
    {
        Tensor normed = ops::layer_norm(out, w.norm1_w, w.norm1_b);

        // in_proj: project to Q, K, V  → [B, T, 3*dim]
        Tensor qkv = ops::linear(normed, w.attn_in_proj_w, w.attn_in_proj_b);

        // Split into q, k, v each [B, T, dim]
        Tensor q = qkv.slice(2, 0, dim);
        Tensor k = qkv.slice(2, dim, 2 * dim);
        Tensor v = qkv.slice(2, 2 * dim, 3 * dim);

        // Reshape to [B, heads, T, dim_head]
        q = q.reshape({B, T, (int64_t)heads, (int64_t)dim_head}).permute({0, 2, 1, 3}).contiguous();
        k = k.reshape({B, T, (int64_t)heads, (int64_t)dim_head}).permute({0, 2, 1, 3}).contiguous();
        v = v.reshape({B, T, (int64_t)heads, (int64_t)dim_head}).permute({0, 2, 1, 3}).contiguous();

        // Scaled dot product attention
        Tensor att_out = ops::scaled_dot_product_attention(q, k, v);

        // Reshape back to [B, T, dim]
        att_out = att_out.permute({0, 2, 1, 3}).contiguous().reshape({B, T, dim});

        // Out projection
        att_out = ops::linear(att_out, w.attn_out_proj_w, w.attn_out_proj_b);

        // LayerScale gamma_1 + residual (fused)
        att_out = ops::scale_residual_add(att_out, out, w.gamma_1, -1);
        out = att_out;
    }

    // FFN block
    {
        Tensor normed2 = ops::layer_norm(out, w.norm2_w, w.norm2_b);
        Tensor ffn = ops::linear_gelu(normed2, w.linear1_w, w.linear1_b);
        ffn = ops::linear(ffn, w.linear2_w, w.linear2_b);

        // LayerScale gamma_2 + residual (fused)
        ffn = ops::scale_residual_add(ffn, out, w.gamma_2, -1);
        out = ffn;
    }

    // norm_out: GroupNorm(1) directly on [B, T, C] layout
    if (!w.norm_out_w.is_empty()) {
        out = ops::group_norm_1_btc(out, w.norm_out_w, w.norm_out_b);
    }

    return out;
}

// ============================================================================
// apply_cross_attention — Cross-attention transformer layer
// ============================================================================

Tensor HTDemucs::apply_cross_attention(const Tensor& q_input, const Tensor& kv_input,
                                        const TransformerLayer& w) {
    // q_input: [B, T_q, dim], kv_input: [B, T_kv, dim]
    int64_t B = q_input.size(0);
    int64_t T_q = q_input.size(1);
    int64_t T_kv = kv_input.size(1);
    int64_t dim = q_input.size(2);
    int heads = cfg_.t_heads;
    int dim_head = (int)(dim / heads);

    Tensor out = q_input;

    // Cross-attention block
    {
        Tensor normed_q = ops::layer_norm(out, w.norm1_w, w.norm1_b);
        Tensor normed_kv = ops::layer_norm(kv_input, w.norm2_w, w.norm2_b);

        // Split in_proj_weight into W_q, W_k, W_v
        Tensor W_q = w.attn_in_proj_w.slice(0, 0, dim);
        Tensor W_k = w.attn_in_proj_w.slice(0, dim, 2 * dim);
        Tensor W_v = w.attn_in_proj_w.slice(0, 2 * dim, 3 * dim);
        Tensor b_q = w.attn_in_proj_b.slice(0, 0, dim);
        Tensor b_k = w.attn_in_proj_b.slice(0, dim, 2 * dim);
        Tensor b_v = w.attn_in_proj_b.slice(0, 2 * dim, 3 * dim);

        // Q from q_input, K/V from kv_input
        Tensor q = ops::linear(normed_q, W_q, b_q);
        Tensor k = ops::linear(normed_kv, W_k, b_k);
        Tensor v = ops::linear(normed_kv, W_v, b_v);

        // Reshape to [B, heads, T, dim_head]
        q = q.reshape({B, T_q, (int64_t)heads, (int64_t)dim_head}).permute({0, 2, 1, 3}).contiguous();
        k = k.reshape({B, T_kv, (int64_t)heads, (int64_t)dim_head}).permute({0, 2, 1, 3}).contiguous();
        v = v.reshape({B, T_kv, (int64_t)heads, (int64_t)dim_head}).permute({0, 2, 1, 3}).contiguous();

        // Scaled dot product attention
        Tensor att_out = ops::scaled_dot_product_attention(q, k, v);

        // Reshape back to [B, T_q, dim]
        att_out = att_out.permute({0, 2, 1, 3}).contiguous().reshape({B, T_q, dim});

        // Out projection
        att_out = ops::linear(att_out, w.attn_out_proj_w, w.attn_out_proj_b);

        // LayerScale gamma_1 + residual (fused)
        att_out = ops::scale_residual_add(att_out, out, w.gamma_1, -1);
        out = att_out;
    }

    // FFN block: uses norm3 for cross-attention layer
    {
        Tensor normed3 = ops::layer_norm(out, w.norm3_w, w.norm3_b);
        Tensor ffn = ops::linear_gelu(normed3, w.linear1_w, w.linear1_b);
        ffn = ops::linear(ffn, w.linear2_w, w.linear2_b);

        // LayerScale gamma_2 + residual (fused)
        ffn = ops::scale_residual_add(ffn, out, w.gamma_2, -1);
        out = ffn;
    }

    // norm_out: GroupNorm(1) directly on [B, T, C] layout
    if (!w.norm_out_w.is_empty()) {
        out = ops::group_norm_1_btc(out, w.norm_out_w, w.norm_out_b);
    }

    return out;
}

// ============================================================================
// create_sin_embedding_1d — sinusoidal positional embedding
// ============================================================================

Tensor HTDemucs::create_sin_embedding_1d(int length, int dim) {
    // Returns [length, 1, dim] (TBC format)
    // pos: [length, 1, 1]
    // phase = pos / (max_period ^ (adim / (half_dim - 1)))
    // Concatenate cos and sin
    assert(dim % 2 == 0);
    int half_dim = dim / 2;
    float max_period = 10000.0f;

    std::vector<float> data(length * dim);
    for (int t = 0; t < length; t++) {
        for (int d = 0; d < half_dim; d++) {
            float phase = (float)t / std::pow(max_period, (float)d / (half_dim - 1));
            data[t * dim + d] = std::cos(phase);              // first half: cos
            data[t * dim + half_dim + d] = std::sin(phase);   // second half: sin
        }
    }
    return Tensor::from_cpu_f32(data.data(), {(int64_t)length, 1, (int64_t)dim});
}

// ============================================================================
// create_sin_embedding_2d — 2D sinusoidal positional embedding
// ============================================================================

Tensor HTDemucs::create_sin_embedding_2d(int dim, int height, int width) {
    // Returns [1, dim, height, width]
    assert(dim % 4 == 0);
    float max_period = 10000.0f;
    int d_model_half = dim / 2;

    std::vector<float> pe(dim * height * width, 0.0f);

    // div_term = exp(arange(0, d_model_half, 2) * -(log(max_period) / d_model_half))
    int num_terms = d_model_half / 2;
    std::vector<float> div_term(num_terms);
    for (int i = 0; i < num_terms; i++) {
        div_term[i] = std::exp((float)(2 * i) * (-std::log(max_period) / d_model_half));
    }

    // pe[0:d_model_half:2, :, :] = sin(pos_w * div_term).T.unsqueeze(1).repeat(1, height, 1)
    // pe[1:d_model_half:2, :, :] = cos(pos_w * div_term).T.unsqueeze(1).repeat(1, height, 1)
    for (int i = 0; i < num_terms; i++) {
        int c_sin = 2 * i;       // channel for sin
        int c_cos = 2 * i + 1;   // channel for cos
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                float val_w = (float)w * div_term[i];
                pe[c_sin * height * width + h * width + w] = std::sin(val_w);
                pe[c_cos * height * width + h * width + w] = std::cos(val_w);
            }
        }
    }

    // pe[d_model_half::2, :, :] = sin(pos_h * div_term).T.unsqueeze(2).repeat(1, 1, width)
    // pe[d_model_half+1::2, :, :] = cos(pos_h * div_term).T.unsqueeze(2).repeat(1, 1, width)
    for (int i = 0; i < num_terms; i++) {
        int c_sin = d_model_half + 2 * i;
        int c_cos = d_model_half + 2 * i + 1;
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                float val_h = (float)h * div_term[i];
                pe[c_sin * height * width + h * width + w] = std::sin(val_h);
                pe[c_cos * height * width + h * width + w] = std::cos(val_h);
            }
        }
    }

    return Tensor::from_cpu_f32(pe.data(), {1, (int64_t)dim, (int64_t)height, (int64_t)width});
}

// ============================================================================
// load — load all model weights
// ============================================================================

void HTDemucs::load(const ModelWeights& weights) {
    // 1. Parse config
    cfg_ = HTDemucsConfig::from_json(weights.config());

    // 2. Create STFT window
    stft_window_ = ops::hann_window(cfg_.nfft);

    // 3. Compute layer parameters
    // We need to figure out the channel progression and which layers are freq/time/empty
    int depth = cfg_.depth;
    int channels = cfg_.channels;
    int growth = cfg_.growth;
    int audio_channels = cfg_.audio_channels;
    int kernel_size = cfg_.kernel_size;
    int stride = cfg_.stride;
    int nfft = cfg_.nfft;
    int S = cfg_.num_sources;

    int chin = audio_channels;
    int chin_z = chin;
    if (cfg_.cac) chin_z *= 2;
    if (cfg_.num_subbands > 1) chin_z *= cfg_.num_subbands;

    int chout = channels;
    int chout_z = channels;
    int freqs = nfft / 2;

    // Track per-layer parameters
    struct LayerParams {
        int chin, chout;
        int kernel, stride_val, pad;
        bool freq;
        bool empty;    // (tencoder only) last_freq marker
        bool norm;
        bool last_freq;
    };

    std::vector<LayerParams> enc_params, tenc_params;
    std::vector<LayerParams> dec_params, tdec_params;

    // Build encoder parameter progression (following Python __init__ exactly)
    int chin_copy = chin;
    int chin_z_copy = chin_z;
    int chout_copy = chout;
    int chout_z_copy = chout_z;
    int freqs_copy = freqs;

    for (int index = 0; index < depth; index++) {
        bool norm = index >= cfg_.norm_starts;
        bool freq = freqs_copy > 1;
        int stri = stride;
        int ker = kernel_size;
        bool pad_flag = true;
        bool last_freq = false;

        if (!freq) {
            ker = cfg_.time_stride * 2;
            stri = cfg_.time_stride;
        }

        if (freq && freqs_copy <= kernel_size) {
            ker = freqs_copy;
            pad_flag = false;
            last_freq = true;
        }

        int pad_val = pad_flag ? ker / 4 : 0;

        if (last_freq) {
            chout_z_copy = std::max(chout_copy, chout_z_copy);
            chout_copy = chout_z_copy;
        }

        // Encoder freq layer
        LayerParams ep;
        ep.chin = chin_z_copy;
        ep.chout = chout_z_copy;
        ep.kernel = ker;
        ep.stride_val = stri;
        ep.pad = pad_val;
        ep.freq = freq;
        ep.empty = false;
        ep.norm = norm;
        ep.last_freq = last_freq;
        enc_params.push_back(ep);

        // Time encoder layer (only when freq=true)
        if (freq) {
            LayerParams tp;
            tp.chin = chin_copy;
            tp.chout = chout_copy;
            tp.kernel = kernel_size;
            tp.stride_val = stride;
            tp.pad = kernel_size / 4;
            tp.freq = false;
            tp.empty = last_freq;
            tp.norm = norm;
            tp.last_freq = false;
            tenc_params.push_back(tp);
        }

        // Decoder freq layer (inserted at front → stored reverse order)
        // chin/chout are swapped for decoder
        // For index==0, output channels expand by S (sources)
        int dec_chin_z = chout_z_copy;
        int dec_chout_z = (index == 0) ? (audio_channels * S * (cfg_.cac ? 2 : 1) *
                                           cfg_.num_subbands) : chin_z_copy;
        LayerParams dp;
        dp.chin = dec_chin_z;
        dp.chout = dec_chout_z;
        dp.kernel = ker;
        dp.stride_val = stri;
        dp.pad = pad_val;
        dp.freq = freq;
        dp.empty = false;
        dp.norm = norm;
        dp.last_freq = last_freq;
        dec_params.push_back(dp);

        // Time decoder (only when freq=true)
        if (freq) {
            int dec_chin_t = chout_copy;
            int dec_chout_t = (index == 0) ? (audio_channels * S) : chin_copy;
            LayerParams tdp;
            tdp.chin = dec_chin_t;
            tdp.chout = dec_chout_t;
            tdp.kernel = kernel_size;
            tdp.stride_val = stride;
            tdp.pad = kernel_size / 4;
            tdp.freq = false;
            tdp.empty = last_freq;
            tdp.norm = norm;
            tdp.last_freq = false;
            tdec_params.push_back(tdp);
        }

        // Update for next layer
        if (index == 0) {
            chin_copy = audio_channels * S;
            chin_z_copy = chin_copy;
            if (cfg_.cac) chin_z_copy *= 2;
            if (cfg_.num_subbands > 1) chin_z_copy *= cfg_.num_subbands;
        }
        chin_copy = chout_copy;
        chin_z_copy = chout_z_copy;
        chout_copy = (int)(growth * chout_copy);
        chout_z_copy = (int)(growth * chout_z_copy);
        if (freq) {
            if (freqs_copy <= kernel_size) {
                freqs_copy = 1;
            } else {
                freqs_copy /= stride;
            }
        }
    }

    // Reverse dec_params and tdec_params (decoder is stored in reverse)
    std::reverse(dec_params.begin(), dec_params.end());
    std::reverse(tdec_params.begin(), tdec_params.end());

    // Mark last decoder layer
    if (!dec_params.empty()) {
        dec_params.back().empty = false;  // last in reversed = first layer = index 0
    }

    // 4. Load DConv weights helper
    auto load_dconv = [&](const std::string& prefix, DConvLayer& dconv) {
        dconv.levels.resize(cfg_.dconv_depth);
        for (int d = 0; d < cfg_.dconv_depth; d++) {
            auto& lev = dconv.levels[d];
            std::string lp = prefix + ".layers." + std::to_string(d);
            lev.conv1_w = weights.get(lp + ".0.weight");
            lev.conv1_b = weights.get(lp + ".0.bias");
            lev.gn1_w   = weights.get(lp + ".1.weight");
            lev.gn1_b   = weights.get(lp + ".1.bias");
            // index 2 = GELU (no params)
            lev.conv2_w = weights.get(lp + ".3.weight");
            lev.conv2_b = weights.get(lp + ".3.bias");
            lev.gn2_w   = weights.get(lp + ".4.weight");
            lev.gn2_b   = weights.get(lp + ".4.bias");
            // index 5 = GLU (no params)
            lev.layer_scale = weights.get(lp + ".6.scale");
        }
    };

    // 5. Load encoder layers
    encoder_.resize(depth);
    for (int i = 0; i < depth; i++) {
        auto& ew = encoder_[i];
        std::string prefix = "encoder." + std::to_string(i);
        ew.conv_w = weights.get(prefix + ".conv.weight");
        ew.conv_b = weights.get(prefix + ".conv.bias");
        ew.is_freq = enc_params[i].freq;
        ew.use_norm = enc_params[i].norm;

        // Check if this is an empty layer equivalent (only conv, no rewrite/dconv)
        // In Python: empty layers only exist in tencoder. Encoder layers always have full structure.
        // But for the last_freq layer, encoder is not empty — it has full dconv/rewrite.
        if (weights.has(prefix + ".rewrite.weight")) {
            ew.rewrite_w = weights.get(prefix + ".rewrite.weight");
            ew.rewrite_b = weights.get(prefix + ".rewrite.bias");
        }
        // DConv
        if ((cfg_.dconv_mode & 1) && weights.has(prefix + ".dconv.layers.0.0.weight")) {
            load_dconv(prefix + ".dconv", ew.dconv);
        }
    }

    // 6. Load time encoder layers
    tencoder_.resize(tenc_params.size());
    for (int i = 0; i < (int)tenc_params.size(); i++) {
        auto& ew = tencoder_[i];
        std::string prefix = "tencoder." + std::to_string(i);
        ew.conv_w = weights.get(prefix + ".conv.weight");
        ew.conv_b = weights.get(prefix + ".conv.bias");
        ew.is_freq = false;
        ew.use_norm = tenc_params[i].norm;

        if (tenc_params[i].empty) {
            // Empty layer: only conv, no norm/rewrite/dconv
            // Leave rewrite_w empty
        } else {
            if (weights.has(prefix + ".rewrite.weight")) {
                ew.rewrite_w = weights.get(prefix + ".rewrite.weight");
                ew.rewrite_b = weights.get(prefix + ".rewrite.bias");
            }
            if ((cfg_.dconv_mode & 1) && weights.has(prefix + ".dconv.layers.0.0.weight")) {
                load_dconv(prefix + ".dconv", ew.dconv);
            }
        }
    }

    // 7. Load decoder layers
    decoder_.resize(depth);
    for (int i = 0; i < depth; i++) {
        auto& dw = decoder_[i];
        std::string prefix = "decoder." + std::to_string(i);
        dw.conv_tr_w = weights.get(prefix + ".conv_tr.weight");
        dw.conv_tr_b = weights.get(prefix + ".conv_tr.bias");
        dw.is_freq = dec_params[i].freq;
        dw.use_norm = dec_params[i].norm;
        dw.is_last = (i == depth - 1);

        if (weights.has(prefix + ".rewrite.weight")) {
            dw.rewrite_w = weights.get(prefix + ".rewrite.weight");
            dw.rewrite_b = weights.get(prefix + ".rewrite.bias");
        }
        if ((cfg_.dconv_mode & 2) && weights.has(prefix + ".dconv.layers.0.0.weight")) {
            load_dconv(prefix + ".dconv", dw.dconv);
        }
    }

    // 8. Load time decoder layers
    tdecoder_.resize(tdec_params.size());
    for (int i = 0; i < (int)tdec_params.size(); i++) {
        auto& dw = tdecoder_[i];
        std::string prefix = "tdecoder." + std::to_string(i);
        dw.conv_tr_w = weights.get(prefix + ".conv_tr.weight");
        dw.conv_tr_b = weights.get(prefix + ".conv_tr.bias");
        dw.is_freq = false;
        dw.use_norm = tdec_params[i].norm;
        // tdecoder is stored reversed in Python; tdecoder[0] corresponds to the
        // last layer in the decoder loop (deepest). is_last for the outermost one.
        dw.is_last = (i == (int)tdec_params.size() - 1);

        if (tdec_params[i].empty) {
            // Empty layer
        } else {
            if (weights.has(prefix + ".rewrite.weight")) {
                dw.rewrite_w = weights.get(prefix + ".rewrite.weight");
                dw.rewrite_b = weights.get(prefix + ".rewrite.bias");
            }
            if ((cfg_.dconv_mode & 2) && weights.has(prefix + ".dconv.layers.0.0.weight")) {
                load_dconv(prefix + ".dconv", dw.dconv);
            }
        }
    }

    // 9. Load frequency embedding
    if (cfg_.freq_emb > 0.0f) {
        freq_emb_weight_ = weights.get("freq_emb.embedding.weight");
    }

    // 10. Load channel up/downsamplers (1×1 Conv1d)
    if (cfg_.bottom_channels > 0) {
        ch_up_w_   = weights.get("channel_upsampler.weight");
        ch_up_b_   = weights.get("channel_upsampler.bias");
        ch_down_w_ = weights.get("channel_downsampler.weight");
        ch_down_b_ = weights.get("channel_downsampler.bias");
        ch_up_t_w_   = weights.get("channel_upsampler_t.weight");
        ch_up_t_b_   = weights.get("channel_upsampler_t.bias");
        ch_down_t_w_ = weights.get("channel_downsampler_t.weight");
        ch_down_t_b_ = weights.get("channel_downsampler_t.bias");
    }

    // 11. Load CrossTransformer weights
    if (cfg_.t_layers > 0) {
        // norm_in
        ct_norm_in_w_   = weights.get("crosstransformer.norm_in.weight");
        ct_norm_in_b_   = weights.get("crosstransformer.norm_in.bias");
        ct_norm_in_t_w_ = weights.get("crosstransformer.norm_in_t.weight");
        ct_norm_in_t_b_ = weights.get("crosstransformer.norm_in_t.bias");

        // classic_parity: cross_first=False → parity=0 → idx%2==0 is classic (self), idx%2==1 is cross
        int classic_parity = 0;  // t_cross_first=False

        ct_layers_.resize(cfg_.t_layers);
        ct_layers_t_.resize(cfg_.t_layers);

        for (int i = 0; i < cfg_.t_layers; i++) {
            bool is_cross = (i % 2 != classic_parity);
            std::string lp  = "crosstransformer.layers." + std::to_string(i);
            std::string lpt = "crosstransformer.layers_t." + std::to_string(i);

            auto load_transformer_layer = [&](const std::string& prefix,
                                               TransformerLayer& tl, bool cross) {
                tl.is_cross = cross;

                if (cross) {
                    // CrossTransformerEncoderLayer uses nn.MultiheadAttention
                    tl.attn_in_proj_w = weights.get(prefix + ".cross_attn.in_proj_weight");
                    tl.attn_in_proj_b = weights.get(prefix + ".cross_attn.in_proj_bias");
                    tl.attn_out_proj_w = weights.get(prefix + ".cross_attn.out_proj.weight");
                    tl.attn_out_proj_b = weights.get(prefix + ".cross_attn.out_proj.bias");
                } else {
                    // MyTransformerEncoderLayer uses nn.MultiheadAttention (self_attn)
                    tl.attn_in_proj_w = weights.get(prefix + ".self_attn.in_proj_weight");
                    tl.attn_in_proj_b = weights.get(prefix + ".self_attn.in_proj_bias");
                    tl.attn_out_proj_w = weights.get(prefix + ".self_attn.out_proj.weight");
                    tl.attn_out_proj_b = weights.get(prefix + ".self_attn.out_proj.bias");
                }

                // FFN
                tl.linear1_w = weights.get(prefix + ".linear1.weight");
                tl.linear1_b = weights.get(prefix + ".linear1.bias");
                tl.linear2_w = weights.get(prefix + ".linear2.weight");
                tl.linear2_b = weights.get(prefix + ".linear2.bias");

                // Norms
                tl.norm1_w = weights.get(prefix + ".norm1.weight");
                tl.norm1_b = weights.get(prefix + ".norm1.bias");
                tl.norm2_w = weights.get(prefix + ".norm2.weight");
                tl.norm2_b = weights.get(prefix + ".norm2.bias");

                if (cross) {
                    tl.norm3_w = weights.get(prefix + ".norm3.weight");
                    tl.norm3_b = weights.get(prefix + ".norm3.bias");
                }

                // LayerScale
                tl.gamma_1 = weights.get(prefix + ".gamma_1.scale");
                tl.gamma_2 = weights.get(prefix + ".gamma_2.scale");

                // norm_out (GroupNorm(1))
                if (weights.has(prefix + ".norm_out.weight")) {
                    tl.norm_out_w = weights.get(prefix + ".norm_out.weight");
                    tl.norm_out_b = weights.get(prefix + ".norm_out.bias");
                }
            };

            load_transformer_layer(lp, ct_layers_[i], is_cross);
            load_transformer_layer(lpt, ct_layers_t_[i], is_cross);
        }
    }

    // Done
    std::cout << "[HTDemucs] Model loaded:"
              << " depth=" << depth
              << " channels=" << channels
              << " growth=" << growth
              << " nfft=" << nfft
              << " sources=" << S
              << " tencoder_layers=" << tencoder_.size()
              << " tdecoder_layers=" << tdecoder_.size()
              << " t_layers=" << cfg_.t_layers
              << " bottom_channels=" << cfg_.bottom_channels
              << std::endl;
}

// ============================================================================
// forward — full HTDemucs inference
// ============================================================================

Tensor HTDemucs::forward(const Tensor& mix) {
    // mix: [B, audio_channels, L]
    int64_t B = mix.size(0);
    int64_t L = mix.size(-1);
    int depth = cfg_.depth;
    int S = cfg_.num_sources;

    // ========================================================================
    // 0. Handle use_train_segment: pad to training length if needed
    // ========================================================================
    Tensor mix_padded = mix;
    int64_t length_pre_pad = 0;
    int training_length = (int)(cfg_.segment * cfg_.samplerate);
    bool padded = false;

    if (cfg_.use_train_segment) {
        if (L < training_length) {
            length_pre_pad = L;
            mix_padded = mix.pad({0, (int64_t)(training_length - L)}, 0.0f);
            padded = true;
        }
    }

    int64_t L_eff = mix_padded.size(-1);

    // ========================================================================
    // 1. STFT → CaC conversion
    // ========================================================================
    Tensor z = spec(mix_padded);
    // z: [B, audio_channels, F, T_frames, 2]

    Tensor mag = magnitude(z);
    // mag: [B, audio_channels*2, F, T_frames]

    Tensor x = mag;

    // Subband rearrangement (if num_subbands > 1)
    if (cfg_.num_subbands > 1) {
        int64_t b = x.size(0), c = x.size(1), f = x.size(2), t = x.size(3);
        int k = cfg_.num_subbands;
        x = x.reshape({b, c * k, f / k, t});
    }

    int64_t Fq = x.size(2);
    int64_t T_frames = x.size(3);

    // ========================================================================
    // 2. Normalize frequency branch
    // ========================================================================
    // mean, std over (C, F, T) dims
    Tensor mean_x = x.mean(1, true).mean(2, true).mean(3, true);  // [B, 1, 1, 1]

    // Compute std = sqrt(mean((x - mean)^2))
    Tensor x_centered = x - mean_x;
    Tensor x_var = (x_centered * x_centered).mean(1, true).mean(2, true).mean(3, true);
    // Compute std = sqrt(var) on GPU
    Tensor std_x = ops::sqrt(x_var);

    // Normalize
    Tensor std_x_safe = std_x.clone();
    std_x_safe.add_scalar_(1e-5f);
    x = x_centered / std_x_safe;

    // ========================================================================
    // 3. Normalize time branch
    // ========================================================================
    Tensor xt = mix_padded;
    Tensor mean_t = xt.mean(1, true).mean(2, true);  // [B, 1, 1]

    Tensor xt_centered = xt - mean_t;
    Tensor xt_var = (xt_centered * xt_centered).mean(1, true).mean(2, true);
    Tensor std_t = ops::sqrt(xt_var);

    Tensor std_t_safe = std_t.clone();
    std_t_safe.add_scalar_(1e-5f);
    xt = xt_centered / std_t_safe;

    // ========================================================================
    // 4. Encoder
    // ========================================================================
    std::vector<Tensor> saved;      // freq skip connections
    std::vector<Tensor> saved_t;    // time skip connections
    std::vector<int64_t> lengths;   // freq branch saved lengths
    std::vector<int64_t> lengths_t; // time branch saved lengths

    for (int idx = 0; idx < depth; idx++) {
        lengths.push_back(x.size(-1));
        Tensor inject;

        if (idx < (int)tencoder_.size()) {
            lengths_t.push_back(xt.size(-1));
            xt = apply_enc_layer(xt, tencoder_[idx]);

            if (!tencoder_[idx].rewrite_w.is_empty()) {
                saved_t.push_back(xt);
            } else {
                inject = xt;
            }
        }

        x = apply_enc_layer(x, encoder_[idx], inject);

        if (idx == 0 && cfg_.freq_emb > 0.0f && !freq_emb_weight_.is_empty()) {
            int64_t num_freqs = x.size(-2);
            if (freq_arange_cache_.numel() == 0) {
                freq_arange_cache_ = Tensor::arange(0, num_freqs, DType::Int64);
            }
            Tensor frs = freq_arange_cache_;
            Tensor emb = ops::embedding(frs, freq_emb_weight_);
            emb = emb.transpose(0, 1).contiguous();
            emb = emb.unsqueeze(0).unsqueeze(-1);
            float total_scale = cfg_.emb_scale * cfg_.freq_emb;
            Tensor scaled_emb = emb.contiguous();
            scaled_emb.mul_scalar_(total_scale);
            x = x + scaled_emb;
        }

        saved.push_back(x);
    }

    // ========================================================================
    // 5. CrossTransformer
    // ========================================================================
    int64_t f_before_ct = x.size(2);  // save freq dim for reshape later

    if (cfg_.t_layers > 0) {
        // Channel upsampler (freq branch)
        if (cfg_.bottom_channels > 0) {
            int64_t b_ = x.size(0), c_ = x.size(1), f_ = x.size(2), t_ = x.size(3);
            // [B, C, F, T] → [B, C, F*T]
            Tensor x_flat = x.reshape({b_, c_, f_ * t_});
            x_flat = ops::conv1d(x_flat, ch_up_w_, ch_up_b_);
            // [B, bottom, F*T] → [B, bottom, F, T]
            x = x_flat.reshape({b_, (int64_t)cfg_.bottom_channels, f_, t_});

            // Channel upsampler (time branch)
            xt = ops::conv1d(xt, ch_up_t_w_, ch_up_t_b_);
        }

        // Prepare for transformer: reshape to batch_first [B, seq_len, dim]
        int64_t B_ = x.size(0), C_ct = x.size(1), Fr = x.size(2), T1 = x.size(3);

        // 2D positional embedding for freq branch (cached)
        int64_t key_2d = ((int64_t)C_ct << 32) | ((int64_t)Fr << 16) | (int64_t)T1;
        auto& cached_2d = pos_emb_2d_cache_[key_2d];
        if (cached_2d.numel() == 0) {
            Tensor pos_emb_2d = create_sin_embedding_2d((int)C_ct, (int)Fr, (int)T1);
            // Rearrange: [1, C, Fr, T1] → [1, T1*Fr, C]
            cached_2d = pos_emb_2d.permute({0, 3, 2, 1}).contiguous().reshape({1, T1 * Fr, C_ct});
        }
        Tensor pos_emb_2d_flat = cached_2d;

        // x: [B, C, Fr, T1] → [B, T1*Fr, C]
        Tensor x_tf = x.permute({0, 3, 2, 1}).contiguous().reshape({B_, T1 * Fr, C_ct});

        // Norm in
        x_tf = ops::layer_norm(x_tf, ct_norm_in_w_, ct_norm_in_b_);

        // Add positional embedding
        Tensor pos_2d_expanded = pos_emb_2d_flat.expand({B_, T1 * Fr, C_ct});
        // weight_pos_embed = 1.0 (default)
        x_tf = x_tf + pos_2d_expanded;

        // 1D positional embedding for time branch (cached)
        int64_t T2 = xt.size(-1);
        int64_t key_1d = ((int64_t)T2 << 16) | (int64_t)C_ct;
        auto& cached_1d = pos_emb_1d_cache_[key_1d];
        if (cached_1d.numel() == 0) {
            Tensor pos_emb_1d = create_sin_embedding_1d((int)T2, (int)C_ct);
            // [T2, 1, C] → [1, T2, C]
            cached_1d = pos_emb_1d.permute({1, 0, 2}).contiguous();
        }
        Tensor pos_1d = cached_1d;

        // xt: [B, C, T2] → [B, T2, C]
        Tensor xt_tf = xt.transpose(1, 2).contiguous();

        // Norm in
        xt_tf = ops::layer_norm(xt_tf, ct_norm_in_t_w_, ct_norm_in_t_b_);

        // Add positional embedding
        Tensor pos_1d_expanded = pos_1d.expand({B_, T2, C_ct});
        xt_tf = xt_tf + pos_1d_expanded;

        // Apply transformer layers
        int classic_parity = 0;
        for (int idx = 0; idx < cfg_.t_layers; idx++) {
            if (idx % 2 == classic_parity) {
                x_tf = apply_self_attention(x_tf, ct_layers_[idx]);
                xt_tf = apply_self_attention(xt_tf, ct_layers_t_[idx]);
            } else {
                Tensor old_x = x_tf;
                x_tf = apply_cross_attention(x_tf, xt_tf, ct_layers_[idx]);
                xt_tf = apply_cross_attention(xt_tf, old_x, ct_layers_t_[idx]);
            }
        }

        // Reshape back
        // x_tf: [B, T1*Fr, C] → [B, C, Fr, T1]
        x = x_tf.reshape({B_, T1, Fr, C_ct}).permute({0, 3, 2, 1}).contiguous();

        // xt_tf: [B, T2, C] → [B, C, T2]
        xt = xt_tf.transpose(1, 2).contiguous();

        // Channel downsampler
        if (cfg_.bottom_channels > 0) {
            int64_t b_ = x.size(0), c_ = x.size(1), f_ = x.size(2), t_ = x.size(3);
            // [B, bottom, F, T] → [B, bottom, F*T]
            Tensor x_flat = x.reshape({b_, c_, f_ * t_});
            x_flat = ops::conv1d(x_flat, ch_down_w_, ch_down_b_);
            // [B, C_orig, F*T] → [B, C_orig, F, T]
            int64_t C_orig = ch_down_w_.size(0);
            x = x_flat.reshape({b_, C_orig, f_, t_});

            // Time downsampler
            xt = ops::conv1d(xt, ch_down_t_w_, ch_down_t_b_);
        }
    }

    // ========================================================================
    // 6. Decoder
    // ========================================================================
    for (int idx = 0; idx < depth; idx++) {
        Tensor skip = saved.back();
        saved.pop_back();
        int64_t tgt_len = lengths.back();
        lengths.pop_back();

        auto [x_out, pre] = apply_dec_layer(x, skip, decoder_[idx], tgt_len);
        x = x_out;

        int offset = depth - (int)tdecoder_.size();
        if (idx >= offset) {
            int tidx = idx - offset;

            if (tdecoder_[tidx].rewrite_w.is_empty()) {
                Tensor pre_t = pre;
                if (pre_t.ndim() == 4 && pre_t.size(2) == 1) {
                    pre_t = pre_t.squeeze(2);
                }
                int64_t len_t = lengths_t.back();
                lengths_t.pop_back();
                auto [xt_out, _] = apply_dec_layer(pre_t, Tensor(), tdecoder_[tidx], len_t);
                xt = xt_out;
            } else {
                Tensor skip_t = saved_t.back();
                saved_t.pop_back();
                int64_t len_t = lengths_t.back();
                lengths_t.pop_back();
                auto [xt_out, _] = apply_dec_layer(xt, skip_t, tdecoder_[tidx], len_t);
                xt = xt_out;
            }
        }
    }

    // Verify all skip connections used
    assert(saved.empty());
    assert(saved_t.empty());
    assert(lengths_t.empty());

    // ========================================================================
    // 7. Output: combine freq and time branches
    // ========================================================================
    int64_t C_out = cfg_.audio_channels;

    // Freq branch: subband inverse if needed
    if (cfg_.num_subbands > 1) {
        int64_t b = x.size(0), c = x.size(1), f = x.size(2), t = x.size(3);
        x = x.reshape({b, -1, Fq, t});  // restore original freq layout
        int k = cfg_.num_subbands;
        int64_t c2 = x.size(1);
        x = x.reshape({b, c2 / k, f * k, t});
    }

    // x: [B, S*C_cac, Fq*num_subbands, T_frames] where C_cac = audio_channels * 2
    int64_t C_cac = C_out * 2;
    x = x.reshape({B, (int64_t)S, C_cac, Fq * cfg_.num_subbands, T_frames});

    // Denormalize freq branch
    // std_x: [B, 1, 1, 1] → [B, 1, 1, 1, 1] for broadcast with [B, S, C_cac, F, T]
    Tensor std_x_5d = std_x.unsqueeze(1);       // [B, 1, 1, 1, 1]
    Tensor mean_x_5d = mean_x.unsqueeze(1);
    x = x * std_x_5d + mean_x_5d;

    // Apply CaC mask → iSTFT
    // z: [B, audio_channels, F, T_frames, 2]
    Tensor zout = mask(z, x);
    // zout: [B, S, audio_channels, F, T_frames, 2]

    // Flatten for iSTFT: [B*S*audio_channels, F, T_frames, 2]
    int64_t F_dim = zout.size(3);
    int64_t T_dim = zout.size(4);
    Tensor zout_flat = zout.reshape({B * S * C_out, F_dim, T_dim, 2});

    int64_t istft_length = cfg_.use_train_segment ? training_length : L;
    Tensor audio_f = ispec(zout_flat, istft_length);
    // audio_f: [B*S*C_out, istft_length]
    audio_f = audio_f.reshape({B, (int64_t)S, C_out, istft_length});

    // Time branch: reshape and denormalize
    // xt: [B, S*audio_channels, L_eff]
    xt = xt.reshape({B, (int64_t)S, C_out, -1});

    // Denormalize time branch
    // std_t: [B, 1, 1], mean_t: [B, 1, 1]
    Tensor std_t_4d = std_t.unsqueeze(1);   // [B, 1, 1, 1]
    Tensor mean_t_4d = mean_t.unsqueeze(1);
    xt = xt * std_t_4d + mean_t_4d;

    // If time branch length doesn't match, slice/pad
    int64_t xt_len = xt.size(-1);
    if (xt_len > istft_length) {
        xt = xt.slice(-1, 0, istft_length);
    } else if (xt_len < istft_length) {
        xt = xt.pad({0, istft_length - xt_len}, 0.0f);
    }

    // Combine
    Tensor output = audio_f + xt;

    // Trim if padded
    if (padded && length_pre_pad > 0) {
        output = output.slice(-1, 0, length_pre_pad);
    }

    return output;  // [B, S, audio_channels, length]
}

} // namespace cudasep
