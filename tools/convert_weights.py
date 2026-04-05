#!/usr/bin/env python3
"""
convert_weights.py – Convert a PyTorch checkpoint + YAML config to the .csm
binary format consumed by the CudaInfer C++/CUDA engine.

Usage
-----
    python convert_weights.py \
        --checkpoint path/to/model.ckpt \
        --config path/to/config.yaml \
        --output model.csm \
        [--half]

.csm layout
-----------
    [4  B] Magic "CSM\0"
    [4  B] Version (uint32 = 1)
    [4  B] Config JSON length (uint32)
    [N  B] Config JSON string
    [4  B] Number of tensors (uint32)
    Per tensor:
        [4  B] Name length (uint32)
        [N  B] Name string
        [4  B] Number of dimensions (uint32)
        [D*8 B] Shape (int64[])
        [4  B] DType (uint32: 0=float32, 1=float16)
        [M  B] Raw tensor data (contiguous, row-major)
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import torch
import yaml

# Optional pretty progress bar
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ---- helpers ---------------------------------------------------------------

MAGIC = b"CSM\x00"
VERSION = 1

DTYPE_MAP = {
    torch.float32: 0,
    torch.float16: 1,
    torch.int64: 2,
}


def _flatten_config(yaml_cfg: dict) -> dict:
    """Merge relevant top-level sections into a single flat dict."""
    out: dict = {}

    # 'audio' section (chunk_size, sample_rate, n_fft, hop_length …)
    if "audio" in yaml_cfg and isinstance(yaml_cfg["audio"], dict):
        out.update(yaml_cfg["audio"])

    # 'model' section — can be a dict (most models) or a string (HTDemucs)
    model_val = yaml_cfg.get("model")
    if isinstance(model_val, dict):
        out.update(model_val)
    elif isinstance(model_val, str):
        # HTDemucs-style: model: "htdemucs" with a separate htdemucs: {} section
        out["model_type"] = model_val
        if model_val in yaml_cfg and isinstance(yaml_cfg[model_val], dict):
            out.update(yaml_cfg[model_val])

    # 'training' section — extract instruments, samplerate, segment, normalize
    if "training" in yaml_cfg and isinstance(yaml_cfg["training"], dict):
        tr = yaml_cfg["training"]
        for key in ("instruments", "samplerate", "segment", "normalize",
                     "target_instrument", "channels"):
            if key in tr:
                if key == "samplerate" and "sample_rate" not in out:
                    out["sample_rate"] = tr[key]
                elif key not in out:
                    out[key] = tr[key]

    # Top-level scalars that don't belong to a sub-dict (legacy configs)
    skip = {"audio", "model", "training", "augmentations", "inference"}
    for k, v in yaml_cfg.items():
        if k not in skip and not isinstance(v, dict):
            out[k] = v

    # Try to infer model_type if not set explicitly
    if "model_type" not in out:
        if isinstance(model_val, dict) and "model_type" in model_val:
            out["model_type"] = model_val["model_type"]

    return out


def _load_state_dict(ckpt_path: str) -> tuple[dict[str, torch.Tensor], dict | None]:
    """Load a checkpoint and return its state_dict and optional kwargs."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    extra_config = None
    if isinstance(ckpt, dict):
        # HTDemucs-style: has 'state' + 'kwargs' with model params
        if "kwargs" in ckpt and isinstance(ckpt.get("kwargs"), dict):
            extra_config = dict(ckpt["kwargs"])
            # Convert non-serializable types (but preserve bool/int/float)
            for k, v in list(extra_config.items()):
                if isinstance(v, bool):
                    pass  # keep as bool (json.dumps handles it)
                elif hasattr(v, 'numerator') and not isinstance(v, (int, float)):
                    extra_config[k] = float(v)  # Fraction → float
        # Common keys used by various training frameworks
        for key in ("state_dict", "model_state_dict", "model", "net", "state"):
            if key in ckpt:
                return ckpt[key], extra_config
        # If none of the known keys exist, assume the dict *is* a state_dict
        # (check that values are tensors)
        sample = next(iter(ckpt.values()), None)
        if isinstance(sample, torch.Tensor):
            return ckpt, extra_config
        raise RuntimeError(
            f"Cannot locate state_dict in checkpoint. Top-level keys: "
            f"{list(ckpt.keys())}"
        )
    raise RuntimeError(
        f"Unsupported checkpoint type: {type(ckpt).__name__}"
    )


def _write_csm(
    state_dict: dict[str, torch.Tensor],
    config_json: str,
    output_path: str,
    use_half: bool,
) -> None:
    """Serialise *state_dict* and *config_json* into a .csm file."""

    tensors: list[tuple[str, torch.Tensor]] = []
    for name, param in state_dict.items():
        t = param.detach()
        if use_half and t.dtype == torch.float32:
            t = t.half()
        elif not use_half and t.dtype == torch.float16:
            t = t.float()  # Ensure float32 when not using half
        t = t.contiguous()
        tensors.append((name, t))

    config_bytes = config_json.encode("utf-8")

    with open(output_path, "wb") as f:
        # Header
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", len(config_bytes)))
        f.write(config_bytes)

        # Number of tensors
        f.write(struct.pack("<I", len(tensors)))

        # Progress
        iterator = tensors
        if tqdm is not None:
            iterator = tqdm(tensors, desc="Writing tensors", unit="tensor")
        else:
            print(f"Writing {len(tensors)} tensors …")

        total_bytes = 0
        for idx, (name, t) in enumerate(iterator):
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)

            # Shape
            ndim = t.ndim
            f.write(struct.pack("<I", ndim))
            for s in t.shape:
                f.write(struct.pack("<q", s))  # int64

            # DType
            dtype_id = DTYPE_MAP.get(t.dtype)
            if dtype_id is None:
                raise ValueError(
                    f"Unsupported dtype {t.dtype} for tensor '{name}'. "
                    f"Only float32, float16, and int64 are supported."
                )
            f.write(struct.pack("<I", dtype_id))

            # Raw data
            data = t.numpy().tobytes()
            f.write(data)
            total_bytes += len(data)

            if tqdm is None and (idx + 1) % 50 == 0:
                print(f"  [{idx + 1}/{len(tensors)}] …")

    file_size = os.path.getsize(output_path)
    print(f"\nDone!  {output_path}")
    print(f"  Tensors : {len(tensors)}")
    print(f"  Data    : {total_bytes / 1e6:.2f} MB")
    print(f"  File    : {file_size / 1e6:.2f} MB")
    print(f"  Dtype   : {'float16' if use_half else 'float32'}")


# ---- main ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch checkpoint + YAML config to .csm format."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to PyTorch .ckpt / .pth file"
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--output", required=True, help="Output .csm file path"
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Convert float32 weights to float16",
    )
    args = parser.parse_args()

    # ---- load config ----
    print(f"Loading config: {args.config}")
    with open(args.config, "r", encoding="utf-8") as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
    config = _flatten_config(yaml_cfg)
    config_json = json.dumps(config, ensure_ascii=False)
    print(f"  Config keys: {list(config.keys())}")

    # ---- load checkpoint ----
    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict, extra_config = _load_state_dict(args.checkpoint)
    print(f"  Parameters: {len(state_dict)}")

    # Merge extra config from checkpoint (e.g. HTDemucs kwargs)
    if extra_config:
        for k, v in extra_config.items():
            if k not in config:
                # Convert lists with non-serializable items
                if isinstance(v, (list, tuple)):
                    v = [float(x) if hasattr(x, 'numerator') else x for x in v]
                config[k] = v
        print(f"  Merged {len(extra_config)} keys from checkpoint kwargs")

    # ---- compute precomputed mel band data for MelBandRoformer ----
    # If config indicates this is a MelBandRoformer, compute freq_indices using librosa
    is_mbr = "num_bands" in config and "stft_n_fft" in config
    if is_mbr:
        print("  Detected MelBandRoformer, computing mel band indices with librosa...")
        try:
            import librosa
        except ImportError:
            print("  WARNING: librosa not installed, cannot compute mel bands")
            is_mbr = False

    if is_mbr:
        import numpy as np
        from einops import reduce, repeat, rearrange

        sample_rate = config.get("sample_rate", 44100)
        stft_n_fft = config.get("stft_n_fft", 2048)
        num_bands = config.get("num_bands", 60)
        stereo = config.get("stereo", True)
        audio_channels = 2 if stereo else 1
        num_fft_bins = stft_n_fft // 2 + 1

        # Use librosa for Slaney mel scale (matching the Python model)
        mel_filter_bank_numpy = librosa.filters.mel(
            sr=sample_rate, n_fft=stft_n_fft, n_mels=num_bands
        )
        mel_filter_bank = torch.from_numpy(mel_filter_bank_numpy)

        # Force endpoints
        mel_filter_bank[0][0] = 1.0
        mel_filter_bank[-1, -1] = 1.0

        # Binary threshold
        freqs_per_band = mel_filter_bank > 0

        # freq_indices
        repeated_freq_indices = repeat(
            torch.arange(num_fft_bins), 'f -> b f', b=num_bands
        )
        freq_indices = repeated_freq_indices[freqs_per_band]

        if stereo:
            freq_indices = repeat(freq_indices, 'f -> f s', s=2)
            freq_indices = freq_indices * 2 + torch.arange(2)
            freq_indices = rearrange(freq_indices, 'f s -> (f s)')

        num_freqs_per_band = reduce(freqs_per_band, 'b f -> b', 'sum')
        num_bands_per_freq = reduce(freqs_per_band, 'b f -> f', 'sum')

        # Store as extra tensors
        state_dict["__precomputed__.freq_indices"] = freq_indices.to(torch.int64)
        state_dict["__precomputed__.num_freqs_per_band"] = num_freqs_per_band.to(torch.int64)
        state_dict["__precomputed__.num_bands_per_freq"] = num_bands_per_freq.to(torch.int64)

        # Also store band_freq_dims as config
        band_freq_dims = [2 * int(f) * audio_channels for f in num_freqs_per_band.tolist()]
        config["__band_freq_dims__"] = band_freq_dims

        print(f"    freq_indices: {len(freq_indices)}, bands: {num_bands}")
        print(f"    band_freq_dims sum: {sum(band_freq_dims)}")
        print(f"    num_freqs_per_band: {num_freqs_per_band.tolist()[:5]}...")

    # Regenerate config_json after possible config changes
    config_json = json.dumps(config, ensure_ascii=False)

    # ---- write .csm ----
    print(f"Writing: {args.output}  (half={args.half})")
    _write_csm(state_dict, config_json, args.output, use_half=args.half)


if __name__ == "__main__":
    main()
