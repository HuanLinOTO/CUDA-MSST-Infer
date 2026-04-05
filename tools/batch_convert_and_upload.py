#!/usr/bin/env python3
"""
batch_convert_and_upload.py — Convert all supported pretrained models to .csm format
and upload to HuggingFace Hub.

Produces both FP32 and FP16 versions for each model.
Upload happens concurrently with conversion.

Usage:
    python batch_convert_and_upload.py [--no-upload] [--output-dir DIR] [--only MODEL_NAME]
"""

import argparse
import os
import subprocess
import sys
import threading
import queue
import time
from pathlib import Path
from dataclasses import dataclass

# ---- Configuration ---------------------------------------------------------

PRETRAIN_ROOT = Path(__file__).resolve().parent.parent.parent / "pretrain"
CONFIGS_ROOT = Path(__file__).resolve().parent.parent.parent / "Music-Source-Separation-Training" / "configs"
CONVERTER = Path(__file__).resolve().parent / "convert_weights.py"
HF_REPO = "SVCFusion/Cuda-MSST-Infer-Models"


@dataclass
class ModelEntry:
    """One pretrained model to convert."""
    name: str               # Output name (without .csm)
    checkpoint: str          # Relative to PRETRAIN_ROOT
    config: str              # Relative to CONFIGS_ROOT
    model_type: str          # MBR / BSR / MDX23C / HTDemucs
    description: str = ""    # Human-readable description


# All supported models to convert
# Organized by model type for clarity
MODELS: list[ModelEntry] = [
    # =========================================================================
    # MelBandRoformer — Kim config (dim=384, depth=6)
    # =========================================================================
    ModelEntry(
        name="Kim_MelBandRoformer",
        checkpoint="vocal_models/Kim_MelBandRoformer.ckpt",
        config="KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml",
        model_type="MBR",
        description="KimberleyJensen MelBandRoformer vocals (SDR 10.98)",
    ),
    ModelEntry(
        name="inst_v1e_MelBandRoformer",
        checkpoint="vocal_models/inst_v1e.ckpt",
        config="KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml",
        model_type="MBR",
        description="Instrumental v1e MelBandRoformer",
    ),
    ModelEntry(
        name="karaoke_MelBandRoformer",
        checkpoint="vocal_models/model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        config="KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml",
        model_type="MBR",
        description="Karaoke MelBandRoformer by aufr33 & viperx (SDR 10.20)",
    ),
    ModelEntry(
        name="denoise_MelBandRoformer",
        checkpoint="single_stem_models/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
        config="KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml",
        model_type="MBR",
        description="Denoise MelBandRoformer by aufr33 (SDR 28.00)",
    ),
    ModelEntry(
        name="denoise_aggr_MelBandRoformer",
        checkpoint="single_stem_models/denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt",
        config="KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml",
        model_type="MBR",
        description="Denoise Aggressive MelBandRoformer by aufr33 (SDR 27.98)",
    ),
    ModelEntry(
        name="dereverb_MelBandRoformer",
        checkpoint="single_stem_models/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
        config="KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml",
        model_type="MBR",
        description="Dereverb MelBandRoformer by anvuew (SDR 19.17)",
    ),
    ModelEntry(
        name="dereverb_less_aggr_MelBandRoformer",
        checkpoint="single_stem_models/dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt",
        config="KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml",
        model_type="MBR",
        description="Dereverb Less Aggressive MelBandRoformer by anvuew (SDR 18.81)",
    ),
    ModelEntry(
        name="crowd_MelBandRoformer",
        checkpoint="single_stem_models/mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt",
        config="KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml",
        model_type="MBR",
        description="Crowd noise MelBandRoformer by aufr33 & viperx (SDR 8.71)",
    ),

    # =========================================================================
    # MelBandRoformer — viperx config (dim=384, depth=12)
    # =========================================================================
    ModelEntry(
        name="viperx_MelBandRoformer",
        checkpoint="vocal_models/model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
        config="viperx/model_mel_band_roformer_ep_3005_sdr_11.4360.yaml",
        model_type="MBR",
        description="Viperx MelBandRoformer vocals (SDR 11.44)",
    ),

    # =========================================================================
    # MelBandRoformer — default vocals config (dim=192, depth=8)
    # =========================================================================
    ModelEntry(
        name="vocals_MelBandRoformer",
        checkpoint="vocal_models/model_vocals_mel_band_roformer_sdr_8.42.ckpt",
        config="config_vocals_mel_band_roformer.yaml",
        model_type="MBR",
        description="Default vocals MelBandRoformer (SDR 8.42)",
    ),

    # =========================================================================
    # BSRoformer
    # =========================================================================
    ModelEntry(
        name="BSRoformer_ep317",
        checkpoint="vocal_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        config="viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml",
        model_type="BSR",
        description="BSRoformer vocals ep317 by viperx (SDR 12.98)",
    ),
    ModelEntry(
        name="BSRoformer_ep368",
        checkpoint="vocal_models/model_bs_roformer_ep_368_sdr_12.9628.ckpt",
        config="viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml",
        model_type="BSR",
        description="BSRoformer vocals ep368 by viperx (SDR 12.96)",
    ),
    ModelEntry(
        name="BSRoformer_ep937_other",
        checkpoint="single_stem_models/model_bs_roformer_ep_937_sdr_10.5309.ckpt",
        config="viperx/model_bs_roformer_ep_937_sdr_10.5309.yaml",
        model_type="BSR",
        description="BSRoformer other stem ep937 by viperx (SDR 10.53)",
    ),

    # =========================================================================
    # MDX23C
    # =========================================================================
    ModelEntry(
        name="MDX23C_vocals",
        checkpoint="vocal_models/model_vocals_mdx23c_sdr_10.17.ckpt",
        config="config_vocals_mdx23c.yaml",
        model_type="MDX23C",
        description="MDX23C vocals (SDR 10.17)",
    ),
    ModelEntry(
        name="MDX23C_dereverb",
        checkpoint="single_stem_models/dereverb_mdx23c_sdr_6.9096.ckpt",
        config="config_vocals_mdx23c.yaml",
        model_type="MDX23C",
        description="MDX23C dereverb (SDR 6.91)",
    ),
    ModelEntry(
        name="MDX23C_musdb18",
        checkpoint="multi_stem_models/model_mdx23c_ep_168_sdr_7.0207.ckpt",
        config="config_musdb18_mdx23c.yaml",
        model_type="MDX23C",
        description="MDX23C 4-stem MUSDB18 (SDR 7.02)",
    ),

    # =========================================================================
    # HTDemucs
    # =========================================================================
    ModelEntry(
        name="HTDemucs4",
        checkpoint="multi_stem_models/HTDemucs4.th",
        config="config_musdb18_htdemucs.yaml",
        model_type="HTDemucs",
        description="HTDemucs4 4-stem (bass/drums/vocals/other)",
    ),
    ModelEntry(
        name="HTDemucs4_6stems",
        checkpoint="multi_stem_models/HTDemucs4_6stems.th",
        config="config_htdemucs_6stems.yaml",
        model_type="HTDemucs",
        description="HTDemucs4 6-stem (bass/drums/vocals/other/guitar/piano)",
    ),
    ModelEntry(
        name="HTDemucs4_FT_vocals",
        checkpoint="single_stem_models/HTDemucs4_FT_vocals_official.th",
        config="config_musdb18_htdemucs.yaml",
        model_type="HTDemucs",
        description="HTDemucs4 fine-tuned vocals (SDR 8.38)",
    ),
    ModelEntry(
        name="HTDemucs4_FT_drums",
        checkpoint="single_stem_models/HTDemucs4_FT_drums.th",
        config="config_musdb18_htdemucs.yaml",
        model_type="HTDemucs",
        description="HTDemucs4 fine-tuned drums (SDR 11.13)",
    ),
    ModelEntry(
        name="HTDemucs4_FT_bass",
        checkpoint="single_stem_models/HTDemucs4_FT_bass.th",
        config="config_musdb18_htdemucs.yaml",
        model_type="HTDemucs",
        description="HTDemucs4 fine-tuned bass (SDR 11.96)",
    ),
    ModelEntry(
        name="HTDemucs4_FT_other",
        checkpoint="single_stem_models/HTDemucs4_FT_other.th",
        config="config_musdb18_htdemucs.yaml",
        model_type="HTDemucs",
        description="HTDemucs4 fine-tuned other (SDR 5.85)",
    ),
    ModelEntry(
        name="HTDemucs4_vocals",
        checkpoint="vocal_models/model_vocals_htdemucs_sdr_8.78.ckpt",
        config="config_vocals_htdemucs.yaml",
        model_type="HTDemucs",
        description="HTDemucs4 vocals MVSep finetuned (SDR 8.78)",
    ),
]


# ---- Upload worker ----------------------------------------------------------

def upload_worker(upload_queue: queue.Queue, hf_repo: str, stop_event: threading.Event):
    """Background thread that uploads files from the queue."""
    while not stop_event.is_set() or not upload_queue.empty():
        try:
            file_path, remote_path = upload_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        print(f"  [UPLOAD] Uploading {Path(file_path).name} → {hf_repo}/{remote_path}")
        try:
            cmd = [
                sys.executable, "-m", "huggingface_hub", "upload",
                hf_repo,
                str(file_path),
                remote_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode != 0:
                # Try huggingface-cli as fallback
                cmd2 = [
                    "huggingface-cli", "upload",
                    hf_repo,
                    str(file_path),
                    remote_path,
                ]
                result = subprocess.run(cmd2, capture_output=True, text=True, timeout=3600)
                if result.returncode != 0:
                    print(f"  [UPLOAD] FAILED: {result.stderr.strip()}")
                    upload_queue.task_done()
                    continue
            print(f"  [UPLOAD] Done: {Path(file_path).name}")
        except Exception as e:
            print(f"  [UPLOAD] ERROR: {e}")
        upload_queue.task_done()


def convert_model(entry: ModelEntry, output_dir: Path, half: bool = False) -> Path | None:
    """Convert a single model. Returns output path or None on failure."""
    suffix = "_fp16" if half else ""
    out_name = f"{entry.name}{suffix}.csm"
    out_path = output_dir / out_name

    ckpt_path = PRETRAIN_ROOT / entry.checkpoint
    config_path = CONFIGS_ROOT / entry.config

    if not ckpt_path.exists():
        print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
        return None
    if not config_path.exists():
        print(f"  [SKIP] Config not found: {config_path}")
        return None

    if out_path.exists():
        print(f"  [SKIP] Already exists: {out_path}")
        return out_path

    cmd = [
        sys.executable, str(CONVERTER),
        "--checkpoint", str(ckpt_path),
        "--config", str(config_path),
        "--output", str(out_path),
    ]
    if half:
        cmd.append("--half")

    print(f"  [{entry.model_type}] Converting {entry.name} ({'FP16' if half else 'FP32'})...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  [ERROR] Conversion failed for {entry.name}:")
            print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            return None
        # Print size info
        size_mb = out_path.stat().st_size / 1e6
        print(f"  [OK] {out_name} ({size_mb:.1f} MB)")
        return out_path
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Timeout converting {entry.name}")
        return None
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Batch convert and upload models")
    parser.add_argument("--no-upload", action="store_true", help="Skip HuggingFace upload")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "csm_models",
                        help="Output directory for .csm files")
    parser.add_argument("--only", type=str, default=None,
                        help="Only convert models matching this name (substring match)")
    parser.add_argument("--fp16-only", action="store_true", help="Only generate FP16 versions")
    parser.add_argument("--fp32-only", action="store_true", help="Only generate FP32 versions")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter models
    models = MODELS
    if args.only:
        models = [m for m in models if args.only.lower() in m.name.lower()]
        if not models:
            print(f"No models matching '{args.only}'")
            return

    print(f"Converting {len(models)} models to {output_dir}")
    print(f"Pretrain root: {PRETRAIN_ROOT}")
    print(f"Configs root: {CONFIGS_ROOT}")
    print(f"Converter: {CONVERTER}")
    print()

    # Start upload worker
    upload_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    upload_thread = None
    if not args.no_upload:
        upload_thread = threading.Thread(
            target=upload_worker, args=(upload_queue, HF_REPO, stop_event), daemon=True
        )
        upload_thread.start()
        print(f"Upload thread started → {HF_REPO}")
        print()

    # Convert and queue uploads
    success_count = 0
    fail_count = 0
    skip_count = 0

    for entry in models:
        print(f"--- {entry.name} ({entry.model_type}) ---")
        print(f"    {entry.description}")

        # FP32 version
        if not args.fp16_only:
            fp32_path = convert_model(entry, output_dir, half=False)
            if fp32_path:
                success_count += 1
                if not args.no_upload:
                    remote = f"fp32/{fp32_path.name}"
                    upload_queue.put((str(fp32_path), remote))
            elif fp32_path is None:
                ckpt = PRETRAIN_ROOT / entry.checkpoint
                if not ckpt.exists():
                    skip_count += 1
                else:
                    fail_count += 1

        # FP16 version
        if not args.fp32_only:
            fp16_path = convert_model(entry, output_dir, half=True)
            if fp16_path:
                success_count += 1
                if not args.no_upload:
                    remote = f"fp16/{fp16_path.name}"
                    upload_queue.put((str(fp16_path), remote))
            elif fp16_path is None:
                ckpt = PRETRAIN_ROOT / entry.checkpoint
                if not ckpt.exists():
                    skip_count += 1
                else:
                    fail_count += 1

        print()

    # Wait for uploads to finish
    if upload_thread:
        print("Waiting for uploads to complete...")
        upload_queue.join()
        stop_event.set()
        upload_thread.join(timeout=10)

    print("=" * 60)
    print(f"Conversion complete: {success_count} success, {fail_count} failed, {skip_count} skipped")

    # Generate README for HuggingFace
    readme_path = output_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("---\n")
        f.write("license: cc\n")
        f.write("tags:\n")
        f.write("  - music-source-separation\n")
        f.write("  - cuda\n")
        f.write("  - audio\n")
        f.write("---\n\n")
        f.write("# Cuda-MSST Inference Models\n\n")
        f.write("Pre-converted `.csm` weight files for the [CudaInfer](https://github.com/) C++/CUDA music source separation engine.\n\n")
        f.write("## Directory Structure\n\n")
        f.write("```\n")
        f.write("fp32/     # Full-precision FP32 weights\n")
        f.write("fp16/     # Half-precision FP16 weights (smaller, faster for some models)\n")
        f.write("```\n\n")
        f.write("## Available Models\n\n")
        f.write("| Model | Type | Description |\n")
        f.write("|-------|------|-------------|\n")
        for m in MODELS:
            f.write(f"| `{m.name}` | {m.model_type} | {m.description} |\n")
        f.write("\n## Usage\n\n")
        f.write("```bash\n")
        f.write("# Download a model\n")
        f.write("huggingface-cli download SVCFusion/Cuda-MSST-Infer-Models fp32/Kim_MelBandRoformer.csm\n\n")
        f.write("# Run inference\n")
        f.write("cudasep_infer --model Kim_MelBandRoformer.csm --input song.mp3 --output vocals.wav\n")
        f.write("```\n")

    if not args.no_upload:
        # Upload README
        cmd = [
            sys.executable, "-m", "huggingface_hub", "upload",
            HF_REPO, str(readme_path), "README.md",
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        print("README.md uploaded")

    print("Done!")


if __name__ == "__main__":
    main()
