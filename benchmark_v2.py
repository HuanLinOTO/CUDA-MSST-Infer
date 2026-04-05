"""
benchmark_v2.py — Compare PyTorch (MSST) vs CudaInfer (C++/CUDA) inference speed.

Usage:
    cd d:\Projects\Joy\CudaMSST
    python CudaInfer/benchmark_v2.py --audio "水饺 Half.mp3"

Workflow:
  1. Convert each model's weights to .csm format (if not already present)
  2. Run PyTorch inference via MSST's demix() — 3 timed runs
  3. Run CudaInfer inference via cudasep_infer.exe — 3 timed runs
  4. Print comparison table
"""

import os
import sys
import time
import argparse
import subprocess
import json
import struct
import gc
import traceback
import numpy as np

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)  # CudaMSST
MSST_ROOT = os.path.join(ROOT, "Music-Source-Separation-Training")
CUDASEP_ROOT = os.path.join(ROOT, "CudaSep")
CUDAINFER_ROOT = SCRIPT_DIR
PRETRAIN_DIR = os.path.join(ROOT, "pretrain")
CONVERT_SCRIPT = os.path.join(CUDAINFER_ROOT, "tools", "convert_weights.py")
INFER_EXE = os.path.join(CUDAINFER_ROOT, "build", "cudasep_infer.exe")
CSM_DIR = os.path.join(CUDAINFER_ROOT, "converted_models")

sys.path.insert(0, MSST_ROOT)

import torch


# =============================================================================
# Model registry: (name, model_type, config_path, checkpoint_path)
# =============================================================================

def find_models():
    """Return list of (description, model_type, config_path, ckpt_path)."""
    models = []
    msst_cfgs = os.path.join(MSST_ROOT, "configs")
    cudasep_cfgs = os.path.join(CUDASEP_ROOT, "configs")

    # --- MelBandRoformer ---
    kim = os.path.join(PRETRAIN_DIR, "vocal_models", "Kim_MelBandRoformer.ckpt")
    kim_cfg = os.path.join(cudasep_cfgs, "config_Kim_MelBandRoformer.yaml")
    ep3005 = os.path.join(PRETRAIN_DIR, "vocal_models",
                          "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt")
    ep3005_cfg = os.path.join(msst_cfgs, "viperx",
                              "model_mel_band_roformer_ep_3005_sdr_11.4360.yaml")
    std_mbr = os.path.join(PRETRAIN_DIR, "vocal_models",
                           "model_vocals_mel_band_roformer_sdr_8.42.ckpt")
    std_mbr_cfg = os.path.join(msst_cfgs, "config_vocals_mel_band_roformer.yaml")

    if os.path.exists(kim) and os.path.exists(kim_cfg):
        models.append(("MelBandRoformer (Kim)", "mel_band_roformer", kim_cfg, kim))
    elif os.path.exists(ep3005) and os.path.exists(ep3005_cfg):
        models.append(("MelBandRoformer (ep3005)", "mel_band_roformer", ep3005_cfg, ep3005))
    elif os.path.exists(std_mbr) and os.path.exists(std_mbr_cfg):
        models.append(("MelBandRoformer (std)", "mel_band_roformer", std_mbr_cfg, std_mbr))

    # --- BSRoformer ---
    bsr317 = os.path.join(PRETRAIN_DIR, "vocal_models",
                          "model_bs_roformer_ep_317_sdr_12.9755.ckpt")
    bsr317_cfg = os.path.join(msst_cfgs, "viperx",
                              "model_bs_roformer_ep_317_sdr_12.9755.yaml")
    bsr937 = os.path.join(PRETRAIN_DIR, "single_stem_models",
                          "model_bs_roformer_ep_937_sdr_10.5309.ckpt")
    bsr937_cfg = os.path.join(msst_cfgs, "viperx",
                              "model_bs_roformer_ep_937_sdr_10.5309.yaml")

    if os.path.exists(bsr317) and os.path.exists(bsr317_cfg):
        models.append(("BSRoformer (ep317)", "bs_roformer", bsr317_cfg, bsr317))
    elif os.path.exists(bsr937) and os.path.exists(bsr937_cfg):
        models.append(("BSRoformer (ep937)", "bs_roformer", bsr937_cfg, bsr937))

    # --- HTDemucs ---
    htd4 = os.path.join(PRETRAIN_DIR, "multi_stem_models", "HTDemucs4.th")
    htd4_cfg = os.path.join(msst_cfgs, "config_musdb18_htdemucs.yaml")

    if os.path.exists(htd4) and os.path.exists(htd4_cfg):
        models.append(("HTDemucs (4-stem)", "htdemucs", htd4_cfg, htd4))

    # --- MDX23C ---
    mdx_v = os.path.join(PRETRAIN_DIR, "vocal_models",
                         "model_vocals_mdx23c_sdr_10.17.ckpt")
    mdx_v_cfg = os.path.join(msst_cfgs, "config_vocals_mdx23c.yaml")

    if os.path.exists(mdx_v) and os.path.exists(mdx_v_cfg):
        models.append(("MDX23C (vocals)", "mdx23c", mdx_v_cfg, mdx_v))

    return models


# =============================================================================
# PyTorch baseline benchmark
# =============================================================================

def load_audio_np(path, sr=44100):
    """Load audio as numpy [channels, samples]."""
    try:
        import soundfile as sf
        data, _sr = sf.read(path, dtype='float32')
        if _sr != sr:
            import librosa
            data = librosa.resample(data.T, orig_sr=_sr, target_sr=sr).T
        if data.ndim == 1:
            data = data[:, None]
        return data.T  # [channels, samples]
    except Exception:
        import librosa
        data, _ = librosa.load(path, sr=sr, mono=False)
        if data.ndim == 1:
            data = data[None, :]
        return data


def benchmark_pytorch(model_type, config_path, ckpt_path, audio_np, device, n_runs=3):
    """Run PyTorch MSST inference and return (mean_time, std_time)."""
    from utils.settings import get_model_from_config
    from utils.model_utils import demix

    model, config = get_model_from_config(model_type, config_path)

    # Load weights
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt
    if isinstance(sd, dict):
        for k in ("state_dict", "state", "model_state_dict"):
            if k in sd:
                sd = sd[k]
                break
    new_sd = {k.replace("module.", ""): v for k, v in sd.items() if isinstance(v, torch.Tensor)}
    model.load_state_dict(new_sd)
    model = model.to(device).eval()

    mix = audio_np.copy()

    # Warmup
    with torch.inference_mode():
        _ = demix(config, model, mix, device, model_type=model_type)
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for i in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = demix(config, model, mix, device, model_type=model_type)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return np.mean(times), np.std(times)


# =============================================================================
# CudaInfer benchmark
# =============================================================================

def convert_to_csm(config_path, ckpt_path, desc):
    """Convert weights to .csm if not already done. Return .csm path."""
    os.makedirs(CSM_DIR, exist_ok=True)
    basename = os.path.splitext(os.path.basename(ckpt_path))[0]
    csm_path = os.path.join(CSM_DIR, basename + ".csm")

    if os.path.exists(csm_path):
        print(f"    [skip] {csm_path} already exists")
        return csm_path

    print(f"    Converting {desc} → {csm_path} ...")
    cmd = [
        sys.executable, CONVERT_SCRIPT,
        "--checkpoint", ckpt_path,
        "--config", config_path,
        "--output", csm_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"    CONVERT FAILED:\n{result.stderr}")
        return None
    print(f"    Done: {os.path.getsize(csm_path) / 1e6:.1f} MB")
    return csm_path


def benchmark_cudainfer(csm_path, audio_path, n_runs=3):
    """Run CudaInfer C++ inference and return (mean_time, std_time).
    
    We run the exe n_runs+1 times (first is warmup), measure wall-clock time.
    """
    if not os.path.exists(INFER_EXE):
        print(f"    ERROR: {INFER_EXE} not found!")
        return float('inf'), 0

    output_dir = os.path.join(CUDAINFER_ROOT, "benchmark_output")
    os.makedirs(output_dir, exist_ok=True)

    times = []

    for i in range(n_runs + 1):  # first run = warmup
        t0 = time.perf_counter()
        result = subprocess.run(
            [INFER_EXE, "--model", csm_path, "--input", audio_path,
             "--output", output_dir, "--stem", "0"],
            capture_output=True, text=True, timeout=600,
        )
        t1 = time.perf_counter()

        if result.returncode != 0:
            print(f"    CudaInfer FAILED (run {i}):")
            stderr_lines = result.stderr.strip().split('\n')
            stdout_lines = result.stdout.strip().split('\n')
            for line in (stdout_lines + stderr_lines)[-20:]:
                print(f"      {line}")
            return float('inf'), 0

        if i == 0:
            # Print some info from warmup run
            for line in result.stdout.strip().split('\n'):
                if 'Inference time' in line or 'Output shape' in line:
                    print(f"    {line.strip()}")
        else:
            times.append(t1 - t0)

    return np.mean(times), np.std(times)


def benchmark_cudainfer_internal(csm_path, audio_path, n_runs=3):
    """Run CudaInfer and parse its internal timing (more accurate than wall-clock)."""
    if not os.path.exists(INFER_EXE):
        return float('inf'), 0

    output_dir = os.path.join(CUDAINFER_ROOT, "benchmark_output")
    os.makedirs(output_dir, exist_ok=True)

    times = []

    for i in range(n_runs + 1):
        result = subprocess.run(
            [INFER_EXE, "--model", csm_path, "--input", audio_path,
             "--output", output_dir, "--stem", "0"],
            capture_output=True, text=True, timeout=600,
        )

        if result.returncode != 0:
            stderr_lines = result.stderr.strip().split('\n') if result.stderr else []
            stdout_lines = result.stdout.strip().split('\n') if result.stdout else []
            print(f"    CudaInfer FAILED (run {i}):")
            for line in (stdout_lines + stderr_lines)[-15:]:
                print(f"      {line}")
            return float('inf'), 0

        # Parse internal inference time from stdout
        for line in result.stdout.strip().split('\n'):
            if 'Inference time' in line:
                # "  Inference time: 1234.5 ms (RTF: 5.67x)"
                try:
                    ms_str = line.split(':')[1].strip().split(' ms')[0].strip()
                    ms = float(ms_str)
                    if i > 0:  # skip warmup
                        times.append(ms / 1000.0)
                except (ValueError, IndexError):
                    pass

    if not times:
        return float('inf'), 0

    return np.mean(times), np.std(times)


# =============================================================================
# Main benchmark runner
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CudaInfer v2 Benchmark")
    parser.add_argument("--audio", required=True, help="Path to test audio file")
    parser.add_argument("--n_runs", type=int, default=3, help="Number of timed runs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-pytorch", action="store_true",
                        help="Skip PyTorch baseline (use cached CudaSep v1 results)")
    parser.add_argument("--skip-cudainfer", action="store_true",
                        help="Skip CudaInfer C++ tests")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Filter by model type: mel_band_roformer bs_roformer htdemucs mdx23c")
    args = parser.parse_args()

    audio_path = os.path.abspath(args.audio)
    if not os.path.exists(audio_path):
        print(f"ERROR: Audio file not found: {audio_path}")
        sys.exit(1)

    # Load audio
    audio_np = load_audio_np(audio_path, sr=44100)
    audio_duration = audio_np.shape[-1] / 44100.0

    print(f"\n{'='*100}")
    print(f"{'CudaInfer v2 Benchmark':^100}")
    print(f"{'='*100}")
    print(f"Audio     : {audio_path}")
    print(f"Duration  : {audio_duration:.1f}s | Shape: {audio_np.shape}")
    print(f"GPU       : {torch.cuda.get_device_name(0)}")
    print(f"Runs      : {args.n_runs}")
    print(f"CudaInfer : {INFER_EXE}")
    print(f"{'='*100}\n")

    models = find_models()

    if args.models:
        models = [(d, mt, c, p) for d, mt, c, p in models if mt in args.models]

    if not models:
        print("No models found! Check pretrain directory.")
        return

    for desc, mt, cfg, ckpt in models:
        print(f"  * {desc}  [{mt}]")
    print()

    results = []

    for desc, model_type, config_path, ckpt_path in models:
        print(f"\n{'─'*80}")
        print(f"  {desc}")
        print(f"{'─'*80}")

        # --- PyTorch baseline ---
        pt_mean, pt_std = float('inf'), 0
        if not args.skip_pytorch:
            print("  [PyTorch] Running baseline...", flush=True)
            try:
                pt_mean, pt_std = benchmark_pytorch(
                    model_type, config_path, ckpt_path, audio_np,
                    args.device, args.n_runs
                )
                print(f"  [PyTorch] {pt_mean:.3f}s ± {pt_std:.3f}s")
            except Exception as e:
                print(f"  [PyTorch] FAILED: {e}")
                traceback.print_exc()

        # --- CudaInfer ---
        ci_mean, ci_std = float('inf'), 0
        if not args.skip_cudainfer:
            print("  [CudaInfer] Converting weights...", flush=True)
            csm_path = convert_to_csm(config_path, ckpt_path, desc)

            if csm_path:
                print("  [CudaInfer] Running inference...", flush=True)
                ci_mean, ci_std = benchmark_cudainfer_internal(
                    csm_path, audio_path, args.n_runs
                )
                if ci_mean != float('inf'):
                    print(f"  [CudaInfer] {ci_mean:.3f}s ± {ci_std:.3f}s")
                else:
                    print(f"  [CudaInfer] FAILED")

        # Speedup
        speedup = pt_mean / ci_mean if (ci_mean > 0 and ci_mean != float('inf')
                                        and pt_mean != float('inf')) else 0

        results.append({
            'desc': desc,
            'model_type': model_type,
            'pt_mean': pt_mean,
            'pt_std': pt_std,
            'ci_mean': ci_mean,
            'ci_std': ci_std,
            'speedup': speedup,
        })

    # =========================================================================
    # Summary table
    # =========================================================================
    print(f"\n\n{'='*120}")
    print(f"{'BENCHMARK RESULTS':^120}")
    print(f"{'='*120}")
    print(f"Audio: {audio_duration:.1f}s | GPU: {torch.cuda.get_device_name(0)} | Runs: {args.n_runs}")
    print(f"{'─'*120}")
    print(f"{'Model':<35} {'PyTorch (s)':<20} {'CudaInfer (s)':<20} "
          f"{'Speedup':<12} {'RTF (PT)':<12} {'RTF (CI)':<12}")
    print(f"{'─'*120}")

    for r in results:
        pt_s = (f"{r['pt_mean']:.3f} ± {r['pt_std']:.3f}"
                if r['pt_mean'] != float('inf') else "FAILED")
        ci_s = (f"{r['ci_mean']:.3f} ± {r['ci_std']:.3f}"
                if r['ci_mean'] != float('inf') else "FAILED")
        sp_s = f"{r['speedup']:.2f}x" if r['speedup'] > 0 else "N/A"
        rtf_pt = (f"{r['pt_mean'] / audio_duration:.3f}"
                  if r['pt_mean'] != float('inf') else "N/A")
        rtf_ci = (f"{r['ci_mean'] / audio_duration:.3f}"
                  if r['ci_mean'] != float('inf') else "N/A")

        print(f"{r['desc']:<35} {pt_s:<20} {ci_s:<20} "
              f"{sp_s:<12} {rtf_pt:<12} {rtf_ci:<12}")

    print(f"{'─'*120}")
    print(f"RTF = Real-Time Factor (lower is better, <1.0 = faster than real-time)")
    print(f"{'='*120}")


if __name__ == "__main__":
    main()
