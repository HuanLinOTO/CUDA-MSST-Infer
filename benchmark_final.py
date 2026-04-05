#!/usr/bin/env python3
"""
Benchmark: PyTorch MSST vs CudaInfer C++/CUDA binary
Measures wall-clock inference time for each model on the same audio.
"""
import os, sys, time, subprocess
import torch
import numpy as np

WORKSPACE = r"d:\Projects\Joy\CudaMSST"
MSST_DIR = os.path.join(WORKSPACE, "Music-Source-Separation-Training")
CUDASEP_EXE = os.path.join(WORKSPACE, "CudaInfer", "build", "cudasep_infer.exe")
AUDIO_FILE = os.path.join(WORKSPACE, "水饺 Half.mp3")
CONVERTED_DIR = os.path.join(WORKSPACE, "CudaInfer", "converted_models")

sys.path.insert(0, MSST_DIR)

MODELS = {
    "MelBandRoformer": {
        "checkpoint": os.path.join(WORKSPACE, "pretrain", "vocal_models", "Kim_MelBandRoformer.ckpt"),
        "config": os.path.join(WORKSPACE, "CudaSep", "configs", "config_Kim_MelBandRoformer.yaml"),
        "csm": os.path.join(CONVERTED_DIR, "Kim_MelBandRoformer.csm"),
        "model_type": "mel_band_roformer",
    },
    "BSRoformer": {
        "checkpoint": os.path.join(WORKSPACE, "pretrain", "vocal_models", "model_bs_roformer_ep_317_sdr_12.9755.ckpt"),
        "config": os.path.join(WORKSPACE, "Music-Source-Separation-Training", "configs", "viperx", "model_bs_roformer_ep_317_sdr_12.9755.yaml"),
        "csm": os.path.join(CONVERTED_DIR, "BSRoformer_ep317.csm"),
        "model_type": "bs_roformer",
    },
    "HTDemucs": {
        "checkpoint": os.path.join(WORKSPACE, "pretrain", "multi_stem_models", "HTDemucs4.th"),
        "config": os.path.join(WORKSPACE, "Music-Source-Separation-Training", "configs", "config_musdb18_htdemucs.yaml"),
        "csm": os.path.join(CONVERTED_DIR, "HTDemucs4.csm"),
        "model_type": "htdemucs",
    },
    "MDX23C": {
        "checkpoint": os.path.join(WORKSPACE, "pretrain", "vocal_models", "model_vocals_mdx23c_sdr_10.17.ckpt"),
        "config": os.path.join(WORKSPACE, "Music-Source-Separation-Training", "configs", "config_vocals_mdx23c.yaml"),
        "csm": os.path.join(CONVERTED_DIR, "MDX23C_vocals.csm"),
        "model_type": "mdx23c",
    },
}

N_WARMUP = 1
N_RUNS = 3


def get_audio_duration():
    import soundfile as sf
    info = sf.info(AUDIO_FILE)
    return info.duration


def benchmark_pytorch(model_name, model_info, audio_duration):
    """Run PyTorch inference and return average time."""
    import yaml
    import soundfile as sf

    model_type = model_info["model_type"]
    config_path = model_info["config"]
    checkpoint_path = model_info["checkpoint"]

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Load model
    if model_type == "mel_band_roformer":
        from models.bs_roformer.mel_band_roformer import MelBandRoformer
        model_cfg = config.get("model", config)
        model = MelBandRoformer(**{k: v for k, v in model_cfg.items()})
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        model.load_state_dict(sd)
    elif model_type == "bs_roformer":
        from models.bs_roformer.bs_roformer import BSRoformer
        model_cfg = config.get("model", config)
        model = BSRoformer(**{k: v for k, v in model_cfg.items()})
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        model.load_state_dict(sd)
    elif model_type == "htdemucs":
        from models.demucs4ht import get_model as get_demucs_model
        from utils.settings import load_config
        cfg = load_config(model_type, config_path)
        model = get_demucs_model(cfg)
        # Load checkpoint directly (load_start_checkpoint requires argparse Namespace)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if 'state' in ckpt:
            ckpt = ckpt['state']
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        model.load_state_dict(ckpt)
    elif model_type == "mdx23c":
        from models.mdx23c_tfc_tdf_v3 import TFC_TDF_net
        from utils.settings import load_config
        cfg = load_config(model_type, config_path)
        model = TFC_TDF_net(cfg)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        model.load_state_dict(sd)
    else:
        print(f"  [SKIP] Unknown model type: {model_type}")
        return None

    model.cuda().eval()

    # Load audio
    audio_data, sr = sf.read(AUDIO_FILE, dtype='float32')
    if audio_data.ndim == 1:
        audio_data = np.stack([audio_data, audio_data])
    else:
        audio_data = audio_data.T
    mix = torch.tensor(audio_data, dtype=torch.float32)

    from utils.model_utils import demix
    from utils.settings import load_config
    cfg = load_config(model_type, config_path)

    # Warmup
    with torch.no_grad(), torch.cuda.amp.autocast():
        for _ in range(N_WARMUP):
            _ = demix(cfg, model, mix, device=torch.device("cuda"), model_type=model_type)

    # Benchmark
    times = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for run in range(N_RUNS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = demix(cfg, model, mix, device=torch.device("cuda"), model_type=model_type)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
            print(f"    run {run+1}: {times[-1]:.3f}s")

    del model
    torch.cuda.empty_cache()

    avg = np.mean(times)
    std = np.std(times)
    return avg, std, times


def benchmark_cudainfer(model_name, model_info, audio_duration):
    """Run CudaInfer binary and parse inference time from output."""
    csm_path = model_info["csm"]
    if not os.path.exists(csm_path):
        print(f"  [CudaInfer] Missing: {csm_path}")
        return None

    output_dir = os.path.join(WORKSPACE, "CudaInfer", "bench_output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{model_name}_output.wav")

    cmd = [CUDASEP_EXE, "--model", csm_path, "--input", AUDIO_FILE, "--output", output_file]

    # Warmup
    for _ in range(N_WARMUP):
        subprocess.run(cmd, capture_output=True, timeout=600)

    # Benchmark
    times = []
    for run in range(N_RUNS):
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  [FAILED] Run {run+1}: {result.stderr[:200]}")
            return None

        # Parse "Inference time: XXXX.X ms" from stdout
        for line in result.stdout.split("\n"):
            if "Inference time:" in line:
                try:
                    ms_str = line.split("Inference time:")[1].split("ms")[0].strip()
                    infer_time = float(ms_str) / 1000.0
                    times.append(infer_time)
                    print(f"    run {run+1}: {infer_time:.3f}s")
                    break
                except:
                    pass

    if not times:
        return None

    avg = np.mean(times)
    std = np.std(times)
    return avg, std, times


def main():
    print("=" * 80)
    print("CudaInfer Benchmark: PyTorch MSST vs C++/CUDA Binary")
    print("=" * 80)

    audio_duration = get_audio_duration()
    print(f"Audio: {os.path.basename(AUDIO_FILE)} ({audio_duration:.1f}s)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Runs: {N_RUNS} (warmup: {N_WARMUP})")
    print()

    results = {}

    for model_name, model_info in MODELS.items():
        print(f"--- {model_name} ---")

        # PyTorch benchmark
        print(f"  [PyTorch] Running...")
        try:
            pt_result = benchmark_pytorch(model_name, model_info, audio_duration)
            if pt_result:
                pt_avg, pt_std, _ = pt_result
                print(f"  [PyTorch] {pt_avg:.3f}s +/- {pt_std:.3f}s  (RTF: {audio_duration/pt_avg:.2f}x)")
            else:
                pt_avg = None
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [PyTorch] ERROR: {e}")
            pt_avg = None

        # CudaInfer benchmark
        print(f"  [CudaInfer] Running...")
        try:
            ci_result = benchmark_cudainfer(model_name, model_info, audio_duration)
            if ci_result:
                ci_avg, ci_std, _ = ci_result
                print(f"  [CudaInfer] {ci_avg:.3f}s +/- {ci_std:.3f}s  (RTF: {audio_duration/ci_avg:.2f}x)")
            else:
                ci_avg = None
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [CudaInfer] ERROR: {e}")
            ci_avg = None

        # Comparison
        if pt_avg and ci_avg:
            ratio = pt_avg / ci_avg
            if ratio >= 1:
                print(f"  => CudaInfer is {ratio:.2f}x FASTER than PyTorch")
            else:
                print(f"  => CudaInfer is {1/ratio:.2f}x SLOWER than PyTorch")
        print()

        results[model_name] = {
            "pytorch": pt_avg,
            "cudainfer": ci_avg,
        }

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Model':<20} {'PyTorch (s)':<14} {'CudaInfer (s)':<14} {'Ratio':<16}")
    print("-" * 64)
    for model_name, r in results.items():
        pt_str = f"{r['pytorch']:.3f}" if r['pytorch'] else "FAIL"
        ci_str = f"{r['cudainfer']:.3f}" if r['cudainfer'] else "FAIL"
        if r['pytorch'] and r['cudainfer']:
            ratio = r['pytorch'] / r['cudainfer']
            if ratio >= 1:
                sp_str = f"{ratio:.2f}x faster"
            else:
                sp_str = f"{1/ratio:.2f}x slower"
        else:
            sp_str = "N/A"
        print(f"{model_name:<20} {pt_str:<14} {ci_str:<14} {sp_str:<16}")
    print(f"\nAudio: {audio_duration:.1f}s | GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
