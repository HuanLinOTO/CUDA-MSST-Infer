#!/usr/bin/env python3
"""
Quick single-chunk benchmark: PyTorch vs CudaInfer
Measures just the model forward pass (no overlap-add overhead).
"""
import os, sys, time, subprocess
import torch
import numpy as np

WORKSPACE = r"d:\Projects\Joy\CudaMSST"
MSST_DIR = os.path.join(WORKSPACE, "Music-Source-Separation-Training")
sys.path.insert(0, MSST_DIR)

def benchmark_pytorch_mbr():
    """Single chunk PyTorch MBR benchmark"""
    import yaml
    config_path = os.path.join(WORKSPACE, "CudaSep", "configs", "config_Kim_MelBandRoformer.yaml")
    ckpt_path = os.path.join(WORKSPACE, "pretrain", "vocal_models", "Kim_MelBandRoformer.ckpt")

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    from models.bs_roformer.mel_band_roformer import MelBandRoformer
    model_cfg = config.get("model", config)
    model = MelBandRoformer(**{k: v for k, v in model_cfg.items() if k != "model_type"})
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    model.cuda().eval()

    chunk_size = config["audio"]["chunk_size"]  # 352800
    # Create a random chunk
    chunk = torch.randn(1, 2, chunk_size, device="cuda")

    # Warmup
    with torch.no_grad(), torch.cuda.amp.autocast():
        for _ in range(3):
            _ = model(chunk)
            torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for _ in range(5):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(chunk)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

    del model
    torch.cuda.empty_cache()
    return np.mean(times), np.std(times), chunk_size


if __name__ == "__main__":
    print("Single-chunk MBR benchmark (PyTorch)")
    avg, std, cs = benchmark_pytorch_mbr()
    dur = cs / 44100.0
    print(f"  Chunk: {cs} samples ({dur:.2f}s)")
    print(f"  Forward: {avg*1000:.1f}ms ± {std*1000:.1f}ms")
    print(f"  RTF: {avg/dur:.3f}")
