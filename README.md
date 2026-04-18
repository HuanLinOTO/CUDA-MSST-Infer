# CUDA-MSST-Infer

高性能 C++/CUDA 音乐源分离推理引擎。作为 [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) 的 PyTorch 推理的高性能替代方案。

## 支持的模型

| 架构                        | 支持状态 |
| --------------------------- | -------- |
| BSRoformer                  | ✅       |
| MelBandRoformer             | ✅       |
| HTDemucs4 (4 音轨 & 6 音轨) | ✅       |
| MDX23C                      | ✅       |

## 性能表现

测试环境：RTX 4060 Ti 16GB, CUDA 12.5, cuDNN 9.20, 68.5 秒立体声音频 @ 44.1kHz

| 模型                  | PyTorch (ms) | CudaInfer (ms) | 加速比    |
| --------------------- | ------------ | -------------- | --------- |
| MelBandRoformer (Kim) | 7957         | 5608           | **1.42x** |
| BSRoformer (ep317)    | 17295        | 13021          | **1.33x** |
| MDX23C (vocals)       | 7462         | 4624           | **1.61x** |
| HTDemucs4 (4-stem)    | 2980         | 1557           | **1.91x** |

> PyTorch: 2.2.0+cu121, AMP autocast 开启, 1 次预热 + 3 次计时取均值

## 环境要求

- NVIDIA GPU (计算能力 ≥ 8.0, Ampere 或更新架构)
- CUDA Toolkit 12.0+
- cuDNN 9.x ([nvidia-cudnn-cu12](https://pypi.org/project/nvidia-cudnn-cu12/))
- CMake 3.20+, Ninja
- C++17 编译器 (Windows 用 MSVC 2019+, Linux 用 GCC 11+)
- FFmpeg (用于音频 I/O)

## 编译构建

### Windows

```powershell
# 1. 安装 cuDNN
pip install nvidia-cudnn-cu12==9.5.1.17

# 2. 查找 cuDNN 路径
$CUDNN = python -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__))"

# 3. 生成导入库 (需在 VS 开发者命令行中执行)
# 详见 docs/build_windows.md

# 4. 配置并编译
cmake -S . -B build -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" `
  -DCUDNN_INCLUDE_DIR="$CUDNN/include" `
  -DCUDNN_LIB_DIR="cudnn_libs"

cmake --build build -j
```

### Linux

```bash
pip install nvidia-cudnn-cu12==9.5.1.17
CUDNN=$(python3 -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__))")

cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
  -DCUDNN_INCLUDE_DIR="$CUDNN/include" \
  -DCUDNN_LIB_DIR="$CUDNN/lib"

cmake --build build -j$(nproc)
```

### CI / GitHub Actions

本项目提供 Windows 和 Linux 的自动化构建，详见 `.github/workflows/build.yml`。
预编译二进制文件可在 [Releases](../../releases) 页面下载。

## 使用方法

```bash
# 基础推理
./cudasep_infer --model path/to/model.csm --input song.mp3 --output vocals/

# 使用 FP16 权重 (在支持的模型上更快)
./cudasep_infer --model path/to/model_fp16.csm --input song.mp3 --output vocals/ --fp16

# 重叠参数 (单位：秒，默认 2)
./cudasep_infer --model model.csm --input song.mp3 --output out/ --overlap 4

# HTTP 服务器模式
./cudasep_infer --serve --model-dir ./csm_models --host 127.0.0.1 --port 8080
```

启动服务器模式后，在浏览器打开 `http://127.0.0.1:8080/`，即可上传音频、选择 `.csm` 模型，并在前端查看计时、进度、日志和全部输出轨道。若模型本身不包含 `other`，服务器会自动用原混音减去所有已分离轨道生成 `other`。

## 预转换权重

全部 23 个支持模型的预转换 `.csm` 权重文件 (FP32 和 FP16 版本) 已上传至 HuggingFace：

👉 **[SVCFusion/Cuda-MSST-Infer-Models](https://huggingface.co/SVCFusion/Cuda-MSST-Infer-Models)**

```bash
# 下载模型
huggingface-cli download SVCFusion/Cuda-MSST-Infer-Models fp32/Kim_MelBandRoformer.csm --local-dir ./models
```

## 权重转换

将 PyTorch `.ckpt` / `.th` 权重转换为 `.csm` 格式：

```bash
# FP32
python tools/convert_weights.py \
  --checkpoint path/to/model.ckpt \
  --config path/to/config.yaml \
  --output model.csm

# FP16 (半精度)
python tools/convert_weights.py \
  --checkpoint path/to/model.ckpt \
  --config path/to/config.yaml \
  --output model_fp16.csm \
  --half
```

## CSM 格式说明

`.csm` (CudaSep Model) 是一种简单的二进制格式：

```
魔数      : "CSM\0"   (4 字节)
版本      : uint32    (= 1)
配置长度   : uint32    (JSON 字节长度)
配置      : UTF-8 JSON (模型超参数 + 元数据)
张量数量   : uint32
张量数据   : [name_len, name, rank, shape[], dtype, raw_data] × N
```

`dtype`: `0` = float32, `1` = float16, `2` = int64

## 项目结构

```
src/
  main.cpp              — 命令行入口
  model_bs_roformer.*   — BSRoformer 推理
  model_mel_band_roformer.* — MelBandRoformer 推理
  model_htdemucs.*      — HTDemucs4 推理
  model_mdx23c.*        — MDX23C 推理
  ops_linear.cu         — GEMM / 线性层 (TF32 + FP16)
  ops_attention.cu      — 多头注意力 + Flash Attention
  ops_fused.cu          — 融合算子 (linear+GELU, linear+sigmoid 等)
  ops_norm.cu           — LayerNorm, GroupNorm, RMSNorm
  ops_stft.cu           — STFT / iSTFT (基于 cuFFT)
  ops_conv.cu           — 卷积 (基于 cuDNN)
  tensor.cu/h           — GPU 张量抽象
  weights.cu/h          — 权重加载 + FP16 转换
tools/
  convert_weights.py    — PyTorch → .csm 转换器
  batch_convert_and_upload.py — 批量转换 + 上传 HuggingFace
```

## 许可证

AGPL-3.0
