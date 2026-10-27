

<h1 align="center">DPDFNet</h1>

<p align="center">
  <strong>Real-time speech enhancement for recordings, live streams, and edge devices.</strong><br>
  Pretrained 8, 16, and 48 kHz models for the CLI, Python API, and stateful streaming.
</p>

<p align="center">
  <a href="https://pypi.org/project/dpdfnet/"><img src="https://img.shields.io/pypi/v/dpdfnet?label=PyPI&style=for-the-badge" alt="PyPI version"></a>&nbsp;
  <a href="https://arxiv.org/abs/2512.16420"><img src="https://img.shields.io/badge/arXiv-2512.16420-b31b1b?style=for-the-badge" alt="arXiv paper"></a>&nbsp;
  <a href="https://huggingface.co/Ceva-IP/DPDFNet"><img src="https://img.shields.io/badge/Hugging%20Face-Models-FFD21E?style=for-the-badge" alt="Hugging Face models"></a>&nbsp;
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge" alt="Apache 2.0 license"></a>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/Ceva-IP/DPDFNetDemo"><strong>Live demo</strong></a>
  &nbsp;&middot;&nbsp;
  <a href="https://ceva-ip.github.io/DPDFNet/"><strong>Audio examples</strong></a>
  &nbsp;&middot;&nbsp;
  <a href="https://huggingface.co/Ceva-IP/DPDFNet"><strong>Pretrained models</strong></a>
  &nbsp;&middot;&nbsp;
  <a href="https://arxiv.org/abs/2512.16420"><strong>Paper</strong></a>
</p>

<br>

<p align="center">
  <img src="figures/dpdfnet2_48khz_hr_gif.gif" width="760" alt="DPDFNet noisy-to-enhanced spectrogram comparison" />
</p>

## Model Profile

### 8 kHz models

| Model | Params [M] | MACs [G] | TFLite Size [MB] | ONNX Size [MB] |
| --- | :---: | :---: | :---: | :---: |
| dpdfnet2_8khz | 2.51 | 1.29 | 10.5 | 9.7 |
| dpdfnet8_8khz | 3.56 | 3.99 | 16.5 | 13.8 |

### 16 kHz models

| Model | Params [M] | MACs [G] | TFLite Size [MB] | ONNX Size [MB] |
| --- | :---: | :---: | :---: | :---: |
| baseline | 2.31 | 0.36 | 8.5 | 8.3 |
| dpdfnet2 | 2.49 | 1.35 | 10.7 | 9.7 |
| dpdfnet4 | 2.84 | 2.36 | 12.9 | 11.1 |
| dpdfnet8 | 3.54 | 4.37 | 17.2 | 13.9 |

### 48 kHz models

| Model | Params [M] | MACs [G] | TFLite Size [MB] | ONNX Size [MB] |
| --- | :---: | :---: | :---: | :---: |
| dpdfnet2_48khz_hr | 2.58 | 2.42 | 11.6 | 10.0 |
| dpdfnet8_48khz_hr | 3.63 | 7.17 | 18.7 | 14.2 |

### Optimized native inference

The [optimized C runtime](optimized_inference/README.md) supports
`dpdfnet2_48khz_hr` and `dpdfnet8_48khz_hr` with the final W7A8 kernels and
FP32 recurrent state. It includes both model weights, a shared C API,
assembly kernels, examples, contract tests and build documentation.

The supported target is Linux x86-64 with AVX2/FMA and GCC, including WSL2.
Inference uses one calling thread and requires no ONNX Runtime.
See [performance and output quality](optimized_inference/docs/PERFORMANCE.md)
for measurements and their scope; the 50 ms algorithmic delay is unchanged.

The native library is also available as an opt-in `StreamEnhancer` backend,
sharing the Python streaming audio frontend; see
[native streaming usage](package/README.md#real-time-microphone-enhancement).

## Install the PyPI Package

For CPU-only ONNX inference using the packaged CLI and Python API:

```bash
pip install dpdfnet
```

### CLI Example

```bash
# Enhance one file
dpdfnet enhance noisy.wav enhanced.wav --model dpdfnet4 --attn-limit-db 12

# Enhance a directory (uses all CPU cores by default)
dpdfnet enhance-dir ./noisy_wavs ./enhanced_wavs --model dpdfnet2 --attn-limit-db 12

# Enhance a directory with a fixed worker count
dpdfnet enhance-dir ./noisy_wavs ./enhanced_wavs --model dpdfnet2 --workers 4 --attn-limit-db 12

# Download models
dpdfnet download
dpdfnet download dpdfnet8
dpdfnet download dpdfnet2_8khz
dpdfnet download dpdfnet4 --force
```

### Python API Example

```python
import soundfile as sf
import dpdfnet

# In-memory enhancement:
audio, sr = sf.read("noisy.wav")
enhanced = dpdfnet.enhance(audio, sample_rate=sr, model="dpdfnet4", attn_limit_db=12)
sf.write("enhanced.wav", enhanced, sr)

# Enhance one file:
out_path = dpdfnet.enhance_file("noisy.wav", model="dpdfnet2", attn_limit_db=12)
print(out_path)

# Model listing:
for row in dpdfnet.available_models():
    print(row["name"], row["ready"], row["cached"])

# Download models:
dpdfnet.download()				# All models
dpdfnet.download("dpdfnet4")	# Specific model
dpdfnet.download("dpdfnet2_8khz")
```

### Streaming (Real-Time) API

For `dpdfnet8_48khz_hr`, use native 48 kHz audio and 480-sample (10 ms) chunks:

```python
import dpdfnet

# Construct outside the audio callback; construction includes warmup.
enhancer = dpdfnet.StreamEnhancer(model="dpdfnet8_48khz_hr")
# In your callback, with a mono float32 input block:
# enhanced = enhancer.process(input_block, sample_rate=enhancer.sample_rate)
```

The optional float32 OpenVINO CPU backend lowers processing latency on the
tested Intel CPU, using more memory. Install `dpdfnet[openvino]` and pass
`backend="openvino"` to select it. The portable default remains ONNX Runtime.
Neither backend changes model weights, skips frames, or removes lookahead.

See the [microphone example and streaming API guidance](package/README.md#real-time-microphone-enhancement)
for callback buffering, sample rates, and reset/flush behavior, and the
[CPU benchmark report](docs/cpu-streaming-performance.md) for measured tradeoffs.

## Run From Source

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Download models

Model files are not bundled in this repository.\
Download PyTorch checkpoints, TFLite, and ONNX models from Hugging Face:

```bash
pip install -U "huggingface_hub[cli]"

# create target dirs
mkdir -p model_zoo/{checkpoints,onnx,tflite}

# PyTorch checkpoints (HF path: checkpoints/* -> local: model_zoo/checkpoints/*)
hf download Ceva-IP/DPDFNet \
  --include "checkpoints/*.pth" \
  --local-dir model_zoo \

# ONNX models (HF path: onnx/* -> local: model_zoo/onnx/*)
hf download Ceva-IP/DPDFNet \
  --include "onnx/*.onnx" \
  --local-dir model_zoo \

# TFLite models (HF path: *.tflite at repo root -> local: model_zoo/tflite/*)
hf download Ceva-IP/DPDFNet \
  --include "*.tflite" \
  --local-dir model_zoo/tflite \
```

### 3) Run offline enhancement

Put one or more `*.wav` files in `./noisy_wavs`, then choose one:

#### Option A: `TFLite`

```bash
python -m tflite_model.infer_dpdfnet_tflite \
	--noisy_dir ./noisy_wavs \
	--enhanced_dir ./enhanced_wavs \
	--model_name dpdfnet4 \
	--workers 5 \
	--attn-limit-db 12
```

#### Option B: `ONNX`

```bash
python -m onnx_model.infer_dpdfnet_onnx \
  	--noisy_dir ./noisy_wavs \
	--enhanced_dir ./enhanced_wavs \
	--model_name dpdfnet4 \
	--workers 5 \
	--attn-limit-db 12
```

To export ONNX models from checkpoints, use the exporter that matches the sample-rate family.
Set `--dprnn-num-blocks` to match the checkpoint variant; for 8 kHz and 48 kHz HR exports,
this also determines the ONNX metadata profile:

```bash
# 8 kHz
python -m onnx_model.export_dpdfnet_8khz_to_onnx \
	--checkpoint model_zoo/checkpoints/dpdfnet2_8khz.pth \
	--output model_zoo/onnx/dpdfnet2_8khz.onnx \
	--dprnn-num-blocks 2

# 16 kHz
python -m onnx_model.export_dpdfnet_to_onnx \
	--checkpoint model_zoo/checkpoints/dpdfnet4.pth \
	--output model_zoo/onnx/dpdfnet4.onnx \
	--dprnn-num-blocks 4

# 48 kHz HR
python -m onnx_model.export_dpdfnet_48khz_hr_to_onnx \
	--checkpoint model_zoo/checkpoints/dpdfnet8_48khz_hr.pth \
	--output model_zoo/onnx/dpdfnet8_48khz_hr.onnx \
	--dprnn-num-blocks 8
```

Enhanced files are written as:

```text
<original_stem>_<model_name>.wav
```

## Audio Samples & Demo

- Project page with examples: https://ceva-ip.github.io/DPDFNet/
- Gradio application: https://huggingface.co/spaces/Ceva-IP/DPDFNetDemo
- Hugging Face model hub: https://huggingface.co/Ceva-IP/DPDFNet
- Evaluation dataset used in the paper: https://huggingface.co/datasets/Ceva-IP/DPDFNet_EvalSet

## Real-Time Demo

![Real-time DPDFNet demo screenshot](figures/live_demo.png)

Run:

```bash
pip install -e ./package
python -m real_time_demo --model dpdfnet8_48khz_hr

# Optional faster CPU runtime:
pip install -e './package[openvino]'
python -m real_time_demo --model dpdfnet8_48khz_hr --backend openvino
```

How it works:
- Captures microphone audio in streaming hops.
- Enhances each hop frame-by-frame with ONNX.
- Displays live noisy vs enhanced spectrograms.
- Allows you to control the noise‑reduction level during playback: `0` for the raw stream and `1` for the fully enhanced stream.
- Enables the use of AGC during playback.

Select another model with `--model`; use `--help` for options. The display's
timing includes STFT, inference, and ISTFT. Plot FFTs run on the UI thread,
not in the audio callback.

## Troubleshooting / FAQ

#### `Q: Model files are missing (TFLite / ONNX / checkpoints)`
- Run the Hugging Face download commands from the `Run From Source` section.
- Confirm files are in:
  - `model_zoo/tflite/`
  - `model_zoo/onnx/`
  - `model_zoo/checkpoints/`

#### `Q: No .wav files found`
- Both offline scripts scan only the exact folder given by `--noisy_dir` (non-recursive).
- Ensure input files use `.wav` extension.

#### `Q: Real-time demo has audio device errors`
- Check microphone permissions and default input/output device settings.
- Install host audio dependencies for `sounddevice` (PortAudio packages on your OS).

#### `Q: Real-time GUI does not open`
- Ensure Qt dependencies from `requirements.txt` installed successfully.
- On headless servers, run offline enhancement instead.

#### `Q: I get import/module errors when running commands`
- Run from repo root and use module form exactly as documented (`python -m ...`).
- Activate your virtual environment before running commands.

#### `Q: CPU is too slow for my target`
- Use the model's native sample rate, one-hop callbacks, and construct `StreamEnhancer` before starting the device. Try its optional `openvino` backend; see the [CPU benchmark report](docs/cpu-streaming-performance.md).
- Try smaller models (`baseline`, `dpdfnet2`).
- Benchmark ONNX runtime using `python -m onnx_model.infer_dpdfnet_onnx ...` and compare RTF.

## Evaluation Metrics

To compute *intrusive* and *non-intrusive* metrics on our [DPDFNet EvalSet](https://huggingface.co/datasets/Ceva-IP/DPDFNet_EvalSet), we use the tools listed below. For aggregate quality reporting, we rely on PRISM, the scale‑normalized composite metric introduced in the DPDFNet paper.

### Intrusive metrics: PESQ, STOI, SI-SNR
We provide a dedicated script, `pesq_stoi_sisnr_calc.py`, which computes **PESQ**, **STOI**, and **SI-SNR** for paired *reference* and *enhanced* audio. The script includes a built-in auto-alignment step that corrects small start-time offsets and drift between the reference and the enhanced signals before scoring, to ensure fair comparisons.

### Non-intrusive metrics
- **DNSMOS (P.835 & P.808)** - We use the **official** DNSMOS local inference script from the DNS Challenge repository: [`dnsmos_local.py`](https://github.com/microsoft/DNS-Challenge/blob/master/DNSMOS/dnsmos_local.py). Please follow their installation and model download instructions in that project before running. 
- **NISQA v2** - We use the **official** NISQA project: <https://github.com/gabrielmittag/NISQA>. Refer to their README for environment setup, pretrained model weights, and inference commands (*e.g.*, running `nisqa_predict.py` on a folder of WAVs).

## Built with DPDFNet

Explore [applications, plugins, libraries and research projects](COMMUNITY.md)
built with DPDFNet.

Using DPDFNet in your project? Open an issue or submit a pull request
to add it to the list.

## Citation

```bibtex
@article{rika2025dpdfnet,
 title = {DPDFNet: Boosting DeepFilterNet2 via Dual-Path RNN},
 author = {Rika, Daniel and Sapir, Nino and Gus, Ido},
 year = {2025},
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
