

<h1 align="center">DPDFNet</h1>

<p align="center">
  <strong>Real-time speech enhancement for recordings, live streams, and edge devices.</strong><br>
  Pretrained 8, 16, and 48 kHz models, and a native CPU runtime for the 48 kHz models with a CLI, Python API, and stateful streaming.
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

### Inference in this repository

The `dpdfnet` package runs `dpdfnet2_48khz_hr` and `dpdfnet8_48khz_hr` on the
[optimized C runtime](optimized_inference/README.md): W7A8 integer kernels with
FP32 recurrent state, hand-written AVX2/FMA (and AVX-VNNI) assembly, and the
calling thread plus one helper thread, which cuts per-hop latency by 25–45% at
identical output. On the tested CPUs a hop takes 5–7× less time than the
original FP32 ONNX model under ONNX Runtime, which the package no longer uses.
See [performance and output quality](optimized_inference/docs/PERFORMANCE.md)
for measurements and their scope; the 50 ms algorithmic delay is unchanged.

It runs on Linux and Windows 10 1607+ on x86-64 CPUs with AVX2 and FMA. The 8
and 16 kHz models above are not part of the package: their PyTorch
checkpoints and ONNX/TFLite exports stay on
[Hugging Face](https://huggingface.co/Ceva-IP/DPDFNet) for research use, and
`dpdfnet` 0.x on PyPI is the last release that ran them.

## Install the Package

Wheels for Linux (glibc 2.27+) and Windows x86-64 bundle the library and
both models' weights, so nothing is downloaded at run time. The repository's
`Package` workflow builds and tests them. Once 1.0 is on PyPI, install it with
`pip install "dpdfnet>=1"`; without the `>=1`, unsupported platforms silently
get the 0.x ONNX release, whose API differs. To build a wheel from a checkout, with
GCC 12+ and CMake 3.18+ (MSYS2 UCRT64 GCC on Windows, see
[BUILD.md](optimized_inference/docs/BUILD.md)):

```bash
pip install ./package
```

This compiles the profile-guided release build and runs all of the native
runtime's contract tests before packaging, which takes a few minutes.

### CLI Example

```bash
# Enhance one file (default model: dpdfnet2_48khz_hr)
dpdfnet enhance noisy.wav enhanced.wav --model dpdfnet8_48khz_hr --attn-limit-db 12

# Enhance a directory (uses all CPU cores by default)
dpdfnet enhance-dir ./noisy_wavs ./enhanced_wavs --attn-limit-db 12

# Enhance a directory with a fixed worker count
dpdfnet enhance-dir ./noisy_wavs ./enhanced_wavs --workers 4 --attn-limit-db 12

# List the models and where the native library was found
dpdfnet models
```

Input at any sample rate is resampled to 48 kHz for the model and back.

### Python API Example

```python
import soundfile as sf
import dpdfnet

# In-memory enhancement:
audio, sr = sf.read("noisy.wav")
enhanced = dpdfnet.enhance(audio, sample_rate=sr, model="dpdfnet8_48khz_hr", attn_limit_db=12)
sf.write("enhanced.wav", enhanced, sr)

# Enhance one file:
out_path = dpdfnet.enhance_file("noisy.wav", attn_limit_db=12)
print(out_path)

# Model listing:
for row in dpdfnet.available_models():
    print(row["name"], row["ready"])
```

### Streaming (Real-Time) API

Use 48 kHz audio and 480-sample (10 ms) chunks:

```python
import dpdfnet

# Construct outside the audio callback; construction includes warmup.
enhancer = dpdfnet.StreamEnhancer(model="dpdfnet8_48khz_hr")
# In your callback, with a mono float32 input block:
# enhanced = enhancer.process(input_block, sample_rate=enhancer.sample_rate)
```

Pass `single_thread=True` to run each hop on the calling thread only, for
example when running many streams in parallel. Output is identical either way.

See the [microphone example and streaming API guidance](package/README.md#real-time-microphone-enhancement)
for callback buffering, sample rates, and reset/flush behavior.

## Run From Source

### 1) Build the native runtime and install the package for development

```bash
python optimized_inference/tools/build.py   # PGO build and contract tests
python -m venv .venv
source .venv/bin/activate
pip install -e './package[test]'
python -m pytest package/tests
```

An editable install finds the library in `optimized_inference/build`. Set
`DPDFNET_NATIVE_LIBRARY` to use a library built elsewhere.

### 2) Research, training and ONNX export

`requirements.txt` holds the PyTorch environment for the model code in
`model/` and the ONNX exporters in `onnx_model/`:

```bash
pip install -r requirements.txt
```

Model files are not bundled in this repository. Download PyTorch
checkpoints from Hugging Face:

```bash
pip install -U "huggingface_hub[cli]"
mkdir -p model_zoo/checkpoints

# PyTorch checkpoints (HF path: checkpoints/* -> local: model_zoo/checkpoints/*)
hf download Ceva-IP/DPDFNet \
  --include "checkpoints/*.pth" \
  --local-dir model_zoo
```

To export ONNX models from checkpoints, use the exporter that matches the sample-rate family.
Set `--dprnn-num-blocks` to match the checkpoint variant; for 8 kHz and 48 kHz HR exports,
this also determines the ONNX metadata profile. Each exporter checks its
output against PyTorch with ONNX Runtime:

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

The native weights in `optimized_inference/models` were fitted from the
48 kHz HR ONNX exports and record each source file's SHA-256. The native
graphs are maintained by hand, so a retrained checkpoint needs a new native
port before the package can run it.

## Audio Samples & Demo

- Project page with examples: https://ceva-ip.github.io/DPDFNet/
- Gradio application: https://huggingface.co/spaces/Ceva-IP/DPDFNetDemo
- Hugging Face model hub: https://huggingface.co/Ceva-IP/DPDFNet
- Evaluation dataset used in the paper: https://huggingface.co/datasets/Ceva-IP/DPDFNet_EvalSet

## Real-Time Demo

![Real-time DPDFNet demo screenshot](figures/live_demo.png)

With the package installed as in [Run From Source](#run-from-source) and the
demo's dependencies from `requirements.txt`:

```bash
python -m real_time_demo --model dpdfnet8_48khz_hr
```

How it works:
- Captures microphone audio in 10 ms hops at 48 kHz.
- Enhances each hop on the native runtime.
- Displays live noisy vs enhanced spectrograms.
- Allows you to control the noise‑reduction level during playback: `0` for the raw stream and `1` for the fully enhanced stream.
- Enables the use of AGC during playback.

Select the other model with `--model`, or run inference on the audio thread
only with `--single-thread`; use `--help` for options. The display's timing
includes STFT, inference, and ISTFT. Plot FFTs run on the UI thread, not in
the audio callback.

## Troubleshooting / FAQ

#### `Q: "The native DPDFNet library was not found"`
- Install a wheel, or in a source checkout run `python optimized_inference/tools/build.py`.
- If you built the library elsewhere, point `DPDFNET_NATIVE_LIBRARY` at it.

#### `Q: "needs an x86-64 CPU with AVX2 and FMA"`
- The runtime has no fallback for older x86 CPUs, ARM or macOS. Use `dpdfnet` 0.x (ONNX Runtime) there.

#### `Q: Real-time demo has audio device errors`
- Check microphone permissions and default input/output device settings.
- Install host audio dependencies for `sounddevice` (PortAudio packages on your OS).

#### `Q: Real-time GUI does not open`
- Ensure Qt dependencies from `requirements.txt` installed successfully.
- On headless servers, run offline enhancement instead.

#### `Q: CPU is too slow for my target`
- Use 48 kHz input, one-hop callbacks, and construct `StreamEnhancer` before starting the device.
- Try `dpdfnet2_48khz_hr`, which takes about half the time per hop.
- Measure with `python benchmarks/streaming.py --paced`; see [PERFORMANCE.md](optimized_inference/docs/PERFORMANCE.md).

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
