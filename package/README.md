# dpdfnet

CPU speech enhancement from pretrained ONNX models, with offline and stateful
streaming APIs.

## Installation

```bash
pip install dpdfnet
```

## Requirements

- Python `>=3.11`
- OS support for `soundfile` / `libsndfile`

Runtime dependencies are installed automatically:
- `numpy`
- `librosa`
- `soundfile`
- `onnxruntime`
- `filelock`
- `tqdm`

## Supported Models

**8 kHz models**

| Model           | Native sample rate |          Notes          |
| :-------------- | :----------------: | :---------------------: |
| `dpdfnet2_8khz` |        8 kHz       | Low-bandwidth DPDFNet-2 |
| `dpdfnet8_8khz` |        8 kHz       | Low-bandwidth DPDFNet-8 |

**16 kHz models**

| Model      | Native sample rate |             Notes            |
| :--------- | :----------------: | :--------------------------: |
| `baseline` |       16 kHz       |     Fastest 16 kHz model     |
| `dpdfnet2` |       16 kHz       |     Balanced 16 kHz model    |
| `dpdfnet4` |       16 kHz       |  Higher-quality 16 kHz model |
| `dpdfnet8` |       16 kHz       | Highest-quality 16 kHz model |

**48 kHz models**

| Model               | Native sample rate |           Notes           |
| :------------------ | :----------------: | :-----------------------: |
| `dpdfnet2_48khz_hr` |       48 kHz       | High-resolution DPDFNet-2 |
| `dpdfnet8_48khz_hr` |       48 kHz       | High-resolution DPDFNet-8 |

## Supported Audio Formats

The following input formats are supported out of the box (via `soundfile`/libsndfile):

| Format | Extensions |
|--------|-----------|
| WAV | `.wav` |
| FLAC | `.flac` |
| Ogg Vorbis | `.ogg` |
| AIFF | `.aiff`, `.aif` |
| AU/SND | `.au`, `.snd` |

MP3 and other compressed formats require the optional `pydub` dependency and
[ffmpeg](https://ffmpeg.org/) on your PATH:

```bash
pip install 'dpdfnet[mp3]'
# also install ffmpeg, e.g.:
#   Ubuntu/Debian:  sudo apt install ffmpeg
#   macOS:          brew install ffmpeg
#   Windows:        https://ffmpeg.org/download.html
```

Once installed, these additional formats are supported:

| Format | Extensions |
|--------|-----------|
| MP3 | `.mp3` |
| AAC / M4A | `.aac`, `.m4a` |
| WMA | `.wma` |
| Opus | `.opus` |

Output is always written as PCM16 `.wav` regardless of the input format.

## CLI

Show help:

```bash
dpdfnet --help
```

Commands:

1. `dpdfnet models`
- List supported models and local availability.

2. `dpdfnet enhance <input> <output.wav> [--model <name>] [--attn-limit-db DB] [-v|--verbose]`
- Enhance one audio file (any supported format; output is always `.wav`).

3. `dpdfnet enhance-dir <input_dir> <output_dir> [--model <name>] [--workers N] [--attn-limit-db DB] [-v|--verbose]`
- Enhance all supported audio files in a directory (non-recursive).
- Files are processed concurrently; `--workers` sets the thread count (default: CPU count).

4. `dpdfnet download [model] [--force|--refresh] [-q|--quiet | -v|--verbose]`
- Download all models when `model` is omitted, or one model when provided.

CLI examples:

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

## Python API

Top-level exports:
- `dpdfnet.enhance`
- `dpdfnet.enhance_file`
- `dpdfnet.available_models`
- `dpdfnet.download`
- `dpdfnet.StreamEnhancer`

In-memory enhancement:

```python
import soundfile as sf
import dpdfnet

audio, sr = sf.read("noisy.wav")
enhanced = dpdfnet.enhance(audio, sample_rate=sr, model="dpdfnet4", attn_limit_db=12)
sf.write("enhanced.wav", enhanced, sr)
```

Enhance one file:

```python
import dpdfnet

out_path = dpdfnet.enhance_file("noisy.wav", model="dpdfnet2", attn_limit_db=12)
print(out_path)
```

Model listing:

```python
import dpdfnet

for row in dpdfnet.available_models():
    print(row["name"], row["ready"], row["cached"])
```

Download models via API:

```python
import dpdfnet

dpdfnet.download()
dpdfnet.download("dpdfnet4")
dpdfnet.download("dpdfnet2_8khz")
```

### Real-time Microphone Enhancement

Install `sounddevice` (not included in `dpdfnet` dependencies):

```bash
pip install sounddevice
```

`StreamEnhancer` preserves recurrent state across calls and accepts arbitrary
chunk sizes. For low CPU latency with `dpdfnet8_48khz_hr`, use native 48 kHz
audio and 480-sample callbacks. The portable ONNX backend needs no extra runtime.

The optional OpenVINO CPU backend reduces processing time on the tested Intel
CPU, at the cost of more memory and longer construction. Install it with:

```bash
pip install 'dpdfnet[openvino]'
```

Select it with `backend="openvino"`; use `backend="onnx"` (the default) for ONNX
Runtime. Both use float32, one CPU thread, the original model weights and one
frame per inference. OpenVINO requires a platform supported by its
[CPU runtime](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino/system-requirements.html).
It does not use a GPU or NPU and does not create a compiled-model disk cache.

For the two 48 kHz HR models, you can also use the separately built
[native W7A8 runtime](https://github.com/perennialtech/DPDFNet/tree/main/optimized_inference) through this streaming
API. Build the C library first; this option does not load or download ONNX
weights and uses quantized inference, so its output need not match float32.

```python
import dpdfnet

enhancer = dpdfnet.StreamEnhancer(
    model="dpdfnet2_48khz_hr",
    backend="native",
    native_library="optimized_inference/build/libdpdfnet.so",
    native_weights="optimized_inference/models/dpdfnet2_48khz_hr/weights.f32",
)
```

The existing microphone demo accepts `--backend native`, `--native-library`
and `--native-weights`; so does `benchmarks/streaming.py` for comparable
end-to-end callback timings. Native inference requires the CPU and build
environment described in the [native build guide](https://github.com/perennialtech/DPDFNet/blob/main/optimized_inference/docs/BUILD.md).

```python
import sounddevice as sd
import dpdfnet

# Construct and warm the model before starting the audio device.
enhancer = dpdfnet.StreamEnhancer(
    model="dpdfnet8_48khz_hr",
    backend="onnx",  # Or "openvino" with the optional runtime installed.
)
INPUT_SR = enhancer.sample_rate  # 48000
BLOCK_SIZE = enhancer.hop_size   # 480 samples, or 10 ms

def callback(indata, outdata, frames, time, status):
    mono_in = indata[:, 0] if indata.ndim > 1 else indata.ravel()
    enhanced = enhancer.process(mono_in, sample_rate=INPUT_SR)
    n = min(len(enhanced), frames)
    outdata[:n, 0] = enhanced[:n]
    if n < frames:
        outdata[n:] = 0.0   # silence while the first window accumulates

with sd.Stream(
    samplerate=INPUT_SR,
    blocksize=BLOCK_SIZE,
    channels=1,
    dtype="float32",
    callback=callback,
):
    print("Enhancing microphone input - press Ctrl+C to stop")
    try:
        while True:
            sd.sleep(100)
    except KeyboardInterrupt:
        pass

# Optional: drain the final partial window at the end of a recording
tail = enhancer.flush()
```

The first callback returns no samples while the 20 ms window fills. Subsequent
10 ms callbacks return one enhanced hop. Smaller chunks do not shorten the
model hop; larger chunks delay output until the caller submits them. These
buffering times are **not total microphone-to-speaker latency**: inference,
the model's internal lookahead, and audio-device buffering add delay. Neither
backend removes the model's lookahead.

Use the model's native rate whenever the device supports it. Other rates are
accepted and return audio at the input rate, but per-chunk resampling adds work
and can introduce boundary artifacts; it is not a stateful resampling API.
Keep the input rate constant until `reset()`.

Create one enhancer per stream, and do not call its methods concurrently.
Call `reset()` between independent recordings. Returned arrays belong to the
caller and remain valid after later calls. `flush()` pads one final STFT frame
and returns at most one model hop at the input rate; repeated calls return
empty arrays. It does not drain every delayed model frame or guarantee a
recording-length output. Use the offline API when exact file length is needed.

See the [CPU streaming measurements and benchmark](https://github.com/perennialtech/DPDFNet/blob/main/docs/cpu-streaming-performance.md)
for latency, memory, and numerical-parity results. Contributor test instructions
are in [CONTRIBUTING.md](https://github.com/perennialtech/DPDFNet/blob/main/CONTRIBUTING.md).

## Links

- Homepage: https://github.com/ceva-ip/DPDFNet
- Issues: https://github.com/ceva-ip/DPDFNet/issues
