# dpdfnet-native

Native CPU speech enhancement with DPDFNet's 48 kHz high-resolution models,
with offline and stateful streaming APIs.

This is an independent fork of [Ceva-IP/DPDFNet](https://github.com/ceva-ip/DPDFNet),
maintained at [perennialtech/DPDFNet](https://github.com/perennialtech/DPDFNet).
It is not Ceva's `dpdfnet` package, which runs all of DPDFNet's 8, 16 and
48 kHz models on ONNX Runtime. The two install side by side: this one imports
as `dpdfnet_native` and its command is `dpdfnet-native`.

Inference runs on the [optimized C runtime](https://github.com/perennialtech/DPDFNet/tree/main/optimized_inference):
W7A8 integer kernels with FP32 recurrent state and hand-written AVX2/FMA
assembly. The wheel bundles the library and both models' weights, so nothing
is downloaded and no ONNX Runtime is needed.

## Installation

```bash
pip install dpdfnet-native
```

It is released as wheels only, for Linux (glibc 2.27+) and Windows on
x86-64, Python 3.11 and later; elsewhere pip finds no matching distribution.
To build a wheel from a repository checkout, with GCC and CMake 3.18+ (MSYS2
UCRT64 GCC on Windows):

```bash
pip install ./package
```

This runs the native runtime's profile-guided release build and all of its
contract tests before packaging.

## Requirements

- Python `>=3.11`
- Linux, or Windows 10 version 1607 or later
- An x86-64 CPU with AVX2 and FMA (Intel Haswell, AMD Excavator or later)
- OS support for `soundfile` / `libsndfile`

On other CPUs the first enhancer or `enhance()` call raises an error saying so.
AVX-VNNI kernels are selected automatically on CPUs that have them.

Runtime dependencies are installed automatically:
- `numpy`
- `librosa`
- `soundfile`
- `tqdm`

## Supported Models

| Model               | Sample rate |           Notes            |
| :------------------ | :---------: | :------------------------: |
| `dpdfnet2_48khz_hr` |    48 kHz   | DPDFNet-2, the default     |
| `dpdfnet8_48khz_hr` |    48 kHz   | DPDFNet-8, higher quality, about twice the time per hop |

Input at other sample rates is resampled to 48 kHz and back.

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
pip install 'dpdfnet-native[mp3]'
# also install ffmpeg, e.g.:
#   Ubuntu/Debian:  sudo apt install ffmpeg
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
dpdfnet-native --help
```

Commands:

1. `dpdfnet-native models`
- List the models and where the native library was found.

2. `dpdfnet-native enhance <input> <output.wav> [--model <name>] [--attn-limit-db DB]`
- Enhance one audio file (any supported format; output is always `.wav`).

3. `dpdfnet-native enhance-dir <input_dir> <output_dir> [--model <name>] [--workers N] [--attn-limit-db DB]`
- Enhance all supported audio files in a directory (non-recursive).
- Files are processed concurrently; `--workers` sets the thread count (default: CPU count).
  With more than one worker, each runs single-threaded inference.

CLI examples:

```bash
# Enhance one file
dpdfnet-native enhance noisy.wav enhanced.wav --model dpdfnet8_48khz_hr --attn-limit-db 12

# Enhance a directory (uses all CPU cores by default)
dpdfnet-native enhance-dir ./noisy_wavs ./enhanced_wavs --attn-limit-db 12

# Enhance a directory with a fixed worker count
dpdfnet-native enhance-dir ./noisy_wavs ./enhanced_wavs --workers 4 --attn-limit-db 12
```

## Python API

Top-level exports:
- `dpdfnet_native.enhance`
- `dpdfnet_native.enhance_file`
- `dpdfnet_native.available_models`
- `dpdfnet_native.StreamEnhancer`

In-memory enhancement:

```python
import soundfile as sf
import dpdfnet_native

audio, sr = sf.read("noisy.wav")
enhanced = dpdfnet_native.enhance(audio, sample_rate=sr, model="dpdfnet8_48khz_hr", attn_limit_db=12)
sf.write("enhanced.wav", enhanced, sr)
```

Enhance one file:

```python
import dpdfnet_native

out_path = dpdfnet_native.enhance_file("noisy.wav", attn_limit_db=12)
print(out_path)
```

Model listing:

```python
import dpdfnet_native

for row in dpdfnet_native.available_models():
    print(row["name"], row["ready"], row["weights_path"])
```

`enhance()` and `enhance_file()` take `single_thread=True` as well; see below.

### Real-time Microphone Enhancement

Install `sounddevice` (not included in `dpdfnet-native`'s dependencies):

```bash
pip install sounddevice
```

`StreamEnhancer` preserves recurrent state across calls and accepts arbitrary
chunk sizes. For the lowest latency, use 48 kHz audio and 480-sample
callbacks.

Each enhancer owns one native context with one helper thread that runs the
model's two branches concurrently, so each hop finishes 25–45% sooner on two
free cores, with identical output and more total CPU time. The helper takes
the CPU affinity and scheduling priority of the thread that constructs the
enhancer, not of the audio callback thread: if the callback runs at raised
priority, the helper does not. Pass `single_thread=True` to run on the calling
thread only, for example when running many streams at once. Call `close()` to
release the context and its thread before garbage collection would.

```python
import sounddevice as sd
import dpdfnet_native

# Construct and warm the model before starting the audio device.
enhancer = dpdfnet_native.StreamEnhancer(model="dpdfnet8_48khz_hr")
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
the model's internal lookahead, and audio-device buffering add delay. The
runtime does not remove the model's lookahead.

Use 48 kHz whenever the device supports it. Other rates are accepted and
return audio at the input rate, but per-chunk resampling adds work and can
introduce boundary artifacts; it is not a stateful resampling API. Keep the
input rate constant until `reset()`.

Create one enhancer per stream, and do not call its methods concurrently.
Call `reset()` between independent recordings. Returned arrays belong to the
caller and remain valid after later calls. `flush()` pads one final STFT frame
and returns at most one model hop at the input rate; repeated calls return
empty arrays. It does not drain every delayed model frame or guarantee a
recording-length output. Use the offline API when exact file length is needed.

The output is quantized inference, not the FP32 model: on the package's test
signal it agrees with the original FP32 ONNX model to 27–35 dB. See
[performance and output quality](https://github.com/perennialtech/DPDFNet/blob/main/optimized_inference/docs/PERFORMANCE.md)
for latency and quality measurements. Contributor test instructions are in
[CONTRIBUTING.md](https://github.com/perennialtech/DPDFNet/blob/main/CONTRIBUTING.md).

### Library location

Wheels load the library bundled in `dpdfnet_native/_native`. An editable install
from a checkout uses `optimized_inference/build`. Set `DPDFNET_NATIVE_LIBRARY`
to the path of a `libdpdfnet.so` or `libdpdfnet.dll` to use another build of
the same version.

## Links

- Homepage: https://github.com/perennialtech/DPDFNet
- Issues: https://github.com/perennialtech/DPDFNet/issues
- Upstream (models, paper, Ceva's `dpdfnet`): https://github.com/ceva-ip/DPDFNet
