# CPU streaming performance

`StreamEnhancer` supports the original pretrained ONNX files on both its
portable ONNX Runtime backend and optional OpenVINO CPU backend. Use native
model-rate float32 audio and one-hop callbacks. The
[streaming API guide](../package/README.md#real-time-microphone-enhancement)
contains installation and microphone examples.

## Measurements

Measurements on 2026-10-22 used an Intel Core i5-12600K, Linux x86-64,
Python 3.12.13, NumPy 2.4.6, ONNX Runtime 1.27.0, and OpenVINO 2026.4.1.
Each runtime used one CPU thread. Each process was pinned to CPU 0. No GPU,
quantization, precision reduction, frame skipping, or multi-frame batching
was used.

The input was the first 30 seconds of the official
[Arabic airport SNR 0 evaluation recording](https://huggingface.co/datasets/Ceva-IP/DPDFNet_EvalSet/blob/main/Noisy/arabic_airport_snr0_rt60none_0_mixture_noisy.wav),
resampled to 48 kHz before timing. The recording's SHA-256 was
`aa144f4ada2f02b1fc0c570114ffbc5711d34bc4ea7ad9a2685b6bd7fc60de47`.
The model was `dpdfnet8_48khz_hr`, SHA-256
`56a4367e09c188ce0c4aa02d5e0c8a067f182e500abaa8ca0cff6c1c7b47939b`.
The baseline inference source matches public commit
`9bd9844a227bb6aa57e55588d8d0e961fcff1c46`.

The table gives the median of each statistic across three separate-process
runs, with runtime order rotated between rounds. Each run submitted 3,000
480-sample hops without sleeping and excluded the first 100 calls from
steady-state statistics. Timings cover the entire `process()` call, including
STFT, inference, state handling, and ISTFT.

| Runtime | Median call (ms) | p99 call (ms) | Constructor (ms) | Peak process RSS (MiB) |
| --- | ---: | ---: | ---: | ---: |
| Baseline ONNX | 3.96 | 4.72 | 176 | 335 |
| Optimized ONNX | 3.51 | 4.53 | 120 | 334 |
| OpenVINO CPU, float32 | 2.95 | 3.75 | 515 | 446 |

Median call time fell by 11% with ONNX and 26% with OpenVINO. No measured call
exceeded the 10 ms hop interval. OpenVINO added about 112 MiB of peak process
memory. These RSS measurements include Python, runtime libraries, the complete
input/output arrays, and offline input resampling; they are not model-weight
sizes. Constructor time excludes imports and model downloads. It includes the
new backends' warmup inference.

A separate run submitted the same speech at 10 ms intervals instead of
running flat out. It excluded the first 100 of 3,000 calls:

| Runtime | Median call (ms) | p99 call (ms) | Missed completion deadlines |
| --- | ---: | ---: | ---: |
| Baseline ONNX | 3.87 | 5.88 | 0 |
| Optimized ONNX | 3.65 | 4.73 | 0 |
| OpenVINO CPU, float32 | 3.17 | 4.54 | 0 |

Completion deadlines include scheduler lateness as well as processing time.
These paced results are one run per case, not the three-run aggregate above.
A separate 10-second native-rate synthetic run, without loading the offline
resampler or retaining output, peaked at 126 MiB for the baseline, 122 MiB
for optimized ONNX, and 240 MiB for OpenVINO. Working audio history remains
bounded by one window; OpenVINO's extra runtime memory does not grow with
stream duration.

The optimized ONNX output was bit-identical to the baseline over 1,440,000
returned samples. OpenVINO's maximum absolute difference was `7.15e-7`, with
RMS difference `2.32e-8` (120.6 dB signal-to-error ratio). This checks numerical
preservation, not an improvement in perceptual quality. Automated tests also
compare output and recurrent state for all eight models, multiple chunk sizes,
and transitions through silence.

These are processing-time measurements, **not microphone-to-speaker latency**.
The 20 ms input window, internal model lookahead, audio driver, and device
buffers remain. Returning initial silence sooner would not reduce meaningful
audio latency. Host load and power management affect results; benchmark the
target device rather than assuming this CPU's speedup applies everywhere.

## Reproduce

Install the package and optional runtime using the
[contributor setup](../CONTRIBUTING.md). On Linux, run from the repository root:

```bash
PYTHONPATH=package/src .venv/bin/python benchmarks/streaming.py \
  --onnx-path model_zoo/onnx/dpdfnet8_48khz_hr.onnx \
  --backend onnx --seconds 30 --cpu 0

PYTHONPATH=package/src .venv/bin/python benchmarks/streaming.py \
  --onnx-path model_zoo/onnx/dpdfnet8_48khz_hr.onnx \
  --backend openvino --seconds 30 --cpu 0
```

The default workload is deterministic harmonics, noise, and silence, not
speech. Add `--audio /path/to/recording.wav` for recorded audio. Resampling
happens before timing. Add `--paced` to submit a block every 10 ms; the report
then includes scheduler lateness and missed completion deadlines as well as
inference service times. This simulates arrivals, not a real audio device.
Use `--json report.json` to save the environment and statistics, or
`--save-output enhanced.npy` to compare complete float32 output without PCM
rounding. See `--help` for the complete benchmark interface.

To compare with another revision, run the same script with `PYTHONPATH`
pointing at that revision's `package/src` in a separate checkout. The ONNX
benchmark intentionally uses the original constructor arguments. Run each
case in a fresh process and rotate run order; do not run cases concurrently.

## Implementation constraints

ONNX sessions specialize the exported `batch` and `time` dimensions to one,
allowing shape computations to fold. The package's offline and streaming paths
both submit one frame at a time. Streaming binds persistent spectrum buffers
and two separate state buffers. Input and output state must not alias because
graph nodes may still read the previous state while producing the next one.

OpenVINO reads the ONNX metadata directly, compiles for CPU float32, and uses
one inference request with request-local recurrent state. Snippets fusion is
disabled because it cannot reliably compile the exported GRU-cell graph.
No temporary ONNX Runtime session, converted weights, or persistent compiled
cache is needed. OpenVINO remains optional because of platform support,
compiler startup cost, and extra memory.

Construction warms each backend and then restores the metadata's initial
state. Per-stream input history is bounded by one STFT window, even for large
chunks, and returned output arrays are caller-owned. Plot FFTs in the GUI run
on the UI thread; its callback does not copy unused five-second audio buffers.

Fused-GRU re-exports, channel-last dual-path layouts, and additional inference
threads were tested but did not provide reliable improvements on this CPU.
They are not enabled or shipped as alternate weights.
