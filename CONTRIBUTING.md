# Contributing

## Set up and test the CPU package

From the repository root, with [uv](https://docs.astral.sh/uv/) installed:

```bash
uv venv --python 3.12
uv pip install -e './package[test,openvino]'
uv run --no-project python -m pytest package/tests -q
```

Omit `openvino` from the extras on unsupported platforms. Tests for that
backend are skipped when it is absent. No PyTorch, TensorFlow, GPU, or audio
device is needed for package tests.

The tests construct small real ONNX graphs to check reusable buffers, state
transitions, warmup, reset, and flush. To also check numerical parity with
every pretrained model, download the models into a directory of your choice:

```bash
export DPDFNET_MODEL_DIR="$PWD/model_zoo/onnx"
export DPDFNET_TEST_MODEL_DIR="$DPDFNET_MODEL_DIR"
uv run --no-project dpdfnet download
uv run --no-project python -m pytest package/tests -q
```

Model tests skip files that are absent; they never download implicitly.
Both backends must preserve the original float32 streaming equations across
chunk sizes, silence, state reset, and all model families. Performance claims
must identify the hardware, runtime versions, input, warmup, and memory cost.

## Benchmark streaming

See [CPU streaming performance](docs/cpu-streaming-performance.md) for commands
and the measurement contract. Keep benchmark timing out of pass/fail tests:
host load and scheduling can change latency without a code regression.

## Test the GUI without an audio device

With the HR ONNX model downloaded as above:

```bash
uv pip install sounddevice pyqtgraph PyQt5
QT_QPA_PLATFORM=offscreen uv run --no-project python -m pytest tests/test_real_time_demo.py -q
```

This exercises the native 48 kHz, 480-sample callbacks on both backends with a
mock audio device and no GUI event loop. It does not test a physical device's
driver, playback quality, or hardware latency.
