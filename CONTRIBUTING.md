# Contributing

## Set up and test the package

The package runs on the native runtime in `optimized_inference/`. From the
repository root, with GCC, CMake 3.18+ and [uv](https://docs.astral.sh/uv/)
installed (on Windows, an MSYS2 UCRT64 shell; see
[BUILD.md](optimized_inference/docs/BUILD.md)):

```bash
python optimized_inference/tools/build.py   # PGO build and all C contracts
uv venv --python 3.12
uv pip install -e './package[test]'
uv run --no-project python -m pytest package/tests -q
```

An editable install finds the library in `optimized_inference/build`; set
`DPDFNET_NATIVE_LIBRARY` to test another build. No PyTorch, GPU, network
access or audio device is needed. Without a built library, the tests that need
it are skipped and the rest still run.

Most package tests replace the native runtime with a fake frame runner to
check buffering, state reset, flush, attenuation limiting and the CLI. The
tests in `test_native_streaming.py` use the real library. They check chunk
invariance, reset, two-thread versus single-thread equality, weight checksums,
and output against the original FP32 ONNX model: `package/tests/data/onnx_reference.npz`
holds that model's output on a synthetic signal, made once by
`make_onnx_reference.py` with Ceva's `dpdfnet` 0.6 (the ONNX Runtime package,
which installs alongside this one). Native output must agree with it
to at least 20 dB.

To test a wheel as users get it, build and install it, then run the tests
with `DPDFNET_TEST_INSTALLED=1` so they import the installed package rather
than `package/src`:

```bash
uv build --wheel --out-dir dist package     # builds the library too
uv pip install "$(ls dist/*.whl)[test]"
DPDFNET_TEST_INSTALLED=1 uv run --no-project python -m pytest package/tests -q
```

Set `DPDFNET_NATIVE_BUILD_DIR` to a directory with an already built library
to package it instead of building again.

## Benchmark streaming

```bash
uv run --no-project python benchmarks/streaming.py --model dpdfnet8_48khz_hr --paced
```

This times complete `process()` calls, including STFT and ISTFT, and reports
deadline misses for 10 ms hops. See
[PERFORMANCE.md](optimized_inference/docs/PERFORMANCE.md) for the native
measurement contract. Keep benchmark timing out of pass/fail tests: host load
and scheduling can change latency without a code regression. Performance
claims must identify the hardware, compiler, input, warmup, and memory cost.

## Test the GUI without an audio device

```bash
uv pip install sounddevice pyqtgraph PyQt5
QT_QPA_PLATFORM=offscreen uv run --no-project python -m pytest tests/test_real_time_demo.py -q
```

This exercises the 48 kHz, 480-sample callbacks, with and without the helper
thread, using a mock audio device and no GUI event loop. It does not test a
physical device's driver, playback quality, or hardware latency.

## Releasing

Releases are wheels only: upload the `wheel-linux` and `wheel-windows`
artifacts from the `Package` workflow, and never an sdist. `setup.py`
refuses to build one, since it could not compile without
`optimized_inference/`, and on platforms without a wheel pip should report
that nothing matches rather than attempt a build.
