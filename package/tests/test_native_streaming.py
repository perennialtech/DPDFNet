"""End-to-end tests for the separately built W7A8 C runtime."""

import os
from pathlib import Path

import numpy as np
import pytest

from dpdfnet import StreamEnhancer
from dpdfnet.native_backend import NativeFrameRunner


_ROOT = Path(__file__).resolve().parents[2] / "optimized_inference"


@pytest.fixture(scope="module")
def native_library():
    build = Path(os.environ.get("DPDFNET_NATIVE_BUILD", _ROOT / "build"))
    library = build / "libdpdfnet.so"
    if not library.is_file():
        pytest.skip("Native C library not built (set DPDFNET_NATIVE_BUILD)")
    return library


@pytest.mark.parametrize("size", [2, 8])
def test_native_spectrum_contract_and_reset(native_library, size):
    model = f"dpdfnet{size}_48khz_hr"
    weights = _ROOT / "models" / model / "weights.f32"
    runner = NativeFrameRunner(model, native_library, weights)
    assert runner.sample_rate == 48000
    assert runner.win_len == 960
    assert runner.hop_size == 480
    assert runner.spec.shape == (1, 1, 481, 2)
    rng = np.random.default_rng(size)
    runner.spec[:] = rng.normal(0, 0.01, runner.spec.shape)
    first = runner.run().copy()
    assert np.isfinite(first).all()
    runner.run()
    runner.reset()
    np.testing.assert_array_equal(runner.run(), first)
    runner.close()


@pytest.mark.parametrize("size", [2, 8])
def test_native_streaming_is_chunk_invariant(native_library, size, monkeypatch):
    model = f"dpdfnet{size}_48khz_hr"
    weights = _ROOT / "models" / model / "weights.f32"

    def unexpected_download(*args, **kwargs):
        raise AssertionError("Native must not resolve or download an ONNX model")

    monkeypatch.setattr("dpdfnet.stream.resolve_model", unexpected_download)
    options = dict(
        model=model, backend="native", native_library=native_library,
        native_weights=weights,
    )
    audio = np.random.default_rng(size + 10).normal(0, 0.05, 9600).astype(np.float32)
    first = StreamEnhancer(**options)
    whole = first.process(audio)
    assert whole.shape == (9120,)
    assert np.isfinite(whole).all()
    first.reset()
    np.testing.assert_array_equal(first.process(audio), whole)

    second = StreamEnhancer(**options)
    chunks = [audio[:325], audio[325:7000], audio[7000:]]
    streamed = np.concatenate([second.process(chunk) for chunk in chunks])
    np.testing.assert_allclose(streamed, whole, rtol=0, atol=1e-6)
    assert not np.shares_memory(whole, second._runner._output)


def test_native_backend_validates_inputs(native_library, tmp_path):
    model = "dpdfnet2_48khz_hr"
    weights = _ROOT / "models" / model / "weights.f32"
    with pytest.raises(ValueError, match="native_library and native_weights"):
        StreamEnhancer(model=model, backend="native")
    with pytest.raises(ValueError, match="48 kHz HR"):
        StreamEnhancer(model="dpdfnet2", backend="native",
                       native_library=native_library, native_weights=weights)
    with pytest.raises(ValueError, match="onnx_path"):
        StreamEnhancer(model=model, backend="native", onnx_path="ignore.onnx",
                       native_library=native_library, native_weights=weights)
    corrupt = tmp_path / "weights.f32"
    data = bytearray(weights.read_bytes())
    data[0] ^= 1
    corrupt.write_bytes(data)
    with pytest.raises(ValueError, match="SHA-256"):
        NativeFrameRunner(model, native_library, corrupt)
