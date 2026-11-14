"""End-to-end tests against the real W7A8 C runtime.

They use the bundled library of an installed wheel, ``DPDFNET_NATIVE_LIBRARY``,
or ``optimized_inference/build`` in a source checkout, and skip when none
exists, except when testing an installed wheel, which must bundle it.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from dpdfnet import StreamEnhancer, enhance
from dpdfnet.models import find_native_library, supported_models, weights_path
from dpdfnet.native_backend import NativeFrameRunner
from synthetic import SAMPLE_RATE, agreement_db, noisy_vowel


_REFERENCE = Path(__file__).with_name("data") / "onnx_reference.npz"
# Measured on Linux and Windows: 29-32 dB streaming, 27-35 dB offline. The
# remaining difference is the runtime's W7A8 quantization of the FP32 model.
# Injected faults (state reset every frame or frozen, real/imaginary swapped,
# conjugated spectra, input passed through) all scored 14 dB or less.
_MIN_ONNX_AGREEMENT_DB = 20.0


@pytest.fixture(scope="module", autouse=True)
def native_library():
    library = find_native_library()
    if library is None:
        if os.environ.get("DPDFNET_TEST_INSTALLED"):
            pytest.fail("The installed package has no native library")
        pytest.skip("Native C library not built (set DPDFNET_NATIVE_LIBRARY)")
    return library


@pytest.mark.parametrize("model", supported_models())
def test_native_spectrum_contract_and_reset(model):
    runner = NativeFrameRunner(model)
    assert runner.sample_rate == 48000
    assert runner.win_len == 960
    assert runner.hop_size == 480
    assert runner.spec.shape == (1, 1, 481, 2)
    rng = np.random.default_rng(int(model[7]))
    runner.spec[:] = rng.normal(0, 0.01, runner.spec.shape)
    first = runner.run().copy()
    assert np.isfinite(first).all()
    runner.run()
    runner.reset()
    np.testing.assert_array_equal(runner.run(), first)
    runner.close()


@pytest.mark.parametrize("model", supported_models())
def test_native_streaming_is_chunk_invariant(model):
    audio = np.random.default_rng(int(model[7]) + 10).normal(0, 0.05, 9600).astype(np.float32)
    first = StreamEnhancer(model)
    whole = first.process(audio)
    assert whole.shape == (9120,)
    assert np.isfinite(whole).all()
    first.reset()
    np.testing.assert_array_equal(first.process(audio), whole)

    second = StreamEnhancer(model)
    chunks = [audio[:325], audio[325:7000], audio[7000:]]
    streamed = np.concatenate([second.process(chunk) for chunk in chunks])
    np.testing.assert_allclose(streamed, whole, rtol=0, atol=1e-6)
    assert not np.shares_memory(whole, second._runner._output)
    assert second.flush().shape == (480,)
    assert second.flush().size == 0
    first.close()
    second.close()


def test_native_instances_do_not_share_state():
    audio = np.random.default_rng(30).normal(0, 0.05, 4800).astype(np.float32)
    alone = StreamEnhancer().process(audio)
    a, b = StreamEnhancer(), StreamEnhancer()
    b.process(np.random.default_rng(31).normal(0, 0.05, 4800).astype(np.float32))
    np.testing.assert_array_equal(a.process(audio), alone)


def test_native_backend_validates_inputs(tmp_path):
    model = "dpdfnet2_48khz_hr"
    with pytest.raises(ValueError, match="Unsupported model"):
        StreamEnhancer("dpdfnet2")
    corrupt = tmp_path / "weights.f32"
    data = bytearray(weights_path(model).read_bytes())
    data[0] ^= 1
    corrupt.write_bytes(data)
    with pytest.raises(ValueError, match="SHA-256"):
        NativeFrameRunner(model, weights=corrupt)
    # One model's weights do not pass for the other.
    with pytest.raises(ValueError, match="SHA-256"):
        NativeFrameRunner(model, weights=weights_path("dpdfnet8_48khz_hr"))


@pytest.mark.parametrize("model", supported_models())
def test_native_default_matches_single_thread(model):
    audio = np.random.default_rng(int(model[7]) + 20).normal(0, 0.05, 9600).astype(np.float32)
    single = StreamEnhancer(model, single_thread=True)
    worker = StreamEnhancer(model)
    expected = single.process(audio)
    np.testing.assert_array_equal(worker.process(audio), expected)
    worker.reset()
    np.testing.assert_array_equal(worker.process(audio), expected)
    np.testing.assert_array_equal(worker.flush(), single.flush())
    np.testing.assert_array_equal(
        enhance(audio, SAMPLE_RATE, model=model),
        enhance(audio, SAMPLE_RATE, model=model, single_thread=True),
    )
    worker.close()
    single.close()


@pytest.mark.parametrize("model", supported_models())
def test_native_output_tracks_the_fp32_onnx_model(model):
    """The independent oracle: output of the original FP32 ONNX export.

    See data/make_onnx_reference.py for how the fixture was made.
    """
    reference = np.load(_REFERENCE)
    noisy, _clean = noisy_vowel()

    stream = StreamEnhancer(model)
    streamed = np.concatenate([stream.process(noisy), stream.flush()])
    expected = reference[f"{model}_stream"].astype(np.float32)
    assert streamed.shape == expected.shape
    assert agreement_db(expected, streamed) >= _MIN_ONNX_AGREEMENT_DB

    offline = enhance(noisy, SAMPLE_RATE, model=model)
    expected = reference[f"{model}_offline"].astype(np.float32)
    assert offline.shape == expected.shape
    assert agreement_db(expected, offline) >= _MIN_ONNX_AGREEMENT_DB


def test_offline_enhance_resamples_and_keeps_length():
    noisy, _clean = noisy_vowel(seconds=0.5)
    at_16k = noisy[::3].copy()
    enhanced = enhance(at_16k, 16000)
    assert enhanced.shape == at_16k.shape
    assert enhanced.dtype == np.float32
    assert np.isfinite(enhanced).all()


@pytest.mark.skipif(
    not os.environ.get("DPDFNET_TEST_INSTALLED"),
    reason="Checks an installed wheel (set DPDFNET_TEST_INSTALLED)",
)
def test_installed_wheel_uses_its_bundled_files(monkeypatch):
    import dpdfnet

    monkeypatch.delenv("DPDFNET_NATIVE_LIBRARY", raising=False)
    bundled = Path(dpdfnet.__file__).resolve().parent / "_native"
    assert find_native_library().parent == bundled
    for model in supported_models():
        assert weights_path(model) == bundled / f"{model}.f32"
