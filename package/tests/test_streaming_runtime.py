from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from dpdfnet import StreamEnhancer
from dpdfnet.onnx_backend import OnnxFrameRunner, build_runtime_model


@pytest.fixture
def tiny_model(tmp_path):
    """A real, stateful ONNX graph; no downloaded weights or mocked inference."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Mul", ["spec", "state_in"], ["spec_e"]),
                helper.make_node("Add", ["state_in", "step"], ["state_out"]),
            ],
            "stream-test",
            [
                helper.make_tensor_value_info(
                    "spec", TensorProto.FLOAT, ["batch", "time", 5, 2]
                ),
                helper.make_tensor_value_info("state_in", TensorProto.FLOAT, [2]),
            ],
            [
                helper.make_tensor_value_info(
                    "spec_e", TensorProto.FLOAT, ["batch", "time", 5, 2]
                ),
                helper.make_tensor_value_info("state_out", TensorProto.FLOAT, [2]),
            ],
            [numpy_helper.from_array(np.array([0.01], dtype=np.float32), "step")],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )
    helper.set_model_props(
        model,
        {
            "state_size": "2",
            "erb_norm_state_size": "1",
            "spec_norm_state_size": "1",
            "erb_norm_init": "1",
            "spec_norm_init": "1",
        },
    )
    path = tmp_path / "tiny.onnx"
    onnx.save(model, path)
    return path


def test_bound_onnx_runner_matches_session_and_reset(tiny_model):
    runtime = build_runtime_model(tiny_model)
    runner = OnnxFrameRunner(runtime, 8)
    state = runtime.init_state.copy()
    rng = np.random.default_rng(4)
    for _ in range(100):
        runner.spec[:] = rng.standard_normal(runner.spec.shape)
        expected, state = runtime.session.run(
            None,
            {
                runtime.in_spec_name: runner.spec,
                runtime.in_state_name: state,
            },
        )
        np.testing.assert_array_equal(runner.run(), expected)
        np.testing.assert_array_equal(runner.state, state)
    assert not np.shares_memory(*runner._states)
    runner.reset()
    np.testing.assert_array_equal(runner.state, runtime.init_state)
    assert runtime.session.get_inputs()[0].shape == [1, 1, 5, 2]


@pytest.mark.parametrize("backend", ["onnx", "openvino"])
def test_constructor_warmup_does_not_advance_state(tiny_model, backend):
    if backend == "openvino":
        pytest.importorskip("openvino")
    enhancer = StreamEnhancer(onnx_path=tiny_model, backend=backend)
    assert enhancer.sample_rate == 16000
    assert enhancer.hop_size == 4
    np.testing.assert_array_equal(enhancer._runner.state, [1.0, 1.0])
    audio = np.random.default_rng(5).standard_normal(80).astype(np.float32)
    first = enhancer.process(audio)
    enhancer.reset()
    np.testing.assert_array_equal(enhancer._runner.state, [1.0, 1.0])
    np.testing.assert_allclose(enhancer.process(audio), first, atol=1e-6)


def test_input_storage_is_bounded_and_outputs_are_owned(tiny_model):
    enhancer = StreamEnhancer(onnx_path=tiny_model)
    audio = np.ones(8000, dtype=np.float32)
    output = enhancer.process(audio)
    saved = output.copy()
    assert enhancer._in_buf.size == 8
    assert enhancer._in_len < 8
    assert not np.shares_memory(enhancer._in_buf, audio)
    enhancer.process(audio)
    np.testing.assert_array_equal(output, saved)


def test_flush_is_idempotent_and_preserves_input_sample_rate(tiny_model):
    enhancer = StreamEnhancer(onnx_path=tiny_model)
    enhancer.process(np.ones(12, dtype=np.float32), sample_rate=48000)
    output = enhancer.flush()
    assert output.size == 12
    assert enhancer._input_sr == 48000
    assert enhancer.flush().size == 0
    enhancer.reset()
    assert enhancer.flush().size == 0


def test_invalid_backend_fails_before_model_resolution():
    with pytest.raises(ValueError, match="backend"):
        StreamEnhancer(model="not-a-model", backend="gpu")


def test_openvino_is_optional(tiny_model, monkeypatch):
    from dpdfnet.openvino_backend import OpenVINOFrameRunner

    monkeypatch.setitem(sys.modules, "openvino", None)
    with pytest.raises(ImportError, match=r"dpdfnet\[openvino\]"):
        OpenVINOFrameRunner(tiny_model)
    enhancer = StreamEnhancer(onnx_path=tiny_model)
    assert enhancer.process(np.ones(8, dtype=np.float32)).size == 4


@pytest.mark.parametrize("backend", ["onnx", "openvino"])
def test_state_shape_guard(tiny_model, backend):
    if backend == "openvino":
        pytest.importorskip("openvino")
    import onnx

    model = onnx.load(tiny_model)
    for entry in model.metadata_props:
        if entry.key == "state_size":
            entry.value = "3"
    onnx.save(model, tiny_model)
    with pytest.raises(ValueError, match="Initial state shape mismatch"):
        StreamEnhancer(onnx_path=tiny_model, backend=backend)


@pytest.mark.parametrize("backend", ["onnx", "openvino"])
def test_instances_do_not_share_recurrent_state(tiny_model, backend):
    if backend == "openvino":
        pytest.importorskip("openvino")
    first = StreamEnhancer(onnx_path=tiny_model, backend=backend)
    second = StreamEnhancer(onnx_path=tiny_model, backend=backend)
    first.process(np.ones(40, dtype=np.float32))
    np.testing.assert_array_equal(second._runner.state, [1.0, 1.0])
    first.reset()
    np.testing.assert_array_equal(
        first.process(np.ones(40, dtype=np.float32)),
        second.process(np.ones(40, dtype=np.float32)),
    )


def _legacy_reference(audio, runtime, win_len):
    """Original streaming equations, with session.run and independent state."""
    from dpdfnet.audio import vorbis_window

    hop = win_len // 2
    window = vorbis_window(win_len)
    state = runtime.init_state.copy()
    overlap = np.zeros(win_len, np.float32)
    parts = []
    # Include the single padded frame emitted by flush().
    padded = np.pad(audio, (0, win_len))
    frames = max(0, (len(audio) - win_len) // hop + 1) + 1
    for index in range(frames):
        spec = np.fft.rfft(padded[index * hop : index * hop + win_len] * window)
        spec_ri = np.stack(
            [spec.real.astype(np.float32), spec.imag.astype(np.float32)], axis=-1
        )[None, None]
        enhanced, state = runtime.session.run(
            None,
            {
                runtime.in_spec_name: spec_ri,
                runtime.in_state_name: state,
            },
        )
        ri = enhanced[0, 0]
        overlap += (np.fft.irfft(ri[:, 0] + 1j * ri[:, 1], n=win_len) * window).astype(
            np.float32
        )
        parts.append(overlap[:hop].copy())
        overlap[:-hop] = overlap[hop:]
        overlap[-hop:] = 0
    return np.concatenate(parts), state


MODEL_DIR = Path(
    os.environ.get(
        "DPDFNET_TEST_MODEL_DIR",
        Path(__file__).resolve().parents[2] / "model_zoo" / "onnx",
    )
)


@pytest.mark.parametrize(
    "name",
    [
        "baseline",
        "dpdfnet2",
        "dpdfnet4",
        "dpdfnet8",
        "dpdfnet2_8khz",
        "dpdfnet8_8khz",
        "dpdfnet2_48khz_hr",
        "dpdfnet8_48khz_hr",
    ],
)
@pytest.mark.parametrize("backend", ["onnx", "openvino"])
def test_pretrained_streaming_parity(name, backend):
    path = MODEL_DIR / f"{name}.onnx"
    if not path.is_file():
        pytest.skip(f"Place {name}.onnx in DPDFNET_TEST_MODEL_DIR to run model parity")
    if backend == "openvino" and importlib.util.find_spec("openvino") is None:
        pytest.skip("OpenVINO is not installed")
    enhancer = StreamEnhancer(model=name, onnx_path=path, backend=backend)
    sr = enhancer.sample_rate
    rng = np.random.default_rng(23)
    audio = (rng.standard_normal(sr) * 0.04).astype(np.float32)
    t = np.arange(sr) / sr
    audio += (0.15 * np.sin(2 * np.pi * 230 * t)).astype(np.float32)
    audio[sr // 4 : sr // 2] = 0
    runtime = build_runtime_model(path)
    expected, state = _legacy_reference(audio, runtime, enhancer.hop_size * 2)
    for block_size in [enhancer.hop_size, 137, 1024, sr]:
        enhancer.reset()
        parts = [
            enhancer.process(audio[i : i + block_size])
            for i in range(0, sr, block_size)
        ]
        parts.append(enhancer.flush())
        actual = np.concatenate(parts)
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=1e-4)
        np.testing.assert_allclose(enhancer._runner.state, state, atol=3e-5, rtol=1e-4)
        assert np.isfinite(actual).all()
