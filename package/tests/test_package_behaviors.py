from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


class _FakeRunner:
    """A frame runner with the native interface and no native library.

    It returns zeros, the input spectrum (``passthrough``), or successive
    frames of ``outputs``.
    """

    def __init__(self, win_len=8, sample_rate=16000, passthrough=False, outputs=None):
        self.win_len = win_len
        self.hop_size = win_len // 2
        self.sample_rate = sample_rate
        self.spec = np.zeros((1, 1, win_len // 2 + 1, 2), dtype=np.float32)
        self._output = np.empty_like(self.spec)
        self.passthrough = passthrough
        self.outputs = outputs
        self.calls = 0
        self.resets = 0
        self.closed = False

    def reset(self):
        self.resets += 1

    def run(self):
        if self.outputs is not None:
            self._output[...] = self.outputs[self.calls]
        elif self.passthrough:
            self._output[...] = self.spec
        else:
            self._output[...] = 0.0
        self.calls += 1
        return self._output

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def _use_fake_runner(monkeypatch, **kwargs):
    """Make every NativeFrameRunner a _FakeRunner; returns the ones created."""
    from dpdfnet_native import native_backend

    created = []

    def _factory(model, *_args, single_thread=False, **_kw):
        runner = _FakeRunner(**kwargs)
        runner.model = model
        runner.single_thread = single_thread
        created.append(runner)
        return runner

    monkeypatch.setattr(native_backend, "NativeFrameRunner", _factory)
    return created


def test_import_surface() -> None:
    import dpdfnet_native

    assert hasattr(dpdfnet_native, "enhance")
    assert hasattr(dpdfnet_native, "enhance_file")
    assert hasattr(dpdfnet_native, "StreamEnhancer")
    assert hasattr(dpdfnet_native, "available_models")
    assert not hasattr(dpdfnet_native, "download")


def _patch_offline_audio(monkeypatch, spec_r, postprocess=None):
    import dpdfnet_native.audio as audio_mod

    monkeypatch.setattr(audio_mod, "to_mono", lambda x: np.asarray(x, dtype=np.float32))
    monkeypatch.setattr(audio_mod, "ensure_sample_rate", lambda x, _a, _b: np.asarray(x, dtype=np.float32))
    monkeypatch.setattr(audio_mod, "make_stft_config", lambda _win: type("Cfg", (), {"win_len": 4})())
    monkeypatch.setattr(audio_mod, "preprocess_waveform", lambda _w, _cfg: spec_r.copy())
    monkeypatch.setattr(
        audio_mod, "postprocess_spec",
        postprocess or (lambda _spec, _cfg: np.zeros((8,), dtype=np.float32)),
    )
    monkeypatch.setattr(audio_mod, "fit_length", lambda x, _n: np.asarray(x, dtype=np.float32))


def test_enhance_progress_callback_reports_frame_progress(monkeypatch) -> None:
    from dpdfnet_native import api

    created = _use_fake_runner(monkeypatch, win_len=4)
    _patch_offline_audio(monkeypatch, np.zeros((1, 4, 3, 2), dtype=np.float32))

    updates = []
    api.enhance(
        audio=np.zeros((8,), dtype=np.float32),
        sample_rate=16000,
        progress_callback=lambda done, total: updates.append((done, total)),
    )

    assert updates[0] == (0, 4)
    assert updates[-1] == (4, 4)
    assert len(updates) == 5
    assert created[0].model == "dpdfnet2_48khz_hr"
    assert created[0].closed


def test_enhance_applies_attn_limit_before_postprocess(monkeypatch) -> None:
    from dpdfnet_native import api

    spec_r = np.arange(1, 1 + (1 * 4 * 3 * 2), dtype=np.float32).reshape(1, 4, 3, 2)
    outputs = np.full((4, 1, 1, 3, 2), 77.0, dtype=np.float32)
    captured: dict[str, np.ndarray] = {}

    def _capture_postprocess(spec, _cfg):
        captured["spec"] = np.asarray(spec, dtype=np.float32).copy()
        return np.zeros((8,), dtype=np.float32)

    _use_fake_runner(monkeypatch, win_len=4, outputs=outputs)
    _patch_offline_audio(monkeypatch, spec_r, _capture_postprocess)

    api.enhance(
        audio=np.zeros((8,), dtype=np.float32),
        sample_rate=16000,
        attn_limit_db=0.0,
    )

    expected = np.zeros_like(spec_r)
    expected[:, 4:, :, :] = spec_r[:, :-4, :, :]
    np.testing.assert_array_equal(captured["spec"], expected)


def test_enhance_feeds_each_frame_and_resets_reused_runner(monkeypatch) -> None:
    from dpdfnet_native import api

    spec_r = np.random.default_rng(3).normal(size=(1, 5, 3, 2)).astype(np.float32)
    captured = []
    _patch_offline_audio(
        monkeypatch, spec_r, lambda spec, _cfg: captured.append(spec.copy()) or np.zeros(8, np.float32)
    )
    runner = _FakeRunner(win_len=4, passthrough=True)

    for _ in range(2):
        api._enhance_with_runner(np.zeros(8, np.float32), 16000, runner=runner)

    assert runner.resets == 2
    assert runner.calls == 10
    for spec in captured:
        np.testing.assert_array_equal(spec, spec_r)


def test_enhance_passes_single_thread_to_the_runner(monkeypatch) -> None:
    from dpdfnet_native import api

    created = _use_fake_runner(monkeypatch, win_len=4)
    _patch_offline_audio(monkeypatch, np.zeros((1, 2, 3, 2), dtype=np.float32))
    api.enhance(np.zeros(8, np.float32), 16000, model="dpdfnet8_48khz_hr", single_thread=True)
    assert created[0].model == "dpdfnet8_48khz_hr"
    assert created[0].single_thread


# ---------------------------------------------------------------------------
# StreamEnhancer tests
# ---------------------------------------------------------------------------

def _make_fake_stream_enhancer(monkeypatch, win_len: int = 8, model_sr: int = 16000):
    """Return a StreamEnhancer whose model outputs zeros."""
    _use_fake_runner(monkeypatch, win_len=win_len, sample_rate=model_sr)
    from dpdfnet_native import StreamEnhancer

    return StreamEnhancer()


def test_stream_enhancer_buffers_small_chunks(monkeypatch) -> None:
    """Small chunks are buffered; output appears only after win_len samples arrive."""
    e = _make_fake_stream_enhancer(monkeypatch, win_len=8)  # hop_size=4
    # 3 samples < win_len=8 → nothing to process yet
    out = e.process(np.zeros(3, dtype=np.float32), sample_rate=16000)
    assert len(out) == 0
    # 5 more → total 8 = win_len → one frame committed → hop_size=4 output samples
    out = e.process(np.zeros(5, dtype=np.float32), sample_rate=16000)
    assert len(out) == 4


def test_stream_enhancer_misaligned_block_size(monkeypatch) -> None:
    """Chunk sizes not aligned to hop_size must not drop or duplicate samples.

    Simulates sounddevice with BLOCK_SIZE=512 at 48 kHz against a 16 kHz model
    by feeding 171-sample chunks (≈ 512 * 16000 / 48000) at model SR directly.
    """
    WIN = 320
    HOP = 160
    e = _make_fake_stream_enhancer(monkeypatch, win_len=WIN)

    total_input = int(16000 * 1.0)  # 1 second at model SR
    CHUNK = 171  # not a multiple of HOP

    outputs = []
    fed = 0
    while fed < total_input:
        n = min(CHUNK, total_input - fed)
        out = e.process(np.zeros(n, dtype=np.float32), sample_rate=16000)
        outputs.append(out)
        fed += n

    total_output = sum(len(o) for o in outputs)
    expected_frames = max(0, (total_input - WIN) // HOP + 1)
    assert total_output == expected_frames * HOP
    for o in outputs:
        assert o.dtype == np.float32


def test_stream_enhancer_reset_clears_state(monkeypatch) -> None:
    """After reset(), the buffer is empty and a full win_len is needed again."""
    e = _make_fake_stream_enhancer(monkeypatch, win_len=8)
    e.process(np.zeros(5, dtype=np.float32), sample_rate=16000)  # partial buffer
    e.reset()
    out = e.process(np.zeros(5, dtype=np.float32), sample_rate=16000)
    assert len(out) == 0


def test_stream_enhancer_flush_drains_remainder(monkeypatch) -> None:
    """flush() zero-pads the last partial window and returns enhanced samples."""
    e = _make_fake_stream_enhancer(monkeypatch, win_len=8)
    e.process(np.zeros(5, dtype=np.float32), sample_rate=16000)  # 5 in buf, no output
    out = e.flush()
    assert len(out) > 0
    assert out.dtype == np.float32


def test_stream_enhancer_flush_on_empty_buffer_returns_empty(monkeypatch) -> None:
    """flush() on a fresh (or fully-drained) instance returns an empty array."""
    e = _make_fake_stream_enhancer(monkeypatch, win_len=8)
    out = e.flush()
    assert len(out) == 0
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# StreamEnhancer correctness: passthrough reconstruction & block-size invariance
# ---------------------------------------------------------------------------

def _make_passthrough_stream_enhancer(monkeypatch, win_len: int = 320, model_sr: int = 16000):
    """StreamEnhancer whose model returns the input spectrum unchanged.

    Unlike ``_make_fake_stream_enhancer`` which outputs zeros, this lets the
    real causal STFT / overlap-add pipeline run, making it suitable for
    testing signal reconstruction properties.
    """
    _use_fake_runner(monkeypatch, win_len=win_len, sample_rate=model_sr, passthrough=True)
    from dpdfnet_native import StreamEnhancer

    return StreamEnhancer()


def _stream_enhance(enhancer, signal: np.ndarray, block_size: int, sr: int = 16000):
    """Feed *signal* through *enhancer* in chunks of *block_size* and return concatenated output."""
    parts: list[np.ndarray] = []
    for start in range(0, len(signal), block_size):
        out = enhancer.process(signal[start : start + block_size], sample_rate=sr)
        parts.append(out)
    parts.append(enhancer.flush())
    return np.concatenate(parts)


def test_stream_passthrough_reconstructs_signal(monkeypatch) -> None:
    """With a passthrough model, the causal STFT → ISTFT pipeline perfectly
    reconstructs the input (after the one-hop COLA ramp-up)."""
    WIN = 320
    HOP = WIN // 2

    e = _make_passthrough_stream_enhancer(monkeypatch, win_len=WIN)

    np.random.seed(123)
    signal = np.random.randn(8000).astype(np.float32) * 0.5

    out = e.process(signal, sample_rate=16000)
    # Skip the first HOP samples: only one window contributes there (half COLA).
    # From sample HOP onward, two overlapping windows sum to 1.
    np.testing.assert_allclose(
        out[HOP:], signal[HOP : len(out)], atol=1e-5,
        err_msg="Passthrough streaming did not reconstruct input after initial latency",
    )


@pytest.mark.parametrize("block_size", [7, 64, 160, 171, 320, 512, 1000])
def test_stream_block_size_matches_reference(monkeypatch, block_size: int) -> None:
    """Streaming output must be identical regardless of how the input is chunked.

    Uses block_size=1 as the reference (one sample at a time), then parametrises
    over a range of realistic and adversarial block sizes.
    """
    WIN = 320

    np.random.seed(42)
    signal = np.random.randn(4000).astype(np.float32) * 0.5

    # Reference: one-sample-at-a-time (guarantees every buffering edge case)
    ref_enhancer = _make_passthrough_stream_enhancer(monkeypatch, win_len=WIN)
    ref = _stream_enhance(ref_enhancer, signal, block_size=1)

    # Test block size
    enhancer = _make_passthrough_stream_enhancer(monkeypatch, win_len=WIN)
    result = _stream_enhance(enhancer, signal, block_size=block_size)

    assert len(result) == len(ref), (
        f"block_size={block_size}: length {len(result)} != ref length {len(ref)}"
    )
    np.testing.assert_allclose(
        result, ref, atol=1e-5,
        err_msg=f"block_size={block_size} differs from sample-by-sample reference",
    )


def test_stream_and_offline_both_reconstruct_with_passthrough(monkeypatch) -> None:
    """With a passthrough model, both enhance() and StreamEnhancer approximately
    reconstruct the input signal (each in its own alignment convention).

    The offline path uses center=True STFT; the streaming path uses center=False.
    They are not bit-identical, but both should faithfully represent the input.
    """
    WIN = 320
    HOP = WIN // 2
    MODEL_SR = 16000

    from dpdfnet_native import api

    _use_fake_runner(monkeypatch, win_len=WIN, sample_rate=MODEL_SR, passthrough=True)

    np.random.seed(99)
    signal = np.random.randn(8000).astype(np.float32) * 0.5

    # Offline
    offline_out = api.enhance(audio=signal, sample_rate=MODEL_SR)

    # Streaming (single large chunk)
    from dpdfnet_native import StreamEnhancer

    se = StreamEnhancer()
    stream_out = np.concatenate([
        se.process(signal, sample_rate=MODEL_SR),
        se.flush(),
    ])

    # Both should produce output of expected length
    assert len(offline_out) == len(signal)
    assert len(stream_out) > 0

    # Both should contain non-trivial (non-zero) samples
    assert np.any(offline_out != 0), "Offline output is all zeros"
    assert np.any(stream_out != 0), "Stream output is all zeros"

    # Streaming path: perfect reconstruction after first HOP (verified separately)
    np.testing.assert_allclose(
        stream_out[HOP : len(stream_out)],
        signal[HOP : len(stream_out)],
        atol=1e-5,
        err_msg="Streaming path did not reconstruct input",
    )

    # Offline path: postprocess_spec strips the first 2*win_len samples from the
    # ISTFT output, so offline_out[k] corresponds to signal[k + 2*WIN].
    shift = 2 * WIN
    valid_len = len(signal) - shift  # samples where real signal exists
    np.testing.assert_allclose(
        offline_out[:valid_len],
        signal[shift : shift + valid_len],
        atol=1e-4,
        err_msg="Offline passthrough did not reconstruct the shifted input",
    )


def test_stream_enhancer_sample_rate_mismatch_raises(monkeypatch) -> None:
    """process() raises ValueError if sample_rate changes between calls."""
    e = _make_fake_stream_enhancer(monkeypatch, win_len=8)
    e.process(np.zeros(3, dtype=np.float32), sample_rate=16000)
    with pytest.raises(ValueError, match="Sample rate changed"):
        e.process(np.zeros(3, dtype=np.float32), sample_rate=8000)


def test_stream_enhancer_empty_chunk_returns_empty(monkeypatch) -> None:
    """process() with an empty array returns an empty float32 array."""
    e = _make_fake_stream_enhancer(monkeypatch, win_len=8)
    out = e.process(np.zeros(0, dtype=np.float32), sample_rate=16000)
    assert len(out) == 0
    assert out.dtype == np.float32


def test_stream_enhancer_stereo_converted_to_mono(monkeypatch) -> None:
    """Stereo (2D) chunks are averaged to mono before processing."""
    e = _make_fake_stream_enhancer(monkeypatch, win_len=8)
    stereo = np.zeros((10, 2), dtype=np.float32)
    # Should not raise; stereo is silently averaged to mono
    out = e.process(stereo, sample_rate=16000)
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# audio.py unit tests
# ---------------------------------------------------------------------------

def test_to_mono_passthrough_for_1d() -> None:
    from dpdfnet_native.audio import to_mono
    x = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    result = to_mono(x)
    np.testing.assert_array_equal(result, x)
    assert result.dtype == np.float32


def test_to_mono_averages_stereo_channels() -> None:
    from dpdfnet_native.audio import to_mono
    # shape (samples, channels) — axis-1 is channels, axis-0 is time
    left = np.array([1.0, 0.0], dtype=np.float32)
    right = np.array([0.0, 1.0], dtype=np.float32)
    stereo = np.stack([left, right], axis=1)  # (2, 2)
    result = to_mono(stereo)
    np.testing.assert_allclose(result, [0.5, 0.5], atol=1e-6)
    assert result.dtype == np.float32


def test_to_mono_invalid_shape_raises() -> None:
    from dpdfnet_native.audio import to_mono
    bad = np.zeros((2, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="Expected mono/stereo"):
        to_mono(bad)


def test_fit_length_truncates() -> None:
    from dpdfnet_native.audio import fit_length
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    result = fit_length(x, 2)
    assert len(result) == 2
    np.testing.assert_array_equal(result, [1.0, 2.0])


def test_fit_length_pads_with_zeros() -> None:
    from dpdfnet_native.audio import fit_length
    x = np.array([1.0, 2.0], dtype=np.float32)
    result = fit_length(x, 5)
    assert len(result) == 5
    np.testing.assert_array_equal(result, [1.0, 2.0, 0.0, 0.0, 0.0])


def test_fit_length_exact_unchanged() -> None:
    from dpdfnet_native.audio import fit_length
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = fit_length(x, 3)
    np.testing.assert_array_equal(result, x)


def test_pcm16_safe_clips_and_converts() -> None:
    from dpdfnet_native.audio import pcm16_safe
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    result = pcm16_safe(x)
    assert result.dtype == np.int16
    assert result[0] == int(-1.0 * 32767)
    assert result[1] == int(-1.0 * 32767)
    assert result[2] == 0
    assert result[3] == 32767
    assert result[4] == 32767


def test_vorbis_window_dtype_and_length() -> None:
    from dpdfnet_native.audio import vorbis_window
    win = vorbis_window(16)
    assert win.dtype == np.float32
    assert len(win) == 16


def test_vorbis_window_cola_at_50pct_overlap() -> None:
    """w[n]^2 + w[n + hop]^2 == 1 for all n in [0, hop)."""
    from dpdfnet_native.audio import vorbis_window
    win_len = 32
    hop = win_len // 2
    w = vorbis_window(win_len).astype(np.float64)
    cola = w[:hop] ** 2 + w[hop:] ** 2
    np.testing.assert_allclose(cola, 1.0, atol=1e-6)


def test_make_stft_config_hop_is_half_win() -> None:
    from dpdfnet_native.audio import make_stft_config
    cfg = make_stft_config(64)
    assert cfg.hop_size == 32
    assert cfg.win_len == 64
    assert len(cfg.window) == 64


def test_ensure_sample_rate_identity_when_same_sr() -> None:
    from dpdfnet_native.audio import ensure_sample_rate
    x = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    result = ensure_sample_rate(x, 16000, 16000)
    np.testing.assert_array_equal(result, x)
    assert result.dtype == np.float32


def test_apply_attn_limit_none_returns_enhanced() -> None:
    from dpdfnet_native.audio import apply_attn_limit

    spec_noisy = np.ones((1, 4, 2, 2), dtype=np.float32)
    spec_enh = np.full((1, 4, 2, 2), 3.0, dtype=np.float32)

    result = apply_attn_limit(spec_noisy, spec_enh, None)

    np.testing.assert_array_equal(result, spec_enh)
    assert result.dtype == np.float32


def test_apply_attn_limit_zero_db_returns_shifted_noisy_reference() -> None:
    from dpdfnet_native.audio import apply_attn_limit

    spec_noisy = np.arange(1, 1 + (1 * 7 * 1 * 2), dtype=np.float32).reshape(1, 7, 1, 2)
    spec_enh = np.full_like(spec_noisy, 99.0)

    result = apply_attn_limit(spec_noisy, spec_enh, 0.0)

    expected = np.zeros_like(spec_noisy)
    expected[:, 4:, :, :] = spec_noisy[:, :-4, :, :]
    np.testing.assert_array_equal(result, expected)


def test_apply_attn_limit_finite_db_blends_shifted_noisy_and_enhanced() -> None:
    from dpdfnet_native.audio import apply_attn_limit

    spec_noisy = np.arange(1, 1 + (1 * 7 * 1 * 2), dtype=np.float32).reshape(1, 7, 1, 2)
    spec_enh = np.full_like(spec_noisy, 8.0)
    attn_limit_db = 6.0

    result = apply_attn_limit(spec_noisy, spec_enh, attn_limit_db)

    alpha = 10.0 ** (-attn_limit_db / 20.0)
    aligned = np.zeros_like(spec_noisy)
    aligned[:, 4:, :, :] = spec_noisy[:, :-4, :, :]
    expected = alpha * aligned + (1.0 - alpha) * spec_enh
    np.testing.assert_allclose(result, expected.astype(np.float32), atol=1e-6)


def test_apply_attn_limit_inf_returns_enhanced() -> None:
    from dpdfnet_native.audio import apply_attn_limit

    spec_noisy = np.ones((1, 4, 2, 2), dtype=np.float32)
    spec_enh = np.full((1, 4, 2, 2), 7.0, dtype=np.float32)

    result = apply_attn_limit(spec_noisy, spec_enh, np.inf)

    np.testing.assert_array_equal(result, spec_enh)


@pytest.mark.parametrize("attn_limit_db", [-1.0, np.nan])
def test_apply_attn_limit_invalid_values_raise(attn_limit_db: float) -> None:
    from dpdfnet_native.audio import apply_attn_limit

    spec = np.zeros((1, 4, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="attn_limit_db"):
        apply_attn_limit(spec, spec, attn_limit_db)


# ---------------------------------------------------------------------------
# models.py unit tests
# ---------------------------------------------------------------------------

def test_get_model_info_invalid_raises() -> None:
    from dpdfnet_native.models import get_model_info
    with pytest.raises(ValueError, match="Unsupported model"):
        get_model_info("nonexistent_model_xyz")


def test_registry_holds_the_two_native_models() -> None:
    from dpdfnet_native import models

    assert models.supported_models() == ["dpdfnet2_48khz_hr", "dpdfnet8_48khz_hr"]
    assert models.DEFAULT_MODEL in models.MODEL_REGISTRY
    for name in models.supported_models():
        info = models.get_model_info(name)
        assert info.sample_rate == 48000
        assert info.native_id == int(name[7])


def test_available_model_entries_keys() -> None:
    from dpdfnet_native import models

    entries = models.available_model_entries()
    assert [entry["name"] for entry in entries] == models.supported_models()
    for entry in entries:
        for key in ("name", "sample_rate", "description", "weights_path", "ready"):
            assert key in entry, f"Missing key '{key}' in entry {entry}"


def test_weights_are_found_in_the_source_tree() -> None:
    from dpdfnet_native import models

    for name in models.supported_models():
        assert models.weights_path(name).name in {f"{name}.f32", "weights.f32"}


def test_missing_library_override_names_the_variable(tmp_path, monkeypatch) -> None:
    from dpdfnet_native import models

    monkeypatch.setenv("DPDFNET_NATIVE_LIBRARY", str(tmp_path / "missing.so"))
    assert models.find_native_library() is None
    assert models.available_model_entries()[0]["ready"] is False
    with pytest.raises(FileNotFoundError, match="DPDFNET_NATIVE_LIBRARY"):
        models.native_library_path()


def test_library_override_is_used(tmp_path, monkeypatch) -> None:
    from dpdfnet_native import models

    library = tmp_path / "libcustom.so"
    library.write_bytes(b"stub")
    monkeypatch.setenv("DPDFNET_NATIVE_LIBRARY", str(library))
    assert models.native_library_path() == library.resolve()


# ---------------------------------------------------------------------------
# api.py unit tests
# ---------------------------------------------------------------------------

def test_enhance_file_missing_input_raises(tmp_path) -> None:
    from dpdfnet_native.api import enhance_file
    with pytest.raises(FileNotFoundError, match="Input file not found"):
        enhance_file(tmp_path / "nonexistent.wav", tmp_path / "out.wav")


def test_enhance_file_default_output_path(tmp_path, monkeypatch) -> None:
    """When output_path is None, the output file is <stem>_enhanced.wav next to input."""
    import soundfile as sf
    from dpdfnet_native import api

    # Write a minimal valid WAV
    in_path = tmp_path / "speech.wav"
    sf.write(str(in_path), np.zeros(160, dtype=np.float32), 16000)

    # Patch enhance() to return a trivial array so we don't need a real model
    monkeypatch.setattr(api, "enhance", lambda audio, sample_rate, **_kw: np.zeros_like(audio))

    result = api.enhance_file(in_path)
    assert result == in_path.with_name("speech_enhanced.wav")
    assert result.is_file()


def test_enhance_file_forwards_attn_limit_db(tmp_path, monkeypatch) -> None:
    """enhance_file() forwards attn_limit_db to enhance()."""
    import soundfile as sf
    from dpdfnet_native import api

    in_path = tmp_path / "speech.wav"
    out_path = tmp_path / "speech_out.wav"
    sf.write(str(in_path), np.zeros(160, dtype=np.float32), 16000)

    captured: dict[str, float] = {}

    def _fake_enhance(audio, sample_rate, **kwargs):
        captured["attn_limit_db"] = kwargs["attn_limit_db"]
        return np.zeros_like(np.asarray(audio, dtype=np.float32))

    monkeypatch.setattr(api, "enhance", _fake_enhance)

    result = api.enhance_file(in_path, out_path, attn_limit_db=12.0)

    assert result == out_path.resolve()
    assert captured["attn_limit_db"] == 12.0


def test__enhance_file_with_runner_forwards_attn_limit_db(tmp_path, monkeypatch) -> None:
    """_enhance_file_with_runner() forwards attn_limit_db to _enhance_with_runner()."""
    import soundfile as sf
    from dpdfnet_native import api

    in_path = tmp_path / "speech.wav"
    out_path = tmp_path / "speech_out.wav"
    sf.write(str(in_path), np.zeros(160, dtype=np.float32), 16000)

    captured: dict[str, float] = {}

    def _fake_enhance_with_runner(audio, sample_rate, **kwargs):
        captured["attn_limit_db"] = kwargs["attn_limit_db"]
        return np.zeros_like(np.asarray(audio, dtype=np.float32))

    monkeypatch.setattr(api, "_enhance_with_runner", _fake_enhance_with_runner)

    result = api._enhance_file_with_runner(
        input_path=in_path,
        output_path=out_path,
        runner=object(),
        attn_limit_db=9.0,
    )

    assert result == out_path.resolve()
    assert captured["attn_limit_db"] == 9.0


def test_read_audio_unsupported_extension_raises(tmp_path) -> None:
    from dpdfnet_native.api import _read_audio
    bad = tmp_path / "clip.xyz"
    bad.write_bytes(b"data")
    with pytest.raises(ValueError, match="Unsupported audio format"):
        _read_audio(bad)


def test_read_audio_pydub_missing_import_error(tmp_path, monkeypatch) -> None:
    """Reading an .mp3 file without pydub installed raises a helpful ImportError."""
    import builtins
    from dpdfnet_native.api import _read_audio

    mp3 = tmp_path / "clip.mp3"
    mp3.write_bytes(b"fake-mp3")

    real_import = builtins.__import__

    def _block_pydub(name, *args, **kwargs):
        if name == "pydub":
            raise ImportError("No module named 'pydub'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_pydub)
    with pytest.raises(ImportError, match="pydub"):
        _read_audio(mp3)


# ---------------------------------------------------------------------------
# cli.py unit tests
# ---------------------------------------------------------------------------

def test_cli_no_command_returns_0(capsys) -> None:
    from dpdfnet_native import cli
    exit_code = cli.main([])
    assert exit_code == 0


def test_cli_models_command_lists_all_models(capsys) -> None:
    from dpdfnet_native import cli, models
    exit_code = cli.main(["models"])
    assert exit_code == 0
    output = capsys.readouterr().out
    for name in models.supported_models():
        assert name in output


def test_cli_enhance_dir_nonexistent_input_returns_error(tmp_path, capsys) -> None:
    from dpdfnet_native import cli
    exit_code = cli.main(["enhance-dir", str(tmp_path / "nope"), str(tmp_path / "out")])
    assert exit_code == 2


def test_cli_enhance_dir_empty_dir_returns_error(tmp_path, capsys) -> None:
    from dpdfnet_native import cli
    in_dir = tmp_path / "empty"
    in_dir.mkdir()
    exit_code = cli.main(["enhance-dir", str(in_dir), str(tmp_path / "out")])
    assert exit_code == 2


def test_cli_enhance_single_file(tmp_path, monkeypatch, capsys) -> None:
    """CLI enhance command calls enhance_file and reports the output path."""
    import soundfile as sf
    from dpdfnet_native import cli, api

    in_path = tmp_path / "input.wav"
    out_path = tmp_path / "output.wav"
    sf.write(str(in_path), np.zeros(160, dtype=np.float32), 16000)

    monkeypatch.setattr(api, "enhance_file", lambda *_a, **_kw: out_path)
    # Also patch _run_enhance's import of enhance_file
    import dpdfnet_native.cli as cli_mod
    monkeypatch.setattr(cli_mod, "_run_enhance",
                        lambda _args: (print(f"Wrote enhanced audio: {out_path}") or 0))

    exit_code = cli.main(["enhance", str(in_path), str(out_path)])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "output.wav" in output


def test_cli_enhance_forwards_attn_limit_db(tmp_path, monkeypatch) -> None:
    import soundfile as sf
    from dpdfnet_native import api, cli

    in_path = tmp_path / "input.wav"
    out_path = tmp_path / "output.wav"
    sf.write(str(in_path), np.zeros(160, dtype=np.float32), 16000)

    captured: dict[str, float] = {}

    def _fake_enhance_file(*_args, **kwargs):
        captured["attn_limit_db"] = kwargs["attn_limit_db"]
        return out_path

    monkeypatch.setattr(cli, "print_banner", lambda **_kwargs: None)
    monkeypatch.setattr(api, "enhance_file", _fake_enhance_file)

    exit_code = cli.main(
        ["enhance", str(in_path), str(out_path), "--attn_limit_db", "12"]
    )

    assert exit_code == 0
    assert captured["attn_limit_db"] == 12.0


def test_cli_enhance_dir_forwards_attn_limit_db(tmp_path, monkeypatch) -> None:
    import soundfile as sf
    from dpdfnet_native import api, cli

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    sf.write(str(in_dir / "input.wav"), np.zeros(160, dtype=np.float32), 16000)

    captured: dict[str, float] = {}

    def _fake_enhance_file_with_runner(*_args, **kwargs):
        captured["attn_limit_db"] = kwargs["attn_limit_db"]
        return Path(kwargs["output_path"])

    monkeypatch.setattr(cli, "print_banner", lambda **_kwargs: None)
    _use_fake_runner(monkeypatch)
    monkeypatch.setattr(api, "_enhance_file_with_runner", _fake_enhance_file_with_runner)

    exit_code = cli.main(
        ["enhance-dir", str(in_dir), str(out_dir), "--attn-limit-db", "9"]
    )

    assert exit_code == 0
    assert captured["attn_limit_db"] == 9.0


@pytest.mark.parametrize("workers", [1, 3])
def test_cli_enhance_dir_threads_runners_by_worker_count(tmp_path, monkeypatch, workers) -> None:
    """One runner per worker thread, single-threaded when files run in parallel."""
    import soundfile as sf
    from dpdfnet_native import cli

    in_dir = tmp_path / "in"
    in_dir.mkdir()
    for index in range(4):
        sf.write(str(in_dir / f"clip{index}.wav"), np.zeros(320, dtype=np.float32), 16000)

    monkeypatch.setattr(cli, "print_banner", lambda **_kwargs: None)
    created = _use_fake_runner(monkeypatch, win_len=8, passthrough=True)

    exit_code = cli.main(
        ["enhance-dir", str(in_dir), str(tmp_path / "out"), "--workers", str(workers),
         "--model", "dpdfnet8_48khz_hr"]
    )

    assert exit_code == 0
    assert len(list((tmp_path / "out").glob("*_enhanced.wav"))) == 4
    assert 1 <= len(created) <= workers
    assert all(runner.single_thread == (workers > 1) for runner in created)
    assert all(runner.model == "dpdfnet8_48khz_hr" for runner in created)
    assert all(runner.closed for runner in created)
    assert sum(runner.resets for runner in created) == 4


def test_cli_has_no_download_command(capsys) -> None:
    from dpdfnet_native import cli

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["download"])
    assert exc_info.value.code == 2


def test_removed_backend_options_are_rejected() -> None:
    from dpdfnet_native import StreamEnhancer, enhance

    with pytest.raises(TypeError):
        StreamEnhancer(backend="onnx")
    with pytest.raises(TypeError):
        StreamEnhancer("dpdfnet2_48khz_hr", "model.onnx")
    with pytest.raises(TypeError):
        enhance(np.zeros(8, np.float32), 48000, onnx_path="model.onnx")
