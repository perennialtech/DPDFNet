from __future__ import annotations

from typing import Optional, Union
from pathlib import Path

import numpy as np

from .audio import ensure_sample_rate, make_stft_config, to_mono
from .models import DEFAULT_MODEL, resolve_model
from .onnx_backend import OnnxFrameRunner, build_runtime_model, infer_win_len


class StreamEnhancer:
    """Process audio chunk-by-chunk while preserving RNN state across calls.

    Maintains an internal input buffer and overlap-add output buffer so that
    arbitrary chunk sizes can be passed without introducing boundary artefacts.

    The STFT is causal (``center=False``). One model window (20 ms) is buffered
    before the first output hop; subsequent complete input hops produce one
    output hop each (10 ms). This does not include inference time, the model's
    internal lookahead, or audio-device buffering.

    .. note::
        This path uses a hand-rolled causal STFT / ISTFT and will **not**
        produce bit-identical output to :func:`dpdfnet.enhance`, which uses
        ``center=True`` reflection padding.  Both are algorithmically correct;
        the causal variant is necessary for real-time use.

    Args:
        model: Model name (default: ``"dpdfnet2"``).
        onnx_path: Optional path to a custom ONNX file; overrides *model*.
        verbose: Print model resolution / download progress.
        backend: ``"onnx"`` (portable default) or ``"openvino"`` (optional
            float32 CPU runtime). Both run one frame at a time on one CPU thread.
            Construction includes a warmup inference and clears its state before
            accepting audio. Construct the enhancer outside the audio callback.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        onnx_path: Optional[Union[str, Path]] = None,
        verbose: bool = False,
        *,
        backend: str = "onnx",
    ) -> None:
        if backend not in {"onnx", "openvino"}:
            raise ValueError("backend must be 'onnx' or 'openvino'.")
        resolved = resolve_model(
            model=model,
            onnx_path=onnx_path,
            auto_download=True,
            verbose=verbose,
        )
        self._model_sr: int = resolved.info.sample_rate
        if backend == "onnx":
            runtime = build_runtime_model(resolved.onnx_path)
            self._win_len = infer_win_len(runtime.session, self._model_sr)
            self._runner = OnnxFrameRunner(runtime, self._win_len)
        else:
            from .openvino_backend import OpenVINOFrameRunner

            self._runner = OpenVINOFrameRunner(resolved.onnx_path)
            self._win_len = self._runner.win_len
        cfg = make_stft_config(self._win_len)
        self._hop_size: int = cfg.hop_size
        self._window: np.ndarray = cfg.window
        self._freq_bins: int = self._win_len // 2 + 1
        self._windowed = np.empty(self._win_len, dtype=np.float32)
        self._time_frame = np.empty(self._win_len, dtype=np.float32)

        # Pay lazy kernel/allocation costs before the audio callback. Warmup
        # must not advance the state of the user's stream.
        self._runner.run()

        self._input_sr: Optional[int] = None
        self.reset()

    @property
    def sample_rate(self) -> int:
        """Native model sample rate; use it to avoid per-chunk resampling."""
        return self._model_sr

    @property
    def hop_size(self) -> int:
        """Samples per model hop at :attr:`sample_rate` (10 ms)."""
        return self._hop_size

    def reset(self) -> None:
        """Reset RNN state and internal buffers.

        Call this between independent audio segments (e.g. between speakers or
        separate recordings) so that the RNN hidden state does not leak across
        streams.
        """
        self._runner.reset()
        self._in_buf: np.ndarray = np.zeros(self._win_len, dtype=np.float32)
        self._in_len = 0
        self._out_buf: np.ndarray = np.zeros(self._win_len, dtype=np.float32)
        self._input_sr = None

    def process(
        self,
        chunk: np.ndarray,
        sample_rate: Optional[int] = None,
    ) -> np.ndarray:
        """Enhance a chunk of audio, returning enhanced samples.

        Returns an empty array when not enough data has accumulated for the
        first model frame.  Thereafter each call returns enhanced audio whose
        length (at the input sample rate) is proportional to the number of
        complete model frames processed.

        Args:
            chunk: Float32 audio samples.  Stereo input is averaged to mono.
            sample_rate: Sample rate of *chunk*.  May be different from the
                model's native rate; resampling is handled internally.  Omit
                (or pass ``None``) to use the model's native sample rate.
                Must be consistent across calls on the same stream; call
                :meth:`reset` before switching sample rates.

        Returns:
            Enhanced float32 mono samples at *sample_rate* (or model SR if
            *sample_rate* was ``None``).  May have length 0.
        """
        chunk = to_mono(np.asarray(chunk, dtype=np.float32))
        if chunk.size == 0:
            return np.zeros(0, dtype=np.float32)

        sr_in = sample_rate if sample_rate is not None else self._model_sr

        if self._input_sr is None:
            self._input_sr = sr_in
        elif self._input_sr != sr_in:
            raise ValueError(
                f"Sample rate changed from {self._input_sr} to {sr_in} between "
                "process() calls.  Call reset() before processing a new stream."
            )

        chunk_model = ensure_sample_rate(chunk, sr_in, self._model_sr)
        enhanced_model_sr = self._process_model_chunk(chunk_model)
        if sr_in != self._model_sr and enhanced_model_sr.size:
            return ensure_sample_rate(enhanced_model_sr, self._model_sr, sr_in)
        return enhanced_model_sr

    def _process_model_chunk(self, chunk: np.ndarray) -> np.ndarray:
        frames = max(
            0, (self._in_len + chunk.size - self._win_len) // self._hop_size + 1
        )
        output = np.empty(frames * self._hop_size, dtype=np.float32)
        consumed = 0
        written = 0
        while consumed < chunk.size:
            count = min(self._win_len - self._in_len, chunk.size - consumed)
            self._in_buf[self._in_len : self._in_len + count] = chunk[
                consumed : consumed + count
            ]
            consumed += count
            self._in_len += count
            if self._in_len < self._win_len:
                break
            # --- Analysis STFT (causal / center=False) ---
            np.multiply(self._in_buf, self._window, out=self._windowed)
            spec_complex = np.fft.rfft(self._windowed, n=self._win_len)
            self._runner.spec[0, 0, :, 0] = spec_complex.real
            self._runner.spec[0, 0, :, 1] = spec_complex.imag

            # --- Stateful model inference ---
            spec_e_t = self._runner.run()

            # --- Per-frame ISTFT + overlap-add ---
            complex_frame = spec_e_t.view(np.complex64).reshape(self._freq_bins)
            time_frame = np.fft.irfft(complex_frame, n=self._win_len)
            np.multiply(
                time_frame, self._window, out=self._time_frame, casting="unsafe"
            )
            self._out_buf += self._time_frame

            # The Vorbis window satisfies COLA at 50% overlap:
            #   w[n]^2 + w[n+hop]^2 == 1  for all n
            # so the first hop_size samples are fully committed after each frame.
            output[written : written + self._hop_size] = self._out_buf[: self._hop_size]
            written += self._hop_size
            self._out_buf[: self._win_len - self._hop_size] = self._out_buf[
                self._hop_size :
            ]
            self._out_buf[self._win_len - self._hop_size :] = 0.0

            self._in_len = self._win_len - self._hop_size
            self._in_buf[: self._in_len] = self._in_buf[self._hop_size :]
        return output

    def flush(self) -> np.ndarray:
        """Drain any remaining buffered audio by zero-padding to a full frame.

        Call this at the end of a stream to retrieve enhanced audio for the
        last partial window.  The output length equals one hop interval at the
        most-recently-used sample rate (or the model's native rate if
        :meth:`process` was never called).

        Does **not** call :meth:`reset`; call that explicitly if you want to
        reuse the instance for a new stream.

        This pads one STFT frame, not all of the model's internal lookahead.
        It does not guarantee that output length equals the recording length.

        Returns:
            Enhanced float32 samples for the final partial window, or an empty
            array if the input buffer is already empty.
        """
        if self._in_len == 0:
            return np.zeros(0, dtype=np.float32)

        remainder = self._in_len
        sr_in = self._input_sr or self._model_sr

        # Pad at model rate without changing the stream's input sample rate.
        pad = np.zeros(self._win_len - remainder, dtype=np.float32)
        out = self._process_model_chunk(pad)
        self._in_len = 0

        # Trim to only the samples that came from real (non-padded) input.
        # One hop's worth of real audio maps to one hop's worth of output.
        real_out = min(self._hop_size, len(out))
        trimmed = out[:real_out] if len(out) > 0 else out

        if sr_in != self._model_sr:
            trimmed = ensure_sample_rate(trimmed, self._model_sr, sr_in)

        return trimmed.astype(np.float32)
