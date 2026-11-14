from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from . import native_backend
from .models import DEFAULT_MODEL, available_model_entries


def available_models() -> List[Dict[str, Any]]:
    return available_model_entries()


def enhance(
    audio: np.ndarray,
    sample_rate: int,
    *,
    model: str = DEFAULT_MODEL,
    attn_limit_db: Optional[float] = None,
    single_thread: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    with native_backend.NativeFrameRunner(model, single_thread=single_thread) as runner:
        return _enhance_with_runner(
            audio,
            sample_rate,
            runner=runner,
            attn_limit_db=attn_limit_db,
            progress_callback=progress_callback,
        )


def _enhance_with_runner(
    audio: np.ndarray,
    sample_rate: int,
    *,
    runner,
    attn_limit_db: Optional[float] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """Offline enhancement of one recording on an existing frame runner.

    The runner's state is reset first, so one runner can serve many files.
    """
    from .audio import (
        apply_attn_limit,
        ensure_sample_rate,
        fit_length,
        make_stft_config,
        postprocess_spec,
        preprocess_waveform,
        to_mono,
    )

    waveform = to_mono(np.asarray(audio, dtype=np.float32))
    sr_in = int(sample_rate)
    model_sr = runner.sample_rate

    waveform_model_sr = ensure_sample_rate(waveform, sr_in, model_sr)
    cfg = make_stft_config(runner.win_len)

    # Keep alignment behavior from the original scripts.
    waveform_padded = np.pad(waveform_model_sr, (0, cfg.win_len), mode="constant")
    spec_r = preprocess_waveform(waveform_padded, cfg)

    runner.reset()
    total_frames = int(spec_r.shape[1])
    if progress_callback is not None:
        progress_callback(0, total_frames)
    if total_frames == 0:
        return waveform.copy()
    spec_e = np.empty_like(spec_r)
    for t in range(total_frames):
        runner.spec[0, 0] = spec_r[0, t]
        spec_e[0, t] = runner.run()[0, 0]
        if progress_callback is not None:
            progress_callback(t + 1, total_frames)

    spec_e = apply_attn_limit(spec_r, spec_e, attn_limit_db)
    enhanced_model_sr = postprocess_spec(spec_e, cfg)
    enhanced = ensure_sample_rate(enhanced_model_sr, model_sr, sr_in)
    return fit_length(enhanced, waveform.shape[0]).astype(np.float32, copy=False)


def _enhance_file_with_runner(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    runner,
    attn_limit_db: Optional[float] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """Like `enhance_file()` but reuses an existing frame runner."""
    import soundfile as sf

    from .audio import pcm16_safe

    in_path = Path(input_path).expanduser().resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    audio, sr = _read_audio(in_path)
    enhanced = _enhance_with_runner(
        audio=audio,
        sample_rate=int(sr),
        runner=runner,
        attn_limit_db=attn_limit_db,
        progress_callback=progress_callback,
    )

    out_path = Path(output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), pcm16_safe(enhanced), int(sr), subtype="PCM_16")
    return out_path


# Extensions handled natively by soundfile (libsndfile)
_SF_EXTENSIONS: frozenset[str] = frozenset({".wav", ".flac", ".ogg", ".aiff", ".aif", ".au", ".snd"})
# Extensions that require pydub + ffmpeg
_PYDUB_EXTENSIONS: frozenset[str] = frozenset({".mp3", ".m4a", ".aac", ".wma", ".opus"})
# All supported input extensions
SUPPORTED_EXTENSIONS: frozenset[str] = _SF_EXTENSIONS | _PYDUB_EXTENSIONS


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    """Return (audio float32, sample_rate) for any supported audio format."""
    suffix = path.suffix.lower()
    if suffix in _SF_EXTENSIONS:
        import soundfile as sf
        audio, sr = sf.read(str(path), always_2d=False)
        return np.asarray(audio, dtype=np.float32), int(sr)
    if suffix in _PYDUB_EXTENSIONS:
        try:
            from pydub import AudioSegment
        except ImportError:
            raise ImportError(
                f"Reading {suffix!r} files requires the 'pydub' package and ffmpeg.\n"
                f"Install them with:  pip install 'dpdfnet[mp3]'\n"
                f"and ensure ffmpeg is available on your PATH."
            ) from None
        seg = AudioSegment.from_file(str(path))
        sr = seg.frame_rate
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        samples /= float(1 << (seg.sample_width * 8 - 1))
        if seg.channels > 1:
            samples = samples.reshape(-1, seg.channels)
        return samples, sr
    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    raise ValueError(
        f"Unsupported audio format {suffix!r} for file: {path}\n"
        f"Supported extensions: {supported}"
    )


def enhance_file(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    *,
    model: str = DEFAULT_MODEL,
    attn_limit_db: Optional[float] = None,
    single_thread: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Path:
    import soundfile as sf

    from .audio import pcm16_safe

    in_path = Path(input_path).expanduser().resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    audio, sr = _read_audio(in_path)
    enhanced = enhance(
        audio=audio,
        sample_rate=int(sr),
        model=model,
        attn_limit_db=attn_limit_db,
        single_thread=single_thread,
        progress_callback=progress_callback,
    )

    if output_path is None:
        out_path = in_path.with_name(f"{in_path.stem}_enhanced.wav")
    else:
        out_path = Path(output_path).expanduser().resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), pcm16_safe(enhanced), int(sr), subtype="PCM_16")
    return out_path
