"""Deterministic speech-like test audio: a gliding vowel in white noise."""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 48000


def noisy_vowel(seconds: float = 1.0, snr_db: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(noisy, clean)`` float32 audio at 48 kHz.

    The clean part is a voiced source with three formants gliding between /a/
    and /i/, under a syllable-rate envelope. Both models keep most of it,
    unlike a plain harmonic buzz, which they suppress as noise.
    """
    rng = np.random.default_rng(2026)
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    f0 = 120 * (1 + 0.08 * np.sin(2 * np.pi * 3 * t) + 0.03 * np.sin(2 * np.pi * 7 * t))
    phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE
    glide = 0.5 * (1 - np.cos(2 * np.pi * 1.5 * t))
    formants = [
        (730 + (270 - 730) * glide, 80),
        (1090 + (2290 - 1090) * glide, 90),
        (2440 + (3010 - 2440) * glide, 120),
    ]
    voiced = np.zeros_like(t)
    for k in range(1, 60):
        amplitude = sum(1 / (1 + ((k * f0 - centre) / width) ** 2) for centre, width in formants)
        voiced += amplitude * np.sin(k * phase)
    envelope = np.clip(np.sin(2 * np.pi * 2.5 * t), 0, None) ** 0.6
    clean = 0.1 * envelope * voiced / np.max(np.abs(voiced))
    noise = rng.standard_normal(t.size)
    noise *= np.sqrt(np.mean(clean**2) / np.mean(noise**2) / 10 ** (snr_db / 10))
    return (clean + noise).astype(np.float32), clean.astype(np.float32)


def agreement_db(reference: np.ndarray, test: np.ndarray) -> float:
    """Signal-to-difference ratio of *test* against *reference*, in dB."""
    reference = np.asarray(reference, dtype=np.float64)
    error = reference - np.asarray(test, dtype=np.float64)
    return float(10 * np.log10(np.sum(reference**2) / np.sum(error**2)))
