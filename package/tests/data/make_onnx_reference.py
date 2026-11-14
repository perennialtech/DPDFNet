"""Regenerate onnx_reference.npz from the FP32 ONNX models.

This needs Ceva's `dpdfnet` package (made with 0.6.0), which runs the FP32
ONNX models on ONNX Runtime and installs alongside `dpdfnet-native`, and the
two 48 kHz ONNX exports (from Hugging Face, Ceva-IP/DPDFNet, onnx/):

    pip install 'dpdfnet==0.6.0'
    python make_onnx_reference.py path/to/onnx/dir

The fixture is the independent oracle for the native runtime: the native
output must stay close to what the original model produces on the same input.
"""

from __future__ import annotations

import hashlib
from importlib.metadata import version
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synthetic import SAMPLE_RATE, noisy_vowel  # noqa: E402
from dpdfnet import StreamEnhancer, enhance  # noqa: E402

MODELS = ("dpdfnet2_48khz_hr", "dpdfnet8_48khz_hr")


def main() -> None:
    onnx_dir = Path(sys.argv[1])
    noisy, _clean = noisy_vowel()
    arrays = {}
    for model in MODELS:
        onnx_path = onnx_dir / f"{model}.onnx"
        stream = StreamEnhancer(model=model, onnx_path=onnx_path)
        streamed = np.concatenate([stream.process(noisy), stream.flush()])
        offline = enhance(noisy, SAMPLE_RATE, model=model, onnx_path=onnx_path)
        arrays[f"{model}_stream"] = streamed.astype(np.float16)
        arrays[f"{model}_offline"] = offline.astype(np.float16)
        arrays[f"{model}_onnx_sha256"] = np.array(
            hashlib.sha256(onnx_path.read_bytes()).hexdigest()
        )
    arrays["dpdfnet_version"] = np.array(version("dpdfnet"))
    np.savez_compressed(Path(__file__).with_name("onnx_reference.npz"), **arrays)


if __name__ == "__main__":
    main()
