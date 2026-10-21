from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("backend", ["onnx", "openvino"])
def test_hr_demo_without_audio_device(monkeypatch, backend):
    pytest.importorskip("pyqtgraph")
    pytest.importorskip("sounddevice")
    if backend == "openvino":
        pytest.importorskip("openvino")
    root = Path(__file__).resolve().parents[1]
    model_dir = Path(
        os.environ.get("DPDFNET_TEST_MODEL_DIR", root / "model_zoo" / "onnx")
    )
    if not (model_dir / "dpdfnet8_48khz_hr.onnx").is_file():
        pytest.skip("Download the HR ONNX model to run the GUI smoke test")
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.syspath_prepend(str(root / "package" / "src"))
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    import real_time_demo as demo

    app = demo.QtWidgets.QApplication.instance() or demo.QtWidgets.QApplication([])
    streams = []

    class DeviceFreeStream:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            streams.append(kwargs)

        def __enter__(self):
            assert self.kwargs["samplerate"] == 48000
            assert self.kwargs["blocksize"] == 480
            for _ in range(3):
                audio = np.full((480, 1), 0.1, dtype=np.float32)
                output = np.empty_like(audio)
                self.kwargs["callback"](audio, output, 480, None, None)
                assert np.isfinite(output).all()
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(demo.sd, "Stream", DeviceFreeStream)
    monkeypatch.setattr(demo.QtWidgets.QApplication, "exec", lambda self: 0)
    demo.main(
        [
            "--model",
            "dpdfnet8_48khz_hr",
            "--backend",
            backend,
            "--onnx-dir",
            str(model_dir),
        ]
    )
    assert len(streams) == 1
    app.closeAllWindows()
