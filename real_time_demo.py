import argparse
from importlib.metadata import version
import queue
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import sounddevice as sd
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from dpdfnet import StreamEnhancer
from dpdfnet.models import supported_models

# =======================
# CONFIG (GLOBAL STYLE)
# =======================
ONNX_DIR = Path("./model_zoo/onnx")
MODEL_NAME = "dpdfnet8_48khz_hr"

BUFFER_SECONDS = 5.0
PLAYBACK_MIX = 0.0
ONNX_MS_EMA_ALPHA = 0.02


def align_signal_length(x: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.shape[0] == target_len:
        return x
    if x.shape[0] > target_len:
        return x[:target_len]
    out = np.zeros(target_len, dtype=np.float32)
    out[: x.shape[0]] = x
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Real-time CPU speech enhancement")
    parser.add_argument("--model", choices=supported_models(), default=MODEL_NAME)
    parser.add_argument(
        "--backend", choices=["onnx", "openvino", "native"], default="onnx"
    )
    parser.add_argument("--onnx-dir", type=Path, default=ONNX_DIR)
    parser.add_argument("--native-library", type=Path)
    parser.add_argument("--native-weights", type=Path)
    args = parser.parse_args(argv)
    # Must be set before creating ImageItem/plots so spectrogram arrays
    # shaped [freq, time] map to y=freq and x=time.
    pg.setConfigOptions(antialias=True, imageAxisOrder="row-major")

    native = args.backend == "native"
    if native and (not args.native_library or not args.native_weights):
        parser.error("native needs --native-library and --native-weights")
    if not native and (args.native_library or args.native_weights):
        parser.error("native paths require --backend native")
    onnx_path = (
        None
        if native
        else (args.onnx_dir / f"{args.model}.onnx").expanduser().resolve()
    )
    if onnx_path is not None and not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    enhancer = StreamEnhancer(
        model=args.model,
        onnx_path=onnx_path,
        backend=args.backend,
        native_library=args.native_library,
        native_weights=args.native_weights,
    )
    sample_rate, hop_size = enhancer.sample_rate, enhancer.hop_size
    buffer_seconds = float(BUFFER_SECONDS)
    playback_mix = float(np.clip(PLAYBACK_MIX, 0.0, 1.0))

    spec_n_fft = 1024
    spec_max_freq = sample_rate / 2.0
    spec_cols = int(buffer_seconds * sample_rate / hop_size)
    spec_window = np.hanning(spec_n_fft).astype(np.float32)
    db_min, db_max = -80.0, 0.0
    eps = 1e-10

    spec_noisy = np.zeros((spec_n_fft // 2 + 1, spec_cols), dtype=np.float32)
    spec_enh = np.zeros_like(spec_noisy)
    db_spec_noisy = 20 * np.log10(spec_noisy + eps)
    db_spec_enh = db_spec_noisy.copy()
    spec_td_noisy = np.zeros(spec_n_fft, dtype=np.float32)
    spec_td_enh = np.zeros(spec_n_fft, dtype=np.float32)
    spec_queue: queue.Queue[tuple[np.ndarray, np.ndarray]] = queue.Queue(maxsize=500)

    agc_enabled = True
    agc_target_rms = 0.12
    agc_rms_floor = 1e-3
    agc_min_gain = 0.25
    agc_max_gain = 8.0
    agc_attack_sec = 0.03
    agc_release_sec = 0.30
    agc_gain = 1.0

    onnx_ms_ema = float("nan")

    def buffer_to_mag(buf: np.ndarray) -> np.ndarray:
        frame = buf.astype(np.float32, copy=True) * spec_window
        return np.abs(np.fft.rfft(frame)).astype(np.float32)

    def apply_output_agc(x: np.ndarray, frames: int) -> np.ndarray:
        nonlocal agc_gain
        x = np.asarray(x, dtype=np.float32)
        if (not agc_enabled) or x.size == 0:
            return np.clip(x, -1.0, 1.0)
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        desired_gain = float(
            np.clip(
                agc_target_rms / max(rms, agc_rms_floor), agc_min_gain, agc_max_gain
            )
        )
        attack_samples = max(1.0, agc_attack_sec * sample_rate)
        release_samples = max(1.0, agc_release_sec * sample_rate)
        attack_coeff = float(np.exp(-frames / attack_samples))
        release_coeff = float(np.exp(-frames / release_samples))
        coeff = attack_coeff if desired_gain < agc_gain else release_coeff
        agc_gain = coeff * agc_gain + (1.0 - coeff) * desired_gain
        return np.clip(x * agc_gain, -1.0, 1.0)

    def enhance_frame(noisy_frame: np.ndarray) -> np.ndarray:
        nonlocal onnx_ms_ema
        t0 = time.perf_counter()
        enhanced = enhancer.process(noisy_frame)
        frame_ms = (time.perf_counter() - t0) * 1000.0
        if np.isnan(onnx_ms_ema):
            onnx_ms_ema = frame_ms
        else:
            onnx_ms_ema = (
                1.0 - ONNX_MS_EMA_ALPHA
            ) * onnx_ms_ema + ONNX_MS_EMA_ALPHA * frame_ms
        return enhanced

    def update_mix_label() -> None:
        if mix_value_label is not None:
            mix_value_label.setText(f"{playback_mix:.1f}")

    def set_playback_mix(value: float) -> None:
        nonlocal playback_mix
        playback_mix = float(np.clip(value, 0.0, 1.0))
        update_mix_label()

    def on_mix_slider_changed(value: int) -> None:
        set_playback_mix(value / 10.0)

    def update_agc_button_text() -> None:
        agc_button.setText(f"AGC: {'ON' if agc_enabled else 'OFF'}")

    def on_agc_button_toggled(checked: bool) -> None:
        nonlocal agc_enabled
        agc_enabled = bool(checked)
        update_agc_button_text()

    try:
        cmap = pg.colormap.get("magma")
        cmap_lut = cmap.getLookupTable(0.0, 1.0, 256)
    except Exception:
        cmap_lut = None

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    win = QtWidgets.QWidget()
    win.setWindowTitle(f"DPDFNet {args.model}: Real-Time CPU Enhancement")
    main_layout = QtWidgets.QVBoxLayout()
    win.setLayout(main_layout)
    glw = pg.GraphicsLayoutWidget()
    main_layout.addWidget(glw)

    p1 = glw.addPlot(row=0, col=0)
    p1.setTitle("Noisy")
    p1.setLabel("left", "Frequency", units="Hz")
    p1.setLabel("bottom", "Time", units="s")
    img_noisy = pg.ImageItem()
    p1.addItem(img_noisy)

    glw.nextRow()
    p2 = glw.addPlot(row=1, col=0)
    p2.setTitle("Enhanced")
    p2.setLabel("left", "Frequency", units="Hz")
    p2.setLabel("bottom", "Time", units="s")
    img_enh = pg.ImageItem()
    p2.addItem(img_enh)

    img_noisy.setImage(db_spec_noisy, autoLevels=False)
    img_enh.setImage(db_spec_noisy, autoLevels=False)

    freq_bins = spec_n_fft // 2 + 1
    dx = buffer_seconds / spec_cols
    dy = spec_max_freq / freq_bins
    t_noisy = QtGui.QTransform()
    t_noisy.scale(dx, dy)
    img_noisy.setTransform(t_noisy)
    t_enh = QtGui.QTransform()
    t_enh.scale(dx, dy)
    img_enh.setTransform(t_enh)
    if cmap_lut is not None:
        img_noisy.setLookupTable(cmap_lut)
        img_enh.setLookupTable(cmap_lut)
    img_noisy.setLevels((db_min, db_max))
    img_enh.setLevels((db_min, db_max))
    p1.setRange(xRange=(0, buffer_seconds), yRange=(0, spec_max_freq))
    p2.setRange(xRange=(0, buffer_seconds), yRange=(0, spec_max_freq))

    mix_layout = QtWidgets.QHBoxLayout()
    main_layout.addLayout(mix_layout)
    mix_title = QtWidgets.QLabel("Playback Mix (Noisy -> Enhanced)")
    lbl_noisy = QtWidgets.QLabel("Noisy")
    lbl_enh = QtWidgets.QLabel("Enhanced")

    mix_slider_local = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    mix_slider_local.setRange(0, 10)  # 0.0 ... 1.0 in 0.1 steps
    mix_slider_local.setSingleStep(1)
    mix_slider_local.setPageStep(1)
    mix_slider_local.setTickInterval(1)
    mix_slider_local.setTickPosition(QtWidgets.QSlider.TicksBelow)
    mix_slider_local.setValue(int(round(playback_mix * 10)))

    mix_value_label = QtWidgets.QLabel(f"{playback_mix:.1f}")
    mix_value_label.setAlignment(QtCore.Qt.AlignCenter)
    mix_value_label.setMinimumWidth(36)
    mix_layout.addWidget(mix_title)
    mix_layout.addWidget(lbl_noisy)
    mix_layout.addWidget(mix_slider_local, 1)
    mix_layout.addWidget(lbl_enh)
    mix_layout.addWidget(mix_value_label)
    mix_slider_local.valueChanged.connect(on_mix_slider_changed)

    agc_layout = QtWidgets.QHBoxLayout()
    main_layout.addLayout(agc_layout)
    agc_title = QtWidgets.QLabel("Output AGC")
    agc_button = QtWidgets.QPushButton()
    agc_button.setCheckable(True)
    agc_button.setChecked(agc_enabled)
    agc_button.toggled.connect(on_agc_button_toggled)
    update_agc_button_text()
    agc_layout.addWidget(agc_title)
    agc_layout.addStretch(1)
    agc_layout.addWidget(agc_button)

    perf_layout = QtWidgets.QHBoxLayout()
    main_layout.addLayout(perf_layout)
    perf_title = QtWidgets.QLabel("Enhancement including STFT/ISTFT (EMA)")
    perf_value_label = QtWidgets.QLabel("-- ms/frame")
    perf_value_label.setAlignment(QtCore.Qt.AlignCenter)
    perf_value_label.setMinimumWidth(120)
    perf_layout.addWidget(perf_title)
    perf_layout.addStretch(1)
    perf_layout.addWidget(perf_value_label)

    def audio_callback(indata, outdata, frames, time_info, status):
        # Suppress sounddevice callback status spam (e.g., underflow/overflow).

        noisy = align_signal_length(indata[:, 0], frames)
        enhanced = align_signal_length(enhance_frame(noisy), frames)
        mixed = (1.0 - playback_mix) * noisy + playback_mix * enhanced
        outdata[:, 0] = apply_output_agc(mixed, frames)

        # Only copy the small audio blocks here; plotting FFTs belong on the UI
        # thread so a slow display cannot delay playback.
        try:
            spec_queue.put_nowait((noisy.copy(), enhanced.copy()))
        except queue.Full:
            pass

    def update_plot():
        nonlocal spec_noisy, spec_enh, db_spec_noisy, db_spec_enh, onnx_ms_ema
        nonlocal spec_td_noisy, spec_td_enh
        updated = False
        try:
            while True:
                noisy, enhanced = spec_queue.get_nowait()
                length = min(len(noisy), spec_n_fft)
                spec_td_noisy = np.roll(spec_td_noisy, -length)
                spec_td_enh = np.roll(spec_td_enh, -length)
                spec_td_noisy[-length:] = noisy[-length:]
                spec_td_enh[-length:] = enhanced[-length:]
                noisy_mag = buffer_to_mag(spec_td_noisy)
                enh_mag = buffer_to_mag(spec_td_enh)
                spec_noisy = np.roll(spec_noisy, -1, axis=1)
                spec_enh = np.roll(spec_enh, -1, axis=1)
                db_spec_noisy = np.roll(db_spec_noisy, -1, axis=1)
                db_spec_enh = np.roll(db_spec_enh, -1, axis=1)
                spec_noisy[:, -1] = noisy_mag
                spec_enh[:, -1] = enh_mag
                db_spec_noisy[:, -1] = 20 * np.log10(noisy_mag + eps)
                db_spec_enh[:, -1] = 20 * np.log10(enh_mag + eps)
                updated = True
        except queue.Empty:
            pass

        if updated:
            img_noisy.setImage(db_spec_noisy, autoLevels=False)
            img_enh.setImage(db_spec_enh, autoLevels=False)
            img_noisy.setLevels((db_min, db_max))
            img_enh.setLevels((db_min, db_max))
        if np.isnan(onnx_ms_ema):
            perf_value_label.setText("-- ms/frame")
        else:
            perf_value_label.setText(f"{onnx_ms_ema:.3f} ms/frame")

    runtime_package = "onnxruntime" if args.backend == "onnx" else "openvino"
    print(f"[INFO] CPU backend: {args.backend} {version(runtime_package)}")
    print(f"[INFO] ONNX model: {onnx_path}")
    print(f"[INFO] Audio params: sr={sample_rate}, hop={hop_size}")
    print("Streaming... speak into the mic. Close the window to stop.")

    timer = QtCore.QTimer()
    timer.timeout.connect(update_plot)
    timer.start(30)

    with sd.Stream(
        samplerate=sample_rate,
        blocksize=hop_size,
        dtype="float32",
        channels=1,
        callback=audio_callback,
    ):
        win.show()
        qt_app = QtWidgets.QApplication.instance()
        if hasattr(qt_app, "exec"):
            qt_app.exec()
        else:
            qt_app.exec_()


if __name__ == "__main__":
    main()
