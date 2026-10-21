from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, Union

import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class RuntimeModel:
    session: ort.InferenceSession
    init_state: np.ndarray
    in_spec_name: str
    in_state_name: str
    out_spec_name: str
    out_state_name: str


def create_cpu_session(onnx_path: Union[str, Path]) -> ort.InferenceSession:
    path = Path(onnx_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ONNX model file not found: {path}")

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Both package inference paths submit exactly one frame from one stream.
    # Specialization lets ORT fold the dynamic exports' shape subgraphs.
    options.add_free_dimension_override_by_name("batch", 1)
    options.add_free_dimension_override_by_name("time", 1)

    try:
        session = ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize ONNX Runtime CPU session. "
            "Install the CPU package 'onnxruntime' and avoid GPU-only providers."
        ) from exc

    providers = session.get_providers()
    if "CPUExecutionProvider" not in providers:
        raise RuntimeError(
            f"CPUExecutionProvider is not active. Active providers: {providers}"
        )

    return session


def load_initial_state_from_metadata(session: ort.InferenceSession) -> np.ndarray:
    if len(session.get_inputs()) < 2:
        raise ValueError(
            "Expected streaming ONNX model with two inputs: (spec, state)."
        )

    return initial_state_from_metadata(session.get_modelmeta().custom_metadata_map)


def initial_state_from_metadata(meta: Mapping[str, str]) -> np.ndarray:
    try:
        state_size = int(meta["state_size"])
        erb_norm_state_size = int(meta["erb_norm_state_size"])
        spec_norm_state_size = int(meta["spec_norm_state_size"])
        erb_norm_init = np.array(
            [float(x) for x in meta["erb_norm_init"].split(",")], dtype=np.float32
        )
        spec_norm_init = np.array(
            [float(x) for x in meta["spec_norm_init"].split(",")], dtype=np.float32
        )
    except KeyError as exc:
        raise ValueError(
            f"ONNX model is missing required metadata key: {exc}. "
            "Re-export the model to embed state initialisation metadata."
        ) from exc

    init_state = np.zeros(state_size, dtype=np.float32)
    init_state[0:erb_norm_state_size] = erb_norm_init
    init_state[erb_norm_state_size : erb_norm_state_size + spec_norm_state_size] = (
        spec_norm_init
    )
    return np.ascontiguousarray(init_state)


def build_runtime_model(
    onnx_path: Union[str, Path],
) -> RuntimeModel:
    session = create_cpu_session(onnx_path)
    init_state = load_initial_state_from_metadata(session)

    if len(session.get_inputs()) < 2 or len(session.get_outputs()) < 2:
        raise ValueError(
            "Expected streaming ONNX signature with 2 inputs and 2 outputs."
        )
    validate_state_shape(session.get_inputs()[1].shape, init_state)

    return RuntimeModel(
        session=session,
        init_state=init_state,
        in_spec_name=session.get_inputs()[0].name,
        in_state_name=session.get_inputs()[1].name,
        out_spec_name=session.get_outputs()[0].name,
        out_state_name=session.get_outputs()[1].name,
    )


def infer_win_len(session: ort.InferenceSession, default_sr: int) -> int:
    spec_shape = session.get_inputs()[0].shape
    freq_bins = spec_shape[-2] if len(spec_shape) >= 2 else None
    if isinstance(freq_bins, int) and freq_bins > 1:
        return int((freq_bins - 1) * 2)
    return int(round(default_sr * 0.02))


def validate_state_shape(shape: Sequence, state: np.ndarray) -> None:
    if len(shape) != state.ndim or any(
        isinstance(expected, int) and expected != actual
        for expected, actual in zip(shape, state.shape)
    ):
        raise ValueError(
            f"Initial state shape mismatch: expected={shape}, actual={state.shape}"
        )


class OnnxFrameRunner:
    """Single-stream CPU inference with owned, reusable input/output storage."""

    def __init__(self, runtime: RuntimeModel, win_len: int) -> None:
        self._runtime = runtime
        self.spec = np.zeros((1, 1, win_len // 2 + 1, 2), dtype=np.float32)
        self._output = np.empty_like(self.spec)
        self._states = [runtime.init_state.copy(), runtime.init_state.copy()]
        self._bindings = []
        for index in range(2):
            binding = runtime.session.io_binding()
            binding.bind_cpu_input(runtime.in_spec_name, self.spec)
            binding.bind_cpu_input(runtime.in_state_name, self._states[index])
            for name, array in (
                (runtime.out_spec_name, self._output),
                (runtime.out_state_name, self._states[1 - index]),
            ):
                binding.bind_output(
                    name, "cpu", 0, np.float32, array.shape, array.ctypes.data
                )
            self._bindings.append(binding)
        self._index = 0

    @property
    def state(self) -> np.ndarray:
        return self._states[self._index]

    def reset(self) -> None:
        for state in self._states:
            np.copyto(state, self._runtime.init_state)
        self._index = 0

    def run(self) -> np.ndarray:
        # Input and output states must not alias: some graph nodes still read
        # the previous state after other nodes have produced the next state.
        self._runtime.session.run_with_iobinding(self._bindings[self._index])
        self._index = 1 - self._index
        return self._output
