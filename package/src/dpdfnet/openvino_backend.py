from __future__ import annotations

from pathlib import Path

import numpy as np

from .onnx_backend import initial_state_from_metadata, validate_state_shape


class OpenVINOFrameRunner:
    """Float32, single-threaded CPU inference with request-local RNN state."""

    def __init__(self, onnx_path: Path) -> None:
        try:
            import openvino as ov
            from openvino import passes
        except ImportError as exc:
            raise ImportError(
                "The OpenVINO streaming backend requires the optional CPU runtime. "
                "Install it with: pip install 'dpdfnet[openvino]'"
            ) from exc

        core = ov.Core()
        model = core.read_model(str(onnx_path))
        if len(model.inputs) != 2 or len(model.outputs) != 2:
            raise ValueError(
                "Expected streaming ONNX signature with 2 inputs and 2 outputs."
            )
        try:
            metadata = {
                key: value.value
                for key, value in model.get_rt_info()["framework"].items()
            }
        except KeyError as exc:
            raise ValueError(
                "ONNX model is missing state initialisation metadata."
            ) from exc
        self._init_state = initial_state_from_metadata(metadata)
        state_shape = model.input(1).partial_shape
        validate_state_shape(
            [
                dimension.get_length() if dimension.is_static else None
                for dimension in state_shape
            ],
            self._init_state,
        )
        shape = model.input(0).partial_shape
        if shape.rank.get_length() != 4 or shape[2].is_dynamic:
            raise ValueError("Expected a spectrum input shaped [1, 1, freq_bins, 2].")
        freq_bins = shape[2].get_length()
        self.win_len = (freq_bins - 1) * 2
        model.reshape(
            {
                model.input(0): [1, 1, freq_bins, 2],
                model.input(1): self._init_state.shape,
            }
        )

        # Internal state removes the full state tensor from the per-hop Python
        # boundary. Each enhancer owns its request; independent streams cannot
        # share a mutable request or its state.
        manager = passes.Manager()
        manager.register_pass(
            passes.MakeStateful({model.input(1).any_name: model.output(1).any_name})
        )
        manager.run_passes(model)
        compiled = core.compile_model(
            model,
            "CPU",
            {
                "INFERENCE_PRECISION_HINT": "f32",
                "INFERENCE_NUM_THREADS": 1,
                "NUM_STREAMS": 1,
                "ENABLE_CPU_PINNING": False,
                # Snippets fusion cannot compile the exported GRU-cell graph
                # reliably. Keep the CPU kernels without that fusion pass.
                "SNIPPETS_MODE": "DISABLE",
            },
        )
        self._request = compiled.create_infer_request()
        self._variable = self._request.query_state()[0]
        self.spec = np.zeros((1, 1, freq_bins, 2), dtype=np.float32)
        self._output = np.empty_like(self.spec)
        self._input_tensor = ov.Tensor(self.spec, shared_memory=True)
        self._output_tensor = ov.Tensor(self._output, shared_memory=True)
        self._request.set_input_tensor(self._input_tensor)
        self._request.set_output_tensor(self._output_tensor)
        self._initial_tensor = ov.Tensor(self._init_state)
        self.reset()

    @property
    def state(self) -> np.ndarray:
        return self._variable.state.data

    def reset(self) -> None:
        self._variable.state = self._initial_tensor

    def run(self) -> np.ndarray:
        # infer() otherwise copies every output into its returned dictionary,
        # even when the caller has already supplied output storage.
        self._request.infer(share_outputs=True)
        return self._output
