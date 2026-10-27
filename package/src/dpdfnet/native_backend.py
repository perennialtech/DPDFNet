"""Opt-in adapter for the separately built optimized_inference C library."""

from __future__ import annotations

import ctypes as ct
import hashlib
from pathlib import Path

import numpy as np


_FLOAT_PTR = ct.POINTER(ct.c_float)
_MODEL_IDS = {"dpdfnet2_48khz_hr": 2, "dpdfnet8_48khz_hr": 8}


class _ModelInfo(ct.Structure):
    _fields_ = [
        ("abi_version", ct.c_uint32),
        ("name", ct.c_char_p),
        ("weights_sha256", ct.c_char_p),
        ("sample_rate", ct.c_uint32),
        ("hop_samples", ct.c_uint32),
        ("spectrum_floats", ct.c_size_t),
        ("state_floats", ct.c_size_t),
        ("weight_floats", ct.c_size_t),
    ]


def _ptr(array: np.ndarray) -> _FLOAT_PTR:
    return array.ctypes.data_as(_FLOAT_PTR)


class NativeFrameRunner:
    """One C context and recurrent state, with the same frame interface as ONNX."""

    def __init__(self, model: str, library: Path, weights: Path) -> None:
        self._handle = None
        if model not in _MODEL_IDS:
            raise ValueError("Native inference supports only the two 48 kHz HR models")
        self._model_id = _MODEL_IDS[model]
        self._lib = lib = ct.CDLL(str(Path(library).expanduser().resolve()))
        lib.dpdfnet_get_model_info.argtypes = [ct.c_int]
        lib.dpdfnet_get_model_info.restype = ct.POINTER(_ModelInfo)
        lib.dpdfnet_cpu_supported.argtypes = []
        lib.dpdfnet_cpu_supported.restype = ct.c_int
        lib.dpdfnet_create.argtypes = [ct.c_int, _FLOAT_PTR, ct.c_size_t]
        lib.dpdfnet_create.restype = ct.c_void_p
        lib.dpdfnet_init_state.argtypes = [ct.c_int, _FLOAT_PTR]
        lib.dpdfnet_init_state.restype = ct.c_int
        lib.dpdfnet_process.argtypes = [
            ct.c_void_p, _FLOAT_PTR, _FLOAT_PTR, _FLOAT_PTR, _FLOAT_PTR
        ]
        lib.dpdfnet_process.restype = ct.c_int
        lib.dpdfnet_destroy.argtypes = [ct.c_void_p]
        lib.dpdfnet_destroy.restype = None

        info_ptr = lib.dpdfnet_get_model_info(self._model_id)
        if not info_ptr or info_ptr.contents.abi_version != 1:
            raise RuntimeError("Incompatible native DPDFNet library ABI")
        info = info_ptr.contents
        if not lib.dpdfnet_cpu_supported():
            raise RuntimeError("Native inference requires x86-64 AVX2 and FMA")

        data = Path(weights).expanduser().read_bytes()
        if hashlib.sha256(data).hexdigest() != info.weights_sha256.decode("ascii"):
            raise ValueError("Native model weights failed the library SHA-256 check")
        weight_array = np.frombuffer(data, dtype="<f4")
        if weight_array.size != info.weight_floats:
            raise ValueError("Native model weight count does not match the library")

        self._handle = lib.dpdfnet_create(
            self._model_id, _ptr(weight_array), weight_array.size
        )
        if not self._handle:
            raise RuntimeError("Could not construct the native inference context")
        try:
            self.sample_rate = int(info.sample_rate)
            self.hop_size = int(info.hop_samples)
            self.win_len = int(info.spectrum_floats - 2)
            if self.win_len != self.hop_size * 2:
                raise RuntimeError("Unsupported native spectral frame layout")
            self.spec = np.zeros(
                (1, 1, info.spectrum_floats // 2, 2), dtype=np.float32
            )
            self._output = np.empty_like(self.spec)
            self.state = np.empty(info.state_floats, dtype=np.float32)
            self.reset()
        except BaseException:
            self.close()
            raise

    def reset(self) -> None:
        if not self._handle or self._lib.dpdfnet_init_state(
            self._model_id, _ptr(self.state)
        ):
            raise RuntimeError("Could not reset native inference state")

    def run(self) -> np.ndarray:
        if not self._handle or self._lib.dpdfnet_process(
            self._handle,
            _ptr(self.spec),
            _ptr(self.state),
            _ptr(self._output),
            _ptr(self.state),
        ):
            raise RuntimeError("Native inference failed")
        return self._output

    def close(self) -> None:
        if self._handle:
            self._lib.dpdfnet_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()
