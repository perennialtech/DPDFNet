"""ctypes adapter for the bundled W7A8 C runtime (optimized_inference)."""

from __future__ import annotations

import ctypes as ct
from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .models import get_model_info, native_library_path, weights_path


_FLOAT_PTR = ct.POINTER(ct.c_float)
_CREATE_SINGLE_THREAD = 0x1


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


@lru_cache(maxsize=None)
def _load(library: Path) -> ct.CDLL:
    lib = ct.CDLL(str(library))
    lib.dpdfnet_get_model_info.argtypes = [ct.c_int]
    lib.dpdfnet_get_model_info.restype = ct.POINTER(_ModelInfo)
    lib.dpdfnet_cpu_supported.argtypes = []
    lib.dpdfnet_cpu_supported.restype = ct.c_int
    lib.dpdfnet_create_with_flags.argtypes = [
        ct.c_int, _FLOAT_PTR, ct.c_size_t, ct.c_uint32
    ]
    lib.dpdfnet_create_with_flags.restype = ct.c_void_p
    lib.dpdfnet_init_state.argtypes = [ct.c_int, _FLOAT_PTR]
    lib.dpdfnet_init_state.restype = ct.c_int
    lib.dpdfnet_process.argtypes = [
        ct.c_void_p, _FLOAT_PTR, _FLOAT_PTR, _FLOAT_PTR, _FLOAT_PTR
    ]
    lib.dpdfnet_process.restype = ct.c_int
    lib.dpdfnet_destroy.argtypes = [ct.c_void_p]
    lib.dpdfnet_destroy.restype = None
    if not lib.dpdfnet_cpu_supported():
        raise RuntimeError(
            "DPDFNet's native runtime needs an x86-64 CPU with AVX2 and FMA "
            "(Intel Haswell or AMD Excavator and later)."
        )
    return lib


def _load_weights(model_id: int, library: Path, weights: Path) -> np.ndarray:
    # The context copies the weights, so this buffer can go after creation.
    info = _load(library).dpdfnet_get_model_info(model_id)
    if not info or info.contents.abi_version != 1:
        raise RuntimeError("Incompatible native DPDFNet library ABI")
    data = weights.read_bytes()
    if hashlib.sha256(data).hexdigest() != info.contents.weights_sha256.decode("ascii"):
        raise ValueError("Native model weights failed the library SHA-256 check")
    array = np.frombuffer(data, dtype="<f4")
    if array.size != info.contents.weight_floats:
        raise ValueError("Native model weight count does not match the library")
    return array


class NativeFrameRunner:
    """One C context and its recurrent state, run one spectral frame at a time.

    Write a frame into :attr:`spec`, call :meth:`run`, and read the enhanced
    frame from the returned array, which is reused by the next call.

    By default the context owns one helper thread that runs the model's two
    branches concurrently, with bit-identical results. ``single_thread`` runs
    every hop on the calling thread instead, which suits running many
    contexts in parallel. A runner is not thread-safe; use one per thread.
    """

    def __init__(
        self,
        model: str,
        library: Optional[Union[str, Path]] = None,
        weights: Optional[Union[str, Path]] = None,
        *,
        single_thread: bool = False,
    ) -> None:
        self._handle = None
        self._model_id = get_model_info(model).native_id
        library = Path(library or native_library_path()).expanduser().resolve()
        weights = Path(weights or weights_path(model)).expanduser().resolve()
        self._lib = lib = _load(library)
        weight_array = _load_weights(self._model_id, library, weights)
        info = lib.dpdfnet_get_model_info(self._model_id).contents

        self._handle = lib.dpdfnet_create_with_flags(
            self._model_id, _ptr(weight_array), weight_array.size,
            _CREATE_SINGLE_THREAD if single_thread else 0,
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

    def __enter__(self) -> "NativeFrameRunner":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
