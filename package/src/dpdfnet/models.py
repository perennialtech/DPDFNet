from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ModelInfo:
    name: str
    sample_rate: int
    frame_ms: float
    description: str
    native_id: int


MODEL_REGISTRY: Dict[str, ModelInfo] = {
    "dpdfnet2_48khz_hr": ModelInfo(
        name="dpdfnet2_48khz_hr",
        sample_rate=48000,
        frame_ms=20.0,
        description="High-resolution 48 kHz DPDFNet-2 model.",
        native_id=2,
    ),
    "dpdfnet8_48khz_hr": ModelInfo(
        name="dpdfnet8_48khz_hr",
        sample_rate=48000,
        frame_ms=20.0,
        description="High-resolution 48 kHz DPDFNet-8 model.",
        native_id=8,
    ),
}

DEFAULT_MODEL = "dpdfnet2_48khz_hr"

LIBRARY_NAME = "libdpdfnet.dll" if os.name == "nt" else "libdpdfnet.so"

# Wheels bundle the library and weights here.
_BUNDLED_DIR = Path(__file__).resolve().with_name("_native")
# In a source checkout (an editable install), fall back to the C runtime's own
# build and model directories: package/src/dpdfnet -> repository root.
_SOURCE_DIR = Path(__file__).resolve().parents[3] / "optimized_inference"


def get_model_info(model: str) -> ModelInfo:
    try:
        return MODEL_REGISTRY[model]
    except KeyError as exc:
        supported = ", ".join(supported_models())
        raise ValueError(f"Unsupported model '{model}'. Supported: {supported}") from exc


def supported_models() -> List[str]:
    return sorted(MODEL_REGISTRY)


def _first_file(candidates: List[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def find_native_library() -> Optional[Path]:
    """The native library: ``DPDFNET_NATIVE_LIBRARY``, the wheel's copy, or a source build."""
    override = os.environ.get("DPDFNET_NATIVE_LIBRARY")
    if override:
        return _first_file([Path(override).expanduser()])
    return _first_file([_BUNDLED_DIR / LIBRARY_NAME, _SOURCE_DIR / "build" / LIBRARY_NAME])


def find_weights(model: str) -> Optional[Path]:
    """The checksummed weights for *model*: the wheel's copy or the source tree's."""
    name = get_model_info(model).name
    return _first_file([
        _BUNDLED_DIR / f"{name}.f32",
        _SOURCE_DIR / "models" / name / "weights.f32",
    ])


def native_library_path() -> Path:
    library = find_native_library()
    if library is None:
        override = os.environ.get("DPDFNET_NATIVE_LIBRARY")
        where = f"DPDFNET_NATIVE_LIBRARY={override!r} is not a file" if override else (
            "this installation does not include it"
        )
        raise FileNotFoundError(
            f"The native DPDFNet library was not found: {where}. Install a dpdfnet "
            "wheel for Linux or Windows x86-64, or build optimized_inference "
            "(python optimized_inference/tools/build.py) and point "
            "DPDFNET_NATIVE_LIBRARY at the built library."
        )
    return library


def weights_path(model: str) -> Path:
    weights = find_weights(model)
    if weights is None:
        raise FileNotFoundError(
            f"Native weights for '{model}' were not found in this installation."
        )
    return weights


def available_model_entries() -> List[Dict[str, Any]]:
    library_found = find_native_library() is not None
    entries: List[Dict[str, Any]] = []
    for name in supported_models():
        weights = find_weights(name)
        row = asdict(MODEL_REGISTRY[name])
        row["weights_path"] = str(weights) if weights else None
        row["ready"] = library_found and weights is not None
        entries.append(row)
    return entries
