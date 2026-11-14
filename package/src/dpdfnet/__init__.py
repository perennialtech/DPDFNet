from typing import TYPE_CHECKING

__all__ = [
    "enhance",
    "enhance_file",
    "available_models",
    "StreamEnhancer",
]

if TYPE_CHECKING:
    from .api import available_models, enhance, enhance_file
    from .stream import StreamEnhancer


def __getattr__(name: str):
    if name in {"enhance", "enhance_file", "available_models"}:
        from . import api

        return getattr(api, name)
    if name == "StreamEnhancer":
        from .stream import StreamEnhancer

        return StreamEnhancer
    raise AttributeError(f"module 'dpdfnet' has no attribute '{name}'")
