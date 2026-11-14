"""Bundle the native runtime and its weights into a platform wheel.

A wheel build runs ``optimized_inference/tools/build.py`` (the PGO release
build and all of its contracts) and copies the library and both checksummed
weight files into ``dpdfnet_native/_native``. Set ``DPDFNET_NATIVE_BUILD_DIR`` to a
directory holding an already built library to skip that step, as CI does
after building in MSYS2 on Windows.

The library is loaded with ctypes, not imported, so the wheel works with any
Python 3 and is tagged ``py3-none-<platform>``. Editable installs bundle
nothing: the package then finds the library in ``optimized_inference/build``.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist
from setuptools.dist import Distribution

HERE = Path(__file__).resolve().parent
NATIVE_SOURCE = HERE.parent / "optimized_inference"
MODELS = ("dpdfnet2_48khz_hr", "dpdfnet8_48khz_hr")
LIBRARY = "libdpdfnet.dll" if os.name == "nt" else "libdpdfnet.so"


def _check_weights(library: Path) -> None:
    """Refuse a library (such as a stale prebuilt one) that rejects these weights."""

    class Info(ctypes.Structure):
        _fields_ = [
            ("abi_version", ctypes.c_uint32),
            ("name", ctypes.c_char_p),
            ("weights_sha256", ctypes.c_char_p),
            ("sample_rate", ctypes.c_uint32),
            ("hop_samples", ctypes.c_uint32),
            ("spectrum_floats", ctypes.c_size_t),
            ("state_floats", ctypes.c_size_t),
            ("weight_floats", ctypes.c_size_t),
        ]

    lib = ctypes.CDLL(str(library))
    lib.dpdfnet_get_model_info.argtypes = [ctypes.c_int]
    lib.dpdfnet_get_model_info.restype = ctypes.POINTER(Info)
    for model in MODELS:
        info = lib.dpdfnet_get_model_info(int(model[7]))
        digest = hashlib.sha256((NATIVE_SOURCE / "models" / model / "weights.f32").read_bytes()).hexdigest()
        if not info or info.contents.weights_sha256.decode("ascii") != digest:
            raise SystemExit(f"{library} does not accept this checkout's {model} weights; rebuild it.")


class build_py_with_native(build_py):
    def run(self) -> None:
        super().run()
        if self.editable_mode:
            return
        if not (NATIVE_SOURCE / "CMakeLists.txt").is_file():
            raise SystemExit(
                "dpdfnet-native bundles a native library built from the repository's "
                "optimized_inference directory, which this source tree (an sdist?) "
                "lacks. Build the wheel directly from a repository checkout: "
                "pip install ./package, pip wheel ./package or python -m build --wheel package."
            )
        library = self._native_library()
        _check_weights(library)
        target = Path(self.build_lib) / "dpdfnet_native" / "_native"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(library, target / LIBRARY)
        for model in MODELS:
            shutil.copy2(NATIVE_SOURCE / "models" / model / "weights.f32", target / f"{model}.f32")

    def _native_library(self) -> Path:
        prebuilt = os.environ.get("DPDFNET_NATIVE_BUILD_DIR")
        if prebuilt:
            # Use an absolute path: build frontends may run this from package/.
            library = Path(prebuilt).expanduser().resolve() / LIBRARY
            if not library.is_file():
                raise SystemExit(f"DPDFNET_NATIVE_BUILD_DIR has no {LIBRARY}: {library}")
            return library
        build_dir = Path(self.get_finalized_command("build").build_temp).resolve() / "native"
        subprocess.run(
            [
                sys.executable,
                str(NATIVE_SOURCE / "tools" / "build.py"),
                "--build-dir",
                str(build_dir),
                "--jobs",
                str(os.cpu_count() or 2),
            ],
            check=True,
        )
        return build_dir / LIBRARY


class no_sdist(sdist):
    """Releases are wheels only.

    A source distribution cannot build: it would lack optimized_inference.
    On a platform without a wheel, pip would try it and fail to compile,
    instead of reporting that no distribution matches.
    """

    def run(self) -> None:
        raise SystemExit(
            "dpdfnet-native is released as wheels only; build one with "
            "pip wheel ./package or python -m build --wheel package."
        )


class BinaryDistribution(Distribution):
    """Platform-specific: the wheel carries a native library."""

    def has_ext_modules(self) -> bool:
        return True


class platform_wheel(bdist_wheel):
    def get_tag(self) -> tuple[str, str, str]:
        _python, _abi, platform = super().get_tag()
        return "py3", "none", platform


setup(
    distclass=BinaryDistribution,
    cmdclass={"build_py": build_py_with_native, "bdist_wheel": platform_wheel, "sdist": no_sdist},
)
