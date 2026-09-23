"""Explicit, independently cached Linux CUDA extension for Wiener–Butterworth RL."""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
import threading
from pathlib import Path

import torch

from ._cuda import _torch_cufft

_module = None
_lock = threading.Lock()


def load():
    """Build only when the CUDA backend is explicitly constructed; never fall back."""
    global _module
    if _module is not None:
        return _module
    with _lock:
        if _module is not None:
            return _module
        if sys.platform != "linux":
            raise RuntimeError("Native Wiener–Butterworth RL requires Linux")
        if torch.version.cuda is None or not torch.cuda.is_available():
            raise RuntimeError("Native Wiener–Butterworth RL requires CUDA-enabled Torch and a GPU")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Construct WienerButterworthRL outside CUDA graph capture")
        from torch.utils.cpp_extension import CUDA_HOME
        from torch.utils.cpp_extension import load as load_extension

        if CUDA_HOME is None or not (Path(CUDA_HOME) / "bin" / "nvcc").is_file():
            raise RuntimeError("Native Wiener–Butterworth RL requires a matching CUDA toolkit; set CUDA_HOME")
        if shutil.which("ninja") is None:
            raise RuntimeError("Native Wiener–Butterworth RL requires Ninja")
        source_dir = Path(__file__).resolve().parents[1] / "csrc"
        sources = [source_dir / "wiener_butterworth.cpp", source_dir / "wiener_butterworth.cu"]
        if not all(source.is_file() for source in sources):
            raise RuntimeError("Install packaged wiener_butterworth.cpp and wiener_butterworth.cu")
        runtime, _ = _torch_cufft()
        cache_root = Path(
            os.environ.get("TORCH_EXTENSIONS_DIR", str(Path(tempfile.gettempdir()) / f"waveorder-native-{os.getuid()}"))
        ).expanduser()
        tag = re.sub(r"[^A-Za-z0-9_.-]", "_", f"py{sys.version_info.major}{sys.version_info.minor}-{torch.__version__}")
        directory = cache_root / f"wiener-butterworth-{tag}"
        directory.mkdir(parents=True, exist_ok=True)
        try:
            module = load_extension(
                name="waveorder_wiener_butterworth_cuda",
                sources=[str(source) for source in sources],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3"],
                extra_ldflags=[str(runtime), f"-Wl,-rpath,{runtime.parent}"],
                build_directory=str(directory),
                with_cuda=True,
                verbose=False,
            )
        except (OSError, RuntimeError) as error:
            raise RuntimeError(
                "Unable to build native Wiener–Butterworth RL; check CUDA_HOME, C++17 compiler, "
                f"Ninja and TORCH_EXTENSIONS_DIR. Native error: {error}"
            ) from error
        _module = module
        return module
