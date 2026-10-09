"""Explicit Linux CUDA extension loader for matched-adjoint RLGC inference."""

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
    """Build once on explicit native selection; never substitute the Torch backend."""
    global _module
    if _module is not None:
        return _module
    with _lock:
        if _module is not None:
            return _module
        if sys.platform != "linux":
            raise RuntimeError("Native gradient-consensus reconstruction requires Linux")
        if torch.version.cuda is None or not torch.cuda.is_available():
            raise RuntimeError("Native gradient-consensus reconstruction requires CUDA-enabled Torch and a GPU")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Construct GradientConsensusRL before CUDA graph capture")
        from torch.utils.cpp_extension import CUDA_HOME
        from torch.utils.cpp_extension import load as load_extension

        toolkit = Path(CUDA_HOME) if CUDA_HOME is not None else None
        nvcc = toolkit / "bin" / "nvcc" if toolkit is not None else None
        if nvcc is None or not nvcc.is_file():
            raise RuntimeError(
                "Native gradient-consensus reconstruction requires a matching CUDA toolkit; set CUDA_HOME"
            )
        if shutil.which("ninja") is None:
            raise RuntimeError("Native gradient-consensus reconstruction requires Ninja on PATH")
        # nvcc -V reports the toolkit version; PyTorch requires the same CUDA minor.
        import subprocess

        version = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True, check=True).stdout
        match = re.search(r"release\s+(\d+)\.(\d+)", version)
        if match is None or tuple(map(int, match.groups())) != tuple(map(int, torch.version.cuda.split(".")[:2])):
            raise RuntimeError(
                f"Native gradient-consensus reconstruction requires CUDA {torch.version.cuda} toolkit (nvcc reports {version.strip()})"
            )
        runtime, _ = _torch_cufft()
        sources = [
            Path(__file__).resolve().parents[1] / "csrc" / filename
            for filename in ("gradient_consensus.cpp", "gradient_consensus.cu")
        ]
        if not all(source.is_file() for source in sources):
            raise RuntimeError("Native gradient-consensus C++ and CUDA sources are not installed")
        cache = Path(
            os.environ.get("TORCH_EXTENSIONS_DIR", str(Path(tempfile.gettempdir()) / f"waveorder-native-{os.getuid()}"))
        ).expanduser()
        tag = re.sub(r"[^A-Za-z0-9_.-]", "_", f"py{sys.version_info.major}{sys.version_info.minor}-{torch.__version__}")
        build = cache / f"gradient-consensus-{tag}"
        build.mkdir(parents=True, exist_ok=True)
        try:
            _module = load_extension(
                name="waveorder_gradient_consensus_cuda",
                sources=[str(source) for source in sources],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3"],
                extra_ldflags=[str(runtime), f"-Wl,-rpath,{runtime.parent}"],
                build_directory=str(build),
                with_cuda=True,
                verbose=False,
            )
        except (OSError, RuntimeError) as error:
            raise RuntimeError(
                f"Unable to build native gradient-consensus reconstruction; check CUDA_HOME, C++17, Ninja, and TORCH_EXTENSIONS_DIR: {error}"
            ) from error
        return _module
