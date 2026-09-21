"""Explicitly loaded Linux CUDA reconstruction extension. No import-time build."""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
import threading
from pathlib import Path

import torch

_module = None
_load_lock = threading.Lock()
_runtime_info = None


def _torch_cufft():
    """Locate cuFFT through Torch's loaded dependency, not a toolkit substitute."""
    import ctypes

    torch_cuda = Path(torch.__file__).resolve().parent / "lib" / "libtorch_cuda.so"
    if not torch_cuda.is_file():
        raise RuntimeError("Cannot locate installed Torch's libtorch_cuda.so for cuFFT linkage")
    try:
        library = ctypes.CDLL(str(torch_cuda))
        version_function = library.cufftGetVersion
        version_function.argtypes = [ctypes.POINTER(ctypes.c_int)]
        version_function.restype = ctypes.c_int

        class DlInfo(ctypes.Structure):
            _fields_ = [
                ("filename", ctypes.c_char_p), ("base", ctypes.c_void_p),
                ("symbol", ctypes.c_char_p), ("address", ctypes.c_void_p),
            ]

        dladdr = ctypes.CDLL(None).dladdr
        dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
        dladdr.restype = ctypes.c_int
        info = DlInfo()
        if not dladdr(ctypes.cast(version_function, ctypes.c_void_p), ctypes.byref(info)) or not info.filename:
            raise RuntimeError("dladdr could not resolve Torch's cuFFT dependency")
        runtime = Path(os.fsdecode(info.filename)).resolve()
        if not runtime.is_file() or "libcufft.so" not in runtime.name:
            raise RuntimeError(f"Unexpected Torch cuFFT dependency: {runtime}")
        version = ctypes.c_int()
        if version_function(ctypes.byref(version)) != 0:
            raise RuntimeError("Torch's cufftGetVersion failed")
        return runtime, version.value
    except (AttributeError, OSError) as error:
        raise RuntimeError("Cannot resolve cuFFT from Torch's loaded CUDA dependency; no toolkit fallback is used") from error


def load():
    """Build/load on explicit request, linking the exact cuFFT used by Torch."""
    global _module, _runtime_info
    if _module is not None:
        return _module
    with _load_lock:
        if _module is not None:
            return _module
        if sys.platform != "linux":
            raise RuntimeError("The native reconstruction backend currently supports Linux; use backend='torch'")
        if torch.version.cuda is None or not torch.cuda.is_available():
            raise RuntimeError("The native reconstruction backend requires CUDA-enabled Torch and an available GPU")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Construct PhaseReconstruction before CUDA graph capture")
        from torch.utils.cpp_extension import CUDA_HOME
        from torch.utils.cpp_extension import load as load_extension

        if CUDA_HOME is None or not (Path(CUDA_HOME) / "bin" / "nvcc").is_file():
            raise RuntimeError("The native reconstruction backend requires a compatible CUDA toolkit; set CUDA_HOME")
        if shutil.which("ninja") is None:
            raise RuntimeError("The native reconstruction backend requires Ninja; install waveorder[native]")
        source_directory = Path(__file__).resolve().parents[1] / "csrc"
        sources = [source_directory / "reconstruction.cpp", source_directory / "reconstruction.cu"]
        if not all(source.is_file() for source in sources):
            raise RuntimeError("Reinstall waveorder with its packaged csrc/reconstruction.cpp and reconstruction.cu files")
        runtime, version = _torch_cufft()
        cache_root = Path(os.environ.get(
            "TORCH_EXTENSIONS_DIR", str(Path(tempfile.gettempdir()) / f"waveorder-native-{os.getuid()}")
        )).expanduser()
        tag = re.sub(r"[^A-Za-z0-9_.-]", "_", f"py{sys.version_info.major}{sys.version_info.minor}-{torch.__version__}")
        build_directory = cache_root / f"reconstruction-{tag}"
        build_directory.mkdir(parents=True, exist_ok=True)
        try:
            module = load_extension(
                name="waveorder_reconstruction_cuda",
                sources=[str(source) for source in sources],
                extra_cflags=["-O3"], extra_cuda_cflags=["-O3"],
                extra_ldflags=[str(runtime), f"-Wl,-rpath,{runtime.parent}"],
                build_directory=str(build_directory), with_cuda=True, verbose=False,
            )
        except (OSError, RuntimeError) as error:
            raise RuntimeError(
                "Unable to build/load native reconstruction. Check CUDA_HOME, compiler/toolkit compatibility "
                f"and TORCH_EXTENSIONS_DIR. Native error: {error}"
            ) from error
        _runtime_info = {"cufft_path": str(runtime), "cufft_version": version, "torch": torch.__version__}
        _module = module
        return module
