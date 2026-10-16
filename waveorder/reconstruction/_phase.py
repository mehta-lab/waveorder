"""One stateful 3D phase reconstruction interface with private execution adapters."""

from __future__ import annotations

import math
import numbers
import operator
import threading
from typing import Literal

import torch
from torch import Tensor

from waveorder._pixel_size import YXPixelSize

from . import _torch


_TENSOR_PARAMETERS = (
    "numerical_aperture_illumination", "numerical_aperture_detection",
    "tilt_angle_zenith", "tilt_angle_azimuth", "regularization_strength", "absorption_ratio",
)


def _integer(value, name, *, minimum):
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer >= {minimum}") from error
    if isinstance(value, bool) or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def _real_scalar(value, name, *, positive=False, regularization=False):
    if isinstance(value, Tensor):
        if value.ndim != 0 or value.is_complex() or value.dtype == torch.bool:
            raise ValueError(f"{name} must be a finite real scalar")
        valid = bool(torch.isfinite(value))
    else:
        valid = isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)
    if not valid:
        raise ValueError(f"{name} must be a finite real scalar")
    if positive and not bool(value > 0):
        raise ValueError(f"{name} must be positive")
    if regularization and not bool((value >= 0) & (value <= torch.finfo(torch.float32).max)):
        raise ValueError("regularization_strength must be nonnegative and within float32 range")
    return value


def _validate_parameters(parameters):
    for name in _TENSOR_PARAMETERS:
        value = parameters[name]
        if name == "absorption_ratio" and value is None:
            continue
        _real_scalar(
            value, name,
            positive=name in ("numerical_aperture_illumination", "numerical_aperture_detection"),
            regularization=name == "regularization_strength",
        )


def _overlaps(first, second):
    first_start, second_start = first.data_ptr(), second.data_ptr()
    return (
        first_start < second_start + second.numel() * second.element_size()
        and second_start < first_start + first.numel() * first.element_size()
    )


class PhaseReconstruction:
    """Reusable full-volume 3D phase reconstruction with Torch or native CUDA.

    This new interface is separate from ``waveorder.models.phase_thick_3d`` and
    the xarray interfaces, whose broader contracts remain unchanged. It accepts
    one contiguous float32 ZYX volume, Tikhonov regularization, and no apodization.
    Outputs are phase in cycles per voxel on ``device``. Both backends preserve
    full transverse and axial Fourier domains; optical slabs are not spatial tiles.

    ``backend="torch"`` supports CPU/CUDA and gradients for the input, NA, tilt,
    regularization and absorption ratio. Pass scalar Tensor parameters as leaves;
    in-place optimizer updates are read on every call with a fresh graph. Any
    Tensor-valued parameter causes fresh optical preparation, even without gradients.
    Python scalar settings are fixed and cache one compact filter. That cached
    filter still allows new input-gradient graphs on successive calls. Sampling
    sizes, wavelength, medium, padding, phase inversion and steepness are fixed.
    Checkpointing does not guarantee that full-volume backward fits in memory.

    ``backend="cuda"`` is an optional Linux inference implementation. Construction
    explicitly loads/builds the extension, snapshots no-gradient scalar parameters,
    and prepares one filter and cuFFT workspace. Later mutations of constructor
    tensors do not alter it. Gradient-bearing parameters or inputs are rejected;
    there is no automatic fallback. Optical preparation still uses bounded Torch
    operations; the real-FFT solve uses native kernels and cuFFT.

    ``absorption_ratio=None`` omits absorption. Every supplied ratio, including a
    learnable zero, retains the absorption term. Regularization must be a finite
    real scalar in [0, float32 maximum], checked before conversion. Zero keeps
    reference singularities; no epsilon is added.

    Default outputs remain valid after later calls or ``close()``. ``out=`` allows
    explicit caller-owned reuse without a native output clone. It requires matching
    contiguous float32 storage, no input overlap, and no gradient-bearing input or
    parameters. The caller must finish prior reads before reusing that output.
    Input/parameter producers on another CUDA stream must be ordered before the
    current stream. Calls serialize this object's workspace, not external output
    consumption. Separate instances own separate workspaces; no global GPU cache
    is used. Use a context manager or ``close()`` to release prepared state.
    """

    def __init__(
        self,
        zyx_shape: tuple[int, int, int],
        *,
        yx_pixel_size: float | YXPixelSize,
        z_pixel_size: float,
        wavelength_illumination: float,
        z_padding: int,
        index_of_refraction_media: float,
        numerical_aperture_illumination: float | Tensor = 0.9,
        numerical_aperture_detection: float | Tensor = 1.2,
        invert_phase_contrast: bool = False,
        tilt_angle_zenith: float | Tensor = 0.0,
        tilt_angle_azimuth: float | Tensor = 0.0,
        pupil_steepness: float = 1e4,
        regularization_strength: float | Tensor = 1e-3,
        absorption_ratio: float | Tensor | None = None,
        backend: Literal["torch", "cuda"] = "torch",
        device: str | torch.device = "cpu",
    ):
        if backend not in ("torch", "cuda"):
            raise ValueError("backend must be 'torch' or 'cuda'")
        if len(zyx_shape) != 3:
            raise ValueError("zyx_shape must contain three positive integers")
        shape = tuple(_integer(size, "zyx_shape entry", minimum=1) for size in zyx_shape)
        padding = _integer(z_padding, "z_padding", minimum=0)
        target = torch.device(device)
        if target.type not in ("cpu", "cuda"):
            raise ValueError("PhaseReconstruction supports CPU or CUDA devices")
        if backend == "cuda" and target.type != "cuda":
            raise ValueError("backend='cuda' requires a CUDA device")
        if isinstance(yx_pixel_size, Tensor) or (
            isinstance(yx_pixel_size, dict) and any(isinstance(value, Tensor) for value in yx_pixel_size.values())
        ):
            raise ValueError("yx_pixel_size must contain fixed Python scalars")
        pixels = YXPixelSize.from_value(yx_pixel_size)
        for name, value in (
            ("yx_pixel_size.y", pixels.y), ("yx_pixel_size.x", pixels.x),
            ("z_pixel_size", z_pixel_size), ("wavelength_illumination", wavelength_illumination),
            ("index_of_refraction_media", index_of_refraction_media), ("pupil_steepness", pupil_steepness),
        ):
            if isinstance(value, Tensor):
                raise ValueError(f"{name} must be a fixed Python scalar")
            _real_scalar(value, name, positive=True)
        if not isinstance(invert_phase_contrast, bool):
            raise ValueError("invert_phase_contrast must be bool")
        parameters = dict(
            yx_pixel_size=pixels, z_pixel_size=float(z_pixel_size),
            wavelength_illumination=float(wavelength_illumination), z_padding=padding,
            index_of_refraction_media=float(index_of_refraction_media),
            numerical_aperture_illumination=numerical_aperture_illumination,
            numerical_aperture_detection=numerical_aperture_detection,
            invert_phase_contrast=invert_phase_contrast,
            tilt_angle_zenith=tilt_angle_zenith, tilt_angle_azimuth=tilt_angle_azimuth,
            pupil_steepness=float(pupil_steepness), regularization_strength=regularization_strength,
            absorption_ratio=absorption_ratio,
        )
        if backend == "cuda" and any(
            isinstance(value, Tensor) and value.requires_grad for value in parameters.values()
        ):
            raise ValueError("The CUDA reconstruction backend is inference-only; parameters must not require gradients")
        _validate_parameters(parameters)
        if backend == "cuda":
            for name in _TENSOR_PARAMETERS:
                if isinstance(parameters[name], Tensor):
                    parameters[name] = parameters[name].item()
        if target.type == "cuda" and target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
        self._shape = shape
        self._device = target
        self._backend = backend
        self._parameters = parameters
        self._dynamic = any(isinstance(value, Tensor) for value in parameters.values())
        self._lock = threading.RLock()
        self._closed = False
        self._filter = None
        self._native = None
        self._completion = None
        if backend == "cuda":
            from ._cuda import load

            native = load()
        if not self._dynamic:
            # Keep the cached filter usable by future input-gradient graphs even
            # when the caller constructs this object inside inference_mode().
            with torch.inference_mode(False), torch.no_grad():
                self._filter = _torch.prepare_filter(shape, **parameters, device=target)
            if backend == "cuda":
                self._native = native.Plan(self._filter.values, self._filter.logical_shape[-1], padding)
            elif target.type == "cuda":
                self._completion = torch.cuda.Event()
                self._completion.record(torch.cuda.current_stream(target))

    @property
    def backend(self):
        return self._backend

    @property
    def device(self):
        return self._device

    @property
    def shape(self):
        return self._shape

    def _check_tensor(self, value, name):
        if not isinstance(value, Tensor):
            raise ValueError(f"{name} must be a torch.Tensor")
        if (
            value.layout != torch.strided or tuple(value.shape) != self._shape
            or value.dtype != torch.float32 or value.device != self._device
            or not value.is_contiguous() or value.is_conj() or value.is_neg()
        ):
            raise ValueError(f"{name} must be contiguous float32 ZYX with shape {self._shape} on {self._device}")

    def __call__(self, data: Tensor, *, out: Tensor | None = None) -> Tensor:
        with self._lock:
            if self._closed:
                raise RuntimeError("PhaseReconstruction is closed")
            self._check_tensor(data, "data")
            parameters_require_grad = any(
                isinstance(value, Tensor) and value.requires_grad for value in self._parameters.values()
            )
            if self._backend == "cuda" and data.requires_grad:
                raise ValueError("The CUDA reconstruction backend is inference-only; data must not require gradients")
            if out is not None:
                self._check_tensor(out, "out")
                if data.requires_grad or out.requires_grad or parameters_require_grad:
                    raise ValueError("out= does not support tensors requiring gradients")
                if _overlaps(data, out):
                    raise ValueError("out must not overlap data")
            # Updated Tensor parameters are checked before allocating optical state.
            if self._dynamic:
                _validate_parameters(self._parameters)
            if self._backend == "cuda":
                result = torch.empty(self._shape, dtype=torch.float32, device=self._device) if out is None else out
                return self._native.run(data, result)
            stream = None
            if self._device.type == "cuda":
                stream = torch.cuda.current_stream(self._device)
                if self._completion is None:
                    self._completion = torch.cuda.Event()
                else:
                    stream.wait_event(self._completion)
                data.record_stream(stream)
                if out is not None:
                    out.record_stream(stream)
            try:
                inverse = self._filter
                if inverse is None:
                    inverse = _torch.prepare_filter(self._shape, **self._parameters, device=self._device)
                if stream is not None:
                    inverse.values.record_stream(stream)
                return _torch.reconstruct(data, inverse, out)
            finally:
                if stream is not None:
                    self._completion.record(stream)

    def close(self):
        """Wait for this object's queued work and release prepared state. Idempotent."""
        with self._lock:
            if self._closed:
                return
            if self._native is not None:
                self._native.close()
            if self._completion is not None:
                self._completion.synchronize()
            self._native = None
            self._filter = None
            self._completion = None
            self._parameters = {}
            self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("PhaseReconstruction is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
