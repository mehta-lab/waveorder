"""Reusable unmatched Wiener–Butterworth Richardson–Lucy inference on ZYX tiles."""

from __future__ import annotations

import math
import numbers
import operator
import threading
import warnings

import torch
from torch import Tensor

from waveorder import backprojector, rlgc, util


def _positive_integer(value, name, minimum):
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer >= {minimum}") from error
    if isinstance(value, bool) or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def _scalar(value, name, *, positive=False):
    if (
        not isinstance(value, numbers.Real)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or (positive and value <= 0)
    ):
        raise ValueError(f"{name} must be a finite {'positive ' if positive else ''}real scalar")
    return float(value)


def _overlaps(a: Tensor, b: Tensor) -> bool:
    return (
        a.data_ptr() < b.data_ptr() + b.numel() * b.element_size()
        and b.data_ptr() < a.data_ptr() + a.numel() * a.element_size()
    )


def _check_hermitian(filter_: Tensor, name: str) -> None:
    """Validate real-convolution symmetry without another full complex volume."""
    z, y, x = filter_.shape
    device = filter_.device
    partner_y = (-torch.arange(y, device=device)) % y
    partner_x = (-torch.arange(x, device=device)) % x
    # A float32 FFT of a real PSF can differ by several ulps at nearly-zero
    # coefficients. Scale absolute tolerance by the OTF gain, not by 1.
    atol = 2e-6 * float(filter_.abs().amax())
    for start in range(0, z, 8):
        slab = filter_[start : start + 8]
        if not bool(torch.isfinite(slab).all()):
            raise ValueError(f"{name} must contain only finite values")
        partner_z = (-torch.arange(start, min(z, start + 8), device=device)) % z
        partner = filter_.index_select(0, partner_z).index_select(1, partner_y).index_select(2, partner_x)
        if not bool(torch.allclose(slab, partner.conj(), rtol=2e-5, atol=atol)):
            raise ValueError(f"{name} must be Hermitian across all three axes for real FFT reconstruction")


class WienerButterworthRL:
    """Cache a physical H and one unmatched B for repeated 3D RL inference.

    ``otf`` is a precomputed, unshifted complex64 Hermitian (Z+2*padding,Y,X)
    transfer function; inputs and results are contiguous float32 (Z,Y,X) on
    ``device``. This is inference only: constructor filters and inputs are
    snapshotted/read without autograd. The forward/transpose convention is
    ``real(ifftn(fftn(x) * H_or_B))``; RFFT is used only after validating both
    full filters' Hermitian symmetry. Axial padding uses edge-inclusive reflection
    if padding < Z, otherwise zeros. The estimate starts at one, uses the
    rlgc epsilon=1e-12 floor, and is cropped only after the last iteration.
    ``resolution_mode='manual'`` accepts positive ``resolution_zyx_px`` to
    reconstruct when an undersampled PSF prevents FWHM estimation.

    Default results own their storage and remain valid after later calls/close.
    ``out=`` requires nonoverlapping caller-owned storage; callers must order
    any reuse/consumption across CUDA streams. Calls on one instance serialize
    its workspace; input producers must be ordered on the current stream.
    The native backend explicitly builds a Linux CUDA extension using a matching
    CUDA toolkit, C++17 compiler and Ninja; it never falls back to Torch.
    """

    def __init__(
        self,
        otf: Tensor,
        *,
        z_padding=0,
        alpha=None,
        beta=None,
        order=8,
        resolution_mode="fwhm",
        resolution_zyx_px=None,
        beta_convention="reference",
        iterations=1,
        background=0.0,
        backend="torch",
        device=None,
    ):
        if backend not in ("torch", "cuda"):
            raise ValueError("backend must be 'torch' or 'cuda'")
        if not isinstance(otf, Tensor) or otf.ndim != 3 or not all(s > 0 for s in otf.shape):
            raise ValueError("otf must be a nonempty complex64 ZYX tensor")
        if (
            otf.dtype != torch.complex64
            or otf.layout != torch.strided
            or not otf.is_contiguous()
            or otf.is_conj()
            or otf.is_neg()
            or otf.requires_grad
        ):
            raise ValueError("otf must be contiguous, physical complex64 ZYX without gradients")
        padding = _positive_integer(z_padding, "z_padding", 0)
        if otf.shape[0] <= 2 * padding:
            raise ValueError("otf Z must exceed twice z_padding")
        iterations = _positive_integer(iterations, "iterations", 1)
        order = _positive_integer(order, "order", 1)
        if resolution_mode not in ("fwhm", "fwhm_over_sqrt2", "manual"):
            raise ValueError("resolution_mode must be 'fwhm', 'fwhm_over_sqrt2', or 'manual'")
        if resolution_mode == "manual":
            if not isinstance(resolution_zyx_px, (tuple, list)) or len(resolution_zyx_px) != 3:
                raise ValueError("resolution_mode='manual' requires three resolution_zyx_px values")
            resolution_zyx_px = tuple(
                _scalar(value, f"resolution_zyx_px[{axis}]", positive=True)
                for axis, value in enumerate(resolution_zyx_px)
            )
        elif resolution_zyx_px is not None:
            raise ValueError("resolution_zyx_px requires resolution_mode='manual'")
        if beta_convention not in ("reference", "paper"):
            raise ValueError("beta_convention must be 'reference' or 'paper'")
        if alpha is not None:
            alpha = _scalar(alpha, "alpha", positive=True)
        if beta is not None:
            beta = _scalar(beta, "beta", positive=True)
        background = _scalar(background, "background")
        target = torch.device(otf.device if device is None else device)
        if target.type not in ("cpu", "cuda"):
            raise ValueError("device must be CPU or CUDA")
        if backend == "cuda" and target.type != "cuda":
            raise ValueError("backend='cuda' requires a CUDA device")
        if target.type == "cuda" and target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
        if otf.device != target:
            raise ValueError("otf must already reside on device")
        if iterations > 5:
            warnings.warn(
                f"rl_iterations={iterations} with rl_back_projector='wiener_butterworth': "
                "unmatched back projectors reach a resolution-limited result in 1-5 iterations "
                "and introduce artifacts beyond that. Consider rl_iterations=1.",
                UserWarning,
                stacklevel=2,
            )
        self._lock = threading.RLock()
        self._closed = False
        self.backend = backend
        self.device = target
        self.shape = (otf.shape[0] - 2 * padding, otf.shape[1], otf.shape[2])
        self.z_padding = padding
        self.iterations = iterations
        self.background = background
        self._native = None
        self._completion = None
        self._h = self._b = self._normalizer = self._initial_rate = None
        with torch.no_grad():
            _check_hermitian(otf, "otf")
            h = otf.detach().clone()
            b = backprojector.calculate_back_projector(
                h,
                "wiener_butterworth",
                alpha=alpha,
                beta=beta,
                order=order,
                resolution_mode=resolution_mode,
                resolution_zyx_px=resolution_zyx_px,
                beta_convention=beta_convention,
            ).contiguous()
            _check_hermitian(b, "Wiener–Butterworth back projector")
            if backend == "cuda":
                from ._wiener_butterworth_cuda import load

                self._native = load().Plan(
                    h[..., : h.shape[-1] // 2 + 1].contiguous(),
                    b[..., : b.shape[-1] // 2 + 1].contiguous(),
                    h.shape[-1],
                    padding,
                    iterations,
                    background,
                )
            else:
                self._h = h[..., : h.shape[-1] // 2 + 1].contiguous()
                self._b = b[..., : b.shape[-1] // 2 + 1].contiguous()
                # For circular convolution, H_T(ones) equals B at DC at every
                # voxel. Keep one scalar instead of an entire padded volume.
                self._normalizer = rlgc.clip(self._b[0, 0, 0].real)
                # H(ones) is constant on the padded circular grid; retain its
                # true DC gain rather than assuming a normalized forward PSF.
                self._initial_rate = rlgc.clip(self._h[0, 0, 0].real + background)
                if target.type == "cuda":
                    self._completion = torch.cuda.Event()
                    self._completion.record(torch.cuda.current_stream(target))

    def _check_tensor(self, value, name):
        if (
            not isinstance(value, Tensor)
            or tuple(value.shape) != self.shape
            or value.dtype != torch.float32
            or value.device != self.device
            or value.layout != torch.strided
            or not value.is_contiguous()
            or value.is_conj()
            or value.is_neg()
            or value.requires_grad
        ):
            raise ValueError(
                f"{name} must be contiguous float32 ZYX, shape {self.shape}, on {self.device}, without gradients"
            )

    def __call__(self, measured: Tensor, *, out: Tensor | None = None) -> Tensor:
        with self._lock:
            if self._closed:
                raise RuntimeError("WienerButterworthRL is closed")
            self._check_tensor(measured, "measured")
            if out is not None:
                self._check_tensor(out, "out")
                if _overlaps(measured, out):
                    raise ValueError("out must not overlap measured")
            if self.backend == "cuda":
                result = torch.empty(self.shape, dtype=torch.float32, device=self.device) if out is None else out
                return self._native.run(measured, result)
            stream = torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None
            if stream is not None:
                stream.wait_event(self._completion)
                measured.record_stream(stream)
                if out is not None:
                    out.record_stream(stream)
            try:
                with torch.no_grad():
                    padded = util.pad_zyx_along_z(measured, self.z_padding).clamp_min(0.0)
                    padded_shape = padded.shape
                    estimate = torch.ones_like(padded) if self.iterations > 1 else None
                    for n in range(self.iterations):
                        if n == 0:
                            ratio = padded / self._initial_rate - 1.0
                            if self.iterations == 1:
                                del padded
                        else:
                            spectrum = torch.fft.rfftn(estimate)
                            spectrum.mul_(self._h)
                            rates = torch.fft.irfftn(spectrum, s=padded_shape)
                            ratio = padded / rlgc.clip(rates.add_(self.background)) - 1.0
                        spectrum = torch.fft.rfftn(ratio)
                        spectrum.mul_(self._b)
                        gradient = torch.fft.irfftn(spectrum, s=padded_shape)
                        if estimate is None:
                            estimate = rlgc.clip(gradient.mul_(self._normalizer.reciprocal()).add_(1.0))
                        else:
                            estimate = rlgc.clip(estimate + gradient * (estimate / self._normalizer))
                    result = estimate[self.z_padding : -self.z_padding] if self.z_padding else estimate
                    if out is not None:
                        out.copy_(result)
                        return out
                    return result.contiguous()
            finally:
                if stream is not None:
                    self._completion.record(stream)

    def close(self):
        """Release cached filters and native cuFFT workspace; idempotent."""
        with self._lock:
            if self._closed:
                return
            if self._native is not None:
                self._native.close()
            if self._completion is not None:
                self._completion.synchronize()
            self._native = self._h = self._b = self._normalizer = self._initial_rate = self._completion = None
            self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("WienerButterworthRL is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
