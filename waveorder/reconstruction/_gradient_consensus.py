"""Reusable matched-adjoint gradient-consensus deconvolution for 3D fluorescence."""

from __future__ import annotations

import math
import numbers
import operator
import threading
from typing import Literal

import torch
from torch import Tensor

from waveorder import rlgc, util

_EPS = 1e-12


def _integer(value, name, minimum):
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer >= {minimum}") from error
    if isinstance(value, bool) or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def _overlaps(first: Tensor, second: Tensor) -> bool:
    a, b = first.data_ptr(), second.data_ptr()
    return a < b + second.numel() * second.element_size() and b < a + first.numel() * first.element_size()


def _compact_hermitian(otf: Tensor) -> Tensor:
    """Check conjugate partners in bounded axial slabs before discarding half the OTF."""
    z, y, x = otf.shape
    zy = (-torch.arange(z, device=otf.device)) % z
    yy = (-torch.arange(y, device=otf.device)) % y
    xx = (-torch.arange(x, device=otf.device)) % x
    # Indexing a few z-planes at a time avoids a second full complex OTF.
    for start in range(0, z, 8):
        partner = otf.index_select(0, zy[start : start + 8]).index_select(1, yy).index_select(2, xx)
        if not torch.allclose(otf[start : start + 8], partner.conj(), rtol=2e-5, atol=2e-6):
            raise ValueError("otf must be Hermitian to use compact real FFT convolution")
    return otf[..., : x // 2 + 1].clone(memory_format=torch.contiguous_format)


class _TorchWorkspace:
    def __init__(self, compact: Tensor, shape: tuple[int, int, int]):
        self.h = compact
        self.adjoint = compact.conj().resolve_conj()
        self.shape = shape
        options = dict(device=compact.device, dtype=torch.float32)
        self.spectrum = torch.empty((*shape[:2], shape[2] // 2 + 1), device=compact.device, dtype=torch.complex64)
        self.estimate = torch.empty(shape, **options)
        self.rates = torch.empty(shape, **options)
        self.gradient = torch.empty(shape, **options)
        self.heads_gradient = torch.empty(shape, **options)
        self.scratch = torch.empty(shape, **options)
        self.step_size = torch.empty(shape, **options)
        # Circular convolution sends a constant field to its DC gain.
        self.transpose_ones = self.adjoint[0, 0, 0].real.clamp_min(_EPS)
        self.forward_ones = self.h[0, 0, 0].real

    def _convolve(self, data: Tensor, transfer: Tensor, output: Tensor, second_transfer: Tensor | None = None):
        torch.fft.rfftn(data, dim=(-3, -2, -1), out=self.spectrum)
        self.spectrum.mul_(transfer)
        if second_transfer is not None:
            self.spectrum.mul_(second_transfer)
        torch.fft.irfftn(self.spectrum, s=self.shape, dim=(-3, -2, -1), out=output)

    def run(
        self,
        measured: Tensor,
        iterations: int,
        background: float,
        tolerance: float | None,
        generator: torch.Generator | None,
        output: Tensor,
        padding: int,
    ):
        self.estimate.fill_(1.0)
        for iteration in range(iterations):
            if iteration == 0:
                self.rates.fill_(self.forward_ones)
            else:
                self._convolve(self.estimate, self.h, self.rates)
            self.rates.add_(background).clamp_(min=_EPS)
            torch.div(measured, self.rates, out=self.scratch)
            self.scratch.sub_(1.0)
            self._convolve(self.scratch, self.adjoint, self.gradient)
            heads = rlgc._coinflip(measured, 0.5, generator)
            torch.div(heads, self.rates, out=self.scratch)
            self.scratch.sub_(0.5)
            self._convolve(self.scratch, self.adjoint, self.heads_gradient)
            torch.sub(self.gradient, self.heads_gradient, out=self.scratch)
            self.scratch.mul_(self.heads_gradient)
            # HT(H(v)) has the compact real-spectrum multiplier H * conj(H).
            # Both ratios are finished; rates can hold the local mask and next estimate.
            self._convolve(self.scratch, self.h, self.rates, self.adjoint)
            torch.div(self.estimate, self.transpose_ones, out=self.step_size)
            self.step_size.masked_fill_(self.rates <= 0, 0.0)
            torch.addcmul(self.estimate, self.gradient, self.step_size, out=self.rates)
            self.rates.clamp_(min=_EPS)
            frozen = bool(torch.count_nonzero(self.step_size) == 0)
            converged = False
            if not frozen and tolerance is not None:
                torch.sub(self.rates, self.estimate, out=self.scratch)
                change = torch.linalg.vector_norm(self.scratch)
                scale = torch.linalg.vector_norm(self.estimate)
                converged = bool((scale > 0) & (change / scale < tolerance))
            self.estimate, self.rates = self.rates, self.estimate
            if frozen or converged:
                break
        result = self.estimate[padding:-padding] if padding else self.estimate
        output.copy_(result)


class GradientConsensusRL:
    """Reusable matched-adjoint RLGC inference with a physical 3D OTF.

    ``otf`` must be contiguous complex64 Hermitian ZYX on the target device and
    sized for the padded measurement. The compact spectrum is validated and
    copied once. Input and output are contiguous float32 unbatched ZYX. The
    Torch backend supports CPU and CUDA; native CUDA builds only when selected
    and never falls back to Torch. Both consume ``generator`` through the same
    Torch binomial photon split as :func:`waveorder.rlgc._coinflip`.

    Calls share a private workspace and serialize across streams. Default
    results own independent storage and remain valid after subsequent calls.
    ``out=`` accepts non-overlapping caller-owned storage for explicit reuse.
    Close the instance to release its OTF, cuFFT plan, and workspaces.
    """

    def __init__(
        self,
        otf: Tensor,
        *,
        z_padding: int = 0,
        iterations: int = 25,
        background: float = 0.0,
        stopping_tolerance: float | None = None,
        backend: Literal["torch", "cuda"] = "torch",
        device: str | torch.device | None = None,
    ):
        if backend not in ("torch", "cuda"):
            raise ValueError("backend must be 'torch' or 'cuda'")
        padding = _integer(z_padding, "z_padding", 0)
        count = _integer(iterations, "iterations", 1)
        if not isinstance(background, numbers.Real) or isinstance(background, bool) or not math.isfinite(background):
            raise ValueError("background must be a finite real scalar")
        if abs(background) > torch.finfo(torch.float32).max:
            raise ValueError("background must fit float32")
        if stopping_tolerance is not None and (
            not isinstance(stopping_tolerance, numbers.Real)
            or isinstance(stopping_tolerance, bool)
            or not math.isfinite(stopping_tolerance)
            or stopping_tolerance < 0
        ):
            raise ValueError("stopping_tolerance must be a finite nonnegative real scalar or None")
        if (
            not isinstance(otf, Tensor)
            or otf.ndim != 3
            or otf.dtype != torch.complex64
            or not otf.is_contiguous()
            or otf.is_conj()
            or otf.is_neg()
            or any(n < 1 for n in otf.shape)
            or otf.requires_grad
        ):
            raise ValueError("otf must be a contiguous complex64 ZYX tensor without gradients")
        target = torch.device(otf.device if device is None else device)
        if target.type not in ("cpu", "cuda"):
            raise ValueError("device must be CPU or CUDA")
        if target.type == "cuda" and target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
        if otf.device != target:
            raise ValueError("otf must already be on the requested device")
        if backend == "cuda" and target.type != "cuda":
            raise ValueError("backend='cuda' requires a CUDA device")
        if padding > (otf.shape[0] - 1) // 2:
            raise ValueError("z_padding must leave at least one measured z-plane")
        self._shape = (otf.shape[0] - 2 * padding, *otf.shape[1:])
        self._padding = padding
        self._iterations = count
        self._background = float(background)
        self._tolerance = stopping_tolerance
        self._device = target
        self._backend = backend
        self._lock = threading.RLock()
        self._closed = False
        self._completion = None
        self._workspace = None
        with torch.no_grad():
            compact = _compact_hermitian(otf)
            if backend == "cuda":
                from ._gradient_consensus_cuda import load

                self._workspace = load().Plan(compact, otf.shape[-1])
            else:
                self._workspace = _TorchWorkspace(compact, tuple(otf.shape))
                if target.type == "cuda":
                    self._completion = torch.cuda.Event()
                    self._completion.record(torch.cuda.current_stream(target))

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    def _check_tensor(self, value: Tensor, name: str):
        if (
            not isinstance(value, Tensor)
            or value.layout != torch.strided
            or tuple(value.shape) != self._shape
            or value.dtype != torch.float32
            or value.device != self._device
            or not value.is_contiguous()
            or value.is_conj()
            or value.is_neg()
            or value.requires_grad
        ):
            raise ValueError(
                f"{name} must be contiguous float32 ZYX with shape {self._shape} on {self._device}, without gradients"
            )

    def __call__(
        self, measured: Tensor, *, generator: torch.Generator | None = None, out: Tensor | None = None
    ) -> Tensor:
        with self._lock:
            if self._closed:
                raise RuntimeError("GradientConsensusRL is closed")
            self._check_tensor(measured, "measured")
            if generator is not None:
                if not isinstance(generator, torch.Generator):
                    raise ValueError("generator must be a torch.Generator on the target device")
                source = generator.device
                source_index = (
                    torch.cuda.current_device() if source.type == "cuda" and source.index is None else source.index
                )
                if source.type != self._device.type or (source.type == "cuda" and source_index != self._device.index):
                    raise ValueError("generator must be a torch.Generator on the target device")
            if out is not None:
                self._check_tensor(out, "out")
                if _overlaps(out, measured):
                    raise ValueError("out must not overlap measured")
            stream = torch.cuda.current_stream(self._device) if self._device.type == "cuda" else None
            if self._completion is not None:
                stream.wait_event(self._completion)
            if stream is not None:
                measured.record_stream(stream)
                if out is not None:
                    out.record_stream(stream)
            try:
                with torch.no_grad():
                    # pad_zyx_along_z uses edge-inclusive reflection for padding < Z;
                    # it switches to a zero halo when padding >= Z.
                    padded = util.pad_zyx_along_z(measured, self._padding)
                    if self._padding:
                        padded.clamp_(min=0.0)
                    else:
                        padded = measured.clamp_min(0.0)
                    result = (
                        out if out is not None else torch.empty(self._shape, dtype=torch.float32, device=self._device)
                    )
                    if self._backend == "cuda":
                        self._workspace.reset()
                        for _ in range(self._iterations):
                            heads = rlgc._coinflip(padded, 0.5, generator)
                            frozen, relative_change = self._workspace.step(padded, heads, self._background)
                            if frozen or (self._tolerance is not None and relative_change < self._tolerance):
                                break
                        self._workspace.copy_output(result, self._padding)
                    else:
                        self._workspace.run(
                            padded,
                            self._iterations,
                            self._background,
                            self._tolerance,
                            generator,
                            result,
                            self._padding,
                        )
                    return result
            finally:
                if self._completion is not None:
                    self._completion.record(stream)

    def close(self):
        """Release cached OTF and workspace after queued operations finish."""
        with self._lock:
            if self._closed:
                return
            if self._workspace is not None and self._backend == "cuda":
                self._workspace.close()
            if self._completion is not None:
                self._completion.synchronize()
            self._workspace = None
            self._completion = None
            self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("GradientConsensusRL is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
