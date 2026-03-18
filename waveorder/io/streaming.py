"""Streaming pipeline for zarr → GPU → zarr reconstruction.

Overlaps disk I/O with GPU compute for multi-FOV processing.
Reads each FOV once from zarr (avoiding per-tile decompression),
slices tiles in memory, and processes them on GPU while the next
FOV is being read.

Example
-------
>>> from waveorder.io.streaming import StreamingReconstructor, PinnedBufferPool
>>>
>>> reconstructor = StreamingReconstructor(
...     input_store=input_plate,
...     output_store=output_plate,
...     tile_size=128,
...     batch_size=16,
...     reconstruct_fn=my_optimize_fn,
...     device="cuda",
... )
>>> stats = reconstructor.run(position_names=["A/1/029029", "A/1/029030"])
>>> print(stats)
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from queue import Queue
from typing import Any

import numpy as np
import torch
import xarray as xr


class PinnedBufferPool:
    """Reusable pool of page-locked (pinned) memory buffers.

    Pre-allocating pinned buffers avoids the expensive ``cudaMallocHost``
    syscall on every H2D transfer.

    Parameters
    ----------
    shape : tuple
        Shape of each buffer.
    dtype : torch.dtype
        Data type (default float32).
    n_buffers : int
        Number of buffers in the pool.
    """

    def __init__(
        self, shape: tuple, dtype: torch.dtype = torch.float32, n_buffers: int = 3,
    ):
        self._queue: Queue = Queue(maxsize=n_buffers)
        for _ in range(n_buffers):
            self._queue.put(torch.empty(shape, dtype=dtype, pin_memory=True))
        self.shape = shape
        self.dtype = dtype

    def get(self, timeout: float | None = None) -> torch.Tensor:
        """Acquire a pinned buffer (blocks if none available)."""
        return self._queue.get(timeout=timeout)

    def put(self, buf: torch.Tensor) -> None:
        """Return a buffer to the pool."""
        self._queue.put(buf)

    @property
    def available(self) -> int:
        return self._queue.qsize()


def make_tile_batches(
    fov_y: int, fov_x: int, tile_size: int, batch_size: int,
) -> list[list[tuple[int, int, int, int]]]:
    """Split a FOV into non-overlapping tile batches.

    Returns a list of batches, where each batch is a list of
    ``(y0, y1, x0, x1)`` bounds. Edge tiles that don't match
    ``tile_size`` exactly are dropped.
    """
    all_bounds = []
    for y in range(0, fov_y, tile_size):
        for x in range(0, fov_x, tile_size):
            y_end = y + tile_size
            x_end = x + tile_size
            if y_end <= fov_y and x_end <= fov_x:
                all_bounds.append((y, y_end, x, x_end))
    return [all_bounds[i : i + batch_size] for i in range(0, len(all_bounds), batch_size)]


def slice_tiles(
    fov: np.ndarray, tile_batches: list[list[tuple[int, int, int, int]]],
) -> list[list[xr.DataArray]]:
    """Slice a preloaded FOV into batches of CZYX xr.DataArrays.

    Parameters
    ----------
    fov : ndarray
        Shape ``(Z, Y, X)`` float32.
    tile_batches : list of list of (y0, y1, x0, x1)
        From :func:`make_tile_batches`.

    Returns
    -------
    list of list of xr.DataArray
        Each inner list is one batch of CZYX tiles.
    """
    all_tiles = []
    for batch_bounds in tile_batches:
        tiles = []
        for y0, y1, x0, x1 in batch_bounds:
            tile_np = fov[:, y0:y1, x0:x1].copy()
            tiles.append(
                xr.DataArray(
                    tile_np[np.newaxis].astype("float32"),
                    dims=("c", "z", "y", "x"),
                )
            )
        all_tiles.append(tiles)
    return all_tiles


@dataclass
class PipelineStats:
    """Statistics from a streaming pipeline run."""

    total_time: float = 0.0
    n_fovs: int = 0
    n_fovs_processed: int = 0
    read_times: list[float] = field(default_factory=list)
    compute_times: list[float] = field(default_factory=list)
    write_times: list[float] = field(default_factory=list)
    total_bytes_read: int = 0
    peak_gpu_memory_mb: float = 0.0

    @property
    def avg_read_ms(self) -> float:
        return sum(self.read_times) / len(self.read_times) * 1000 if self.read_times else 0

    @property
    def avg_compute_ms(self) -> float:
        return sum(self.compute_times) / len(self.compute_times) * 1000 if self.compute_times else 0

    @property
    def avg_write_ms(self) -> float:
        return sum(self.write_times) / len(self.write_times) * 1000 if self.write_times else 0

    @property
    def read_bandwidth_mbs(self) -> float:
        total_read_time = sum(self.read_times)
        return self.total_bytes_read / total_read_time / 1e6 if total_read_time > 0 else 0

    @property
    def per_fov_s(self) -> float:
        return self.total_time / self.n_fovs_processed if self.n_fovs_processed > 0 else 0

    def __str__(self) -> str:
        lines = [
            f"StreamingPipeline: {self.n_fovs_processed}/{self.n_fovs} FOVs "
            f"in {self.total_time:.1f}s ({self.per_fov_s:.1f}s/FOV)",
            f"  Read:    {self.avg_read_ms:.0f}ms/FOV ({self.read_bandwidth_mbs:.0f} MB/s)",
            f"  Compute: {self.avg_compute_ms:.0f}ms/FOV",
            f"  Write:   {self.avg_write_ms:.0f}ms/FOV",
        ]
        if self.peak_gpu_memory_mb > 0:
            lines.append(f"  GPU mem: {self.peak_gpu_memory_mb:.0f} MB peak")
        return "\n".join(lines)


class StreamingReconstructor:
    """Multi-FOV streaming pipeline: zarr in → GPU reconstruct → zarr out.

    Reads each FOV in a background thread, processes tile batches on GPU,
    and writes results in a background thread. The three stages overlap
    to minimize the GPU waiting for data from I/O.

    Parameters
    ----------
    input_store
        iohub plate opened with ``open_ome_zarr(path, mode="r")``.
    tile_size : int
        Tile width/height in pixels.
    batch_size : int
        Number of tiles per GPU batch.
    reconstruct_fn : callable
        ``reconstruct_fn(tiles: list[xr.DataArray]) -> list[xr.DataArray]``
        Takes a batch of CZYX tiles and returns reconstructed tiles.
    device : str
        GPU device (default ``"cuda"``).
    output_store : optional
        iohub plate opened with ``open_ome_zarr(path, mode="r+")``.
        If provided, results are stitched and written per-FOV.
    input_channel_idx : int
        Channel index to read from input (default 0).
    n_read_workers : int
        Threads for parallel zarr reads (default 4).
    n_write_workers : int
        Threads for parallel zarr writes (default 2).
    n_buffers : int
        Pipeline depth / pinned buffer count (default 3).
    """

    def __init__(
        self,
        input_store,
        tile_size: int,
        batch_size: int,
        reconstruct_fn: Callable[[list[xr.DataArray]], list[xr.DataArray]],
        device: str = "cuda",
        output_store=None,
        input_channel_idx: int = 0,
        n_read_workers: int = 4,
        n_write_workers: int = 2,
        n_buffers: int = 3,
    ):
        self.input_store = input_store
        self.tile_size = tile_size
        self.batch_size = batch_size
        self.reconstruct_fn = reconstruct_fn
        self.device = device
        self.output_store = output_store
        self.input_channel_idx = input_channel_idx
        self.n_read_workers = n_read_workers
        self.n_write_workers = n_write_workers
        self.n_buffers = n_buffers

        self._fov_shape = None  # set lazily on first read

        # CUDA streams for async transfers
        self._transfer_stream = torch.cuda.Stream(device=device)

        # Pinned buffer pool (allocated on first run when shape is known)
        self._pinned_pool: PinnedBufferPool | None = None

    def run(
        self,
        position_names: list[str],
        t_idx: int = 0,
    ) -> PipelineStats:
        """Process multiple FOVs through the streaming pipeline.

        Parameters
        ----------
        position_names : list of str
            Position paths (e.g., ``["A/1/029029", "A/1/029030"]``).
        t_idx : int
            Time index to process.

        Returns
        -------
        PipelineStats
        """
        stats = PipelineStats(n_fovs=len(position_names))

        read_q: Queue[tuple[str, Any] | None] = Queue(maxsize=self.n_buffers)
        write_q: Queue[tuple[str, Any] | None] = Queue(maxsize=self.n_buffers)

        read_pool = ThreadPoolExecutor(
            max_workers=self.n_read_workers, thread_name_prefix="zarr_read",
        )
        write_pool = ThreadPoolExecutor(
            max_workers=self.n_write_workers, thread_name_prefix="zarr_write",
        )

        def read_worker():
            for pos_name in position_names:
                t0 = time.perf_counter()
                future = read_pool.submit(
                    self._read_fov, pos_name, t_idx,
                )
                tile_batches, n_bytes = future.result()
                stats.read_times.append(time.perf_counter() - t0)
                stats.total_bytes_read += n_bytes
                read_q.put((pos_name, tile_batches))
            read_q.put(None)

        def write_worker():
            pending = []
            while True:
                item = write_q.get()
                if item is None:
                    for f in pending:
                        f.result()
                    break
                pos_name, results = item
                t0 = time.perf_counter()
                if self.output_store is not None:
                    future = write_pool.submit(
                        self._write_fov, pos_name, results,
                    )
                    pending.append(future)
                    if len(pending) > self.n_write_workers * 2:
                        pending[0].result()
                        pending.pop(0)
                stats.write_times.append(time.perf_counter() - t0)

        read_thread = threading.Thread(target=read_worker, daemon=True)
        write_thread = threading.Thread(target=write_worker, daemon=True)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t_start = time.perf_counter()
        read_thread.start()
        write_thread.start()

        # --- Compute on main thread ---
        while True:
            item = read_q.get()
            if item is None:
                write_q.put(None)
                break
            pos_name, tile_batches = item
            t0 = time.perf_counter()
            all_results = []
            for tiles in tile_batches:
                results = self.reconstruct_fn(tiles)
                all_results.extend(results)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            stats.compute_times.append(time.perf_counter() - t0)
            stats.n_fovs_processed += 1
            write_q.put((pos_name, all_results))

        read_thread.join()
        write_thread.join()
        read_pool.shutdown(wait=False)
        write_pool.shutdown(wait=False)

        stats.total_time = time.perf_counter() - t_start
        if torch.cuda.is_available():
            stats.peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / 1e6

        return stats

    def _read_fov(
        self, pos_name: str, t_idx: int,
    ) -> tuple[list[list[xr.DataArray]], int]:
        """Read a full FOV from zarr and slice into tile batches."""
        position = self.input_store[pos_name]
        fov_np = np.array(
            position.data.oindex[t_idx, self.input_channel_idx], dtype="float32",
        )
        n_bytes = fov_np.nbytes

        # Allocate pinned pool on first read (now we know the shape)
        if self._pinned_pool is None:
            self._fov_shape = fov_np.shape
            self._pinned_pool = PinnedBufferPool(
                shape=self._fov_shape,
                n_buffers=self.n_buffers,
            )

        # Copy into pinned buffer for faster future H2D
        pinned = self._pinned_pool.get()
        pinned.copy_(torch.from_numpy(fov_np))
        fov = pinned.numpy()

        fov_y, fov_x = fov.shape[-2], fov.shape[-1]
        batches = make_tile_batches(fov_y, fov_x, self.tile_size, self.batch_size)
        tile_batches = slice_tiles(fov, batches)

        self._pinned_pool.put(pinned)
        return tile_batches, n_bytes

    def _write_fov(self, pos_name: str, results: list[xr.DataArray]) -> None:
        """Stitch tile results and write to output zarr via write_xarray()."""
        if self.output_store is None or not results:
            return

        pos = self.output_store[pos_name]
        Y, X = pos.data.shape[-2], pos.data.shape[-1]

        # Stitch tiles into a full TCZYX array
        output = np.zeros((1, 1, 1, Y, X), dtype="float32")  # (T, C, Z, Y, X)

        tile_idx = 0
        for y in range(0, Y, self.tile_size):
            for x in range(0, X, self.tile_size):
                y_end = y + self.tile_size
                x_end = x + self.tile_size
                if y_end <= Y and x_end <= X and tile_idx < len(results):
                    tile = results[tile_idx]
                    tile_np = tile.values if hasattr(tile, "values") else np.asarray(tile)
                    # Extract (Y, X) from any shape: (C,Z,Y,X), (C,Y,X), or (Y,X)
                    if tile_np.ndim == 4:
                        output[0, 0, 0, y:y_end, x:x_end] = tile_np[0, 0]
                    elif tile_np.ndim == 3:
                        output[0, 0, 0, y:y_end, x:x_end] = tile_np[0]
                    elif tile_np.ndim == 2:
                        output[0, 0, 0, y:y_end, x:x_end] = tile_np
                    tile_idx += 1

        # write_xarray() requires channel coordinate to match store's channel_names
        channel_names = pos.channel_names  # e.g. ["Phase2D"]
        output_da = xr.DataArray(
            output,
            dims=("t", "c", "z", "y", "x"),
            coords={"c": channel_names[:output.shape[1]]},
        )
        pos.write_xarray(output_da)


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def stream_optimize_positions(
    input_zarr: str,
    output_zarr: str,
    position_names: list[str],
    settings,
    opt_iterations: int = 50,
    tile_size: int = 128,
    batch_size: int = 16,
    device: str = "auto",
    input_channel_names: list[str] | None = None,
    n_read_workers: int = 4,
    n_buffers: int = 3,
    t_idx: int = 0,
) -> "PipelineStats":
    """Stream phase optimization across multiple FOVs.

    Reads each FOV once from zarr, runs ``phase.optimize`` per tile,
    stitches results, and writes to the output zarr — all with I/O
    overlapped against GPU compute.

    Parameters
    ----------
    input_zarr : str
        Path to input OME-Zarr plate store.
    output_zarr : str
        Path to output OME-Zarr plate store. Created if it does not exist.
    position_names : list of str
        HCS position paths to process (e.g. ``["A/1/029029", "A/1/029030"]``).
    settings : phase.Settings
        Phase reconstruction settings. Optimizable parameters (those with
        ``lr > 0``) will be optimized per tile.
    opt_iterations : int
        Number of optimizer steps per tile (default 50).
    tile_size : int
        Tile width/height in pixels (default 128).
    batch_size : int
        Number of tiles per GPU batch (default 16).
    device : str
        Compute device: ``"auto"``, ``"cuda"``, ``"cuda:0"``, etc.
    input_channel_names : list of str, optional
        Channel names to use from input. Defaults to the first channel.
    n_read_workers : int
        Parallel zarr read threads (default 4).
    n_buffers : int
        Pipeline depth / pinned buffer count (default 3).
    t_idx : int
        Time index to process (default 0).

    Returns
    -------
    PipelineStats
        Timing and throughput statistics.

    Example
    -------
    >>> from waveorder.api import phase
    >>> from waveorder.io.streaming import stream_optimize_positions
    >>> from waveorder.optim import OptimizableFloat
    >>>
    >>> settings = phase.Settings(
    ...     transfer_function=phase.TransferFunctionSettings(
    ...         wavelength_illumination=0.45,
    ...         yx_pixel_size=0.325,
    ...         z_pixel_size=2.0,
    ...         z_focus_offset=OptimizableFloat(init=0.1, lr=0.02),
    ...         tilt_angle_zenith=OptimizableFloat(init=0.1, lr=0.02),
    ...         tilt_angle_azimuth=OptimizableFloat(init=0.1, lr=0.02),
    ...     ),
    ... )
    >>>
    >>> stats = stream_optimize_positions(
    ...     input_zarr="/data/input.zarr",
    ...     output_zarr="/data/output.zarr",
    ...     position_names=["A/1/028028", "A/1/029029", "A/1/030030"],
    ...     settings=settings,
    ...     opt_iterations=50,
    ...     device="cuda",
    ... )
    >>> print(stats)
    """
    from pathlib import Path

    import xarray as xr
    from iohub.ngff import open_ome_zarr

    from waveorder.api import phase as phase_api
    from waveorder.device import resolve_device
    from waveorder.optim.losses import MidbandPowerLossSettings

    dev = resolve_device(device)

    # Resolve input channel index
    input_store = open_ome_zarr(input_zarr, mode="r")
    if input_channel_names is not None:
        store_channels = input_store.channel_names
        input_channel_idx = store_channels.index(input_channel_names[0])
    else:
        input_channel_idx = 0

    # Create output store if it doesn't exist
    output_path = Path(output_zarr)
    if not output_path.exists():
        _create_output_store(input_store, output_zarr, position_names)
    output_store = open_ome_zarr(output_zarr, mode="r+")

    # Build the per-tile optimize function using phase.optimize
    def optimize_fn(tiles: list[xr.DataArray]) -> list[xr.DataArray]:
        results = []
        for tile in tiles:
            _, recon = phase_api.optimize(
                tile,
                recon_dim=2,
                settings=settings,
                max_iterations=opt_iterations,
                convergence_tol=None,
                convergence_patience=None,
                loss_settings=MidbandPowerLossSettings(
                    midband_fractions=[0.125, 0.25],
                ),
                device=dev,
            )
            results.append(recon)
        return results

    reconstructor = StreamingReconstructor(
        input_store=input_store,
        tile_size=tile_size,
        batch_size=batch_size,
        reconstruct_fn=optimize_fn,
        device=dev,
        output_store=output_store,
        input_channel_idx=input_channel_idx,
        n_read_workers=n_read_workers,
        n_buffers=n_buffers,
    )

    try:
        stats = reconstructor.run(position_names, t_idx=t_idx)
    finally:
        input_store.close()
        output_store.close()

    return stats


def _create_output_store(
    input_store, output_zarr: str, position_names: list[str],
) -> None:
    """Create an OME-Zarr v0.5 output store matching input FOV shape."""
    import numpy as np
    from iohub.ngff import open_ome_zarr

    pos0 = input_store[position_names[0]]
    Y, X = pos0.data.shape[-2], pos0.data.shape[-1]
    output_shape = (1, 1, 1, Y, X)

    with open_ome_zarr(
        output_zarr, layout="hcs", mode="w",
        channel_names=["Phase2D"], version="0.5",
    ) as out:
        for pos_path in position_names:
            parts = pos_path.split("/")
            row, col, pos_name = parts[0], parts[1], parts[2]
            pos = out.create_position(row, col, pos_name)
            pos.create_zeros(
                name="0", shape=output_shape,
                chunks=output_shape, dtype=np.float32,
            )
