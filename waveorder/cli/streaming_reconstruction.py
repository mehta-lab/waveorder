"""
CUDA streaming reconstruction for overlapping I/O and compute.

This module provides functions to reconstruct multiple positions with pipelined
CUDA streams, overlapping:
- CPU/Disk I/O: Load data for position N
- Stream 1: CPU → GPU data transfer for position N
- Default stream: GPU computation for position N-1
- Stream 2: GPU → CPU transfer + disk write for position N-2

This pipelining overlaps I/O with compute for better GPU utilization.
"""

import torch
import numpy as np
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
from iohub.ngff import open_ome_zarr
from waveorder.models import phase_thick_3d
from waveorder.cli.settings import ReconstructionSettings
from waveorder.io import utils


def _check_nan_n_zeros(array):
    """Check if array is all zeros or NaN"""
    return np.all(array == 0) or np.all(np.isnan(array))


def reconstruct_positions_pipelined(
    position_names: list[str],
    input_store_path: Path,
    transfer_function_path: Path,
    config_path: Path,
    output_store_path: Path,
    output_channel_names: list[str],
    t_idx: int = 0,
) -> None:
    """
    Reconstruct multiple positions using CUDA streams for I/O overlap.

    This function processes positions with a 3-stage pipeline:
    - Stage 0: Load position N from disk (CPU)
    - Stage 1: Transfer position N to GPU (CUDA stream 1, async)
    - Stage 2: Compute position N-1 on GPU (default stream)
    - Stage 3: Write position N-2 to disk (CUDA stream 2, async)

    Parameters
    ----------
    position_names : list[str]
        List of position names to reconstruct (e.g., ["020020", "020021", ...])
    input_store_path : Path
        Path to input OME-Zarr store containing raw data
    transfer_function_path : Path
        Path to transfer function OME-Zarr
    config_path : Path
        Path to phase config YAML file
    output_store_path : Path
        Path to output OME-Zarr store
    output_channel_names : list[str]
        Channel names for output (e.g., ["Phase3D"])
    t_idx : int
        Time index to process (default: 0)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA streams require GPU")

    device = torch.device("cuda")

    # Read config
    settings = utils.yaml_to_model(config_path, ReconstructionSettings)
    recon_settings = settings.phase.apply_inverse
    tf_settings = settings.phase.transfer_function

    # Create CUDA streams
    transfer_stream = torch.cuda.Stream()  # For H2D transfers
    write_stream = torch.cuda.Stream()     # For D2H transfers
    # Default stream for compute

    # Pipeline depth: number of buffer slots per worker
    # Optimal at 3: tested depth=10 (780s, 27% GPU, 81GB) vs depth=3 (636s, 43% GPU, 70GB)
    # Deeper pipeline causes memory pressure at 99% capacity, hurting performance
    # depth=3 with multi-threaded I/O is better than depth=10
    pipeline_depth = 3

    # Pipeline buffers
    class PipelineBuffer:
        def __init__(self):
            self.pos_name = None
            self.cpu_data = None      # Pinned CPU tensor
            self.gpu_data = None      # GPU tensor
            self.gpu_result = None    # GPU result tensor
            self.skip = False         # Flag for zero/NaN data

    buffers = [PipelineBuffer() for _ in range(pipeline_depth)]

    # Open datasets (reuse across all positions)
    input_dataset = open_ome_zarr(input_store_path, mode="r")
    output_dataset = open_ome_zarr(output_store_path, mode="r+")
    tf_dataset = open_ome_zarr(transfer_function_path, mode="r")

    # Load transfer functions once and move to GPU
    real_tf_np = tf_dataset["real_potential_transfer_function"][0, 0]
    imag_tf_np = tf_dataset["imaginary_potential_transfer_function"][0, 0]
    real_tf = torch.tensor(real_tf_np, dtype=torch.float32).to(device)
    imag_tf = torch.tensor(imag_tf_np, dtype=torch.float32).to(device)

    # Get input/output channel indices (assuming Phase3D reconstruction)
    # For pheno: input channels are typically 0-4 (5 z-stacks)
    pos_example = input_dataset[position_names[0]]
    num_input_channels = pos_example.data.shape[1]
    input_channel_indices = list(range(num_input_channels))
    output_channel_indices = [0]  # Single Phase3D output channel

    # Thread pool for parallel zarr I/O (4 threads per worker)
    # Zarr chunks are independent files, so loading channels in parallel speeds up disk I/O
    io_thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="zarr_io")

    def load_single_channel(pos_dataset, channel_idx):
        """Load a single channel from zarr (for parallel I/O)"""
        return pos_dataset.data.oindex[t_idx, channel_idx]

    def stage_load(buf_idx: int, pos_name: str):
        """Stage 0: Load data from disk to pinned CPU memory (multi-threaded I/O)"""
        buf = buffers[buf_idx]
        buf.pos_name = pos_name
        buf.skip = False

        try:
            # Load data with multi-threaded I/O for zarr chunks
            pos_dataset = input_dataset[pos_name]

            # Load each channel in parallel using thread pool
            channel_futures = [
                io_thread_pool.submit(load_single_channel, pos_dataset, ch)
                for ch in input_channel_indices
            ]

            # Gather results (blocks until all channels loaded)
            channel_arrays = [future.result() for future in channel_futures]
            czyx_uint16 = np.stack(channel_arrays, axis=0)

            # Check for zero/NaN
            if _check_nan_n_zeros(czyx_uint16):
                print(f"  [{pos_name}] All zeros/NaN, skipping", file=sys.stderr)
                buf.skip = True
                return

            # Convert to float32 in pinned memory for fast async transfer
            czyx_int32 = np.int32(czyx_uint16)
            czyx_float32 = torch.tensor(czyx_int32, dtype=torch.float32)
            buf.cpu_data = czyx_float32.pin_memory()

        except Exception as e:
            print(f"  [{pos_name}] Load error: {e}", file=sys.stderr)
            buf.skip = True

    def stage_transfer(buf_idx: int):
        """Stage 1: Transfer from pinned CPU to GPU (async)"""
        buf = buffers[buf_idx]
        if buf.skip or buf.cpu_data is None:
            return

        # Async transfer on dedicated stream
        with torch.cuda.stream(transfer_stream):
            buf.gpu_data = buf.cpu_data.to(device, non_blocking=True)

    def stage_compute(buf_idx: int):
        """Stage 2: GPU computation on default stream"""
        buf = buffers[buf_idx]
        if buf.skip or buf.gpu_data is None:
            return

        # Wait for H2D transfer to complete before computing
        # Required to avoid race condition where compute operates on uninitialized GPU data
        torch.cuda.current_stream().wait_stream(transfer_stream)

        try:
            # Apply phase reconstruction (on default stream)
            # czyx_data shape: (C, Z, Y, X) -> extract first channel for processing
            zyx_data = buf.gpu_data[0]  # Take first channel

            result = phase_thick_3d.apply_inverse_transfer_function(
                zyx_data=zyx_data,
                real_potential_transfer_function=real_tf,
                imaginary_potential_transfer_function=imag_tf,
                z_padding=tf_settings.z_padding,
                absorption_ratio=0.0,
                reconstruction_algorithm=recon_settings.reconstruction_algorithm,
                regularization_strength=recon_settings.regularization_strength,
                TV_rho_strength=recon_settings.TV_rho_strength,
                TV_iterations=recon_settings.TV_iterations,
            )

            # Expand to CZYX
            buf.gpu_result = result.unsqueeze(0)  # (Z, Y, X) -> (1, Z, Y, X)

        except Exception as e:
            print(f"  [{buf.pos_name}] Compute error: {e}", file=sys.stderr)
            buf.skip = True

    def stage_write(buf_idx: int):
        """Stage 3: Transfer to CPU and write to disk"""
        buf = buffers[buf_idx]
        if buf.skip or buf.gpu_result is None:
            return

        # Async transfer to CPU on write stream
        with torch.cuda.stream(write_stream):
            cpu_result = buf.gpu_result.cpu()

        # Wait for D2H transfer to complete before writing to disk
        # Required to avoid race condition where disk write uses uninitialized CPU data
        write_stream.synchronize()

        try:
            # Convert torch tensor to numpy for zarr compatibility
            # (torch.dtype has no .name attribute, which zarr requires)
            cpu_result_np = cpu_result.numpy()

            # Write to disk
            output_dataset[buf.pos_name][0].oindex[
                t_idx, output_channel_indices
            ] = cpu_result_np

        except Exception as e:
            print(f"  [{buf.pos_name}] Write error: {e}", file=sys.stderr)

        # Clear buffer
        buf.pos_name = None
        buf.cpu_data = None
        buf.gpu_data = None
        buf.gpu_result = None
        buf.skip = False

    # ===== PIPELINE EXECUTION =====
    num_positions = len(position_names)

    # Prime the pipeline
    for i in range(min(pipeline_depth, num_positions)):
        idx = i % pipeline_depth
        stage_load(idx, position_names[i])
        if i >= 1:
            stage_transfer((i - 1) % pipeline_depth)
        if i >= 2:
            stage_compute((i - 2) % pipeline_depth)

    # Steady state
    for i in range(pipeline_depth, num_positions + pipeline_depth):
        idx = i % pipeline_depth

        # Write completed result
        if i >= pipeline_depth:
            stage_write((i - pipeline_depth) % pipeline_depth)

        # Load new position
        if i < num_positions:
            stage_load(idx, position_names[i])

        # Transfer previous
        if i >= 1 and (i - 1) < num_positions:
            stage_transfer((i - 1) % pipeline_depth)

        # Compute position from 2 stages ago
        if i >= 2 and (i - 2) < num_positions:
            stage_compute((i - 2) % pipeline_depth)

    # Drain pipeline
    for i in range(max(0, num_positions - pipeline_depth), num_positions):
        stage_write(i % pipeline_depth)

    # Cleanup
    io_thread_pool.shutdown(wait=True)  # Ensure all I/O threads complete
    input_dataset.close()
    output_dataset.close()
    tf_dataset.close()

    print(f"Reconstructed {num_positions} positions with CUDA streams (multi-threaded I/O)", file=sys.stderr)
