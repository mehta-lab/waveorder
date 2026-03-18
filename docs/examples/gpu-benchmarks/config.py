"""Configuration, GPU detection, and CLI argument parsing."""

import argparse

import torch

from waveorder.api import phase


def get_gpu_info() -> dict:
    """Return GPU name and VRAM. Works on any CUDA device."""
    if not torch.cuda.is_available():
        return {"name": "CPU", "vram_gb": 0}
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "vram_gb": round(props.total_memory / 1e9, 1),
    }


def make_settings() -> phase.Settings:
    """Phase 2D reconstruction settings matching 20x OPS pipeline."""
    return phase.Settings(
        transfer_function=phase.TransferFunctionSettings(
            wavelength_illumination=0.45,
            yx_pixel_size=0.325,
            z_pixel_size=2.0,
            z_focus_offset=0,
            numerical_aperture_illumination=0.4,
            numerical_aperture_detection=0.55,
            index_of_refraction_media=1.0,
            invert_phase_contrast=False,
        ),
        apply_inverse=phase.ApplyInverseSettings(
            reconstruction_algorithm="Tikhonov",
            regularization_strength=0.001,
        ),
    )


# 20x OPS physics params as a dict (for direct model-level calls)
PHYSICS_20X = {
    "wavelength_illumination": 0.45,
    "yx_pixel_size": 0.325,
    "z_pixel_size": 2.0,
    "index_of_refraction_media": 1.0,
    "numerical_aperture_illumination": 0.4,
    "numerical_aperture_detection": 0.55,
    "invert_phase_contrast": False,
    "regularization_strength": 0.001,
    "pupil_steepness": 100.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark I/O + GPU reconstruction throughput",
    )
    parser.add_argument("input_zarr", help="Path to input OME-Zarr store")
    parser.add_argument(
        "--position", default="A/1/002026", help="Position path within zarr",
    )
    parser.add_argument(
        "--device", default="auto", help="Device: auto, cpu, cuda:0",
    )
    parser.add_argument(
        "--tile-sizes",
        nargs="+",
        type=int,
        default=[128, 256, 512, 1024],
        help="Tile sizes to sweep",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 4, 16, 64, 256],
        help="Batch sizes to sweep",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Read full FOV once into memory, then slice tiles from numpy "
             "(vs per-tile zarr oindex reads)",
    )
    parser.add_argument(
        "--opt-iterations",
        type=int,
        default=50,
        help="Max iterations for the optimization loop benchmark",
    )
    parser.add_argument(
        "--forward-backward",
        action="store_true",
        help="Run forward/backward per-iteration profiling with substage breakdown",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run torch.profiler trace on a representative forward/backward pass. "
             "Exports Chrome trace to profile_trace_<tile>_<batch>.json",
    )
    parser.add_argument(
        "--nsys",
        action="store_true",
        help="Run nsys-profiled optimization iterations. "
             "Exports .nsys-rep file viewable in Nsight Systems GUI. "
             "Requires: ml load nsight/2025.3.1",
    )

    # Stage selection flags
    parser.add_argument(
        "--stages",
        action="store_true",
        help="Run per-stage breakdown (zarr read, H2D, TF, reconstruct, D2H, etc.)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run per-tile optimization benchmark",
    )
    parser.add_argument(
        "--optimize-batch",
        action="store_true",
        help="Run batched optimization benchmark (B tiles with per-tile params)",
    )
    parser.add_argument(
        "--svd-backend",
        choices=["closed_form", "torch", "both"],
        default="both",
        help="SVD backend for forward/backward benchmarks. 'both' runs both for comparison.",
    )

    args = parser.parse_args()

    # Default: if no flags set, run everything (except profile/nsys)
    if not any([args.stages, args.optimize, args.optimize_batch, args.forward_backward, args.profile, args.nsys]):
        args.stages = True
        args.optimize = True
        args.optimize_batch = True
        args.forward_backward = True
    args.device = resolve_device(args.device)
    return args


def resolve_device(device: str) -> str:
    """Resolve 'auto' to the best available torch device string."""
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device
