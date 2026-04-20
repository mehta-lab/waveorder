"""Core benchmark runner: phantom → simulate → wo rec → metrics.

Each synthetic case shells out to ``wo rec`` so the benchmark proves
the exact CLI workflow. The ``cli_command.sh`` saved alongside each
case can be copy-pasted for production use.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

logging.getLogger("iohub").setLevel(logging.ERROR)
logging.getLogger("iohub.ngff").setLevel(logging.ERROR)
logging.getLogger("iohub.ngff.nodes").setLevel(logging.ERROR)

import numpy as np
import torch
import yaml
from iohub.ngff import open_ome_zarr
from iohub.ngff.models import TransformationMeta

from benchmarks.config import PhantomConfig
from benchmarks.metrics import compute_metrics
from benchmarks.simulate import simulate_fluorescence_3d, simulate_phase_3d
from benchmarks.utils import TimingTree
from waveorder import phantoms

# Map phantom function names to callables
_PHANTOM_FUNCTIONS = {
    "single_bead": phantoms.single_bead,
    "random_beads": phantoms.random_beads,
}


def _extract_optics(tf_settings: dict) -> tuple[float, float, float]:
    """Pull ``(NA_det, wavelength, pixel_size)`` from a transfer_function block.

    Falls back to values appropriate for a typical microscope when keys
    are missing. Covers both illumination and emission wavelength keys.
    """
    NA_det = tf_settings.get("numerical_aperture_detection", 1.2)
    wavelength = tf_settings.get(
        "wavelength_illumination",
        tf_settings.get("wavelength_emission", 0.532),
    )
    pixel_size = tf_settings.get("yx_pixel_size", 0.1)
    return NA_det, wavelength, pixel_size


def _run_wo_rec(position_path: Path, config_path: Path, recon_path: Path) -> None:
    """Invoke ``wo rec`` as a subprocess, raising on non-zero exit."""
    env = {**os.environ, "PYTHONWARNINGS": "ignore::UserWarning,ignore::DeprecationWarning"}
    result = subprocess.run(
        ["wo", "rec", "-i", str(position_path), "-c", str(config_path), "-o", str(recon_path)],
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"wo rec failed:\n{result.stderr}")


# Storage cap applied to per-case output zarrs. Small synthetic outputs
# (~4 MB for a bead phantom) pass through; large HPC 3D recons and big
# simulations are deleted after their metrics are computed. Override
# with ``save_all=True``.
SIZE_LIMIT_BYTES = 25 * 1024 * 1024


def _dir_size_bytes(path: Path) -> int:
    """Sum the sizes of all files under ``path`` (recursive)."""
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _cleanup_large_outputs(case_dir: Path, save_all: bool) -> None:
    """Remove transfer-function zarrs and oversized output zarrs.

    Transfer-function zarrs are always deleted when ``save_all`` is
    False — they can be regenerated from config and are typically the
    largest intermediate. ``simulated.zarr`` and ``reconstruction.zarr``
    are deleted only if they exceed :data:`SIZE_LIMIT_BYTES`.
    """
    if save_all:
        return
    for p in case_dir.glob("transfer_function_*.zarr"):
        if p.is_dir():
            shutil.rmtree(p)
    for name in ("simulated.zarr", "reconstruction.zarr"):
        p = case_dir / name
        if p.is_dir() and _dir_size_bytes(p) > SIZE_LIMIT_BYTES:
            shutil.rmtree(p)


def _build_phantom(phantom_config: PhantomConfig | dict) -> phantoms.Phantom:
    """Build a phantom from a config.

    Parameters
    ----------
    phantom_config : PhantomConfig or dict
        Phantom configuration. If dict, will be validated as PhantomConfig.

    Returns
    -------
    Phantom
        Generated phantom.
    """
    if isinstance(phantom_config, dict):
        phantom_config = PhantomConfig(**phantom_config)
    kwargs = phantom_config.model_dump()
    func_name = kwargs.pop("function")
    func = _PHANTOM_FUNCTIONS[func_name]
    return func(**kwargs)


def _write_zarr(data: torch.Tensor, path: Path, channel_name: str, pixel_sizes: tuple[float, float, float]):
    """Write a ZYX tensor as an HCS OME-Zarr readable by ``wo rec``.

    Parameters
    ----------
    data : Tensor
        3D volume (Z, Y, X).
    path : Path
        Output zarr path.
    channel_name : str
        Name for the single channel.
    pixel_sizes : tuple[float, float, float]
        (z, y, x) pixel sizes.
    """
    czyx = data.numpy()[None, ...]  # (1, Z, Y, X)
    dataset = open_ome_zarr(path, layout="hcs", mode="w", channel_names=[channel_name])
    position = dataset.create_position("0", "0", "0")
    position.create_zeros(
        "0",
        (1, *czyx.shape),
        dtype=np.float32,
        transform=[
            TransformationMeta(
                type="scale",
                scale=[1, 1, float(pixel_sizes[0]), float(pixel_sizes[1]), float(pixel_sizes[2])],
            )
        ],
    )
    position["0"][0] = czyx
    dataset.close()


def _read_zarr(path: Path) -> torch.Tensor:
    """Read first position, first timepoint from an HCS OME-Zarr.

    Parameters
    ----------
    path : Path
        Path to OME-Zarr.

    Returns
    -------
    Tensor
        Volume as (Z, Y, X) or (C, Z, Y, X) float32 tensor.
    """
    dataset = open_ome_zarr(path, layout="hcs", mode="r")
    position = list(dataset.positions())[0][1]
    data = torch.tensor(np.array(position["0"][0]), dtype=torch.float32)
    dataset.close()
    return data


def run_synthetic_case(
    phantom_config: dict,
    recon_config: dict,
    case_dir: Path,
    modality: str = "phase",
    save_all: bool = False,
) -> dict:
    """Run a single synthetic benchmark case via ``wo rec``.

    Parameters
    ----------
    phantom_config : dict
        Phantom generation parameters (see experiment YAML).
    recon_config : dict
        Reconstruction config dict (parsed from YAML).
    case_dir : Path
        Output directory for this case.
    modality : str
        "phase" or "fluorescence".
    save_all : bool
        Keep all intermediate outputs. When False (default), transfer
        function zarrs are deleted and ``simulated.zarr`` /
        ``reconstruction.zarr`` larger than
        :data:`SIZE_LIMIT_BYTES` are deleted after metrics are computed.

    Returns
    -------
    dict
        Metrics dict from compute_metrics.
    """
    case_dir.mkdir(parents=True, exist_ok=True)
    tt = TimingTree()

    with tt.time("total"):
        # Generate phantom
        with tt.time("phantom"):
            phantom = _build_phantom(phantom_config)

        (case_dir / "phantom_config.json").write_text(json.dumps(phantom.metadata, indent=2))

        # Simulate measurement
        with tt.time("simulate"):
            tf_settings = recon_config.get(modality, {}).get("transfer_function", {})
            if modality == "phase":
                data = simulate_phase_3d(
                    phantom,
                    wavelength_illumination=tf_settings.get("wavelength_illumination", 0.532),
                    index_of_refraction_media=tf_settings.get("index_of_refraction_media", 1.3),
                    numerical_aperture_illumination=tf_settings.get("numerical_aperture_illumination", 0.9),
                    numerical_aperture_detection=tf_settings.get("numerical_aperture_detection", 1.2),
                )
                channel_name = recon_config.get("input_channel_names", ["Brightfield"])[0]
            elif modality == "fluorescence":
                data = simulate_fluorescence_3d(
                    phantom,
                    wavelength_emission=tf_settings.get("wavelength_emission", 0.532),
                    index_of_refraction_media=tf_settings.get("index_of_refraction_media", 1.3),
                    numerical_aperture_detection=tf_settings.get("numerical_aperture_detection", 1.2),
                )
                channel_name = recon_config.get("input_channel_names", ["GFP"])[0]
            else:
                raise ValueError(f"Unknown modality: {modality}")

        # Write simulated data as OME-Zarr
        simulated_path = case_dir / "simulated.zarr"
        _write_zarr(data, simulated_path, channel_name, phantom.pixel_sizes)

        # Write reconstruction config
        config_path = case_dir / "config.yml"
        config_path.write_text(yaml.dump(recon_config, default_flow_style=False))

        # Build and save CLI command
        position_path = simulated_path / "0" / "0" / "0"
        recon_path = case_dir / "reconstruction.zarr"
        cli_cmd = f"wo rec -i {position_path} -c {config_path} -o {recon_path}"
        (case_dir / "cli_command.sh").write_text(f"#!/bin/bash\n{cli_cmd}\n")

        # Reconstruct via wo rec
        with tt.time("reconstruct"):
            _run_wo_rec(position_path, config_path, recon_path)

        # Load reconstruction and compute metrics
        with tt.time("metrics"):
            recon_data = _read_zarr(recon_path)
            # Remove batch dims — get to (Z, Y, X)
            while recon_data.ndim > 3:
                recon_data = recon_data[0]

            NA_det, wavelength, pixel_size = _extract_optics(tf_settings)
            ground_truth = phantom.phase if modality == "phase" else phantom.fluorescence

            metrics = compute_metrics(
                recon_data,
                NA_det=NA_det,
                wavelength=wavelength,
                pixel_size=pixel_size,
                phantom=ground_truth,
            )

    tt.save(case_dir / "timing.json")
    (case_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _cleanup_large_outputs(case_dir, save_all)

    return metrics


def run_hpc_case(
    input_path: str,
    position: str,
    recon_config: dict,
    case_dir: Path,
    save_all: bool = False,
) -> dict:
    """Run a single HPC benchmark case on existing data via ``wo rec``.

    Parameters
    ----------
    input_path : str
        Path to input OME-Zarr store.
    position : str
        Position key within the store (e.g. "A/1/029029").
    recon_config : dict
        Reconstruction config dict.
    case_dir : Path
        Output directory for this case.
    save_all : bool
        Keep all intermediate outputs. When False (default), the
        transfer function zarr is deleted and ``reconstruction.zarr``
        is deleted if it exceeds :data:`SIZE_LIMIT_BYTES` after metrics
        are computed.

    Returns
    -------
    dict
        Metrics dict from compute_metrics.
    """
    case_dir.mkdir(parents=True, exist_ok=True)
    tt = TimingTree()

    # Determine modality and optics from config
    modality = "phase" if "phase" in recon_config else "fluorescence"
    tf_settings = recon_config.get(modality, {}).get("transfer_function", {})

    with tt.time("total"):
        # Write reconstruction config
        config_path = case_dir / "config.yml"
        config_path.write_text(yaml.dump(recon_config, default_flow_style=False))

        # Build CLI command
        position_path = Path(input_path) / position
        recon_path = case_dir / "reconstruction.zarr"
        cli_cmd = f"wo rec -i {position_path} -c {config_path} -o {recon_path}"
        (case_dir / "cli_command.sh").write_text(f"#!/bin/bash\n{cli_cmd}\n")

        # Reconstruct via wo rec
        with tt.time("reconstruct"):
            _run_wo_rec(position_path, config_path, recon_path)

        # Load reconstruction and compute metrics (image_quality only, no phantom)
        with tt.time("metrics"):
            recon_data = _read_zarr(recon_path)
            while recon_data.ndim > 3:
                recon_data = recon_data[0]

            NA_det, wavelength, pixel_size = _extract_optics(tf_settings)

            metrics = compute_metrics(
                recon_data,
                NA_det=NA_det,
                wavelength=wavelength,
                pixel_size=pixel_size,
            )

    tt.save(case_dir / "timing.json")
    (case_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _cleanup_large_outputs(case_dir, save_all)

    return metrics
