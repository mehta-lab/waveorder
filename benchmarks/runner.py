"""Core benchmark runner: phantom → simulate → wo rec → metrics.

Each synthetic case shells out to ``wo rec`` so the benchmark proves
the exact CLI workflow. The ``cli_command.sh`` saved alongside each
case can be copy-pasted for production use.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml
from iohub.ngff import open_ome_zarr
from iohub.ngff.models import TransformationMeta

from benchmarks.config import load_experiment
from benchmarks.metrics import compute_metrics
from benchmarks.simulate import simulate_fluorescence_3d, simulate_phase_3d
from benchmarks.utils import TimingTree, collect_metadata
from waveorder import phantoms

# Map phantom function names to callables
_PHANTOM_FUNCTIONS = {
    "single_bead": phantoms.single_bead,
    "random_beads": phantoms.random_beads,
}


def _build_phantom(phantom_config: dict) -> phantoms.Phantom:
    """Build a phantom from a config dict.

    Parameters
    ----------
    phantom_config : dict
        Must include "function" key. Remaining keys are passed
        as kwargs to the phantom function.

    Returns
    -------
    Phantom
        Generated phantom.
    """
    config = dict(phantom_config)
    func_name = config.pop("function")
    func = _PHANTOM_FUNCTIONS[func_name]
    for key in ("shape", "pixel_sizes"):
        if key in config:
            config[key] = tuple(config[key])
    return func(**config)


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
            result = subprocess.run(
                ["wo", "rec", "-i", str(position_path), "-c", str(config_path), "-o", str(recon_path)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"wo rec failed:\n{result.stderr}")

        # Load reconstruction and compute metrics
        with tt.time("metrics"):
            recon_data = _read_zarr(recon_path)
            # Remove batch dims — get to (Z, Y, X)
            while recon_data.ndim > 3:
                recon_data = recon_data[0]

            NA_det = tf_settings.get("numerical_aperture_detection", 1.2)
            wavelength = tf_settings.get(
                "wavelength_illumination",
                tf_settings.get("wavelength_emission", 0.532),
            )
            pixel_size = tf_settings.get("yx_pixel_size", 0.1)

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

    return metrics


def run_experiment(
    experiment_path: str | Path,
    output_dir: str | Path = ".",
) -> dict:
    """Run all cases in an experiment.

    Parameters
    ----------
    experiment_path : str or Path
        Path to experiment YAML.
    output_dir : str or Path
        Root output directory.

    Returns
    -------
    dict
        Per-case metrics keyed by case name.
    """
    experiment = load_experiment(experiment_path)
    output_dir = Path(output_dir)

    metadata = collect_metadata()
    run_name = f"{metadata['git_hash']}_{experiment.get('name', 'experiment')}"
    run_dir = output_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    results = {}
    for case_name, case in experiment.get("cases", {}).items():
        case_type = case.get("type", "synthetic")
        if case_type != "synthetic":
            print(f"Skipping non-synthetic case: {case_name}")
            continue

        case_dir = run_dir / "cases" / case_name

        # Resolve config
        recon_config = case.get("_resolved_config", {})
        if not recon_config and "config" in case:
            config_path = Path(case["config"])
            if not config_path.is_absolute():
                config_path = Path(experiment_path).parent / config_path
            with open(config_path) as f:
                recon_config = yaml.safe_load(f)

        modality = "phase" if "phase" in recon_config else "fluorescence"

        results[case_name] = run_synthetic_case(
            phantom_config=case["phantom"],
            recon_config=recon_config,
            case_dir=case_dir,
            modality=modality,
        )

    (run_dir / "summary.json").write_text(json.dumps(results, indent=2))

    return results
