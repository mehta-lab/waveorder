from pathlib import Path
from typing import Annotated, Optional

import typer

from waveorder.cli.parsing import (
    ConfigFilepath,
    InputPositionDirpaths,
    OutputDirpath,
)

PriorTransferFunctionDirpath = Annotated[
    Optional[Path],
    typer.Option(
        "--transfer-function-dirpath",
        "-t",
        help="tf.zarr whose parameters (and covariance, if non-zero) set the prior on the "
        "estimated parameters, overriding the configuration's tolerances for the parameters "
        "it covers.",
    ),
]
InitTransferFunctionDirpath = Annotated[
    Optional[Path],
    typer.Option(
        "--init",
        help="tf.zarr whose parameters set the starting point of the search without changing the prior.",
    ),
]


def calibrate_transfer_function_cli(
    input_position_dirpath: Path,
    config_filepath: Path,
    output_dirpath: Path,
    prior_tf_dirpath: Path | None = None,
    init_tf_dirpath: Path | None = None,
) -> None:
    """Estimate the free imaging-system parameters from one position.

    Writes a ``tf.zarr`` with ``parameters``, ``parameters_covariance``,
    ``parameter_names``, ``tile_grid`` and ``provenance`` (design document
    section 3).
    """
    # Deferred imports: these pull in torch, iohub, numpy, etc.
    # Only loaded when the command runs, keeping wo calibrate-tf -h fast.
    import hashlib
    import warnings

    import numpy as np
    import yaml
    from iohub.ngff import open_ome_zarr

    from waveorder.calibrate.beads import run_bead_calibration
    from waveorder.calibrate.evidence_search import run_evidence_calibration
    from waveorder.calibrate.run import (
        run_zernike_calibration,
        zernike_calibration_requested,
    )
    from waveorder.calibrate.synthesis import fluorescence_otf
    from waveorder.calibrate.tfzarr import read_tf_zarr, write_tf_zarr
    from waveorder.calibrate.zernike_names import name_to_noll
    from waveorder.cli.printing import echo_headline
    from waveorder.cli.utils import resolve_time_indices
    from waveorder.config.loader import load_v4_config
    from waveorder.config.migrate import detect_version, migrate_3x_dict
    from waveorder.config.schema import PriorBeads, free_parameters

    config_text = Path(config_filepath).read_text()
    raw = yaml.safe_load(config_text)
    if detect_version(raw) >= 4:
        cfg = load_v4_config(config_filepath)
    else:
        warnings.warn(
            f"{config_filepath} is a 3.x configuration; converting in memory. "
            f"Convert it with: waveorder migrate-config {config_filepath}",
            DeprecationWarning,
            stacklevel=2,
        )
        cfg = migrate_3x_dict(raw)

    if not free_parameters(cfg.imaging_system):
        raise ValueError(
            "calibrate-tf with no tolerances: nothing to estimate. Give at least one "
            "imaging_system number as {nominal: x, tolerance: t}."
        )
    if not zernike_calibration_requested(cfg):
        raise NotImplementedError(
            "calibrate-tf currently estimates pupil_aberrations terms; estimate "
            "numerical_aperture / tilt / z_focus_offset through 'wo reconstruct' "
            "with solver.calibration for now."
        )

    echo_headline(f"Calibrating transfer function from {input_position_dirpath}\n")

    input_dataset = open_ome_zarr(input_position_dirpath, layout="fov", mode="r")
    try:
        if not set(cfg.data.input_channel_names).issubset(input_dataset.channel_names):
            raise ValueError(
                f"input_channel_names = {cfg.data.input_channel_names} must appear in "
                f"{input_position_dirpath} (channels: {input_dataset.channel_names})"
            )
        t_index = resolve_time_indices(cfg.data.time_indices, input_dataset.data.shape[0])[0]
        czyx = input_dataset.to_xarray().isel(t=t_index).sel(c=cfg.data.input_channel_names)
        zyx_data = np.asarray(czyx.values[0], dtype=np.float32)
    finally:
        input_dataset.close()

    calibration = cfg.solver.calibration
    score = calibration.score if calibration is not None else "evidence"
    extra_provenance = {}
    score_arrays = {}
    covariance = None
    provenance_kind = "refined"
    prior_store = read_tf_zarr(prior_tf_dirpath) if prior_tf_dirpath is not None else None
    init_store = read_tf_zarr(init_tf_dirpath) if init_tf_dirpath is not None else None
    if score not in ("evidence", "gcv") and (prior_store is not None or init_store is not None):
        raise NotImplementedError("-t / --init are supported with score: evidence or gcv for now")
    if isinstance(cfg.object.prior, PriorBeads):
        if prior_store is not None or init_store is not None:
            raise NotImplementedError("-t / --init with prior: beads is not supported yet")
        bead_result = run_bead_calibration(cfg, zyx_data)
        parameters = bead_result.parameters
        parameter_names = bead_result.parameter_names
        covariance = bead_result.parameters_covariance
        tile_grid = {
            "size_yx": [int(zyx_data.shape[1]), int(zyx_data.shape[2])],
            "overlap_yx": [0, 0],
            "centers_yx": [[0.0, 0.0]],
        }
        extra_provenance = {
            "best_score": bead_result.best_score,
            "noise_sigma": bead_result.noise_sigma,
            "background": bead_result.background,
            "n_beads": int(bead_result.positions_zyx.shape[0]),
        }
        provenance_kind = "beads"
        if "score_trace" in (cfg.outputs if cfg.outputs != "all" else ["score_trace"]):
            score_arrays["score_trace"] = np.asarray(bead_result.score_trace)
        score_arrays["bead_positions_zyx"] = bead_result.positions_zyx
        score_arrays["bead_amplitudes"] = bead_result.amplitudes
        final_loss = bead_result.best_score
    elif score in ("evidence", "gcv"):
        evidence_result = run_evidence_calibration(cfg, zyx_data, prior_store=prior_store, init_store=init_store)
        parameters = evidence_result.parameters
        parameter_names = evidence_result.parameter_names
        covariance = evidence_result.parameters_covariance
        tile_grid = {
            "size_yx": [int(zyx_data.shape[1]), int(zyx_data.shape[2])],
            "overlap_yx": [0, 0],
            "centers_yx": [[0.0, 0.0]],
        }
        extra_provenance = {
            "best_score": evidence_result.best_score,
            "noise_sigma": evidence_result.noise_sigma,
            "weight": evidence_result.weight,
        }
        wants = cfg.outputs if cfg.outputs != "all" else ["score_trace", "score_grid"]
        if "score_trace" in wants and evidence_result.score_trace:
            score_arrays["score_trace"] = np.asarray(evidence_result.score_trace)
        if "score_grid" in wants and evidence_result.score_grid is not None:
            score_arrays["score_grid"] = evidence_result.score_grid
        final_loss = evidence_result.best_score
    elif score == "sharpness":
        sharpness_result = run_zernike_calibration(cfg, zyx_data)
        parameters = sharpness_result.parameters
        parameter_names = sharpness_result.parameter_names
        tile_grid = sharpness_result.tile_grid
        final_loss = sharpness_result.loss_history[-1]
    else:
        raise NotImplementedError(f"score: {score} is not implemented in calibrate-tf")

    # Shift-invariant stores always carry the transfer function; tiled stores
    # only on request, and are otherwise regenerated per tile at apply time.
    transfer_function = None
    n_tiles = parameters.shape[0]
    wants_tf = cfg.outputs == "all" or "transfer_function" in cfg.outputs
    if n_tiles == 1 and wants_tf:
        detection = cfg.imaging_system.detection
        yx = cfg.data.pixel_size.yx
        coefficients = {name_to_noll(name): value for name, value in zip(parameter_names, parameters[0])}
        transfer_function = (
            fluorescence_otf(
                zyx_shape=zyx_data.shape,
                yx_pixel_size=float(yx) if isinstance(yx, (int, float)) else float(yx.y),
                z_pixel_size=cfg.data.pixel_size.z,
                wavelength_emission=detection.wavelength,
                index_of_refraction_media=cfg.imaging_system.index_of_refraction_media,
                numerical_aperture_detection=(
                    detection.numerical_aperture.nominal
                    if hasattr(detection.numerical_aperture, "nominal")
                    else detection.numerical_aperture
                ),
                zernike_coefficients=coefficients,
            )
            .numpy()
            .astype(np.complex64)
        )

    provenance = {
        "kind": provenance_kind,
        "sources": [str(input_position_dirpath)],
        "config_hash": hashlib.sha256(config_text.encode()).hexdigest(),
        "score": score,
        "time_index": int(t_index),
        **extra_provenance,
    }
    write_tf_zarr(
        output_dirpath,
        parameters=parameters,
        parameter_names=parameter_names,
        tile_grid=tile_grid,
        provenance=provenance,
        parameters_covariance=covariance,
        transfer_function=transfer_function,
        extra_arrays=score_arrays,
    )

    echo_headline(
        f"Estimated {len(parameter_names)} parameters on {n_tiles} tile(s); final {score} score {final_loss:.4g}"
    )
    if n_tiles == 1:
        for name, value in zip(parameter_names, parameters[0]):
            print(f"    {name:16s} {value:+.4f} waves")
    echo_headline(f"Wrote {output_dirpath}")


def _calibrate_tf_cli(
    input_position_dirpaths: InputPositionDirpaths,
    config_filepath: ConfigFilepath,
    output_dirpath: OutputDirpath,
    prior_tf_dirpath: PriorTransferFunctionDirpath = None,
    init_tf_dirpath: InitTransferFunctionDirpath = None,
) -> None:
    """Estimate free imaging-system parameters (tolerances) from data.

    Every imaging_system number written as {nominal: x, tolerance: t} is
    estimated from the first position in `input-position-dirpaths`, and the
    result is written as a tf.zarr with `parameters`, `parameters_covariance`,
    `parameter_names`, `tile_grid`, and `provenance`.

    A tf.zarr passed with `-t` sets the prior: its parameters move the prior
    center, and its covariance (when non-zero) replaces the configuration's
    tolerances for the parameters it covers. `--init` sets the starting point
    only.

    \b
    Example:
      \033[92mwo calibrate-tf -i ./input.zarr/0/0/0 -c ./config.yml -o ./tf_hat.zarr\033[0m
    """
    if len(input_position_dirpaths) > 1:
        raise NotImplementedError(
            "calibrate-tf with several positions (one shared H) is not implemented yet; pass one position."
        )
    calibrate_transfer_function_cli(
        input_position_dirpaths[0],
        config_filepath,
        output_dirpath,
        prior_tf_dirpath=prior_tf_dirpath,
        init_tf_dirpath=init_tf_dirpath,
    )
