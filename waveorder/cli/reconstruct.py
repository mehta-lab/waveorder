from pathlib import Path

import click

from waveorder.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    output_dirpath,
    processes_option,
    unique_id,
    write_config_scale_to_output,
)


@click.command("reconstruct", no_args_is_help=True)
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@processes_option(default=1)
@unique_id()
@write_config_scale_to_output()
def _reconstruct_cli(
    input_position_dirpaths,
    config_filepath,
    output_dirpath,
    num_processes,
    unique_id,
    write_config_scale_to_output,
):
    """
    Reconstruct a dataset using a configuration file. This is a
    convenience function for a `compute-tf` call followed by a `apply-inv-tf`
    call.

    Calculates the transfer function based on the shape of the first position
    in the list `input-position-dirpaths`, then applies that transfer function
    to all positions in the list `input-position-dirpaths`, so all positions
    must have the same TCZYX shape.

    If any parameter has an `lr` key (OptimizableFloat), an optimization loop
    runs before the standard reconstruction pipeline.

    If `apply_inverse` has an `auto_regularization` block, `regularization_strength`
    is chosen from the data after the transfer function is computed, and the
    resolved settings are written alongside the config.

    See /examples for example configuration files.

    \b
    Example:
      \033[92mwo rec -i ./input.zarr/*/*/* -c ./config.yml -o ./output.zarr\033[0m
    """
    click.echo(click.style("Starting reconstruction...", fg="green"))

    # Deferred imports: these pull in torch, iohub, numpy, etc.
    # Only loaded when the command runs, keeping wo rec -h fast.
    from waveorder.cli.apply_inverse_transfer_function import apply_inverse_transfer_function_cli
    from waveorder.cli.compute_transfer_function import compute_transfer_function_cli
    from waveorder.cli.settings import ReconstructionSettings
    from waveorder.io import utils
    from waveorder.optim import has_optimizable_params

    settings = utils.yaml_to_model(config_filepath, ReconstructionSettings)

    # Check for optimizable parameters and run optimization if needed
    if has_optimizable_params(settings):
        config_filepath = _run_optimization(settings, input_position_dirpaths[0], config_filepath)

    # Handle transfer function path
    transfer_function_path = output_dirpath.parent / Path("transfer_function_" + config_filepath.stem + ".zarr")

    # Compute transfer function
    compute_transfer_function_cli(
        input_position_dirpaths[0],
        config_filepath,
        transfer_function_path,
    )

    # Choose regularization_strength from the data, if asked to. This runs after
    # the transfer function (which it needs, and which does not depend on
    # `apply_inverse`) and before the reconstruction, so the sweep happens once
    # rather than per timepoint inside the process pool.
    if _has_auto_regularization(settings):
        config_filepath = _run_auto_regularization(
            settings,
            input_position_dirpaths[0],
            config_filepath,
        )

    # Apply inverse transfer function
    apply_inverse_transfer_function_cli(
        input_position_dirpaths,
        transfer_function_path,
        config_filepath,
        output_dirpath,
        num_processes,
        write_config_scale_to_output,
    )


def _run_optimization(settings, input_position_dirpath, config_filepath):
    """Run parameter optimization before standard reconstruction."""
    # Deferred imports for the same reason as above
    from iohub.ngff import open_ome_zarr

    from waveorder.api import fluorescence, phase
    from waveorder.cli.settings import OptimizationSettings
    from waveorder.cli.utils import resolve_time_indices
    from waveorder.io import utils

    if settings.birefringence is not None:
        raise NotImplementedError("Parameter optimization is not supported for birefringence reconstructions.")

    opt = settings.optimization if settings.optimization else OptimizationSettings()

    input_dataset = open_ome_zarr(input_position_dirpath, layout="fov", mode="r")
    t_idx = resolve_time_indices(settings.time_indices, input_dataset.data.shape[0])[0]
    czyx_data = input_dataset.to_xarray().isel(t=t_idx)

    recon_dim = settings.reconstruction_dimension
    optimize_kwargs = dict(
        recon_dim=recon_dim,
        max_iterations=opt.max_iterations,
        method=opt.method,
        convergence_tol=opt.convergence_tol,
        convergence_patience=opt.convergence_patience,
        use_gradients=opt.use_gradients,
        grid_points=opt.grid_points,
        loss_settings=opt.loss,
        log_dir=opt.log_dir,
        device=settings.device,
    )

    if settings.phase is not None:
        settings.phase, _ = phase.optimize(czyx_data, settings=settings.phase, **optimize_kwargs)
    elif settings.fluorescence is not None:
        settings.fluorescence, _ = fluorescence.optimize(czyx_data, settings=settings.fluorescence, **optimize_kwargs)

    optimized_path = config_filepath.parent / (config_filepath.stem + "_optimized.yml")
    utils.model_to_yaml(settings, optimized_path)
    print(f"Optimized settings saved to {optimized_path}")

    input_dataset.close()
    return optimized_path


def _has_auto_regularization(settings) -> bool:
    """Whether any reconstruction block asks for an auto-regularization sweep."""
    return any(
        block is not None and block.apply_inverse.auto_regularization is not None
        for block in (settings.phase, settings.fluorescence)
    )


def _run_auto_regularization(settings, input_position_dirpath, config_filepath):
    """Sweep regularization_strength, then write the resolved settings to a new config.

    Returns the path of the resolved config, which the reconstruction then uses.
    """
    # Deferred imports for the same reason as above
    import json

    import torch
    from iohub.ngff import open_ome_zarr

    from waveorder.api import fluorescence, phase
    from waveorder.cli.printing import echo_headline
    from waveorder.cli.utils import resolve_time_indices
    from waveorder.device import resolve_device
    from waveorder.io import utils
    from waveorder.optim import autoreg

    if settings.phase is not None and settings.phase.apply_inverse.auto_regularization is not None:
        contrast, block, api_module = "phase", settings.phase, phase
        tf_key = "real_potential_transfer_function"
    else:
        contrast, block, api_module = "fluorescence", settings.fluorescence, fluorescence
        tf_key = "optical_transfer_function"

    auto_settings = block.apply_inverse.auto_regularization
    device = resolve_device(settings.device)

    echo_headline(f"\nChoosing {contrast} regularization_strength with the '{auto_settings.rule}' rule...")

    input_dataset = open_ome_zarr(input_position_dirpath, layout="fov", mode="r")
    try:
        # Score the first timepoint the config asks for, so `time_indices: [5, 6]`
        # is scored on 5 without a setting of its own.
        t_idx = resolve_time_indices(settings.time_indices, input_dataset.data.shape[0])[0]
        czyx = input_dataset.to_xarray().isel(t=t_idx).sel(c=settings.input_channel_names)

        y0, x0, size = autoreg.select_crop(torch.as_tensor(czyx.values[0]))
        czyx_crop = czyx.isel(y=slice(y0, y0 + size), x=slice(x0, x0 + size))
    finally:
        input_dataset.close()

    # The transfer function cannot be cropped, only recomputed at the crop's shape.
    tf_crop = api_module.compute_transfer_function(
        czyx_crop,
        recon_dim=settings.reconstruction_dimension,
        settings=block,
        device=device,
    )

    result = autoreg.select_regularization(
        torch.tensor(czyx_crop.values[0], dtype=torch.float32, device=device),
        torch.as_tensor(tf_crop[tf_key].values).to(device),
        auto_settings,
        contrast=contrast,
        z_padding=block.transfer_function.z_padding,
        crop=(y0, x0, size),
        apodization_rolloff=block.apply_inverse.apodization_rolloff,
        # No full-frame peak is passed: |H|^2max is set by the optics and the pixel
        # size, not by the frame size, so the crop's own peak is the same number.
        # Reading it back off the full transfer function measured 0.000000 decades
        # of difference on a 2048x2048 frame and cost 10 s against a 0.4 s sweep.
    )
    autoreg.warn_all(result)

    click.echo(
        click.style(
            f"    regularization_strength = {result.regularization_strength:.4g}\n"
            f"    lambda / |H|^2max       = {result.lambda_over_h2max:.4g}\n"
            f"    sweep index             = {result.index} of {len(result.regularization_strengths)}\n"
            f"    scored crop             = {size}x{size} at (y={y0}, x={x0}), t={t_idx}",
            fg="green",
        )
    )

    if auto_settings.report_path is not None:
        report_path = Path(auto_settings.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"Auto-regularization report saved to {report_path}")

    # Freeze the pick into a sibling config, so the reconstruction that follows is
    # reproducible without re-running the sweep.
    block.apply_inverse.regularization_strength = result.regularization_strength
    block.apply_inverse.auto_regularization = None

    resolved_path = config_filepath.parent / (config_filepath.stem + "_autoreg.yml")
    utils.model_to_yaml(settings, resolved_path)
    print(f"Auto-regularized settings saved to {resolved_path}")

    return resolved_path
