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

    A transfer function is only valid for the ZYX shape it was computed from,
    so this command groups the positions in `input-position-dirpaths` by their
    ZYX shape, calculates one transfer function per distinct shape, and applies
    each transfer function to the positions it was computed for.

    If any parameter has an `lr` key (OptimizableFloat), an optimization loop
    runs before the standard reconstruction pipeline.

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
    from waveorder.cli.utils import read_zyx_shapes
    from waveorder.io import utils
    from waveorder.optim import has_optimizable_params

    settings = utils.yaml_to_model(config_filepath, ReconstructionSettings)

    # Check for optimizable parameters and run optimization if needed
    if has_optimizable_params(settings):
        config_filepath = _run_optimization(settings, input_position_dirpaths[0], config_filepath)

    # Positions that share a ZYX shape can share a transfer function, so compute
    # one per distinct shape, using the first position that has that shape.
    position_zyx_shapes = read_zyx_shapes(input_position_dirpaths)
    first_position_by_zyx_shape: dict[tuple[int, ...], Path] = {}
    for zyx_shape, input_position_dirpath in zip(position_zyx_shapes, input_position_dirpaths, strict=True):
        first_position_by_zyx_shape.setdefault(zyx_shape, input_position_dirpath)

    if len(first_position_by_zyx_shape) > 1:
        click.echo(
            click.style(
                f"Found {len(first_position_by_zyx_shape)} distinct ZYX shapes across "
                f"{len(input_position_dirpaths)} positions: computing one transfer "
                "function per shape.",
                fg="yellow",
            )
        )

    transfer_function_paths = {}
    for zyx_shape, input_position_dirpath in first_position_by_zyx_shape.items():
        # Only disambiguate the filename when there is more than one shape, so
        # single-shape runs keep writing the path users already expect.
        suffix = "" if len(first_position_by_zyx_shape) == 1 else "_" + "x".join(str(size) for size in zyx_shape)
        transfer_function_path = output_dirpath.parent / f"transfer_function_{config_filepath.stem}{suffix}.zarr"
        compute_transfer_function_cli(input_position_dirpath, config_filepath, transfer_function_path)
        transfer_function_paths[zyx_shape] = transfer_function_path

    # Apply each position's matching inverse transfer function
    apply_inverse_transfer_function_cli(
        input_position_dirpaths,
        transfer_function_paths,
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
