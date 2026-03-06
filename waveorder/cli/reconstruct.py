from pathlib import Path

import click
from iohub.ngff import open_ome_zarr

from waveorder.api import fluorescence, phase
from waveorder.cli.apply_inverse_transfer_function import (
    apply_inverse_transfer_function_cli,
)
from waveorder.cli.compute_transfer_function import (
    compute_transfer_function_cli,
)
from waveorder.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    output_dirpath,
    processes_option,
    unique_id,
)
from waveorder.cli.settings import OptimizationSettings, ReconstructionSettings
from waveorder.cli.utils import resolve_time_indices
from waveorder.io import utils
from waveorder.optim import has_optimizable_params


@click.command("reconstruct")
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@processes_option(default=1)
@unique_id()
def _reconstruct_cli(
    input_position_dirpaths,
    config_filepath,
    output_dirpath,
    num_processes,
    unique_id,
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

    See /examples for example configuration files.

    >> waveorder reconstruct -i ./input.zarr/*/*/* -c ./examples/birefringence.yml -o ./output.zarr
    """
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

    # Apply inverse transfer function
    apply_inverse_transfer_function_cli(
        input_position_dirpaths,
        transfer_function_path,
        config_filepath,
        output_dirpath,
        num_processes,
    )


def _run_optimization(settings, input_position_dirpath, config_filepath):
    """Run parameter optimization before standard reconstruction."""
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
