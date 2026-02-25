from pathlib import Path

import click

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
    from iohub.ngff import open_ome_zarr

    from waveorder.api import fluorescence, phase
    from waveorder.io import utils

    print("Detected optimizable parameters. Running optimization...")

    if settings.birefringence is not None:
        raise NotImplementedError("Parameter optimization is not supported for birefringence reconstructions.")

    # Load data
    input_dataset = open_ome_zarr(input_position_dirpath, layout="fov", mode="r")
    czyx_data = input_dataset.to_xarray().isel(t=0)

    opt_settings = settings.optimization
    num_iterations = opt_settings.num_iterations if opt_settings else 10
    log_dir = opt_settings.log_dir if opt_settings else None
    midband_fractions = (0.125, 0.25)
    if opt_settings and opt_settings.loss:
        midband_fractions = opt_settings.loss.midband_fractions

    recon_dim = settings.reconstruction_dimension

    if settings.phase is not None:
        new_settings, _ = phase.optimize(
            czyx_data,
            recon_dim=recon_dim,
            settings=settings.phase,
            num_iterations=num_iterations,
            midband_fractions=midband_fractions,
            log_dir=log_dir,
        )
        settings.phase = new_settings
        print("Phase optimization complete. Updated settings.")

    elif settings.fluorescence is not None:
        new_settings, _ = fluorescence.optimize(
            czyx_data,
            recon_dim=recon_dim,
            settings=settings.fluorescence,
            num_iterations=num_iterations,
            midband_fractions=midband_fractions,
            log_dir=log_dir,
        )
        settings.fluorescence = new_settings
        print("Fluorescence optimization complete. Updated settings.")

    # Save optimized settings back to config file
    optimized_path = config_filepath.parent / (config_filepath.stem + "_optimized.yml")
    utils.model_to_yaml(settings, optimized_path)
    print(f"Optimized settings saved to {optimized_path}")

    input_dataset.close()

    return optimized_path
