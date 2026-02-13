from pathlib import Path

import click
import numpy as np
import xarray as xr
from iohub.ngff import open_ome_zarr
from iohub.ngff.models import TransformationMeta

from waveorder.api import (
    birefringence,
    birefringence_and_phase,
    fluorescence,
    phase,
)
from waveorder.cli.parsing import config_filepath, output_dirpath
from waveorder.cli.printing import echo_headline
from waveorder.cli.settings import ReconstructionSettings
from waveorder.io import utils


def _write_czyx(czyx: xr.DataArray, path: Path):
    """Write a CZYX xr.DataArray to an HCS OME-Zarr."""
    channel_names = list(czyx.coords["c"].values)
    dataset = open_ome_zarr(path, layout="hcs", mode="w", channel_names=channel_names)
    position = dataset.create_position("0", "0", "0")
    position.create_zeros(
        "0",
        (1, *czyx.shape),
        dtype=np.float32,
        transform=[
            TransformationMeta(
                type="scale",
                scale=[
                    1,
                    1,
                    float(czyx.z[1] - czyx.z[0]),
                    float(czyx.y[1] - czyx.y[0]),
                    float(czyx.x[1] - czyx.x[0]),
                ],
            )
        ],
    )
    position["0"][0] = czyx.values

    dataset.close()


@click.command("simulate")
@config_filepath()
@output_dirpath()
def _simulate_cli(config_filepath: Path, output_dirpath: Path):
    """Simulate phantom data matching a configuration file.

    Writes a single zarr with both the ground-truth phantom and
    simulated measurement as separate channels.

    \b
    >> wo sim -c ./phase.yml -o ./phase.zarr
    """
    settings = utils.yaml_to_model(config_filepath, ReconstructionSettings)

    recon_dim = settings.reconstruction_dimension

    if settings.birefringence is not None and settings.phase is not None:
        echo_headline("Simulating birefringence + phase data")
        scheme = f"{len(settings.input_channel_names)}-State"
        phantom, data = birefringence_and_phase.simulate(
            settings.birefringence,
            settings.phase,
            scheme=scheme,
        )
    elif settings.birefringence is not None:
        echo_headline("Simulating birefringence data")
        scheme = f"{len(settings.input_channel_names)}-State"
        phantom, data = birefringence.simulate(
            settings.birefringence,
            scheme=scheme,
        )
    elif settings.phase is not None:
        echo_headline("Simulating phase data")
        phantom, data = phase.simulate(
            settings.phase,
            recon_dim=recon_dim,
        )
    elif settings.fluorescence is not None:
        echo_headline("Simulating fluorescence data")
        phantom, data = fluorescence.simulate(
            settings.fluorescence,
            recon_dim=recon_dim,
            channel_name=settings.input_channel_names[0],
        )
    else:
        raise click.UsageError("Config must contain birefringence, phase, or fluorescence settings")

    czyx = xr.concat([phantom, data], dim="c")
    _write_czyx(czyx, output_dirpath)

    echo_headline(f"Wrote simulated data to {output_dirpath}")
