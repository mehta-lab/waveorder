from pathlib import Path

import typer

from waveorder.cli.parsing import ConfigFilepath, OutputDirpath


def _simulate_cli(config_filepath: ConfigFilepath, output_dirpath: OutputDirpath):
    """Simulate phantom data matching a configuration file.

    For a 3.x configuration (or a 4.0 configuration without a `simulate`
    block), writes a single zarr with both the ground-truth phantom and
    simulated measurement as separate channels.

    A 4.0 configuration with a `simulate: {seed, shape}` block describes the
    true imaging system (design section 5) and writes three stores into the
    output directory: `data.zarr`, `object.zarr`, and `tf.zarr`.

    \b
    Example:
      \033[92mwo sim -c ./phase.yml -o ./phase.zarr\033[0m
    """
    typer.echo(typer.style("Starting simulation...", fg=typer.colors.GREEN))

    # Deferred imports: these pull in torch, iohub, numpy, etc.
    # Only loaded when the command runs, keeping wo sim -h fast.
    import numpy as np
    import xarray as xr
    import yaml
    from iohub.ngff import open_ome_zarr
    from iohub.ngff.models import TransformationMeta

    from waveorder.api import (
        birefringence,
        birefringence_and_phase,
        fluorescence,
        phase,
    )
    from waveorder.cli.printing import echo_headline
    from waveorder.config.loader import load_reconstruction_settings, load_v4_config
    from waveorder.config.migrate import detect_version
    from waveorder.config.simulate import run_simulation

    with open(config_filepath) as f:
        raw = yaml.safe_load(f)
    if detect_version(raw) >= 4 and raw.get("simulate") is not None:
        cfg = load_v4_config(config_filepath)
        truth = run_simulation(cfg, output_dirpath)
        echo_headline(f"Wrote data.zarr, object.zarr, tf.zarr to {output_dirpath}")
        for name, value in truth.items():
            print(f"    true {name:16s} {value:+.4f} waves")
        return

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

    settings = load_reconstruction_settings(config_filepath)

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
        raise typer.BadParameter("Config must contain birefringence, phase, or fluorescence settings")

    czyx = xr.concat([phantom, data], dim="c")
    _write_czyx(czyx, output_dirpath)

    echo_headline(f"Wrote simulated data to {output_dirpath}")
