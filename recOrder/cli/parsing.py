import click
from typing import Callable
from iohub.ngff import open_ome_zarr, Plate


def _validate_fov_path(
    ctx: click.Context, opt: click.Option, value: str
) -> None:
    dataset = open_ome_zarr(value)
    if isinstance(dataset, Plate):
        raise ValueError(
            "Please supply a single position instead of an HCS plate. Likely fix: replace 'input.zarr' with 'input.zarr/0/0/0'"
        )
    return value


def input_data_path_argument() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.argument(
            "input-data-path",
            type=click.Path(exists=True),
            callback=_validate_fov_path,
            nargs=1,
        )(f)

    return decorator


def config_path_option() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--config-path", "-c", required=True, help="Path to config.yml"
        )(f)

    return decorator


def output_dataset_option(default) -> Callable:
    click_options = [
        click.option(
            "--output-path",
            "-o",
            default=default,
            help="Path to output.zarr",
        )
    ]
    # good place to add chunking, overwrite flag, etc

    def decorator(f: Callable) -> Callable:
        for opt in click_options:
            f = opt(f)
        return f

    return decorator
