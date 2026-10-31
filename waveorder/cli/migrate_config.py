from pathlib import Path
from typing import Annotated, Optional

import typer


def _migrate_config_cli(
    config_filepath: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="Path to a 3.x YAML configuration file.",
        ),
    ],
    output_filepath: Annotated[
        Optional[Path],
        typer.Option(
            "--output-filepath",
            "-o",
            help="Path for the 4.0 configuration. Defaults to '<input stem>_v4.yml' beside the input.",
        ),
    ] = None,
):
    """
    Migrate a 3.x configuration file to the 4.0 format.

    The migration is deterministic and lossless: the 4.0 configuration
    selects the same code path with the same numerical parameters and
    produces the same reconstruction.

    \b
    Example:
      \033[92mwaveorder migrate-config ./config.yml -o ./config_v4.yml\033[0m
    """
    # Deferred imports: these pull in torch via the settings models.
    # Only loaded when the command runs, keeping the help fast.
    import yaml

    from waveorder.config.migrate import (
        MigrationError,
        detect_version,
        migrate_3x_dict,
        v4_to_yaml_text,
    )

    with open(config_filepath) as f:
        raw = yaml.safe_load(f)

    if detect_version(raw) >= 4:
        typer.echo(typer.style(f"{config_filepath} is already a 4.0 configuration.", fg=typer.colors.YELLOW))
        raise typer.Exit()

    try:
        migrated = migrate_3x_dict(raw)
    except MigrationError as e:
        raise typer.BadParameter(str(e)) from e

    if output_filepath is None:
        output_filepath = config_filepath.parent / f"{config_filepath.stem}_v4.yml"

    text = v4_to_yaml_text(migrated)
    output_filepath.write_text(text)
    typer.echo(text)
    typer.echo(typer.style(f"Wrote 4.0 configuration to {output_filepath}", fg=typer.colors.GREEN))
