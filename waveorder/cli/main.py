import inspect
import logging
import os
import warnings
from importlib.metadata import version
from typing import Annotated

import typer
from iohub.cli import OptionEatAll
from typer.core import TyperCommand, TyperGroup, TyperOption

# Suppress noisy CUDA and dependency warnings.
# PYTHONWARNINGS catches warnings from torch's C++ queued callbacks that fire
# after import but before Python filterwarnings can intercept them.
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning,ignore::DeprecationWarning")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("iohub").setLevel(logging.ERROR)

from waveorder.cli.apply_inverse_transfer_function import _apply_inverse_transfer_function_cli
from waveorder.cli.bench import benchmark
from waveorder.cli.calibrate_tf import _calibrate_tf_cli
from waveorder.cli.compute_transfer_function import _compute_transfer_function_cli
from waveorder.cli.gui_widget import gui
from waveorder.cli.migrate_config import _migrate_config_cli
from waveorder.cli.reconstruct import _reconstruct_cli
from waveorder.cli.simulate import _simulate_cli
from waveorder.cli.tile_stitch import _tile_stitch_cli
from waveorder.cli.view import _view_cli


class _InputPathsCommand(TyperCommand):
    """Compile input position lists as greedy options."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for param in self.params:
            if isinstance(param, TyperOption) and param.name == "input_position_dirpaths":
                param.__class__ = OptionEatAll


class _RootGroup(TyperGroup):
    """Hide the shell-completion options from the root Options panel.

    Typer prints options before commands, which makes them look like the main
    entry point; the epilog points to them below the command list instead.
    """

    def get_params(self, ctx):
        params = super().get_params(ctx)
        for param in params:
            if param.name in {"install_completion", "show_completion"}:
                param.hidden = True
        return params


def _short_help(alias: str, help_text: str) -> str:
    """First help paragraph, unwrapped, prefixed with the command's shortcut."""
    summary = " ".join(inspect.cleandoc(help_text).split("\n\n")[0].split())
    return f"[green]\\[{alias}][/green] {summary}"


app = typer.Typer(
    name="wo",
    cls=_RootGroup,
    help=(
        "waveorder \\[[green]wo[/green]]: Wave-optical simulation and reconstruction\n\n"
        "Start with [green]wo rec -h[/green] to reconstruct a dataset from a configuration file."
    ),
    epilog="Shell completion: [green]wo --install-completion[/green] or [green]wo --show-completion[/green]",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_enable=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"waveorder {version('waveorder')}")
        raise typer.Exit()


@app.callback()
def _main(
    show_version: Annotated[
        bool | None,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show the version and exit."),
    ] = None,
) -> None:
    pass


_COMMANDS = {
    "reconstruct": (_reconstruct_cli, "rec"),
    "compute-transfer-function": (_compute_transfer_function_cli, "compute-tf"),
    "apply-inverse-transfer-function": (_apply_inverse_transfer_function_cli, "apply-inv-tf"),
    "calibrate-transfer-function": (_calibrate_tf_cli, "calibrate-tf"),
    "tile-stitch": (_tile_stitch_cli, "ts"),
    "simulate": (_simulate_cli, "sim"),
    "migrate-config": (_migrate_config_cli, "mc"),
    "view": (_view_cli, "v"),
    "interactive": (gui, "gui"),
}

for name, (command, alias) in _COMMANDS.items():
    command_options = {"no_args_is_help": True} if name not in {"view", "interactive"} else {}
    if name in {
        "reconstruct",
        "compute-transfer-function",
        "apply-inverse-transfer-function",
        "calibrate-transfer-function",
        "tile-stitch",
    }:
        command_options["cls"] = _InputPathsCommand
    app.command(name, short_help=_short_help(alias, command.__doc__), **command_options)(command)
    app.command(alias, hidden=True, **command_options)(command)

app.add_typer(benchmark, name="benchmark", short_help=_short_help("bm", benchmark.info.help))
app.add_typer(benchmark, name="bm", hidden=True)


if __name__ == "__main__":
    app()
