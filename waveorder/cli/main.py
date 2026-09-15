import logging
import os
import warnings

import typer
from iohub.cli import OptionEatAll
from typer.core import TyperCommand, TyperOption

# Suppress noisy CUDA and dependency warnings.
# PYTHONWARNINGS catches warnings from torch's C++ queued callbacks that fire
# after import but before Python filterwarnings can intercept them.
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning,ignore::DeprecationWarning")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("iohub").setLevel(logging.ERROR)

from waveorder.cli.apply_inverse_transfer_function import _apply_inverse_transfer_function_cli
from waveorder.cli.bench import benchmark
from waveorder.cli.compute_transfer_function import _compute_transfer_function_cli
from waveorder.cli.gui_widget import gui
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


app = typer.Typer(
    name="wo",
    help="waveorder [wo]: Wave-optical simulation and reconstruction",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_enable=False,
)

_COMMANDS = {
    "reconstruct": (_reconstruct_cli, "rec"),
    "compute-transfer-function": (_compute_transfer_function_cli, "compute-tf"),
    "apply-inverse-transfer-function": (_apply_inverse_transfer_function_cli, "apply-inv-tf"),
    "tile-stitch": (_tile_stitch_cli, "ts"),
    "simulate": (_simulate_cli, "sim"),
    "view": (_view_cli, "v"),
    "interactive": (gui, "gui"),
}

for name, (command, alias) in _COMMANDS.items():
    command_options = {"no_args_is_help": True} if name not in {"view", "interactive"} else {}
    if name in {
        "reconstruct",
        "compute-transfer-function",
        "apply-inverse-transfer-function",
        "tile-stitch",
    }:
        command_options["cls"] = _InputPathsCommand
    app.command(name, **command_options)(command)
    app.command(alias, hidden=True, **command_options)(command)

app.add_typer(benchmark, name="benchmark")
app.add_typer(benchmark, name="bm", hidden=True)


if __name__ == "__main__":
    app()
