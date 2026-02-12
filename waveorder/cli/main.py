import click

from waveorder.cli.apply_inverse_transfer_function import (
    _apply_inverse_transfer_function_cli,
)
from waveorder.cli.compute_transfer_function import (
    _compute_transfer_function_cli,
)
from waveorder.cli.download import _download_examples_cli
from waveorder.cli.reconstruct import _reconstruct_cli
from waveorder.cli.simulate import _simulate_cli
from waveorder.cli.view import _view_cli

try:
    from waveorder.cli.gui_widget import gui as _interactive_cli
except:
    _interactive_cli = None

CONTEXT = {"help_option_names": ["-h", "--help"]}


class AliasGroup(click.Group):
    """Click group with command aliases shown in help."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._aliases: dict[str, str] = {}
        self._reverse: dict[str, str] = {}
        self._display_order: list[tuple[str, int]] = []

    def add_alias(self, alias: str, command_name: str):
        self._aliases[alias] = command_name
        self._reverse[command_name] = alias

    def get_command(self, ctx, cmd_name):
        cmd_name = self._aliases.get(cmd_name, cmd_name)
        return super().get_command(ctx, cmd_name)

    def list_commands(self, ctx):
        return list(self.commands.keys())

    def format_usage(self, ctx, formatter):
        formatter.write_usage(
            "wo",
            "[OPTIONS] COMMAND [ARGS]...",
        )

    def format_help(self, ctx, formatter):
        self.format_usage(ctx, formatter)
        formatter.write("\n")
        formatter.write(
            "  \033[92mwaveorder\033[0m [\033[92mwo\033[0m]: "
            "Wave-optical simulation and reconstruction\n"
        )
        formatter.write("\n")

        # Options
        formatter.write("Options:\n")
        formatter.write("  -h, --help  Show this message and exit.\n")
        formatter.write("\n")

        # Commands with aliases, manually ordered
        formatter.write("Commands:\n")
        col = 40  # align aliases to this column
        for name, indent in self._display_order:
            if name not in self.commands:
                continue
            alias = self._reverse.get(name)
            pad = " " * indent
            label = f"  {pad}{name}"
            if alias:
                gap = " " * max(1, col + indent - len(label))
                label += f"{gap}[{alias}]"
            formatter.write(label + "\n")

        formatter.write("\n")


@click.group(context_settings=CONTEXT, cls=AliasGroup)
def cli():
    pass


# Commands
cli.add_command(_simulate_cli, "simulate")
cli.add_command(_reconstruct_cli, "reconstruct")
cli.add_command(_compute_transfer_function_cli, "compute-transfer-function")
cli.add_command(
    _apply_inverse_transfer_function_cli, "apply-inverse-transfer-function"
)
cli.add_command(_view_cli, "view")
if _interactive_cli is not None:
    cli.add_command(_interactive_cli, "interactive")
cli.add_command(_download_examples_cli, "download-examples")

# Display order: (command_name, indent)
cli._display_order = [
    ("reconstruct", 0),
    ("compute-transfer-function", 2),
    ("apply-inverse-transfer-function", 2),
    ("view", 0),
    ("interactive", 0),
    ("simulate", 0),
    ("download-examples", 0),
]

# Aliases
cli.add_alias("rec", "reconstruct")
cli.add_alias("sim", "simulate")
cli.add_alias("dle", "download-examples")
cli.add_alias("compute-tf", "compute-transfer-function")
cli.add_alias("apply-inv-tf", "apply-inverse-transfer-function")
if _interactive_cli is not None:
    cli.add_alias("gui", "interactive")

if __name__ == "__main__":
    cli()
