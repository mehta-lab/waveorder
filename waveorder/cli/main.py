import click

from waveorder.cli.apply_inverse_transfer_function import (
    _apply_inverse_transfer_function_cli,
)
from waveorder.cli.compute_transfer_function import (
    _compute_transfer_function_cli,
)
from waveorder.cli.reconstruct import _reconstruct_cli

try:
    from waveorder.cli.gui_widget import gui
except:
    pass

CONTEXT = {"help_option_names": ["-h", "--help"]}


# `waveorder -h` will show subcommands in the order they are added
class NaturalOrderGroup(click.Group):
    def list_commands(self, ctx):
        return self.commands.keys()


@click.group(context_settings=CONTEXT, cls=NaturalOrderGroup)
def cli():
    """\033[92mwaveorder: Computational Toolkit for Label-Free Imaging\033[0m\n"""


cli.add_command(_reconstruct_cli)
cli.add_command(_compute_transfer_function_cli)
cli.add_command(_apply_inverse_transfer_function_cli)
try:
    cli.add_command(gui)
except:
    pass

if __name__ == "__main__":
    cli()
