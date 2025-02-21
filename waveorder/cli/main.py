import click

from waveorder.cli.apply_inverse_transfer_function import apply_inv_tf
from waveorder.cli.compute_transfer_function import compute_tf
from waveorder.cli.reconstruct import reconstruct

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


cli.add_command(reconstruct)
cli.add_command(compute_tf)
cli.add_command(apply_inv_tf)
try:
    cli.add_command(gui)
except:
    pass

if __name__ == "__main__":
    cli()
