import click

from recOrder.cli.apply_inverse_transfer_function import apply_inv_tf
from recOrder.cli.compute_transfer_function import compute_tf
from recOrder.cli.reconstruct import reconstruct

CONTEXT = {"help_option_names": ["-h", "--help"]}


# `recorder -h` will show subcommands in the order they are added
class NaturalOrderGroup(click.Group):
    def list_commands(self, ctx):
        return self.commands.keys()


@click.group(context_settings=CONTEXT, cls=NaturalOrderGroup)
def cli():
    """\033[92mrecOrder: Computational Toolkit for Label-Free Imaging\033[0m\n"""


cli.add_command(reconstruct)
cli.add_command(compute_tf)
cli.add_command(apply_inv_tf)
