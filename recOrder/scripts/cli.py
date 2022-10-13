import click


@click.command()
@click.help_option("-h", "--help")
def help():
    """\033[92mrecOrder: Computational Toolkit for Label-Free Imaging\033[0m

    To use recOrder\'s napari plugin, use \033[96mnapari -w recOrder-napari\033[0m

    Thank you for using recOrder.
    """
    print(help.__doc__)
