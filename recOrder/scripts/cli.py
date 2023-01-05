import click
import napari
import numpy as np

# Workaround waveorder #97
# Create napari Viewer before other imports
v = napari.Viewer()
v.close()
from recOrder.io.zarr_converter import ZarrConverter
from waveorder.io import WaveorderReader


@click.group()
def cli():
    print(
        "\033[92mrecOrder: Computational Toolkit for Label-Free Imaging\033[0m\n"
    )


@cli.command()
@click.help_option("-h", "--help")
@click.argument("filename")
def info(filename):
    """View a dataset's basic metadata"""
    print(f"Reading file:\t {filename}")
    reader = WaveorderReader(filename)
    print_reader_info(reader)


def print_reader_info(reader):
    print(f"Positions:\t {reader.get_num_positions()}")
    print(f"Time points:\t {reader.shape[0]}")
    print(f"Channels:\t {reader.shape[1]}")
    print(f"(Z, Y, X):\t {reader.shape[2:]}")
    print(f"Channel names:\t {reader.channel_names}")
    print(f"Z step (um):\t {reader.z_step_size}")
    print("")


@cli.command()
@click.help_option("-h", "--help")
@click.argument("filename")
@click.option(
    "--position",
    "-p",
    default=None,
    multiple=True,
    help="Integer positions to open. Accepts multiple positions: -p 0 -p 1 -p 10.",
)
@click.option(
    "--layers",
    "-l",
    default="position",
    type=click.Choice(["position", "channel", "p", "c"]),
    help="Layers as 'position' ('p') or 'channel' ('c')",
)
def view(filename, position=None, layers=None):
    """View a dataset in napari"""
    print(f"Reading file:\t {filename}")
    reader = WaveorderReader(filename)
    print_reader_info(reader)

    if position == ():  # If empty, open all positions
        position = range(reader.get_num_positions())
    position = [int(x) for x in position]

    v = napari.Viewer()
    v.text_overlay.visible = True
    v.text_overlay.color = "green"
    if layers == "position" or layers == "p":
        for i in position:
            try:
                name = reader.stage_positions[i]["Label"]
            except:
                name = "Pos" + str(i)
            v.add_image(reader.get_zarr(i), name=name)
            v.layers[-1].reset_contrast_limits()
        v.dims.axis_labels = ("T", "C", "Z", "Y", "X")

        def text_overlay():
            v.text_overlay.text = (
                f"Channel: {reader.channel_names[v.dims.current_step[1]]}"
            )

        v.dims.events.current_step.connect(text_overlay)
        text_overlay()

    elif layers == "channel" or layers == "c":
        print(
            "WARNING: sending channels to layers is more expensive than sending positions to layers. "
            "Try loading a small number of positions."
        )

        ptzyx = (len(position),) + (reader.shape[0],) + reader.shape[2:]
        for j in range(int(reader.channels)):
            temp_data = np.zeros(ptzyx)
            for k, pos in enumerate(position):
                temp_data[k] = reader.get_array(pos)[:, j, ...]
            v.add_image(temp_data, name=reader.channel_names[j])
            v.layers[-1].reset_contrast_limits()
        v.dims.axis_labels = ("P", "T", "Z", "Y", "X")

    napari.run()


@cli.command()
@click.help_option("-h", "--help")
@click.option(
    "--input",
    required=True,
    type=click.Path(exists=True),
    help="path to the raw data folder containing ome.tifs",
)
@click.option(
    "--output",
    required=True,
    type=str,
    help="full path to save the zarr store (../../Experiment.zarr)",
)
@click.option(
    "--data_type",
    required=False,
    type=str,
    help='Data type, "ometiff", "upti", "zarr"',
)
@click.option(
    "--replace_pos_name",
    required=False,
    type=bool,
    help="whether or not to append position name to data",
)
@click.option(
    "--format_hcs",
    required=False,
    type=bool,
    help='whether or not to format the data as an HCS "well-plate"',
)
def convert(input, output, data_type, replace_pos_name, format_hcs):
    """Convert MicroManager ome-tiff to ome-zarr"""
    converter = ZarrConverter(
        input, output, data_type, replace_pos_name, format_hcs
    )
    converter.run_conversion()
