import click
import napari

#v = napari.Viewer()  # open viewer right away to use on hpc
import numpy as np
from recOrder.io.utils import ret_ori_overlay
from iohub.reader import print_info, _infer_format
from iohub import read_micromanager, open_ome_zarr
from iohub.reader_base import ReaderBase
from iohub.ngff import NGFFNode, Plate, Position, TiledPosition


def _get_reader(filename):
    fmt, extra_info = _infer_format(filename)
    if fmt == "omezarr" and extra_info == "0.4":
        reader = open_ome_zarr(filename, mode="r")
    else:
        reader = read_micromanager(filename, data_type=fmt)
    return reader


def _build_complete_position_list(reader):
    if isinstance(reader, ReaderBase):
        positions = range(reader.get_num_positions())
    elif isinstance(reader, Plate):
        positions = [x[0] for x in list(reader.positions())]
    elif isinstance(reader, Position) or isinstance(reader, TiledPosition):
        positions = ["0"]
    else:
        raise (
            NotImplementedError,
            f"`recOrder view` does not support {type(reader)}.",
        )
    return positions


def _build_array_list(reader, positions, layers):
    arrays = []
    if isinstance(reader, ReaderBase):
        positions = [int(x) for x in positions]

        # Positions as layers
        if layers == "position" or layers == "p":
            for position in positions:
                arrays.append(reader.get_zarr(position))

        # Channels as layers
        elif layers == "channel" or layers == "c":
            print(
                """WARNING: for sending channels to layers is more expensive than 
sending positions to layers. Try loading a small number of positions."""
            )

            ptzyx = (len(positions),) + (reader.shape[0],) + reader.shape[2:]
            for channel in range(int(reader.channels)):
                temp_data = np.zeros(ptzyx)
                for k, position in enumerate(positions):
                    temp_data[k] = reader.get_array(position)[:, channel, ...]
                arrays.append(temp_data)

    elif isinstance(reader, NGFFNode):
        # Positions as layers
        if layers == "position" or layers == "p":
            for position in positions:
                if isinstance(reader, Plate):
                    arrays.append(reader[position].data)
                elif isinstance(reader, Position):
                    arrays.append(reader[position])

        # Channels as layers
        elif layers == "channel" or layers == "c":
            # TODO: Use dask API to lazy load
            if isinstance(reader, Plate):
                T, C, Z, Y, X = reader[positions[0]].data.shape
            elif isinstance(reader, Position):
                T, C, Z, Y, X = reader["0"].shape
            ptzyx = (len(positions),) + (T, Z, Y, X)
            for c in range(C):
                temp_data = np.zeros(ptzyx)
                for k, position in enumerate(positions):
                    if isinstance(reader, Plate):
                        temp_data[k] = reader[position].data[:, c, ...]
                    elif isinstance(reader, Position):
                        temp_data[k] = reader[position][:, c, ...]
                arrays.append(temp_data)

    return arrays


def _build_name_lists(reader, positions, layers):
    layer_names = []
    slice_names = []

    # Build list of position_names
    if isinstance(reader, ReaderBase):
        positions = [int(x) for x in positions]
        position_names = []
        for position in positions:
            try:
                pos_name = reader.stage_positions[position]["Label"]
            except:
                pos_name = "Pos" + str(position)
            position_names.append(pos_name)
    elif isinstance(reader, NGFFNode):
        position_names = positions

    # Assign position and channel names to layers or slices
    if layers == "position" or layers == "p":
        layer_names = position_names
        slice_names = reader.channel_names
    elif layers == "channel" or layers == "c":
        layer_names = reader.channel_names
        slice_names = position_names

    return layer_names, slice_names


def _create_napari_viewer(arrays, layers, layer_names, slice_names, overlay):
    # Add arrays
    for i, array in enumerate(arrays):
        v.add_image(array, name=layer_names[i])
        v.layers[-1].reset_contrast_limits()

        # Add overlays
        if overlay:
            try:
                ret_ind = slice_names.index("Retardance")
                ori_ind = slice_names.index("Orientation")
            except:
                raise ValueError(
                    "No channels named 'Retardance' and 'Orientation'. --overlay option is unavailable."
                )
            ret = array[:, ret_ind, ...]
            ori = array[:, ori_ind, ...]
            overlay = ret_ori_overlay(
                ret,
                ori,
                ret_max=np.percentile(ret, 99.99),
                cmap="HSV",
            )
            v.add_image(
                np.squeeze(overlay),
                name=layer_names[i] + "-BirefringenceOverlay",
                rgb=True,
            )

    # Cosmetic labelling
    v.text_overlay.visible = True
    v.text_overlay.color = "green"

    if layers == "channel" or layers == "c":
        v.dims.axis_labels = ("P", "T", "Z", "Y", "X")

        def text_overlay():
            v.text_overlay.text = (
                f"Position: {slice_names[v.dims.current_step[0]]}"
            )

    elif layers == "position" or layers == "p":
        v.dims.axis_labels = ("T", "C", "Z", "Y", "X")

        def text_overlay():
            v.text_overlay.text = (
                f"Channel: {slice_names[v.dims.current_step[1]]}"
            )

    v.dims.events.current_step.connect(text_overlay)
    text_overlay()

    napari.run()


@click.command()
@click.help_option("-h", "--help")
@click.argument("filename")
@click.option(
    "--positions",
    "-p",
    default=None,
    multiple=True,
    help="""For ome-zarr v0.4 datasets, position name string e.g. \"-p 0/0/0\"
for the 0th row, 0th column, and 0th fov. For other datasets, integer positions
e.g. \"-p 0\" for the 0th position. Accepts multiple positions e.g. \"-p 0/0/0
-p 0/1/10\".""",
)
@click.option(
    "--layers",
    "-l",
    default="position",
    type=click.Choice(["position", "channel", "p", "c"]),
    help="Layers as 'position' ('p') or 'channel' ('c')",
)
@click.option(
    "--overlay",
    "-o",
    is_flag=True,
    default=False,
    help="""
    Computes a retardance-orientation HSV overlay layers. Requires channel 
    names `Retardance` and `Orientation` to be present, and requires `--layers 
    position`
    """,
)
def view(filename, positions, layers, overlay):
    """View a dataset in napari"""
    click.echo(f"Reading file:\t {filename}")
    print_info(filename)

    reader = _get_reader(filename)

    if positions == ():  # If no positions specified, open all positions
        positions = _build_complete_position_list(reader)

    arrays = _build_array_list(reader, positions, layers)
    layer_names, slice_names = _build_name_lists(reader, positions, layers)

    _create_napari_viewer(arrays, layers, layer_names, slice_names, overlay)
