import click


def _open_transfer_function(viewer, path):
    """Open a transfer function zarr, displaying real/imag parts with bwr colormap."""
    import numpy as np
    import zarr

    root = zarr.open(path, mode="r")
    for name in root.keys():
        arr = np.array(root[name])
        # Remove leading singleton dims (T, C)
        while arr.ndim > 3 and arr.shape[0] == 1:
            arr = arr[0]

        # ifftshift to center the DC component
        arr = np.fft.ifftshift(arr)

        lim = np.max(np.abs(arr))
        if lim == 0:
            lim = 1.0
        viewer.add_image(
            arr.real,
            name=f"Re({name})",
            colormap="bwr",
            contrast_limits=(-lim, lim),
        )
        if np.iscomplexobj(arr):
            viewer.add_image(
                arr.imag,
                name=f"Im({name})",
                colormap="bwr",
                contrast_limits=(-lim, lim),
            )


def _is_transfer_function(path):
    """Check if a zarr store is a transfer function (has settings in attrs)."""
    import zarr

    try:
        root = zarr.open(path, mode="r")
        return "settings" in root.attrs
    except Exception:
        return False


def _open_ome_zarr(viewer, path):
    """Open an OME-Zarr, squeezing singleton dims so 2D results always appear."""
    import numpy as np
    from iohub.ngff import open_ome_zarr

    plate = open_ome_zarr(path, mode="r")
    positions = list(plate.positions())
    multi_position = len(positions) > 1

    for position_key, position in positions:
        data = np.array(position["0"])  # TCZYX
        T, C, Z, Y, X = data.shape
        scale = position.scale

        for c_idx, ch_name in enumerate(position.channel_names):
            ch_data = data[:, c_idx]  # TZYX
            name = f"{ch_name} [{position_key}]" if multi_position else ch_name

            # Squeeze singleton T and Z so napari shows 2D results at every Z
            if T == 1:
                ch_data = ch_data[0]  # ZYX
            if Z == 1:
                ch_data = ch_data[0]  # YX (or TYX if T > 1)

            layer_scale = scale[-ch_data.ndim :]
            viewer.add_image(ch_data, name=name, scale=layer_scale)

            # Apply contrast limits from metadata
            omero_ch = position.metadata.omero.channels[c_idx]
            if omero_ch.window:
                w = omero_ch.window
                if isinstance(w, dict):
                    lo, hi = w.get("start", 0), w.get("end", 1)
                else:
                    lo, hi = w.start, w.end
                if lo < hi:
                    viewer.layers[name].contrast_limits = (lo, hi)

    plate.close()


@click.command("view")
@click.argument("paths", nargs=-1)
def _view_cli(paths):
    """Open OME-Zarr datasets or transfer functions in napari.

    Accepts paths as arguments and/or from stdin (one per line).

    \b
    >> wo view ./input.zarr ./reconstruction.zarr
    >> wo view ./transfer_function.zarr
    """
    import sys

    import napari

    all_paths = list(paths)
    if not sys.stdin.isatty():
        for line in sys.stdin:
            line = line.strip()
            if line:
                all_paths.append(line)

    if not all_paths:
        raise click.UsageError("No paths provided.")

    viewer = napari.Viewer()
    for path in all_paths:
        if _is_transfer_function(path):
            _open_transfer_function(viewer, path)
        else:
            _open_ome_zarr(viewer, path)
    viewer.grid.enabled = True
    napari.run()
