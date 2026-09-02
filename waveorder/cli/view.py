from pathlib import PurePath

import click


def _add_transfer_function_image(viewer, name, arr, shift_axes=None):
    """Add one transfer function array to the viewer as real/imag bwr layers."""
    import numpy as np

    # Remove leading singleton dims (T, C)
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]

    # ifftshift to center the DC component (all axes by default)
    arr = np.fft.ifftshift(arr, axes=shift_axes)

    lim = np.max(np.abs(arr))
    if lim == 0:
        lim = 1.0

    viewer.add_image(arr.real, name=f"Re({name})", colormap="bwr", contrast_limits=(-lim, lim))
    if np.iscomplexobj(arr):
        viewer.add_image(arr.imag, name=f"Im({name})", colormap="bwr", contrast_limits=(-lim, lim))


def _open_transfer_function(viewer, path, prefix=""):
    """Open a transfer function zarr in napari.

    2D reconstructions store a singular system (``U``, ``S``, ``Vh``) instead of
    a direct transfer function. In that case the transfer function is
    reconstructed from the SVD, ``H = U @ diag(S) @ Vh`` at every lateral
    frequency, and the resulting transfer function(s) are shown rather than the
    raw singular vectors. All arrays are displayed as real/imag parts with a
    diverging ``bwr`` colormap.
    """
    import numpy as np
    import zarr

    root = zarr.open(path, mode="r")
    names = list(root.keys())

    svd_components = {"singular_system_U", "singular_system_S", "singular_system_Vh"}
    if svd_components.issubset(names):
        # Reconstruct the transfer function from the singular system.
        # Stored shapes: U (1, s, k, Vy, Vx), S (1, 1, k, Vy, Vx),
        # Vh (1, k, Z, Vy, Vx). H[s, z] = sum_k U[s, k] * S[k] * Vh[k, z].
        U = np.asarray(root["singular_system_U"])[0]  # (s, k, Vy, Vx)
        S = np.asarray(root["singular_system_S"])[0, 0]  # (k, Vy, Vx)
        Vh = np.asarray(root["singular_system_Vh"])[0]  # (k, Z, Vy, Vx)
        H = np.einsum("skyx,kyx,kzyx->szyx", U, S.astype(U.dtype), Vh)  # (s, Z, Vy, Vx)

        # Name the output transfer functions by object type.
        labels = {
            1: ["transfer_function"],
            2: ["absorption_transfer_function", "phase_transfer_function"],
        }.get(H.shape[0], [f"transfer_function_{i}" for i in range(H.shape[0])])

        for component, label in zip(H, labels):
            # Z is a real-space defocus axis, so only ifftshift the lateral
            # frequency axes.
            _add_transfer_function_image(viewer, f"{prefix}{label}", component, shift_axes=(-2, -1))

        # Show any remaining (non-SVD) transfer function arrays, if present.
        for name in names:
            if name.startswith("singular_system_"):
                continue
            _add_transfer_function_image(viewer, f"{prefix}{name}", np.asarray(root[name]))
        return

    for name in names:
        _add_transfer_function_image(viewer, f"{prefix}{name}", np.asarray(root[name]))


def _is_transfer_function(path):
    """Check if a zarr store is a transfer function (has settings in attrs)."""
    import zarr

    try:
        root = zarr.open(path, mode="r")
        return "settings" in root.attrs
    except Exception:
        return False


def _open_ome_zarr(viewer, path, prefix=""):
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
            name = f"{prefix}{name}"

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
        # With multiple stores open, prefix layer names with the store name so
        # same-named channels (e.g. two reconstructions) stay distinguishable.
        prefix = f"{PurePath(path).stem}: " if len(all_paths) > 1 else ""
        if _is_transfer_function(path):
            _open_transfer_function(viewer, path, prefix=prefix)
        else:
            _open_ome_zarr(viewer, path, prefix=prefix)
    viewer.grid.enabled = True
    napari.run()
