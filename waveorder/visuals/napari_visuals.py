import numpy as np
import torch

from waveorder.visuals.utils import complex_tensor_to_rgb


def add_transfer_function_to_viewer(
    viewer: "napari.Viewer",
    transfer_function: torch.Tensor,
    zyx_scale: tuple[float, float, float],
    layer_name: str = "Transfer Function",
    clim_factor: float = 1.0,
    complex_rgb: bool = False,
):
    zyx_shape = transfer_function.shape[-3:]
    lim = torch.max(torch.abs(transfer_function)) * clim_factor
    voxel_scale = np.array(
        [
            zyx_shape[0] * zyx_scale[0],
            zyx_shape[1] * zyx_scale[1],
            zyx_shape[2] * zyx_scale[2],
        ]
    )
    shift_dims = (-3, -2, -1)

    if complex_rgb:
        rgb_transfer_function = complex_tensor_to_rgb(
            np.array(torch.fft.ifftshift(transfer_function, dim=shift_dims)),
            saturate_clim_fraction=clim_factor,
        )
        viewer.add_image(
            rgb_transfer_function,
            scale=1 / voxel_scale,
            name=layer_name,
        )
    else:
        viewer.add_image(
            torch.fft.ifftshift(torch.real(transfer_function), dim=shift_dims)
            .cpu()
            .numpy(),
            colormap="bwr",
            contrast_limits=(-lim, lim),
            scale=1 / voxel_scale,
            name="Re(" + layer_name + ")",
        )
        if transfer_function.dtype == torch.complex64:
            viewer.add_image(
                torch.fft.ifftshift(
                    torch.imag(transfer_function), dim=shift_dims
                )
                .cpu()
                .numpy(),
                colormap="bwr",
                contrast_limits=(-lim, lim),
                scale=1 / voxel_scale,
                name="Im(" + layer_name + ")",
            )

    viewer.dims.current_step = (0,) * (transfer_function.ndim - 3) + (
        zyx_shape[0] // 2,
        zyx_shape[1] // 2,
        zyx_shape[2] // 2,
    )

    # Show XZ view by default, and only allow rolling between XY and XZ
    viewer.dims.order = list(range(transfer_function.ndim - 3)) + [
        transfer_function.ndim - 2,
        transfer_function.ndim - 3,
        transfer_function.ndim - 1,
    ]
    viewer.dims.rollable = (False,) * (transfer_function.ndim - 3) + (
        True,
        True,
        False,
    )
    viewer.dims.axis_labels = ("DATA", "OBJECT", "Z", "Y", "X")
