import napari
import numpy as np
import torch


def add_transfer_function_to_viewer(
    viewer: napari.Viewer,
    transfer_function: torch.Tensor,
    zyx_scale: tuple[float, float, float],
    layer_name: str = "Transfer Function",
    clim_factor: float = 1.0,
):
    zyx_shape = transfer_function.shape[-3:]
    lim = torch.max(torch.abs(transfer_function))*clim_factor
    voxel_scale = np.array(
        [
            zyx_shape[0] * zyx_scale[0],
            zyx_shape[1] * zyx_scale[1],
            zyx_shape[2] * zyx_scale[2],
        ]
    )
    shift_dims = (-3, -2, -1)

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
            torch.fft.ifftshift(torch.imag(transfer_function), dim=shift_dims)
            .cpu()
            .numpy(),
            colormap="bwr",
            contrast_limits=(-lim, lim),
            scale=1 / voxel_scale,
            name="Im(" + layer_name + ")",
        )

    viewer.dims.current_step = (0,)*(transfer_function.ndim - 3) + (
        zyx_shape[0] // 2,
        zyx_shape[1] // 2,
        zyx_shape[2] // 2,
    )
    viewer.dims.order = (2, 0, 1)