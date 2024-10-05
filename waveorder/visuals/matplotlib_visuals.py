import matplotlib.pyplot as plt

from waveorder.visuals.utils import complex_tensor_to_rgb
import os
import numpy as np
import re
import torch


def plot_transfer_function(
    sfZYX_data,
    filename,
    zyx_scale,
    z_slice,
    s_labels,
    f_labels,
    rose_path=None,
    inches_per_column=1,
    saturate_clim_fraction=1.0,
):
    sfZYX_data = torch.fft.ifftshift(sfZYX_data, dim=(-3, -2, -1))
    sfZYX_rgb = complex_tensor_to_rgb(sfZYX_data, saturate_clim_fraction)
    sfZYX_rgb[:, :, 0, 0, 0, :] = 0
    sfZYX_rgb[:, :, -1, -1, -1, :] = 0

    S, F, Z, Y, X = sfZYX_data.shape
    assert S == len(s_labels)
    assert F == len(f_labels)

    X_size = 1 * inches_per_column
    Y_size = (zyx_scale[2] / zyx_scale[1]) * inches_per_column
    Z_size = (zyx_scale[2] / zyx_scale[0]) * inches_per_column

    n_rows = (S * 3) + 1
    n_cols = F + 1

    height = S * (Y_size + (2 * Z_size)) + 1
    width = (F + 1) * X_size

    height_ratios = [1] + [
        item
        for sublist in [[Y_size, Z_size, Z_size] for _ in range(S)]
        for item in sublist
    ]

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(width, height),
        gridspec_kw={
            "width_ratios": [1] * n_cols,
            "height_ratios": height_ratios,
            "wspace": 0.1,  # Adjust this value to reduce space between columns
            "hspace": 0.1,  # Adjust this value to reduce space between rows
        },
    )

    rose_path = os.path.join(
        os.path.dirname(__file__),
        f"./assets/rose.png",
    )
    axes[0, 0].imshow(plt.imread(rose_path))

    for i in range(n_rows):
        for j in range(n_cols):
            if (i == 0 and j > 0) or (j == 0 and (i - 1) % 3 == 0):
                if i == 0:
                    folder = "gellman"
                    index = f_labels[j - 1]
                else:
                    folder = "stokes"
                    index = s_labels[int((i - 1) / 3)]
                image_path = os.path.join(
                    os.path.dirname(__file__),
                    f"./assets/{folder}/{index}.png",
                )
                image = plt.imread(image_path)
                axes[i, j].imshow(image)

            if i > 0 and j > 0:
                if (i - 1) % 3 == 0:
                    axes[i, j].imshow(
                        sfZYX_rgb[int((i - 1) / 3), j - 1, z_slice],
                        aspect=zyx_scale[1] / zyx_scale[2],
                    )
                elif (i - 1) % 3 == 1:
                    axes[i, j].imshow(
                        sfZYX_rgb[int((i - 1) / 3), j - 1, :, :, X // 2, :],
                        aspect=zyx_scale[2] / zyx_scale[0],
                    )
                elif (i - 1) % 3 == 2:
                    axes[i, j].imshow(
                        sfZYX_rgb[int((i - 1) / 3), j - 1, :, Y // 2],
                        aspect=zyx_scale[1] / zyx_scale[0],
                    )

            axes[i, j].tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )
            # axes[i, j].spines["top"].set_visible(False)
            # axes[i, j].spines["right"].set_visible(False)
            # axes[i, j].spines["bottom"].set_visible(False)
            # axes[i, j].spines["left"].set_visible(False)

    fig.savefig(filename, bbox_inches="tight")
