import matplotlib.pyplot as plt

from waveorder.visuals.utils import complex_tensor_to_rgb
from waveorder.sampling import nd_fourier_central_cuboid

import numpy as np
import os
import torch


def plot_5d_ortho(
    rczyx_data,
    filename,
    zyx_scale,
    z_slice,
    row_labels,
    column_labels,
    rose_path=None,
    inches_per_column=1,
    saturate_clim_fraction=1.0,
    trim_edges=0,
    fourier_space=True,
):
    R, C, Z, Y, X = rczyx_data.shape

    axis_extent = [
        Z * zyx_scale[0],
        Y * zyx_scale[1],
        X * zyx_scale[2],
    ]
    if fourier_space:
        axis_extent = [1 / x for x in axis_extent]

    rczyx_data = nd_fourier_central_cuboid(
        rczyx_data, (R, C, Z - trim_edges, Y - trim_edges, X - trim_edges)
    )

    R, C, Z, Y, X = rczyx_data.shape  # after cropping

    rczyx_data = torch.fft.ifftshift(rczyx_data, dim=(-3, -2, -1))
    sfZYX_rgb = complex_tensor_to_rgb(rczyx_data, saturate_clim_fraction)

    assert R == len(row_labels)
    assert C == len(column_labels)

    n_rows = (R * 2) + 1
    n_cols = (C * 2) + 1

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols, n_rows),
        gridspec_kw={
            "wspace": 0.1,  # Adjust this value to reduce space between columns
            "hspace": 0.1,  # Adjust this value to reduce space between rows
            "width_ratios": [1]
            + C
            * [
                1,
                Z / X,
            ],
            "height_ratios": [1]
            + R
            * [
                1,
                Z / Y,
            ],
        },
    )

    rose_path = os.path.join(
        os.path.dirname(__file__),
        f"./assets/rose.png",
    )
    axes[0, 0].imshow(plt.imread(rose_path))

    for i in range(n_rows):
        for j in range(n_cols):
            if (i == 0 and (j - 1) % 2 == 0) or (j == 0 and (i - 1) % 2 == 0):
                if i == 0:
                    folder = "gellman"
                    index = column_labels[int((j - 1) / 2)]
                else:
                    folder = "stokes"
                    index = row_labels[int((i - 1) / 2)]

                if isinstance(index, int):
                    image_path = os.path.join(
                        os.path.dirname(__file__),
                        f"./assets/{folder}/{index}.png",
                    )
                    image = plt.imread(image_path)
                    axes[i, j].imshow(image)
                else:
                    axes[i, j].text(
                        0.5,
                        0.5,
                        index,
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=10,
                        color="black",
                    )

            if i > 0 and j > 0:
                # XY
                if (i - 1) % 2 == 0 and (j - 1) % 2 == 0:
                    axes[i, j].imshow(
                        sfZYX_rgb[
                            int((i - 1) / 2),
                            int((j - 1) / 2),
                            (Z // 2) + z_slice,
                        ],
                        aspect=axis_extent[1] / axis_extent[2],
                    )
                # YZ
                elif (i - 1) % 2 == 0 and (j - 1) % 2 == 1:
                    axes[i, j].imshow(
                        sfZYX_rgb[
                            int((i - 1) / 2), int((j - 1) / 2), :, :, X // 2, :
                        ].transpose(1, 0, 2),
                        aspect=axis_extent[1] / axis_extent[0],
                    )
                # XZ
                elif (i - 1) % 2 == 1 and (j - 1) % 2 == 0:
                    axes[i, j].imshow(
                        sfZYX_rgb[
                            int((i - 1) / 2), int((j - 1) / 2), :, Y // 2
                        ],
                        aspect=axis_extent[0] / axis_extent[1],
                    )

            axes[i, j].tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )

            # Draw lines between rows and cols
            top = axes[0, 0].get_position().y1
            bottom = axes[-1, -1].get_position().y0
            left = axes[0, 0].get_position().x0
            right = axes[-1, -1].get_position().x1
            if i == 0 and (j - 1) % 2 == 0:
                left_edge = (
                    axes[0, j].get_position().x0
                    + axes[0, j - 1].get_position().x1
                ) / 2
                fig.add_artist(
                    plt.Line2D(
                        [left_edge, left_edge],
                        [bottom, top],
                        transform=fig.transFigure,
                        color="black",
                        lw=0.5,
                    )
                )
            if j == 0 and (i - 1) % 2 == 0:
                top_edge = (
                    axes[i, 0].get_position().y1
                    + axes[i - 1, 0].get_position().y0
                ) / 2
                fig.add_artist(
                    plt.Line2D(
                        [left, right],
                        [top_edge, top_edge],
                        transform=fig.transFigure,
                        color="black",
                        lw=0.5,
                    )
                )

            axes[i, j].spines["top"].set_visible(False)
            axes[i, j].spines["right"].set_visible(False)
            axes[i, j].spines["bottom"].set_visible(False)
            axes[i, j].spines["left"].set_visible(False)

    # Draw ortho view lines
    fig.add_artist(
        plt.Line2D(
            [Y // 2, Y // 2],
            [0, X],
            transform=axes[1, 1].transData,  # use axis coordinates
            color="red",
            lw=0.5,
        )
    )
    fig.add_artist(
        plt.Line2D(
            [0, X],
            [Y // 2, Y // 2],  # from bottom to top in axis coordinates
            transform=axes[1, 1].transData,  # use axis coordinates
            color="blue",
            lw=0.5,
        )
    )
    fig.add_artist(
        plt.Rectangle(
            (0, 0),  # lower-left corner
            X,  # width
            Y,  # height
            linewidth=0.5,
            edgecolor="green",
            facecolor="none",
            transform=axes[1, 1].transData,  # use axis coordinates
        )
    )
    axes[1, 1].text(
        0.1,
        0.975,
        "x",
        horizontalalignment="left",
        verticalalignment="top",
        transform=axes[1, 1].transAxes,
        fontsize=5,
        color="black",
    )
    axes[1, 1].text(
        0.025,
        0.9,
        "y",
        horizontalalignment="left",
        verticalalignment="top",
        transform=axes[1, 1].transAxes,
        fontsize=5,
        color="black",
    )

    fig.add_artist(
        plt.Line2D(
            [
                Z // 2 + z_slice,
                Z // 2 + z_slice,
            ],
            [0, Y],
            transform=axes[1, 2].transData,  # use axis coordinates
            color="green",
            lw=0.5,
        )
    )
    fig.add_artist(
        plt.Line2D(
            [0, Z],  # from bottom to top in axis coordinates
            [Y // 2, Y // 2],
            transform=axes[1, 2].transData,  # use axis coordinates
            color="blue",
            lw=0.5,
        )
    )

    rect = plt.Rectangle(
        (0, 0),  # lower-left corner
        Z,  # width
        Y,  # height
        linewidth=0.5,
        edgecolor="red",
        facecolor="none",
        transform=axes[1, 2].transData,  # use axis coordinates
    )
    fig.add_artist(rect)

    axes[1, 2].text(
        0.1,
        0.975,
        "z",
        horizontalalignment="left",
        verticalalignment="top",
        transform=axes[1, 2].transAxes,
        fontsize=5,
        color="black",
    )
    axes[1, 2].text(
        0.025,
        0.9,
        "y",
        horizontalalignment="left",
        verticalalignment="top",
        transform=axes[1, 2].transAxes,
        fontsize=5,
        color="black",
    )

    fig.add_artist(
        plt.Line2D(
            [0, X],
            [
                Z // 2 + z_slice,
                Z // 2 + z_slice,
            ],  # from bottom to top in axis coordinates
            transform=axes[2, 1].transData,  # use axis coordinates
            color="green",
            lw=0.5,
        )
    )
    fig.add_artist(
        plt.Line2D(
            [X // 2, X // 2],
            [0, Z],  # from bottom to top in axis coordinates
            transform=axes[2, 1].transData,  # use axis coordinates
            color="red",
            lw=0.5,
        )
    )

    rect = plt.Rectangle(
        (0, 0),  # lower-left corner
        X,  # width
        Z,  # height
        linewidth=0.5,
        edgecolor="blue",
        facecolor="none",
        transform=axes[2, 1].transData,  # use axis coordinates
    )
    fig.add_artist(rect)

    axes[2, 1].text(
        0.1,
        0.975,
        "x",
        horizontalalignment="left",
        verticalalignment="top",
        transform=axes[2, 1].transAxes,
        fontsize=5,
        color="black",
    )
    axes[2, 1].text(
        0.025,
        0.9,
        "z",
        horizontalalignment="left",
        verticalalignment="top",
        transform=axes[2, 1].transAxes,
        fontsize=5,
        color="black",
    )

    print(f"Saving {filename}")
    fig.savefig(filename, dpi=300, format="pdf", bbox_inches="tight")
