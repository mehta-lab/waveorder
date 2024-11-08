import matplotlib.pyplot as plt

from waveorder.visuals.utils import complex_tensor_to_rgb
from waveorder.sampling import nd_fourier_central_cuboid

import numpy as np
import os
import torch


def plot_5d_ortho(
    rcCzyx_data,
    filename,
    voxel_size,
    zyx_slice,
    color_funcs,
    row_labels=None,
    column_labels=None,
    rose_path=None,
    inches_per_column=1.5,
    label_size=1,
    ortho_line_width=0.5,
    row_column_line_width=0.5,
    **kwargs,
):
    R, C, Ch, Z, Y, X = rcCzyx_data.shape

    # Extent
    dZ, dY, dX = Z * voxel_size[0], Y * voxel_size[1], X * voxel_size[2]

    # if fourier_space:
    #     axis_extent = [1 / x for x in axis_extent]

    # rczyx_data = nd_fourier_central_cuboid(
    #     rczyx_data, (R, C, Z - trim_edges, Y - trim_edges, X - trim_edges)
    # )

    # R, C, Z, Y, X = rczyx_data.shape  # after cropping

    # rczyx_data = torch.fft.ifftshift(rczyx_data, dim=(-3, -2, -1))
    # sfZYX_rgb = complex_tensor_to_rgb(rczyx_data, saturate_clim_fraction)

    assert R == len(row_labels)
    assert C == len(column_labels)
    assert zyx_slice[0] < Z and zyx_slice[1] < Y and zyx_slice[2] < X
    assert zyx_slice[0] >= 0 and zyx_slice[1] >= 0 and zyx_slice[2] >= 0

    assert R == len(color_funcs)
    for color_func_row in color_funcs:
        if isinstance(color_func_row, list):
            assert len(color_func_row) == C
        else:
            color_func_row = [color_func_row] * C

    n_rows = 1 + (2 * R)
    n_cols = 1 + (2 * C)

    width_ratios = [label_size] + C * [1, dZ / dX]
    height_ratios = [label_size] + R * [dY / dX, dZ / dX]

    fig_width = np.array(width_ratios).sum() * inches_per_column
    fig_height = np.array(height_ratios).sum() * inches_per_column

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        gridspec_kw={
            "wspace": 0.05,
            "hspace": 0.05,
            "width_ratios": width_ratios,
            "height_ratios": height_ratios,
        },
    )

    if rose_path is not None:
        axes[0, 0].imshow(plt.imread(rose_path))

    for i in range(n_rows):
        for j in range(n_cols):
            # Add labels
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
                        fontsize=10 * label_size,
                        color="black",
                    )

            # Add data
            if i > 0 and j > 0:
                color_func = color_funcs[int((i - 1) / 2)][int((j - 1) / 2)]

                Cyx_data = rcCzyx_data[
                    int((i - 1) / 2), int((j - 1) / 2), :, zyx_slice[0]
                ]
                Cyz_data = rcCzyx_data[
                    int((i - 1) / 2), int((j - 1) / 2), :, :, :, zyx_slice[2]
                ].transpose(0, 2, 1)
                Czx_data = rcCzyx_data[
                    int((i - 1) / 2), int((j - 1) / 2), :, :, zyx_slice[1]
                ]

                # YX
                if (i - 1) % 2 == 0 and (j - 1) % 2 == 0:
                    axes[i, j].imshow(
                        color_func(*Cyx_data, **kwargs),
                        aspect=voxel_size[1] / voxel_size[2],
                    )
                # YZ
                elif (i - 1) % 2 == 0 and (j - 1) % 2 == 1:
                    axes[i, j].imshow(
                        color_func(*Cyz_data, **kwargs),
                        aspect=voxel_size[1] / voxel_size[0],
                    )
                # XZ
                elif (i - 1) % 2 == 1 and (j - 1) % 2 == 0:
                    axes[i, j].imshow(
                        color_func(*Czx_data, **kwargs),
                        aspect=voxel_size[0] / voxel_size[2],
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
                        lw=row_column_line_width,
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
                        lw=row_column_line_width,
                    )
                )

            # Remove ticks and spines
            axes[i, j].tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )
            axes[i, j].spines["top"].set_visible(False)
            axes[i, j].spines["right"].set_visible(False)
            axes[i, j].spines["bottom"].set_visible(False)
            axes[i, j].spines["left"].set_visible(False)

    yx_slice_color = "green"
    yz_slice_color = "red"
    zx_slice_color = "blue"

    # Label orthogonal slices
    add_ortho_lines_to_axis(
        axes[1, 1],
        (zyx_slice[1], zyx_slice[2]),
        ("y", "x"),
        yx_slice_color,
        yz_slice_color,
        zx_slice_color,
        ortho_line_width,
    )  # YX axis

    add_ortho_lines_to_axis(
        axes[2, 1],
        (zyx_slice[0], zyx_slice[2]),
        ("z", "x"),
        zx_slice_color,
        yz_slice_color,
        yx_slice_color,
        ortho_line_width,
    )  # ZX axis

    add_ortho_lines_to_axis(
        axes[1, 2],
        (zyx_slice[1], zyx_slice[0]),
        ("y", "z"),
        yz_slice_color,
        yx_slice_color,
        zx_slice_color,
        ortho_line_width,
    )  # YZ axis

    print(f"Saving {filename}")
    fig.savefig(filename, dpi=400, format="pdf", bbox_inches="tight")


def add_ortho_lines_to_axis(
    axis,
    yx_slice,
    axis_labels,
    outer_color,
    vertical_color,
    horizontal_color,
    line_width=0,
    text_color="white",
):

    xmin, xmax = axis.get_xlim()
    ymin, ymax = axis.get_ylim()

    # Axis labels
    horizontal_axis_label_pos = (0.1, 0.975)
    vertical_axis_label_pos = (0.025, 0.9)
    axis.text(
        horizontal_axis_label_pos[0],
        horizontal_axis_label_pos[1],
        axis_labels[1],
        horizontalalignment="left",
        verticalalignment="top",
        transform=axis.transAxes,
        fontsize=5,
        color=text_color,
    )

    axis.text(
        vertical_axis_label_pos[0],
        vertical_axis_label_pos[1],
        axis_labels[0],
        horizontalalignment="left",
        verticalalignment="top",
        transform=axis.transAxes,
        fontsize=5,
        color=text_color,
    )

    # Outer rectangle
    axis.add_artist(
        plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=line_width,
            edgecolor=outer_color,
            facecolor="none",
            transform=axis.transData,
            clip_on=False,
        )
    )

    # Horizontal line
    axis.add_artist(
        plt.Line2D(
            [xmin, xmax],
            [yx_slice[0], yx_slice[0]],
            transform=axis.transData,
            color=horizontal_color,
            lw=line_width,
        )
    )

    # Vertical line
    axis.add_artist(
        plt.Line2D(
            [yx_slice[1], yx_slice[1]],
            [ymin, ymax],
            transform=axis.transData,
            color=vertical_color,
            lw=line_width,
        )
    )
