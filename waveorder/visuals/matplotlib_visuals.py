import matplotlib.pyplot as plt
import numpy as np


def plot_5d_ortho(
    rcCzyx_data: np.ndarray,
    filename: str,
    voxel_size: tuple[float, float, float],
    zyx_slice: tuple[int, int, int],
    color_funcs: list[list[callable]],
    row_labels: list[str] = None,
    column_labels: list[str] = None,
    rose_path: str = None,
    inches_per_column: float = 1.5,
    label_size: int = 1,
    ortho_line_width: float = 0.5,
    row_column_line_width: float = 0.5,
    xyz_labels: bool = True,
    background_color: str = "white",
    **kwargs: dict,
) -> None:
    """
    Plot 5D multi-channel data in a grid or ortho-slice views.

    Input data is a 6D array with (row, column, channels, Z, Y, X) dimensions.

    `color_funcs` permits different RGB color maps for each row and column.

    Parameters
    ----------
    rcCzyx_data : numpy.ndarray
        5D array with shape (R, C, Ch, Z, Y, X) containing the data to plot.
        [r]ows and [c]olumns form a grid
        [C]hannels contain multiple color channels
        [ZYX] contain 3D volumes.
    filename : str
        Path to save the output plot.
    voxel_size : tuple[float, float, float]
        Size of each voxel in (Z, Y, X) dimensions.
    zyx_slice : tuple[int, int, int]
        Indices of the ortho-slices to plot in (Z, Y, X) indices.
    color_funcs : list[list[callable]]
        A list of lists of callables, one for each element of the plot grid,
        with len(color_funcs) == R and len(colors_funcs[0] == C).
        Each callable accepts [C]hannel arguments and returns RGB color values,
        enabling different RGB color maps for each member of the grid.
    row_labels : list[str], optional
        Labels for the rows, by default None.
    column_labels : list[str], optional
        Labels for the columns, by default None.
    rose_path : str, optional
        Path to an image to display in the top-left corner, by default None.
    inches_per_column : float, optional
        Width of each column in inches, by default 1.5.
    label_size : int, optional
        Size of the labels, by default 1.
    ortho_line_width : float, optional
        Width of the orthogonal lines, by default 0.5.
    row_column_line_width : float, optional
        Width of the lines between rows and columns, by default 0.5.
    xyz_labels : bool, optional
        Whether to display XYZ labels, by default True.
    background_color : str, optional
        Background color of the plot, by default "white".
    **kwargs : dict
        Additional keyword arguments passed to color_funcs.
    """
    R, C, Ch, Z, Y, X = rcCzyx_data.shape

    # Extent
    dZ, dY, dX = Z * voxel_size[0], Y * voxel_size[1], X * voxel_size[2]

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
    fig.patch.set_facecolor(background_color)
    for ax in axes.flat:
        ax.set_facecolor(background_color)

    if rose_path is not None:
        axes[0, 0].imshow(plt.imread(rose_path))

    for i in range(n_rows):
        for j in range(n_cols):
            # Add labels
            if (i == 0 and (j - 1) % 2 == 0) or (j == 0 and (i - 1) % 2 == 0):
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
        ("y", "x") if xyz_labels else ("", ""),
        yx_slice_color,
        yz_slice_color,
        zx_slice_color,
        ortho_line_width,
    )  # YX axis

    add_ortho_lines_to_axis(
        axes[2, 1],
        (zyx_slice[0], zyx_slice[2]),
        ("z", "x") if xyz_labels else ("", ""),
        zx_slice_color,
        yz_slice_color,
        yx_slice_color,
        ortho_line_width,
    )  # ZX axis

    add_ortho_lines_to_axis(
        axes[1, 2],
        (zyx_slice[1], zyx_slice[0]),
        ("y", "z") if xyz_labels else ("", ""),
        yz_slice_color,
        yx_slice_color,
        zx_slice_color,
        ortho_line_width,
    )  # YZ axis

    print(f"Saving {filename}")
    fig.savefig(filename, dpi=400, format="pdf", bbox_inches="tight")


def add_ortho_lines_to_axis(
    axis: plt.Axes,
    yx_slice: tuple[int, int],
    axis_labels: tuple[str, str],
    outer_color: str,
    vertical_color: str,
    horizontal_color: str,
    line_width: float = 0,
    text_color: str = "white",
) -> None:
    """
    Add orthogonal lines and labels to a given axis.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        The axis to which the orthogonal lines and labels will be added.
    yx_slice : tuple[int, int]
        The (Y, X) slice indices for the orthogonal lines.
    axis_labels : tuple[str, str]
        The labels for the Y and X axes.
    outer_color : str
        The color of the outer rectangle.
    vertical_color : str
        The color of the vertical line.
    horizontal_color : str
        The color of the horizontal line.
    line_width : float, optional
        The width of the lines, by default 0.
    text_color : str, optional
        The color of the text labels, by default "white".
    """
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
