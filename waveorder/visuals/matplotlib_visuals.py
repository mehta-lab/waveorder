import matplotlib.pyplot as plt

from waveorder.visuals.utils import complex_tensor_to_rgb
from waveorder.sampling import nd_fourier_central_cuboid

import numpy as np
import os
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
    trim_edges=0,
):
    S, F, Z, Y, X = sfZYX_data.shape
    voxel_scale = [
        Z * zyx_scale[0],
        Y * zyx_scale[1],
        X * zyx_scale[2],
    ]

    sfZYX_data = nd_fourier_central_cuboid(
        sfZYX_data, (S, F, Z - trim_edges, Y - trim_edges, X - trim_edges)
    )

    S, F, Z, Y, X = sfZYX_data.shape
    sfZYX_data = torch.fft.ifftshift(sfZYX_data, dim=(-3, -2, -1))
    sfZYX_rgb = complex_tensor_to_rgb(sfZYX_data, saturate_clim_fraction)

    assert S == len(s_labels)
    assert F == len(f_labels)

    X_size = 1 * inches_per_column
    Y_size = (zyx_scale[2] / zyx_scale[1]) * inches_per_column
    Z_size = (zyx_scale[2] / zyx_scale[0]) * inches_per_column


    n_rows = S + 1
    n_cols = (F * 3) + 1

    height = n_rows
    width = n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(width, height),
        gridspec_kw={
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
            if (i == 0 and (j - 1) % 3 == 1) or (j == 0 and i > 0):
                if i == 0:
                    folder = "gellman"
                    index = f_labels[int((j - 1) / 3)]
                else:
                    folder = "stokes"
                    index = s_labels[i - 1]
                
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
                if (j - 1) % 3 == 0:
                    axes[i, j].imshow(
                        sfZYX_rgb[i - 1, int((j - 1) / 3), (Z // 2) + z_slice],
                        aspect=voxel_scale[1] / voxel_scale[2],
                        #interpolation=None
                    )
                elif (j - 1) % 3 == 1:
                    axes[i, j].imshow(
                        sfZYX_rgb[i - 1, int((j - 1) / 3), :, :, X // 2, :],
                        aspect=voxel_scale[2] / voxel_scale[0],
                    )
                elif (j - 1) % 3 == 2:
                    axes[i, j].imshow(
                        sfZYX_rgb[i - 1, int((j - 1) / 3), :, Y // 2],
                        aspect=voxel_scale[1] / voxel_scale[0],
                    )

            axes[i, j].tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )

            # Draw lines between rows and cols
            top = axes[0, 0].get_position().y1
            bottom = axes[-1, -1].get_position().y0
            left = axes[0, 0].get_position().x0
            right = axes[-1, -1].get_position().x1
            if i == 0 and (j - 1) % 3 == 0:
                left_edge = (axes[0, j].get_position().x0 + axes[0, j - 1].get_position().x1)/2
                fig.add_artist(
                    plt.Line2D(
                        [left_edge, left_edge],
                        [bottom, top],
                        transform=fig.transFigure,
                        color="black",
                        lw=0.5,
                    )
                )
            if j == 0 and i > 0:
                top_edge = (axes[i, 0].get_position().y1 + axes[i - 1, 0].get_position().y0)/2
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
            #[(X/2) - 0.5*X/np.sqrt(2) , (X/2) + 0.5*X/np.sqrt(2)],
            #[(Y/2) - 0.5*Y/np.sqrt(2) , (Y/2) + 0.5*Y/np.sqrt(2)],  
            [Y//2, Y//2],
            [0, X],
            transform=axes[1, 1].transData,  # use axis coordinates
            color="red",
            lw=0.5,
        )
    )
    fig.add_artist(                    
        plt.Line2D(
            [0, X],
            [Y//2, Y//2],  # from bottom to top in axis coordinates
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
            edgecolor='green',
            facecolor='none',
            transform=axes[1, 1].transData  # use axis coordinates
        )
    )
    axes[1, 1].text(
        0.1, 0.975, 'x', 
        horizontalalignment='left', 
        verticalalignment='top', 
        transform=axes[1, 1].transAxes, 
        fontsize=5, 
        color='black'
    )
    axes[1, 1].text(
        0.025, 0.9, 'y', 
        horizontalalignment='left', 
        verticalalignment='top', 
        transform=axes[1, 1].transAxes, 
        fontsize=5, 
        color='black'
    )
    
    fig.add_artist(                    
        plt.Line2D(
            [0, X],
            [Z//2 + z_slice, Z//2 + z_slice],  # from bottom to top in axis coordinates
            transform=axes[1, 2].transData,  # use axis coordinates
            color="green",
            lw=0.5,
        )
    )
    fig.add_artist(                    
        plt.Line2D(
            [X//2, X//2],
            [0, Z],  # from bottom to top in axis coordinates
            transform=axes[1, 2].transData,  # use axis coordinates
            color="blue",
            lw=0.5,
        )
    )

    rect = plt.Rectangle(
        (0, 0),  # lower-left corner
        X,  # width
        Z,  # height
        linewidth=0.5,
        edgecolor='red',
        facecolor='none',
        transform=axes[1, 2].transData  # use axis coordinates
    )
    fig.add_artist(rect)

    axes[1, 2].text(
        0.1, 0.975, 'y', 
        horizontalalignment='left', 
        verticalalignment='top', 
        transform=axes[1, 2].transAxes, 
        fontsize=5, 
        color='black'
    )
    axes[1, 2].text(
        0.025, 0.9, 'z', 
        horizontalalignment='left', 
        verticalalignment='top', 
        transform=axes[1, 2].transAxes, 
        fontsize=5, 
        color='black'
    )
    

    fig.add_artist(                    
        plt.Line2D(
            [0, X],
            [Z//2 + z_slice, Z//2 + z_slice],  # from bottom to top in axis coordinates
            transform=axes[1, 3].transData,  # use axis coordinates
            color="green",
            lw=0.5,
        )
    )
    fig.add_artist(                    
        plt.Line2D(
            [X//2, X//2],
            [0, Z],  # from bottom to top in axis coordinates
            transform=axes[1, 3].transData,  # use axis coordinates
            color="red",
            lw=0.5,
        )
    )

    rect = plt.Rectangle(
        (0, 0),  # lower-left corner
        X,  # width
        Z,  # height
        linewidth=0.5,
        edgecolor='blue',
        facecolor='none',
        transform=axes[1, 3].transData  # use axis coordinates
    )
    fig.add_artist(rect)

    axes[1, 3].text(
        0.1, 0.975, 'x', 
        horizontalalignment='left', 
        verticalalignment='top', 
        transform=axes[1, 3].transAxes, 
        fontsize=5, 
        color='black'
    )
    axes[1, 3].text(
        0.025, 0.9, 'z', 
        horizontalalignment='left', 
        verticalalignment='top', 
        transform=axes[1, 3].transAxes, 
        fontsize=5, 
        color='black'
    )

    print(f"Saving {filename}")
    fig.savefig(filename, dpi=300, format="pdf", bbox_inches="tight")
