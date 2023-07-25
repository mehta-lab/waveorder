import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import os
import io
from PIL import Image as PImage
from ipywidgets import (
    Image,
    Layout,
    interact,
    interactive,
    fixed,
    interact_manual,
    HBox,
    VBox,
)
from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import Normalize
from scipy.ndimage import uniform_filter
from scipy.stats import binned_statistic_2d

from numpy.typing import NDArray


def im_bit_convert(im, bit=16, norm=False, limit=[]):
    im = im.astype(
        np.float32, copy=False
    )  # convert to float32 without making a copy to save memory
    if norm:
        if not limit:
            limit = [
                np.nanmin(im[:]),
                np.nanmax(im[:]),
            ]  # scale each image individually based on its min and max
        im = (im - limit[0]) / (limit[1] - limit[0]) * (2**bit - 1)
    im = np.clip(
        im, 0, 2**bit - 1
    )  # clip the values to avoid wrap-around by np.astype
    if bit == 8:
        im = im.astype(np.uint8, copy=False)  # convert to 8 bit
    else:
        im = im.astype(np.uint16, copy=False)  # convert to 16 bit
    return im


def im_adjust(img, tol=1, bit=8):
    """
    Adjust contrast of the image

    """
    limit = np.percentile(img, [tol, 100 - tol])
    im_adjusted = im_bit_convert(img, bit=bit, norm=True, limit=limit.tolist())
    return im_adjusted


def array2png_bytes(img: NDArray):
    """encode numpy array in 8-bit png bytes"""
    image = PImage.fromarray(img)
    png = io.BytesIO()
    image.save(png, format="png")
    png.seek(0)
    return png.read()


def image_stack_viewer(
    image_stack, size=(10, 10), colormap="gray", origin="upper"
):
    """

    visualize 3D and 4D image stack interactively in jupyter notebook (or jupyter lab)

    Parameters
    ----------
        image_stack : numpy.ndarray
                      a 3D or 4D image stack with the size of (N_stack, Ny, Nx) or (N_stack, Nchannel, Ny, Nx)

        size        : tuple
                      the size of the figure panel (width, height)

        colormap    : str
                      the colormap of the display figure (from the colormap of the matplotlib library)

        origin      : str
                      option to set the origin of the array to the top ('upper') or to the bottom ('lower')

    Returns
    -------
        a interactive widget shown in the output cell of the jupyter notebook (or jupyter lab)

    """

    max_val = np.max(image_stack)
    min_val = np.min(image_stack)

    def interact_plot_3D(stack_idx):
        plt.figure(figsize=size)
        plt.imshow(
            image_stack[stack_idx],
            cmap=colormap,
            vmin=min_val,
            vmax=max_val,
            origin=origin,
        )
        plt.colorbar()
        plt.show()

    def interact_plot_4D(stack_idx_1, stack_idx_2):
        plt.figure(figsize=size)
        plt.imshow(
            image_stack[stack_idx_1, stack_idx_2],
            cmap=colormap,
            vmin=min_val,
            vmax=max_val,
            origin=origin,
        )
        plt.colorbar()
        plt.show()

    if image_stack.ndim == 3:
        return interact(
            interact_plot_3D,
            stack_idx=widgets.IntSlider(
                value=0, min=0, max=len(image_stack) - 1, step=1
            ),
        )
    else:
        return interact(
            interact_plot_4D,
            stack_idx_1=widgets.IntSlider(
                value=0, min=0, max=image_stack.shape[0] - 1, step=1
            ),
            stack_idx_2=widgets.IntSlider(
                value=0, min=0, max=image_stack.shape[1] - 1, step=1
            ),
        )


def image_stack_viewer_fast(
    image_stack, size=(512, 512), origin="upper", vrange=None
):
    """

    faster function to visualize 3D image stack interactively in jupyter notebook (or jupyter lab)

    Parameters
    ----------
        image_stack : numpy.ndarray
                      a 3D image stack with the size of (N_stack, Ny, Nx)

        size        : tuple
                      the size of the figure panel (width, height)

        origin      : str
                      option to set the origin of the array to the top ('upper') or to the bottom ('lower')

    Returns
    -------
        a interactive widget shown in the output cell of the jupyter notebook (or jupyter lab)

    """
    if vrange is None:
        imgs = im_adjust(image_stack, tol=0, bit=16)
    elif vrange[0] < vrange[1]:
        imgs = im_bit_convert(image_stack, bit=16, norm=True, limit=vrange)
    else:
        raise ValueError(
            "vrange needs to be a two element list with vrange[0] < vrange[1]"
        )

    im_dict = {}
    for idx, img in enumerate(imgs):
        if origin == "upper":
            im_dict[idx] = array2png_bytes(img)
        elif origin == "lower":
            im_dict[idx] = array2png_bytes(np.flipud(img))
        else:
            raise ValueError('origin can only be either "upper" or "lower"')

    im_wgt = Image(
        value=im_dict[0],
        layout=Layout(height=str(size[0]) + "px", width=str(size[1]) + "px"),
    )

    def interact_plot_3D(stack_idx):
        im_wgt.value = im_dict[stack_idx]

    interact(
        interact_plot_3D,
        stack_idx=widgets.IntSlider(
            value=0, min=0, max=len(im_dict) - 1, step=1
        ),
    )

    return HBox([im_wgt])


def hsv_stack_viewer(image_stack, max_val=1, size=5, origin="upper"):
    """

    visualize 3D retardance + orientation image stack with hsv colormap (orientation in h, constant in s, retardance in v)
    interactively in jupyter notebook (or jupyter lab)

    Parameters
    ----------
        image_stack : numpy.ndarray
                      a 3D image stack with size of  (Nchannel, N_stack, Ny, Nx)
                      the 0 index corresponds to the orientation stack (range from 0 to pi)
                      the 1 index corresponds to the retardance stack

        max_val     : float
                      raise the brightness of the retardance channel by 1/max_val

        size        : int
                      the size of the figure panel (size, size)

        origin      : str
                      option to set the origin of the array to the top ('upper') or to the bottom ('lower')

    Returns
    -------
        a interactive widget shown in the output cell of the jupyter notebook (or jupyter lab)

    """

    image_stack1 = image_stack[0]
    image_stack2 = image_stack[1]

    N_stack = len(image_stack1)
    I_rgb = np.zeros(image_stack1.shape + (3,))

    for i in range(N_stack):
        I_hsv = np.transpose(
            np.stack(
                [
                    image_stack1[i] / np.pi,
                    np.ones_like(image_stack1[i]),
                    np.minimum(
                        1, image_stack2[i] / np.max(image_stack2[i]) / max_val
                    ),
                ]
            ),
            (1, 2, 0),
        )
        I_rgb[i] = hsv_to_rgb(I_hsv)

    V, H = np.mgrid[0:1:500j, 0:1:500j]
    S = np.ones_like(V)
    HSV = np.dstack((V, S, H))
    RGB = hsv_to_rgb(HSV)

    def interact_plot_hsv(stack_idx):
        f1, ax = plt.subplots(1, 2, figsize=(size + size / 2, size))
        ax[0].imshow(I_rgb[stack_idx], origin=origin)

        ax[1].imshow(RGB, origin="lower", extent=[0, 1, 0, 180], aspect=0.2)
        plt.xlabel("V")
        plt.ylabel("H")
        plt.title("$S_{HSV}=1$")

        plt.tight_layout()
        plt.show()

    return interact(
        interact_plot_hsv,
        stack_idx=widgets.IntSlider(
            value=0, min=0, max=len(image_stack1) - 1, step=1
        ),
    )


def rgb_stack_viewer(image_stack, size=5, origin="upper"):
    """

    visualize 3D rgb image stack interactively in jupyter notebook (or jupyter lab)

    Parameters
    ----------
        image_stack : numpy.ndarray
                      a 3D rgb image stack with the size of  (N_stack, Ny, Nx, 3)

        size        : int
                      the size of the figure panel (size, size)

        origin      : str
                      option to set the origin of the array to the top ('upper') or to the bottom ('lower')

    Returns
    -------
        a interactive widget shown in the output cell of the jupyter notebook (or jupyter lab)

    """

    def interact_plot_rgb(stack_idx):
        plt.figure(figsize=(size, size))
        plt.imshow(image_stack[stack_idx], origin=origin)
        plt.show()

    return interact(
        interact_plot_rgb,
        stack_idx=widgets.IntSlider(
            value=0, min=0, max=len(image_stack) - 1, step=1
        ),
    )


def rgb_stack_viewer_fast(image_stack, size=(256, 256), origin="upper"):
    """

    visualize 3D rgb image stack interactively in jupyter notebook (or jupyter lab)

    Parameters
    ----------
        image_stack : numpy.ndarray
                      a 3D rgb image stack with the size of  (N_stack, Ny, Nx, 3)

        size        : int
                      the size of the figure panel (size, size)

        origin      : str
                      option to set the origin of the array to the top ('upper') or to the bottom ('lower')

    Returns
    -------
        a interactive widget shown in the output cell of the jupyter notebook (or jupyter lab)

    """
    imgs = np.zeros_like(image_stack, dtype=np.uint8)
    imgs[:, :, :, 0] = np.uint8(image_stack[:, :, :, 2] * 255)
    imgs[:, :, :, 1] = np.uint8(image_stack[:, :, :, 1] * 255)
    imgs[:, :, :, 2] = np.uint8(image_stack[:, :, :, 0] * 255)

    im_dict = {}
    for idx, img in enumerate(imgs):
        if origin == "upper":
            im_dict[idx] = array2png_bytes(img)
        elif origin == "lower":
            im_dict[idx] = array2png_bytes(np.flipud(img))
        else:
            raise ValueError('origin can only be either "upper" or "lower"')

    im_wgt = Image(
        value=im_dict[0],
        layout=Layout(height=str(size[0]) + "px", width=str(size[1]) + "px"),
    )

    def interact_plot_3D(stack_idx):
        im_wgt.value = im_dict[stack_idx]

    interact(
        interact_plot_3D,
        stack_idx=widgets.IntSlider(
            value=0, min=0, max=len(im_dict) - 1, step=1
        ),
    )

    return HBox([im_wgt])


def parallel_4D_viewer(
    image_stack,
    num_col=2,
    size=10,
    set_title=False,
    titles=[],
    colormap="gray",
    origin="upper",
    vrange=None,
):
    """

    simultaneous visualize all channels of image stack interactively in jupyter notebook

    Parameters
    ----------
        image_stack : numpy.ndarray
                      a 4D image with the size of (N_stack, Nchannel, N, M)

        num_col     : int
                      number of columns you wish to display

        size        : int
                      the size of one figure panel

        set_title   : bool
                      options for setting up titles of the figures

        titles      : list
                      list of titles for the figures

        colormap    : str
                      the colormap of the display figure (from the colormap of the matplotlib library)

        origin      : str
                      option to set the origin of the array to the top ('upper') or to the bottom ('lower')

        vrange      : list
                      list of range (two numbers) for all the image panels

    Returns
    -------
        a interactive widget shown in the output cell of the jupyter notebook (or jupyter lab)

    """

    N_stack, N_channel, _, _ = image_stack.shape
    if set_title == True and len(titles) != N_channel:
        raise ValueError(
            "number of titles does not match with the number of channels"
        )
    num_row = int(np.ceil(N_channel / num_col))
    figsize = (num_col * size, num_row * size)

    def interact_plot(stack_idx):
        if vrange is None:
            f1, ax = plt.subplots(num_row, num_col, figsize=figsize)
            if num_row == 1:
                for i in range(N_channel):
                    col_idx = np.mod(i, num_col)
                    ax1 = ax[col_idx].imshow(
                        image_stack[stack_idx, i], cmap=colormap, origin=origin
                    )
                    plt.colorbar(ax1, ax=ax[col_idx])
                    if set_title == True:
                        ax[col_idx].set_title(titles[i])
                plt.show()
            elif num_col == 1:
                for i in range(N_channel):
                    row_idx = i // num_col
                    ax1 = ax[row_idx].imshow(
                        image_stack[stack_idx, i], cmap=colormap, origin=origin
                    )
                    plt.colorbar(ax1, ax=ax[row_idx])
                    if set_title == True:
                        ax[row_idx].set_title(titles[i])
                plt.show()
            else:
                for i in range(N_channel):
                    row_idx = i // num_col
                    col_idx = np.mod(i, num_col)
                    ax1 = ax[row_idx, col_idx].imshow(
                        image_stack[stack_idx, i], cmap=colormap, origin=origin
                    )
                    plt.colorbar(ax1, ax=ax[row_idx, col_idx])
                    if set_title == True:
                        ax[row_idx, col_idx].set_title(titles[i])
                plt.show()

        elif len(vrange) == 2 and vrange[0] < vrange[1]:
            f1, ax = plt.subplots(num_row, num_col, figsize=figsize)
            if num_row == 1:
                for i in range(N_channel):
                    col_idx = np.mod(i, num_col)
                    ax1 = ax[col_idx].imshow(
                        image_stack[stack_idx, i],
                        cmap=colormap,
                        origin=origin,
                        vmin=vrange[0],
                        vmax=vrange[1],
                    )
                    plt.colorbar(ax1, ax=ax[col_idx])
                    if set_title == True:
                        ax[col_idx].set_title(titles[i])
                plt.show()
            elif num_col == 1:
                for i in range(N_channel):
                    row_idx = i // num_col
                    ax1 = ax[row_idx].imshow(
                        image_stack[stack_idx, i],
                        cmap=colormap,
                        origin=origin,
                        vmin=vrange[0],
                        vmax=vrange[1],
                    )
                    plt.colorbar(ax1, ax=ax[row_idx])
                    if set_title == True:
                        ax[row_idx].set_title(titles[i])
                plt.show()
            else:
                for i in range(N_channel):
                    row_idx = i // num_col
                    col_idx = np.mod(i, num_col)
                    ax1 = ax[row_idx, col_idx].imshow(
                        image_stack[stack_idx, i],
                        cmap=colormap,
                        origin=origin,
                        vmin=vrange[0],
                        vmax=vrange[1],
                    )
                    plt.colorbar(ax1, ax=ax[row_idx, col_idx])
                    if set_title == True:
                        ax[row_idx, col_idx].set_title(titles[i])
                plt.show()

        else:
            raise ValueError(
                "range should be a list with length of 2 and range[0]<range[1]"
            )

    return interact(
        interact_plot,
        stack_idx=widgets.IntSlider(value=0, min=0, max=N_stack - 1, step=1),
    )


def parallel_4D_viewer_fast(
    image_stack, num_col=2, size=256, origin="upper", vrange=None
):
    """

    simultaneous visualize all channels of image stack interactively in jupyter notebook

    Parameters
    ----------
        image_stack : numpy.ndarray
                      a 4D image with the size of (N_stack, Nchannel, N, M)

        num_col     : int
                      number of columns you wish to display

        size        : int
                      the size of one figure panel

        origin      : str
                      option to set the origin of the array to the top ('upper') or to the bottom ('lower')

        vrange      : list
                      list of range (two numbers) for all the image panels

    Returns
    -------
        a interactive widget shown in the output cell of the jupyter notebook (or jupyter lab)

    """
    N_stack, N_channel, _, _ = image_stack.shape

    list_of_widgets = []
    list_of_img_binaries = []

    for i in range(N_channel):
        if vrange is None:
            imgs = im_adjust(image_stack[:, i], tol=0, bit=16)
        elif vrange[0] < vrange[1]:
            imgs = im_bit_convert(
                image_stack[:, i], bit=16, norm=True, limit=vrange
            )
        else:
            raise ValueError(
                "vrange needs to be a two element list with vrange[0] < vrange[1]"
            )

        list_of_img_binaries.append({})
        for idx, img in enumerate(imgs):
            if origin == "upper":
                list_of_img_binaries[i][idx] = array2png_bytes(img)
            elif origin == "lower":
                list_of_img_binaries[i][idx] = array2png_bytes(np.flipud(img))
            else:
                raise ValueError(
                    'origin can only be either "upper" or "lower"'
                )

        list_of_widgets.append(
            Image(
                value=list_of_img_binaries[i][0],
                layout=Layout(height=str(size) + "px", width=str(size) + "px"),
            )
        )

    def interact_plot(stack_idx):
        for i in range(N_channel):
            list_of_widgets[i].value = list_of_img_binaries[i][stack_idx]

    interact(
        interact_plot,
        stack_idx=widgets.IntSlider(value=0, min=0, max=N_stack - 1, step=1),
    )

    return widgets.GridBox(
        list_of_widgets,
        layout=widgets.Layout(
            grid_template_columns="repeat("
            + str(num_col)
            + ","
            + str(size + 10)
            + "px)"
        ),
    )


def parallel_5D_viewer(
    image_stack,
    num_col=2,
    size=10,
    set_title=False,
    titles=[],
    colormap="gray",
    origin="upper",
):
    """

    simultaneous visualize all channels of image stack interactively in jupyter notebook with two stepping nobs on N_stack and N_pattern

    Parameters
    ----------
        image_stack : numpy.ndarray
                      a 5D image with the size of (N_stack, N_pattern, Nchannel, N, M)

        num_col     : int
                      number of columns you wish to display

        size        : int
                      the size of one figure panel

        set_title   : bool
                      options for setting up titles of the figures

        titles      : list
                      list of titles for the figures

        colormap    : str
                      the colormap of the display figure (from the colormap of the matplotlib library)

        origin      : str
                      option to set the origin of the array to the top ('upper') or to the bottom ('lower')

    Returns
    -------
        a interactive widget shown in the output cell of the jupyter notebook (or jupyter lab)

    """

    N_stack, N_pattern, N_channel, _, _ = image_stack.shape
    if set_title == True and len(titles) != N_channel:
        raise ValueError(
            "number of titles does not match with the number of channels"
        )
    num_row = int(np.ceil(N_channel / num_col))
    figsize = (num_col * size, num_row * size)

    def interact_plot(stack_idx_1, stack_idx_2):
        f1, ax = plt.subplots(num_row, num_col, figsize=figsize)
        if num_row == 1:
            for i in range(N_channel):
                col_idx = np.mod(i, num_col)
                ax1 = ax[col_idx].imshow(
                    image_stack[stack_idx_1, stack_idx_2, i],
                    cmap=colormap,
                    origin=origin,
                )
                plt.colorbar(ax1, ax=ax[col_idx])
                if set_title == True:
                    ax[col_idx].set_title(titles[i])
            plt.show()
        else:
            for i in range(N_channel):
                row_idx = i // num_col
                col_idx = np.mod(i, num_col)
                ax1 = ax[row_idx, col_idx].imshow(
                    image_stack[stack_idx_1, stack_idx_2, i],
                    cmap=colormap,
                    origin=origin,
                )
                plt.colorbar(ax1, ax=ax[row_idx, col_idx])
                if set_title == True:
                    ax[row_idx, col_idx].set_title(titles[i])
            plt.show()

    return interact(
        interact_plot,
        stack_idx_1=widgets.IntSlider(value=0, min=0, max=N_stack - 1, step=1),
        stack_idx_2=widgets.IntSlider(
            value=0, min=0, max=N_pattern - 1, step=1
        ),
    )


def parallel_5D_viewer_fast(
    image_stack, num_col=2, size=256, origin="upper", vrange=None
):
    """

    simultaneous visualize all channels of image stack interactively in jupyter notebook with two stepping nobs on N_stack and N_pattern

    Parameters
    ----------
        image_stack : numpy.ndarray
                      a 5D image with the size of (N_stack, N_pattern, Nchannel, N, M)

        num_col     : int
                      number of columns you wish to display

        size        : int
                      the size of one figure panel

        set_title   : bool
                      options for setting up titles of the figures

        titles      : list
                      list of titles for the figures

        colormap    : str
                      the colormap of the display figure (from the colormap of the matplotlib library)

        origin      : str
                      option to set the origin of the array to the top ('upper') or to the bottom ('lower')

    Returns
    -------
        a interactive widget shown in the output cell of the jupyter notebook (or jupyter lab)

    """

    N_stack, N_pattern, N_channel, _, _ = image_stack.shape

    list_of_widgets = []
    list_of_list_img_binaries = []

    for j in range(N_pattern):
        list_of_img_binaries = []
        for i in range(N_channel):
            if vrange is None:
                imgs = im_adjust(image_stack[:, j, i], tol=0, bit=16)
            elif vrange[0] < vrange[1]:
                imgs = im_bit_convert(
                    image_stack[:, i], bit=16, norm=True, limit=vrange
                )
            else:
                raise ValueError(
                    "vrange needs to be a two element list with vrange[0] < vrange[1]"
                )

            list_of_img_binaries.append({})
            for idx, img in enumerate(imgs):
                if origin == "upper":
                    list_of_img_binaries[i][idx] = array2png_bytes(img)
                elif origin == "lower":
                    list_of_img_binaries[i][idx] = array2png_bytes(
                        np.flipud(img)
                    )
                else:
                    raise ValueError(
                        'origin can only be either "upper" or "lower"'
                    )

            if j == 0:
                list_of_widgets.append(
                    Image(
                        value=list_of_img_binaries[i][0],
                        layout=Layout(
                            height=str(size) + "px", width=str(size) + "px"
                        ),
                    )
                )

        list_of_list_img_binaries.append(list_of_img_binaries)

    def interact_plot(stack_idx_1, stack_idx_2):
        for i in range(N_channel):
            list_of_widgets[i].value = list_of_list_img_binaries[stack_idx_2][
                i
            ][stack_idx_1]

    interact(
        interact_plot,
        stack_idx_1=widgets.IntSlider(value=0, min=0, max=N_stack - 1, step=1),
        stack_idx_2=widgets.IntSlider(
            value=0, min=0, max=N_pattern - 1, step=1
        ),
    )

    return widgets.GridBox(
        list_of_widgets,
        layout=widgets.Layout(
            grid_template_columns="repeat("
            + str(num_col)
            + ","
            + str(size + 10)
            + "px)"
        ),
    )


def plot_multicolumn(
    image_stack,
    num_col=2,
    size=10,
    set_title=False,
    titles=[],
    colormap="gray",
    origin="upper",
):
    """

    plot images in multiple columns

    Parameters
    ----------
        image_stack : numpy.ndarray
                      image stack with the size of (N_stack, N, M)

        num_col     : int
                      number of columns you wish to display

        size        : int
                      the size of one figure panel

        set_title   : bool
                      options for setting up titles of the figures

        titles      : list
                      list of titles for the figures

        colormap    : str
                      the colormap of the display figure (from the colormap of the matplotlib library)

        origin      : str
                      option to set the origin of the array to the top ('upper') or to the bottom ('lower')

    """

    N_stack = len(image_stack)
    num_row = int(np.ceil(N_stack / num_col))
    figsize = (num_col * size, num_row * size)

    f1, ax = plt.subplots(num_row, num_col, figsize=figsize)

    if num_row == 1:
        for i in range(N_stack):
            col_idx = np.mod(i, num_col)
            ax1 = ax[col_idx].imshow(
                image_stack[i], cmap=colormap, origin=origin
            )
            plt.colorbar(ax1, ax=ax[col_idx])

            if set_title == True:
                ax[col_idx].set_title(titles[i])
    else:
        for i in range(N_stack):
            row_idx = i // num_col
            col_idx = np.mod(i, num_col)
            ax1 = ax[row_idx, col_idx].imshow(
                image_stack[i], cmap=colormap, origin=origin
            )
            plt.colorbar(ax1, ax=ax[row_idx, col_idx])

            if set_title == True:
                ax[row_idx, col_idx].set_title(titles[i])


def plot_hsv(image_stack, max_val=1, size=5, origin="upper"):
    """

    visualize retardance + orientation image with hsv colormap (orientation in h, constant in s, retardance in v)

    Parameters
    ----------
        image_stack : numpy.ndarray
                      retardance and orientation image with size of (N_channel, Ny, Nx)
                      the 0 index corresponds to the orientation image (range from 0 to pi)
                      the 1 index corresponds to the retardance image

        max_val     : float
                      raise the brightness of the retardance channel by 1/max_val

        size        : int
                      the size of the figure panel (size, size)

        origin      : str
                      option to set the origin of the array to the top ('upper') or to the bottom ('lower')

    """

    N_channel = len(image_stack)

    if N_channel == 2:
        I_hsv = np.transpose(
            np.array(
                [
                    image_stack[0] / np.pi,
                    np.ones_like(image_stack[0]),
                    np.minimum(
                        1, image_stack[1] / np.max(image_stack[1]) / max_val
                    ),
                ]
            ),
            (1, 2, 0),
        )
        I_rgb = hsv_to_rgb(I_hsv.copy())

        f1, ax = plt.subplots(1, 2, figsize=(size + size / 2, size))
        ax[0].imshow(I_rgb, origin=origin)

        V, H = np.mgrid[0:1:500j, 0:1:500j]
        S = np.ones_like(V)
        HSV = np.dstack((V, S, H))
        RGB = hsv_to_rgb(HSV)

        ax[1].imshow(RGB, origin="lower", extent=[0, 1, 0, 180], aspect=0.2)
        plt.xlabel("V")
        plt.ylabel("H")
        plt.title("$S_{HSV}=1$")

        plt.tight_layout()

    else:
        raise ValueError("plot_hsv does not support N_channel >2 rendering")


def plot_phase_hsv(
    image_stack, max_val_V=1, max_val_S=1, size=5, origin="upper"
):
    """

    visualize retardance + orientation + phase image with hsv colormap (orientation in h, retardance in s, phase in v)

    Parameters
    ----------
        image_stack : numpy.ndarray
                      retardance and orientation image with size of (N_channel, Ny, Nx)
                      the 0 index corresponds to the orientation image (range from 0 to pi)
                      the 1 index corresponds to the retardance image
                      the 2 index corresponds to the phase image

        max_val_V   : float
                      raise the brightness of the phase channel by 1/max_val_V

        max_val_S   : float
                      raise the brightness of the retardance channel by 1/max_val_S

        size        : int
                      the size of the figure panel (size, size)

        origin      : str
                      option to set the origin of the array to the top ('upper') or to the bottom ('lower')

    """

    N_channel = len(image_stack)

    if N_channel == 3:
        I_hsv = np.transpose(
            np.array(
                [
                    image_stack[0] / np.pi,
                    np.clip(
                        image_stack[1] / np.max(image_stack[1]) / max_val_S,
                        0,
                        1,
                    ),
                    np.clip(
                        image_stack[2] / np.max(image_stack[2]) / max_val_V,
                        0,
                        1,
                    ),
                ]
            ),
            (1, 2, 0),
        )
        I_rgb = hsv_to_rgb(I_hsv.copy())

        f1, ax = plt.subplots(1, 2, figsize=(size + size / 2, size))
        ax[0].imshow(I_rgb, origin=origin)

        V, H = np.mgrid[0:1:500j, 0:1:500j]
        S = np.ones_like(V)
        HSV = np.dstack((V, H, S))
        RGB = hsv_to_rgb(HSV)

        ax[1].imshow(RGB, origin="lower", extent=[0, 1, 0, 180], aspect=0.2)
        plt.xlabel("S")
        plt.ylabel("H")
        plt.title("$V_{HSV}=1$")

        plt.tight_layout()

    else:
        raise ValueError("plot_hsv does not support N_channel >3 rendering")


def plotVectorField(
    img,
    orientation,
    anisotropy=1,
    spacing=20,
    window=20,
    linelength=20,
    linewidth=3,
    linecolor="g",
    colorOrient=True,
    cmapOrient="hsv",
    threshold=None,
    alpha=1,
    clim=[None, None],
    cmapImage="gray",
):
    """

    overlays orientation field on the image. Returns matplotlib image axes.

    Parameters
    ----------
        img         : numpy.ndarray
                      image to overlay orientation lines on

        orientation : numpy.ndarray
                      orientation in radian

        anisotropy  : numpy.ndarray
                      magnitude encoded in the line length

        spacing     : int
                      spacing of the line glyphs

        window      : int
                      size of the blurring window for the orientation field

        linelength  : int
                      can be a scalar or an image the same size as the orientation to encode linelength further

        linewidth   : int
                      width of the orientation line

        linecolor   : str
                      for example 'y' -> yellow

        colorOrient : bool
                      if it is True, then color the lines by their orientation

        cmapOrient  : str
                      colormap for coloring the lines by the orientation

        threshold   : numpy.ndarray
                      a binary numpy array, wherever the map is 0, ignore the plotting of the line

        alpha       : int
                      line transparency. [0,1]. lower is more transparent

        clim        : list
                      [min, max], min and max intensities for displaying img

        cmapImage   : str
                      colormap for displaying the image
    Returns
    -------
        im_ax       : obj
                      matplotlib image axes

    """

    # plot vector field representaiton of the orientation map

    # Compute U, V such that they are as long as line-length when anisotropy = 1.
    U, V = anisotropy * linelength * np.cos(
        2 * orientation
    ), anisotropy * linelength * np.sin(2 * orientation)
    USmooth = uniform_filter(U, (window, window))  # plot smoothed vector field
    VSmooth = uniform_filter(V, (window, window))  # plot smoothed vector field
    azimuthSmooth = 0.5 * np.arctan2(VSmooth, USmooth)
    RSmooth = np.sqrt(USmooth**2 + VSmooth**2)
    USmooth, VSmooth = RSmooth * np.cos(azimuthSmooth), RSmooth * np.sin(
        azimuthSmooth
    )

    nY, nX = img.shape
    Y, X = np.mgrid[0:nY, 0:nX]  # notice the reversed order of X and Y

    # Plot sparsely sampled vector lines
    Plotting_X = X[::spacing, ::spacing]
    Plotting_Y = Y[::spacing, ::spacing]
    Plotting_U = linelength * USmooth[::spacing, ::spacing]
    Plotting_V = linelength * VSmooth[::spacing, ::spacing]
    Plotting_R = RSmooth[::spacing, ::spacing]

    if threshold is None:
        threshold = np.ones_like(X)  # no threshold
    Plotting_thres = threshold[::spacing, ::spacing]
    Plotting_orien = (
        ((azimuthSmooth[::spacing, ::spacing]) % np.pi) * 180 / np.pi
    )

    if colorOrient:
        im_ax = plt.imshow(img, cmap=cmapImage, vmin=clim[0], vmax=clim[1])
        plt.title("Orientation map")
        plt.quiver(
            Plotting_X[Plotting_thres == 1],
            Plotting_Y[Plotting_thres == 1],
            Plotting_U[Plotting_thres == 1],
            Plotting_V[Plotting_thres == 1],
            Plotting_orien[Plotting_thres == 1],
            cmap=cmapOrient,
            edgecolor=linecolor,
            facecolor=linecolor,
            units="xy",
            alpha=alpha,
            width=linewidth,
            headwidth=0,
            headlength=0,
            headaxislength=0,
            scale_units="xy",
            scale=1,
            angles="uv",
            pivot="mid",
        )
    else:
        im_ax = plt.imshow(img, cmap=cmapImage, vmin=clim[0], vmax=clim[1])
        plt.title("Orientation map")
        plt.quiver(
            Plotting_X[Plotting_thres == 1],
            Plotting_Y[Plotting_thres == 1],
            Plotting_U[Plotting_thres == 1],
            Plotting_V[Plotting_thres == 1],
            edgecolor=linecolor,
            facecolor=linecolor,
            units="xy",
            alpha=alpha,
            width=linewidth,
            headwidth=0,
            headlength=0,
            headaxislength=0,
            scale_units="xy",
            scale=1,
            angles="uv",
            pivot="mid",
        )

    return im_ax


def orientation_2D_colorwheel(wheelsize=256, circ_size=50):
    """

    generate hsv colorwheel for color-encoded 2D orientation

    Parameters
    ----------
        wheelsize : int
                    size of 2D array to create this colorwheel

        circ_size : int
                    radius of the colorwheel in pixel

    Returns
    -------
        im_ax     : obj
                    matplotlib image axes

    """

    xx_grid, yy_grid = np.meshgrid(
        np.r_[0:wheelsize] - wheelsize // 2,
        np.r_[0:wheelsize] - wheelsize // 2,
    )

    circle_mask = np.zeros_like(xx_grid)
    circle_mask[xx_grid**2 + yy_grid**2 <= circ_size**2] = 1
    orientation = (np.arctan2(yy_grid, xx_grid) % (np.pi)) / np.pi
    value = circle_mask.copy()

    orientation_image = np.transpose(
        np.array([orientation, np.ones_like(circle_mask), value]), (1, 2, 0)
    )
    RGB_image = hsv_to_rgb(orientation_image)

    im_ax = plt.imshow(RGB_image, origin="lower")

    return im_ax


def orientation_3D_colorwheel(
    wheelsize=256,
    circ_size=50,
    interp_belt=40 / 180 * np.pi,
    sat_factor=1,
    discretize=False,
):
    """

    generate colorwheel for color-encoded 3D orientation

    Parameters
    ----------
        wheelsize   : int
                      size of 2D array to create this colorwheel

        circ_size   : int
                      radius of the colorwheel in pixel

        interp_belt : float
                      width of the interpolation belt (in radian) between the top hemi-sphere and bottom hemi-sphere

        sat_factor  : float
                      gamma factor of the saturation value
                      sat_factor > 1 : larger white color range in theta dimension
                      sat_factor < 1 : smaller white color range in theta dimension

        discretize  : bool
                      option to display the discretized top hemisphere of the colorsphere or not

    Returns
    -------
        im_ax       : obj
                      matplotlib image axes

    """

    xx_grid, yy_grid = np.meshgrid(
        np.r_[0:wheelsize] - wheelsize // 2,
        np.r_[0:wheelsize] - wheelsize // 2,
    )

    circle_mask = np.zeros_like(xx_grid)

    if discretize:
        circle_mask[xx_grid**2 + yy_grid**2 <= 1 * circ_size**2] = 1
        rho = ((xx_grid**2 + yy_grid**2) / (circ_size) ** 2) ** (
            0.5
        ) * circle_mask
        inc = np.round(rho * 2, 0) / 2 * np.pi / 2
        orientation = (np.arctan2(yy_grid, xx_grid) % (np.pi * 2)) / 2 / np.pi
        orientation = np.round(orientation * 16, 0) / 16
    else:
        circle_mask[xx_grid**2 + yy_grid**2 <= 4 * circ_size**2] = 1
        rho = ((xx_grid**2 + yy_grid**2) / (circ_size) ** 2) ** (
            0.5
        ) * circle_mask
        inc = rho * np.pi / 2
        orientation = (np.arctan2(yy_grid, xx_grid) % (np.pi * 2)) / 2 / np.pi

    value = circle_mask.copy()

    orientation_image = np.transpose(
        np.array([orientation, inc, value]), (1, 2, 0)
    )
    RGB_image = orientation_3D_to_rgb(
        orientation_image, interp_belt=interp_belt, sat_factor=sat_factor
    )

    theta = np.linspace(0, 2 * np.pi, 1000)

    r = circ_size
    x1 = r * np.cos(theta) + wheelsize // 2
    x2 = r * np.sin(theta) + wheelsize // 2

    im_ax = plt.imshow(RGB_image, origin="lower")
    if not discretize:
        plt.plot(x1, x2, "k")

    return im_ax


def orientation_3D_to_rgb(hsv, interp_belt=20 / 180 * np.pi, sat_factor=1):
    """

    convert [azimuth, theta, retardance] values to rgb according to

    S.-T. Wu, R. Voltoline, and Y. Clarissa Lin,
    "A view-independent line-coding colormap for diffusion tensor imaging," Computer & Graphics, 60, 2016

    Parameters
    ----------
        hsv         : numpy.ndarray
                      array with the shape of (..., 3)
                      [h, s, v] is refered to [hue, saturation(inclination), value]
                      h and v are assumed to be in range [0, 1]
                      s is assumed to be in range of [0, pi]

        interp_belt : float
                      width of the interpolation belt (in radian) between the top hemi-sphere and bottom hemi-sphere

        sat_factor  : float
                      gamma factor of the saturation value
                      sat_factor > 1 : larger white color range in theta dimension
                      sat_factor < 1 : smaller white color range in theta dimension

    Returns
    -------
        rgb         : numpy.ndarray
                      rgb array with the shape of (..., 3)
                      colors converted to RGB values in range [0, 1]

    """

    hsv = np.asarray(hsv)

    # check length of the last dimension, should be _some_ sort of rgb
    if hsv.shape[-1] != 3:
        raise ValueError(
            "Last dimension of input array must be 3; "
            "shape {shp} was found.".format(shp=hsv.shape)
        )

    in_shape = hsv.shape
    hsv = np.array(
        hsv,
        copy=False,
        dtype=np.promote_types(hsv.dtype, np.float32),  # Don't work on ints.
        ndmin=2,  # In case input was 1D.
    )

    s_sign = np.sign(np.pi / 2 - hsv[..., 1])

    theta_merge_1 = (hsv[..., 1] - (np.pi / 2 - interp_belt)) / (
        2 * interp_belt
    )
    theta_merge_2 = 1 - theta_merge_1

    scaling_factor = np.zeros_like(hsv[..., 1])

    idx_scale = theta_merge_1 <= 0
    scaling_factor[idx_scale] = hsv[idx_scale, 1] / (np.pi / 2 - interp_belt)

    idx_scale = np.logical_and(theta_merge_1 > 0, theta_merge_2 > 0)
    scaling_factor[idx_scale] = 1

    idx_scale = theta_merge_2 <= 0
    scaling_factor[idx_scale] = (np.pi - hsv[idx_scale, 1]) / (
        np.pi / 2 - interp_belt
    )

    h = hsv[..., 0]
    s = np.sin(scaling_factor**sat_factor * np.pi / 2)
    v = hsv[..., 2]

    r = np.empty_like(h)
    g = np.empty_like(h)
    b = np.empty_like(h)

    i = (h * 6.0).astype(int)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    idx = np.logical_and(i % 6 == 0, s_sign > 0)
    r[idx] = v[idx]
    g[idx] = p[idx]
    b[idx] = t[idx]

    idx = np.logical_and(i == 1, s_sign > 0)
    r[idx] = q[idx]
    g[idx] = t[idx]
    b[idx] = q[idx]

    idx = np.logical_and(i == 2, s_sign > 0)
    r[idx] = t[idx]
    g[idx] = v[idx]
    b[idx] = p[idx]

    idx = np.logical_and(i == 3, s_sign > 0)
    r[idx] = q[idx]
    g[idx] = q[idx]
    b[idx] = t[idx]

    idx = np.logical_and(i == 4, s_sign > 0)
    r[idx] = p[idx]
    g[idx] = t[idx]
    b[idx] = v[idx]

    idx = np.logical_and(i == 5, s_sign > 0)
    r[idx] = t[idx]
    g[idx] = q[idx]
    b[idx] = q[idx]

    # the other hemisphere

    idx = np.logical_and(i == 3, s_sign < 0)
    r[idx] = v[idx]
    g[idx] = p[idx]
    b[idx] = t[idx]

    idx = np.logical_and(i == 4, s_sign < 0)
    r[idx] = q[idx]
    g[idx] = t[idx]
    b[idx] = q[idx]

    idx = np.logical_and(i == 5, s_sign < 0)
    r[idx] = t[idx]
    g[idx] = v[idx]
    b[idx] = p[idx]

    idx = np.logical_and(i % 6 == 0, s_sign < 0)
    r[idx] = q[idx]
    g[idx] = q[idx]
    b[idx] = t[idx]

    idx = np.logical_and(i == 1, s_sign < 0)
    r[idx] = p[idx]
    g[idx] = t[idx]
    b[idx] = v[idx]

    idx = np.logical_and(i == 2, s_sign < 0)
    r[idx] = t[idx]
    g[idx] = q[idx]
    b[idx] = q[idx]

    # inclination color blending

    idx_blend = np.logical_and(
        i % 6 == 0, np.logical_and(theta_merge_1 > 0, theta_merge_2 > 0)
    )
    r[idx_blend] = (
        v[idx_blend] * theta_merge_2[idx_blend]
        + q[idx_blend] * theta_merge_1[idx_blend]
    )
    g[idx_blend] = (
        p[idx_blend] * theta_merge_2[idx_blend]
        + q[idx_blend] * theta_merge_1[idx_blend]
    )
    b[idx_blend] = (
        t[idx_blend] * theta_merge_2[idx_blend]
        + t[idx_blend] * theta_merge_1[idx_blend]
    )

    idx_blend = np.logical_and(
        i == 1, np.logical_and(theta_merge_1 > 0, theta_merge_2 > 0)
    )
    r[idx_blend] = (
        q[idx_blend] * theta_merge_2[idx_blend]
        + p[idx_blend] * theta_merge_1[idx_blend]
    )
    g[idx_blend] = (
        t[idx_blend] * theta_merge_2[idx_blend]
        + t[idx_blend] * theta_merge_1[idx_blend]
    )
    b[idx_blend] = (
        q[idx_blend] * theta_merge_2[idx_blend]
        + v[idx_blend] * theta_merge_1[idx_blend]
    )

    idx_blend = np.logical_and(
        i == 2, np.logical_and(theta_merge_1 > 0, theta_merge_2 > 0)
    )
    r[idx_blend] = (
        t[idx_blend] * theta_merge_2[idx_blend]
        + t[idx_blend] * theta_merge_1[idx_blend]
    )
    g[idx_blend] = (
        v[idx_blend] * theta_merge_2[idx_blend]
        + q[idx_blend] * theta_merge_1[idx_blend]
    )
    b[idx_blend] = (
        p[idx_blend] * theta_merge_2[idx_blend]
        + q[idx_blend] * theta_merge_1[idx_blend]
    )

    idx_blend = np.logical_and(
        i == 3, np.logical_and(theta_merge_1 > 0, theta_merge_2 > 0)
    )
    r[idx_blend] = (
        q[idx_blend] * theta_merge_2[idx_blend]
        + v[idx_blend] * theta_merge_1[idx_blend]
    )
    g[idx_blend] = (
        q[idx_blend] * theta_merge_2[idx_blend]
        + p[idx_blend] * theta_merge_1[idx_blend]
    )
    b[idx_blend] = (
        t[idx_blend] * theta_merge_2[idx_blend]
        + t[idx_blend] * theta_merge_1[idx_blend]
    )

    idx_blend = np.logical_and(
        i == 4, np.logical_and(theta_merge_1 > 0, theta_merge_2 > 0)
    )
    r[idx_blend] = (
        p[idx_blend] * theta_merge_2[idx_blend]
        + q[idx_blend] * theta_merge_1[idx_blend]
    )
    g[idx_blend] = (
        t[idx_blend] * theta_merge_2[idx_blend]
        + t[idx_blend] * theta_merge_1[idx_blend]
    )
    b[idx_blend] = (
        v[idx_blend] * theta_merge_2[idx_blend]
        + q[idx_blend] * theta_merge_1[idx_blend]
    )

    idx_blend = np.logical_and(
        i == 5, np.logical_and(theta_merge_1 > 0, theta_merge_2 > 0)
    )
    r[idx_blend] = (
        t[idx_blend] * theta_merge_2[idx_blend]
        + t[idx_blend] * theta_merge_1[idx_blend]
    )
    g[idx_blend] = (
        q[idx_blend] * theta_merge_2[idx_blend]
        + v[idx_blend] * theta_merge_1[idx_blend]
    )
    b[idx_blend] = (
        q[idx_blend] * theta_merge_2[idx_blend]
        + p[idx_blend] * theta_merge_1[idx_blend]
    )

    idx = s == 0
    r[idx] = v[idx]
    g[idx] = v[idx]
    b[idx] = v[idx]

    rgb = np.stack([r, g, b], axis=-1)

    return rgb.reshape(in_shape)


def save_stack_to_folder(
    img_stack, dir_name, file_name, min_val=None, max_val=None, rgb=False
):
    """

    save image stack into separate images

    Parameters
    ----------
        img_stack   : numpy.ndarray
                      image stack with the shape of (N_frame, Ny, Nx) or (N_frame, Ny, Nx, 3) for RGB images

        dir_name    : str
                      path to the saving folder

        file_name   : str
                      prefix name of the saving images

        min_val     : float
                      minimal value of the plotting range

        max_val     : float
                      minimal value of the plotting range

        rgb         : bool
                      option to save rgb image or not


    """

    os.system("mkdir " + dir_name)

    if rgb:
        N_frame, N, M, _ = img_stack.shape
    else:
        N_frame, N, M = img_stack.shape

    for i in range(N_frame):
        file_name_i = file_name + str(i) + ".tif"
        file_path = os.path.join(dir_name, file_name_i)

        if rgb:
            plt.imsave(file_path, img_stack[i, :, :, :], format="tiff")
        else:
            plt.imsave(
                file_path,
                img_stack[i, :, :],
                format="tiff",
                cmap=plt.cm.gray,
                vmin=min_val,
                vmax=max_val,
            )


def plot3DVectorField(
    img,
    azimuth,
    theta,
    threshold=None,
    anisotropy=1,
    cmapImage="gray",
    clim=[None, None],
    aspect=1,
    spacing=20,
    window=20,
    linelength=20,
    linewidth=3,
    linecolor="g",
    cmapAzimuth="hsv",
    alpha=1,
    subplot_ax=None,
):
    """

    overlays 3D orientation field (azimuth in line orientation, theta in hsv color, retardance in linelength) on the image

    Parameters
    ----------
        img         : numpy.ndarray
                      image to overlay orientation lines on

        azimuth     : numpy.ndarray
                      orientation in radian

        theta       : numpy.ndarray
                      theta in radian

        threshold   : numpy.ndarray
                      a binary numpy array, wherever the map is 0, ignore the plotting of the line

        anisotropy  : numpy.ndarray
                      magnitude encoded in the line length

        cmapImage   : str
                      colormap for displaying the image

        clim        : list
                      [min, max], min and max intensities for displaying img

        aspect      : float
                      aspect ratio of the 2D image

        spacing     : int
                      spacing of the line glyphs

        window      : int
                      size of the blurring window for the orientation field

        linelength  : int
                      can be a scalar or an image the same size as the orientation to encode linelength further

        linewidth   : int
                      width of the orientation line

        linecolor   : str
                      for example 'y' -> yellow

        cmapAzimuth : str
                      colormap for coloring the lines by the theta


        alpha       : int
                      line transparency. [0,1]. lower is more transparent


        subplot_ax  : obj
                      matplotlib image axes from subplots

    Returns
    -------
        im_ax       : obj
                      matplotlib image axes

    """

    U = anisotropy * linelength * np.cos(2 * azimuth)
    V = anisotropy * linelength * np.sin(2 * azimuth)

    USmooth = uniform_filter(U, (window, window))  # plot smoothed vector field
    VSmooth = uniform_filter(V, (window, window))  # plot smoothed vector field
    azimuthSmooth = 0.5 * np.arctan2(VSmooth, USmooth)
    RSmooth = np.sqrt(USmooth**2 + VSmooth**2)
    USmooth, VSmooth = RSmooth * np.cos(azimuthSmooth), RSmooth * np.sin(
        azimuthSmooth
    )

    nY, nX = img.shape
    Y, X = np.mgrid[0:nY, 0:nX]  # notice the reversed order of X and Y

    # Plot sparsely sampled vector lines
    Plotting_X = X[::-spacing, ::spacing]
    Plotting_Y = Y[::-spacing, ::spacing]
    Plotting_U = linelength * USmooth[::-spacing, ::spacing]
    Plotting_V = linelength * VSmooth[::-spacing, ::spacing]

    Plotting_inc = ((theta[::-spacing, ::spacing]) % np.pi) * 180 / np.pi

    if threshold is None:
        threshold = np.ones_like(X)  # no threshold

    Plotting_thres = threshold[::-spacing, ::spacing]

    if subplot_ax is None:
        im_ax = plt.imshow(
            img,
            cmap=cmapImage,
            vmin=clim[0],
            vmax=clim[1],
            origin="lower",
            aspect=aspect,
        )
        plt.title("3D Orientation map")
        plt.quiver(
            Plotting_X[Plotting_thres == 1],
            Plotting_Y[Plotting_thres == 1],
            Plotting_U[Plotting_thres == 1],
            Plotting_V[Plotting_thres == 1],
            Plotting_inc[Plotting_thres == 1],
            cmap=cmapAzimuth,
            norm=Normalize(vmin=0, vmax=180),
            edgecolor=linecolor,
            facecolor=linecolor,
            units="xy",
            alpha=alpha,
            width=linewidth,
            headwidth=0,
            headlength=0,
            headaxislength=0,
            scale_units="xy",
            scale=1,
            angles="uv",
            pivot="mid",
        )
    else:
        im_ax = subplot_ax.imshow(
            img,
            cmap=cmapImage,
            vmin=clim[0],
            vmax=clim[1],
            origin="lower",
            aspect=aspect,
        )
        subplot_ax.set_title("3D Orientation map")
        subplot_ax.quiver(
            Plotting_X[Plotting_thres == 1],
            Plotting_Y[Plotting_thres == 1],
            Plotting_U[Plotting_thres == 1],
            Plotting_V[Plotting_thres == 1],
            Plotting_inc[Plotting_thres == 1],
            cmap=cmapAzimuth,
            norm=Normalize(vmin=0, vmax=180),
            edgecolor=linecolor,
            facecolor=linecolor,
            units="xy",
            alpha=alpha,
            width=linewidth,
            headwidth=0,
            headlength=0,
            headaxislength=0,
            scale_units="xy",
            scale=1,
            angles="uv",
            pivot="mid",
        )

    return im_ax


def orientation_3D_hist(
    azimuth,
    theta,
    retardance,
    bins=20,
    num_col=1,
    size=10,
    contour_level=100,
    hist_cmap="gray",
    top_hemi=False,
    colorbar=True,
):
    """

    plot histogram of 3D orientation weighted by anisotropy

    Parameters
    ----------
        azimuth       : numpy.ndarray
                        a flatten array or a stack of flatten array of azimuth with the shape of (N,) or (N_stack, N)

        theta         : numpy.ndarray
                        a flatten array or a stack of flatten array of theta with the shape of (N,) or (N_stack, N)

        retardance    : numpy.ndarray
                        a flatten array or a stack of flatten array of retardance with the shape of (N,) or (N_stack, N)

        bins          : int
                        number of bins for both azimuth and theta dimension

        num_col       : int
                        number of columns for display of multiple histograms

        size          : int
                        the size of each figure panel

        contour_level : int
                        number of discrete contour levels of the signal counts

        hist_cmap     : str
                        colormap for the plotted histogram

        top_hemi      : bool
                        option to convert the azimuth and theta from the front hemisphere to the top hemisphere

        colorbar      : bool
                        option to show the colorbar

    Returns
    -------
        fig           : obj
                        matplotlib figure object

        ax            : obj
                        matplotlib axes object

    """

    if top_hemi:
        index_remapping = theta > np.pi / 2
        azimuth[index_remapping] = azimuth[index_remapping] + np.pi
        theta[index_remapping] = np.pi - theta[index_remapping]

    if azimuth.ndim == 1:
        N_hist = 1
        azimuth = azimuth[np.newaxis, :]
        theta = theta[np.newaxis, :]
        retardance = retardance[np.newaxis, :]

    elif azimuth.ndim == 2:
        N_hist, _ = azimuth.shape

    num_row = int(np.ceil(N_hist / num_col))
    figsize = (num_col * size, num_row * size)

    if top_hemi:
        azimuth_edges = np.linspace(0, 2 * np.pi, 2 * bins)
        theta_edges = np.linspace(0, np.pi / 2, bins // 2)
    else:
        azimuth_edges = np.linspace(0, np.pi, bins)
        theta_edges = np.linspace(0, np.pi, bins)

    theta_hist, azimuth_hist = np.meshgrid(theta_edges, azimuth_edges)

    fig, ax = plt.subplots(
        num_row, num_col, subplot_kw=dict(projection="polar"), figsize=figsize
    )

    for i in range(N_hist):
        az = azimuth[i].copy()
        th = theta[i].copy()
        val = retardance[i].copy()

        if top_hemi:
            statistic, aedges, tedges, binnumber = binned_statistic_2d(
                az,
                th,
                val,
                statistic="sum",
                bins=[2 * bins, bins / 2],
                range=[[0, 2 * np.pi], [0, np.pi / 2]],
            )
        else:
            statistic, aedges, tedges, binnumber = binned_statistic_2d(
                az,
                th,
                val,
                statistic="sum",
                bins=bins,
                range=[[0, np.pi], [0, np.pi]],
            )

        row_idx = i // num_col
        col_idx = np.mod(i, num_col)

        if num_row == 1:
            if num_col == 1:
                img = ax.contourf(
                    azimuth_hist,
                    theta_hist / np.pi * 180,
                    statistic,
                    levels=contour_level,
                    cmap=hist_cmap,
                )
                if top_hemi:
                    ax.set_yticks([0, 30, 60])
                    ax.set_xticks(
                        np.pi / 180 * np.linspace(0, 360, 12, endpoint=False)
                    )
                    ax.set_rmax(90)
                else:
                    ax.set_yticks([0, 30, 60, 90, 120, 150, 180])
                    ax.set_thetamax(180)
                if colorbar:
                    fig.colorbar(img, ax=ax)
            else:
                img = ax[col_idx].contourf(
                    azimuth_hist,
                    theta_hist / np.pi * 180,
                    statistic,
                    levels=contour_level,
                    cmap=hist_cmap,
                )
                if top_hemi:
                    ax[col_idx].set_xticks(
                        np.pi / 180 * np.linspace(0, 360, 12, endpoint=False)
                    )
                    ax[col_idx].set_yticks([0, 30, 60])
                    ax[col_idx].set_rmax(90)
                else:
                    ax[col_idx].set_yticks([0, 30, 60, 90, 120, 150, 180])
                    ax[col_idx].set_thetamax(180)

                if colorbar:
                    fig.colorbar(img, ax=ax[col_idx])

        else:
            if num_col == 1:
                img = ax[row_idx].contourf(
                    azimuth_hist,
                    theta_hist / np.pi * 180,
                    statistic,
                    levels=contour_level,
                    cmap=hist_cmap,
                )

                if top_hemi:
                    ax[row_idx].set_xticks(
                        np.pi / 180 * np.linspace(0, 360, 12, endpoint=False)
                    )
                    ax[row_idx].set_yticks([0, 30, 60])
                    ax[row_idx].set_rmax(90)
                else:
                    ax[row_idx].set_yticks([0, 30, 60, 90, 120, 150, 180])
                    ax[row_idx].set_thetamax(180)
                if colorbar:
                    fig.colorbar(img, ax=ax[row_idx])
            else:
                img = ax[row_idx, col_idx].contourf(
                    azimuth_hist,
                    theta_hist / np.pi * 180,
                    statistic,
                    levels=contour_level,
                    cmap=hist_cmap,
                )
                if top_hemi:
                    ax[row_idx, col_idx].set_xticks(
                        np.pi / 180 * np.linspace(0, 360, 12, endpoint=False)
                    )
                    ax[row_idx, col_idx].set_yticks([0, 30, 60])
                    ax[row_idx, col_idx].set_rmax(90)
                else:
                    ax[row_idx, col_idx].set_yticks(
                        [0, 30, 60, 90, 120, 150, 180]
                    )
                    ax[row_idx, col_idx].set_thetamax(180)
                if colorbar:
                    fig.colorbar(img, ax=ax[row_idx, col_idx])

    return fig, ax