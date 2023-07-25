import numpy as np
import matplotlib.pyplot as plt
import pywt
import time
import torch

from numpy.fft import fft, ifft, fft2, ifft2, fftn, ifftn, fftshift, ifftshift
from scipy.ndimage import uniform_filter
from collections import namedtuple
from .optics import scattering_potential_tensor_to_3D_orientation_PN

import re

numbers = re.compile(r"(\d+)")

import torch


def pad_zyx_along_z(zyx_data, z_padding):
    """
    Pad a 3D tensor along the z-dimension.

    Parameters
    ----------
    zyx_data : torch.Tensor
        Input 3D tensor of shape (Z, Y, X).
    z_padding : int
        Number of padding slices to add on both ends of the z-dimension.

    Returns
    -------
    torch.Tensor
        Padded 3D tensor of shape (Z + 2 * z_padding, Y, X).

    Raises
    ------
    ValueError
        If z_padding is negative.

    Notes
    -----
    - If z_padding is 0, the function returns the input tensor unchanged.
    - If z_padding is positive, the function pads the tensor with zeros along the z-dimension.
    - If z_padding is greater than or equal to the number of z-slices in zyx_data, a warning message is included in the returned tensor,
      indicating that zero padding is used instead of reflection padding (less effective).
    - Reflection padding is used when z_padding is smaller than the number of z-slices in zyx_data, providing a symmetric padding.
    """
    if z_padding < 0:
        raise ValueError("z_padding cannot be negative.")
    elif z_padding == 0:
        return zyx_data
    else:
        zyx_padded = torch.nn.functional.pad(
            zyx_data,
            (0, 0, 0, 0, z_padding, z_padding),
            mode="constant",
            value=0,
        )
        if z_padding < zyx_data.shape[0]:
            zyx_padded[:z_padding] = torch.flip(zyx_data[:z_padding], dims=[0])
            zyx_padded[-z_padding:] = torch.flip(
                zyx_data[-z_padding:], dims=[0]
            )
        else:
            warning_msg = "Warning: z_padding is larger than the number of z-slices. Using zero padding instead of reflection padding (less effective)."
            zyx_padded = torch.nn.functional.pad(
                zyx_padded,
                (0, 0, 0, 0, 0, 0),
                mode="constant",
                value=0,
            )
        return zyx_padded


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def generate_star_target(yx_shape, blur_px=2, margin=60):
    """

    generate Siemens star image for simulation

    Parameters
    ----------
        yx_shape  : tuple
                  size of the simulated image in (Y, X)

        blur_px : float
                  the standard deviation of the imposed Gaussian blur on the simulated image

        margin  : int
                  the size of blank margin on the simulated image

    Returns
    -------
        star    : torch.tensor
                  Siemens star with the size of (Y, X)

        theta   : torch.tensor
                  azimuthal angle of the polar coordinate with the size of (Y, X)

        xx      : numpy.ndarray
                  x coordinate array with the size of (Y, X)

    """

    # Construct Siemens star
    Y, X = yx_shape

    x = np.arange(X) - X // 2
    y = np.arange(Y) - Y // 2

    xx, yy = torch.tensor(np.meshgrid(x, y))

    rho = torch.sqrt(xx**2 + yy**2)
    theta = torch.arctan2(yy, xx)

    # star = (1 + np.cos(40*theta))
    # star = np.pad(star[10:-10,10:-10],(10,),mode='constant')
    star = 1 + torch.cos(16 * theta)
    star = torch.nn.functional.pad(
        star[margin:-margin, margin:-margin], 4 * (margin,), mode="constant"
    )
    star[star < 1] = 0

    # Filter to prevent aliasing

    Gaussian = torch.exp(-(rho**2) / (2 * blur_px**2))

    star = torch.clip(
        torch.real(
            torch.fft.ifft2(
                torch.fft.fft2(star)
                * torch.fft.fft2(torch.fft.ifftshift(Gaussian))
            )
        ),
        min=0,
    )
    # star = np.maximum(0, np.real(ifft2(fft2(star) * fft2(ifftshift(Gaussian)))))*(2+np.sin(2*np.pi*(1/5)*rho))
    star /= torch.max(star)

    return star, theta, xx


def genStarTarget_3D(
    img_dim,
    ps,
    psz,
    blur_size=0.1,
    inc_upper_bound=np.pi / 8,
    inc_range=np.pi / 64,
):
    """

    generate 3D star image for simulation

    Parameters
    ----------
        img_dim         : tuple
                          shape of the computed 3D space with size of (Ny, Nx, Nz)

        ps              : float
                          transverse pixel size of the image space

        psz             : float
                          axial step size of the image space

        blur_size       : float
                          the standard deviation of the imposed 3D Gaussian blur on the simulated image

        inc_upper_bound : float
                          the upper bound of the inclination angle of the tilted feature from 0 to pi/2

        inc_range       : float
                          the range of the inclination that defines the axial width of the tilted feature

    Returns
    -------
        star            : numpy.ndarray
                          3D star image with the size of (Ny, Nx, Nz)

        azimuth         : numpy.ndarray
                          azimuthal angle of the 3D polar coordinate with the size of (Ny, Nx, Nz)

        inc_angle       : numpy.ndarray
                          theta angle of the 3D polar coordinate with the size of (Ny, Nx, Nz)

    """

    N, M, L = img_dim

    x = (np.r_[:M] - M // 2) * ps
    y = (np.r_[:N] - N // 2) * ps
    z = (np.r_[:L] - L // 2) * psz

    xx, yy, zz = np.meshgrid(x, y, z)

    rho = np.sqrt(xx**2 + yy**2 + zz**2)
    azimuth = np.arctan2(yy, xx)
    inc_angle = np.arctan2((xx**2 + yy**2) ** (1 / 2), zz)

    star = 1 + np.cos(16 * azimuth)

    star = np.pad(
        star[20:-20, 20:-20, 20:-20], ((20,), (20,), (20,)), mode="constant"
    )
    star[star < 1] = 0
    star[np.abs(inc_angle - np.pi / 2) > inc_upper_bound] = 0
    star[np.abs(inc_angle - np.pi / 2) < inc_upper_bound - inc_range] = 0

    # Filter to prevent aliasing

    Gaussian = np.exp(-(rho**2) / (2 * blur_size**2))

    star = np.maximum(
        0, np.real(ifftn(fftn(star) * fftn(ifftshift(Gaussian))))
    )
    star /= np.max(star)

    return star, azimuth, inc_angle


def generate_sphere_target(
    zyx_shape, yx_pixel_size, z_pixel_size, radius, blur_size=0.1
):
    """

    generate 3D sphere target for simulation

    Parameters
    ----------
        zyx_shape   : tuple
                    shape of the computed 3D space with size of (Z, Y, X)

        yx_pixel_size        : float
                    transverse pixel size of the image space

        z_pixel_size       : float
                    axial step size of the image space

        radius    : float
                    radius of the generated sphere

        blur_size : float
                    the standard deviation of the imposed 3D Gaussian blur on the simulated image


    Returns
    -------
        sphere    : torch.tensor
                    3D star image with the size of (Z, Y, X)

        azimuth   : torch.tensor
                    azimuthal angle of the 3D polar coordinate with the size of (Z, Y, X)

        inc_angle : torch.tensor
                    theta angle of the 3D polar coordinate with the size of (Z, Y, X)

    """

    Z, Y, X = zyx_shape
    x = (torch.arange(X) - X // 2) * yx_pixel_size
    y = (torch.arange(Y) - Y // 2) * yx_pixel_size
    z = (torch.arange(Z) - Z // 2) * z_pixel_size

    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")

    rho = torch.sqrt(xx**2 + yy**2 + zz**2)
    azimuth = torch.arctan2(yy, xx)
    inc_angle = torch.arctan2((xx**2 + yy**2) ** (1 / 2), zz)

    sphere = torch.zeros_like(xx)
    sphere[xx**2 + yy**2 + zz**2 < radius**2] = 1

    Gaussian = np.exp(-(rho**2) / (2 * blur_size**2))

    sphere = np.maximum(
        0,
        torch.real(
            torch.fft.ifftn(
                torch.fft.fftn(sphere)
                * torch.fft.fftn(torch.fft.ifftshift(Gaussian))
            )
        ),
    )
    sphere /= torch.max(sphere)

    return sphere, azimuth, inc_angle


def gen_coordinate(img_dim, ps):
    """

    generate spatial and spatial frequency coordinate arrays

    Input:
        img_dim : tuple
                  shape of the computed 2D space with size of (Ny, Nx)

        ps      : float
                  transverse pixel size of the image space

    Output:
        xx      : numpy.ndarray
                  x coordinate array with the size of (Ny, Nx)

        yy      : numpy.ndarray
                  y coordinate array with the size of (Ny, Nx)

        fxx     : numpy.ndarray
                  x component of 2D spatial frequency array with the size of (Ny, Nx)

        fyy     : numpy.ndarray
                  y component of 2D spatial frequency array with the size of (Ny, Nx)

    """

    N, M = img_dim

    fx = ifftshift((np.r_[:M] - M / 2) / M / ps)
    fy = ifftshift((np.r_[:N] - N / 2) / N / ps)
    x = ifftshift((np.r_[:M] - M / 2) * ps)
    y = ifftshift((np.r_[:N] - N / 2) * ps)

    xx, yy = np.meshgrid(x, y)
    fxx, fyy = np.meshgrid(fx, fy)

    return (xx, yy, fxx, fyy)


def generate_radial_frequencies(img_dim, ps):
    fy = torch.fft.fftfreq(img_dim[0], ps)
    fx = torch.fft.fftfreq(img_dim[1], ps)

    fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")

    return torch.sqrt(fyy**2 + fxx**2)


def axial_upsampling(I_meas, upsamp_factor=1):
    F = lambda x: ifftshift(fft(fftshift(x, axes=2), axis=2), axes=2)
    iF = lambda x: ifftshift(ifft(fftshift(x, axes=2), axis=2), axes=2)

    N, M, Lcrop = I_meas.shape
    L = Lcrop * upsamp_factor

    I_meas_up = np.zeros((N, M, L))
    if (L - Lcrop) // 2 == 0:
        I_meas_up = np.abs(
            iF(
                np.pad(
                    F(I_meas),
                    ((0,), (0,), ((L - Lcrop) // 2,)),
                    mode="constant",
                )
            )
        )
    else:
        I_meas_up = np.abs(
            iF(
                np.pad(
                    F(I_meas),
                    ((0, 0), (0, 0), ((L - Lcrop) // 2 + 1, (L - Lcrop) // 2)),
                    mode="constant",
                )
            )
        )

    return I_meas_up


def softTreshold(x, threshold, use_gpu=False, gpu_id=0):
    """

    compute soft thresholding operation on numpy ndarray with gpu option

    Parameters
    ----------
        x          : numpy.ndarray
                     targeted array for soft thresholding operation with arbitrary size

        threshold  : numpy.ndarray
                     array contains threshold value for each x array position

        use_gpu    : bool
                     option to use gpu or not

        gpu_id     : int
                     number refering to which gpu will be used

    Returns
    -------
        x_threshold : numpy.ndarray
                      thresholded array

    """

    if use_gpu:
        globals()["cp"] = __import__("cupy")
        cp.cuda.Device(gpu_id).use()
        magnitude = cp.abs(x)
        ratio = cp.maximum(0, magnitude - threshold) / (magnitude + 1e-16)
    else:
        magnitude = np.abs(x)
        ratio = np.maximum(0, magnitude - threshold) / (magnitude + 1e-16)

    x_threshold = x * ratio

    return x_threshold


def wavelet_softThreshold(img, wavelet, threshold, level=1, axes=None):
    """

    soft thresholding in the nD wavelet space

    Parameters
    ----------
        img       : numpy.ndarray
                    image or volume in nD space

        wavelet   : str
                    type of wavelet to use (pywt.wavelist() to find the whole list)

        threshold : float
                    threshold value

        axes      : list
                    axes along which to denoise nD volume

    Returns
    -------
        img_thres : numpy.ndarray
                    denoised image or volume in nD space

    """
    shape = np.shape(img)

    padding = []
    unpadding = []
    for dim in shape:
        # No padding
        if dim % 2 == 0:
            padding.append((0, 0))
            unpadding.append(slice(None))

        # pad dimension
        else:
            padding.append((0, 1))
            unpadding.append(slice(0, -1))

    padding = tuple(padding)
    unpadding = tuple(unpadding)

    img_padded = np.pad(img, padding, "edge")

    coeffs = pywt.wavedecn(img_padded, wavelet, level=level, axes=axes)

    for i in range(level + 1):
        if i == 0:
            coeffs[i] = softTreshold(coeffs[i], threshold)
        else:
            for item in coeffs[i]:
                coeffs[i][item] = softTreshold(coeffs[i][item], threshold)

    img_thres = pywt.waverecn(coeffs, wavelet, axes=axes)

    return img_thres[unpadding]


def array_based_4x4_det(a):
    """

    compute array-based determinant on 4 x 4 matrix

    Parameters
    ----------
        a   : numpy.ndarray or cupy.ndarray
              4 x 4 matrix in the nD space with the shape of (4, 4, Ny, Nx, Nz, ...)

    Returns
    -------
        det : numpy.ndarray or cupy.ndarray
              computed determinant in the nD space with the shape of (Ny, Nx, Nz, ...)

    """

    sub_det1 = a[0, 0] * (
        a[1, 1] * (a[2, 2] * a[3, 3] - a[3, 2] * a[2, 3])
        - a[1, 2] * (a[2, 1] * a[3, 3] - a[3, 1] * a[2, 3])
        + a[1, 3] * (a[2, 1] * a[3, 2] - a[3, 1] * a[2, 2])
    )

    sub_det2 = a[0, 1] * (
        a[1, 0] * (a[2, 2] * a[3, 3] - a[3, 2] * a[2, 3])
        - a[1, 2] * (a[2, 0] * a[3, 3] - a[3, 0] * a[2, 3])
        + a[1, 3] * (a[2, 0] * a[3, 2] - a[3, 0] * a[2, 2])
    )

    sub_det3 = a[0, 2] * (
        a[1, 0] * (a[2, 1] * a[3, 3] - a[3, 1] * a[2, 3])
        - a[1, 1] * (a[2, 0] * a[3, 3] - a[3, 0] * a[2, 3])
        + a[1, 3] * (a[2, 0] * a[3, 1] - a[3, 0] * a[2, 1])
    )

    sub_det4 = a[0, 3] * (
        a[1, 0] * (a[2, 1] * a[3, 2] - a[3, 1] * a[2, 2])
        - a[1, 1] * (a[2, 0] * a[3, 2] - a[3, 0] * a[2, 2])
        + a[1, 2] * (a[2, 0] * a[3, 1] - a[3, 0] * a[2, 1])
    )

    det = sub_det1 - sub_det2 + sub_det3 - sub_det4

    return det


def array_based_5x5_det(a):
    """

    compute array-based determinant on 5 x 5 matrix

    Parameters
    ----------
        a   : numpy.ndarray or cupy.ndarray
              5 x 5 matrix in the nD space with the shape of (5, 5, Ny, Nx, Nz, ...)

    Returns
    -------
        det : numpy.ndarray or cupy.ndarray
              computed determinant in the nD space with the shape of (Ny, Nx, Nz, ...)

    """

    det = (
        a[0, 0] * array_based_4x4_det(a[1:, 1:])
        - a[0, 1] * array_based_4x4_det(a[1:, [0, 2, 3, 4]])
        + a[0, 2] * array_based_4x4_det(a[1:, [0, 1, 3, 4]])
        - a[0, 3] * array_based_4x4_det(a[1:, [0, 1, 2, 4]])
        + a[0, 4] * array_based_4x4_det(a[1:, [0, 1, 2, 3]])
    )

    return det


def array_based_6x6_det(a):
    """

    compute array-based determinant on 6 x 6 matrix

    Parameters
    ----------
        a   : numpy.ndarray or cupy.ndarray
              6 x 6 matrix in the nD space with the shape of (6, 6, Ny, Nx, Nz, ...)

    Returns
    -------
        det : numpy.ndarray or cupy.ndarray
              computed determinant in the nD space with the shape of (Ny, Nx, Nz, ...)

    """

    det = (
        a[0, 0] * array_based_5x5_det(a[1:, 1:])
        - a[0, 1] * array_based_5x5_det(a[1:, [0, 2, 3, 4, 5]])
        + a[0, 2] * array_based_5x5_det(a[1:, [0, 1, 3, 4, 5]])
        - a[0, 3] * array_based_5x5_det(a[1:, [0, 1, 2, 4, 5]])
        + a[0, 4] * array_based_5x5_det(a[1:, [0, 1, 2, 3, 5]])
        - a[0, 5] * array_based_5x5_det(a[1:, [0, 1, 2, 3, 4]])
    )

    return det


def array_based_7x7_det(a):
    """

    compute array-based determinant on 7 x 7 matrix

    Parameters
    ----------
        a   : numpy.ndarray or cupy.ndarray
              7 x 7 matrix in the nD space with the shape of (7, 7, Ny, Nx, Nz, ...)

    Returns
    -------
        det : numpy.ndarray or cupy.ndarray
              computed determinant in the nD space with the shape of (Ny, Nx, Nz, ...)

    """

    det = (
        a[0, 0] * array_based_6x6_det(a[1:, 1:])
        - a[0, 1] * array_based_6x6_det(a[1:, [0, 2, 3, 4, 5, 6]])
        + a[0, 2] * array_based_6x6_det(a[1:, [0, 1, 3, 4, 5, 6]])
        - a[0, 3] * array_based_6x6_det(a[1:, [0, 1, 2, 4, 5, 6]])
        + a[0, 4] * array_based_6x6_det(a[1:, [0, 1, 2, 3, 5, 6]])
        - a[0, 5] * array_based_6x6_det(a[1:, [0, 1, 2, 3, 4, 6]])
        + a[0, 6] * array_based_6x6_det(a[1:, [0, 1, 2, 3, 4, 5]])
    )

    return det


def uniform_filter_2D(image, size, use_gpu=False, gpu_id=0):
    """

    compute uniform filter operation on 2D image with gpu option

    Parameters
    ----------
        image          : numpy.ndarray
                         targeted image for filtering with size of (Ny, Nx)

        size           : int
                         size of the kernel for uniform filtering

        use_gpu        : bool
                         option to use gpu or not

        gpu_id         : int
                         number refering to which gpu will be used

    Returns
    -------
        image_filtered : numpy.ndarray
                         filtered image with size of (Ny, Nx)

    """

    N, M = image.shape

    if use_gpu:
        globals()["cp"] = __import__("cupy")
        cp.cuda.Device(gpu_id).use()

        # filter in y direction

        image_cp = cp.array(image)

        kernel_y = cp.zeros((3 * N,))
        kernel_y[3 * N // 2 - size // 2 : 3 * N // 2 + size // 2] = 1
        kernel_y /= cp.sum(kernel_y)
        kernel_y = cp.fft.fft(cp.fft.ifftshift(kernel_y))

        image_bound_y = cp.zeros((3 * N, M))
        image_bound_y[N : 2 * N, :] = image_cp.copy()
        image_bound_y[0:N, :] = cp.flipud(image_cp)
        image_bound_y[2 * N : 3 * N, :] = cp.flipud(image_cp)
        filtered_y = cp.real(
            cp.fft.ifft(
                cp.fft.fft(image_bound_y, axis=0) * kernel_y[:, cp.newaxis],
                axis=0,
            )
        )
        filtered_y = filtered_y[N : 2 * N, :]

        # filter in x direction

        kernel_x = cp.zeros((3 * M,))
        kernel_x[3 * M // 2 - size // 2 : 3 * M // 2 + size // 2] = 1
        kernel_x /= cp.sum(kernel_x)
        kernel_x = cp.fft.fft(cp.fft.ifftshift(kernel_x))

        image_bound_x = cp.zeros((N, 3 * M))
        image_bound_x[:, M : 2 * M] = filtered_y.copy()
        image_bound_x[:, 0:M] = cp.fliplr(filtered_y)
        image_bound_x[:, 2 * M : 3 * M] = cp.fliplr(filtered_y)

        image_filtered = cp.real(
            cp.fft.ifft(
                cp.fft.fft(image_bound_x, axis=1) * kernel_x[cp.newaxis, :],
                axis=1,
            )
        )
        image_filtered = image_filtered[:, M : 2 * M]
    else:
        image_filtered = uniform_filter(image, size=size)

    return image_filtered


def inten_normalization(img_stack, bg_filter=True):
    """

    layer-by-layer intensity normalization to reduce low-frequency phase artifacts

    Parameters
    ----------
        img_stack      : torch.tensor
                         image stack for normalization with size of (Z, Y, X)

        bg_filter      : bool
                         option for slow-varying 2D background normalization with uniform filter

    Returns
    -------
        img_norm_stack : torch.tensor
                         normalized image stack with size of (Z, Y, X)

    """

    Z, Y, X = img_stack.shape

    img_norm_stack = torch.zeros_like(img_stack)

    for i in range(Z):
        if bg_filter:
            img_norm_stack[i] = img_stack[i] / uniform_filter(
                img_stack[i], size=X // 2
            )
        else:
            img_norm_stack[i] = img_stack[i].copy()
        img_norm_stack[i] /= torch.mean(img_norm_stack[i])
        img_norm_stack[i] -= 1

    return img_norm_stack


def inten_normalization_3D(img_stack):
    """

    whole-stack intensity normalization to reduce low-frequency phase artifacts

    Parameters
    ----------
        img_stack      : torch.tensor
                         image stack for normalization with size of (Z, Y, X)

    Returns
    -------
        img_norm_stack : torch.tensor
                         normalized image stack with size of (Z, Y, X)

    """
    img_norm_stack = img_stack / torch.mean(img_stack)
    img_norm_stack -= 1
    return img_norm_stack


def dual_variable_tikhonov_deconvolution_2d(AHA, b_vec, determinant=None):
    """

    2D Tikhonov deconvolution to solve for phase and absorption with weak object transfer function

    Parameters
    ----------
        AHA         : list
                      A^H times A matrix stored with a list of 4 2D numpy array (4 diagonal matrices)
                      | AHA[0]  AHA[1] |
                      | AHA[2]  AHA[3] |

        b_vec       : list
                      measured intensity stored with a list of 2 2D numpy array (2 vectors)
                      | b_vec[0] |
                      | b_vec[1] |

        determinant : optional torch.tensor

    Returns
    -------
        mu_sample   : torch.tensor
                      2D absorption reconstruction with the size of (Y, X)

        phi_sample  : torch.tensor
                      2D phase reconstruction with the size of (Y, X)
    """

    if determinant is None:
        determinant = AHA[0] * AHA[3] - AHA[1] * AHA[2]

    mu_sample_f = (b_vec[0] * AHA[3] - b_vec[1] * AHA[1]) / determinant
    phi_sample_f = (b_vec[1] * AHA[0] - b_vec[0] * AHA[2]) / determinant

    mu_sample = torch.real(torch.fft.ifft2(mu_sample_f))
    phi_sample = torch.real(torch.fft.ifft2(phi_sample_f))

    return mu_sample, phi_sample


def dual_variable_admm_tv_deconv_2d(
    AHA,
    b_vec,
    rho=1e-5,
    lambda_u=1e-3,
    lambda_p=1e-3,
    itr=20,
    verbose=False,
):
    """

    2D TV deconvolution to solve for phase and absorption with weak object transfer function

    ADMM formulation:

        0.5 * || A*x - b ||_2^2 + lambda * || z ||_1 + 0.5 * rho * || D*x - z + u ||_2^2

    Parameters
    ----------
        AHA        : list
                     A^H times A matrix stored with a list of 4 2D numpy array (4 diagonal matrices)
                     | AHA[0]  AHA[1] |
                     | AHA[2]  AHA[3] |

        b_vec      : list
                     measured intensity stored with a list of 2 2D numpy array (2 vectors)
                     | b_vec[0] |
                     | b_vec[1] |

        rho        : float
                     ADMM rho parameter

        lambda_u   : float
                     TV regularization parameter for absorption

        lambda_p   : float
                     TV regularization parameter for phase

        itr        : int
                     number of iterations of ADMM algorithm

        verbose    : bool
                     option to display progress of the computation

        use_gpu    : bool
                     option to use gpu or not

        gpu_id     : int
                     number refering to which gpu will be used

    Returns
    -------
        mu_sample  : numpy.ndarray
                     2D absorption reconstruction with the size of (Ny, Nx)

        phi_sample : numpy.ndarray
                     2D phase reconstruction with the size of (Ny, Nx)
    """

    # ADMM deconvolution with anisotropic TV regularization

    N, M = b_vec[0].shape
    Dx = np.zeros((N, M))
    Dx[0, 0] = 1
    Dx[0, -1] = -1
    Dy = np.zeros((N, M))
    Dy[0, 0] = 1
    Dy[-1, 0] = -1

    Dx = fft2(Dx)
    Dy = fft2(Dy)

    rho_term = rho * (np.conj(Dx) * Dx + np.conj(Dy) * Dy)

    z_para = np.zeros((4, N, M))
    u_para = np.zeros((4, N, M))
    D_vec = np.zeros((4, N, M))

    AHA[0] = AHA[0] + rho_term
    AHA[3] = AHA[3] + rho_term

    determinant = AHA[0] * AHA[3] - AHA[1] * AHA[2]

    for i in range(itr):
        v_para = fft2(z_para - u_para)
        b_vec_new = [
            b_vec[0]
            + rho * (np.conj(Dx) * v_para[0] + np.conj(Dy) * v_para[1]),
            b_vec[1]
            + rho * (np.conj(Dx) * v_para[2] + np.conj(Dy) * v_para[3]),
        ]

        mu_sample, phi_sample = dual_variable_tikhonov_deconvolution_2d(
            AHA, b_vec_new, determinant=determinant
        )

        D_vec[0] = mu_sample - np.roll(mu_sample, -1, axis=1)
        D_vec[1] = mu_sample - np.roll(mu_sample, -1, axis=0)
        D_vec[2] = phi_sample - np.roll(phi_sample, -1, axis=1)
        D_vec[3] = phi_sample - np.roll(phi_sample, -1, axis=0)

        z_para = D_vec + u_para

        z_para[:2, :, :] = softTreshold(z_para[:2, :, :], lambda_u / rho)
        z_para[2:, :, :] = softTreshold(z_para[2:, :, :], lambda_p / rho)

        u_para += D_vec - z_para

        if verbose:
            print("Number of iteration computed (%d / %d)" % (i + 1, itr))

    return mu_sample, phi_sample


def single_variable_tikhonov_deconvolution_3D(
    S0_stack,
    H_eff,
    reg_re=1e-4,
    autotune=False,
    epsilon_auto=0.5,
    output_lambda=False,
    search_range_auto=6,
    verbose=True,
):
    """

    Single variable 3D Tikhonov deconvolution to solve for 3D phase (from defocus, with weak object transfer function) or 3D fluorescence.

    Parameters
    ----------
        S0_stack         : numpy.ndarray
                           S0 z-stack for 3D phase deconvolution with size of (Ny, Nx, Nz)

        H_eff            : numpy.ndarray
                           effective transfer function with size of (Ny, Nx, Nz)

        reg_re           : float
                           Tikhonov regularization parameter

        autotune         :  bool
                           option to use L-curve to automatically choose regularization parameter

        output_lambda   : bool
                          option to return the optimal L-curve value after assessment

        epsilon_auto     : float
                           (if using autotune) the tolerance on the regularization parameter for the stopping condition of iterating along the L curve (in log10 units)

        search_range_auto: float
                           (if using autotune) L-curve search will occur on the range of reg_re +/- search_range_auto

        verbose          : bool
                           option to display detailed progress of computations or not

    Returns
    -------
        f_real           : numpy.ndarray
                           3D unscaled phase reconstruction with the size of (Ny, Nx, Nz)
                           (if using autotune) reconstruction for the automatically chosen parameter, plus two others around that parameter value, size (3, Ny, Nx, Nz)
    """
    S0_stack_f = torch.fft.fftn(S0_stack, dim=(-3, -2, -1))
    Z, Y, X = S0_stack_f.shape
    H_eff_abs_square = torch.abs(H_eff) ** 2
    H_eff_conj = torch.conj(H_eff)

    # a "named tuple" representing a point on the L curve
    # allows for dot notation to access attributes
    Point_L_curve = namedtuple(
        "Point_L_curve", "reg_x reg data_norm reg_norm f_real_f"
    )

    def menger_curvature(points):
        # x is data norm, y is reg norm
        x, y = torch.zeros(3), torch.zeros(3)
        for i in range(3):
            x[i] = points[i].data_norm
            y[i] = points[i].reg_norm
        d0 = ((x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2) ** (1 / 2)
        d1 = ((x[2] - x[1]) ** 2 + (y[2] - y[1]) ** 2) ** (1 / 2)
        d2 = ((x[2] - x[0]) ** 2 + (y[2] - y[0]) ** 2) ** (1 / 2)

        signed_area = 0.5 * (
            (x[1] - x[2]) * (y[1] - y[0]) - (x[1] - x[0]) * (y[1] - y[2])
        )

        return 4 * signed_area / (d0 * d1 * d2)

    # computes f_real_f for a specific lambda
    # used for both autotuning and non-autotuning situation
    def compute_f_real_f(reg_x):
        reg_coeff = 10**reg_x

        # FT{f} (f=scattering potential (whose real part is (scaled) phase))
        f_real_f = S0_stack_f * H_eff_conj / (H_eff_abs_square + reg_coeff)

        return f_real_f

    # evaluate the L curve at a specific lambda
    # returns a new Point_L_curve() tuple.
    # this function is the only place where new points are instantiated
    def eval_L_curve(reg_x, keep_f_real_f=False):
        f_real_f = compute_f_real_f(reg_x)
        S0_est_stack_f = (
            H_eff * f_real_f
        )  # Ax (put estimate through forward model)

        data_norm_eval = torch.log(
            torch.linalg.norm(S0_est_stack_f - S0_stack_f) ** 2 / (Z * Y * X)
        )
        reg_norm_eval = torch.log(
            torch.linalg.norm(f_real_f) ** 2 / (Z * Y * X)
        )

        if not keep_f_real_f:
            f_real_f = None

        return Point_L_curve(
            reg_x, 10**reg_x, data_norm_eval, reg_norm_eval, f_real_f
        )

    # creates return value of the whole function
    # (scaled) phase = real part of inverse FT {scattering potential}
    def ifft_f_real(f_real_f):
        f_real = torch.real(torch.fft.ifftn(f_real_f, dim=(-3, -2, -1)))
        return f_real

    def calc_golden_x(a, b):
        gs_ratio = (1 + torch.sqrt(5)) / 2
        return (a * gs_ratio + b) / (1 + gs_ratio)

    if autotune:
        # initialize golden section search
        reg_x_cent = torch.log10(
            reg_re
        )  # reg_re becomes middle of search range
        reg_x = torch.zeros(4)
        reg_x[0] = (
            reg_x_cent - search_range_auto
        )  # search range = reg_x_cent +/- search_range_auto
        reg_x[3] = reg_x_cent + search_range_auto
        reg_x[1] = calc_golden_x(reg_x[0], reg_x[3])
        reg_x[2] = reg_x[0] + (reg_x[3] - reg_x[1])

        # holds the 4 current points
        curr_pts = []
        for i in range(4):
            curr_pts.append(eval_L_curve(reg_x[i]))

        last_opt = None  # only save the last point to save GPU memory
        itr = 0
        search_range = curr_pts[3].reg_x - curr_pts[0].reg_x
        while search_range > epsilon_auto:
            C1 = menger_curvature(curr_pts[:3])
            C2 = menger_curvature(curr_pts[1:])

            # make sure right curvature is positive!
            # keep moving the right edge inwards (and adjusting the middle points accordingly)
            while C2 < 0:
                new_reg_x = calc_golden_x(curr_pts[0].reg_x, curr_pts[2].reg_x)
                curr_pts[3] = curr_pts[2]
                curr_pts[2] = curr_pts[1]
                curr_pts[1] = eval_L_curve(new_reg_x)
                C2 = menger_curvature(curr_pts[1:])
            C1 = menger_curvature(curr_pts[:3])

            # case 1: left 3 points are better
            # [a, b, c, d] --> [a, (a*phi+c)/(1+phi), b, c]
            if C1 > C2:
                last_opt = curr_pts[1]
                new_reg_x = calc_golden_x(curr_pts[0].reg_x, curr_pts[2].reg_x)
                curr_pts[3] = curr_pts[2]
                curr_pts[2] = curr_pts[1]
                curr_pts[1] = eval_L_curve(new_reg_x)

            # case 2: right 3 points are better
            # [a, b, c, d] --> [b, c, b+d-c, d]
            else:
                last_opt = curr_pts[2]
                new_reg_x = (
                    curr_pts[1].reg_x + curr_pts[3].reg_x - curr_pts[2].reg_x
                )
                curr_pts[0] = curr_pts[1]
                curr_pts[1] = curr_pts[2]
                curr_pts[2] = eval_L_curve(new_reg_x)

            itr += 1
            search_range = curr_pts[3].reg_x - curr_pts[0].reg_x
            if verbose:
                print(
                    "Iteration: %d, deviation of the regularization interval: %.2e"
                    % (itr, search_range)
                )

        if verbose:
            print("Final regularization parameter chosen: %.2e" % last_opt.reg)

        # Return 3 solutions: the parameter chosen by the L-curve +/- epsilon
        # try to not keep intermediate values in memory
        f_real = []
        f_real.append(
            ifft_f_real(compute_f_real_f(last_opt.reg_x - epsilon_auto))
        )
        f_real.append(ifft_f_real(compute_f_real_f(last_opt.reg_x)))
        f_real.append(
            ifft_f_real(compute_f_real_f(last_opt.reg_x + epsilon_auto))
        )

        if output_lambda:
            return np.array(f_real), 10 ** (last_opt.reg_x)
        else:
            return np.array(f_real)

    else:
        f_real_f = compute_f_real_f(np.log10(reg_re))
        f_real = ifft_f_real(f_real_f)
        return f_real


#     return f_real, opt_list # if wanted some kind of plotting option, could save all points visited


def Dual_variable_Tikhonov_deconv_3D(
    AHA, b_vec, determinant=None, use_gpu=False, gpu_id=0, move_cpu=True
):
    """

    3D Tikhonov deconvolution to solve for phase and absorption with weak object transfer function

    Parameters
    ----------
        AHA         : list
                      A^H times A matrix stored with a list of 4 3D numpy array (4 diagonal matrices)
                      | AHA[0]  AHA[1] |
                      | AHA[2]  AHA[3] |

        b_vec       : list
                      measured intensity stored with a list of 2 3D numpy array (2 vectors)
                      | b_vec[0] |
                      | b_vec[1] |

        determinant : numpy.ndarray
                      determinant of the AHA matrix in 3D space

        use_gpu     : bool
                      option to use gpu or not

        gpu_id      : int
                      number refering to which gpu will be used

        move_cpu    : bool
                      option to move the array from gpu to cpu

    Returns
    -------
        f_real      : numpy.ndarray
                      3D real scattering potential (unscaled phase) reconstruction with the size of (Ny, Nx, Nz)

        f_imag      : numpy.ndarray
                      3D imaginary scattering potential (unscaled absorption) reconstruction with the size of (Ny, Nx, Nz)
    """

    if determinant is None:
        determinant = AHA[0] * AHA[3] - AHA[1] * AHA[2]

    f_real_f = (b_vec[0] * AHA[3] - b_vec[1] * AHA[1]) / determinant
    f_imag_f = (b_vec[1] * AHA[0] - b_vec[0] * AHA[2]) / determinant

    if use_gpu:
        globals()["cp"] = __import__("cupy")
        cp.cuda.Device(gpu_id).use()

        f_real = cp.real(cp.fft.ifftn(f_real_f))
        f_imag = cp.real(cp.fft.ifftn(f_imag_f))

        if move_cpu:
            f_real = cp.asnumpy(f_real)
            f_imag = cp.asnumpy(f_imag)

    else:
        f_real = np.real(ifftn(f_real_f))
        f_imag = np.real(ifftn(f_imag_f))

    return f_real, f_imag


def single_variable_admm_tv_deconvolution_3D(
    S0_stack,
    H_eff,
    rho=1e-5,
    reg_re=1e-4,
    lambda_re=1e-3,
    itr=20,
    verbose=False,
    use_gpu=False,
    gpu_id=0,
):
    """

    3D TV deconvolution to solve for phase with weak object transfer function

    ADMM formulation:

        0.5 * || A*x - b ||_2^2 + lambda * || z ||_1 + 0.5 * rho * || D*x - z + u ||_2^2

    Parameters
    ----------
        S0_stack  : numpy.ndarray
                    S0 z-stack for 3D phase deconvolution with size of (Ny, Nx, Nz)

        H_eff     : numpy.ndarray
                    effective transfer function with size of (Ny, Nx, Nz)

        reg_re    : float
                    Tikhonov regularization parameter

        rho       : float
                    ADMM rho parameter

        lambda_re : float
                    TV regularization parameter for phase

        itr       : int
                    number of iterations of ADMM algorithm

        verbose   : bool
                    option to display progress of the computation

        use_gpu   : bool
                    option to use gpu or not

        gpu_id    : int
                    number refering to which gpu will be used

    Returns
    -------
        f_real    : numpy.ndarray
                    3D unscaled phase reconstruction with the size of (Ny, Nx, Nz)
    """

    N, M, N_defocus = S0_stack.shape

    Dx = np.zeros((N, M, N_defocus))
    Dx[0, 0, 0] = 1
    Dx[0, -1, 0] = -1
    Dy = np.zeros((N, M, N_defocus))
    Dy[0, 0, 0] = 1
    Dy[-1, 0, 0] = -1
    Dz = np.zeros((N, M, N_defocus))
    Dz[0, 0, 0] = 1
    Dz[0, 0, -1] = -1

    if use_gpu:
        globals()["cp"] = __import__("cupy")
        cp.cuda.Device(gpu_id).use()

        S0_stack_f = cp.fft.fftn(
            cp.array(S0_stack.astype("float32")), axes=(0, 1, 2)
        )
        H_eff = cp.array(H_eff.astype("complex64"))

        Dx = cp.fft.fftn(cp.array(Dx), axes=(0, 1, 2))
        Dy = cp.fft.fftn(cp.array(Dy), axes=(0, 1, 2))
        Dz = cp.fft.fftn(cp.array(Dz), axes=(0, 1, 2))

        rho_term = (
            rho * (cp.conj(Dx) * Dx + cp.conj(Dy) * Dy + cp.conj(Dz) * Dz)
            + reg_re
        )
        AHA = cp.abs(H_eff) ** 2 + rho_term
        b_vec = S0_stack_f * cp.conj(H_eff)

        z_para = cp.zeros((3, N, M, N_defocus))
        u_para = cp.zeros((3, N, M, N_defocus))
        D_vec = cp.zeros((3, N, M, N_defocus))

        for i in range(itr):
            v_para = cp.fft.fftn(z_para - u_para, axes=(1, 2, 3))
            b_vec_new = b_vec + rho * (
                cp.conj(Dx) * v_para[0]
                + cp.conj(Dy) * v_para[1]
                + cp.conj(Dz) * v_para[2]
            )

            f_real = cp.real(cp.fft.ifftn(b_vec_new / AHA, axes=(0, 1, 2)))

            D_vec[0] = f_real - cp.roll(f_real, -1, axis=1)
            D_vec[1] = f_real - cp.roll(f_real, -1, axis=0)
            D_vec[2] = f_real - cp.roll(f_real, -1, axis=2)

            z_para = D_vec + u_para

            z_para = softTreshold(
                z_para, lambda_re / rho, use_gpu=True, gpu_id=gpu_id
            )

            u_para += D_vec - z_para

            if verbose:
                print("Number of iteration computed (%d / %d)" % (i + 1, itr))

            if i == itr - 1:
                f_real = cp.asnumpy(f_real)

    else:
        S0_stack_f = fftn(S0_stack, axes=(0, 1, 2))

        Dx = fftn(Dx, axes=(0, 1, 2))
        Dy = fftn(Dy, axes=(0, 1, 2))
        Dz = fftn(Dz, axes=(0, 1, 2))

        rho_term = (
            rho * (np.conj(Dx) * Dx + np.conj(Dy) * Dy + np.conj(Dz) * Dz)
            + reg_re
        )
        AHA = np.abs(H_eff) ** 2 + rho_term
        b_vec = S0_stack_f * np.conj(H_eff)

        z_para = np.zeros((3, N, M, N_defocus))
        u_para = np.zeros((3, N, M, N_defocus))
        D_vec = np.zeros((3, N, M, N_defocus))

        for i in range(itr):
            v_para = fftn(z_para - u_para, axes=(1, 2, 3))
            b_vec_new = b_vec + rho * (
                np.conj(Dx) * v_para[0]
                + np.conj(Dy) * v_para[1]
                + np.conj(Dz) * v_para[2]
            )

            f_real = np.real(ifftn(b_vec_new / AHA, axes=(0, 1, 2)))

            D_vec[0] = f_real - np.roll(f_real, -1, axis=1)
            D_vec[1] = f_real - np.roll(f_real, -1, axis=0)
            D_vec[2] = f_real - np.roll(f_real, -1, axis=2)

            z_para = D_vec + u_para

            z_para = softTreshold(z_para, lambda_re / rho)

            u_para += D_vec - z_para

            if verbose:
                print("Number of iteration computed (%d / %d)" % (i + 1, itr))

    return f_real


def Dual_variable_ADMM_TV_deconv_3D(
    AHA,
    b_vec,
    rho,
    lambda_re,
    lambda_im,
    itr,
    verbose,
    use_gpu=False,
    gpu_id=0,
):
    """

    3D TV deconvolution to solve for phase and absorption with weak object transfer function

    ADMM formulation:

        0.5 * || A*x - b ||_2^2 + lambda * || z ||_1 + 0.5 * rho * || D*x - z + u ||_2^2

    Parameters
    ----------
        AHA        : list
                     A^H times A matrix stored with a list of 4 3D numpy array (4 diagonal matrices)
                     | AHA[0]  AHA[1] |
                     | AHA[2]  AHA[3] |

        b_vec      : list
                     measured intensity stored with a list of 2 3D numpy array (2 vectors)
                     | b_vec[0] |
                     | b_vec[1] |

        rho        : float
                     ADMM rho parameter

        lambda_re  : float
                     TV regularization parameter for phase

        lambda_im  : float
                     TV regularization parameter for absorption

        itr        : int
                     number of iterations of ADMM algorithm

        verbose    : bool
                     option to display progress of the computation

        use_gpu    : bool
                     option to use gpu or not

        gpu_id     : int
                     number refering to which gpu will be used

    Returns
    -------
        f_real     : numpy.ndarray
                     3D real scattering potential (unscaled phase) reconstruction with the size of (Ny, Nx, Nz)

        f_imag     : numpy.ndarray
                     3D imaginary scattering potential (unscaled absorption) reconstruction with the size of (Ny, Nx, Nz)
    """

    # ADMM deconvolution with anisotropic TV regularization

    N, M, L = b_vec[0].shape
    Dx = np.zeros((N, M, L))
    Dx[0, 0, 0] = 1
    Dx[0, -1, 0] = -1
    Dy = np.zeros((N, M, L))
    Dy[0, 0, 0] = 1
    Dy[-1, 0, 0] = -1
    Dz = np.zeros((N, M, L))
    Dz[0, 0, 0] = 1
    Dz[0, 0, -1] = -1

    if use_gpu:
        globals()["cp"] = __import__("cupy")
        cp.cuda.Device(gpu_id).use()

        Dx = cp.fft.fftn(cp.array(Dx))
        Dy = cp.fft.fftn(cp.array(Dy))
        Dz = cp.fft.fftn(cp.array(Dz))

        rho_term = rho * (
            cp.conj(Dx) * Dx + cp.conj(Dy) * Dy + cp.conj(Dz) * Dz
        )

        z_para = cp.zeros((6, N, M, L))
        u_para = cp.zeros((6, N, M, L))
        D_vec = cp.zeros((6, N, M, L))

    else:
        Dx = fftn(Dx)
        Dy = fftn(Dy)
        Dz = fftn(Dz)

        rho_term = rho * (
            np.conj(Dx) * Dx + np.conj(Dy) * Dy + np.conj(Dz) * Dz
        )

        z_para = np.zeros((6, N, M, L))
        u_para = np.zeros((6, N, M, L))
        D_vec = np.zeros((6, N, M, L))

    AHA[0] = AHA[0] + rho_term
    AHA[3] = AHA[3] + rho_term

    determinant = AHA[0] * AHA[3] - AHA[1] * AHA[2]

    for i in range(itr):
        if use_gpu:
            v_para = cp.fft.fftn(z_para - u_para, axes=(1, 2, 3))
            b_vec_new = [
                b_vec[0]
                + rho
                * (
                    cp.conj(Dx) * v_para[0]
                    + cp.conj(Dy) * v_para[1]
                    + cp.conj(Dz) * v_para[2]
                ),
                b_vec[1]
                + rho
                * (
                    cp.conj(Dx) * v_para[3]
                    + cp.conj(Dy) * v_para[4]
                    + cp.conj(Dz) * v_para[5]
                ),
            ]

            f_real, f_imag = Dual_variable_Tikhonov_deconv_3D(
                AHA,
                b_vec_new,
                determinant=determinant,
                use_gpu=use_gpu,
                gpu_id=gpu_id,
                move_cpu=not use_gpu,
            )

            D_vec[0] = f_real - cp.roll(f_real, -1, axis=1)
            D_vec[1] = f_real - cp.roll(f_real, -1, axis=0)
            D_vec[2] = f_real - cp.roll(f_real, -1, axis=2)
            D_vec[3] = f_imag - cp.roll(f_imag, -1, axis=1)
            D_vec[4] = f_imag - cp.roll(f_imag, -1, axis=0)
            D_vec[5] = f_imag - cp.roll(f_imag, -1, axis=2)

            z_para = D_vec + u_para

            z_para[:3, :, :] = softTreshold(
                z_para[:3, :, :], lambda_re / rho, use_gpu=True, gpu_id=gpu_id
            )
            z_para[3:, :, :] = softTreshold(
                z_para[3:, :, :], lambda_im / rho, use_gpu=True, gpu_id=gpu_id
            )

            u_para += D_vec - z_para

            if i == itr - 1:
                f_real = cp.asnumpy(f_real)
                f_imag = cp.asnumpy(f_imag)

        else:
            v_para = fftn(z_para - u_para, axes=(1, 2, 3))
            b_vec_new = [
                b_vec[0]
                + rho
                * (
                    np.conj(Dx) * v_para[0]
                    + np.conj(Dy) * v_para[1]
                    + np.conj(Dz) * v_para[2]
                ),
                b_vec[1]
                + rho
                * (
                    np.conj(Dx) * v_para[3]
                    + np.conj(Dy) * v_para[4]
                    + np.conj(Dz) * v_para[5]
                ),
            ]

            f_real, f_imag = Dual_variable_Tikhonov_deconv_3D(
                AHA, b_vec_new, determinant=determinant
            )

            D_vec[0] = f_real - np.roll(f_real, -1, axis=1)
            D_vec[1] = f_real - np.roll(f_real, -1, axis=0)
            D_vec[2] = f_real - np.roll(f_real, -1, axis=2)
            D_vec[3] = f_imag - np.roll(f_imag, -1, axis=1)
            D_vec[4] = f_imag - np.roll(f_imag, -1, axis=0)
            D_vec[5] = f_imag - np.roll(f_imag, -1, axis=2)

            z_para = D_vec + u_para

            z_para[:3, :, :] = softTreshold(z_para[:3, :, :], lambda_re / rho)
            z_para[3:, :, :] = softTreshold(z_para[3:, :, :], lambda_im / rho)

            u_para += D_vec - z_para

        if verbose:
            print("Number of iteration computed (%d / %d)" % (i + 1, itr))

    return f_real, f_imag


def cylindrical_shell_local_orientation(
    VOI, ps, psz, scale, beta=0.5, c_para=0.5, evec_idx=0
):
    """

    segmentation of 3D cylindrical shell structure and the estimation of local orientation of the geometry

    Parameters
    ----------
        VOI      : numpy.ndarray
                   3D volume of interest

        ps       : float
                   transverse pixel size of the image space

        psz      : float
                   axial step size of the image space

        scale    : list
                   list of feature size to scan through for a segmentation including multi-scale feature size

        beta     : float
                   value to control whether the segmentation need to highlight more or less on the shell-like feature
                   larger  -> highlight the most strong shell-like feature (more sparse segmentation)
                   smaller -> highlight somewhat shell-like feature (more connected segmentation)

        c_para   : float
                   value to control whether the segmentation need to highlight more or less on the structure with overall large gradient

        evec_idx : int
                   the index of eigenvector we consider for local orientation
                   0: smallest eigenvector, which cooresponds to the local orientation along the cylindrical shell
                   2: largest eigenvector, which cooresponds to the local orientation normal to the cylindrical shell


    Returns
    -------
        azimuth  : numpy.ndarray
                   the azimuthal angle of the computed local orientation

        theta    : numpy.ndarray
                   the theta part of the computed local orientation

        V_func   : numpy.ndarray
                   the segmentation map of the cylindrical shell structure

        kernel   : numpy.ndarray
                   the kernel corresponding to the highlighting feature sizes in the segmentation with the size of (N_scale, Ny, Nx, Nz)

    """

    # Hessian matrix filtering

    N, M, L = VOI.shape

    x_r = (np.r_[:M] - M // 2) * ps
    y_r = (np.r_[:N] - N // 2) * ps
    z_r = (np.r_[:L] - L // 2) * psz

    xx_r, yy_r, zz_r = np.meshgrid(x_r, y_r, z_r)

    fx_r = ifftshift((np.r_[:M] - M // 2) / ps / M)
    fy_r = ifftshift((np.r_[:N] - N // 2) / ps / N)
    fz_r = ifftshift((np.r_[:L] - L // 2) / psz / L)
    fxx_r, fyy_r, fzz_r = np.meshgrid(fx_r, fy_r, fz_r)

    diff_filter = np.zeros((3, N, M, L), complex)
    diff_filter[0] = 1j * 2 * np.pi * fxx_r
    diff_filter[1] = 1j * 2 * np.pi * fyy_r
    diff_filter[2] = 1j * 2 * np.pi * fzz_r

    V_func = np.zeros_like(VOI)
    kernel = np.zeros((len(scale),) + (N, M, L))

    t0 = time.time()

    for i, s in enumerate(scale):
        kernel[i] = np.exp(
            -(xx_r**2 + yy_r**2 + zz_r**2) / 2 / s**2
        ) / (2 * np.pi * s**2) ** (3 / 2)
        Gaussian_3D_f = fftn(ifftshift(kernel[i])) * (ps * ps * psz)

        VOI_filtered = np.zeros((3, 3, N, M, L))

        for p in range(3):
            for q in range(3):
                Hessian_filter = (
                    ((s) ** 2)
                    * Gaussian_3D_f
                    * diff_filter[p]
                    * diff_filter[q]
                )
                VOI_filtered[p, q] = np.real(
                    ifftn(fftn(VOI) * Hessian_filter) / (ps * ps * psz)
                )

        eigen_val, eigen_vec = np.linalg.eig(
            np.transpose(VOI_filtered, (2, 3, 4, 0, 1))
        )

        eig_val_idx = np.zeros((3, N, M, L))
        for p in range(3):
            eig_val_idx[p] = np.argpartition(np.abs(eigen_val), p, axis=3)[
                :, :, :, p
            ]

        eig_val_sort = np.zeros((3, N, M, L), complex)

        for p in range(3):
            for q in range(3):
                eig_val_sort[q, eig_val_idx[q] == p] = eigen_val[
                    eig_val_idx[q] == p, p
                ]

        RB = np.abs(eig_val_sort[2]) / np.sqrt(
            np.abs(eig_val_sort[0]) * np.abs(eig_val_sort[1])
        )
        S = np.sqrt(np.sum(np.abs(eig_val_sort) ** 2, axis=0))

        c = c_para * np.max(S)

        V_func_temp = (1 - np.exp(-(RB**2) / 2 / beta**2)) * (
            1 - np.exp(-(S**2) / 2 / c**2)
        )
        V_func_temp[np.real(eig_val_sort[2]) > 0] = 0

        if i == 0:
            V_func = V_func_temp.copy()
            orientation_vec = np.zeros((N, M, L, 3))

            for p in range(3):
                orientation_vec[eig_val_idx[evec_idx] == p] = np.real(
                    eigen_vec[eig_val_idx[evec_idx] == p, :, p]
                )
        else:
            larger_V_idx = V_func_temp > V_func
            V_func[larger_V_idx] = V_func_temp[larger_V_idx]

            for p in range(3):
                orientation_vec[
                    np.logical_and(larger_V_idx, eig_val_idx[evec_idx] == p)
                ] = np.real(
                    eigen_vec[
                        np.logical_and(
                            larger_V_idx, eig_val_idx[evec_idx] == p
                        ),
                        :,
                        p,
                    ]
                )

        print(
            "Finish V_map computation for scale = %.2f, elapsed time: %.2f"
            % (s, time.time() - t0)
        )

    orientation_vec = np.transpose(orientation_vec, (3, 0, 1, 2))

    #     orientation_vec[0] = orientation_vec[0]*ps
    #     orientation_vec[1] = orientation_vec[1]*ps
    #     orientation_vec[2] = orientation_vec[2]*psz

    norm = np.sqrt(np.sum(np.abs(orientation_vec) ** 2, axis=0))
    theta = np.arccos(np.clip(orientation_vec[2] / norm, -1, 1))
    azimuth = np.arctan2(orientation_vec[1], orientation_vec[0])
    azimuth = azimuth % (2 * np.pi)
    theta[azimuth > np.pi] = np.pi - theta[azimuth > np.pi]
    azimuth[azimuth > np.pi] = azimuth[azimuth > np.pi] - np.pi

    print(
        "Finish local orientation extraction, elapsed time:"
        + str(time.time() - t0)
    )

    return azimuth, theta, V_func, kernel


def integer_factoring(integer):
    """

    find all the factors of an integer

    Parameters
    ----------
        integer : int
                  integer to be factored
    Returns
    -------
        factors : list
                  list containing all the factors of the input integer

    """

    if not isinstance(integer, int):
        raise ValueError("integer should be an int")

    factors = []

    for half_factor in range(1, int(np.sqrt(integer)) + 1):
        if integer % half_factor == 0:
            factors.append(half_factor)
            if integer // half_factor != half_factor:
                factors.append(integer // half_factor)

    factors.sort()
    return factors


def generate_FOV_splitting_parameters(
    img_size, overlapping_range, max_image_size
):
    """

    calculate the overlap and pixels of increment for sub-FOV processing

    Parameters
    ----------
        img_size          : tuple or list
                            the original size of the image in the format of (Ny, Nx)

        overlapping_range : tuple or list
                            the targeted range for the number of overlapping pixels in the format of (overlap_min, overlap_max)

        max_image_size    : tuple or list
                            the maximal accepted size of the sub-FOV in the format of (Ny, Nx)


    Returns
    -------
        overlap           : int
                            the number of overlapping pixels

        N_space           : int
                            the number of y-increment pixels

        M_space           : int
                            the number of x-increment pixels

    """

    overlap = 0
    N_space = 0
    M_space = 0

    for i in range(overlapping_range[0], overlapping_range[1]):
        pre_N_space = np.max(
            [
                x
                for x in integer_factoring(img_size[0] - i)
                if x <= max_image_size[0] - i
            ]
        )
        pre_M_space = np.max(
            [
                x
                for x in integer_factoring(img_size[1] - i)
                if x <= max_image_size[1] - i
            ]
        )

        if (
            pre_N_space > N_space
            and pre_M_space > M_space
            and (pre_N_space + i) % 2 == 0
            and (pre_M_space + i) % 2 == 0
        ):
            overlap = i
            N_space = pre_N_space
            M_space = pre_M_space

    print("Optimal number of overlapping is %d pixels" % (overlap))
    print("The corresponding maximal N_space is %d pixels" % (N_space))
    print("The corresponding maximal M_space is %d pixels" % (M_space))

    return overlap, N_space, M_space


def generate_sub_FOV_coordinates(img_size, img_space, overlap):
    """

    calculate the starting pixel indices of each sub-FOV

    Parameters
    ----------
        img_size  : tuple or list
                    the original size of the image in the format of (Ny, Nx)

        img_space : tuple or list
                    the number of x- and y-increment pixels in the format of (Ny, Nx)

        overlap   : tuple or list
                    the number of overlapping pixels in y and x directions in the format of (Ny, Nx)


    Returns
    -------
        ns        : int
                    the starting pixel indices in y direction

        ms        : int
                    the starting pixel indices in x direction

    """

    N_full, M_full = img_size
    N_space, M_space = img_space

    Ns = N_space + overlap[0]
    Ms = M_space + overlap[1]

    num_N = np.floor(N_full / N_space)
    num_M = np.floor(M_full / M_space)

    end_N = (num_N - 1) * N_space + Ns
    end_M = (num_M - 1) * M_space + Ms

    if end_N <= N_full:
        ns = np.r_[0:num_N] * N_space
    else:
        ns = np.r_[0 : (num_N - 1)] * N_space
        num_N -= 1
        end_N = (num_N - 1) * N_space + Ns

    if end_M <= M_full:
        ms = np.r_[0:num_M] * M_space
    else:
        ms = np.r_[0 : (num_M - 1)] * M_space
        num_M -= 1
        end_M = (num_M - 1) * M_space + Ms

    print(
        "Last pixel in (y,x) dimension processed is (%d, %d)" % (end_N, end_M)
    )

    ms, ns = np.meshgrid(ms, ns)
    ms = ms.flatten()
    ns = ns.flatten()

    return ns, ms


def image_stitching(
    coord_list, overlap, file_loading_func, gen_ref_map=True, ref_stitch=None
):
    """

    stitch images (with size (Ny, Nx, ...)) with alpha blending algorithm given the image coordinate, overlap, and file_loading_functions

    Parameters
    ----------
        coord_list        : tuple
                            a tuple containing two np.arrays for the y- and x-coordinate of the sub-FOVs
                            e.g. (y_idx_list, x_idx_list)
                            y_idx_list = [0, 0, 1, 1]
                            x_idx_list = [0, 1, 0, 1]
                                         |
                                [0, 0]   |   [0, 1]
                                         |
                            --------------------------  for a FOV like this, the list suggest an order of [0,0] -> [0, 1] -> [1,0] -> [1, 1]
                                         |
                                [1, 0]   |   [1, 1]
                                         |


        overlap           : tuple or list
                            the number of overlapping pixels in y and x directions in the format of (Ny, Nx)

        file_loading_func : func
                            a function handle that receives one integer parameter (p) to load the p-th array in the disk with the shape of (Ny_sub, Nx_sub, ...)

        gen_ref_map       : bool
                            an option to generate a normalization map for the stitching algorithm

        ref_stitch        : numpy.ndarray
                            a precomputed normalization map with the shape of (Ny, Nx)


    Returns
    -------
        img_normalized    : numpy.ndarray
                            the stitched array with the shape of (Ny, Nx, ...)

        ref_stitch        : int
                            the computed normalization map with the shape of (Ny, Nx)

    """

    row_list = coord_list[0]
    column_list = coord_list[1]

    overlap_y = overlap[0]
    overlap_x = overlap[1]

    num_row = int(np.max(np.array(row_list)) + 1)
    num_column = int(np.max(np.array(column_list)) + 1)

    t0 = time.time()
    for i in range(num_row * num_column):
        row_idx = int(row_list[i])
        column_idx = int(column_list[i])

        img_i = file_loading_func(i)

        if i == 0:
            Ns, Ms = img_i.shape[:2]
            N_full = (num_row - 1) * (Ns - overlap_y) + Ns
            M_full = (num_column - 1) * (Ms - overlap_x) + Ms
            if img_i.ndim == 2:
                img_stitch = np.zeros((N_full, M_full))
            else:
                img_stitch = np.zeros((N_full, M_full) + img_i.shape[2:])

            if gen_ref_map:
                ref_stitch = np.zeros((N_full, M_full))
            else:
                if ref_stitch is None:
                    raise ValueError(
                        "Make gen_ref_map True if ref_stitch is None"
                    )

        if gen_ref_map:
            ref_i = np.ones(img_i.shape[:2])

        # center
        if (
            np.sum(row_idx == np.r_[1 : num_row - 1])
            * np.sum(column_idx == np.r_[1 : num_column - 1])
            == 1
        ):
            for p in range(overlap_y):
                img_i[-1 - p, :] = img_i[-1 - p, :] * p / overlap_y
                img_i[p, :] = img_i[p, :] * p / overlap_y
                if gen_ref_map:
                    ref_i[-1 - p, :] = ref_i[-1 - p, :] * p / overlap_y
                    ref_i[p, :] = ref_i[p, :] * p / overlap_y

            for p in range(overlap_x):
                img_i[:, p] = img_i[:, p] * p / overlap_x
                img_i[:, -1 - p] = img_i[:, -1 - p] * p / overlap_x
                if gen_ref_map:
                    ref_i[:, p] = ref_i[:, p] * p / overlap_x
                    ref_i[:, -1 - p] = ref_i[:, -1 - p] * p / overlap_x

        # top
        if (
            np.sum(row_idx == 0)
            * np.sum(column_idx == np.r_[1 : num_column - 1])
            == 1
        ):
            for p in range(overlap_y):
                img_i[-1 - p, :] = img_i[-1 - p, :] * p / overlap_y
                #             img_i[p,:] = img_i[p,:]*p/overlap_y
                if gen_ref_map:
                    ref_i[-1 - p, :] = ref_i[-1 - p, :] * p / overlap_y
            #             ref_i[p,:] = ref_i[p,:]*p/overlap_y

            for p in range(overlap_x):
                img_i[:, p] = img_i[:, p] * p / overlap_x
                img_i[:, -1 - p] = img_i[:, -1 - p] * p / overlap_x
                if gen_ref_map:
                    ref_i[:, p] = ref_i[:, p] * p / overlap_x
                    ref_i[:, -1 - p] = ref_i[:, -1 - p] * p / overlap_x

        # bottom
        if (
            np.sum(row_idx == num_row - 1)
            * np.sum(column_idx == np.r_[1 : num_column - 1])
            == 1
        ):
            for p in range(overlap_y):
                #             img_i[-1-p,:] = img_i[-1-p,:]*p/overlap_y
                img_i[p, :] = img_i[p, :] * p / overlap_y
                if gen_ref_map:
                    #             ref_i[-1-p,:] = ref_i[-1-p,:]*p/overlap_y
                    ref_i[p, :] = ref_i[p, :] * p / overlap_y

            for p in range(overlap_x):
                img_i[:, p] = img_i[:, p] * p / overlap_x
                img_i[:, -1 - p] = img_i[:, -1 - p] * p / overlap_x
                if gen_ref_map:
                    ref_i[:, p] = ref_i[:, p] * p / overlap_x
                    ref_i[:, -1 - p] = ref_i[:, -1 - p] * p / overlap_x

        # left
        if (
            np.sum(row_idx == np.r_[1 : num_row - 1]) * np.sum(column_idx == 0)
            == 1
        ):
            for p in range(overlap_y):
                img_i[-1 - p, :] = img_i[-1 - p, :] * p / overlap_y
                img_i[p, :] = img_i[p, :] * p / overlap_y
                if gen_ref_map:
                    ref_i[-1 - p, :] = ref_i[-1 - p, :] * p / overlap_y
                    ref_i[p, :] = ref_i[p, :] * p / overlap_y

            for p in range(overlap_x):
                #             img_i[:,p] = img_i[:,p]*p/overlap_x
                img_i[:, -1 - p] = img_i[:, -1 - p] * p / overlap_x
                if gen_ref_map:
                    #             ref_i[:,p] = ref_i[:,p]*p/overlap_x
                    ref_i[:, -1 - p] = ref_i[:, -1 - p] * p / overlap_x

        # right
        if (
            np.sum(row_idx == np.r_[1 : num_row - 1])
            * np.sum(column_idx == num_column - 1)
            == 1
        ):
            for p in range(overlap_y):
                img_i[-1 - p, :] = img_i[-1 - p, :] * p / overlap_y
                img_i[p, :] = img_i[p, :] * p / overlap_y
                if gen_ref_map:
                    ref_i[-1 - p, :] = ref_i[-1 - p, :] * p / overlap_y
                    ref_i[p, :] = ref_i[p, :] * p / overlap_y

            for p in range(overlap_x):
                img_i[:, p] = img_i[:, p] * p / overlap_x
                #             img_i[:,-1-p] = img_i[:,-1-p]*p/overlap_x
                if gen_ref_map:
                    ref_i[:, p] = ref_i[:, p] * p / overlap_x
        #             ref_i[:,-1-p] = ref_i[:,-1-p]*p/overlap_x

        # top left
        if np.sum(row_idx == 0) * np.sum(column_idx == 0) == 1:
            for p in range(overlap_y):
                img_i[-1 - p, :] = img_i[-1 - p, :] * p / overlap_y
                #             img_i[p,:] = img_i[p,:]*p/overlap_y
                if gen_ref_map:
                    ref_i[-1 - p, :] = ref_i[-1 - p, :] * p / overlap_y
            #             ref_i[p,:] = ref_i[p,:]*p/overlap_y

            for p in range(overlap_x):
                #             img_i[:,p] = img_i[:,p]*p/overlap_x
                img_i[:, -1 - p] = img_i[:, -1 - p] * p / overlap_x
                if gen_ref_map:
                    #             ref_i[:,p] = ref_i[:,p]*p/overlap_x
                    ref_i[:, -1 - p] = ref_i[:, -1 - p] * p / overlap_x

        # top right
        if np.sum(row_idx == 0) * np.sum(column_idx == num_column - 1) == 1:
            for p in range(overlap_y):
                img_i[-1 - p, :] = img_i[-1 - p, :] * p / overlap_y
                #             img_i[p,:] = img_i[p,:]*p/overlap_y
                if gen_ref_map:
                    ref_i[-1 - p, :] = ref_i[-1 - p, :] * p / overlap_y
            #             ref_i[p,:] = ref_i[p,:]*p/overlap_y

            for p in range(overlap_x):
                img_i[:, p] = img_i[:, p] * p / overlap_x
                #             img_i[:,-1-p] = img_i[:,-1-p]*p/overlap_x
                if gen_ref_map:
                    ref_i[:, p] = ref_i[:, p] * p / overlap_x
        #             ref_i[:,-1-p] = ref_i[:,-1-p]*p/overlap_x

        # bottom left
        if np.sum(row_idx == num_row - 1) * np.sum(column_idx == 0) == 1:
            for p in range(overlap_y):
                #             img_i[-1-p,:] = img_i[-1-p,:]*p/overlap_y
                img_i[p, :] = img_i[p, :] * p / overlap_y
                if gen_ref_map:
                    #             ref_i[-1-p,:] = ref_i[-1-p,:]*p/overlap_y
                    ref_i[p, :] = ref_i[p, :] * p / overlap_y

            for p in range(overlap_x):
                #             img_i[:,p] = img_i[:,p]*p/overlap_x
                img_i[:, -1 - p] = img_i[:, -1 - p] * p / overlap_x
                if gen_ref_map:
                    #             ref_i[:,p] = ref_i[:,p]*p/overlap_x
                    ref_i[:, -1 - p] = ref_i[:, -1 - p] * p / overlap_x

        # bottom right
        if (
            np.sum(row_idx == num_row - 1)
            * np.sum(column_idx == num_column - 1)
            == 1
        ):
            for p in range(overlap_y):
                #             img_i[-1-p,:] = img_i[-1-p,:]*p/overlap_y
                img_i[p, :] = img_i[p, :] * p / overlap_y
                if gen_ref_map:
                    #             ref_i[-1-p,:] = ref_i[-1-p,:]*p/overlap_y
                    ref_i[p, :] = ref_i[p, :] * p / overlap_y

            for p in range(overlap_x):
                img_i[:, p] = img_i[:, p] * p / overlap_x
                #             img_i[:,-1-p] = img_i[:,-1-p]*p/overlap_x
                if gen_ref_map:
                    ref_i[:, p] = ref_i[:, p] * p / overlap_x
        #             ref_i[:,-1-p] = ref_i[:,-1-p]*p/overlap_x

        pad_tuple_ref = (
            (
                row_idx * (Ns - overlap_y),
                (num_row - 1 - row_idx) * (Ns - overlap_y),
            ),
            (
                column_idx * (Ms - overlap_x),
                (num_column - 1 - column_idx) * (Ms - overlap_x),
            ),
        )
        pad_tuple = pad_tuple_ref

        if img_stitch.ndim > ref_stitch.ndim:
            for extend_idx in range(img_stitch.ndim - ref_stitch.ndim):
                pad_tuple += ((0, 0),)

        img_temp = np.pad(img_i, pad_tuple, mode="constant")
        img_stitch += img_temp

        if gen_ref_map:
            ref_temp = np.pad(ref_i, pad_tuple_ref, mode="constant")
            ref_stitch += ref_temp

        if np.mod(i + 1, 1) == 0:
            print(
                "Processed positions (%d / %d), elapsed time: %.2f"
                % (i + 1, num_row * num_column, time.time() - t0)
            )

    ref_stitch_extend = ref_stitch.copy()
    if img_stitch.ndim > ref_stitch.ndim:
        for extend_idx in range(img_stitch.ndim - ref_stitch.ndim):
            ref_stitch_extend = ref_stitch_extend[..., np.newaxis]

    img_normalized = (
        img_stitch * ref_stitch_extend / (ref_stitch_extend + 1e-6)
    )

    if gen_ref_map:
        return img_normalized, ref_stitch
    else:
        return img_normalized


def orientation_3D_continuity_map(
    azimuth, theta, psz_ps_ratio=None, avg_px_size=10, reg_ret_pr=1e-1
):
    """

    calculate the 3D orientation continuity map that is used to suppress noisy retardance measurements

    Parameters
    ----------
        azimuth           : numpy.ndarray
                            reconstructed in-plane orientation with the size of (N, M) for 2D and (N, M, N_defocus) for 3D

        theta             : numpy.ndarray
                            reconstructed out-of-plane inclination with the size of (N, M) for 2D and (N, M, N_defocus) for 3D

        psz_ps_ratio      : float
                            the ratio of the sampling size in z and in xy

        avg_px_size       : int
                            size of the smoothing uniform filter to enforce spatial continuity
                            (larger --> smoother feature, lower --> sharper feature)

        reg_ret_pr        : float
                            regularization parameters for principal retardance estimation


    Returns
    -------
        retardance_pr_avg : numpy.ndarray
                            the computed orientation continuity map with the size of (N, M) for 2D and (N, M, N_defocus) for 3D

    """

    img_size = azimuth.shape
    img_dim = azimuth.ndim

    f_tensor_unit_ret = np.zeros((7,) + img_size)
    f_tensor_unit_ret[2] = (
        -np.ones(img_size) * (np.sin(theta) ** 2) * np.cos(2 * azimuth)
    )
    f_tensor_unit_ret[3] = (
        -np.ones(img_size) * (np.sin(theta) ** 2) * np.sin(2 * azimuth)
    )
    f_tensor_unit_ret[4] = (
        -np.ones(img_size) * (np.sin(2 * theta)) * np.cos(azimuth)
    )
    f_tensor_unit_ret[5] = (
        -np.ones(img_size) * (np.sin(2 * theta)) * np.sin(azimuth)
    )

    f_tensor_blur = np.zeros_like(f_tensor_unit_ret)
    for i in range(4):
        if img_dim == 3 and psz_ps_ratio is not None:
            f_tensor_blur[2 + i] = uniform_filter(
                f_tensor_unit_ret[2 + i],
                (
                    avg_px_size,
                    avg_px_size,
                    int(np.round(avg_px_size / psz_ps_ratio)),
                ),
            )
        elif img_dim == 2:
            f_tensor_blur[2 + i] = uniform_filter(
                f_tensor_unit_ret[2 + i], (avg_px_size, avg_px_size)
            )
        else:
            raise ValueError(
                "azimuth and theta are either 2D or 3D, psz_ps_ratio should not be None for 3D images"
            )

    retardance_pr_avg, _, _ = scattering_potential_tensor_to_3D_orientation_PN(
        f_tensor_blur, material_type="positive", reg_ret_pr=reg_ret_pr
    )
    retardance_pr_avg /= np.max(retardance_pr_avg)

    return retardance_pr_avg
