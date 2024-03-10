import numpy as np
import torch
import matplotlib.pyplot as plt
import gc
import itertools
from numpy.fft import fft, fft2, ifft2, fftn, ifftn, fftshift, ifftshift


def Jones_sample(Ein, t, sa):
    """
    compute output electric field after the interaction between the transparent retardant sample and incident electric field.

    Parameters
    ----------
        Ein  : numpy.ndarray
               incident electric field with the size of (2, Ny, Nx, ...)

        t    : numpy.ndarray or a list
               eigen-transmission of the sample at its optical axis with size of (2, Ny, Nx, ...)

        sa   : numpy.ndarray
               slow-axis orientation in radian with the size of (Ny, Nx, ...)

    Returns
    -------
        Eout : numpy.ndarray
               output electric field with the size of (2, Ny, Nx, ...)
    """

    Eout = np.zeros_like(Ein)
    Eout[0] = Ein[0] * (t[0] * np.cos(sa) ** 2 + t[1] * np.sin(sa) ** 2) + Ein[
        1
    ] * (t[0] - t[1]) * np.sin(sa) * np.cos(sa)
    Eout[1] = Ein[0] * (t[0] - t[1]) * np.sin(sa) * np.cos(sa) + Ein[1] * (
        t[0] * np.sin(sa) ** 2 + t[1] * np.cos(sa) ** 2
    )

    return Eout


def Jones_to_Stokes(Ein, use_gpu=False, gpu_id=0):
    """

    given a coherent electric field, compute the corresponding Stokes vector.

    Parameters
    ----------
        Ein     : numpy.ndarray
                  electric field with the size of (2, Ny, Nx, ...)

        use_gpu : bool
                  option to use gpu or not

        gpu_id  : int
                  number refering to which gpu will be used

    Returns
    -------
        Stokes  : numpy.ndarray
                  corresponding Stokes vector with the size of (4, Ny, Nx, ...)

    """

    if use_gpu:
        globals()["cp"] = __import__("cupy")
        cp.cuda.Device(gpu_id).use()

        S0 = (cp.abs(Ein[0]) ** 2 + cp.abs(Ein[1]) ** 2)[cp.newaxis, ...]
        S1 = (cp.abs(Ein[0]) ** 2 - cp.abs(Ein[1]) ** 2)[cp.newaxis, ...]
        S2 = (cp.real(Ein[0].conj() * Ein[1] + Ein[0] * Ein[1].conj()))[
            cp.newaxis, ...
        ]
        S3 = (
            cp.real(-1j * (Ein[0].conj() * Ein[1] - Ein[0] * Ein[1].conj()))
        )[cp.newaxis, ...]
        Stokes = cp.concatenate((S0, S1, S2, S3), axis=0)

    else:
        Stokes = []
        Stokes.append(np.abs(Ein[0]) ** 2 + np.abs(Ein[1]) ** 2)  # S0
        Stokes.append(np.abs(Ein[0]) ** 2 - np.abs(Ein[1]) ** 2)  # S1
        Stokes.append(
            np.real(Ein[0].conj() * Ein[1] + Ein[0] * Ein[1].conj())
        )  # S2
        Stokes.append(
            np.real(-1j * (Ein[0].conj() * Ein[1] - Ein[0] * Ein[1].conj()))
        )  # S3
        Stokes = np.array(Stokes)

    return Stokes


def analyzer_output(Ein, alpha, beta):
    """

    compute output electric field after passing through an universal analyzer.

    Parameters
    ----------
        Ein   : numpy.ndarray
                incident electric field with the size of (2, Ny, Nx, ...)

        alpha : numpy.ndarray
                retardance of the first LC

        beta  : numpy.ndarray
                retardance of the second LC

    Returns
    -------
        Eout  : numpy.ndarray
                output electric field with the size of (2, Ny, Nx, ...)
    """

    Eout = Ein[0] * np.exp(-1j * beta / 2) * np.cos(alpha / 2) - Ein[
        1
    ] * 1j * np.exp(1j * beta / 2) * np.sin(alpha / 2)

    return Eout


def generate_pupil(frr, NA, lamb_in):
    """

    compute pupil function given spatial frequency, NA, wavelength.

    Parameters
    ----------
        frr       : torch.tensor
                    radial frequency coordinate in units of inverse length

        NA        : float
                    numerical aperture of the pupil function (normalized by the refractive index of the immersion media)

        lamb_in : float
                    wavelength of the light (inside the immersion media)
                    in units of length (inverse of frr's units)

    Returns
    -------
        Pupil     : numpy.ndarray
                    pupil function with the specified parameters with the size of (Ny, Nx)

    """

    Pupil = torch.zeros(frr.shape)
    Pupil[frr < NA / lamb_in] = 1

    return Pupil


def gen_sector_Pupil(fxx, fyy, NA, lamb_in, sector_angle, rotation_angle):
    """

    compute sector pupil functions given spatial frequency, NA, wavelength, sector angle, and the rotational angles.

    Parameters
    ----------
        fxx            : numpy.ndarray
                         x component of 2D spatial frequency array with the size of (Ny, Nx)

        fyy            : numpy.ndarray
                         y component of 2D spatial frequency array with the size of (Ny, Nx)

        NA             : float
                         numerical aperture of the pupil function (normalized by the refractive index of the immersion media)

        lamb_in      : float
                         wavelength of the light (inside the immersion media)

        sector_angle   : float
                         the angle of a sector pupil function (0~180 degrees)

        rotation_angle : float, int, or list of ints or floats
                         the rotational angles of a set of rotated sector pupil functions


    Returns
    -------
        Pupil_sector   : list of numpy.ndarray
                         a set of rotated sector pupil functions with the specified parameters with the size of (Ny, Nx)

    """

    N, M = fxx.shape

    Pupil = np.zeros((N, M))
    fr = (fxx**2 + fyy**2) ** (1 / 2)
    Pupil[fr < NA / lamb_in] = 1
    Pupil_sector = []

    if sector_angle > 180 or sector_angle < 0:
        raise ValueError("sector_angle should be between 0 to 180 degree")

    if isinstance(rotation_angle, int) or isinstance(rotation_angle, float):
        rotation_angle = [rotation_angle]
    elif isinstance(rotation_angle, list):
        if (not isinstance(rotation_angle[0], int)) and (
            not isinstance(rotation_angle[0], float)
        ):
            raise ValueError(
                "Elements in rotation_angle need to be either int or float"
            )
    else:
        raise ValueError(
            "rotation_angle needs to be int, float, or a list of int and float"
        )

    for i in range(len(rotation_angle)):
        deg = rotation_angle[i]
        temp = np.zeros_like(Pupil)
        temp2 = np.zeros_like(Pupil)
        temp[
            fyy * np.cos(np.deg2rad(deg)) - fxx * np.sin(np.deg2rad(deg))
            > 1e-10
        ] = 1
        temp2[
            fyy * np.cos(np.deg2rad(deg - (180 - sector_angle)))
            - fxx * np.sin(np.deg2rad(deg - (180 - sector_angle)))
            > 1e-10
        ] = 1

        Pupil_sector.append(temp * temp2 * Pupil)

    return Pupil_sector


def Source_subsample(Source_cont, NAx_coord, NAy_coord, subsampled_NA=0.1):
    """

    compute the sub-sampled source function with the specified sampling spacing in NA coordinate

    Parameters
    ----------
        Source_cont     : numpy.ndarray
                          continuous illumination source pattern with the size of (Ny, Nx)

        NAx_coord       : numpy.ndarray
                          x component of 2D spatial frequency array multiplied by wavelength with the size of (Ny, Nx)

        NAy_coord       : numpy.ndarray
                          y component of 2D spatial frequency array multiplied by wavelength with the size of (Ny, Nx)

        subsampled_NA   : float
                          subsampled spatial frequency in the unit of numerical aperture (normalized by the refractive index of the immersion media)


    Returns
    -------
        Source_discrete : numpy.ndarray
                          discretized illumination source pattern with dimension of (Ny, Nx)

    """

    N, M = Source_cont.shape

    [idx_y, idx_x] = np.where(Source_cont > 0)

    NAx_list = NAx_coord[idx_y, idx_x]
    NAy_list = NAy_coord[idx_y, idx_x]
    NA_list = ((NAx_list) ** 2 + (NAy_list) ** 2) ** (0.5)
    NA_idx = np.argsort(NA_list)

    illu_list = []

    first_idx = True

    for i in NA_idx:
        if first_idx:
            illu_list.append(i)
            first_idx = False
        elif (
            np.product(
                (NAx_list[i] - NAx_list[illu_list]) ** 2
                + (NAy_list[i] - NAy_list[illu_list]) ** 2
                >= subsampled_NA**2
            )
            == 1
        ):
            illu_list.append(i)

    Source_discrete = np.zeros((N, M))
    Source_discrete[idx_y[illu_list], idx_x[illu_list]] = 1

    return Source_discrete


def generate_propagation_kernel(
    radial_frequencies, pupil_support, wavelength, z_position_list
):
    """
    Parameters
    ----------
        radial_frequencies : torch.tensor
                             radial of 2D spatial frequency array with the size of (Y, X)

        pupil_support   : torch.tensor
                        the array that defines the support of the pupil function with the size of (Y, X)

        wavelength      : float
                        wavelength of the light in the immersion media

        z_position_list : torch.tensor or list
                        1D array of defocused z positions with the size of (Z)

    Returns
    -------
        propagation_kernel  : torch.tensor
                            corresponding propagation kernel with size (Z, Y, X)

    """

    oblique_factor = (
        (1 - wavelength**2 * radial_frequencies**2) * pupil_support
    ) ** (1 / 2) / wavelength

    propagation_kernel = pupil_support[None, :, :] * torch.exp(
        1j
        * 2
        * np.pi
        * torch.tensor(z_position_list)[:, None, None]
        * oblique_factor[None, :, :]
    )

    return propagation_kernel


def generate_greens_function_z(
    radial_frequencies, pupil_support, wavelength_illumination, z_position_list
):
    """

    generate Green's function in u_x, u_y, z space

    Parameters
    ----------
        radial_frequencies             : torch.tensor
                        x component of 2D spatial frequency array with the size of (Y, X)

        pupil_support   : torch.tensor
                        the array that defines the support of the pupil function with the size of (Y, X)

        wavelength_illumination       : float
                        wavelength of the light in the immersion media

        z_position_list      : torch.tensor or list
                        1D array of defocused z position with the size of (Z,)

    Returns
    -------
        greens_function_z       : torch.tensor
                        corresponding Green's function in u_x, u_y, z space with size of (Z, Y, X)

    """

    oblique_factor = (
        (1 - wavelength_illumination**2 * radial_frequencies**2)
        * pupil_support
    ) ** (1 / 2) / wavelength_illumination

    greens_function_z = (
        -1j
        / 4
        / np.pi
        * pupil_support[None, :, :]
        * torch.exp(
            1j
            * 2
            * np.pi
            * torch.tensor(z_position_list)[:, None, None]
            * oblique_factor[None, :, :]
        )
        / (oblique_factor[None, :, :] + 1e-15)
    )

    return greens_function_z


def gen_dyadic_Greens_tensor_z(fxx, fyy, G_fun_z, Pupil_support, lambda_in):
    """

    generate forward dyadic Green's function in u_x, u_y, z space

    Parameters
    ----------
        fxx           : numpy.ndarray
                        x component of 2D spatial frequency array with the size of (Ny, Nx)

        fyy           : numpy.ndarray
                        y component of 2D spatial frequency array with the size of (Ny, Nx)

        G_fun_z       : numpy.ndarray
                        forward Green's function in u_x, u_y, z space with size of (Ny, Nx, Nz)

        Pupil_support : numpy.ndarray
                        the array that defines the support of the pupil function with the size of (Ny, Nx)

        lambda_in     : float
                        wavelength of the light in the immersion media

    Returns
    -------
        G_tensor_z    : numpy.ndarray
                        forward dyadic Green's function in u_x, u_y, z space with the size of (3, 3, Ny, Nx, Nz)
    """

    N, M = fxx.shape
    fr = (fxx**2 + fyy**2) ** (1 / 2)
    oblique_factor = ((1 - lambda_in**2 * fr**2) * Pupil_support) ** (
        1 / 2
    ) / lambda_in

    diff_filter = np.zeros((3,) + G_fun_z.shape, complex)
    diff_filter[0] = (1j * 2 * np.pi * fxx * Pupil_support)[..., np.newaxis]
    diff_filter[1] = (1j * 2 * np.pi * fyy * Pupil_support)[..., np.newaxis]
    diff_filter[2] = (1j * 2 * np.pi * oblique_factor)[..., np.newaxis]

    G_tensor_z = np.zeros((3, 3) + G_fun_z.shape, complex)

    for i in range(3):
        for j in range(3):
            G_tensor_z[i, j] = (
                G_fun_z
                * diff_filter[i]
                * diff_filter[j]
                / (2 * np.pi / lambda_in) ** 2
            )
            if i == j:
                G_tensor_z[i, i] += G_fun_z

    return G_tensor_z


def gen_Greens_function_real(img_size, ps, psz, lambda_in):
    """

    generate Green's function in real space

    Parameters
    ----------
        img_size  : tuple
                    image dimension (Ny, Nx, Nz)

        ps        : float
                    transverse pixel size

        psz       : float
                    axial pixel size

        lambda_in : float
                    wavelength of the light in the immersion media

    Returns
    -------
        G_real    : numpy.ndarray
                    corresponding real-space Green's function with size of (Ny, Nx, Nz)

    """

    N, M, L = img_size

    x_r = (np.r_[:M] - M // 2) * ps
    y_r = (np.r_[:N] - N // 2) * ps
    z_r = (np.r_[:L] - L // 2) * psz

    xx_r, yy_r, zz_r = np.meshgrid(x_r, y_r, z_r)

    # radial coordinate
    rho = (xx_r**2 + yy_r**2 + zz_r**2) ** (0.5)

    # average radius of integration around r=0
    epsilon = (ps * ps * psz / np.pi / 4 * 3) ** (1 / 3)

    # wavenumber
    k = 2 * np.pi / lambda_in

    # average value for Green's function at r=0
    V_epsilon = (
        1
        / 1j
        / k
        * (
            epsilon * np.exp(1j * k * epsilon)
            - 1 / 1j / k * (np.exp(1j * k * epsilon) - 1)
        )
        / ps
        / ps
        / psz
    )

    G_real = np.exp(1j * k * rho) / (rho + 1e-7) / 4 / np.pi
    G_real[rho == 0] = V_epsilon

    return G_real


def gen_dyadic_Greens_tensor(G_real, ps, psz, lambda_in, space="real"):
    """

    generate dyadic Green's function tensor in real space or in Fourier space

    Parameters
    ----------
        G_real    : numpy.ndarray
                    real space Greens function for wave equation with size of (Ny, Nx, Nz)

        ps        : float
                    transverse pixel size

        psz       : float
                    axial pixel size

        lambda_in : float
                    wavelength of the light

        space     : str
                    'real' or 'Fourier' indicate real or Fourier space representation

    Returns
    -------
        G_tensor  : numpy.ndarray
                    corresponding real or Fourier space Green's tensor with size of (3, 3, Ny, Nx, Nz)

    """

    N, M, L = G_real.shape

    fx_r = ifftshift((np.r_[:M] - M // 2) / ps / M)
    fy_r = ifftshift((np.r_[:N] - N // 2) / ps / N)
    fz_r = ifftshift((np.r_[:L] - L // 2) / psz / L)
    fxx_r, fyy_r, fzz_r = np.meshgrid(fx_r, fy_r, fz_r)

    diff_filter = np.zeros((3, N, M, L), complex)
    diff_filter[0] = 1j * 2 * np.pi * fxx_r
    diff_filter[1] = 1j * 2 * np.pi * fyy_r
    diff_filter[2] = 1j * 2 * np.pi * fzz_r

    G_tensor = np.zeros((3, 3, N, M, L), complex)
    G_real_f = fftn(ifftshift(G_real)) * (ps * ps * psz)

    for i in range(3):
        for j in range(3):
            G_tensor[i, j] = (
                G_real_f
                * diff_filter[i]
                * diff_filter[j]
                / (2 * np.pi / lambda_in) ** 2
            )
            if i == j:
                G_tensor[i, i] += G_real_f

    if space == "Fourier":
        return G_tensor

    elif space == "real":
        return (
            fftshift(ifftn(G_tensor, axes=(2, 3, 4)), axes=(2, 3, 4))
            / ps
            / ps
            / psz
        )


def compute_weak_object_transfer_function_2d(
    illumination_pupil, detection_pupil
):
    """

    compute 2D weak object transfer function (2D WOTF)

    Parameters
    ----------
        illumination_pupil  : torch.tensor
                  illumination source pattern with the size of (Y, X)

        detection_pupil   : torch.tensor
                  pupil function with the size of (Y, X)

    Returns
    -------

        absorption_transfer_function      : torch.tensor
                  absorption transfer function with size of (Y, X)

        phase_transfer_function      : torch.tensor
                  phase transfer function with size of (Y, X)

    """

    SP_hat = torch.fft.fft2(illumination_pupil * detection_pupil)
    P_hat = torch.fft.fft2(detection_pupil)

    H1 = torch.fft.ifft2(torch.conj(SP_hat) * P_hat)
    H2 = torch.fft.ifft2(SP_hat * torch.conj(P_hat))
    I_norm = torch.sum(
        illumination_pupil * detection_pupil * torch.conj(detection_pupil)
    )
    absorption_transfer_function = (H1 + H2) / I_norm
    phase_transfer_function = 1j * (H1 - H2) / I_norm

    return absorption_transfer_function, phase_transfer_function


def WOTF_semi_3D_compute(
    Source_support, Source, Pupil, Hz_det, G_fun_z, use_gpu=False, gpu_id=0
):
    """

    compute semi-3D weak object transfer function (semi-3D WOTF)

    Parameters
    ----------
        Source_support : numpy.ndarray
                         illumination source pattern support with size of (Ny, Nx)

        Source         : numpy.ndarray
                         illumination source pattern with spatial frequency modulation with size of (Ny, Nx)

        Pupil          : numpy.ndarray
                         pupil function with the size of (Ny, Nx)

        Hz_det         : numpy.ndarray
                         one slice of propagation kernel with size of (Ny, Nx)

        G_fun_z        : numpy.ndarray
                         one slice of scaled 2D Fourier transform of Green's function in xy-dimension with size of (Ny, Nx)

        use_gpu        : bool
                         option to use gpu or not

        gpu_id         : int
                         number refering to which gpu will be used

    Returns
    -------
        Hu             : numpy.ndarray
                         absorption transfer function with size of (Ny, Nx)

        Hp             : numpy.ndarray
                         phase transfer function with size of (Ny, Nx)

    """

    if use_gpu:
        globals()["cp"] = __import__("cupy")
        cp.cuda.Device(gpu_id).use()

        Source = cp.array(Source)
        Source_support = cp.array(Source_support)
        Pupil = cp.array(Pupil)
        Hz_det = cp.array(Hz_det)
        G_fun_z = cp.array(G_fun_z)

        H1 = cp.fft.ifft2(
            cp.conj(cp.fft.fft2(Source * Pupil * Hz_det))
            * cp.fft.fft2(Pupil * G_fun_z)
        )
        H2 = cp.fft.ifft2(
            cp.fft.fft2(Source * Pupil * Hz_det)
            * cp.conj(cp.fft.fft2(Pupil * G_fun_z))
        )
        I_norm = cp.sum(Source_support * Pupil * cp.conj(Pupil))
        Hu = (H1 + H2) / I_norm
        Hp = 1j * (H1 - H2) / I_norm

        Hu = cp.asnumpy(Hu)
        Hp = cp.asnumpy(Hp)

    else:
        H1 = ifft2(
            fft2(Source * Pupil * Hz_det).conj() * fft2(Pupil * G_fun_z)
        )
        H2 = ifft2(
            fft2(Source * Pupil * Hz_det) * fft2(Pupil * G_fun_z).conj()
        )
        I_norm = np.sum(Source_support * Pupil * Pupil.conj())
        Hu = (H1 + H2) / I_norm
        Hp = 1j * (H1 - H2) / I_norm

    return Hu, Hp


def compute_weak_object_transfer_function_3D(
    illumination_pupil_support,
    illumination_pupil,
    detection_pupil,
    propagation_kernel,
    greens_function_z,
    z_pixel_size,
):
    """
    compute 3D weak object transfer function (3D WOTF)

    Parameters
    ----------
        illumination_pupil_support : torch.tensor
                         illumination source pattern support with the size of (Y, X)

        illumination_pupil         : torch.tensor
                         source with spatial frequency modulation with the size of (Y, X)

        detection_pupil          : torch.tensor
                         pupil function with the size of (Y, X)

        propagation_kernel         : torch.tensor
                         propagation kernel with size of (Z, Y, X)

        greens_function_z        : torch.tensor
                         2D Fourier transform of Green's function in xy-dimension with size of (Z, Y, X)

        z_pixel_size            : float
                         pixel size in the z-dimension

    Returns
    -------
        real_potential_transfer_function           : torch.tensor
                         transfer function of real scattering potential with the size of (Z, Y, X)

        imag_potential_transfer_function           : torch.tensor
                         transfer function of imaginary scattering potential with the size of (Z, Y, X)

    """

    # NOTE: To match numpy's hanning window, I needed `periodic=False. I think
    # I would prefer `periodic=True` so that window[0] == 1, but the difference
    # should be marginal.
    window = torch.fft.ifftshift(
        torch.hann_window(propagation_kernel.shape[0], periodic=False)
    )

    SPHz_hat = torch.fft.fft2(
        (illumination_pupil * detection_pupil)[None, :, :]
        * propagation_kernel,
        dim=(1, 2),
    )
    PG_hat = torch.fft.fft2(
        detection_pupil[None, :, :] * greens_function_z, dim=(1, 2)
    )

    H1 = torch.fft.ifft2(torch.conj(SPHz_hat) * PG_hat, dim=(1, 2))
    H1 = H1 * window[:, None, None]
    H1 = torch.fft.fft(H1, dim=0) * z_pixel_size

    H2 = torch.fft.ifft2(SPHz_hat * torch.conj(PG_hat), dim=(1, 2))
    H2 = H2 * window[:, None, None]
    H2 = torch.fft.fft(H2, dim=0) * z_pixel_size

    I_norm = torch.sum(
        illumination_pupil_support
        * detection_pupil
        * torch.conj(detection_pupil)
    )
    real_potential_transfer_function = (H1 + H2) / I_norm
    imag_potential_transfer_function = 1j * (H1 - H2) / I_norm

    return real_potential_transfer_function, imag_potential_transfer_function


def gen_geometric_inc_matrix(incident_theta, incident_phi, Source):
    """

    compute forward and backward matrix mapping from inclination coefficients to retardance with geometric model

    Parameters
    ----------
        incident_theta           : numpy.ndarray
                                   theta spherical coordinate map in 2D spatial frequency grid with the size of (Ny, Nx)

        incident_phi             : numpy.ndarray
                                   phi spherical coordinate map in 2D spatial frequency grid with the size of (Ny, Nx)

        Source                   : numpy.ndarray
                                   illumination Source pattern in 2D spatial frequency grid with the size of (N_pattern, Ny, Nx)

    Returns
    -------
        geometric_inc_matrix     : numpy.ndarray
                                   forward matrix mapping from inclination coefficients to retardance

        geometric_inc_matrix_inv : numpy.ndarray
                                   pinv of the forward matrix

    """

    N_pattern, _, _ = Source.shape

    geometric_inc_matrix = []

    for i in range(N_pattern):
        idx_y, idx_x = np.where(Source[i])

        geometric_inc_matrix.append(
            [
                1,
                np.mean(0.5 * np.cos(2 * incident_theta[idx_y, idx_x])),
                np.mean(
                    -0.5
                    * np.sin(2 * incident_theta[idx_y, idx_x])
                    * np.cos(incident_phi[idx_y, idx_x])
                ),
                np.mean(
                    -0.5
                    * np.sin(2 * incident_theta[idx_y, idx_x])
                    * np.sin(incident_phi[idx_y, idx_x])
                ),
                np.mean(
                    -0.5
                    * (np.sin(incident_theta[idx_y, idx_x]) ** 2)
                    * np.cos(2 * incident_phi[idx_y, idx_x])
                ),
                np.mean(
                    -0.5
                    * (np.sin(incident_theta[idx_y, idx_x]) ** 2)
                    * np.sin(2 * incident_phi[idx_y, idx_x])
                ),
            ]
        )

    geometric_inc_matrix = np.array(geometric_inc_matrix)
    geometric_inc_matrix_inv = np.linalg.pinv(geometric_inc_matrix)

    return geometric_inc_matrix, geometric_inc_matrix_inv


def SEAGLE_vec_forward(
    E_tot, f_scat_tensor, G_tensor, use_gpu=False, gpu_id=0
):
    """

    compute vectorial SEAGLE forward model

    Parameters
    ----------
        E_tot         : nunmpy.ndarray
                        total electric field (scattered + incident) with the size of (3, Ny, Nx, Nz)

        f_scat_tensor : numpy.ndarray
                        scattering potential tensor with the size of (3, 3, Ny, Nx, Nz)

        G_tensor      : numpy.ndarray
                        dyadic Green's function with the size of (3, 3, Ny, Nx, Nz)

        use_gpu       : bool
                        option to use gpu or not

        gpu_id        : int
                        number refering to which gpu will be used

    Returns
    -------
        E_in_est      : numpy.ndarray
                        estimated incident electric field with the size of (3, Ny, Nx, Nz)

    """

    N, M, L = E_tot.shape[1:]

    if use_gpu:
        globals()["cp"] = __import__("cupy")
        cp.cuda.Device(gpu_id).use()

        pad_convolve_G = lambda x, y, z: cp.fft.ifftn(
            cp.fft.fftn(
                cp.pad(
                    x,
                    ((N // 2, N // 2), (M // 2, M // 2), (L // 2, L // 2)),
                    mode="constant",
                    constant_values=y,
                )
            )
            * z
        )[N // 2 : -N // 2, M // 2 : -M // 2, L // 2 : -L // 2]
        E_interact = cp.zeros((3, N, M, L), complex)
        E_in_est = cp.zeros_like(E_tot, complex)

        for p, q in itertools.product(range(3), range(3)):
            E_interact[p] += f_scat_tensor[p, q] * E_tot[q]

        for p, q in itertools.product(range(3), range(3)):
            E_in_est[p] += pad_convolve_G(
                E_interact[q],
                cp.asnumpy(cp.abs(cp.mean(E_interact[q]))),
                G_tensor[p, q],
            )
            if p == q:
                E_in_est[p] += E_tot[p]

    else:
        pad_convolve_G = lambda x, y, z: ifftn(
            fftn(
                np.pad(
                    x,
                    ((N // 2,), (M // 2,), (L // 2,)),
                    mode="constant",
                    constant_values=y,
                )
            )
            * z
        )[N // 2 : -N // 2, M // 2 : -M // 2, L // 2 : -L // 2]
        E_interact = np.zeros((3, N, M, L), complex)
        E_in_est = np.zeros_like(E_tot, complex)

        for p, q in itertools.product(range(3), range(3)):
            E_interact[p] += f_scat_tensor[p, q] * E_tot[q]

        for p, q in itertools.product(range(3), range(3)):
            E_in_est[p] += pad_convolve_G(
                E_interact[q], np.abs(np.mean(E_interact[q])), G_tensor[p, q]
            )
            if p == q:
                E_in_est[p] += E_tot[p]

    return E_in_est


def SEAGLE_vec_backward(
    E_diff, f_scat_tensor, G_tensor, use_gpu=False, gpu_id=0
):
    """

    compute the adjoint of vectorial SEAGLE forward model

    Parameters
    ----------
        E_diff        : numpy.ndarray
                        difference between estimated and true incident field with the size of (3, Ny, Nx, Nz)

        f_scat_tensor : numpy.ndarray
                        scattering potential tensor with the size of (3, 3, Ny, Nx, Nz)

        G_tensor      : numpy.ndarray
                        dyadic Green's function with the size of (3, 3, Ny, Nx, Nz)

        use_gpu       : bool
                        option to use gpu or not

        gpu_id        : int
                        gpu_id for computation

    Returns
    -------
        grad_E        : numpy.ndarray
                        gradient of the total electric field with the size of (3, Ny, Nx, Nz)

    """

    N, M, L = E_diff.shape[1:]

    if use_gpu:
        globals()["cp"] = __import__("cupy")
        cp.cuda.Device(gpu_id).use()

        pad_convolve_G = lambda x, y, z: cp.fft.ifftn(
            cp.fft.fftn(
                cp.pad(
                    x,
                    ((N // 2, N // 2), (M // 2, M // 2), (L // 2, L // 2)),
                    mode="constant",
                    constant_values=y,
                )
            )
            * z
        )[N // 2 : -N // 2, M // 2 : -M // 2, L // 2 : -L // 2]

        E_diff_conv = cp.zeros_like(E_diff, complex)
        grad_E = cp.zeros_like(E_diff, complex)

        for p, q in itertools.product(range(3), range(3)):
            E_diff_conv[p] += pad_convolve_G(
                E_diff[q],
                cp.asnumpy(cp.abs(cp.mean(E_diff[p]))),
                G_tensor[p, q].conj(),
            )

        for p in range(3):
            E_interact = cp.zeros((N, M, L), complex)
            for q in range(3):
                E_interact += f_scat_tensor[q, p].conj() * E_diff_conv[q]
            grad_E[p] = E_diff[p] + E_interact

    else:
        pad_convolve_G = lambda x, y, z: ifftn(
            fftn(
                np.pad(
                    x,
                    ((N // 2,), (M // 2,), (L // 2,)),
                    mode="constant",
                    constant_values=y,
                )
            )
            * z
        )[N // 2 : -N // 2, M // 2 : -M // 2, L // 2 : -L // 2]

        E_diff_conv = np.zeros_like(E_diff, complex)
        grad_E = np.zeros_like(E_diff, complex)

        for p, q in itertools.product(range(3), range(3)):
            E_diff_conv[p] += pad_convolve_G(
                E_diff[q], np.abs(np.mean(E_diff[p])), G_tensor[p, q].conj()
            )

        for p in range(3):
            E_interact = np.zeros((N, M, L), complex)
            for q in range(3):
                E_interact += f_scat_tensor[q, p].conj() * E_diff_conv[q]
            grad_E[p] = E_diff[p] + E_interact

    return grad_E


def scattering_potential_tensor_to_3D_orientation_PN(
    f_tensor, material_type="positive", reg_ret_pr=1e-1
):
    """

    compute principal retardance and 3D orientation from scattering potential tensor components

    Parameters
    ----------
        f_tensor      : numpy.ndarray
                        scattering potential tensor components with the size of (7, N, M) or (7, N, M, N_defocus) for 3D

        material_type : str
                        'positive' for assumption of positively uniaxial material
                        'negative' for assumption of negatively uniaxial material

        reg_ret_pr    : float
                        regularization parameters for principal retardance estimation

    Returns
    -------
        retardance_pr : numpy.ndarray
                        reconstructed principal retardance with the size of (N, M) for 2D and (N, M, N_defocus) for 3D
                        p: positively uniaxial solution (return retardance_pr_p when 'positive' is specified for material_type)
                        n: negatively uniaxial solution (return retardance_pr_n when 'negative' is specified for material_type)

        azimuth       : numpy.ndarray
                        reconstructed in-plane orientation with the size of (N, M) for 2D and (N, M, N_defocus) for 3D
                        p: positively uniaxial solution (return azimuth_p when 'positive' is specified for material_type)
                        n: negatively uniaxial solution (return azimuth_n when 'negative' is specified for material_type)

        theta         : numpy.ndarray
                        reconstructed out-of-plane inclination with the size of (N, M) for 2D and (N, M, N_defocus) for 3D
                        p: positively uniaxial solution (return theta_p when 'positive' is specified for material_type)
                        n: negatively uniaxial solution (return theta_n when 'negative' is specified for material_type)

    """

    if material_type == "positive":
        # Positive uniaxial material

        azimuth_p = (np.arctan2(-f_tensor[3], -f_tensor[2]) / 2) % np.pi
        del_f_sin_square_p = -f_tensor[2] * np.cos(2 * azimuth_p) - f_tensor[
            3
        ] * np.sin(2 * azimuth_p)
        del_f_sin2theta_p = -f_tensor[4] * np.cos(azimuth_p) - f_tensor[
            5
        ] * np.sin(azimuth_p)
        theta_p = np.arctan2(2 * del_f_sin_square_p, del_f_sin2theta_p)
        retardance_pr_p = (
            del_f_sin_square_p
            * np.sin(theta_p) ** 2
            / (np.sin(theta_p) ** 4 + reg_ret_pr)
        )

        return retardance_pr_p, azimuth_p, theta_p

    elif material_type == "negative":
        # Negative uniaxial material

        azimuth_n = (np.arctan2(f_tensor[3], f_tensor[2]) / 2) % np.pi
        del_f_sin_square_n = f_tensor[2] * np.cos(2 * azimuth_n) + f_tensor[
            3
        ] * np.sin(2 * azimuth_n)
        del_f_sin2theta_n = f_tensor[4] * np.cos(azimuth_n) + f_tensor[
            5
        ] * np.sin(azimuth_n)
        theta_n = np.arctan2(2 * del_f_sin_square_n, del_f_sin2theta_n)
        retardance_pr_n = (
            -del_f_sin_square_n
            * np.sin(theta_n) ** 2
            / (np.sin(theta_n) ** 4 + reg_ret_pr)
        )

        return retardance_pr_n, azimuth_n, theta_n


def phase_inc_correction(f_tensor0, retardance_pr, theta):
    """

    compute the inclination-corrected phase from the principal retardance, inclination, and the 0-th component of the scattering potential tensor

    Parameters
    ----------
        f_tensor0     : numpy.ndarray
                        0-th component of scattering potential tensor with the size of (N, M) or (N, M, N_defocus) for 3D

        retardance_pr : numpy.ndarray
                        reconstructed principal retardance with the size of (N, M) for 2D and (N, M, N_defocus) for 3D

        theta         : numpy.ndarray
                        reconstructed out-of-plane inclination with the size of (N, M) for 2D and (N, M, N_defocus) for 3D

    Returns
    -------
        phase         : numpy.ndarray
                        inclination-corrected phase with the size of (N, M) for 2D and (N, M, N_defocus) for 3D

    """

    phase = -f_tensor0 + retardance_pr * np.cos(theta) ** 2

    return phase


def optic_sign_probability(mat_map, mat_map_thres=0.1):
    """

    compute the optic sign probability from the reconstructed material tendancy

    Parameters
    ----------
        mat_map       : numpy.ndarray
                        reconstructed material tendancy with the size of (2, N, M) for 2D and (2, N, M, N_defocus) for 3D

        mat_map_thres : float
                        the cut-off material tendancy to noisy tendancy estimate (typically 0.05 ~ 0.2)

    Returns
    -------
        p_mat_map     : numpy.ndarray
                        computed optic sign probability for positive uniaxial material with the size of (N, M) for 2D and (N, M, N_defocus) for 3D

    """

    mat_map_norm = mat_map / np.max(np.abs(np.sum(mat_map, axis=0)))
    p_mat_map = np.maximum(mat_map_norm[0], mat_map_thres) / (
        np.maximum(mat_map_norm[0], mat_map_thres)
        + np.maximum(mat_map_norm[1], mat_map_thres)
    )

    return p_mat_map


def unit_conversion_from_scattering_potential_to_permittivity(
    SP_array, lambda_0, n_media=1, imaging_mode="3D"
):
    """

    Convert the scattering potential of the specimen to the relative permittivity.

    Parameters
    ----------
        SP_array    : numpy.ndarray
                      array in the unit of a scattering potential (z-projected for 2D)

        lambda_0    : float
                      wavelength of the light in the free space

        n_media     : float
                      refractive index of the immersing media

        imaging_dim : str
                      option to convert the scattering potential unit for 2D or 3D imaging
                      '2D': convert the unit of a z-projected scattering potential to the unit of phase or absorption (in the unit of nm)
                      '2D-ret': convert the unit of a z-projected scattering potential to the unit of principal retardance (in the unit of nm)
                      '3D': convert the unit of a scattering potential to the unit of a permittivity (unitless)

    Returns
    -------
        P_array     : numpy.ndarray
                      array in the unit of a permittivity (z-projected for 2D)

    """

    k_0 = 2 * np.pi / lambda_0

    if imaging_mode == "2D":
        P_array = SP_array / 2 / k_0**2 / n_media * 1e3

    elif imaging_mode == "2D-ret":
        P_array = SP_array / k_0**2 / n_media * 1e3

    elif imaging_mode == "3D":
        P_array = SP_array / k_0**2

    else:
        raise ValueError(
            "Unsupported option for imaging dimension. imaging_mode must be 2D-phase, 2D-ret or 3D"
        )

    return P_array
