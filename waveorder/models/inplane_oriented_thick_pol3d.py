from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from waveorder import correction, stokes, util


def generate_test_phantom(yx_shape):
    star, theta, _ = util.generate_star_target(yx_shape, blur_px=0.1)
    retardance = 0.25 * star
    orientation = (theta % np.pi) * (star > 1e-3)
    transmittance = 0.9 * torch.ones_like(retardance)
    depolarization = 0.9 * torch.ones_like(retardance)
    return retardance, orientation, transmittance, depolarization


def calculate_transfer_function(
    swing,
    scheme,
):
    return stokes.calculate_intensity_to_stokes_matrix(swing, scheme=scheme)


def visualize_transfer_function(viewer, intensity_to_stokes_matrix):
    viewer.add_image(
        intensity_to_stokes_matrix.cpu().numpy(),
        name="Intensity to stokes matrix",
    )


def apply_transfer_function(
    retardance,
    orientation,
    transmittance,
    depolarization,
    intensity_to_stokes_matrix,
):
    stokes_params = stokes.stokes_after_adr(
        retardance, orientation, transmittance, depolarization
    )
    stokes_to_intensity_matrix = torch.linalg.pinv(intensity_to_stokes_matrix)

    cyx_intensities = stokes.mmul(
        stokes_to_intensity_matrix, torch.stack(stokes_params)
    )

    # Return in czyx shape
    # TODO: make this simulation more realistic with defocussed data
    return cyx_intensities[:, None, ...] + 0.1


def apply_inverse_transfer_function(
    czyx_data: Tensor,
    intensity_to_stokes_matrix: Tensor,
    cyx_no_sample_data: Optional[Tensor] = None,
    remove_estimated_background: bool = False,
    project_stokes_to_2d: bool = False,
    flip_orientation: bool = False,
    rotate_orientation: bool = False,
) -> Tuple[Tensor]:
    """Reconstructs retardance, orientation, transmittance, and depolarization
    from czyx_data and an intensity_to_stokes_matrix, providing options for
    background correction, projection, and orientation transformations.

    Parameters
    ----------
    czyx_data : Tensor
        4D raw data, first dimension is the polarization dimension, remaining
        dimensions are spatial
    intensity_to_stokes_matrix : Tensor
        Forward model, see calculate_transfer_function above
    cyx_no_sample_data : Tensor, optional
        3D raw background data, by default None
        First dimension is the polarization dimension, remaining dimensions are spatial.
        cyx shape must match in this parameter and czxy_data
        If provided, this background will be removed.
        If None, no background will be removed.
    remove_estimated_background : bool, optional
        Estimate a background from the data and remove it, by default False
    project_stokes_to_2d : bool, optional
        Project stokes to 2D for SNR improvement in thin samples, by default False
    flip_orientation : bool, optional
        Flip the reconstructed orientation about the x axis, by default False
    rotate_orientation : bool, optional
        Add 90 degrees to the reconstructed orientation, by default False

    Notes
    -----
    cyx_no_sample_data and remove_estimated_background provide background correction options

    flip_orientation and rotate_orientation modify the reconstructed orientation.
    We recommend using these parameters when a test target with a known orientation
    is available.

    Returns
    -------
    Tuple[Tensor]
        zyx_retardance (radians)
        zyx_orientation (radians)
        zyx_transmittance (unitless)
        zyx_depolarization (unitless)
    """
    data_stokes = stokes.mmul(intensity_to_stokes_matrix, czyx_data)

    # Apply a "Measured" background correction
    if cyx_no_sample_data is None:
        background_corrected_stokes = data_stokes
    else:
        # Find the no-sample Stokes parameters from the background data
        measured_no_sample_stokes = stokes.mmul(
            intensity_to_stokes_matrix, cyx_no_sample_data
        )
        # Estimate the attenuating, depolarizing, retarder's inverse Mueller
        # matrix that caused this background data
        inverse_background_mueller = stokes.mueller_from_stokes(
            *measured_no_sample_stokes, model="adr", direction="inverse"
        )
        # Apply this background-correction Mueller matrix to the data to remove
        # the background contribution
        background_corrected_stokes = stokes.mmul(
            inverse_background_mueller, data_stokes
        )

    # Apply an "Estimated" background correction
    if remove_estimated_background:
        for stokes_index in range(background_corrected_stokes.shape[0]):
            # Project to 2D
            z_projection = torch.mean(
                background_corrected_stokes[stokes_index], dim=0
            )
            # Estimate the background and subtract
            background_corrected_stokes[
                stokes_index
            ] -= correction.estimate_background(
                z_projection, order=2, block_size=32
            )

    # Project to 2D (typically for SNR reasons)
    if project_stokes_to_2d:
        background_corrected_stokes = torch.mean(
            background_corrected_stokes, dim=1
        )[:, None, ...]

    # Estimate an attenuating, depolarizing, retarder's parameters,
    # i.e. (retardance, orientation, transmittance, depolarization)
    # from the background-corrected Stokes values
    adr_parameters = stokes.estimate_adr_from_stokes(
        *background_corrected_stokes
    )

    # Apply orientation transformations
    orientation = stokes.apply_orientation_offset(
        adr_parameters[1], rotate=rotate_orientation, flip=flip_orientation
    )

    # Return (retardance, orientation, transmittance, depolarization)
    return adr_parameters[0], orientation, adr_parameters[2], adr_parameters[3]
