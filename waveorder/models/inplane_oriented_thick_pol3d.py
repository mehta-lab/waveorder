import numpy as np
import torch
from waveorder import background_estimator, stokes, util


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
    czyx_data,
    intensity_to_stokes_matrix,
    wavelength_illumination=0.5,  # TOOD: MOVE THIS PARAM TO OTF? (leaky param)
    cyx_no_sample_data=None,  # if not None, use this data for background correction
    project_stokes_to_2d=False,
    remove_estimated_background=False,  # if True estimate background from czyx_data and remove it
    orientation_flip=False,  # TODO implement
    orientation_rotate=False,  # TODO implement
):
    data_stokes = stokes.mmul(intensity_to_stokes_matrix, czyx_data)

    # "Measured" background correction
    if cyx_no_sample_data is None:
        background_corrected_stokes = data_stokes
    else:
        measured_no_sample_stokes = stokes.mmul(
            intensity_to_stokes_matrix, cyx_no_sample_data
        )
        inverse_background_mueller = stokes.mueller_from_stokes(
            *measured_no_sample_stokes, model="adr", direction="inverse"
        )
        background_corrected_stokes = stokes.mmul(
            inverse_background_mueller, data_stokes
        )

    # "Estimated" background correction
    if remove_estimated_background:
        estimator = background_estimator.BackgroundEstimator2D()
        for stokes_index in range(background_corrected_stokes.shape[0]):
            z_projection = torch.mean(
                background_corrected_stokes[stokes_index], dim=0
            )
            background_corrected_stokes[
                stokes_index
            ] -= estimator.get_background(
                z_projection,
                normalize=False,
            )

    # Project to 2D (typically for SNR reasons)
    if project_stokes_to_2d:
        background_corrected_stokes = torch.mean(
            background_corrected_stokes, dim=1
        )[:, None, ...]

    adr_parameters = stokes.estimate_adr_from_stokes(
        *background_corrected_stokes
    )

    # Return retardance in distance units (matching wavelength_illumination)
    retardance = adr_parameters[0] * wavelength_illumination / (2 * np.pi)

    return retardance, adr_parameters[1], adr_parameters[2], adr_parameters[3]
