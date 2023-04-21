import torch
from waveorder import stokes


def calculate_transfer_function(
    swing,
    polarized_illumination_scheme,
    cyx_background_data=None,  # if not None, apply this as background correction
    czyx_data_for_background_estimate=None,  # if not None, average over z, estimate background, and remove it
):
    # Calculate A matrix
    intensity_to_stokes_matrix = stokes.calculate_intensity_to_stokes_matrix(
        swing, scheme=polarized_illumination_scheme
    )

    # Set default background correction matrix as identity
    inverse_background_mueller = torch.eye(4)

    # Calculate measured background correction
    if cyx_background_data is not None:
        measured_background_stokes = stokes.mmul(
            intensity_to_stokes_matrix, cyx_background_data
        )
        inverse_background_mueller = stokes.mueller_from_stokes(
            *measured_background_stokes, model="adr", direction="inverse"
        )

    # TODO: Implement estimated background correction
    # Calculate estimated background correction
    # if czyx_data_for_background_estimate is not None:
    #    cyx_data = torch.mean(czyx_data_for_background_estimate, dim=1)
    #    measured_stokes = stokes.mmul(intensity_to_stokes_matrix, cyx_data)
    #   inverse_background_mueller = stokes.mueller_from_stokes(
    #       *measured_stokes, model="adr", direction="inverse"
    #   )

    return intensity_to_stokes_matrix, inverse_background_mueller


def visualize_transfer_function(
    viewer, intensity_to_stokes_matrix, inverse_background_mueller
):
    viewer.add_image(
        intensity_to_stokes_matrix.cpu().numpy(),
        name="Intensity to stokes matrix",
    )
    viewer.add_image(
        inverse_background_mueller.cpu().numpy(),
        name="Inverse background mueller",
    )


def apply_transfer_function(
    intensity_to_stokes_matrix,
    inverse_background_mueller,
    retardance,
    orientation,
    transmittance,
    depolarization,
):
    s = stokes.stokes_after_adr(
        retardance, orientation, transmittance, depolarization
    )

    stokes_to_intensity_matrix = torch.linalg.pinv(intensity_to_stokes_matrix)
    reverse_background = torch.linalg.pinv(intensity_to_stokes_matrix)

    return NotImplementedError


def apply_inverse_transfer_function(
    czyx_data, intensity_to_stokes_matrix, inverse_background_mueller
):
    data_stokes = stokes.mmul(intensity_to_stokes_matrix, czyx_data)
    corrected_stokes = stokes.mmul(inverse_background_mueller, data_stokes)
    return stokes.estimate_adr_from_stokes(*corrected_stokes)
